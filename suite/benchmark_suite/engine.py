# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
import json
from pathlib import Path
import queue
import shutil
import sqlite3
import time
from typing import Any, TextIO
import uuid

from .backends import Backend, WorkerHandle
from .contracts import (
    CampaignSpec,
    DATASET_VERSION,
    ContractError,
    PROTOCOL_VERSION,
    canonical,
    content_hash,
    integer,
    mapping,
    text,
)
from .protocol import PendingRequest, Server
from .storage import atomic_json, file_hash, sync_directory
from .store import Store
from .timing import TimingPolicy

READ_METHODS = frozenset(
    {
        "engine.status",
        "catalog.get",
        "parameters.resolve",
        "campaign.preview",
        "campaign.status",
        "results.query",
        "events.read",
        "logs.read",
        "inspection.get",
        "profiles.list",
        "spawns.list",
        "command.status",
    }
)
READ_METHODS = READ_METHODS | {
    "measurements.query",
    "spawns.map",
    "profiles.sweep_preview",
    "profiles.export",
    "profiles.diff",
    "spawns.export",
    "reports.get",
    "exports.get",
    "profiles.get",
    "spawns.get",
    "results.get",
    "campaign.repeat",
}
ACTIVE_STATES = ("RUNNING", "RETRYING", "PAUSING", "PAUSED", "STOPPING", "BLOCKED", "REPORTING")


class Engine:
    def __init__(
        self,
        workspace: Path,
        backend: Backend,
        resolve: Callable[[dict[str, Any]], CampaignSpec] | None = None,
        library: Any = None,
    ):
        self.store = Store(workspace)
        from .exports import ExportManager

        self.exports = ExportManager(self.store)
        self.workspace = self.store.workspace
        self.backend, self.library = backend, library
        self.resolve = resolve or CampaignSpec.accepted
        if not backend.synthetic and resolve is None:
            self.store.close()
            raise ValueError("Native workers require catalog and dependency preflight")
        self.active: dict[str, WorkerHandle] = {}
        self.metrics: dict[str, dict[str, Any]] = {}
        self.timing: dict[str, TimingPolicy] = {}
        self.parallel_exposed: set[str] = set()
        self.current: str | None = None
        self.queue_enabled = False
        self.closed = False
        self.job: Any = None
        self.job_id: str | None = None
        self.job_stage = ""
        self.pending_compactions: list[dict[str, Any]] = []
        self.report_import: TextIO | None = None
        self.report_reference: dict[str, Any] | None = None
        self.requests: queue.Queue[PendingRequest] = queue.Queue(maxsize=128)
        self.server: Server | None = None
        if library:
            library.bind(self.store)
        self._recover()

    def _recover(self) -> None:
        from .jobs import recover

        for report in self.store.rows(
            "SELECT * FROM reports WHERE state IN ('QUEUED','COMPACTING','COMMITTING','RUNNING','INDEXING')"
        ):
            specification = json.loads(report["specification"])
            recover(
                specification.get("token", ""),
                specification.get("pid"),
                specification.get("pid_created"),
            )
            self._remove_report_output(report)
            self.store.db.execute(
                "UPDATE reports SET state='INTERRUPTED',detail='Engine restart; regenerate after Resume' WHERE id=?",
                (report["id"],),
            )
        unfinished = self.store.rows("SELECT * FROM attempts WHERE phase!='FINISHED'")
        if not self.backend.synthetic:
            from .datasets import recover_compactions

            recover_compactions(self.workspace)
        self.backend.recover(unfinished)
        for attempt in unfinished:
            if attempt["discard_intent"]:
                self._discard(attempt)
                continue
            manifest = Path(attempt["directory"]) / "manifest.json"
            finalized = None
            candidate: dict[str, Any] = {}
            if manifest.is_file():
                try:
                    decoded = json.loads(manifest.read_text())
                    candidate = decoded if isinstance(decoded, dict) else {}
                    if candidate.get("phase") == "FINISHED" and isinstance(
                        candidate.get("outcome"), str
                    ):
                        finalized = candidate
                except (OSError, ValueError):
                    candidate = {}
                    finalized = None
            if finalized is not None:
                if not self.backend.synthetic:
                    from .datasets import finalize_attempt

                    try:
                        finalize_attempt(Path(attempt["directory"]))
                    except (OSError, ValueError) as error:
                        finalized["outcome"] = "storage_failure"
                        finalized["detail"] = str(error)
                with self.store.transaction():
                    self.store.db.execute(
                        "UPDATE attempts SET outcome=?,metrics=?,detail=? WHERE id=?",
                        (
                            finalized["outcome"],
                            canonical(finalized.get("final_metrics", {})),
                            finalized.get("detail", ""),
                            attempt["id"],
                        ),
                    )
                self._finish(self.store.attempt(attempt["id"]))
            else:
                directory = Path(attempt["directory"])
                from .datasets import finalize_partial

                try:
                    finalize_partial(
                        directory,
                        "engine_interruption",
                        "Unexpected engine or host interruption; readable checkpoints retained",
                        json.loads(attempt["metrics"] or "{}"),
                    )
                except (OSError, ValueError) as error:
                    interrupted = json.loads(manifest.read_bytes())
                    interrupted["recovery_error"] = str(error)
                    atomic_json(manifest, interrupted)
                with self.store.transaction():
                    self.store.db.execute(
                        "UPDATE attempts SET phase='FINISHED',outcome='engine_interruption',"
                        "terminal_ns=?,cleanup_ns=? WHERE id=?",
                        (time.time_ns(), time.time_ns(), attempt["id"]),
                    )
                    self.store.db.execute(
                        "UPDATE experiments SET state='NOT_DONE',held=0 WHERE id=?",
                        (attempt["experiment_id"],),
                    )
                    self.store.event(
                        "attempt.interrupted",
                        {"retained": True},
                        attempt["campaign_id"],
                        attempt["id"],
                    )
                    self._record_artifacts(attempt, manifest)
        for campaign in self.store.rows(
            "SELECT id FROM campaigns WHERE state IN "
            "('RUNNING','RETRYING','PAUSING','PAUSED','STOPPING','BLOCKED','REPORTING')"
        ):
            self.store.set_state(
                campaign["id"], "RECOVERY_REQUIRED", "Engine restarted; Resume is required"
            )

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            if (
                type(request.get("protocol_version")) is not int
                or request.get("protocol_version") != PROTOCOL_VERSION
            ):
                raise ContractError(
                    "protocol_version", "Use protocol version 1", "version_mismatch"
                )
            request_id = text(request.get("request_id"), "request_id")
            method = text(request.get("method"), "method")
            params = mapping(request.get("params", {}), "params")
            request_hash = content_hash({"method": method, "params": params})
            if method in READ_METHODS:
                result = self._route(method, params, request_id)
            else:
                with self.store.transaction():
                    previous = self.store.one(
                        "SELECT * FROM commands WHERE request_id=?", (request_id,)
                    )
                    if previous:
                        if previous["hash"] != request_hash:
                            raise ContractError(
                                "request_id", "ID already used with different content", "conflict"
                            )
                        return mapping(json.loads(previous["response"]), "response")
                    result = self._route(method, params, request_id)
                    response = {"ok": True, "request_id": request_id, "result": result}
                    self.store.db.execute(
                        "INSERT INTO commands VALUES(?,?,?,?,?)",
                        (request_id, method, request_hash, canonical(response), time.time_ns()),
                    )
            return {"ok": True, "request_id": request_id, "result": result}
        except ContractError as exc:
            return {"ok": False, "request_id": request.get("request_id"), "error": exc.dictionary()}
        except (OSError, sqlite3.Error) as exc:
            self._storage_failure(str(exc))
            return {"ok": False, "error": {"code": "storage_failure", "message": str(exc)}}

    def _route(self, method: str, params: dict[str, Any], operation_id: str) -> Any:
        if method == "engine.status":
            campaigns = self.store.query("campaigns", {}, limit=100, summary=True)
            active = []
            for identity in self.active:
                attempt = self.store.attempt(identity)
                experiment = self.store.one(
                    "SELECT * FROM experiments WHERE id=?", (attempt["experiment_id"],)
                )
                active.append(
                    {
                        "attempt": attempt,
                        "experiment": experiment,
                        "metrics": self.metrics.get(identity, {}),
                    }
                )
            return {
                "backend": "synthetic" if self.backend.synthetic else "gazebo",
                "workspace": str(self.workspace),
                "current_campaign": self.current,
                "campaigns": campaigns,
                "active": active,
                "event_sequence": self._last_sequence(),
                "export_id": self.exports.identity,
                "image_id": getattr(self.backend, "runtime", {}).get("image"),
                "diagnostics": getattr(self.backend, "diagnostics", None),
            }
        if method == "dataset.export":
            if self.current or self.active or self.job or self.exports.identity:
                raise ContractError("campaign_id", "Export requires an idle engine", "conflict")
            return self.exports.submit(text(params.get("campaign_id"), "campaign_id"))
        if method == "exports.get":
            return self.exports.get(text(params.get("export_id"), "export_id"))
        if method == "catalog.get":
            return (
                self.library.catalog()
                if self.library
                else {"source": "synthetic", "robots": [], "worlds": []}
            )
        if method == "campaign.preview":
            spec = self.resolve(mapping(params.get("specification"), "specification"))
            return {
                "count": spec.count,
                "expected_minimum_attempts": spec.count,
                "specification": spec.dictionary(),
                "timing_retry": "One final sequential replacement",
            }
        if method == "campaign.repeat":
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if campaign["state"] != "FINISHED":
                raise ContractError("campaign_id", "Choose a finished campaign", "conflict")
            specification = json.loads(campaign["specification"])
            specification.update(name=specification["name"] + " copy", source_campaign_id=identity)
            return {"specification": specification, "source_campaign_id": identity}
        if method == "campaign.submit":
            spec = self.resolve(mapping(params.get("specification"), "specification"))
            identity = self.store.submit(spec)
            return {
                "campaign_id": identity,
                "experiments": spec.count,
                "state": "QUEUED",
                "operation_id": operation_id,
                "completed": True,
            }
        if method == "campaign.move":
            identity = text(params.get("campaign_id"), "campaign_id")
            direction = params.get("direction")
            if direction not in ("up", "down", "first", "last"):
                raise ContractError("direction", "Use up, down, first or last")
            if self.store.campaign(identity)["state"] != "QUEUED":
                raise ContractError("campaign_id", "Only queued campaigns can move", "conflict")
            rows = self.store.rows(
                "SELECT id,queue_position FROM campaigns WHERE state='QUEUED' ORDER BY queue_position,id"
            )
            ordered = [row["id"] for row in rows]
            index = ordered.index(identity)
            target = {
                "up": max(0, index - 1),
                "down": min(len(rows) - 1, index + 1),
                "first": 0,
                "last": len(rows) - 1,
            }[direction]
            ordered.insert(target, ordered.pop(index))
            self.store.db.executemany(
                "UPDATE campaigns SET queue_position=? WHERE id=?",
                ((row["queue_position"], cid) for row, cid in zip(rows, ordered, strict=True)),
            )
            self.store.event("campaign.reordered", {"direction": direction}, identity)
            return {"operation_id": operation_id, "completed": True, "campaign_id": identity}
        if method == "reports.generate":
            from .analysis import DetectorSettings
            from dataclasses import asdict

            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if (
                self.current
                or self.active
                or self.job
                or self.exports.identity
                or campaign["state"] != "FINISHED"
            ):
                raise ContractError(
                    "campaign_id",
                    "Reports require finished execution and an idle engine",
                    "conflict",
                )
            settings = DetectorSettings.accept(params.get("detectors"))
            report_id, token = str(uuid.uuid4()), uuid.uuid4().hex
            self.store.db.execute(
                "INSERT INTO reports(id,campaign_id,state,specification,created_ns) VALUES(?,?,?,?,?)",
                (
                    report_id,
                    identity,
                    "QUEUED",
                    canonical(
                        dict(
                            token=token,
                            kind="campaign_report",
                            schema_version=DATASET_VERSION,
                            detectors=asdict(settings),
                        )
                    ),
                    time.time_ns(),
                ),
            )
            self.current, self.job_id = identity, report_id
            self.store.db.execute(
                "UPDATE campaigns SET report_state='QUEUED' WHERE id=?", (identity,)
            )
            self.store.set_state(identity, "REPORTING")
            return dict(accepted=True, operation_id=operation_id, report_id=report_id)
        if method == "reports.get":
            from .datasets import inside

            identity = text(params.get("report_id"), "report_id")
            report = self.store.one("SELECT * FROM reports WHERE id=?", (identity,))
            if not report:
                raise ContractError("report_id", "Report does not exist", "not_found")
            offset = integer(params.get("offset", 0), "offset", 0, 2**63 - 1)
            limit = integer(params.get("limit", 100), "limit", 1, 250)
            prefix = (
                str(inside(self.workspace, "reports/" + report["campaign_id"] + "/" + identity))
                + "/"
            )
            clauses, args = (
                "campaign_id=? AND path>=? AND path<?",
                [report["campaign_id"], prefix, prefix[:-1] + "0"],
            )
            if params.get("attempt_id") is not None:
                clauses += " AND attempt_id=?"
                args.append(text(params["attempt_id"], "attempt_id"))
            total = self.store.db.execute(
                "SELECT count(*) FROM artifacts WHERE " + clauses, args
            ).fetchone()[0]
            path = Path(report["path"]) if report["path"] else None
            return dict(
                report=report,
                manifest=json.loads(path.read_bytes()) if path and path.exists() else None,
                artifacts=self.store.rows(
                    "SELECT kind,path,attempt_id,bytes FROM artifacts WHERE "
                    + clauses
                    + " ORDER BY path LIMIT ? OFFSET ?",
                    (*args, limit, offset),
                ),
                total=total,
                offset=offset,
                next_offset=offset + limit if offset + limit < total else None,
            )
        if method in ("campaign.start", "campaign.resume"):
            if self.exports.identity:
                raise ContractError(
                    "campaign_id", "Wait for the active export to finish", "conflict"
                )
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if self.current and self.current != identity:
                raise ContractError("campaign_id", "Another campaign owns execution", "conflict")
            allowed = (
                ("QUEUED",)
                if method.endswith("start")
                else ("PAUSED", "STOPPED", "RECOVERY_REQUIRED", "BLOCKED")
            )
            if campaign["state"] not in allowed:
                raise ContractError(
                    "campaign_id",
                    "Campaign cannot start/resume from " + campaign["state"],
                    "conflict",
                )
            self.current, self.queue_enabled = identity, True
            self.store.db.execute(
                "UPDATE experiments SET held=0 WHERE campaign_id=? AND state='NOT_DONE'",
                (identity,),
            )
            self.store.set_state(identity, "RUNNING")
            return {"operation_id": operation_id, "accepted": True, "state": "RUNNING"}
        if method == "campaign.pause":
            identity = text(params.get("campaign_id"), "campaign_id")
            state = self.store.campaign(identity)["state"]
            if self.current != identity or state not in ("RUNNING", "RETRYING"):
                raise ContractError("campaign_id", "Pause requires active execution", "conflict")
            self.queue_enabled = False
            self.store.set_state(identity, "PAUSING" if self.active else "PAUSED")
            return {"operation_id": operation_id, "accepted": True}
        if method == "campaign.end":
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if self.current != identity or campaign["state"] == "REPORTING":
                raise ContractError(
                    "campaign_id", "End requires an active or paused campaign", "conflict"
                )
            self.queue_enabled = False
            self.store.set_state(identity, "STOPPING")
            self.store.db.execute(
                "UPDATE experiments SET state='NOT_DONE' WHERE campaign_id=? AND state='PENDING'",
                (identity,),
            )
            for attempt_id in self.active:
                self._request_discard(attempt_id, False, operation_id)
            return {"operation_id": operation_id, "accepted": True}
        if method in ("experiment.stop", "experiment.restart"):
            experiment_id = text(params.get("experiment_id"), "experiment_id")
            selected_attempt = self.store.one(
                "SELECT * FROM attempts WHERE experiment_id=? AND phase!='FINISHED'",
                (experiment_id,),
            )
            if not selected_attempt or selected_attempt["id"] not in self.active:
                raise ContractError("experiment_id", "Experiment has no active attempt", "conflict")
            self._request_discard(selected_attempt["id"], method.endswith("restart"), operation_id)
            return {
                "operation_id": operation_id,
                "accepted": True,
                "attempt_id": selected_attempt["id"],
            }
        if method == "campaign.status":
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            campaign["specification"] = json.loads(campaign["specification"])
            return {"campaign": campaign, "counters": self.store.counters(identity)}
        if method == "results.query":
            return self.store.query(
                params.get("table", "experiments"),
                mapping(params.get("filters", {}), "filters"),
                params.get("offset", 0),
                params.get("limit", 100),
                summary=params.get("summary", False),
                cursor=params.get("cursor"),
            )
        if method == "results.get":
            table = params.get("table")
            if table not in ("attempts", "experiments"):
                raise ContractError("table", "Choose attempts or experiments")
            row = self.store.one(
                f"SELECT * FROM {table} WHERE id=?", (text(params.get("id"), "id"),)
            )
            if row is None:
                raise ContractError("id", "No result with this identity", "not_found")
            for field in ("specification", "configuration", "metrics"):
                if row.get(field):
                    row[field] = json.loads(row[field])
            return row
        if method == "measurements.query":
            attempt = self.store.attempt(text(params.get("attempt_id"), "attempt_id"))
            if attempt["discarded"]:
                raise ContractError("attempt_id", "Measurements were discarded", "discarded")
            from .datasets import query

            return query(self.workspace, params)
        if method == "events.read":
            after = integer(params.get("after_sequence", 0), "after_sequence", 0, 2**63 - 1)
            limit = integer(params.get("limit", 200), "limit", 1, 500)
            current = self._last_sequence()
            if after > current:
                return {"events": [], "last_sequence": current, "snapshot_required": True}
            rows = self.store.rows(
                "SELECT * FROM events WHERE sequence>? ORDER BY sequence LIMIT ?", (after, limit)
            )
            for row in rows:
                row["payload"] = json.loads(row["payload"])
            return {
                "events": rows,
                "last_sequence": rows[-1]["sequence"] if rows else after,
                "snapshot_required": False,
            }
        if method == "inspection.get":
            identity = text(params.get("attempt_id"), "attempt_id")
            if identity not in self.active:
                raise ContractError("attempt_id", "Worker is no longer active", "not_found")
            return dict(
                self.active[identity].inspection(
                    integer(params.get("after_sequence", 0), "after_sequence", 0, 2**63 - 1),
                    integer(params.get("map_revision", -1), "map_revision", -1, 2**63 - 1),
                ),
                attempt_id=identity,
            )
        if method == "logs.read":
            return self._logs(params)
        if method == "command.status":
            identity = text(params.get("request_id"), "request_id")
            command = self.store.one("SELECT * FROM commands WHERE request_id=?", (identity,))
            if not command:
                raise ContractError("request_id", "No accepted command", "not_found")
            return {"request_id": identity, "response": json.loads(command["response"])}
        if self.library and method.startswith(("profiles.", "spawns.", "parameters.")):
            return self.library.handle(method, params, self.store)
        raise ContractError("method", "Unsupported operation: " + method, "unsupported")

    def _last_sequence(self) -> int:
        return int(
            self.store.db.execute("SELECT COALESCE(MAX(sequence),0) FROM events").fetchone()[0]
        )

    def _logs(self, params: dict[str, Any]) -> dict[str, Any]:
        attempt = self.store.attempt(text(params.get("attempt_id"), "attempt_id"))
        if attempt["discarded"]:
            raise ContractError("attempt_id", "Partial measurements were discarded", "discarded")
        component = params.get("component", "worker")
        if component not in ("worker", "simulation", "robot", "navigation", "slam", "explorer"):
            raise ContractError("component", "Choose a recorded component")
        offset = integer(params.get("offset", 0), "offset", 0, 2**63 - 1)
        limit = integer(params.get("limit", 65536), "limit", 1, 262144)
        path = Path(attempt["directory"]) / (component + ".log")
        if not path.is_file() and not path.with_suffix(".log.zst").is_file():
            return {"text": "", "next_offset": offset, "available": False}
        from .datasets import read_log

        data = read_log(path, offset, limit)
        return {
            "text": data.decode("utf-8", errors="replace"),
            "next_offset": offset + len(data),
            "available": True,
        }

    def _request_discard(self, identity: str, restart: bool, operation_id: str) -> None:
        attempt = self.store.attempt(identity)
        self.store.db.execute(
            "UPDATE attempts SET discard_intent=1,restart_intent=?,phase='STOPPING' WHERE id=?",
            (int(restart), identity),
        )
        self.store.event(
            "attempt.discard_requested",
            {"restart": restart, "operation_id": operation_id},
            attempt["campaign_id"],
            identity,
        )

    def _discard(self, attempt: dict[str, Any]) -> None:
        directory = Path(attempt["directory"]).resolve()
        if directory != (self.workspace / "attempts" / attempt["id"]).resolve():
            raise ContractError("directory", "Attempt cleanup path is outside its owned directory")
        shutil.rmtree(directory, ignore_errors=False) if directory.exists() else None
        if directory.parent.exists():
            sync_directory(directory.parent)
        with self.store.transaction():
            self.store.db.execute("DELETE FROM artifacts WHERE attempt_id=?", (attempt["id"],))
            self.store.db.execute(
                "DELETE FROM events WHERE attempt_id=? AND kind!='attempt.discard_requested'",
                (attempt["id"],),
            )
            self.store.db.execute(
                "UPDATE attempts SET phase='FINISHED',outcome='manual_discard',"
                "discarded=1,retained=0,configuration='{}',metrics=NULL,directory='',detail='',"
                "pid=NULL,pid_created=NULL,terminal_ns=?,cleanup_ns=? WHERE id=?",
                (time.time_ns(), time.time_ns(), attempt["id"]),
            )
            self.store.db.execute(
                "UPDATE experiments SET state='NOT_DONE',held=? WHERE id=?",
                (0 if attempt["restart_intent"] else 1, attempt["experiment_id"]),
            )
            self.store.event(
                "attempt.discarded",
                {"restart": bool(attempt["restart_intent"])},
                attempt["campaign_id"],
                attempt["id"],
            )

    def _reserve(self, experiment: dict[str, Any], retry: bool, slot: int) -> dict[str, Any]:
        identity, token = str(uuid.uuid4()), uuid.uuid4().hex
        directory = self.workspace / "attempts" / identity
        with self.store.transaction():
            number = self.store.db.execute(
                "SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE experiment_id=?",
                (experiment["id"],),
            ).fetchone()[0]
            self.store.db.execute(
                "INSERT INTO attempts(id,experiment_id,campaign_id,number,slot,domain,token,"
                "partition,phase,retry,reserved_ns,configuration,directory) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    experiment["id"],
                    experiment["campaign_id"],
                    number,
                    slot,
                    61 + slot,
                    token,
                    "suite_" + token,
                    "STARTING",
                    int(retry),
                    time.time_ns(),
                    experiment["specification"],
                    str(directory),
                ),
            )
            self.store.db.execute(
                "UPDATE experiments SET state='ACTIVE' WHERE id=?", (experiment["id"],)
            )
            self.store.event(
                "attempt.reserved",
                {"slot": slot, "retry": retry},
                experiment["campaign_id"],
                identity,
            )
        directory.mkdir(parents=True, exist_ok=False)
        return self.store.attempt(identity)

    def _launch(
        self, experiment: dict[str, Any], retry: bool, slot: int, campaign: dict[str, Any]
    ) -> None:
        attempt = self._reserve(experiment, retry, slot)
        try:
            worker = self.backend.start(attempt, experiment, campaign)
            self.active[attempt["id"]] = worker
            self.timing[attempt["id"]] = TimingPolicy()
            with self.store.transaction():
                self.store.db.execute(
                    "UPDATE attempts SET pid=?,pid_created=?,started_ns=? WHERE id=?",
                    (worker.pid, worker.pid_created, time.time_ns(), attempt["id"]),
                )
                self.store.event(
                    "attempt.launched",
                    {"pid": worker.pid, "overlapping_attempts": list(self.active)},
                    campaign["id"],
                    attempt["id"],
                )
                if len(self.active) > 1:
                    for active_id in self.active:
                        if active_id not in self.parallel_exposed:
                            self.store.event(
                                "attempt.parallel_exposure",
                                {
                                    "overlapping_attempts": list(self.active),
                                    "scope": "owned preparation, simulation or finalization resources",
                                },
                                attempt["campaign_id"],
                                active_id,
                            )
                            self.parallel_exposed.add(active_id)
        except Exception as exc:
            with self.store.transaction():
                self.store.db.execute(
                    "UPDATE attempts SET outcome='startup_failure',detail=? WHERE id=?",
                    (str(exc), attempt["id"]),
                )
            self._finish(self.store.attempt(attempt["id"]))

    def _finish(self, attempt: dict[str, Any]) -> None:
        if attempt["discard_intent"]:
            self._discard(attempt)
            return
        outcome = attempt["outcome"] or "runtime_failure"
        directory = Path(attempt["directory"])
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            atomic_json(
                manifest_path,
                {
                    "schema_version": DATASET_VERSION,
                    "phase": "FINISHED",
                    "outcome": outcome,
                    "source": "synthetic" if self.backend.synthetic else "native",
                    "attempt_id": attempt["id"],
                    "experiment_id": attempt["experiment_id"],
                    "detail": attempt["detail"],
                    "configuration": json.loads(attempt["configuration"]),
                    "final_metrics": json.loads(attempt["metrics"] or "{}"),
                },
            )
        with self.store.transaction():
            self.store.db.execute(
                "UPDATE attempts SET phase='FINISHED',outcome=?,cleanup_ns=?,"
                "terminal_ns=COALESCE(terminal_ns,?) WHERE id=?",
                (outcome, time.time_ns(), time.time_ns(), attempt["id"]),
            )
            self._record_artifacts(attempt, manifest_path)
            parallel = (
                self.store.one(
                    "SELECT 1 FROM events WHERE attempt_id=? AND kind='attempt.parallel_exposure' LIMIT 1",
                    (attempt["id"],),
                )
                is not None
            )
            if outcome == "timing_rejection" and not attempt["retry"] and parallel:
                self.store.db.execute(
                    "UPDATE experiments SET state='RETRY',retry_used=1 WHERE id=?",
                    (attempt["experiment_id"],),
                )
                self.store.db.execute(
                    "INSERT OR REPLACE INTO retry_queue VALUES(?,?,?,?)",
                    (
                        attempt["experiment_id"],
                        attempt["campaign_id"],
                        time.time_ns(),
                        attempt["id"],
                    ),
                )
            else:
                self.store.db.execute(
                    "DELETE FROM retry_queue WHERE experiment_id=?", (attempt["experiment_id"],)
                )
                self.store.db.execute(
                    "UPDATE experiments SET state=?,outcome=?,final_attempt_id=? WHERE id=?",
                    (
                        "UNRESOLVED" if outcome == "timing_rejection" else "DONE",
                        outcome,
                        attempt["id"],
                        attempt["experiment_id"],
                    ),
                )
            self.store.event(
                "attempt.finished",
                {"outcome": outcome, "retry": bool(attempt["retry"])},
                attempt["campaign_id"],
                attempt["id"],
            )

    def _record_artifacts(self, attempt: dict[str, Any], manifest: Path) -> None:
        directory = manifest.parent
        finalization = directory / "finalization.json"
        entries = (
            json.loads(finalization.read_bytes())["artifacts"] if finalization.exists() else []
        )
        if not any(entry["kind"] == "manifest" for entry in entries):
            entries.append(
                dict(
                    kind="manifest",
                    path=manifest.name,
                    sha256=file_hash(manifest),
                    bytes=manifest.stat().st_size,
                )
            )
        recovery = directory / "recovery.json"
        if recovery.exists() and not any(entry["path"] == recovery.name for entry in entries):
            entries.append(
                dict(
                    kind="checkpoint_index",
                    path=recovery.name,
                    sha256=file_hash(recovery),
                    bytes=recovery.stat().st_size,
                )
            )
        for entry in entries:
            self.store.db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    attempt["campaign_id"],
                    attempt["id"],
                    entry["kind"],
                    str(directory / entry["path"]),
                    entry["sha256"],
                    entry["bytes"],
                    entry.get("rows"),
                    time.time_ns(),
                    canonical(entry),
                ),
            )

    def tick(self) -> None:
        if self.exports.identity:
            if self.exports.tick():
                self.queue_enabled = False
            return
        if self.report_import is not None:
            try:
                self._import_report()
            except (OSError, ValueError, KeyError) as error:
                self._report_failure(
                    str(error), storage=isinstance(error, OSError) and error.errno in (28, 122)
                )
            return
        if self.job is not None:
            try:
                self._poll_job()
            except (OSError, ValueError, KeyError) as error:
                self._report_failure(
                    str(error), storage=isinstance(error, OSError) and error.errno in (28, 122)
                )
            return
        if self.current and self.job_id:
            report = self.store.one("SELECT * FROM reports WHERE id=?", (self.job_id,))
            if report and report["state"] == "QUEUED":
                self._start_report(report)
                return
        for identity, worker in tuple(self.active.items()):
            attempt = self.store.attempt(identity)
            if attempt["discard_intent"]:
                worker.stop("manual_discard", "Manual control")
            for event in worker.poll():
                kind = event.get("kind")
                if kind == "metrics":
                    self.metrics[identity] = event
                    if not self.backend.synthetic and attempt["phase"] == "RUNNING":
                        policy = self.timing[identity]
                        if policy.update(event["wall_monotonic"], event["clock"]):
                            worker.stop("timing_rejection", "RTF below 0.90 for ten wall seconds")
                elif kind == "state":
                    phase = event["phase"]
                    if phase in ("STARTING", "READY", "RUNNING", "STOPPING", "FINALIZING"):
                        self.store.db.execute(
                            "UPDATE attempts SET phase=? WHERE id=?", (phase, identity)
                        )
                        self.store.event("attempt.state", event, attempt["campaign_id"], identity)
                elif kind == "terminal":
                    self.store.db.execute(
                        "UPDATE attempts SET phase='FINALIZING',outcome=?,detail=?,"
                        "metrics=?,terminal_ns=COALESCE(terminal_ns,?) WHERE id=?",
                        (
                            event["outcome"],
                            event.get("detail", ""),
                            canonical(event.get("metrics", self.metrics.get(identity, {}))),
                            time.time_ns(),
                            identity,
                        ),
                    )
                    self.store.event("attempt.terminal", event, attempt["campaign_id"], identity)
                    if event["outcome"] == "storage_failure":
                        self._storage_failure(event.get("detail", "Dataset storage failed"))
                elif kind == "cleanup_failure":
                    self._storage_failure(event["detail"])
            if worker.closed:
                self._finish(self.store.attempt(identity))
                del self.active[identity]
                self.metrics.pop(identity, None)
                self.timing.pop(identity, None)
                self.parallel_exposed.discard(identity)
        if not self.current:
            if self.queue_enabled:
                next_campaign = self.store.one(
                    "SELECT id FROM campaigns WHERE state='QUEUED' ORDER BY queue_position LIMIT 1"
                )
                if next_campaign:
                    self.current = next_campaign["id"]
                    self.store.set_state(self.current, "RUNNING")
                else:
                    self.queue_enabled = False
            if not self.current:
                return
        campaign = self.store.campaign(self.current)
        if campaign["state"] in ("PAUSING", "STOPPING"):
            if not self.active:
                self.store.set_state(
                    self.current, "PAUSED" if campaign["state"] == "PAUSING" else "STOPPED"
                )
            return
        if campaign["state"] not in ("RUNNING", "RETRYING"):
            return
        spec = json.loads(campaign["specification"])
        occupied = {self.store.attempt(identity)["slot"] for identity in self.active}
        slots = range(1 if campaign["state"] == "RETRYING" else spec["concurrency"])
        for slot in slots:
            if slot in occupied:
                continue
            experiment = self.store.one(
                "SELECT e.* FROM experiments e LEFT JOIN retry_queue r ON e.id=r.experiment_id "
                "WHERE e.campaign_id=? AND e.state IN ('PENDING','NOT_DONE') AND e.held=0 "
                "AND r.experiment_id IS NULL ORDER BY e.position LIMIT 1",
                (self.current,),
            )
            retry = False
            if not experiment and not self.active:
                experiment = self.store.one(
                    "SELECT e.* FROM retry_queue r JOIN experiments e ON r.experiment_id=e.id "
                    "WHERE r.campaign_id=? AND e.state IN ('RETRY','NOT_DONE') AND e.held=0 "
                    "ORDER BY r.queued_ns LIMIT 1",
                    (self.current,),
                )
                if experiment:
                    retry = True
                    self.store.set_state(self.current, "RETRYING")
            if experiment:
                self._launch(experiment, retry, slot, campaign)
                if retry:
                    break
        remaining = self.store.one(
            "SELECT COUNT(*) count FROM experiments WHERE campaign_id=? "
            "AND state IN ('PENDING','NOT_DONE','RETRY','ACTIVE')",
            (self.current,),
        )
        if not self.active and remaining and remaining["count"] == 0:
            self.store.db.execute(
                "UPDATE campaigns SET execution_finished_ns=?,report_state='NOT_REQUESTED' WHERE id=?",
                (time.time_ns(), self.current),
            )
            if self.backend.synthetic:
                self.store.set_state(self.current, "FINISHED")
                self.current = None
            else:
                self._begin_finalization(self.current)
        elif not self.active:
            held = self.store.one(
                "SELECT COUNT(*) count FROM experiments WHERE campaign_id=? AND held=1",
                (self.current,),
            )
            if held and held["count"]:
                self.queue_enabled = False
                self.store.set_state(self.current, "PAUSED", "Not done experiments await Resume")

    def _begin_finalization(self, campaign_id: str) -> None:
        identity, token = str(uuid.uuid4()), uuid.uuid4().hex
        specification = {
            "token": token,
            "kind": "campaign_report",
            "schema_version": DATASET_VERSION,
        }
        with self.store.transaction():
            self.store.db.execute(
                "INSERT INTO reports(id,campaign_id,state,specification,created_ns) VALUES(?,?,?,?,?)",
                (identity, campaign_id, "COMPACTING", canonical(specification), time.time_ns()),
            )
            self.store.db.execute(
                "UPDATE campaigns SET report_state='COMPACTING' WHERE id=?", (campaign_id,)
            )
            self.store.set_state(campaign_id, "REPORTING")
        self.job_id = identity
        self.pending_compactions = [
            json.loads(row["metadata"])
            for row in self.store.rows(
                "SELECT path,metadata FROM artifacts WHERE campaign_id=? AND kind='dataset_index' ORDER BY committed_ns",
                (campaign_id,),
            )
            if not (Path(row["path"]).parent / "commit.json").exists()
        ]
        if self.pending_compactions:
            self._start_job("recover_commit", self.pending_compactions.pop(0), token)
        else:
            self._start_job(
                "compact", {"workspace": str(self.workspace), "campaign_id": campaign_id}, token
            )

    def _start_job(self, stage: str, payload: dict[str, Any], token: str) -> None:
        from .jobs import Job

        self.job_stage = stage
        try:
            self.job = Job("commit" if stage == "recover_commit" else stage, payload, token)
        except Exception as error:
            self.store.db.execute(
                "UPDATE reports SET state='FAILED',detail=?,finished_ns=? WHERE id=?",
                (str(error), time.time_ns(), self.job_id),
            )
            self.store.db.execute(
                "UPDATE campaigns SET report_state='FAILED' WHERE id=?", (self.current,)
            )
            if self.current:
                self.store.set_state(
                    self.current, "FINISHED", "Execution complete; analysis process could not start"
                )
            self.current, self.job_id = None, None
            return
        report = self.store.one("SELECT specification FROM reports WHERE id=?", (self.job_id,))
        assert report is not None
        specification = dict(
            json.loads(report["specification"]), pid=self.job.pid, pid_created=self.job.created
        )
        self.store.db.execute(
            "UPDATE reports SET specification=? WHERE id=?", (canonical(specification), self.job_id)
        )

    def _poll_job(self) -> None:
        result = self.job.poll()
        if result is None:
            return
        self.job.close()
        self.job = None
        report = self.store.one("SELECT * FROM reports WHERE id=?", (self.job_id,))
        assert report is not None
        campaign_id = report["campaign_id"]
        if not result["ok"]:
            self._report_failure(result["detail"], storage=result.get("errno") in (28, 122))
            return
        if self.job_stage == "recover_commit":
            token = json.loads(report["specification"])["token"]
            if self.pending_compactions:
                self._start_job("recover_commit", self.pending_compactions.pop(0), token)
            else:
                self._start_job(
                    "compact", {"workspace": str(self.workspace), "campaign_id": campaign_id}, token
                )
        elif self.job_stage == "compact":
            reference = result["result"]
            index = self.workspace / reference["index"]
            if file_hash(index) != reference["sha256"]:
                raise OSError("Prepared dataset index failed integrity validation")
            with self.store.transaction():
                self.store.db.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        campaign_id,
                        None,
                        "dataset_index",
                        str(index),
                        reference["sha256"],
                        index.stat().st_size,
                        None,
                        time.time_ns(),
                        canonical(reference),
                    ),
                )
                for shard in json.loads(index.read_bytes())["shards"]:
                    self.store.db.execute(
                        "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            str(uuid.uuid4()),
                            campaign_id,
                            None,
                            "dataset",
                            str(self.workspace / shard["path"]),
                            shard["sha256"],
                            shard["bytes"],
                            shard["rows"],
                            time.time_ns(),
                            canonical({"kind": shard["kind"]}),
                        ),
                    )
                self.store.db.execute(
                    "UPDATE reports SET state='COMMITTING' WHERE id=?", (self.job_id,)
                )
            self._start_job("commit", reference, json.loads(report["specification"])["token"])
        elif self.job_stage == "commit":
            self._start_report(report)
        else:
            from .datasets import inside

            reference = result["result"]
            index = inside(self.workspace, reference["index"])
            manifest = inside(self.workspace, reference["manifest"])
            if (
                file_hash(index) != reference["sha256"]
                or file_hash(manifest) != reference["manifest_sha256"]
            ):
                raise OSError("Generated report failed integrity validation")
            self.report_reference = reference
            self.report_import = index.open(encoding="utf-8")
            self.store.db.execute(
                "UPDATE reports SET state='INDEXING',path=? WHERE id=?",
                (str(manifest), self.job_id),
            )

    def _start_report(self, report: dict[str, Any]) -> None:
        specification = json.loads(report["specification"])
        self.store.db.execute("UPDATE reports SET state='RUNNING' WHERE id=?", (report["id"],))
        self.store.db.execute(
            "UPDATE campaigns SET report_state='RUNNING' WHERE id=?", (report["campaign_id"],)
        )
        self._start_job(
            "report",
            dict(
                workspace=str(self.workspace),
                campaign_id=report["campaign_id"],
                report_id=report["id"],
                detectors=specification.get("detectors", {}),
            ),
            specification["token"],
        )

    def _import_report(self) -> None:
        from .datasets import inside

        assert self.report_import is not None
        campaign_id = self.current
        assert campaign_id is not None
        finished = False
        with self.store.transaction():
            for _ in range(100):
                line = self.report_import.readline()
                if not line:
                    finished = True
                    break
                entry = json.loads(line)
                path = inside(self.workspace, entry["path"])
                if file_hash(path) != entry["sha256"]:
                    raise OSError("Report artifact failed integrity validation")
                self.store.db.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(uuid.uuid4()),
                        campaign_id,
                        entry.get("attempt_id"),
                        entry["kind"],
                        str(path),
                        entry["sha256"],
                        entry["bytes"],
                        None,
                        time.time_ns(),
                        canonical(entry),
                    ),
                )
        if finished:
            self.report_import.close()
            self.report_import = None
            with self.store.transaction():
                self.store.db.execute(
                    "UPDATE reports SET state='FINISHED',finished_ns=? WHERE id=?",
                    (time.time_ns(), self.job_id),
                )
                self.store.db.execute(
                    "UPDATE campaigns SET report_state='FINISHED' WHERE id=?", (campaign_id,)
                )
                self.store.set_state(campaign_id, "FINISHED")
            self.current, self.job_id = None, None

    def _storage_failure(self, detail: str) -> None:
        self.queue_enabled = False
        if self.report_import is not None:
            self._report_failure(detail, storage=True)
            return
        for worker in self.active.values():
            worker.stop("storage_failure", detail)
        if self.current:
            try:
                self.store.set_state(self.current, "BLOCKED", detail)
            except sqlite3.Error:
                pass

    def _remove_report_output(self, report: dict[str, Any]) -> None:
        from .datasets import inside

        directory = inside(self.workspace, "reports/" + report["campaign_id"] + "/" + report["id"])
        prefix = str(directory) + "/"
        self.store.db.execute(
            "DELETE FROM artifacts WHERE campaign_id=? AND path>=? AND path<?",
            (report["campaign_id"], prefix, prefix[:-1] + "0"),
        )
        if directory.exists():
            shutil.rmtree(directory)

    def _report_failure(self, detail: str, *, storage: bool = False) -> None:
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.report_import is not None:
            self.report_import.close()
            self.report_import = None
        report = self.store.one("SELECT * FROM reports WHERE id=?", (self.job_id,))
        assert report is not None
        try:
            self._remove_report_output(report)
        except OSError as error:
            storage = True
            detail += "; report cleanup requires repair: " + str(error)
        with self.store.transaction():
            self.store.db.execute(
                "UPDATE reports SET state='FAILED',detail=?,finished_ns=? WHERE id=?",
                (detail, time.time_ns(), self.job_id),
            )
            self.store.db.execute(
                "UPDATE campaigns SET report_state='FAILED' WHERE id=?", (report["campaign_id"],)
            )
            self.store.set_state(
                report["campaign_id"], "FINISHED", "Execution finished; reporting failed: " + detail
            )
        if storage:
            self.queue_enabled = False
        self.current, self.job_id = None, None

    def run(self, socket_path: Path | None = None) -> None:
        self.server = Server(socket_path or self.workspace / "runtime/control.sock", self.requests)
        try:
            while not self.closed:
                for _ in range(32):
                    try:
                        pending = self.requests.get_nowait()
                    except queue.Empty:
                        break
                    if pending.future.done():
                        continue
                    try:
                        pending.future.set_result(self.handle(pending.request))
                    except Exception as exc:
                        pending.future.set_result(
                            {"ok": False, "error": {"code": "runtime_error", "message": str(exc)}}
                        )
                try:
                    self.tick()
                except (OSError, sqlite3.Error) as exc:
                    self._storage_failure(str(exc))
                time.sleep(0.02)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.closed = True
        if self.server:
            self.server.close()
        self.exports.close()
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.report_import is not None:
            self.report_import.close()
            self.report_import = None
        for worker in self.active.values():
            worker.stop("engine_interruption", "Engine shutdown")
        deadline = time.monotonic() + 50
        while self.active and time.monotonic() < deadline:
            for identity, worker in tuple(self.active.items()):
                worker.poll()
                if worker.closed:
                    del self.active[identity]
            time.sleep(0.02)
        if self.active:
            attempts = [self.store.attempt(identity) for identity in self.active]
            self.backend.recover(attempts)
            self.active.clear()
        self.store.close()
