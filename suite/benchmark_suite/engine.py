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
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import queue
import re
import shutil
import sqlite3
import time
from typing import Any
import uuid

from .backends import Backend, WorkerHandle
from .contracts import (
    CampaignSpec,
    DATASET_VERSION,
    ContractError,
    COMPATIBLE_PROTOCOLS,
    PROTOCOL_VERSION,
    STUDY_ID,
    UNASSIGNED,
    canonical,
    content_hash,
    integer,
    known_keys,
    mapping,
    text,
)
from . import provenance
from .deletion import Remover, campaign_ids, delete_rows, plan
from .monitoring import MONITOR_LAYERS
from .protocol import PendingRequest, Server
from .replay import WINDOW_LIMIT_NS, ReplayWindows
from .run_map import RunMaps
from .run_records import RunRecords
from .exports import Exports, studies_root
from .files import BUILDERS, TABLE_SUFFIXES, BackgroundReads, text as file_text
from .storage import atomic_json, file_hash, sync_directory
from .store import ACTIVE_STATES, BUSY_TIMEOUT_MS, TUNING_CAMPAIGN, Store
from .studies import (
    attempt_record,
    last_repetition,
    provenance_document,
    read_study,
    run_counts,
    runs_page,
    studies_page,
)
from .timelines import timeline, traces

READ_METHODS = frozenset(
    {
        "engine.status",
        "catalog.get",
        "parameters.resolve",
        "campaign.preview",
        "campaign.delete_preview",
        "campaign.status",
        "results.query",
        "events.read",
        "logs.read",
        "monitor.get",
        "profiles.list",
        "spawns.list",
        "command.status",
    }
)
READ_METHODS = READ_METHODS | {
    "measurements.query",
    "spawns.map",
    "profiles.export",
    "spawns.export",
    "reports.get",
    "profiles.get",
    "spawns.get",
    "results.get",
    "results.map",
    "replay.window",
    "campaign.repeat",
    "study.export_status",
    "studies.list",
    "study.get",
    "files.list",
    "files.table",
    "files.log",
    "results.timeline",
    "study.traces",
}
COPY_SUFFIX = re.compile(r" \(copy (\d+)\)$")
COLLECT_RETRY_S = (
    30.0  # a busy database is asked again for the object collection and VACUUM after this
)


def study_identity(value: Any, field: str) -> str:
    """A study ID as requests name it: a ``contracts.Study`` ID or ``unassigned``."""
    identity = text(value, field, maximum=64)
    if identity != UNASSIGNED and not STUDY_ID.match(identity):
        raise ContractError(field, "Unknown study identity")
    return identity


def is_tuning(specification: dict[str, Any]) -> bool:
    """Whether a stored campaign specification is a tuning run (``CampaignSpec.tuning``)."""
    return specification.get("tuning") is True


def previewed(specification: dict[str, Any]) -> dict[str, Any]:
    """A preview checks the plan as it is now: a plan hash the specification carries (an exported or accepted
    campaign) is the engine's to set again here, and is compared only at submission."""
    study = specification.get("study")
    if not isinstance(study, dict) or "plan_sha256" not in study:
        return specification
    return dict(specification, study={k: v for k, v in study.items() if k != "plan_sha256"})


class Engine:
    def __init__(
        self,
        workspace: Path,
        backend: Backend,
        resolve: Callable[[dict[str, Any]], CampaignSpec] | None = None,
        library: Any = None,
    ):
        self.store = Store(workspace)
        self.workspace = self.store.workspace
        # Code identity and run environment of this engine (provenance.py); run records name it.
        self.reservations = 0  # attempts reserved by this engine process (runs_since_engine_start)
        self.provenance_id = self.store.put_provenance(provenance.document(backend.root))
        self.backend, self.library = backend, library
        self.resolve = resolve or CampaignSpec.accepted
        if not backend.synthetic and resolve is None:
            self.store.close()
            raise ValueError("Native workers require catalog and dependency preflight")
        self.active: dict[str, WorkerHandle] = {}
        self.metrics: dict[str, dict[str, Any]] = {}
        self.parallel_exposed: set[str] = set()
        self.current: str | None = None
        self.queue_enabled = False
        self.closed = False
        self.job: Any = None
        self.job_id: str | None = None
        self.job_stage = ""
        self.pending_compactions: list[dict[str, Any]] = []
        self.requests: queue.Queue[PendingRequest] = queue.Queue(maxsize=128)
        self.server: Server | None = None
        # Campaign deletion whose files are being removed: (campaign ID, remover).
        self.removal: tuple[str, Remover] | None = None
        # a deletion finished: collect orphan objects and VACUUM when idle (set at start when one finished after
        # the last collection); a busy database is tried again after COLLECT_RETRY_S
        self.collect_pending = False
        self.collect_retry_at = 0.0
        self.collected = (0, 0)  # objects and bytes removed by an attempt that has not finished yet
        self.run_maps = RunMaps(self.workspace)
        self.replay_windows = ReplayWindows(self.workspace)
        self.run_records = RunRecords(
            self.store,
            self.workspace,
            self.provenance_id,
            library.world_families() if library else None,
        )
        self.exports = Exports(self.store, self.workspace)
        self.reads = BackgroundReads()
        if library:
            library.bind(self.store)
        self._recover()

    def _recover(self) -> None:
        from .jobs import recover

        # Removals that failed are retried at every engine start; their rows are already gone.
        self.store.db.execute(
            "UPDATE deletions SET state='REMOVING',detail='' WHERE state='FAILED'"
        )
        for report in self.store.rows(
            # QUEUED, RUNNING and INDEXING: report rows of engines before R7
            "SELECT * FROM reports WHERE state IN ('QUEUED','COMPACTING','COMMITTING','RUNNING','INDEXING')"
        ):
            specification = json.loads(report["specification"])
            recover(
                specification.get("token", ""),
                specification.get("pid"),
                specification.get("pid_created"),
            )
            self._remove_report_output(report)  # a legacy report's partial files (before R7)
            self.store.db.execute(
                "UPDATE reports SET state='INTERRUPTED',detail='Engine restart; finalized again after Resume' "
                "WHERE id=?",
                (report["id"],),
            )
        # a deletion that finished after the last collection (an engine stopped before its idle tick)
        self.collect_pending = bool(
            self.store.one(
                "SELECT 1 FROM deletions WHERE state='DELETED' AND finished_ns>COALESCE((SELECT "
                "MAX(utc_ns) FROM events WHERE kind IN ('objects.collected','objects.collect_failed')),0)"
            )
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

                # A persisted stop request is honoured even when the engine stopped first.
                stopped = self._stop_requested(attempt["id"])
                outcome, detail = (
                    (
                        "manual_stop",
                        "Stopped by the researcher; engine interrupted before finalization",
                    )
                    if stopped
                    else (
                        "engine_interruption",
                        "Unexpected engine or host interruption; readable checkpoints retained",
                    )
                )
                try:
                    finalize_partial(
                        directory, outcome, detail, json.loads(attempt["metrics"] or "{}")
                    )
                except (OSError, ValueError) as error:
                    interrupted = json.loads(manifest.read_bytes())
                    interrupted["recovery_error"] = str(error)
                    atomic_json(manifest, interrupted)
                if stopped:
                    with self.store.transaction():
                        self.store.db.execute(
                            "UPDATE attempts SET outcome='manual_stop',detail=? WHERE id=?",
                            (detail, attempt["id"]),
                        )
                    self._finish(self.store.attempt(attempt["id"]))
                    continue
                with self.store.transaction():
                    self.store.db.execute(
                        "UPDATE attempts SET phase='FINISHED',outcome='engine_interruption',"
                        "terminal_ns=?,cleanup_ns=? WHERE id=?",
                        (time.time_ns(), time.time_ns(), attempt["id"]),
                    )
                    # Its files are incomplete: a stub record (MEASUREMENTS, run-record-v1).
                    self.run_records.stub(self.store.attempt(attempt["id"]), None)
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
            "SELECT id,state,specification FROM campaigns WHERE state IN ("
            + ",".join("?" * len(ACTIVE_STATES))
            + ")",
            ACTIVE_STATES,
        ):
            if campaign["state"] == "STOPPING" or is_tuning(json.loads(campaign["specification"])):
                self.store.set_state(campaign["id"], "STOPPED")
            else:
                self.store.set_state(
                    campaign["id"], "RECOVERY_REQUIRED", "Engine restarted; Resume is required"
                )

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            if (
                type(request.get("protocol_version")) is not int
                or request.get("protocol_version") not in COMPATIBLE_PROTOCOLS
            ):
                raise ContractError(
                    "protocol_version",
                    f"Use protocol version {PROTOCOL_VERSION}",
                    "version_mismatch",
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
                        "stop_requested": self._stop_requested(identity),
                    }
                )
            return {
                "backend": "synthetic" if self.backend.synthetic else "gazebo",
                "workspace": str(self.workspace),
                "current_campaign": self.current,
                "campaigns": campaigns,
                "active": active,
                "event_sequence": self._last_sequence(),
                "image_id": getattr(self.backend, "runtime", {}).get("image"),
                "diagnostics": getattr(self.backend, "diagnostics", None),
                "deletions": {
                    "removing": self.store.db.execute(
                        "SELECT count(*) FROM deletions WHERE state='REMOVING'"
                    ).fetchone()[0],
                    "failed": self.store.rows(
                        "SELECT campaign_id,name,detail FROM deletions WHERE state='FAILED' "
                        "ORDER BY requested_ns DESC LIMIT 20"
                    ),
                },
            }
        if method == "catalog.get":
            return (
                self.library.catalog()
                if self.library
                else {"source": "synthetic", "robots": [], "worlds": []}
            )
        if method == "campaign.preview":
            spec = self._in_study(
                self.resolve(previewed(mapping(params.get("specification"), "specification")))
            )
            return {
                "count": spec.count,
                "expected_minimum_attempts": spec.count,
                "specification": spec.dictionary(),
                "timing_retry": "One final sequential replacement",
            }
        if method == "studies.list":
            known_keys(params, {"offset", "limit"}, "params")
            return studies_page(self.store.db, params.get("offset", 0), params.get("limit", 100))
        if method in ("study.export", "study.export_status", "study.get"):
            study_id = study_identity(params.get("study_id"), "study_id")
            if method == "study.get":
                known_keys(params, {"study_id"}, "params")
                reply = self.reads.get(
                    ("study.get", study_id),
                    partial(read_study, self.workspace / "suite.sqlite3", study_id),
                    dict(study_id=study_id),
                )
                if reply["state"] == "ready":
                    reply["export_status"] = self.exports.status(study_id)
                return reply
            if method == "study.export_status":
                export_id = params.get("export_id")
                return self.exports.status(
                    study_id,
                    None if export_id is None else text(export_id, "export_id", maximum=64),
                )
            # RO-10: never beside runs, finalization (compaction) or a deletion (they write what it reads).
            busy = bool(
                self.active or self.current or self.queue_enabled or self.job or self.removal
            )
            return self.exports.start(study_id, busy=busy)
        if method == "campaign.repeat":
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if campaign["state"] != "FINISHED":
                raise ContractError("campaign_id", "Choose a finished campaign", "conflict")
            specification = json.loads(campaign["specification"])
            if is_tuning(specification):
                raise ContractError("campaign_id", "A tuning run cannot be repeated", "conflict")
            specification.update(name=specification["name"] + " copy", source_campaign_id=identity)
            return {"specification": specification, "source_campaign_id": identity}
        if method == "campaign.submit":
            spec = self._in_study(
                self.resolve(mapping(params.get("specification"), "specification"))
            )
            identity = self.store.submit(spec)
            self._note_repetitions(spec)
            return {
                "campaign_id": identity,
                "experiments": spec.count,
                "state": "QUEUED",
                "operation_id": operation_id,
                "completed": True,
            }
        if method == "campaign.delete_preview":
            entries = plan(self.store, campaign_ids(params.get("campaign_ids")), self.current)
            return {
                "campaigns": [
                    {key: value for key, value in entry.items() if key not in ("paths", "summary")}
                    for entry in entries
                ]
            }
        if method == "campaign.delete":
            # The maintainer's GUI action: all or nothing; rows go in this transaction with a
            # tombstone, files follow in tick (deletion.py).
            if self.exports.job is not None:
                raise ContractError(
                    "campaign_ids",
                    "A study export is reading the data; delete when it ends",
                    "conflict",
                )
            entries = plan(self.store, campaign_ids(params.get("campaign_ids")), self.current)
            blocked = [
                entry
                for entry in entries
                if not entry["deletable"] and entry["reason"] != "already deleted"
            ]
            if blocked:
                first = blocked[0]
                raise ContractError(
                    "campaign_ids",
                    f"{first['name'] or first['campaign_id']} cannot be deleted: {first['reason']}",
                    "conflict",
                )
            deletable = [entry for entry in entries if entry["deletable"]]
            self.run_records.cancel([entry["campaign_id"] for entry in deletable])
            delete_rows(self.store, deletable, operation_id, self.provenance_id)
            return {
                "operation_id": operation_id,
                "deleted": [entry["campaign_id"] for entry in deletable],
                "already_deleted": [
                    entry["campaign_id"] for entry in entries if not entry["deletable"]
                ],
                "files_pending": bool(deletable),
            }
        if method == "campaign.move":
            identity = text(params.get("campaign_id"), "campaign_id")
            direction = params.get("direction")
            if direction not in ("up", "down", "first", "last"):
                raise ContractError("direction", "Use up, down, first or last")
            if self.store.campaign(identity)["state"] != "QUEUED":
                raise ContractError("campaign_id", "Only queued campaigns can move", "conflict")
            self._move_queued(identity, direction)
            self.store.event("campaign.reordered", {"direction": direction}, identity)
            return {"operation_id": operation_id, "completed": True, "campaign_id": identity}
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
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if is_tuning(json.loads(campaign["specification"])):
                raise ContractError("campaign_id", "Start a new tuning run instead", "conflict")
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
            self.backend.preflight()
            self.current, self.queue_enabled = identity, True
            self.store.db.execute(
                "UPDATE experiments SET held=0 WHERE campaign_id=? AND state='NOT_DONE'",
                (identity,),
            )
            self.store.set_state(identity, "RUNNING")
            return {"operation_id": operation_id, "accepted": True, "state": "RUNNING"}
        if method == "tuning.start":
            return self._start_tuning(params, operation_id)
        if method == "campaign.pause":
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if is_tuning(json.loads(campaign["specification"])):
                raise ContractError("campaign_id", "A tuning run cannot be paused", "conflict")
            state = campaign["state"]
            if self.current != identity or state not in ("RUNNING", "RETRYING"):
                raise ContractError("campaign_id", "Pause requires active execution", "conflict")
            self.queue_enabled = False
            self.store.set_state(identity, "PAUSING" if self.active else "PAUSED")
            return {"operation_id": operation_id, "accepted": True}
        if method in ("campaign.stop", "campaign.end"):
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            if campaign["state"] not in {
                "QUEUED",
                "RUNNING",
                "RETRYING",
                "PAUSING",
                "PAUSED",
                "BLOCKED",
                "RECOVERY_REQUIRED",
            }:
                raise ContractError(
                    "campaign_id", "Campaign cannot stop from " + campaign["state"], "conflict"
                )
            state = self._stop_campaign(identity, operation_id)
            return {"operation_id": operation_id, "accepted": True, "state": state}
        if method == "campaign.restart":
            # Retired: it stopped the source. A distinct method keeps an old GUI or engine from silently
            # applying the other semantics.
            raise ContractError(
                "method",
                "campaign.restart was replaced by campaign.restart_copy, which keeps the source campaign",
                "unsupported",
            )
        if method == "campaign.restart_copy":
            # Runs a copy of the accepted specification now, or first in the queue when the engine is
            # busy; the source campaign, its runs and its data are never touched.
            identity = text(params.get("campaign_id"), "campaign_id")
            source = self.store.campaign(identity)
            specification = json.loads(source["specification"])
            if is_tuning(specification):
                raise ContractError("campaign_id", "A tuning run cannot be restarted", "conflict")
            if self.library:
                catalog = self.library.catalog()
                missing = sorted(
                    set(specification["robots"]) - {robot["id"] for robot in catalog["robots"]}
                ) + sorted(
                    set(specification["worlds"]) - {world["id"] for world in catalog["worlds"]}
                )
                if missing:
                    raise ContractError(
                        "campaign_id", "No longer in the catalog: " + ", ".join(missing), "conflict"
                    )
            idle = not (self.current or self.active or self.job)
            if idle:
                self.backend.preflight()
            name = self._copy_name(source["name"])
            specification.update(name=name, source_campaign_id=identity)
            spec = CampaignSpec.accepted(specification)
            copy = self.store.submit(spec)
            self.store.event(
                "campaign.restarted_from",
                {
                    "source_campaign_id": identity,
                    "source_name": source["name"],
                    "operation_id": operation_id,
                },
                copy,
            )
            if idle:
                self.current, self.queue_enabled = copy, True
                self.store.set_state(copy, "RUNNING")
            else:
                self._move_queued(copy, "first")
            return {
                "operation_id": operation_id,
                "accepted": True,
                "campaign_id": copy,
                "name": name,
                "experiments": spec.count,
                "source_campaign_id": identity,
                "state": self.store.campaign(copy)["state"],
            }
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
        if method == "attempt.stop":
            # Ends one run as manual_stop: its data is kept, the experiment is resolved and not run
            # again, and study exports leave it out (run_record.report_class, reported-outcomes-v2).
            known_keys(params, {"attempt_id"}, "params")
            identity = text(params.get("attempt_id"), "attempt_id")
            if identity not in self.active:
                raise ContractError("attempt_id", "Run is no longer active", "conflict")
            attempt = self.store.attempt(identity)
            if attempt["discard_intent"]:
                raise ContractError("attempt_id", "Run is already being discarded", "conflict")
            response = {"operation_id": operation_id, "accepted": True, "attempt_id": identity}
            if self._stop_requested(identity):
                return dict(response, already_requested=True)
            if attempt["phase"] not in ("STARTING", "READY", "RUNNING"):
                raise ContractError("attempt_id", "Run is already ending", "conflict")
            # Persisted with the command; tick signals the worker only after this commits.
            self.store.event(
                "attempt.stop_requested",
                {"operation_id": operation_id},
                attempt["campaign_id"],
                identity,
            )
            return response
        if method == "campaign.status":
            identity = text(params.get("campaign_id"), "campaign_id")
            campaign = self.store.campaign(identity)
            campaign["specification"] = json.loads(campaign["specification"])
            return {"campaign": campaign, "counters": self.store.counters(identity)}
        if method == "results.query" and params.get("table") == "runs":
            if params.get("cursor") is not None or params.get("summary", False) is not False:
                raise ContractError("table", "Run pages take no cursor or summary")
            include = params.get("include", [])
            if include not in ([], ["counts"]):
                raise ContractError("include", "Choose counts for run pages")
            filters = mapping(params.get("filters", {}), "filters")
            page = runs_page(
                self.store.db,
                filters,
                params.get("offset", 0),
                params.get("limit", 100),
                params.get("sort_by"),
                params.get("sort_descending", False),
            )
            if include:
                page["counts"] = run_counts(self.store.db, filters)
            return page
        if method == "results.query":
            include = params.get("include", [])
            if include not in ([], ["study"]) or (include and params.get("table") != "campaigns"):
                raise ContractError("include", "Choose study for campaign pages")
            page = self.store.query(
                params.get("table", "experiments"),
                mapping(params.get("filters", {}), "filters"),
                params.get("offset", 0),
                params.get("limit", 100),
                summary=params.get("summary", False),
                cursor=params.get("cursor"),
                sort_by=params.get("sort_by"),
                sort_descending=params.get("sort_descending", False),
            )
            if include:
                identities = [row["id"] for row in page["rows"]]
                studies = {
                    row["id"]: row
                    for row in self.store.rows(
                        "SELECT id,json_extract(specification,'$.study.id') study_id,"
                        "json_extract(specification,'$.study.evidence_level') evidence_level "
                        f"FROM campaigns WHERE id IN ({','.join('?' * len(identities))})",
                        tuple(identities),
                    )
                }
                for row in page["rows"]:
                    found = studies.get(row["id"]) or {}
                    row["study_id"] = found.get("study_id") or UNASSIGNED
                    row["evidence_level"] = found.get("evidence_level") or "exploratory"
            return page
        if method == "results.get":
            table = params.get("table")
            if table not in ("attempts", "experiments"):
                raise ContractError("table", "Choose attempts or experiments")
            row = self.store.one(
                f"SELECT * FROM {table} WHERE id=?", (text(params.get("id"), "id"),)
            )
            if row is None:
                raise ContractError("id", "No result with this identity", "not_found")
            row.pop("metrics", None)  # results come from the run record (RD-44; R7)
            for field in ("specification", "configuration"):
                if row.get(field):
                    row[field] = json.loads(row[field])
            # Optional parts (R4b): the attempt's run record and the provenance it names; null until built.
            include = params.get("include", [])
            if (
                not isinstance(include, list)
                or not all(type(part) is str for part in include)
                or not set(include) <= {"record", "provenance"}
                or len(set(include)) != len(include)
                or (include and table != "attempts")
            ):
                raise ContractError("include", "Choose record and provenance for an attempt")
            if include:
                record = attempt_record(self.store.db, row["id"])
                if "record" in include:
                    row["record"] = record
                if "provenance" in include:
                    row["provenance"] = provenance_document(
                        self.store.db, record["provenance_id"] if record else None
                    )
            return row
        if method == "results.timeline":
            # The run view's time series of a finished run (timelines.py), built off the engine loop.
            known_keys(params, {"attempt_id"}, "params")
            identity = text(params.get("attempt_id"), "attempt_id")
            attempt = self.store.attempt(identity)
            if attempt["discarded"] or not attempt["directory"]:
                raise ContractError("attempt_id", "Measurements were discarded", "discarded")
            if identity in self.active or attempt["phase"] != "FINISHED":
                raise ContractError(
                    "attempt_id", "Run in progress; its view opens when it has finished", "conflict"
                )
            return self.reads.get(
                ("results.timeline", identity),
                partial(timeline, Path(attempt["directory"])),
                dict(attempt_id=identity),
            )
        if method == "study.traces":
            known_keys(params, {"study_id", "world_id", "spawn_id", "robot_id"}, "params")
            study_id = study_identity(params.get("study_id"), "study_id")
            world = text(params.get("world_id"), "world_id")
            spawn, robot = (
                None if params.get(key) is None else text(params.get(key), key)
                for key in ("spawn_id", "robot_id")
            )
            return self.reads.get(
                ("study.traces", study_id, world, spawn, robot),
                partial(traces, self.workspace / "suite.sqlite3", study_id, world, spawn, robot),
                dict(study_id=study_id, world_id=world, spawn_id=spawn, robot_id=robot),
            )
        if method == "results.map":
            # Coverage and path map of a finished run, built off the engine loop (run_map.py).
            known_keys(params, {"attempt_id", "after_sequence"}, "params")
            identity = text(params.get("attempt_id"), "attempt_id")
            after = integer(params.get("after_sequence", 0), "after_sequence", 0, 2**63 - 1)
            attempt = self.store.attempt(identity)
            if attempt["discarded"] or not attempt["directory"]:
                return dict(
                    attempt_id=identity, state="unavailable", reason="Measurements were discarded"
                )
            if identity in self.active or attempt["phase"] != "FINISHED":
                return dict(
                    attempt_id=identity,
                    state="running",
                    reason="Run in progress; its live map is on the Monitor",
                )
            return self.run_maps.get(identity, after)
        if method == "replay.window":
            # What the Monitor showed during one window of a finished run (replay.py), built off the loop.
            known_keys(params, {"attempt_id", "start_ns", "end_ns"}, "params")
            identity = text(params.get("attempt_id"), "attempt_id")
            start = integer(params.get("start_ns"), "start_ns", 0, 2**63 - 1 - WINDOW_LIMIT_NS)
            end = integer(params.get("end_ns"), "end_ns", start + 1, start + WINDOW_LIMIT_NS)
            attempt = self.store.attempt(identity)
            if attempt["discarded"] or not attempt["directory"]:
                return dict(
                    attempt_id=identity, state="unavailable", reason="Measurements were discarded"
                )
            if identity in self.active or attempt["phase"] != "FINISHED":
                return dict(
                    attempt_id=identity,
                    state="running",
                    reason="Run in progress; replay opens when it has finished",
                )
            return self.replay_windows.get(identity, start, end)
        if method == "measurements.query":
            attempt = self.store.attempt(text(params.get("attempt_id"), "attempt_id"))
            if attempt["discarded"]:
                raise ContractError("attempt_id", "Measurements were discarded", "discarded")
            from .datasets import query

            return query(self.workspace, params)
        if method == "events.read":
            after = integer(params.get("after_sequence", 0), "after_sequence", 0, 2**63 - 1)
            limit = integer(params.get("limit", 200), "limit", 1, 500)
            # Optional scopes: one campaign's or one attempt's events.
            scope = {
                key: text(params[key], key)
                for key in ("campaign_id", "attempt_id")
                if params.get(key) is not None
            }
            current = self._last_sequence()
            if after > current:
                return {"events": [], "last_sequence": current, "snapshot_required": True}
            terms = "".join(f" AND {key}=?" for key in scope)
            rows = self.store.rows(
                f"SELECT * FROM events WHERE sequence>?{terms} ORDER BY sequence LIMIT ?",
                (after, *scope.values(), limit),
            )
            for row in rows:
                row["payload"] = json.loads(row["payload"])
            if scope and len(rows) < limit:
                # Every matching event up to ``current`` was returned: the reader may skip ahead.
                last = current
            else:
                last = rows[-1]["sequence"] if rows else after
            return {"events": rows, "last_sequence": last, "snapshot_required": False}
        if method == "monitor.get":
            identity = text(params.get("attempt_id"), "attempt_id")
            if identity not in self.active:
                raise ContractError("attempt_id", "Worker is no longer active", "not_found")
            layers = params.get("layers", [])
            if (
                not isinstance(layers, list)
                or not all(layer in MONITOR_LAYERS for layer in layers)
                or len(set(layers)) != len(layers)
            ):
                raise ContractError(
                    "layers", "Choose distinct layers from: " + ", ".join(MONITOR_LAYERS)
                )
            revisions = mapping(params.get("revisions", {}), "revisions")
            known_keys(revisions, set(MONITOR_LAYERS), "revisions")
            return dict(
                self.active[identity].monitor(
                    integer(params.get("after_sequence", 0), "after_sequence", 0, 2**63 - 1),
                    list(layers),
                    {
                        name: integer(value, "revisions." + name, -1, 2**63 - 1)
                        for name, value in revisions.items()
                    },
                ),
                attempt_id=identity,
            )
        if method == "logs.read":
            return self._logs(params)
        if method in ("files.list", "files.table", "files.log"):
            return self._files(method, params)
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

    def _files(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """File viewer reads (files.py): a finished run's directory or a study's exports directory."""
        known_keys(params, {"root", "id", "path", "offset", "limit", "search"}, "params")
        if params.get("root") == "attempt":
            identity = text(params.get("id"), "id", maximum=64)
            attempt = self.store.attempt(identity)
            if attempt["discarded"] or not attempt["directory"]:
                raise ContractError("id", "Measurements were discarded", "discarded")
            if identity in self.active or attempt["phase"] != "FINISHED":
                raise ContractError(
                    "id", "Run in progress; its files open when it has finished", "conflict"
                )
            root = Path(attempt["directory"])
        elif params.get("root") == "study":
            identity = study_identity(params.get("id"), "id")
            root = studies_root(self.workspace) / identity
            if not root.is_dir():
                raise ContractError("id", "The study has no exports yet", "not_found")
        else:
            raise ContractError("root", "Choose attempt or study")
        relative = params.get("path", "")
        if not isinstance(relative, str) or len(relative) > 1024 or relative.startswith("/"):
            raise ContractError("path", "Use a path relative to the run's or study's directory")
        if method == "files.table" and not relative.endswith(TABLE_SUFFIXES):
            raise ContractError("path", "Tables are Parquet or CSV files")
        offset = integer(params.get("offset", 0), "offset", 0, 2**63 - 1)
        if method == "files.log":
            limit = integer(params.get("limit", 65536), "limit", 1, 262144)
        else:
            limit = integer(params.get("limit", 100), "limit", 1, 250)
        search = params.get("search")
        if search is not None:
            if method != "files.log":
                raise ContractError("search", "Search applies to files.log")
            if (
                not isinstance(search, str)
                or not 0 < len(search) <= 200
                or any(c in search for c in ("\n", "\r", "\x00"))
            ):
                raise ContractError("search", "Search for one line of 1–200 characters")
        build = (
            partial(file_text, root, relative, offset, limit, search)
            if search is not None
            else partial(BUILDERS[method], root, relative, offset, limit)
        )
        return self.reads.get(
            (method, str(root), relative, offset, limit, search),
            build,
            dict(root=params["root"], id=identity, path=relative, offset=offset),
        )

    def _logs(self, params: dict[str, Any]) -> dict[str, Any]:
        attempt = self.store.attempt(text(params.get("attempt_id"), "attempt_id"))
        if attempt["discarded"]:
            raise ContractError("attempt_id", "Partial measurements were discarded", "discarded")
        component = params.get("component", "worker")
        if component not in (
            "worker",
            "simulation",
            "robot",
            "navigation",
            "slam",
            "explorer",
            "insight",
        ):
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

    def _stop_requested(self, identity: str) -> bool:
        return (
            self.store.one(
                "SELECT 1 FROM events WHERE attempt_id=? AND kind='attempt.stop_requested' LIMIT 1",
                (identity,),
            )
            is not None
        )

    def _move_queued(self, identity: str, direction: str) -> None:
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

    def _copy_name(self, name: str) -> str:
        """Next "<base> (copy N)" name; copies of copies do not stack the suffix."""
        base = COPY_SUFFIX.sub("", name)
        numbers = [0]
        for row in self.store.rows("SELECT name FROM campaigns"):
            match = COPY_SUFFIX.search(row["name"])
            # Long names were cut to fit 160 characters together with their own suffix.
            if match and row["name"][: match.start()] == base[: 160 - len(match.group(0))]:
                numbers.append(int(match.group(1)))
        suffix = f" (copy {max(numbers) + 1})"
        return base[: 160 - len(suffix)] + suffix

    def _stop_campaign(self, identity: str, operation_id: str) -> str:
        self.store.db.execute(
            "UPDATE experiments SET state='NOT_DONE' WHERE campaign_id=? "
            "AND state IN ('PENDING','RETRY')",
            (identity,),
        )
        state = "STOPPING" if self.current == identity else "STOPPED"
        self.store.set_state(identity, state)
        if self.current == identity:
            self.queue_enabled = False
            for attempt_id in self.active:
                self._request_discard(attempt_id, False, operation_id)
        return state

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

    def _in_study(self, spec: CampaignSpec) -> CampaignSpec:
        """Number a new campaign's repetitions after the study's existing ones (RD-30).

        Blocks are (robot, world, spawn, repetition) in the study's namespace, so a later campaign of the
        study only adds blocks. A restarted copy keeps its source's numbers (``campaign.restart_copy``).
        """
        if spec.study is None:
            return spec
        last = last_repetition(self.store.db, spec.study.id)
        first = 0 if last is None else last + 1
        return replace(spec, study=replace(spec.study, first_repetition=first))

    def _note_repetitions(self, spec: CampaignSpec) -> None:
        """Keep a study's highest repetition, so deleting its campaigns never frees a block ID."""
        if spec.study is not None:
            self.store.db.execute(
                "INSERT INTO study_repetitions VALUES(?,?) ON CONFLICT(study_id) DO UPDATE SET "
                "last_repetition=MAX(last_repetition,excluded.last_repetition)",
                (spec.study.id, spec.study.first_repetition + spec.repetitions - 1),
            )

    def _discard(self, attempt: dict[str, Any]) -> None:
        directory = Path(attempt["directory"]).resolve()
        if directory != (self.workspace / "attempts" / attempt["id"]).resolve():
            raise ContractError("directory", "Attempt cleanup path is outside its owned directory")
        shutil.rmtree(directory, ignore_errors=False) if directory.exists() else None
        if directory.parent.exists():
            sync_directory(directory.parent)
        with self.store.transaction():
            # The stub reads the attempt's reservation event, which the next statements delete.
            campaign = self.store.campaign(attempt["campaign_id"])
            self.run_records.stub(
                dict(attempt, outcome="manual_discard", detail="", terminal_ns=time.time_ns()),
                "campaign_stop"
                if campaign["state"] in ("STOPPING", "STOPPED")
                else "experiment_restart"
                if attempt["restart_intent"]
                else "experiment_stop",
            )
            self.store.db.execute("DELETE FROM artifacts WHERE attempt_id=?", (attempt["id"],))
            self.store.db.execute(
                "DELETE FROM events WHERE attempt_id=? "
                "AND kind NOT IN ('attempt.discard_requested','attempt.stop_requested')",
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
            # The engine provenance and run order of the attempt (run record fields, MEASUREMENTS).
            self.store.event(
                "attempt.reserved",
                {
                    "slot": slot,
                    "retry": retry,
                    "provenance_id": self.provenance_id,
                    "runs_since_engine_start": self.reservations,
                },
                experiment["campaign_id"],
                identity,
            )
        self.reservations += 1
        directory.mkdir(parents=True, exist_ok=False)
        return self.store.attempt(identity)

    def _launch(
        self, experiment: dict[str, Any], retry: bool, slot: int, campaign: dict[str, Any]
    ) -> bool:
        try:
            self.backend.preflight()
        except ContractError as exc:
            self.queue_enabled = False
            self.store.set_state(campaign["id"], "BLOCKED", str(exc))
            return False
        attempt = self._reserve(experiment, retry, slot)
        try:
            worker = self.backend.start(attempt, experiment, campaign)
            self.active[attempt["id"]] = worker
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
        return True

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
            if outcome != "manual_stop" and self._stop_requested(attempt["id"]):
                # The run ended on its own before the stop took effect; its own outcome stands.
                self.store.event(
                    "attempt.stop_superseded",
                    {"outcome": outcome},
                    attempt["campaign_id"],
                    attempt["id"],
                )
            if outcome == "engine_interruption":
                self.run_records.stub(self.store.attempt(attempt["id"]), None)
                # The experiment returns to Not done until Resume (docs/MEASUREMENTS.md), as for
                # an interrupted attempt without a finished manifest; an interrupted sequential
                # replacement keeps its retry_queue entry and runs again as a replacement.
                self.store.db.execute(
                    "UPDATE experiments SET state='NOT_DONE',held=0 WHERE id=?",
                    (attempt["experiment_id"],),
                )
            elif outcome == "timing_rejection" and not attempt["retry"] and parallel:
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

    def _advance_deletions(self) -> None:
        """Remove deleted campaigns' files for about 10 ms per tick, oldest deletion first."""
        if self.removal is None:
            row = self.store.one(
                "SELECT campaign_id,paths FROM deletions WHERE state='REMOVING' "
                "ORDER BY requested_ns LIMIT 1"
            )
            if row is None:
                return
            self.removal = row["campaign_id"], Remover(self.workspace, json.loads(row["paths"]))
        identity, remover = self.removal
        try:
            finished = remover.advance(0.01)
        except OSError as error:
            self.removal = None
            self.store.db.execute(
                "UPDATE deletions SET state='FAILED',detail=?,freed_bytes=? WHERE campaign_id=?",
                (str(error), remover.freed_bytes, identity),
            )
            return
        if finished:
            self.removal = None
            with self.store.transaction():
                self.store.db.execute(
                    "UPDATE deletions SET state='DELETED',detail='',finished_ns=?,freed_bytes=? "
                    "WHERE campaign_id=?",
                    (time.time_ns(), remover.freed_bytes, identity),
                )
                self.store.event(
                    "campaign.files_removed", {"freed_bytes": remover.freed_bytes}, identity
                )
            self.collect_pending = True

    def _collect(self) -> None:
        """After deletions: the shared objects nothing names any more, then VACUUM and a WAL checkpoint (RD-44,
        R7). Only while nothing runs, finalizes, exports or removes, in this thread, so no reference appears
        meanwhile. The database never waits for a reader here: a busy or locked one is tried again after
        ``COLLECT_RETRY_S``; any other failure is reported once (``objects.collect_failed``) and dropped."""
        from .deletion import collect_objects

        if (
            not self.collect_pending
            or time.monotonic() < self.collect_retry_at
            or self.active
            or self.current
            or self.job
            or self.removal
            or self.exports.job is not None
            or self.store.one("SELECT 1 FROM deletions WHERE state='REMOVING' LIMIT 1")
        ):
            return
        started = time.monotonic()
        try:
            removed, freed = collect_objects(self.workspace, self.store.db)
            self.collected = (self.collected[0] + removed, self.collected[1] + freed)
            self.store.db.execute("PRAGMA busy_timeout=0")
            try:
                self.store.db.execute("VACUUM")
                # in WAL mode the rewritten pages reach the database file only at a checkpoint
                busy = self.store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
            finally:
                self.store.db.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                self._collect_failed(str(error))
            else:
                self.collect_retry_at = time.monotonic() + COLLECT_RETRY_S
            return
        except OSError as error:
            self._collect_failed(str(error))
            return
        if busy:  # a reader held the log: the file shrinks at a later attempt
            self.collect_retry_at = time.monotonic() + COLLECT_RETRY_S
            return
        self.collect_pending = False
        (removed, freed), self.collected = self.collected, (0, 0)
        self.store.event(
            "objects.collected",
            {
                "objects": removed,
                "freed_bytes": freed,
                "seconds": round(time.monotonic() - started, 3),
            },
        )

    def _collect_failed(self, detail: str) -> None:
        self.collect_pending, self.collected = False, (0, 0)
        self.store.event("objects.collect_failed", {"detail": detail})

    def tick(self) -> None:
        self._advance_deletions()
        self._collect()
        self.run_maps.poll()
        self.run_records.poll(self.current)
        self.exports.poll()
        if self.job is not None:
            try:
                self._poll_job()
            except (OSError, ValueError, KeyError) as error:
                self._finalization_failure(
                    str(error), storage=isinstance(error, OSError) and error.errno in (28, 122)
                )
            return
        for identity, worker in tuple(self.active.items()):
            attempt = self.store.attempt(identity)
            if attempt["discard_intent"]:
                worker.stop("manual_discard", "Manual control")
            elif self._stop_requested(identity):
                worker.stop("manual_stop", "Stopped by the researcher from the Monitor")
            for event in worker.poll():
                kind = event.get("kind")
                if kind == "metrics":
                    # Timing rejection is the worker's (timing-rule-v2, the recorder's windows).
                    self.metrics[identity] = event
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
        if (
            campaign["state"] == "BLOCKED"
            and not self.active
            and is_tuning(json.loads(campaign["specification"]))
        ):
            # A tuning run is never resumed: one that cannot continue releases execution.
            self.store.set_state(self.current, "STOPPED", campaign["detail"])
            self.current = None
            return
        if campaign["state"] in ("PAUSING", "STOPPING"):
            if not self.active:
                self.store.set_state(
                    self.current, "PAUSED" if campaign["state"] == "PAUSING" else "STOPPED"
                )
                if campaign["state"] == "STOPPING":
                    self.current = None
            return
        if campaign["state"] not in ("RUNNING", "RETRYING"):
            return
        if self.exports.job is not None:
            return  # an export reads the store and the attempts' files: no new runs until it ends
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
                if not self._launch(experiment, retry, slot, campaign):
                    return
                if retry:
                    break
        remaining = self.store.one(
            "SELECT COUNT(*) count FROM experiments WHERE campaign_id=? "
            "AND state IN ('PENDING','NOT_DONE','RETRY','ACTIVE')",
            (self.current,),
        )
        if not self.active and remaining and remaining["count"] == 0:
            # Once: the record gate below may keep the campaign here for several ticks.
            self.store.db.execute(
                "UPDATE campaigns SET execution_finished_ns=?,report_state='NOT_REQUESTED' "
                "WHERE id=? AND execution_finished_ns IS NULL",
                (time.time_ns(), self.current),
            )
            if self.backend.synthetic or is_tuning(spec):
                # A tuning run keeps its attempt directory as recorded until the next tuning run.
                self.store.set_state(self.current, "FINISHED")
                self.current = None
            elif not self.run_records.pending(self.current) and self.exports.job is None:
                # Compaction removes the segments the record job reads: records first.
                self._begin_finalization(self.current)
        elif not self.active:
            held = self.store.one(
                "SELECT COUNT(*) count FROM experiments WHERE campaign_id=? AND held=1",
                (self.current,),
            )
            if held and held["count"] and is_tuning(spec):
                self.store.set_state(self.current, "STOPPED", "Tuning run ended without finishing")
                self.current = None
            elif held and held["count"]:
                self.queue_enabled = False
                self.store.set_state(self.current, "PAUSED", "Not done experiments await Resume")

    def _start_tuning(self, params: dict[str, Any], operation_id: str) -> dict[str, Any]:
        """tuning.start: one run that starts at once, is hidden from lists and replaces the previous one
        (plans/SUITE_TUNING_RUN.md). Refused while anything owns execution or the queue runs."""
        if self.current or self.active or self.job or self.queue_enabled:
            raise ContractError(
                "specification",
                "A campaign is running; a tuning run can start when the engine is idle",
                "conflict",
            )
        if self.exports.job is not None:
            raise ContractError(
                "specification",
                "A study export is running; start the tuning run when it ends",
                "conflict",
            )
        raw = mapping(params.get("specification"), "specification")
        # A tuning run picks one predefined spawn of a collection world (plans/worlds/I9 §8.1).
        spec = (
            self.library.resolve_campaign(raw, tuning=True)
            if self.library is not None
            else self.resolve(raw)
        )
        if spec.study is not None:
            raise ContractError("study", "A tuning run belongs to no study")
        single = (
            len(spec.robots) == 1
            and len(spec.robots[0][1]) == 1
            and len(spec.worlds) == 1
            and len(spec.worlds[0][1]) == 1
        )
        if not single or spec.repetitions != 1 or spec.concurrency != 1 or spec.exclusions:
            raise ContractError(
                "specification",
                "A tuning run is one robot, one profile, one world and one start pose",
            )
        spec = replace(spec, name="Tuning run", tuning=True)
        previous = [
            row["id"]
            for row in self.store.rows(
                f"SELECT id FROM campaigns WHERE {TUNING_CAMPAIGN} ORDER BY created_ns"
            )
        ]
        entries = plan(self.store, previous, self.current)
        blocked = [entry for entry in entries if not entry["deletable"]]
        if blocked:
            raise ContractError(
                "specification",
                f"The previous tuning run cannot be removed yet: {blocked[0]['reason']}",
                "conflict",
            )
        self.backend.preflight()
        self.run_records.cancel([entry["campaign_id"] for entry in entries])
        delete_rows(self.store, entries, operation_id, self.provenance_id)
        identity = self.store.submit(spec)
        self.current = identity
        self.store.set_state(identity, "RUNNING")
        return {
            "operation_id": operation_id,
            "accepted": True,
            "campaign_id": identity,
            "replaced": previous,
            "state": "RUNNING",
        }

    def _begin_finalization(self, campaign_id: str) -> None:
        identity, token = str(uuid.uuid4()), uuid.uuid4().hex
        specification = {
            "token": token,
            "kind": "campaign_finalization",
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
                    self.current, "FINISHED", "Execution complete; finalization could not start"
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
            self._finalization_failure(result["detail"], storage=result.get("errno") in (28, 122))
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
            self._finish_finalization(report)

    def _finish_finalization(self, report: dict[str, Any]) -> None:
        """Compacted and committed: the campaign is FINISHED (RESULTS_OVERHAUL A16; no report stage since R7).
        The attempts' checkpoint indexes went with their segments (``commit_compaction``): their artifact rows
        go too."""
        campaign_id = report["campaign_id"]
        with self.store.transaction():
            for row in self.store.rows(
                "SELECT id,path FROM artifacts WHERE campaign_id=? AND kind='checkpoint_index'",
                (campaign_id,),
            ):
                if not Path(row["path"]).exists():
                    self.store.db.execute("DELETE FROM artifacts WHERE id=?", (row["id"],))
            self.store.db.execute(
                "UPDATE reports SET state='FINISHED',finished_ns=? WHERE id=?",
                (time.time_ns(), report["id"]),
            )
            self.store.db.execute(
                "UPDATE campaigns SET report_state='FINISHED' WHERE id=?", (campaign_id,)
            )
            self.store.set_state(campaign_id, "FINISHED")
        self.current, self.job_id = None, None

    def _storage_failure(self, detail: str) -> None:
        self.queue_enabled = False
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

    def _finalization_failure(self, detail: str, *, storage: bool = False) -> None:
        """Compaction or commit failed: the campaign is FINISHED with the reason (its runs and records stand;
        a failed compaction leaves the attempts' own segments, which every reader still uses)."""
        if self.job is not None:
            self.job.close()
            self.job = None
        report = self.store.one("SELECT * FROM reports WHERE id=?", (self.job_id,))
        assert report is not None
        with self.store.transaction():
            self.store.db.execute(
                "UPDATE reports SET state='FAILED',detail=?,finished_ns=? WHERE id=?",
                (detail, time.time_ns(), self.job_id),
            )
            self.store.db.execute(
                "UPDATE campaigns SET report_state='FAILED' WHERE id=?", (report["campaign_id"],)
            )
            self.store.set_state(
                report["campaign_id"],
                "FINISHED",
                "Execution finished; finalization failed: " + detail,
            )
        if storage:
            self.queue_enabled = False
        self.current, self.job_id = None, None

    def run(self, socket_path: Path | None = None) -> None:
        self.server = Server(socket_path or self.workspace / "runtime/control.sock", self.requests)
        try:
            while not self.closed:
                # Requests are served as they arrive; each tick starts 20 ms after the previous
                # one ended, or when the request being handled at that moment finishes.
                deadline = time.monotonic() + 0.02
                while (remaining := deadline - time.monotonic()) > 0:
                    try:
                        pending = self.requests.get(timeout=remaining)
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
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.closed = True
        if self.server:
            self.server.close()
        self.run_maps.close()
        self.run_records.close()
        self.exports.close()
        self.replay_windows.close()
        self.reads.close()
        if self.job is not None:
            self.job.close()
            self.job = None
        for identity, worker in self.active.items():
            if self._stop_requested(identity):
                worker.stop("manual_stop", "Stopped by the researcher from the Monitor")
            else:
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
