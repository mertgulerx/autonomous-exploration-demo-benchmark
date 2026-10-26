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

"""Writes the run record of every finished attempt once (RESULTS_RESEARCH RD-47; ``run_record.py``).

The queue is implicit: finished attempts without a record of the current version, rebuilt from the store
at every engine start. One spawned job at a time measures an attempt's files; the engine loop adds its own
facts and inserts the row. Attempts whose files are gone (a discard) or incomplete (an engine
interruption) get a stub, written in the transaction that ends them. A campaign is finalised (its segments
compacted) only after its records exist, because the job reads the segments.
"""

from collections import OrderedDict
from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
import uuid

from .contracts import UNASSIGNED, canonical
from .run_record import (
    ELIGIBILITY_VERSION,
    INDEX_COLUMNS,
    NAMES,
    RUN_RECORD_VERSION,
    detail_code,
    ordered,
    validate,
)
from .storage import load_object
from .store import Store
from .strata import COMPARISON_KEY_VERSION

if TYPE_CHECKING:
    from .jobs import Job

# The explorer of profiles without ``explorer`` (frontier_exploration_ros2's profiles.yaml ID) and its
# run-record explorer_id (its package, as in run-record-v4).
DEFAULT_EXPLORER = "default"
EXPLORER_ID = "frontier_exploration_ros2"
EXPLORER_LABEL = (
    "org.benchmark.runtime.repository.exploration_packages.frontier_exploration_ros2.commit"
)
# Attempts whose record failed are retried at the next engine start, not in a loop (bounded memory).
FAILED_MEMORY = 1000
# A record job that runs longer is stopped and counted as failed (a hung job never blocks finalisation).
JOB_TIMEOUT_SECONDS = 600.0


class RunRecords:
    def __init__(
        self,
        store: Store,
        workspace: Path,
        provenance_id: str,
        world_families: Mapping[str, str | None] | None = None,
        explorer_ids: Mapping[str, str] | None = None,
    ):
        self.store = store
        self.workspace = workspace
        self.provenance_id = provenance_id
        # world_family per world from the registry (Library.world_families): the family of a generated
        # world, null for every other tier and for worlds the engine's registry does not list.
        self.world_families = dict(world_families or {})
        # explorer_id per profiles.yaml explorer ID (Library.explorer_ids): the explorer's package.
        self.explorer_ids = {DEFAULT_EXPLORER: EXPLORER_ID, **(explorer_ids or {})}
        self.job: Job | None = None
        self.building: str | None = None
        self.building_campaign: str | None = None
        self.started: float = 0.0
        self.failed: OrderedDict[str, str] = OrderedDict()
        columns = {row["name"] for row in store.rows("PRAGMA table_info(run_records)")}
        if not set(INDEX_COLUMNS) <= columns:
            raise RuntimeError(
                "run_records lacks index columns: "
                + ", ".join(sorted(set(INDEX_COLUMNS) - columns))
            )
        self.carry_stubs()

    def carry_stubs(self) -> None:
        """Copy the stubs of an older record version to the current one (engine start).

        A stub's attempt has no files to measure again (a discard) or incomplete ones (an
        engine interruption), and the reservation event its facts came from may be deleted,
        so ``pending`` never rebuilds it: the newest older stub is copied with the current
        ``run_record_version``, fields the older version lacked null. Older rows stay.
        """
        rows = self.store.rows(
            "SELECT r.attempt_id,r.run_record_version,r.record FROM run_records r "
            "WHERE r.record_kind='stub' AND r.run_record_version<? AND NOT EXISTS "
            "(SELECT 1 FROM run_records n WHERE n.attempt_id=r.attempt_id "
            "AND n.run_record_version=?) ORDER BY r.attempt_id,r.run_record_version DESC",
            (RUN_RECORD_VERSION, RUN_RECORD_VERSION),
        )
        carried: set[str] = set()
        with self.store.transaction():
            for row in rows:
                if row["attempt_id"] in carried:
                    continue
                older = json.loads(row["record"])
                record = ordered(
                    {
                        **dict.fromkeys(NAMES),
                        **{name: value for name, value in older.items() if name in NAMES},
                        "run_record_version": RUN_RECORD_VERSION,
                    }
                )
                try:
                    self._store({"id": row["attempt_id"]}, record)
                except ValueError as exc:
                    # one malformed old stub must not stop the engine from starting
                    self.store.event(
                        "run_record.failed",
                        {"detail": f"stub not carried to version {RUN_RECORD_VERSION}: {exc}"},
                        None,
                        row["attempt_id"],
                    )
                    continue
                carried.add(row["attempt_id"])
            if carried:
                self.store.event(
                    "run_record.stubs_carried",
                    dict(version=RUN_RECORD_VERSION, stubs=len(carried)),
                )

    def pending(self, campaign_id: str | None = None) -> list[str]:
        """Finished attempts that still need a record, oldest first (excluding failures until restart)."""
        rows = self.store.rows(
            "SELECT a.id FROM attempts a WHERE a.phase='FINISHED' AND a.discarded=0 "
            "AND COALESCE(a.outcome,'')!='engine_interruption' "
            + ("AND a.campaign_id=? " if campaign_id else "")
            + "AND NOT EXISTS (SELECT 1 FROM run_records r WHERE r.attempt_id=a.id "
            "AND r.run_record_version=?) ORDER BY a.terminal_ns,a.id",
            ((campaign_id,) if campaign_id else ()) + (RUN_RECORD_VERSION,),
        )
        return [row["id"] for row in rows if row["id"] not in self.failed]

    def poll(self, priority: str | None = None) -> None:
        """Collect a finished job, then start the next one (``priority``: the campaign that runs, whose
        finalisation waits for its records). Never reads recorded files itself."""
        if self.job is not None:
            result = self.job.poll()
            if result is None and time.monotonic() - self.started > JOB_TIMEOUT_SECONDS:
                result = {"ok": False, "detail": f"Record job exceeded {JOB_TIMEOUT_SECONDS:g} s"}
            if result is None:
                return
            self.job.close()
            identity, self.job, self.building, self.building_campaign = (
                self.building,
                None,
                None,
                None,
            )
            assert identity is not None
            attempt = self.store.one("SELECT * FROM attempts WHERE id=?", (identity,))
            if attempt is None:
                return  # deleted while it was measured
            if result["ok"]:
                try:
                    with self.store.transaction():
                        self.insert(attempt, result["result"])
                except (ValueError, KeyError, TypeError) as error:
                    self._fail(attempt, f"{type(error).__name__}: {error}")
            else:
                self._fail(attempt, str(result["detail"]))
        if self.job is None:
            waiting = (self.pending(priority) if priority else []) or self.pending()
            if waiting:
                self._start(waiting[0])

    def _start(self, identity: str) -> None:
        from .jobs import Job

        attempt = self.store.one("SELECT * FROM attempts WHERE id=?", (identity,))
        if attempt is None:
            return
        try:
            self.job = Job(
                "run_record",
                {
                    "workspace": str(self.workspace),
                    "attempt_id": identity,
                    "outcome": attempt["outcome"],
                },
                uuid.uuid4().hex,
            )
        except Exception as error:
            self._fail(attempt, f"Record job could not start: {error}")
            return
        self.building, self.building_campaign = identity, attempt["campaign_id"]
        self.started = time.monotonic()

    def cancel(self, campaign_ids: list[str]) -> None:
        """Stop a record job whose attempt belongs to a campaign about to be deleted (its files go)."""
        if self.job is not None and self.building_campaign in campaign_ids:
            self.job.close()
            self.job, self.building, self.building_campaign = None, None, None

    def _fail(self, attempt: dict[str, Any], detail: str) -> None:
        self.failed[attempt["id"]] = detail
        while len(self.failed) > FAILED_MEMORY:
            self.failed.popitem(last=False)
        self.store.event(
            "run_record.failed", {"detail": detail}, attempt["campaign_id"], attempt["id"]
        )

    def engine_facts(self, attempt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The record fields the engine knows, and the campaign specification."""
        experiment = self.store.one(
            "SELECT * FROM experiments WHERE id=?", (attempt["experiment_id"],)
        )
        campaign = self.store.one(
            "SELECT specification FROM campaigns WHERE id=?", (attempt["campaign_id"],)
        )
        assert experiment is not None and campaign is not None
        specification = json.loads(campaign["specification"])
        study = specification.get("study") or {}
        reserved = self.store.one(
            "SELECT payload FROM events WHERE attempt_id=? AND kind='attempt.reserved' "
            "ORDER BY sequence LIMIT 1",
            (attempt["id"],),
        )
        reservation = json.loads(reserved["payload"]) if reserved else {}
        block = self.store.one(
            "SELECT COUNT(*) n FROM experiments WHERE campaign_id=? AND block_id=? AND position<?",
            (experiment["campaign_id"], experiment["block_id"], experiment["position"]),
        )
        # Attempts reserved before suite image 17 name no provenance and no run order: null, not this
        # engine's (never attributed after the fact).
        provenance_id = reservation.get("provenance_id")
        document = (
            self.store.one("SELECT document FROM provenance WHERE id=?", (provenance_id,))
            if provenance_id
            else None
        )
        labels = (
            (json.loads(document["document"]).get("runtime_image") or {}).get("labels") or {}
            if document
            else {}
        )
        profile = json.loads(experiment["specification"]).get("profile") or {}
        explorer = self._explorer(profile)
        outcome = attempt["outcome"] or ""
        return dict(
            run_record_version=RUN_RECORD_VERSION,
            eligibility_version=ELIGIBILITY_VERSION,
            comparison_key_version=COMPARISON_KEY_VERSION,
            study_id=study.get("id") or UNASSIGNED,
            evidence_level=study.get("evidence_level") or "exploratory",
            campaign_id=attempt["campaign_id"],
            experiment_id=attempt["experiment_id"],
            attempt_id=attempt["id"],
            attempt_number=attempt["number"],
            robot_id=experiment["robot_id"],
            world_id=experiment["world_id"],
            spawn_id=experiment["spawn_id"],
            repetition=experiment["repetition"],
            block_id=experiment["block_id"],
            world_family=self.world_families.get(experiment["world_id"]),
            profile_id=experiment["profile_id"],
            profile_revision=profile.get("revision"),
            explorer_id=self.explorer_ids.get(explorer, explorer),
            # the image labels name frontier_exploration_ros2's built commit only; an external explorer's
            # pinned commit is in explorer_source
            explorer_version=labels.get(EXPLORER_LABEL) if explorer == DEFAULT_EXPLORER else None,
            campaign_position=experiment["position"],
            block_position=block["n"] if block else 0,
            wall_start_ns=attempt["started_ns"],
            wall_end_ns=attempt["terminal_ns"],
            worker_slot=attempt["slot"],
            runs_since_engine_start=reservation.get("runs_since_engine_start"),
            outcome=outcome,
            outcome_detail=attempt["detail"] or "",
            outcome_detail_code=detail_code(outcome, attempt["detail"] or ""),
            provenance_id=provenance_id,
            contact_events=None,
            ghost_cells=None,
            moving_object_min_distance_m=None,
            moving_object_time_within_0_5_m_s=None,
            **self._concurrency(attempt),
        ), specification

    def _explorer(self, profile: dict[str, Any]) -> str:
        """The profiles.yaml explorer ID of an experiment's profile (stored inline or as an object)."""
        if "explorer" not in profile and isinstance(profile.get("object"), str):
            try:
                profile = load_object(self.workspace, profile["object"])
            except OSError as exc:
                raise ValueError(f"Profile object unreadable: {exc}") from exc
        explorer = profile.get("explorer", DEFAULT_EXPLORER)
        if not isinstance(explorer, str) or not explorer:
            raise ValueError("The profile names no explorer ID")
        return explorer

    def _concurrency(self, attempt: dict[str, Any]) -> dict[str, Any]:
        """Largest number of attempts running at once with this one, and its overlapped wall share."""
        start, end = attempt["started_ns"], attempt["terminal_ns"]
        if start is None or end is None or end <= start:
            return dict(parallel_max=None, concurrent_wall_fraction=None)
        others = self.store.rows(
            "SELECT started_ns,terminal_ns FROM attempts WHERE id!=? AND started_ns IS NOT NULL "
            "AND started_ns<? AND COALESCE(terminal_ns,?)>?",
            (attempt["id"], end, end, start),
        )
        intervals = [
            (max(start, row["started_ns"]), min(end, row["terminal_ns"] or end)) for row in others
        ]
        points = sorted(
            [(a, 1) for a, b in intervals if b > a] + [(b, -1) for a, b in intervals if b > a]
        )
        running = highest = 0
        covered, opened = 0, None
        for moment, change in points:
            if running == 0 and change > 0:
                opened = moment
            running += change
            highest = max(highest, running)
            if running == 0 and opened is not None:
                covered += moment - opened
        return dict(parallel_max=1 + highest, concurrent_wall_fraction=covered / (end - start))

    def insert(self, attempt: dict[str, Any], measured: dict[str, Any]) -> None:
        """Store a full record from a job's result; the caller holds a transaction."""
        facts, specification = self.engine_facts(attempt)
        fields = dict(measured["fields"])
        defaults = (specification.get("default_parameter_hashes") or {}).get(facts["robot_id"])
        record = ordered(
            dict(
                fields,
                **facts,
                record_kind="full",
                is_baseline=None if defaults is None else fields["parameter_hash"] == defaults,
                discard_trigger=None,
                recorded_ns=time.time_ns(),
            )
        )
        self._store(attempt, record)
        for table in measured.get("derived", []):
            self.store.db.execute(
                "INSERT OR REPLACE INTO artifacts(id,campaign_id,attempt_id,kind,path,hash,bytes,rows,"
                "committed_ns,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    attempt["campaign_id"],
                    attempt["id"],
                    table["kind"],
                    table["path"],
                    table["sha256"],
                    table["bytes"],
                    table["rows"],
                    time.time_ns(),
                    canonical(table),
                ),
            )
        self.store.event(
            "run_record.built",
            dict(
                version=RUN_RECORD_VERSION,
                job_cpu_s=measured.get("cost", {}).get("cpu_s"),
                wall_s=round(time.monotonic() - self.started, 3) if self.started else None,
            ),
            attempt["campaign_id"],
            attempt["id"],
        )

    def stub(self, attempt: dict[str, Any], discard_trigger: str | None) -> None:
        """Store a stub record (identity and outcome only); the caller holds a transaction.

        Runs inside a discard or a recovery: a failure is recorded as an event and never stops them.
        """
        try:
            facts, _ = self.engine_facts(attempt)
            record = ordered(
                {
                    **dict.fromkeys(NAMES),
                    **facts,
                    "record_kind": "stub",
                    "discard_trigger": discard_trigger,
                    "recorded_ns": time.time_ns(),
                }
            )
            self._store(attempt, record)
        except (ValueError, KeyError, TypeError, AssertionError) as error:
            self.store.event(
                "run_record.failed",
                {"detail": f"Stub: {type(error).__name__}: {error}"},
                attempt["campaign_id"],
                attempt["id"],
            )

    def _store(self, attempt: dict[str, Any], record: dict[str, Any]) -> None:
        validate(record)
        columns = ("attempt_id", "run_record_version", *INDEX_COLUMNS, "record")
        values = [
            attempt["id"],
            RUN_RECORD_VERSION,
            *(record[name] for name in INDEX_COLUMNS),
            canonical(record),
        ]
        self.store.db.execute(
            f"INSERT OR IGNORE INTO run_records({','.join(columns)}) "
            f"VALUES({','.join('?' * len(columns))})",
            values,
        )

    def close(self) -> None:
        if self.job is not None:
            self.job.close()
            self.job = None
