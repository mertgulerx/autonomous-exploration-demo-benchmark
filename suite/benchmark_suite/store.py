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

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Iterator
import uuid

from .contracts import CampaignSpec, ContractError, canonical, integer, known_keys, mapping, text
from .design import ORDERING_VERSION, experiments
from .storage import put_object, require_local_storage

# A campaign row started by tuning.start (CampaignSpec.tuning; absent from other specifications).
TUNING_CAMPAIGN = "coalesce(json_extract(specification,'$.tuning'),0)=1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
INSERT INTO schema_version SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM schema_version);
CREATE TABLE IF NOT EXISTS catalogs(hash TEXT PRIMARY KEY, snapshot TEXT NOT NULL, created_ns INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS library(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, owner TEXT NOT NULL, name TEXT NOT NULL,
 family_id TEXT NOT NULL, revision INTEGER NOT NULL, snapshot TEXT NOT NULL,
 hash TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0, created_ns INTEGER NOT NULL,
 UNIQUE(family_id, revision));
CREATE INDEX IF NOT EXISTS library_selection ON library(kind,owner,archived,created_ns);
CREATE TABLE IF NOT EXISTS campaigns(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL, specification TEXT NOT NULL,
 count INTEGER NOT NULL, queue_position INTEGER NOT NULL, created_ns INTEGER NOT NULL,
 execution_finished_ns INTEGER, report_state TEXT NOT NULL DEFAULT 'PENDING',
 detail TEXT NOT NULL DEFAULT '', ordering_version TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS campaign_queue ON campaigns(queue_position,state);
CREATE TABLE IF NOT EXISTS experiments(
 id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL REFERENCES campaigns(id),
 position INTEGER NOT NULL, block_id TEXT NOT NULL, robot_id TEXT NOT NULL, world_id TEXT NOT NULL,
 profile_id TEXT NOT NULL, profile_name TEXT NOT NULL, spawn_id TEXT NOT NULL, spawn_name TEXT NOT NULL,
 repetition INTEGER NOT NULL, specification TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'PENDING',
 outcome TEXT, final_attempt_id TEXT, held INTEGER NOT NULL DEFAULT 0, retry_used INTEGER NOT NULL DEFAULT 0,
 UNIQUE(campaign_id,position));
CREATE INDEX IF NOT EXISTS dispatch_queue ON experiments(campaign_id,state,held,position);
CREATE INDEX IF NOT EXISTS factor_queries ON experiments(campaign_id,robot_id,world_id,profile_id,spawn_id);
CREATE TABLE IF NOT EXISTS attempts(
 id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
 campaign_id TEXT NOT NULL REFERENCES campaigns(id), number INTEGER NOT NULL, slot INTEGER NOT NULL,
 domain INTEGER NOT NULL, token TEXT NOT NULL UNIQUE, partition TEXT NOT NULL, phase TEXT NOT NULL,
 retry INTEGER NOT NULL, reserved_ns INTEGER NOT NULL, started_ns INTEGER, terminal_ns INTEGER,
 cleanup_ns INTEGER, pid INTEGER, pid_created REAL, configuration TEXT NOT NULL,
 directory TEXT NOT NULL, outcome TEXT, detail TEXT NOT NULL DEFAULT '', metrics TEXT,
 discard_intent INTEGER NOT NULL DEFAULT 0, restart_intent INTEGER NOT NULL DEFAULT 0,
 discarded INTEGER NOT NULL DEFAULT 0, retained INTEGER NOT NULL DEFAULT 1,
 UNIQUE(experiment_id,number));
CREATE INDEX IF NOT EXISTS attempts_lookup ON attempts(campaign_id,experiment_id,number);
CREATE INDEX IF NOT EXISTS attempts_campaign_time ON attempts(campaign_id,reserved_ns DESC);
CREATE INDEX IF NOT EXISTS attempts_reserved_time ON attempts(reserved_ns DESC);
CREATE INDEX IF NOT EXISTS attempts_campaign_outcome ON attempts(campaign_id,outcome);
CREATE TABLE IF NOT EXISTS commands(
 request_id TEXT PRIMARY KEY, method TEXT NOT NULL, hash TEXT NOT NULL, response TEXT NOT NULL,
 created_ns INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS events(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, utc_ns INTEGER NOT NULL, kind TEXT NOT NULL,
 campaign_id TEXT, attempt_id TEXT, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS event_campaign ON events(campaign_id,sequence);
CREATE INDEX IF NOT EXISTS event_attempt ON events(attempt_id,kind);
CREATE TABLE IF NOT EXISTS retry_queue(
 experiment_id TEXT PRIMARY KEY REFERENCES experiments(id), campaign_id TEXT NOT NULL,
 queued_ns INTEGER NOT NULL, source_attempt_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts(
 id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, attempt_id TEXT, kind TEXT NOT NULL,
 path TEXT NOT NULL UNIQUE, hash TEXT NOT NULL, bytes INTEGER NOT NULL, rows INTEGER,
 committed_ns INTEGER NOT NULL, metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reports(
 id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, state TEXT NOT NULL, specification TEXT NOT NULL,
 path TEXT, detail TEXT NOT NULL DEFAULT '', created_ns INTEGER NOT NULL, finished_ns INTEGER);
CREATE INDEX IF NOT EXISTS artifacts_campaign ON artifacts(campaign_id,attempt_id);
CREATE TABLE IF NOT EXISTS deletions(
 campaign_id TEXT PRIMARY KEY, name TEXT NOT NULL, request_id TEXT NOT NULL, requested_ns INTEGER NOT NULL,
 finished_ns INTEGER, state TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '', freed_bytes INTEGER,
 paths TEXT NOT NULL, summary TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS deletions_state ON deletions(state,requested_ns);
"""


# Campaign states in the order the Campaigns table lists them with ``sort_by="attention"``.
ATTENTION_ORDER = (
    "RECOVERY_REQUIRED",
    "BLOCKED",
    "RUNNING",
    "RETRYING",
    "PAUSING",
    "STOPPING",
    "PAUSED",
    "QUEUED",
    "STOPPED",
    "REPORTING",
    "FINISHED",
)
ATTENTION_RANK = (
    "CASE state "
    + " ".join(f"WHEN '{state}' THEN {rank}" for rank, state in enumerate(ATTENTION_ORDER))
    + f" ELSE {len(ATTENTION_ORDER)} END"
)
CAMPAIGN_SORT_KEYS = frozenset(
    {"name", "state", "count", "report_state", "created_ns", "attention"}
)
# Metrics copied into attempt summaries (``results.query`` with ``summary``).
SUMMARY_METRICS = ("coverage_percent", "simulation_elapsed", "distance_m")


class Store:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        require_local_storage(self.workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.owner = threading.get_ident()
        self._depth = 0
        self.lock = (self.workspace / "engine.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise ContractError(
                "workspace", "Another engine owns this workspace", "conflict"
            ) from exc
        try:
            self.db = sqlite3.connect(self.workspace / "suite.sqlite3", isolation_level=None)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA busy_timeout=5000")
            self.db.executescript(SCHEMA)
            if self.db.execute("SELECT version FROM schema_version").fetchone()[0] != 1:
                raise ContractError("workspace", "Unsupported operational database version")
            check = self.db.execute("PRAGMA quick_check").fetchone()[0]
            if check != "ok":
                raise ContractError("workspace", "Database integrity check failed: " + check)
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.lock.close()
            raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._assert_owner()
        outer = self._depth == 0
        if outer:
            self.db.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            if outer:
                self.db.execute("ROLLBACK")
            raise
        else:
            if outer:
                self.db.execute("COMMIT")
        finally:
            self._depth -= 1

    def _assert_owner(self) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the scheduler thread may access the operational database")

    def one(self, sql: str, arguments: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        self._assert_owner()
        row = self.db.execute(sql, arguments).fetchone()
        return dict(row) if row else None

    def rows(self, sql: str, arguments: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        self._assert_owner()
        return [dict(row) for row in self.db.execute(sql, arguments)]

    def event(
        self, kind: str, payload: Any, campaign_id: str | None = None, attempt_id: str | None = None
    ) -> int:
        self._assert_owner()
        cursor = self.db.execute(
            "INSERT INTO events(utc_ns,kind,campaign_id,attempt_id,payload) VALUES(?,?,?,?,?)",
            (time.time_ns(), kind, campaign_id, attempt_id, canonical(payload)),
        )
        return int(cursor.lastrowid or 0)

    def submit(self, spec: CampaignSpec, campaign_id: str | None = None) -> str:
        identity = campaign_id or str(uuid.uuid4())
        profiles = {
            (robot, profile.id): put_object(self.workspace, profile.dictionary())
            for robot, choices in spec.robots
            for profile in choices
        }

        def specification(experiment: Any) -> str:
            value = experiment.dictionary()
            value["profile"] = dict(
                id=experiment.profile.id,
                name=experiment.profile.name,
                object=profiles[(experiment.robot_id, experiment.profile.id)],
            )
            return canonical(value)

        with self.transaction():
            position = self.db.execute(
                "SELECT COALESCE(MAX(queue_position),0)+1 FROM campaigns"
            ).fetchone()[0]
            self.db.execute(
                "INSERT INTO campaigns(id,name,state,specification,count,queue_position,"
                "created_ns,ordering_version) VALUES(?,?,?,?,?,?,?,?)",
                (
                    identity,
                    spec.name,
                    "QUEUED",
                    canonical(spec.dictionary()),
                    spec.count,
                    position,
                    time.time_ns(),
                    ORDERING_VERSION,
                ),
            )
            self.db.executemany(
                "INSERT INTO experiments(id,campaign_id,position,block_id,robot_id,world_id,"
                "profile_id,profile_name,spawn_id,spawn_name,repetition,specification) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    (
                        e.id,
                        identity,
                        e.position,
                        e.block_id,
                        e.robot_id,
                        e.world_id,
                        e.profile.id,
                        e.profile.name,
                        e.spawn.id,
                        e.spawn.name,
                        e.repetition,
                        specification(e),
                    )
                    for e in experiments(spec, identity)
                ),
            )
            self.event("campaign.submitted", {"experiments": spec.count}, identity)
        return identity

    def campaign(self, identity: str) -> dict[str, Any]:
        result = self.one("SELECT * FROM campaigns WHERE id=?", (identity,))
        if result is None:
            raise ContractError("campaign_id", "Campaign does not exist", "not_found")
        return result

    def attempt(self, identity: str) -> dict[str, Any]:
        result = self.one("SELECT * FROM attempts WHERE id=?", (identity,))
        if result is None:
            raise ContractError("attempt_id", "Attempt does not exist", "not_found")
        return result

    def set_state(self, identity: str, state: str, detail: str = "") -> None:
        with self.transaction():
            self.db.execute(
                "UPDATE campaigns SET state=?,detail=? WHERE id=?", (state, detail, identity)
            )
            self.event("campaign.state", {"state": state, "detail": detail}, identity)

    def counters(self, campaign_id: str) -> dict[str, Any]:
        states = {
            r["state"]: r["count"]
            for r in self.rows(
                "SELECT state,COUNT(*) count FROM experiments WHERE campaign_id=? GROUP BY state",
                (campaign_id,),
            )
        }
        outcomes = {
            r["outcome"]: r["count"]
            for r in self.rows(
                "SELECT outcome,COUNT(*) count FROM experiments WHERE campaign_id=? AND outcome IS NOT NULL "
                "GROUP BY outcome",
                (campaign_id,),
            )
        }
        attempts = self.one(
            "SELECT COUNT(*) executed, SUM(retry) sequential_replacements, "
            "SUM(outcome='timing_rejection') timing_rejections, SUM(discarded) discards "
            "FROM attempts WHERE campaign_id=?",
            (campaign_id,),
        )
        return {"states": states, "outcomes": outcomes, "attempts": attempts}

    def query(
        self,
        table: str,
        filters: dict[str, Any],
        offset: int = 0,
        limit: int = 100,
        *,
        summary: bool = False,
        cursor: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_descending: bool = False,
    ) -> dict[str, Any]:
        columns = {
            "experiments": {
                "campaign_id",
                "robot_id",
                "world_id",
                "profile_id",
                "spawn_id",
                "state",
                "outcome",
            },
            "attempts": {
                "campaign_id",
                "experiment_id",
                "phase",
                "outcome",
                "retry",
                "discarded",
                "robot_id",
                "world_id",
                "profile_id",
                "spawn_id",
            },
            "campaigns": {"state", "id", "hide_finished"},
            "reports": {"campaign_id", "state"},
            "library": {"kind", "owner", "archived", "family_id"},
            "deletions": {"campaign_id", "state"},
        }
        if not isinstance(table, str) or table not in columns or set(filters) - columns[table]:
            raise ContractError("filters", "Unsupported table or filter")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 250:
            raise ContractError("page", "Use a nonnegative offset and page size 1–250")
        order = {
            "experiments": "position",
            "attempts": "a.reserved_ns DESC,a.id DESC",
            "campaigns": "(state='FINISHED') ASC,queue_position,id",
            "reports": "created_ns DESC",
            "library": "created_ns DESC",
            "deletions": "requested_ns DESC",
        }[table]
        if type(summary) is not bool:
            raise ContractError("summary", "Expected boolean")
        if type(sort_descending) is not bool:
            raise ContractError("sort_descending", "Expected boolean")
        if sort_by is not None:
            if (
                table != "campaigns"
                or type(sort_by) is not str
                or sort_by not in CAMPAIGN_SORT_KEYS
            ):
                raise ContractError("sort_by", "Choose a campaign table column")
            direction = "DESC" if sort_descending else "ASC"
            if sort_by == "attention":
                # Campaigns that need the researcher first, newest first within a state.
                order = f"{ATTENTION_RANK} {direction},created_ns DESC,id"
            else:
                nocase = sort_by in {"name", "state", "report_state"}
                collation = " COLLATE NOCASE" if nocase else ""
                order = f"(state='FINISHED') ASC,{sort_by}{collation} {direction},queue_position,id"
        elif sort_descending:
            raise ContractError("sort_by", "Choose a column before setting the sort direction")
        terms: list[str] = []
        arguments: list[Any] = []
        for key, value in filters.items():
            if key == "hide_finished":
                if type(value) is not bool:
                    raise ContractError("filters.hide_finished", "Expected boolean")
                if value:
                    terms.append("state != ?")
                    arguments.append("FINISHED")
                continue
            if value is not None and type(value) not in (str, int, float, bool):
                raise ContractError("filters." + key, "Use a string, number, boolean or null")
            if type(value) is float and not math.isfinite(value):
                raise ContractError("filters." + key, "Use a finite number")
            if type(value) is int and not -(2**63) <= value < 2**63:
                raise ContractError("filters." + key, "Integer is outside the supported range")
            column = (
                ("e." if key in {"robot_id", "world_id", "profile_id", "spawn_id"} else "a.") + key
                if table == "attempts"
                else key
            )
            if value is None:
                terms.append(column + " IS NULL")
            else:
                terms.append(column + "=?")
                arguments.append(value)
        # Tuning runs (tuning.start) never appear in lists; reads by ID, and lists filtered by the tuning
        # run's own campaign ID (Monitor's events), still reach them.
        if table == "campaigns":
            terms.append(f"NOT {TUNING_CAMPAIGN}")
        elif table in ("experiments", "attempts", "reports") and "campaign_id" not in filters:
            prefix = "a." if table == "attempts" else ""
            terms.append(
                f"{prefix}campaign_id NOT IN (SELECT id FROM campaigns WHERE {TUNING_CAMPAIGN})"
            )
        where = " WHERE " + " AND ".join(terms) if terms else ""
        source = (
            "attempts a JOIN experiments e ON e.id=a.experiment_id"
            if table == "attempts"
            else table
        )
        selection = (
            "a.*,e.robot_id,e.world_id,e.profile_id,e.profile_name,e.spawn_id,e.spawn_name,e.repetition"
            if table == "attempts"
            else "*"
        )
        if summary:
            fields = [
                row[1]
                for row in self.db.execute(f"PRAGMA table_info({table})")
                if row[1] not in {"specification", "configuration", "snapshot", "metrics", "paths"}
            ]
            selection = ",".join(("a." if table == "attempts" else "") + field for field in fields)
            if table == "attempts":
                selection += ",e.robot_id,e.world_id,e.profile_id,e.profile_name,e.spawn_id,e.spawn_name,e.repetition"
                # Values a run table shows without the metrics document; null when not recorded.
                selection += "".join(
                    f",json_extract(a.metrics,'$.{key}') {key}" for key in SUMMARY_METRICS
                )
        page_where, page_arguments, page_offset = where, list(arguments), offset
        backwards = False
        if cursor is not None:
            if table != "attempts":
                raise ContractError("cursor", "Stable cursors apply to attempt pages")
            cursor = mapping(cursor, "cursor")
            known_keys(cursor, {"reserved_ns", "id", "direction"}, "cursor")
            stamp = integer(cursor.get("reserved_ns"), "cursor.reserved_ns", 0, 2**63 - 1)
            identity = text(cursor.get("id"), "cursor.id")
            if cursor.get("direction") not in ("after", "before"):
                raise ContractError("cursor.direction", "Use after or before")
            backwards = cursor["direction"] == "before"
            operator = ">" if backwards else "<"
            page_where += (
                " AND " if page_where else " WHERE "
            ) + f"(a.reserved_ns,a.id){operator}(?,?)"
            page_arguments.extend((stamp, identity))
            page_offset = 0
            if backwards:
                order = "a.reserved_ns ASC,a.id ASC"
        rows = self.rows(
            f"SELECT {selection} FROM {source}{page_where} ORDER BY {order} LIMIT ? OFFSET ?",
            tuple(page_arguments + [limit + 1, page_offset]),
        )
        more = len(rows) > limit
        rows = rows[:limit]
        if backwards:
            rows.reverse()
        count_source = (
            "attempts a"
            if table == "attempts"
            and not set(filters).intersection({"robot_id", "world_id", "profile_id", "spawn_id"})
            else source
        )
        total = self.db.execute(
            f"SELECT COUNT(*) FROM {count_source}{where}", arguments
        ).fetchone()[0]
        for row in rows:
            for key in (
                "specification",
                "configuration",
                "snapshot",
                "metrics",
                "paths",
                "summary",
            ):
                if row.get(key):
                    row[key] = json.loads(row[key])
        return {
            "rows": rows,
            "total": total,
            "offset": offset,
            "next_cursor": dict(
                reserved_ns=rows[-1]["reserved_ns"], id=rows[-1]["id"], direction="after"
            )
            if table == "attempts" and rows and (more or backwards)
            else None,
            "previous_cursor": dict(
                reserved_ns=rows[0]["reserved_ns"], id=rows[0]["id"], direction="before"
            )
            if table == "attempts"
            and rows
            and (more if backwards else offset > 0 or cursor is not None)
            else None,
            "next_offset": offset + len(rows) if offset + len(rows) < total else None,
        }

    def close(self) -> None:
        self._assert_owner()
        self.db.close()
        self.lock.close()
