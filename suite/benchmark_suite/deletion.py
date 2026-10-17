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

"""Campaign deletion: one plan for preview and delete, rows in the command transaction, files after.

Rows go first, with a tombstone listing the owned paths, so an interruption can only leave orphan files,
never a visible campaign with missing files; the engine removes the listed paths a slice at a time and
resumes at its next start (docs/SUITE_ARCHITECTURE.md, "Campaign deletion").
"""

from collections.abc import Iterator
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from .contracts import CAMPAIGN_DELETION_LIMIT, ContractError, canonical, content_hash
from .storage import sync_directory
from .store import Store

DELETABLE_STATES = frozenset({"FINISHED", "STOPPED"})
# The only workspace subdirectories whose entries deletion may remove; each entry is named by a UUID.
# Legacy dataset-export rows and bundles (the export feature was removed) are left untouched.
ROOTS = ("attempts", "datasets", "reports")


def is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def campaign_ids(value: Any) -> list[str]:
    """1 to CAMPAIGN_DELETION_LIMIT distinct campaign IDs in canonical UUID form."""
    if not isinstance(value, list) or not 1 <= len(value) <= CAMPAIGN_DELETION_LIMIT:
        raise ContractError("campaign_ids", f"Choose 1–{CAMPAIGN_DELETION_LIMIT} campaigns")
    if not all(isinstance(item, str) and is_uuid(item) for item in value):
        raise ContractError("campaign_ids", "Use campaign IDs")
    if len(set(value)) != len(value):
        raise ContractError("campaign_ids", "Each campaign may appear once")
    return list(value)


def plan(store: Store, ids: list[str], current: str | None) -> list[dict[str, Any]]:
    """Per campaign: whether it can be deleted now, why not, what it holds and the paths it owns."""
    plans = []
    for identity in ids:
        campaign = store.one("SELECT * FROM campaigns WHERE id=?", (identity,))
        if campaign is None:
            tombstone = store.one("SELECT name FROM deletions WHERE campaign_id=?", (identity,))
            plans.append(
                dict(
                    campaign_id=identity,
                    name=tombstone["name"] if tombstone else "",
                    deletable=False,
                    reason="already deleted" if tombstone else "not found",
                )
            )
            continue
        reason = None
        if campaign["state"] not in DELETABLE_STATES:
            reason = campaign["state"].lower().replace("_", " ") + ": stop it first"
        elif identity == current:
            reason = "it owns execution"
        elif store.one(
            "SELECT 1 FROM attempts WHERE campaign_id=? AND phase!='FINISHED' LIMIT 1", (identity,)
        ):
            reason = "a run is still finishing"
        paths = [f"datasets/{identity}", f"reports/{identity}"]
        attempts = store.rows(
            "SELECT id,directory,outcome FROM attempts WHERE campaign_id=?", (identity,)
        )
        for attempt in attempts:
            if not attempt["directory"]:
                continue  # discarded attempts own no files
            owned = store.workspace / "attempts" / attempt["id"]
            if not is_uuid(attempt["id"]) or Path(attempt["directory"]).resolve() != owned:
                reason = reason or "a run directory lies outside the workspace"
                continue
            paths.append("attempts/" + attempt["id"])
        experiments = store.rows(
            "SELECT outcome,count(*) n FROM experiments WHERE campaign_id=? GROUP BY outcome",
            (identity,),
        )
        reports = store.one("SELECT count(*) n FROM reports WHERE campaign_id=?", (identity,))
        recorded = store.one(
            "SELECT COALESCE(SUM(bytes),0) n FROM artifacts WHERE campaign_id=?", (identity,)
        )
        specification = json.loads(campaign["specification"])
        summary = dict(
            experiments=sum(row["n"] for row in experiments),
            attempts=len(attempts),
            reports=reports["n"] if reports else 0,
            outcomes={row["outcome"] or "unresolved": row["n"] for row in experiments},
            created_ns=campaign["created_ns"],
            execution_finished_ns=campaign["execution_finished_ns"],
            specification_sha256=content_hash(specification),
            source_campaign_id=specification.get("source_campaign_id"),
        )
        plans.append(
            dict(
                campaign_id=identity,
                name=campaign["name"],
                state=campaign["state"],
                deletable=reason is None,
                reason=reason,
                experiments=summary["experiments"],
                attempts=summary["attempts"],
                reports=summary["reports"],
                recorded_bytes=recorded["n"] if recorded else 0,
                paths=paths,
                summary=summary,
            )
        )
    return plans


def delete_rows(store: Store, plans: list[dict[str, Any]], request_id: str) -> None:
    """Remove every row the campaigns own and leave a tombstone; runs in the command transaction."""
    now = time.time_ns()
    for entry in plans:
        identity = entry["campaign_id"]
        store.db.execute("DELETE FROM retry_queue WHERE campaign_id=?", (identity,))
        store.db.execute("DELETE FROM artifacts WHERE campaign_id=?", (identity,))
        store.db.execute(
            "DELETE FROM events WHERE attempt_id IN (SELECT id FROM attempts WHERE campaign_id=?)",
            (identity,),
        )
        store.db.execute("DELETE FROM events WHERE campaign_id=?", (identity,))
        store.db.execute("DELETE FROM attempts WHERE campaign_id=?", (identity,))
        store.db.execute("DELETE FROM experiments WHERE campaign_id=?", (identity,))
        store.db.execute("DELETE FROM reports WHERE campaign_id=?", (identity,))
        store.db.execute("DELETE FROM campaigns WHERE id=?", (identity,))
        store.db.execute(
            "INSERT INTO deletions(campaign_id,name,request_id,requested_ns,state,paths,summary) "
            "VALUES(?,?,?,?,'REMOVING',?,?)",
            (
                identity,
                entry["name"],
                request_id,
                now,
                canonical(entry["paths"]),
                canonical(entry["summary"]),
            ),
        )
        store.event(
            "campaign.deleted",
            {"name": entry["name"], "attempts": entry["attempts"], "request_id": request_id},
            identity,
        )


class Remover:
    """Removes one deletion's committed paths bottom-up, a time slice at a time; never follows links."""

    def __init__(self, workspace: Path, paths: list[str]):
        self.workspace = workspace
        self.pending = list(paths)
        self.walk: Iterator[tuple[str, list[str], list[str]]] | None = None
        self.freed_bytes = 0

    def root(self, relative: str) -> Path:
        top, _, name = relative.partition("/")
        valid = top in ROOTS and is_uuid(name)
        path = self.workspace / top / name
        if not valid or path.parent.resolve() != (self.workspace / top).resolve():
            raise OSError(f"Refusing to remove {relative}: not an owned workspace entry")
        return path

    def unlink(self, path: Path) -> None:
        try:
            self.freed_bytes += os.lstat(path).st_blocks * 512
            path.unlink()
        except FileNotFoundError:
            pass

    def advance(self, budget_seconds: float) -> bool:
        """Remove for at most budget_seconds; True once every path is gone."""
        deadline = time.monotonic() + budget_seconds
        while self.pending:
            path = self.root(self.pending[0])
            if path.is_symlink() or path.is_file():
                self.unlink(path)
            elif path.is_dir():
                if self.walk is None:
                    self.walk = os.walk(path, topdown=False, followlinks=False)
                for directory, folders, files in self.walk:
                    for name in files:
                        self.unlink(Path(directory) / name)
                    for name in folders:
                        # A link to a directory is listed here and removed as a link, never walked.
                        child = Path(directory) / name
                        if child.is_symlink():
                            self.unlink(child)
                    if Path(directory) != path:
                        self.rmdir(Path(directory))
                    if time.monotonic() > deadline:
                        return False
                self.rmdir(path)
            if path.parent.exists():
                sync_directory(path.parent)
            self.walk = None
            self.pending.pop(0)
            if time.monotonic() > deadline:
                return not self.pending
        return True

    def rmdir(self, path: Path) -> None:
        try:
            self.freed_bytes += os.lstat(path).st_blocks * 512
            path.rmdir()
        except FileNotFoundError:
            pass
