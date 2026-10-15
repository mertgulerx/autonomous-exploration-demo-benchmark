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

from collections.abc import Callable, Iterator
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any
import uuid

from .contracts import DATASET_VERSION, ContractError, canonical, text
from .jobs import Job, recover
from .storage import atomic_json, file_hash, inside, sync_directory
from .store import Store


def object_references(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        if set(value) == {"object"} and isinstance(value["object"], str):
            yield value["object"]
        for key, item in value.items():
            if key == "objects" and isinstance(item, dict):
                yield from (v for v in item.values() if isinstance(v, str))
            else:
                yield from object_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from object_references(item)


def export_campaign(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(payload["workspace"]).resolve()
    identity = payload["export_id"]
    campaign_id = payload["campaign_id"]
    parent = workspace / "exports"
    parent.mkdir(exist_ok=True)
    staging, destination = parent / (identity + ".staging"), parent / identity
    staging.mkdir()
    total_bytes, total_files = 0, 0
    index_path = staging / "files.jsonl"
    try:
        with index_path.open("x", encoding="utf-8") as index:

            def include(path: Path, expected: str | None = None) -> None:
                nonlocal total_bytes, total_files
                relative = path.relative_to(workspace)
                source = inside(workspace, str(relative))
                if path.is_symlink() or not source.is_file():
                    raise OSError("Export requires an ordinary retained file: " + str(path))
                target = staging / relative
                if target.exists():
                    if expected and file_hash(target) != expected:
                        raise OSError("Export source hash mismatch: " + str(source))
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with source.open("rb") as incoming, target.open("xb") as outgoing:
                    while block := incoming.read(2 * 1024 * 1024):
                        digest.update(block)
                        outgoing.write(block)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                sha = digest.hexdigest()
                if expected and sha != expected:
                    raise OSError("Export source hash mismatch: " + str(source))
                size = target.stat().st_size
                index.write(canonical(dict(path=str(relative), sha256=sha, bytes=size)) + "\n")
                total_bytes += size
                total_files += 1
                if source.suffix == ".json":
                    include_objects(json.loads(source.read_bytes()))

            def include_objects(value: Any) -> None:
                for digest in object_references(value):
                    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise OSError("Export contains an invalid shared-object identity")
                    path = workspace / "objects" / digest[:2] / (digest + ".json")
                    if not path.is_file():
                        path = path.with_name(digest)
                    include(path, digest)

            with sqlite3.connect(
                f"file:{workspace / 'suite.sqlite3'}?mode=ro", uri=True
            ) as database:
                database.row_factory = sqlite3.Row
                database.execute("BEGIN")
                campaign = database.execute(
                    "SELECT * FROM campaigns WHERE id=?", (campaign_id,)
                ).fetchone()
                if campaign is None or campaign["state"] not in ("FINISHED", "STOPPED"):
                    raise ValueError("Export requires finished or stopped execution")
                atomic_json(staging / "campaign.json", dict(campaign))
                include_objects(json.loads(campaign["specification"]))
                for table in ("experiments", "attempts", "events", "artifacts", "reports"):
                    metadata = staging / (table + ".jsonl")
                    with metadata.open("x", encoding="utf-8") as output:
                        for row in database.execute(
                            f"SELECT * FROM {table} WHERE campaign_id=?", (campaign_id,)
                        ):
                            entry = dict(row)
                            output.write(canonical(entry) + "\n")
                            for field in ("specification", "configuration", "metrics"):
                                if entry.get(field):
                                    include_objects(json.loads(entry[field]))
                            if table == "artifacts":
                                source = Path(entry["path"])
                                include(source, entry["hash"])
                            elif (
                                table == "attempts" and entry["retained"] and not entry["discarded"]
                            ):
                                directory = inside(workspace, "attempts/" + entry["id"])
                                for root, directories, names in os.walk(
                                    directory, followlinks=False
                                ):
                                    directories[:] = [
                                        name for name in directories if name != "ros_logs"
                                    ]
                                    if any(
                                        (Path(root) / name).is_symlink() for name in directories
                                    ):
                                        raise OSError("Export attempt contains a linked directory")
                                    for name in names:
                                        include(Path(root) / name)
                    with metadata.open("rb") as handle:
                        os.fsync(handle.fileno())
                database.rollback()
            index.flush()
            os.fsync(index.fileno())
        metadata = [
            staging / "campaign.json",
            *(
                staging / (name + ".jsonl")
                for name in ("experiments", "attempts", "events", "artifacts", "reports")
            ),
        ]
        manifest = dict(
            export_version=1,
            dataset_version=DATASET_VERSION,
            campaign_id=campaign_id,
            export_id=identity,
            source_workspace=str(workspace),
            created_ns=time.time_ns(),
            files=total_files,
            bytes=total_bytes,
            index="files.jsonl",
            index_sha256=file_hash(index_path),
            metadata=[
                dict(path=p.name, sha256=file_hash(p), bytes=p.stat().st_size) for p in metadata
            ],
            layout="Original workspace-relative paths; scientific source files are unchanged",
        )
        atomic_json(staging / "export.json", manifest)
        for root, _, _ in os.walk(staging, topdown=False):
            sync_directory(Path(root))
        staging.rename(destination)
        sync_directory(parent)
        return dict(
            path=str(destination),
            manifest_sha256=file_hash(destination / "export.json"),
            files=total_files,
            bytes=total_bytes,
        )
    except BaseException:
        shutil.rmtree(staging)
        sync_directory(parent)
        raise


def verify_export(directory: Path, cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    manifest = json.loads((directory / "export.json").read_bytes())
    if manifest.get("export_version") != 1 or manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Unsupported export version")
    index = inside(directory, manifest["index"])
    if file_hash(index) != manifest["index_sha256"]:
        raise OSError("Export file index failed integrity validation")
    count, size = 0, 0
    for entry in manifest["metadata"]:
        path = inside(directory, entry["path"])
        if file_hash(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise OSError("Export metadata failed integrity validation: " + str(path))
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            if cancelled():
                raise InterruptedError("Export verification cancelled")
            entry = json.loads(line)
            path = inside(directory, entry["path"])
            if file_hash(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise OSError("Export file failed integrity validation: " + str(path))
            count += 1
            size += entry["bytes"]
    if count != manifest["files"] or size != manifest["bytes"]:
        raise OSError("Export counts differ from its receipt")
    return dict(manifest)


def copy_export(
    source: Path,
    destination: Path,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> dict[str, Any]:
    source, destination = source.resolve(), destination.expanduser().resolve()
    if (
        destination.exists()
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Choose a new export directory outside the source bundle")
    manifest = json.loads((source / "export.json").read_bytes())
    if manifest.get("export_version") != 1 or manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Unsupported export version")
    index = inside(source, manifest["index"])
    if file_hash(index) != manifest["index_sha256"]:
        raise OSError("Export index failed integrity validation")
    staging = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".staging")
    staging.mkdir(parents=True)
    done, last_progress, count = 0, 0.0, 0
    try:

        def copy(entry: dict[str, Any]) -> None:
            nonlocal done, last_progress
            incoming_path = inside(source, entry["path"])
            target = inside(staging, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with incoming_path.open("rb") as incoming, target.open("xb") as outgoing:
                while block := incoming.read(2 * 1024 * 1024):
                    if cancelled():
                        raise InterruptedError("Export copy cancelled")
                    outgoing.write(block)
                    digest.update(block)
                    size += len(block)
                    done += len(block)
                    if time.monotonic() - last_progress >= 0.25:
                        progress(done, manifest["bytes"])
                        last_progress = time.monotonic()
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if digest.hexdigest() != entry["sha256"] or size != entry["bytes"]:
                raise OSError("Export file failed integrity validation: " + str(incoming_path))

        with index.open(encoding="utf-8") as handle:
            for line in handle:
                copy(json.loads(line))
                count += 1
        if count != manifest["files"] or done != manifest["bytes"]:
            raise OSError("Export counts differ from its receipt")
        for entry in manifest["metadata"]:
            copy(entry)
        copy(
            dict(
                path=manifest["index"], sha256=manifest["index_sha256"], bytes=index.stat().st_size
            )
        )
        copy(
            dict(
                path="export.json",
                sha256=file_hash(source / "export.json"),
                bytes=(source / "export.json").stat().st_size,
            )
        )
        for root, _, _ in os.walk(staging, topdown=False):
            sync_directory(Path(root))
        staging.rename(destination)
        sync_directory(destination.parent)
        progress(manifest["bytes"], manifest["bytes"])
        return dict(path=str(destination), files=count, bytes=manifest["bytes"])
    except BaseException:
        shutil.rmtree(staging)
        raise


class ExportManager:
    def __init__(self, store: Store):
        self.store = store
        self.identity: str | None = None
        self.job: Job | None = None
        for entry in store.rows("SELECT * FROM exports WHERE state IN ('QUEUED','RUNNING')"):
            spec = json.loads(entry["specification"])
            recover(spec["token"], spec.get("pid"), spec.get("pid_created"))
            parent = store.workspace / "exports"
            for suffix in ("", ".staging"):
                path = parent / (entry["id"] + suffix)
                if path.exists():
                    shutil.rmtree(path)
            store.db.execute(
                "UPDATE exports SET state='INTERRUPTED',detail='Engine restart; request a fresh export',finished_ns=? WHERE id=?",
                (time.time_ns(), entry["id"]),
            )

    def submit(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.store.campaign(campaign_id)
        if campaign["state"] not in ("FINISHED", "STOPPED"):
            raise ContractError(
                "campaign_id", "Export requires finished or stopped execution", "conflict"
            )
        identity = str(uuid.uuid4())
        specification = dict(
            workspace=str(self.store.workspace),
            campaign_id=campaign_id,
            export_id=identity,
            token=uuid.uuid4().hex,
        )
        self.store.db.execute(
            "INSERT INTO exports(id,campaign_id,state,specification,created_ns) VALUES(?,?,?,?,?)",
            (identity, campaign_id, "QUEUED", canonical(specification), time.time_ns()),
        )
        self.store.event("export.queued", {"export_id": identity}, campaign_id)
        self.identity = identity
        return dict(export_id=identity, accepted=True)

    def get(self, identity: str) -> dict[str, Any]:
        entry = self.store.one("SELECT * FROM exports WHERE id=?", (text(identity, "export_id"),))
        if entry is None:
            raise ContractError("export_id", "No export with this identity", "not_found")
        entry["specification"] = json.loads(entry["specification"])
        return entry

    def tick(self) -> bool:
        assert self.identity is not None
        entry = self.get(self.identity)
        specification = entry["specification"]
        storage_failure = False
        try:
            if self.job is None:
                self.job = Job("export", specification, specification["token"])
                specification.update(pid=self.job.pid, pid_created=self.job.created)
                self.store.db.execute(
                    "UPDATE exports SET state='RUNNING',specification=? WHERE id=?",
                    (canonical(specification), self.identity),
                )
                return False
            result = self.job.poll()
            if result is None:
                return False
            self.job.close()
            self.job = None
            if not result["ok"]:
                raise OSError(result.get("errno") or 0, result["detail"])
            receipt = result["result"]
            expected = self.store.workspace / "exports" / self.identity
            if (
                Path(receipt["path"]) != expected
                or file_hash(expected / "export.json") != receipt["manifest_sha256"]
            ):
                raise OSError("Export result failed integrity validation")
            self.store.db.execute(
                "UPDATE exports SET state='FINISHED',path=?,receipt=?,finished_ns=? WHERE id=?",
                (str(expected), canonical(receipt), time.time_ns(), self.identity),
            )
            self.store.event(
                "export.finished", dict(receipt, export_id=self.identity), entry["campaign_id"]
            )
        except (OSError, ValueError, RuntimeError) as error:
            storage_failure = isinstance(error, OSError) and error.errno in (28, 122)
            if self.job:
                self.job.close()
                self.job = None
            parent = self.store.workspace / "exports"
            for suffix in ("", ".staging"):
                owned = parent / (self.identity + suffix)
                if owned.exists():
                    shutil.rmtree(owned)
            self.store.db.execute(
                "UPDATE exports SET state='FAILED',detail=?,finished_ns=? WHERE id=?",
                (str(error), time.time_ns(), self.identity),
            )
            self.store.event(
                "export.failed",
                {"export_id": self.identity, "detail": str(error)},
                entry["campaign_id"],
            )
        self.identity = None
        return storage_failure

    def close(self) -> None:
        if self.job:
            self.job.close()
            self.job = None
