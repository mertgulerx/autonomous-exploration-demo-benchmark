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

import base64
from collections import defaultdict
from collections.abc import Iterator
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import time
from typing import Any, cast
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from .contracts import DATASET_VERSION, ContractError, integer, text
from .schemas import ADDED_FIELDS, LEGACY_FIELDS, SCHEMAS, SCHEMAS_BY_VERSION
from .storage import atomic_json, file_hash, inside as inside, load_object, sync_directory


def materialize_manifest(directory: Path) -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads((directory / "manifest.json").read_bytes()))
    workspace = directory.parent.parent
    for name in ("configuration", "robot", "requested_parameters", "effective_parameters"):
        value = manifest.get(name)
        if isinstance(value, dict) and set(value) == {"object"}:
            manifest[name] = load_object(workspace, value["object"])
    generated = manifest.get("robot_configuration", {})
    for name in ("robot_description", "visualization_description"):
        value = generated.get(name)
        if isinstance(value, dict) and set(value) == {"object"}:
            generated[name] = load_object(workspace, value["object"])
    return manifest


def validate_file(path: Path, entry: dict[str, Any], kind: str) -> None:
    if file_hash(path) != entry["sha256"]:
        raise OSError("Artifact hash mismatch: " + str(path))
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != entry["rows"] or not supported_schema(
        kind, parquet.schema_arrow
    ):
        raise OSError("Artifact schema or row count mismatch: " + str(path))


def supported_schema(kind: str, schema: pa.Schema) -> bool:
    """Whether a shard has the schema some dataset version wrote for its stream."""
    for schemas in SCHEMAS_BY_VERSION.values():
        expected = schemas.get(kind)
        if expected is None:
            continue
        if schema == expected or (
            kind == "trajectory"
            and schema == expected.remove(expected.get_field_index("command_vy"))
        ):
            return True
    return False


def recorded_version(directory: Path) -> int:
    """The dataset schema an attempt's files were written with, from their Arrow metadata.

    Versions without their own layout (before 9) read as 9, whose layout they share.
    """
    for kind in SCHEMAS:
        for path in files(directory, kind):
            metadata = pq.read_schema(path).metadata or {}
            version = int(metadata.get(b"schema_version", b"0"))
            return version if version in SCHEMAS_BY_VERSION else 9
    return 9


def stream_relation(connection: Any, kind: str, paths: list[str]) -> str:
    """SQL for the stream rows of ``paths`` under the current schema's names (one ``?``: the paths).

    One attempt's files share one schema version; shards written before schema 10 are read through
    ``LEGACY_FIELDS`` and have null ``ADDED_FIELDS``. Recorded files are never rewritten.
    """
    names = {
        column[0]
        for column in connection.execute(
            "SELECT * FROM read_parquet(?,union_by_name=true) LIMIT 0", (paths,)
        ).description
    }
    renames = [
        f'"{old}" AS "{new}"' for old, new in LEGACY_FIELDS.get(kind, {}).items() if old in names
    ]
    added = "".join(
        f', NULL::{sql_type} AS "{name}"' for name, sql_type in ADDED_FIELDS if name not in names
    )
    star = "*" + (" RENAME (" + ", ".join(renames) + ")" if renames else "")
    return f"(SELECT {star}{added} FROM read_parquet(?,union_by_name=true))"


def validate_attempt(directory: Path) -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads((directory / "manifest.json").read_bytes()))
    index = checkpoint_index(directory)
    if (
        index.name == "segments.json"
        and manifest.get("segments_hash")
        and file_hash(index) != manifest["segments_hash"]
    ):
        raise OSError("Attempt checkpoint index hash mismatch")
    if not index.exists():
        if manifest.get("source") == "gazebo" and manifest.get("outcome") in (
            "completed",
            "time_limit",
        ):
            raise OSError("Successful native attempt is missing its measurement checkpoint index")
        return manifest
    segments = json.loads(index.read_bytes())["segments"]
    compacted = directory / "compacted.json"
    if compacted.exists():
        replacement = json.loads(compacted.read_bytes())
        root = compacted_root(directory, replacement)
        dataset_index = inside(root, replacement["index"])
        if file_hash(dataset_index) != replacement["sha256"]:
            raise OSError("Compacted dataset index hash mismatch")
        dataset = json.loads(dataset_index.read_bytes())
        for entry in replacement.get("shards", dataset["shards"]):
            if manifest.get("identity", {}).get("attempt_id") in entry["attempts"]:
                validate_file(inside(root, entry["path"]), entry, entry["kind"])
    else:
        for entry in segments:
            if entry["kind"] not in SCHEMAS:
                raise OSError("Unknown checkpoint dataset kind")
            validate_file(inside(directory, entry["path"]), entry, entry["kind"])
    return manifest


def checkpoint_index(directory: Path) -> Path:
    recovered = directory / "recovery.json"
    return recovered if recovered.exists() else directory / "segments.json"


def compacted_root(directory: Path, reference: dict[str, Any]) -> Path:
    # Bundles written by the removed dataset export are read in place, not through their source workspace.
    bundle = directory.parent.parent
    return bundle if (bundle / "export.json").is_file() else Path(reference["root"])


def files(directory: Path, kind: str) -> list[Path]:
    compacted = directory / "compacted.json"
    if compacted.exists():
        reference = json.loads(compacted.read_bytes())
        root = compacted_root(directory, reference)
        entries = reference.get("shards")
        if entries is None:
            entries = json.loads(inside(root, reference["index"]).read_bytes())["shards"]
        attempt = directory.name
        return [
            inside(root, entry["path"])
            for entry in entries
            if entry["kind"] == kind and attempt in entry["attempts"]
        ]
    index = checkpoint_index(directory)
    if not index.exists():
        return []
    return [
        inside(directory, entry["path"])
        for entry in json.loads(index.read_bytes())["segments"]
        if entry["kind"] == kind
    ]


def observations(directory: Path, kind: str, columns: list[str] | None = None) -> pa.Table:
    if kind not in SCHEMAS:
        raise ContractError("kind", "Unknown measurement dataset")
    paths = files(directory, kind)
    schema = SCHEMAS[kind]
    selected = columns or schema.names
    if any(column not in schema.names for column in selected):
        raise ContractError("columns", "Choose fields present in the dataset schema")
    if not paths:
        return pa.Table.from_pylist([], schema=pa.schema([schema.field(c) for c in selected]))
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as connection:
        columns_sql = ",".join('"' + name + '"' for name in selected)
        sources = [str(path) for path in paths]
        return connection.execute(
            f"SELECT {columns_sql} FROM {stream_relation(connection, kind, sources)} "
            "WHERE attempt_id=? ORDER BY receipt_monotonic_ns",
            (sources, directory.name),
        ).fetch_arrow_table()


def iter_rows(directory: Path, kind: str) -> Iterator[dict[str, Any]]:
    paths = files(directory, kind)
    if not paths:
        return
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as connection:
        sources = [str(path) for path in paths]
        batches = connection.execute(
            f"SELECT * FROM {stream_relation(connection, kind, sources)} "
            "WHERE attempt_id=? ORDER BY clock_ns NULLS FIRST, receipt_monotonic_ns",
            (sources, directory.name),
        ).fetch_record_batch(4096)
        for batch in batches:
            yield from batch.to_pylist()


def preview(directory: Path, kind: str, maximum: int = 20000, scope: str | None = None) -> pa.Table:
    paths = files(directory, kind)
    if not paths:
        return pa.Table.from_pylist([], schema=SCHEMAS[kind])
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as connection:
        where = "attempt_id=?" + (" AND scope=?" if scope is not None else "")
        source = stream_relation(connection, kind, [str(path) for path in paths])
        args = (
            [str(path) for path in paths],
            directory.name,
            *((scope,) if scope is not None else ()),
        )
        count = connection.execute(
            "SELECT count(*) FROM " + source + " WHERE " + where, args
        ).fetchone()
        assert count is not None
        stride = max(1, (count[0] + maximum - 1) // maximum)
        gap = (
            500_000_000
            if kind in ("trajectory", "ground_truth")
            else 1_500_000_000
            if kind == "coverage"
            else 5_000_000_000
        )
        valid = "valid AND status='valid'" if kind == "coverage" else "valid"
        table = connection.execute(
            "WITH raw AS (SELECT *,row_number() OVER ordering AS n,clock_ns-lag(clock_ns) OVER ordering AS gap,"
            + valid
            + " AS present,lag("
            + valid
            + ") OVER ordering AS previous_present FROM "
            + source
            + " WHERE "
            + where
            + " WINDOW ordering AS (ORDER BY clock_ns NULLS FIRST,receipt_monotonic_ns)), segmented AS (SELECT * EXCLUDE(gap,present,previous_present),sum(CASE WHEN NOT present OR NOT previous_present OR gap<=0 OR gap>? THEN 1 ELSE 0 END) OVER(ORDER BY n) AS presentation_segment FROM raw) SELECT * EXCLUDE(n) FROM segmented WHERE (n-1)%?=0 ORDER BY n",
            (*args, gap, stride),
        ).fetch_arrow_table()
        return table.replace_schema_metadata(
            {**(table.schema.metadata or {}), b"presentation_stride": str(stride).encode()}
        )


def recover_checkpoints(directory: Path) -> dict[str, Any]:
    accepted, rejected = [], []
    previous = directory / "segments.json"
    indexed = (
        {entry["path"]: entry for entry in json.loads(previous.read_bytes()).get("segments", [])}
        if previous.exists()
        else {}
    )
    for path in sorted((directory / "segments").glob("*/*.parquet")):
        kind = path.parent.name
        try:
            parquet = pq.ParquetFile(path)
            if kind not in SCHEMAS or not supported_schema(kind, parquet.schema_arrow):
                raise OSError("Schema mismatch")
            if str(path.relative_to(directory)) in indexed:
                validate_file(path, indexed[str(path.relative_to(directory))], kind)
            accepted.append(
                dict(
                    kind=kind,
                    path=str(path.relative_to(directory)),
                    rows=parquet.metadata.num_rows,
                    bytes=path.stat().st_size,
                    sha256=file_hash(path),
                )
            )
        except (OSError, ValueError, pa.ArrowException) as error:
            rejected.append(dict(path=str(path.relative_to(directory)), reason=str(error)))
    result = dict(
        schema_version=DATASET_VERSION,
        segments=accepted,
        rejected=rejected,
        truncated=[
            str(p.relative_to(directory)) for p in (directory / "segments").glob("*/*.staging")
        ],
    )
    atomic_json(directory / "recovery.json", result)
    return result


def compress_logs(directory: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(directory.rglob("*.log")):
        destination = path.with_suffix(path.suffix + ".zst")
        temporary = destination.with_suffix(".staging")
        original_hash = file_hash(path)
        try:
            frames = []
            digest = hashlib.sha256()
            raw_offset = 0
            with path.open("rb") as source, temporary.open("wb") as output:
                compressor = zstd.ZstdCompressor(level=3)
                while chunk := source.read(1024 * 1024):
                    frame = compressor.compress(chunk)
                    if zstd.ZstdDecompressor().decompress(frame) != chunk:
                        raise OSError("Compressed log integrity check failed")
                    frames.append(
                        dict(
                            raw_offset=raw_offset,
                            raw_bytes=len(chunk),
                            offset=output.tell(),
                            bytes=len(frame),
                        )
                    )
                    output.write(frame)
                    raw_offset += len(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != original_hash:
                raise OSError("Compressed log integrity check failed")
            temporary.replace(destination)
            sync_directory(destination.parent)
            atomic_json(
                destination.with_suffix(destination.suffix + ".index.json"),
                dict(frames=frames, original_sha256=original_hash, sha256=file_hash(destination)),
            )
            artifacts.append(
                dict(
                    path=str(destination.relative_to(directory)),
                    sha256=file_hash(destination),
                    original_sha256=original_hash,
                    bytes=destination.stat().st_size,
                    uncompressed_bytes=path.stat().st_size,
                )
            )
            path.unlink()
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return artifacts


def finalize_attempt(directory: Path) -> dict[str, Any]:
    manifest = validate_attempt(directory)
    materialize_manifest(directory)
    for digest in manifest.get("objects", {}).values():
        if file_hash(directory.parent.parent / "objects" / digest[:2] / digest) != digest:
            raise OSError("Shared reference object failed integrity validation")
    logs = compress_logs(directory)
    resources = [directory / "manifest.json"]
    resources.extend(
        path
        for path in (
            directory / "segments.json",
            directory / "recovery.json",
            directory / "final_map.npz",
        )
        if path.exists()
    )
    artifacts = [
        dict(
            path=str(path.relative_to(directory)),
            sha256=file_hash(path),
            bytes=path.stat().st_size,
            kind="map"
            if path.suffix == ".npz"
            else "manifest"
            if path.name == "manifest.json"
            else "checkpoint_index",
        )
        for path in resources
    ]
    artifacts.extend(dict(entry, kind="log") for entry in logs)
    for path in directory.rglob("*.log.zst"):
        if not any(entry["path"] == str(path.relative_to(directory)) for entry in artifacts):
            artifacts.append(
                dict(
                    path=str(path.relative_to(directory)),
                    sha256=file_hash(path),
                    bytes=path.stat().st_size,
                    kind="log",
                )
            )
    for path in directory.rglob("*.log.zst.index.json"):
        artifacts.append(
            dict(
                path=str(path.relative_to(directory)),
                sha256=file_hash(path),
                bytes=path.stat().st_size,
                kind="log_index",
            )
        )
    result = dict(
        schema_version=DATASET_VERSION,
        phase="FINISHED",
        outcome=manifest.get("outcome"),
        artifacts=artifacts,
    )
    atomic_json(directory / "finalization.json", result)
    return result


def finalize_partial(
    directory: Path, outcome: str, detail: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    path = directory / "manifest.json"
    try:
        decoded = json.loads(path.read_bytes())
        manifest = decoded if isinstance(decoded, dict) else {}
    except (OSError, ValueError):
        manifest = {}
    manifest.update(
        # A recovered attempt keeps the schema it was recorded with (another image may recover it).
        schema_version=manifest.get("schema_version", DATASET_VERSION),
        phase="FINISHED",
        partial=True,
        outcome=outcome,
        detail=detail,
        recovered_utc_ns=time.time_ns(),
        final_metrics=dict(metrics, frozen=True),
        eligibility={"timing": False, "coverage": False, "motion": False, "navigation": False},
    )
    atomic_json(path, manifest)
    recover_checkpoints(directory)
    finalize_attempt(directory)
    return manifest


def read_log(path: Path, offset: int, limit: int) -> bytes:
    if path.exists():
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(limit)
    compressed = path.with_suffix(path.suffix + ".zst")
    if not compressed.exists():
        return b""
    index = compressed.with_suffix(compressed.suffix + ".index.json")
    if index.exists():
        frames = json.loads(index.read_bytes())["frames"]
        position = max(0, bisect_right([frame["raw_offset"] for frame in frames], offset) - 1)
        result = bytearray()
        with compressed.open("rb") as source:
            for frame in frames[position:]:
                source.seek(frame["offset"])
                data = zstd.ZstdDecompressor().decompress(
                    source.read(frame["bytes"]), max_output_size=frame["raw_bytes"]
                )
                result.extend(data[max(0, offset - frame["raw_offset"]) :][: limit - len(result)])
                if len(result) >= limit:
                    break
        return bytes(result)
    with compressed.open("rb") as source, zstd.ZstdDecompressor().stream_reader(source) as reader:
        reader.seek(offset)
        return reader.read(limit)


def compact(
    workspace: Path, campaign_id: str, directories: list[str], target_bytes: int = 128 * 1024**2
) -> dict[str, Any]:
    destination = workspace / "datasets" / campaign_id / uuid.uuid4().hex
    destination.mkdir(parents=True)
    try:
        return _compact(workspace, campaign_id, directories, target_bytes, destination)
    except BaseException:
        shutil.rmtree(destination)
        sync_directory(destination.parent)
        raise


def recover_compactions(workspace: Path) -> None:
    with sqlite3.connect(f"file:{workspace / 'suite.sqlite3'}?mode=ro", uri=True) as database:
        committed = {
            Path(row[0]).parent.resolve()
            for row in database.execute("SELECT path FROM artifacts WHERE kind='dataset_index'")
        }
    for path in (workspace / "datasets").glob("*/*"):
        if (
            path.is_dir()
            and len(path.name) == 32
            and all(c in "0123456789abcdef" for c in path.name)
            and path.resolve() not in committed
        ):
            shutil.rmtree(path)
            sync_directory(path.parent)


def _compact(
    workspace: Path, campaign_id: str, directories: list[str], target_bytes: int, destination: Path
) -> dict[str, Any]:
    # Shards hold one dataset schema each: attempts recorded before schema 10 keep the schema-9 layout
    # (older attempts were already written as 9), so a campaign resumed on a newer image is not relabelled.
    groups: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    for directory in sorted(map(Path, directories)):
        manifest = validate_attempt(directory)
        if (directory / "compacted.json").exists() or not checkpoint_index(directory).exists():
            continue
        identity = manifest.get("identity", {})
        version = recorded_version(directory)
        groups[
            (identity.get("robot_id", "unknown"), identity.get("world_id", "unknown"), version)
        ].append(directory)
        sources.append(
            dict(
                directory=str(directory),
                index=checkpoint_index(directory).name,
                sha256=file_hash(checkpoint_index(directory)),
            )
        )
    shards: list[dict[str, Any]] = []
    for (robot, world, version), attempts in groups.items():
        schemas = SCHEMAS_BY_VERSION[version]
        for kind in schemas:
            partition = destination / robot / world / f"schema-{version}" / kind
            partition.mkdir(parents=True)
            writer = None
            path = None
            rows = size = 0
            counts: dict[str, int] = {}

            def close_shard() -> None:
                nonlocal writer, path, rows, size, counts
                if writer is None or path is None:
                    return
                writer.close()
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                final = path.with_suffix(".parquet")
                path.replace(final)
                sync_directory(partition)
                entry = dict(
                    path=str(final.relative_to(workspace)),
                    sha256=file_hash(final),
                    bytes=final.stat().st_size,
                    rows=rows,
                    kind=kind,
                    schema_version=version,
                    attempts=counts,
                )
                validate_file(final, entry, kind)
                shards.append(entry)
                writer, path, rows, size, counts = None, None, 0, 0, {}

            try:
                for directory in attempts:
                    paths = files(directory, kind)
                    if not paths:
                        continue
                    with duckdb.connect(
                        config={
                            "threads": 1,
                            "memory_limit": "256MB",
                            "temp_directory": str(destination / "sort-staging"),
                        }
                    ) as connection:
                        batches = connection.execute(
                            "SELECT * FROM read_parquet(?,union_by_name=true) WHERE attempt_id=? ORDER BY clock_ns NULLS FIRST, receipt_monotonic_ns",
                            ([str(path) for path in paths], directory.name),
                        ).fetch_record_batch(32768)
                        for batch in batches:
                            table = pa.Table.from_batches([batch])
                            if kind == "trajectory" and "command_vy" not in table.column_names:
                                table = table.add_column(
                                    schemas[kind].get_field_index("command_vy"),
                                    "command_vy",
                                    pa.nulls(len(table), pa.float64()),
                                )
                            table = table.cast(schemas[kind])
                            if writer is None:
                                path = partition / (uuid.uuid4().hex + ".staging")
                                writer = pq.ParquetWriter(
                                    path, schemas[kind], compression="zstd", use_dictionary=True
                                )
                            writer.write_table(table, row_group_size=32768)
                            rows += len(table)
                            size += table.nbytes
                            counts[directory.name] = counts.get(directory.name, 0) + len(table)
                            if size >= target_bytes:
                                close_shard()
                close_shard()
            finally:
                if writer:
                    writer.close()
    result = dict(
        schema_version=DATASET_VERSION,
        campaign_id=campaign_id,
        shards=shards,
        sources=sources,
        root=str(workspace),
        phase="PREPARED",
        target_uncompressed_bytes=target_bytes,
    )
    atomic_json(destination / "index.json", result)
    return dict(
        index=str((destination / "index.json").relative_to(workspace)),
        sha256=file_hash(destination / "index.json"),
        root=str(workspace),
    )


def commit_compaction(reference: dict[str, Any]) -> None:
    root = Path(reference["root"])
    index_path = inside(root, reference["index"])
    if file_hash(index_path) != reference["sha256"]:
        raise OSError("Compaction index hash mismatch")
    index = json.loads(index_path.read_bytes())
    with sqlite3.connect(f"file:{root / 'suite.sqlite3'}?mode=ro", uri=True) as database:
        receipt = database.execute(
            "SELECT hash FROM artifacts WHERE kind='dataset_index' AND path=?",
            (str(index_path),),
        ).fetchone()
    if receipt is None or receipt[0] != reference["sha256"]:
        raise OSError("Compaction replacement has not been durably committed by the engine")
    for entry in index["shards"]:
        validate_file(inside(root, entry["path"]), entry, entry["kind"])
    for source in index["sources"]:
        directory = inside(root / "attempts", Path(source["directory"]).name)
        source_index = inside(directory, source.get("index", "segments.json"))
        if file_hash(source_index) != source["sha256"]:
            raise OSError("Source checkpoint index changed after compaction")
        segments = json.loads(source_index.read_bytes())["segments"]
        for entry in segments:
            path = inside(directory, entry["path"])
            if path.exists():
                validate_file(path, entry, entry["kind"])
            elif not (directory / "compacted.json").exists():
                raise OSError("Source segment disappeared before its replacement was committed")
    for source in index["sources"]:
        directory = inside(root / "attempts", Path(source["directory"]).name)
        alias = dict(
            reference,
            shards=[entry for entry in index["shards"] if directory.name in entry["attempts"]],
        )
        atomic_json(directory / "compacted.json", alias)
    for source in index["sources"]:
        directory = inside(root / "attempts", Path(source["directory"]).name)
        for entry in json.loads(
            inside(directory, source.get("index", "segments.json")).read_bytes()
        )["segments"]:
            path = inside(directory, entry["path"])
            path.unlink(missing_ok=True)
            sync_directory(path.parent)
    atomic_json(
        index_path.parent / "commit.json", dict(phase="COMMITTED", index_sha256=reference["sha256"])
    )


def query(workspace: Path, params: dict[str, Any]) -> dict[str, Any]:
    attempt = text(params.get("attempt_id"), "attempt_id")
    directory = inside(workspace / "attempts", attempt)
    kind = text(params.get("kind"), "kind")
    if kind not in SCHEMAS:
        raise ContractError("kind", "Unknown measurement dataset")
    offset = integer(params.get("offset", 0), "offset", 0, 2**63 - 1)
    limit = integer(params.get("limit", 100), "limit", 1, 250)
    paths = files(directory, kind)
    if not paths:
        return dict(rows=[], total=0, next_offset=None)
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as connection:
        arguments = ([str(path) for path in paths], attempt)
        source = stream_relation(connection, kind, arguments[0])
        count = connection.execute(
            "SELECT count(*) FROM " + source + " WHERE attempt_id=?", arguments
        ).fetchone()
        assert count is not None
        total = count[0]
        table = connection.execute(
            "SELECT * FROM "
            + source
            + " WHERE attempt_id=? ORDER BY receipt_monotonic_ns LIMIT ? OFFSET ?",
            (*arguments, limit, offset),
        ).fetch_arrow_table()
    # JSON replies carry binary columns (chunk states, explored masks) as base64 text.
    binary = [field.name for field in SCHEMAS[kind] if pa.types.is_binary(field.type)]
    rows = table.to_pylist()
    for row in rows:
        for name in binary:
            if row[name] is not None:
                row[name] = base64.b64encode(row[name]).decode()
    return dict(
        rows=rows,
        total=total,
        next_offset=offset + limit if offset + limit < total else None,
        binary_columns=dict.fromkeys(binary, "base64"),
    )
