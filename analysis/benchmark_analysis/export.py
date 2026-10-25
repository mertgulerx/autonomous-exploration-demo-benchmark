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

"""A frozen study export, checked before anything is read (RESULTS_OVERHAUL §6, R4).

Every file must match ``SHA256SUMS`` and no other file may be present, so an output always names exactly the
data it came from. Export and run-record versions the module does not know are refused.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .errors import Refused

# 2 adds the worlds table and per-reference descriptors and region rasters (plans/worlds/I2 §3.10); optional
# for readers, so a version-1 export analyses as before.
EXPORT_VERSIONS = (1, 2)
RUN_RECORD_VERSIONS = (1, 2)  # 2 (I10): mapping_condition
CHUNK = 1 << 20


class UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        keys = [key.value for key, _ in node.value if isinstance(key, yaml.ScalarNode)]
        duplicate = next((key for key in keys if keys.count(key) > 1), None)
        if duplicate is not None:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {duplicate!r}", node.start_mark
            )
        return super().construct_mapping(node, deep=deep)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Export:
    directory: Path
    study: dict[str, Any]
    sums_sha256: str  # SHA-256 of SHA256SUMS: the identity of the whole export
    files: dict[str, str]  # relative path -> SHA-256
    provenance: dict[str, Any]  # provenance.json: documents, references, homogeneity

    @property
    def study_id(self) -> str:
        return str(self.study["study_id"])

    def table(self, name: str, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.directory / f"{name}.parquet", columns=columns)

    def input_hashes(self, *names: str) -> dict[str, str]:
        """The listed SHA-256 of the named files, for a sidecar."""
        return {name: self.files[name] for name in sorted(names)}


def open_export(directory: Path) -> Export:
    """Check an export against its ``SHA256SUMS`` and read its ``study.yaml``."""
    directory = directory.resolve()
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise Refused(f"{directory.name}: no SHA256SUMS (not a finished study export)")
    files: dict[str, str] = {}
    for number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name or name in files:
            raise Refused(f"SHA256SUMS line {number} is malformed")
        files[name] = digest
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != sums
    }
    if present != set(files):
        extra, missing = sorted(present - set(files)), sorted(set(files) - present)
        raise Refused(
            f"the export differs from SHA256SUMS: extra {extra[:5]}, missing {missing[:5]}"
        )
    for name, digest in sorted(files.items()):
        if file_sha256(directory / name) != digest:
            raise Refused(f"{name} does not match SHA256SUMS")
    try:
        study = yaml.load((directory / "study.yaml").read_text(encoding="utf-8"), UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise Refused(f"study.yaml: {exc}") from exc
    if not isinstance(study, dict):
        raise Refused("study.yaml is not a mapping")
    for name, known in (
        ("export_version", EXPORT_VERSIONS),
        ("run_record_version", RUN_RECORD_VERSIONS),
    ):
        if type(study.get(name)) is not int or study[name] not in known:
            raise Refused(f"{name} {study.get(name)!r} is not supported")
    provenance_file = directory / "provenance.json"
    provenance = (
        json.loads(provenance_file.read_text(encoding="utf-8")) if provenance_file.is_file() else {}
    )
    return Export(directory, study, file_sha256(sums), files, provenance)
