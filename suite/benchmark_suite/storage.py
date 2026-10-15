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

import hashlib
import os
from pathlib import Path
from typing import Any
import uuid

from .contracts import ContractError, canonical


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise OSError("Artifact path leaves its dataset directory")
    return path


def require_local_storage(path: Path) -> None:
    mounts = Path("/proc/self/mountinfo")
    if not mounts.is_file():
        raise ContractError(
            "workspace", "Active campaign storage requires Linux local filesystem information"
        )
    remote = {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "9p",
        "ceph",
        "glusterfs",
        "fuse.sshfs",
        "fuse.rclone",
        "fuse.s3fs",
        "fuse.gcsfuse",
    }
    match: tuple[int, str] | None = None
    for line in mounts.read_text().splitlines():
        fields = line.split()
        separator = fields.index("-")
        location = fields[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
        if path.resolve().is_relative_to(Path(location)):
            candidate = len(location), fields[separator + 1]
            if match is None or candidate[0] > match[0]:
                match = candidate
    if match is None or match[1] in remote:
        raise ContractError(
            "workspace",
            "Choose local disk for active SQLite and measurement storage; export completed data to remote storage",
        )


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".staging")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (canonical(value) + "\n").encode())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def put_object(workspace: Path, value: Any) -> str:
    data = canonical(value).encode()
    digest = hashlib.sha256(data).hexdigest()
    path = workspace / "objects" / digest[:2] / (digest + ".json")
    if path.exists():
        if file_hash(path) != digest:
            raise OSError("Shared object failed integrity check: " + str(path))
    else:
        atomic_bytes(path, data)
    return digest


def load_object(workspace: Path, digest: str) -> Any:
    import json

    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid content object identity")
    path = workspace / "objects" / digest[:2] / (digest + ".json")
    if file_hash(path) != digest:
        raise OSError("Shared object failed integrity check: " + str(path))
    return json.loads(path.read_bytes())


def object_store(directory: Path, data: bytes) -> tuple[str, Path]:
    digest = hashlib.sha256(data).hexdigest()
    path = directory / digest[:2] / digest
    if path.exists():
        if file_hash(path) != digest:
            raise OSError("Shared object failed integrity check: " + str(path))
    else:
        atomic_bytes(path, data)
    return digest, path
