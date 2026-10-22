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

"""Read-only file views for the Results page's file viewer (RESULTS_OVERHAUL §8, A14; RESULTS_RESEARCH RD-41).

The GUI environment has no DuckDB, PyArrow or zstandard, so the engine pages what the GUI cannot read:
``files.list`` lists a directory, ``files.table`` pages a Parquet or CSV file and ``files.log`` pages the text
of a plain, zstd or gzip file. The GUI renders JSON, YAML, text, PDF, PNG, SVG, PGM and ``.npz`` files itself
from the listed paths. Every path stays inside one finished attempt's directory or one study's directory
under ``results/studies``. ``BackgroundReads`` builds these replies, ``study.get``, ``results.timeline`` and
``study.traces`` on one background thread so reads never block the engine loop: ``get`` hands over a ready reply once, or queues the request (the
latest requests win) and answers ``loading``. Suffixes are matched as the suite writes them (lower case).
"""

from collections import OrderedDict, deque
from collections.abc import Callable, Hashable
import datetime
import decimal
import gzip
import math
from pathlib import Path
import threading
import time
from typing import Any, BinaryIO, cast

import duckdb
import zstandard

from .contracts import ContractError, canonical
from .datasets import read_log
from .protocol import MAX_MESSAGE

# Ready replies must fit one protocol message with room for the envelope.
REPLY_LIMIT = MAX_MESSAGE - 64 * 1024
CELL_LIMIT = 1000  # characters of one table cell; longer values are cut and marked
PENDING_READS = 2  # requests waiting for the thread; older ones are dropped (the latest win)
KEPT_REPLIES = 4  # built replies waiting to be collected
# A reply nobody collected within this time is dropped, so a later request reads anew.
REPLY_AGE_S = 5.0
# files.log search reads the text in 1 MiB chunks and scans at most 64 MiB per request; the reply says where
# it stopped. For zstd and gzip text, reaching a later offset decompresses everything before it again.
SCAN_CHUNK = 1024 * 1024
SCAN_BYTES = 64 * 1024 * 1024
KINDS = {
    ".parquet": "table",
    ".csv": "table",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".log": "text",
    ".txt": "text",
    ".md": "text",
    ".zst": "log",
    ".gz": "log",
    ".pdf": "pdf",
    ".png": "image",
    ".svg": "image",
    ".pgm": "image",
    ".npz": "map",
}
TEXT_NAMES = frozenset({"SHA256SUMS", "LICENSE"})
TABLE_SUFFIXES = (".parquet", ".csv")


def kind_of(path: Path) -> str:
    """What the viewer does with a file: table, json, yaml, text, log, pdf, image, map or unknown."""
    if path.name in TEXT_NAMES:
        return "text"
    if path.name.endswith(".index.json"):
        return "json"
    return KINDS.get(path.suffix, "unknown")


def confined(root: Path, relative: str) -> Path:
    """``relative`` inside ``root`` (symbolic links resolved), else a contract error."""
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ContractError("path", "Path leaves the run's or study's directory")
    return path


def listing(root: Path, relative: str, offset: int, limit: int) -> dict[str, Any]:
    """One page of a directory: subdirectories first, then files, by name."""
    directory = confined(root, relative)
    if not directory.is_dir():
        raise ContractError("path", "Not a directory", "not_found")
    base = root.resolve()
    entries = []
    for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        resolved = child.resolve()
        if not resolved.is_relative_to(base) or not (resolved.is_dir() or resolved.is_file()):
            continue  # a link out of the directory, or a socket or device
        status = resolved.stat()
        entries.append(
            dict(
                name=child.name,
                path=child.relative_to(base).as_posix(),
                type="directory" if resolved.is_dir() else "file",
                kind=None if resolved.is_dir() else kind_of(child),
                size=None if resolved.is_dir() else status.st_size,
                modified_ns=status.st_mtime_ns,
            )
        )
    page = entries[offset : offset + limit]
    return dict(
        root=str(base),
        path=directory.relative_to(base).as_posix() if directory != base else "",
        entries=page,
        total=len(entries),
        offset=offset,
        next_offset=offset + len(page) if offset + len(page) < len(entries) else None,
    )


def cell(value: Any) -> Any:
    """A table value as JSON: non-finite numbers and binary values as text, long values cut."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(value)} bytes>"
    if isinstance(value, datetime.date | datetime.time | datetime.timedelta | decimal.Decimal):
        return str(value)
    if isinstance(value, list | tuple):
        converted: Any = [cell(item) for item in value]
    elif isinstance(value, dict):
        converted = {str(key): cell(item) for key, item in value.items()}
    else:
        converted = str(value)
    text = converted if isinstance(converted, str) else canonical(converted)
    if len(text) <= CELL_LIMIT:
        return converted
    return text[:CELL_LIMIT] + f"… ({len(text):,} characters)"


def table(root: Path, relative: str, offset: int, limit: int) -> dict[str, Any]:
    """One page of a Parquet or CSV file in file order, cut to fit one protocol message."""
    path = confined(root, relative)
    if not path.is_file():
        raise ContractError("path", "Not a file", "not_found")
    reader = "read_parquet" if path.suffix == ".parquet" else "read_csv_auto"
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as connection:
        relation = f"{reader}(?)"
        columns = [
            dict(name=row[0], type=row[1])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {relation}", (str(path),)
            ).fetchall()
        ]
        total = connection.execute(f"SELECT COUNT(*) FROM {relation}", (str(path),)).fetchone()
        rows = connection.execute(
            f"SELECT * FROM {relation} LIMIT ? OFFSET ?", (str(path), limit, offset)
        ).fetchall()
    count = int(total[0]) if total else 0
    page: list[list[Any]] = []
    size = len(canonical(columns)) + 1024
    for row in rows:
        values = [cell(value) for value in row]
        size += len(canonical(values)) + 1
        if size > REPLY_LIMIT:
            if not page:
                raise ContractError(
                    "path", "One row of this table is larger than a reply", "too_large"
                )
            break
        page.append(values)
    return dict(
        path=path.relative_to(root.resolve()).as_posix(),
        columns=columns,
        rows=page,
        total=count,
        offset=offset,
        next_offset=offset + len(page) if offset + len(page) < count else None,
    )


def _stream(path: Path, offset: int) -> BinaryIO:
    """The file's text as a binary stream positioned at ``offset`` (zstd and gzip decompressed)."""
    if path.suffix == ".zst":
        stream: BinaryIO = zstandard.ZstdDecompressor().stream_reader(path.open("rb"), closefd=True)
    elif path.suffix == ".gz":
        stream = cast(BinaryIO, gzip.open(path, "rb"))
    else:
        stream = path.open("rb")
    stream.seek(offset)
    return stream


def locate(path: Path, offset: int, needle: str) -> tuple[int | None, int | None, int, bool]:
    """The first occurrence of ``needle`` at or after ``offset``, ASCII letters compared without case: the
    start of its line and its own offset (None, None without a match), then where a further search should
    continue and whether the scan reached the end of the text. At most ``SCAN_BYTES`` are read per call."""
    pattern = needle.encode().lower()
    keep = len(pattern) - 1
    position, line_start, tail = offset, offset, b""
    with _stream(path, offset) as stream:
        while position - offset < SCAN_BYTES:
            chunk = stream.read(SCAN_CHUNK)
            if not chunk:
                return None, None, position, True
            buffer = tail + chunk
            base = position - len(tail)
            found = buffer.lower().find(pattern)
            newline = buffer.rfind(b"\n", 0, found if found >= 0 else len(buffer))
            if newline >= 0:
                line_start = max(line_start, base + newline + 1)
            if found >= 0:
                return line_start, base + found, base + found + 1, False
            position += len(chunk)
            tail = buffer[-keep:] if keep else b""
    # The last bytes were only seen as the start of a match: the next call reads them again.
    return None, None, position - len(tail), False


def text(
    root: Path, relative: str, offset: int, limit: int, search: str | None = None
) -> dict[str, Any]:
    """``limit`` bytes of a file's text from ``offset``, decompressed when it is zstd or gzip. With ``search``,
    the window holds the first match at or after ``offset`` (``match_offset``), starting at its line or half a
    window before it, whichever is later; without a match it is empty, ``searched_to`` says where a further
    search continues and ``end`` whether the scan reached the end of the text."""
    path = confined(root, relative)
    if not path.is_file():
        raise ContractError("path", "Not a file", "not_found")
    found: dict[str, Any] = {}
    if search is not None:
        start, match, reached, exhausted = locate(path, offset, search)
        if start is None or match is None:
            return dict(
                path=path.relative_to(root.resolve()).as_posix(),
                text="",
                offset=offset,
                next_offset=reached,
                end=exhausted,
                match_offset=None,
                searched_to=reached,
                stored_bytes=path.stat().st_size,
            )
        offset, found = max(start, match - limit // 2), dict(match_offset=match)
    if path.suffix == ".zst":
        data = read_log(path.with_suffix(""), offset, limit)
    else:
        with _stream(path, offset) as stream:
            data = stream.read(limit)
    end = len(data) < limit
    if not end:
        data = whole_characters(data)
    return dict(
        path=path.relative_to(root.resolve()).as_posix(),
        text=data.decode("utf-8", errors="replace"),
        offset=offset,
        next_offset=offset + len(data),
        end=end,
        stored_bytes=path.stat().st_size,  # on disk: compressed for .zst and .gz, unlike the offsets
        **found,
    )


def whole_characters(data: bytes) -> bytes:
    """``data`` without a UTF-8 character cut at its end (the next page starts with it); unchanged when
    nothing would remain, so a page always advances."""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 != 0x80:  # ASCII or the lead byte of the last character
            width = (
                2
                if byte >> 5 == 0b110
                else 3
                if byte >> 4 == 0b1110
                else 4
                if byte >> 3 == 0b11110
                else 1
            )
            return data[:-back] if width > back and len(data) > back else data
    return data


BUILDERS: dict[str, Callable[[Path, str, int, int], dict[str, Any]]] = {
    "files.list": listing,
    "files.table": table,
    "files.log": text,
}
Key = tuple[Hashable, ...]
Read = tuple[Key, Callable[[], dict[str, Any]], dict[str, Any]]


class BackgroundReads:
    """Replies built on one background thread (file views, ``study.get``, ``results.timeline`` and
    ``study.traces``); each ready or failed
    reply is handed over once, and one nobody collects within ``REPLY_AGE_S`` is dropped, so a repeated
    request reads the current state again."""

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.done: OrderedDict[Key, tuple[float, dict[str, Any]]] = OrderedDict()
        self.wanted: deque[Read] = deque(maxlen=PENDING_READS)
        self.building: Key | None = None
        self.closing = False
        self.thread: threading.Thread | None = None

    def get(
        self, key: Key, build: Callable[[], dict[str, Any]], echo: dict[str, Any]
    ) -> dict[str, Any]:
        """The reply for ``key`` once ``build`` has made it (``state`` ready or failed); ``echo``
        identifies the request in the ``loading`` and ``failed`` replies."""
        with self.lock:
            now = time.monotonic()
            while self.done and now - next(iter(self.done.values()))[0] > REPLY_AGE_S:
                self.done.popitem(last=False)
            if key in self.done:
                return self.done.pop(key)[1]
            if key != self.building and all(key != wanted[0] for wanted in self.wanted):
                self.wanted.append((key, build, echo))
                if self.thread is None or not self.thread.is_alive():
                    self.thread = threading.Thread(
                        target=self._run, name="background-reads", daemon=True
                    )
                    self.thread.start()
                self.lock.notify()
        return dict(echo, state="loading")

    def _run(self) -> None:
        while True:
            with self.lock:
                while not self.wanted and not self.closing:
                    self.lock.wait()
                if self.closing:
                    return
                key, build, echo = self.wanted.popleft()
                self.building = key
            reply = dict(echo, state="failed")
            try:
                result = build()
                size = len(canonical(result))
                if size > REPLY_LIMIT:
                    reply.update(reason=f"Reply too large ({size:,} bytes)", code="too_large")
                else:
                    reply = dict(result, state="ready")
            except ContractError as exc:
                reply.update(reason=exc.message, code=exc.code)
            except Exception as exc:
                # The thread boundary: report any failure once and keep serving (a file can be
                # unreadable, or removed by compaction or deletion while it is read).
                reply.update(reason=f"Read failed: {exc}", code="unreadable")
            finally:
                with self.lock:
                    self.building = None
                    self.done[key] = (time.monotonic(), reply)
                    while len(self.done) > KEPT_REPLIES:
                        self.done.popitem(last=False)

    def close(self) -> None:
        with self.lock:
            self.closing = True
            self.lock.notify()
        if self.thread is not None:
            self.thread.join(timeout=5)
