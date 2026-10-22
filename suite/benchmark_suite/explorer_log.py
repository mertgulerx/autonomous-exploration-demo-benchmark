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

"""The explorer's own log lines of one attempt: dispatches, exit status, and their simulation times.

Contract: docs/MEASUREMENTS.md, "Explorer log formats". Version 1 reads the lines the explorer has logged at
INFO since ``1992202``; version 2 (the logging release, RESULTS_OVERHAUL R8) adds decision, gate and cancel
lines. Explorer stamps are wall clock (UTC); they are mapped to simulation time through the 1 Hz ``quality``
rows, which carry both clocks, and every mapped time is an estimate.
"""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import zstandard

PARSER_VERSION = 1
# A pair is accepted when its Nav2 goal lies at most this long after the mapped dispatch.
MATCH_WINDOW_NS = 1_000_000_000
# ... or at most this long before it: the wall-to-simulation mapping's error (up to 31 ms on the
# recorded attempts of 21 October 2026).
MAPPING_TOLERANCE_NS = 100_000_000

EXPLORER_LINE = re.compile(
    r"^\[frontier_explorer-\d+\] \[(?P<level>[A-Z]+)\] \[(?P<stamp>\d+)\.(?P<fraction>\d+)\] "
    r"\[[^\]]*\]: (?P<text>.*)$"
)
CLEAN_EXIT = re.compile(r"^\[INFO\] \[frontier_explorer-\d+\]: process has finished cleanly")
DIED = re.compile(
    r"^\[ERROR\] \[frontier_explorer-\d+\]: process has died \[pid \d+, exit code (?P<code>-?\d+)"
)
SIGINT_SENT = re.compile(r"^\[INFO\] \[frontier_explorer-\d+\]: sending signal 'SIGINT'")
NUMBER = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
FRONTIER = re.compile(
    rf"frontier reference=\((?P<fx>{NUMBER}), (?P<fy>{NUMBER})\), "
    rf"dispatch=\((?P<dx>{NUMBER}), (?P<dy>{NUMBER})\), size=(?P<size>\d+)"
)
# Every dispatch the explorer logs (frontier_explorer_core_dispatch.cpp and ..._suppression.cpp).
DISPATCHES = (
    ("frontier", re.compile(r"^Sending frontier goal \((?P<mode>[^)]+)\): ")),
    ("updated", re.compile(r"^Sending updated frontier goal \((?P<mode>[^)]+)\): ")),
    ("after_blocked_skip", re.compile(r"^Sending frontier goal after blocked-goal skip: ")),
    ("return_to_start", re.compile(r"^Exploration finished, returning to start pose: ")),
    (
        "suppressed_return_to_start",
        re.compile(r"^All frontiers are temporarily suppressed, returning to start pose while "),
    ),
)

# RO-5 exit statuses.
EXITED_0, EXITED_NONZERO, SIGNALLED, STOPPED_BY_WORKER, NOT_STARTED = (
    "exited_0",
    "exited_nonzero",
    "signalled",
    "stopped_by_worker",
    "not_started",
)


@dataclass(frozen=True, slots=True)
class Dispatch:
    """One dispatch line: its wall stamp, kind and, for frontier goals, the logged frontier."""

    utc_ns: int
    kind: str
    mode: str | None
    frontier_x: float | None
    frontier_y: float | None
    dispatch_x: float | None
    dispatch_y: float | None
    frontier_size: int | None


@dataclass(frozen=True, slots=True)
class ExplorerLog:
    version: int
    lines: int
    dispatches: tuple[Dispatch, ...]
    exit_status: str
    exit_code: int | None
    exited_before_stop: bool | None


def read_text(directory: Path, name: str = "explorer") -> str | None:
    """One of the attempt's logs, plain or compressed after finalisation; None when it has none."""
    plain, packed = directory / f"{name}.log", directory / f"{name}.log.zst"
    if packed.is_file():
        with packed.open("rb") as handle:
            return (
                zstandard.ZstdDecompressor().stream_reader(handle).read().decode(errors="replace")
            )
    if plain.is_file():
        return plain.read_text(errors="replace")
    return None


def parse(text: str | None) -> ExplorerLog:
    """Dispatch lines and the exit of the explorer process from one log."""
    if text is None:
        return ExplorerLog(PARSER_VERSION, 0, (), NOT_STARTED, None, None)
    dispatches: list[Dispatch] = []
    status, code, signal_line, exit_line = STOPPED_BY_WORKER, None, None, None
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if CLEAN_EXIT.match(line):
            status, code, exit_line = EXITED_0, 0, number
            continue
        died = DIED.match(line)
        if died:
            code = int(died["code"])
            status, exit_line = (EXITED_NONZERO if code > 0 else SIGNALLED), number
            continue
        if signal_line is None and SIGINT_SENT.match(line):
            signal_line = number
            continue
        match = EXPLORER_LINE.match(line)
        if match is None:
            continue
        text_part = match["text"]
        for kind, pattern in DISPATCHES:
            head = pattern.match(text_part)
            if head is None:
                continue
            frontier = FRONTIER.search(text_part)
            dispatches.append(
                Dispatch(
                    utc_ns=int(match["stamp"]) * 1_000_000_000
                    + int(match["fraction"].ljust(9, "0")[:9]),
                    kind=kind,
                    mode=head.groupdict().get("mode"),
                    frontier_x=float(frontier["fx"]) if frontier else None,
                    frontier_y=float(frontier["fy"]) if frontier else None,
                    dispatch_x=float(frontier["dx"]) if frontier else None,
                    dispatch_y=float(frontier["dy"]) if frontier else None,
                    frontier_size=int(frontier["size"]) if frontier else None,
                )
            )
            break
    before = None if exit_line is None else signal_line is None or exit_line < signal_line
    return ExplorerLog(PARSER_VERSION, len(lines), tuple(dispatches), status, code, before)


class ClockMap:
    """Wall clock (UTC ns) to simulation clock (ns) by linear interpolation between anchor rows."""

    def __init__(self, anchors: Iterable[tuple[int, int]]):
        pairs = sorted({utc: clock for utc, clock in anchors}.items())
        self.utc = [utc for utc, _ in pairs]
        self.clock = [clock for _, clock in pairs]

    def __call__(self, utc_ns: int) -> int | None:
        """The simulation time of a wall stamp, or None outside the anchors' span."""
        index = bisect_right(self.utc, utc_ns)
        if index == 0 or (index == len(self.utc) and utc_ns != self.utc[-1]):
            return None
        if index == len(self.utc):
            return self.clock[-1]
        left, right = index - 1, index
        span = self.utc[right] - self.utc[left]
        fraction = (utc_ns - self.utc[left]) / span if span else 0.0
        return round(self.clock[left] + fraction * (self.clock[right] - self.clock[left]))


def match(
    dispatches: Sequence[Dispatch], goals: Sequence[tuple[int, int | None]], clock: ClockMap
) -> tuple[dict[int, Dispatch], int, int]:
    """Pair dispatch lines with Nav2 goals in order (``goals``: sequence and dispatch clock).

    Walking both in time order, a dispatch pairs with the next goal whose clock lies at most
    ``MATCH_WINDOW_NS`` after the dispatch's mapped time, or at most ``MAPPING_TOLERANCE_NS`` before it;
    whatever is skipped on either side counts as a mismatch. Goals without a clock never pair. Returns
    the pairs by goal sequence, the mismatches and the dispatch lines outside the clock map's span.
    """
    pairs: dict[int, Dispatch] = {}
    mismatches = unmapped = 0
    timed = sorted((goal for goal in goals if goal[1] is not None), key=lambda goal: (goal[1], goal[0]))
    mismatches += len(goals) - len(timed)
    index = 0
    for dispatch in dispatches:
        mapped = clock(dispatch.utc_ns)
        if mapped is None:
            unmapped += 1
            mismatches += 1
            continue
        while index < len(timed) and (timed[index][1] or 0) < mapped - MAPPING_TOLERANCE_NS:
            index += 1  # a goal with no dispatch line before it
            mismatches += 1
        if index < len(timed) and (timed[index][1] or 0) - mapped <= MATCH_WINDOW_NS:
            pairs[timed[index][0]] = dispatch
            index += 1
        else:
            mismatches += 1  # a dispatch line with no Nav2 goal after it
    mismatches += len(timed) - index
    return pairs, mismatches, unmapped


def decision_rows(
    log: ExplorerLog, clock: ClockMap, origin_ns: int | None, activation_ns: int | None
) -> list[dict[str, Any]]:
    """``explorer_decisions`` rows; version 1 logs only dispatches, so the decision columns stay null."""
    rows = []
    for dispatch in log.dispatches:
        mapped = clock(dispatch.utc_ns)
        rows.append(
            dict(
                time_s=(mapped - origin_ns) / 1e9
                if mapped is not None and origin_ns is not None
                else None,
                t_activation_s=(mapped - activation_ns) / 1e9
                if mapped is not None and activation_ns is not None
                else None,
                time_mapped=mapped is not None,
                utc_ns=dispatch.utc_ns,
                kind=dispatch.kind,
                selection_mode=dispatch.mode,
                candidates=None,
                pool_size=None,
                route_length=None,
                search_ms=None,
                ordering_ms=None,
                decision_map_cache_hits=None,
                decision_map_cache_misses=None,
                frontier_cache_hits=None,
                frontier_cache_misses=None,
                order_cache_hits=None,
                order_cache_misses=None,
                log_format=log.version,
            )
        )
    return rows
