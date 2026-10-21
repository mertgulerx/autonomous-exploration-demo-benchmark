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

"""Replay of a finished run on the Monitor (docs/suite-gui-design/MONITOR_INSIGHT.md §2 Q4, screen 20).

``ReplayData`` holds what the engine sent for one attempt: the Results run map's ground-truth vertices with
their simulation stamps (``results.map``) and the replay windows (``replay.window``, 60 simulation s each:
analysis ticks in the live layers' shapes, the goal history, explored-cell snapshots and coverage samples).
``frame`` gives the state at one simulation time, as the live Monitor showed it then. ``ReplayBar`` is the 48 px
bar: play or pause, step back and forward, the time, the timeline (played part, goal starts, coverage
milestones, thumb) and the speed (presets and a box for any speed).
"""

import base64
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any
import zlib

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .gui_controls import DoubleSpinBox
from .gui_insight import ChunkGrid, Decisions, Goal
from .gui_maps import FREE, LivePath, WorldMap
from .gui_support import Transport
from .gui_theme import ThemeManager, snap
from .gui_widgets import (
    Button,
    SegmentedControl,
    TextStyle,
    Themed,
    draw_text,
    shadow_image,
    shadow_margin,
)

# The engine's replay window (``replay.WINDOW_LIMIT_NS``; the GUI does not import the engine's modules).
WINDOW_NS = 60_000_000_000
SPEEDS = (1.0, 4.0, 16.0)
SPEED_RANGE = (0.25, 64.0)  # the speed box
SPEED_FINE = 0.25  # its arrows step by 1× from 1×, by 0.25× below it
KEPT_WINDOWS = 6  # windows held in memory: the ones around the replay time and the last one


def clock(seconds: float) -> str:
    """Simulation time as m:ss (h:mm:ss past an hour)."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass(slots=True)
class Tick:
    simulation_ns: int
    decisions: Decisions | None
    chunks: ChunkGrid | None
    figures: Mapping[str, Any]


@dataclass(slots=True)
class Window:
    """One ``replay.window`` reply, parsed."""

    index: int
    ticks: list[Tick]
    goals: list[Goal]
    snapshots: list[Mapping[str, Any]]
    coverage: list[Mapping[str, Any]]
    # One sample per simulation second: raw-odometry motion and the explorer's process accounting.
    motion: list[Mapping[str, Any]] = field(default_factory=list)
    resources: list[Mapping[str, Any]] = field(default_factory=list)
    explored: dict[int, np.ndarray[Any, np.dtype[np.bool_]]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Frame:
    """What the Monitor shows at one replay time."""

    time_ns: int
    vertices: int  # path vertices up to this time
    pose: tuple[float, float, float] | None
    explored: np.ndarray[Any, np.dtype[np.uint8]] | None
    tick: Tick | None
    goals: dict[int, Goal]
    goal: tuple[float, float] | None  # the goal Nav2 received last and has not ended
    coverage: Mapping[str, Any] | None  # the latest coverage sample
    motion: Mapping[str, Any] | None  # the latest motion sample (distance, speed, turning)
    resources: Mapping[str, Any] | None  # the latest explorer process sample (CPU, RAM, RTF, flags)


class ReplayData:
    """The replay of one attempt (see the module docstring). ``zero_ns`` is the timeline's start: the first
    movement (as the Monitor's simulation times), else the explorer's activation (no movement)."""

    def __init__(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        self.extent: Mapping[str, Any] | None = None
        self.analysis: Mapping[str, Any] = {}
        self.windows: dict[int, Window] = {}
        self.all_goals: list[Goal] = []
        self.goals_from = -1  # the window whose goal history ``all_goals`` is
        self.x: np.ndarray[Any, np.dtype[np.float64]] = np.empty(0)
        self.y: np.ndarray[Any, np.dtype[np.float64]] = np.empty(0)
        self.flags: np.ndarray[Any, np.dtype[np.uint8]] = np.empty(0, dtype=np.uint8)
        self.stamps: np.ndarray[Any, np.dtype[np.int64]] = np.empty(0, dtype=np.int64)
        self.path_complete = False
        self.path_after = 0  # the run map's next ``after_sequence``
        self.path_note = ""
        # The attempt's own world reference (the run map's ``base``); its target cells order the masks.
        self.world: WorldMap | None = None
        self.target: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
        # The last explored image and its snapshot (the map caches its base image by identity).
        self._pixels: tuple[tuple[int, int], np.ndarray[Any, np.dtype[np.uint8]]] | None = None

    # ---------- inputs ----------

    @property
    def moved(self) -> bool:
        return isinstance((self.extent or {}).get("first_motion_ns"), int)

    @property
    def zero_ns(self) -> int:
        extent = self.extent or {}
        for key in ("first_motion_ns", "simulation_start_ns"):
            value = extent.get(key)
            if isinstance(value, int):
                return value
        return 0

    @property
    def end_ns(self) -> int:
        """The terminal trigger, else the last path stamp (for example a failed run)."""
        extent = self.extent or {}
        end = extent.get("end_ns")
        if isinstance(end, int) and end > self.zero_ns:
            return end
        return max(self.zero_ns, int(self.stamps[-1]) if len(self.stamps) else self.zero_ns)

    @property
    def ready(self) -> bool:
        return self.extent is not None

    def set_world(self, base: Mapping[str, Any]) -> None:
        """The run map's ``base``: the reference the attempt recorded (``spawns.map`` may be newer)."""
        self.world = WorldMap.from_reply(dict(base))
        rows, columns = np.nonzero(self.world.pixels == FREE)
        self.target = (rows, columns)
        self._pixels = None

    def add_path_page(self, reply: Mapping[str, Any]) -> int | None:
        """A ``results.map`` page; returns the next ``after_sequence`` (None when complete)."""
        points, stamps = reply.get("points") or [], reply.get("stamps")
        if stamps is None or len(points) != len(stamps):
            raise ValueError("the run map has no vertex times: the engine image predates replay")
        if reply.get("base"):
            self.set_world(reply["base"])
        if points:
            array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            self.x = np.concatenate([self.x, array[:, 0]])
            self.y = np.concatenate([self.y, array[:, 1]])
            self.flags = np.concatenate([self.flags, array[:, 2].astype(np.uint8)])
            self.stamps = np.concatenate([self.stamps, np.asarray(stamps, dtype=np.int64)])
        notes = reply.get("notes") or []
        self.path_note = " · ".join(str(note) for note in notes)
        last = int(reply.get("last_sequence") or 0)
        sequence = int(reply.get("from_sequence") or 1) + len(points) - 1
        self.path_after = max(self.path_after, sequence)
        if not points or sequence >= last:
            self.path_complete = True
            return None
        return sequence

    def window_index(self, time_ns: int) -> int:
        return max(0, (time_ns - self.zero_ns) // WINDOW_NS)

    @property
    def last_window(self) -> int:
        return self.window_index(max(self.zero_ns, self.end_ns - 1))

    def window_span(self, index: int) -> tuple[int, int]:
        start = self.zero_ns + index * WINDOW_NS
        return start, start + WINDOW_NS

    def add_window(self, reply: Mapping[str, Any], index: int | None) -> None:
        """A ``replay.window`` reply; ``index`` None for the probe that only brings the extent."""
        self.extent = reply.get("extent") or {}
        self.analysis = reply.get("analysis") or {}
        if index is None:
            return
        goals = [Goal.from_entry(entry) for entry in reply.get("goals") or []]
        ticks = []
        for entry in reply.get("ticks") or []:
            ticks.append(
                Tick(
                    int(entry["simulation_ns"]),
                    Decisions.from_layer(entry["decisions"]) if entry.get("decisions") else None,
                    ChunkGrid.from_layer(entry["chunks"]) if entry.get("chunks") else None,
                    entry.get("figures") or {},
                )
            )
        explored = reply.get("explored") or {}
        self.windows[index] = Window(
            index,
            ticks,
            goals,
            list(explored.get("snapshots") or []),
            list(reply.get("coverage") or []),
            list(reply.get("motion") or []),
            list(reply.get("resources") or []),
        )
        if index >= self.goals_from:  # a later window's history covers every earlier goal
            self.all_goals, self.goals_from = goals, index

    def needed(self, time_ns: int) -> list[int]:
        """Windows the frame at ``time_ns`` needs: its own and the one before (for the latest tick)."""
        index = self.window_index(time_ns)
        return [i for i in (index, index - 1) if i >= 0]

    def missing(self, time_ns: int) -> list[int]:
        return [i for i in self.needed(time_ns) if i not in self.windows]

    def evict(self, time_ns: int) -> None:
        """Keep the windows around ``time_ns`` and the last one (the full goal history)."""
        index = self.window_index(time_ns)
        keep = {index - 1, index, index + 1, self.last_window}
        for position in sorted(self.windows, key=lambda i: abs(i - index), reverse=True):
            if len(self.windows) <= KEPT_WINDOWS:
                break
            if position not in keep:
                del self.windows[position]

    def ticks_near(self, time_ns: int) -> list[int]:
        """Tick times of the windows around ``time_ns`` (a step never jumps over a window not loaded)."""
        index = self.window_index(time_ns)
        stamps: list[int] = []
        for position in (index - 1, index, index + 1):
            window = self.windows.get(position)
            if window is not None:
                stamps.extend(tick.simulation_ns for tick in window.ticks)
        return sorted(stamps)

    # ---------- frames ----------

    def frame(self, time_ns: int) -> Frame:
        count = int(np.searchsorted(self.stamps, time_ns, side="right"))
        pose = None
        if count:
            x, y = float(self.x[count - 1]), float(self.y[count - 1])
            heading = 0.0
            for back in range(count - 2, max(-1, count - 40), -1):
                dx, dy = x - float(self.x[back]), y - float(self.y[back])
                if math.hypot(dx, dy) > 0.02:
                    heading = math.atan2(dy, dx)
                    break
            pose = (x, y, heading)
        index = self.window_index(time_ns)
        window = self.windows.get(index)
        tick = None
        for candidate in (window, self.windows.get(index - 1)):
            if candidate is None:
                continue
            stamps = [t.simulation_ns for t in candidate.ticks]
            position = bisect_right(stamps, time_ns)
            if position:
                tick = candidate.ticks[position - 1]
                break
        goals: dict[int, Goal] = {}
        current: Goal | None = None
        for goal in self.all_goals:
            if goal.start_ns > time_ns:
                continue
            ended = goal.end_ns is not None and goal.end_ns <= time_ns
            shown = goal if ended else Goal(goal.number, goal.x, goal.y, goal.start_ns, None, None)
            goals[goal.number] = shown
            if (
                not ended
                and goal.x is not None
                and (current is None or goal.start_ns > current.start_ns)
            ):
                current = goal  # the open goal started last
        return Frame(
            time_ns,
            count,
            pose,
            self.explored_at(window, time_ns) if window is not None else None,
            tick,
            goals,
            None
            if current is None or current.x is None or current.y is None
            else (current.x, current.y),
            self.latest(index, "coverage", time_ns),
            self.latest(index, "motion", time_ns),
            self.latest(index, "resources", time_ns),
        )

    def latest(self, index: int, stream: str, time_ns: int) -> Mapping[str, Any] | None:
        """The latest sample of ``stream`` at or before ``time_ns`` in window ``index`` or the one before."""
        for position in (index, index - 1):
            window = self.windows.get(position)
            if window is None:
                continue
            rows: list[Mapping[str, Any]] = getattr(window, stream)
            found = bisect_right([int(row["clock_ns"]) for row in rows], time_ns)
            if found:
                return rows[found - 1]
        return None

    def explored_at(
        self, window: Window, time_ns: int
    ) -> np.ndarray[Any, np.dtype[np.uint8]] | None:
        """The explored-cell image of the latest snapshot at or before ``time_ns`` (keyframe plus changes)."""
        if self.target is None or self.world is None:
            return None
        stamps = [int(snapshot["clock_ns"]) for snapshot in window.snapshots]
        last = bisect_right(stamps, time_ns) - 1
        if last < 0:
            return None
        if self._pixels is not None and self._pixels[0] == (window.index, last):
            return self._pixels[1]
        mask = window.explored.get(last)
        if mask is None:
            first = max(
                (i for i in range(last + 1) if window.snapshots[i]["kind"] == "keyframe"),
                default=-1,
            )
            if first < 0:
                return None
            for position in range(first, last + 1):
                bits = self._decode(window.snapshots[position])
                if bits is None:
                    return None
                mask = bits if mask is None else mask ^ bits
                window.explored[position] = mask
            if mask is None:
                return None
        rows, columns = self.target
        if mask.shape != rows.shape:
            return None
        pixels = np.zeros(self.world.pixels.shape, dtype=np.uint8)
        pixels[rows[mask], columns[mask]] = 255
        self._pixels = ((window.index, last), pixels)
        return pixels

    @staticmethod
    def _decode(snapshot: Mapping[str, Any]) -> np.ndarray[Any, np.dtype[np.bool_]] | None:
        try:
            data = zlib.decompress(base64.b64decode(snapshot["mask"]))
        except (ValueError, zlib.error, KeyError):
            return None
        cells = int(snapshot["reference_cells"])
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        if len(bits) < cells:
            return None
        return bits[:cells].astype(bool)

    def path_to(self, path: LivePath, count: int) -> None:
        """``path`` as the vertices up to ``count`` (extended when it grows, rebuilt when it shrinks)."""
        if count < len(path):
            path.clear()
        start = len(path)
        if count > start:
            path.extend(
                zip(
                    self.x[start:count].tolist(),
                    self.y[start:count].tolist(),
                    self.flags[start:count].tolist(),
                    strict=True,
                )
            )

    def goal_starts(self) -> list[int]:
        return [goal.start_ns for goal in self.all_goals]


class ReplayController(QtCore.QObject):
    """Loads and plays one replay: requests (tagged with a generation, so replies of an earlier replay are
    ignored), the waiting list asked again every second (``retry``), the failed windows (asked again only on a
    seek or step), the 50 ms playback timer and the replay time. It emits ``changed`` when the frame should be
    drawn again, ``started`` once the run's extent is known, ``note`` for a failure and ``playing``."""

    changed = QtCore.Signal()
    started = QtCore.Signal()
    note = QtCore.Signal(str)
    playing = QtCore.Signal(bool)

    def __init__(self, transport: Transport, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.transport = transport
        self.data: ReplayData | None = None
        self.generation = 0
        self.time = 0
        self.speed = SPEEDS[0]
        self.moved_by_reader = False
        self.inflight: set[str] = set()
        self.waiting: dict[str, tuple[str, dict[str, Any]]] = {}
        self.failed_windows: dict[int, str] = {}
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.advance)
        # The engine's own words for an image older than replay (set by the page).
        self.stale: Callable[[str], str | None] = lambda error: None

    @property
    def active(self) -> bool:
        return self.data is not None

    @property
    def is_playing(self) -> bool:
        return self.timer.isActive()

    def start(self, attempt_id: str) -> None:
        self.stop()
        self.data = ReplayData(attempt_id)
        self._request(
            "probe", "replay.window", {"attempt_id": attempt_id, "start_ns": 0, "end_ns": WINDOW_NS}
        )
        self._request("map", "results.map", {"attempt_id": attempt_id, "after_sequence": 0})

    def stop(self) -> None:
        self.pause()
        self.generation += 1
        self.data = None
        self.moved_by_reader = False
        self.inflight.clear()
        self.waiting.clear()
        self.failed_windows.clear()

    # requests

    def _tag(self, kind: str) -> str:
        return f"replay:{self.generation}:{kind}"

    def _request(self, kind: str, method: str, params: dict[str, Any]) -> None:
        tag = self._tag(kind)
        if tag in self.inflight:
            return
        self.inflight.add(tag)
        self.waiting.pop(tag, None)
        self.transport.submit(tag, method, params)

    def retry(self) -> None:
        """Every second: ask again for what was loading or still finishing."""
        for tag, (method, params) in list(self.waiting.items()):
            kind = tag.split(":", 2)[2]
            self._request(kind, method, params)

    def request_windows(self, *, ahead: bool = False, again: bool = False) -> None:
        """The windows the current time needs (and the next one while playing); ``again`` retries failures."""
        data = self.data
        if data is None or not data.ready:
            return
        wanted = data.missing(self.time)
        index = data.window_index(self.time)
        if ahead and index + 1 <= data.last_window and index + 1 not in data.windows:
            wanted.append(index + 1)
        for position in wanted:
            if position in self.failed_windows and not again:
                continue
            self.failed_windows.pop(position, None)
            tag = self._tag(f"window:{position}")
            if tag in self.waiting:
                continue  # asked again by ``retry``, once a second
            start, end = data.window_span(position)
            self._request(
                f"window:{position}",
                "replay.window",
                {"attempt_id": data.attempt_id, "start_ns": start, "end_ns": end},
            )

    def received(self, tag: str, result: dict[str, Any]) -> bool:
        if not tag.startswith("replay:"):
            return False
        self.inflight.discard(tag)
        _, generation, kind = tag.split(":", 2)
        data = self.data
        if data is None or int(generation) != self.generation:
            return True  # an earlier replay's reply
        state = result.get("state")
        if state in ("loading", "building", "running"):
            self.waiting[tag] = self._params(kind, data)
            return True
        if state != "ready":
            self._failed(kind, str(result.get("reason") or state))
            return True
        try:
            if kind == "map":
                after = data.add_path_page(result)
                if after is not None:
                    self._request(
                        "map",
                        "results.map",
                        {"attempt_id": data.attempt_id, "after_sequence": after},
                    )
                elif not self.moved_by_reader:
                    self.time = data.end_ns  # the end may come from the path (no terminal trigger)
            elif kind == "probe":
                first = not data.ready
                data.add_window(result, None)
                if first:
                    self.time = data.end_ns
                    self.request_windows()
                    self.started.emit()
            else:
                index = int(kind.split(":")[1])
                data.add_window(result, index)
                self.failed_windows.pop(index, None)
                data.evict(self.time)
        except (KeyError, TypeError, ValueError) as exc:
            self._failed(kind, str(exc))
            return True
        self.changed.emit()
        return True

    def failed_reply(self, tag: str, error: str) -> bool:
        if not tag.startswith("replay:"):
            return False
        self.inflight.discard(tag)
        _, generation, kind = tag.split(":", 2)
        if self.data is not None and int(generation) == self.generation:
            self._failed(kind, self.stale(error) or error)
        return True

    def _params(self, kind: str, data: ReplayData) -> tuple[str, dict[str, Any]]:
        identity = data.attempt_id
        if kind == "map":
            return "results.map", {"attempt_id": identity, "after_sequence": data.path_after}
        if kind == "probe":
            return "replay.window", {"attempt_id": identity, "start_ns": 0, "end_ns": WINDOW_NS}
        start, end = data.window_span(int(kind.split(":")[1]))
        return "replay.window", {"attempt_id": identity, "start_ns": start, "end_ns": end}

    def _failed(self, kind: str, reason: str) -> None:
        if kind.startswith("window:"):
            self.failed_windows[int(kind.split(":")[1])] = reason
            self.pause()  # a deterministic failure must not be asked for 20 times a second
        self.note.emit(f"Replay unavailable: {reason}")

    # playback

    def pause(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.playing.emit(False)

    def toggle(self) -> None:
        data = self.data
        if data is None or not data.ready:
            return
        if self.timer.isActive():
            self.pause()
        else:
            if self.time >= data.end_ns:
                self.time = data.zero_ns  # play again from the start
            self.moved_by_reader = True
            self.timer.start()
            self.playing.emit(True)
            self.request_windows(ahead=True, again=True)
        self.changed.emit()

    def advance(self) -> None:
        data = self.data
        if data is None or not data.ready:
            self.pause()
            return
        self.request_windows(ahead=True)
        if data.missing(self.time):
            return  # waiting for the window at this time
        step = round(self.timer.interval() * 1e6 * self.speed)
        self.time = min(data.end_ns, self.time + step)
        if self.time >= data.end_ns:
            self.pause()
        self.changed.emit()

    def step(self, direction: int) -> None:
        """To the next or previous analysis tick of the neighbouring windows (else one simulation second)."""
        data = self.data
        if data is None or not data.ready:
            return
        stamps = data.ticks_near(self.time)
        if direction > 0:
            target = next((stamp for stamp in stamps if stamp > self.time), None)
        else:
            target = next((stamp for stamp in reversed(stamps) if stamp < self.time), None)
        if target is None:
            target = self.time + direction * 1_000_000_000
        self._move(target)

    def seek(self, fraction: float) -> None:
        data = self.data
        if data is None or not data.ready:
            return
        self._move(data.zero_ns + round(fraction * (data.end_ns - data.zero_ns)))

    def _move(self, time_ns: int) -> None:
        data = self.data
        assert data is not None
        self.moved_by_reader = True
        self.time = min(data.end_ns, max(data.zero_ns, time_ns))
        for index in data.needed(self.time):
            self.failed_windows.pop(index, None)  # a seek or step asks again
        self.request_windows(again=True)
        self.changed.emit()

    def set_speed(self, speed: float) -> None:
        self.speed = speed


# ---------- the replay bar ----------

TIME_TEXT = TextStyle(13, 400, 18, tabular=True)
TIME_VALUE = TextStyle(14, 600, 18, tabular=True)


class ReplayTrack(Themed):
    """The timeline (28 px): the rail (``data.remaining``), the played part (``accent.primary``), goal starts
    as 5 px ticks, coverage milestones as hollow 7 px circles, the 14 px thumb. Click or drag to seek."""

    seek = QtCore.Signal(float)  # fraction of the run

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.fraction = 1.0
        self.goals: list[float] = []
        self.milestones: list[float] = []
        self.setFixedHeight(28)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Replay time")

    def set_marks(self, goals: Sequence[float], milestones: Sequence[float]) -> None:
        self.goals, self.milestones = list(goals), list(milestones)
        self.update()

    def set_fraction(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        if fraction != self.fraction:
            self.fraction = fraction
            self.update()

    def _emit(self, x: float) -> None:
        width = max(1.0, self.width() - 14.0)
        self.seek.emit(min(1.0, max(0.0, (x - 7) / width)))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._emit(event.position().x())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._emit(event.position().x())

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        step = {QtCore.Qt.Key.Key_Left: -0.01, QtCore.Qt.Key.Key_Right: 0.01}.get(
            QtCore.Qt.Key(event.key())
        )
        if step is None:
            super().keyPressEvent(event)
            return
        self.seek.emit(min(1.0, max(0.0, self.fraction + step)))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.theme
        left, width = 7.0, max(1.0, self.width() - 14.0)
        accent = theme.color("accent.primary")
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("data.remaining"))
        painter.drawRoundedRect(QtCore.QRectF(left, 12, width, 4), 2, 2)
        painter.setBrush(accent)
        painter.drawRoundedRect(QtCore.QRectF(left, 12, width * self.fraction, 4), 2, 2)
        tick = QtGui.QColor(theme.color("text.tertiary"))
        tick.setAlphaF(0.7)
        for fraction in self.goals:
            painter.fillRect(QtCore.QRectF(snap(left + width * fraction), 20, 1, 5), tick)
        for fraction in self.milestones:
            center = QtCore.QPointF(left + width * fraction, 14)
            painter.setPen(QtGui.QPen(accent, 1.5))
            painter.setBrush(theme.color("surface.card"))
            painter.drawEllipse(center, 2.75, 2.75)
        thumb = QtCore.QPointF(left + width * self.fraction, 14)
        layers = ((0, 1, 2, 0, theme.tokens["overlay.shadowSmall"]),)
        margin = shadow_margin(layers)
        painter.drawImage(
            QtCore.QPointF(thumb.x() - 7 - margin, thumb.y() - 7 - margin),
            shadow_image(14, 14, 7, layers),
        )
        painter.setPen(QtGui.QPen(accent, 2))
        painter.setBrush(theme.color("surface.card"))
        painter.drawEllipse(thumb, 6, 6)
        if self.hasFocus():
            painter.setPen(QtGui.QPen(theme.color("text.ink"), 1, QtCore.Qt.PenStyle.DotLine))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(thumb, 9, 9)


class SpeedBox(DoubleSpinBox):
    """The replay speed in multiples of real time: typed (0.25× to 64×, two decimals) or stepped one step at a
    time by its arrows, the wheel while focused and the arrow keys: to the next whole speed from 1× up, to the
    next quarter below 1× (a typed 2.5× steps to 3× or 2×). ``valueChanged`` fires on Enter, on leaving the box
    and on each step, not on every keystroke."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, "×", parent)
        self.setLocale(QtCore.QLocale.c())
        self.setDecimals(2)
        self.setRange(*SPEED_RANGE)
        self.setValue(SPEEDS[0])
        self.setKeyboardTracking(False)
        self.setFixedSize(88, 32)
        self.setToolTip(
            "Replay speed (× real time): type any speed; the arrows step it by 1× (0.25× below 1×)"
        )
        self.setAccessibleName("Replay speed")

    def stepBy(self, steps: int) -> None:
        value = self.value()
        for _ in range(abs(steps)):
            value = self.next_speed(value, 1 if steps > 0 else -1)
        self.setValue(value)

    @staticmethod
    def next_speed(value: float, direction: int) -> float:
        """The next step above (``direction`` 1) or below (-1) ``value``: whole speeds from 1×, quarters below."""
        if direction > 0:
            if value < 1.0:
                return min(1.0, (math.floor(value / SPEED_FINE + 1e-9) + 1) * SPEED_FINE)
            return float(math.floor(value + 1e-9) + 1)
        if value <= 1.0:
            return (math.ceil(value / SPEED_FINE - 1e-9) - 1) * SPEED_FINE
        return float(math.ceil(value - 1e-9) - 1)

    def textFromValue(self, value: float) -> str:
        return f"{value:g}"


class ReplayBar(Themed):
    """``.rbar``: 48 px with a bottom rule, padding 0 16 0 12, gap 8: play or pause, step back, step forward,
    "0:41 of 1:12 · sim", the timeline and the speed: presets (1×, 4×, 16×) and a box for any speed (a typed
    speed that is not a preset leaves no preset selected)."""

    play = QtCore.Signal()
    step = QtCore.Signal(int)  # -1 or +1 analysis tick
    seek = QtCore.Signal(float)
    speed = QtCore.Signal(float)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.toggle = Button(themes, icon="play", parent=self)
        self.toggle.setToolTip("Play")
        self.toggle.setAccessibleName("Play")
        self.toggle.clicked.connect(self.play)
        self.back = Button(themes, icon="skipback", variant="ghost", parent=self)
        self.back.setToolTip("Previous analysis tick")
        self.back.setAccessibleName("Previous analysis tick")
        self.back.clicked.connect(lambda: self.step.emit(-1))
        self.forward = Button(themes, icon="skipfwd", variant="ghost", parent=self)
        self.forward.setToolTip("Next analysis tick")
        self.forward.setAccessibleName("Next analysis tick")
        self.forward.clicked.connect(lambda: self.step.emit(1))
        self.track = ReplayTrack(themes, self)
        self.track.seek.connect(self.seek)
        self.speeds = SegmentedControl(
            themes,
            [("1×", None), ("4×", None), ("16×", None)],
            tooltips=("Real time", "4 × real time", "16 × real time"),
            parent=self,
        )
        self.speeds.changed.connect(self._preset)
        self.speed_box = SpeedBox(themes, self)
        self.speed_box.valueChanged.connect(self._typed)
        self.position, self.length = 0.0, 0.0
        self.note = ""
        self.setFixedHeight(48)

    def _preset(self, index: int) -> None:
        speed = SPEEDS[index]
        if self.speed_box.value() != speed:
            blocked = self.speed_box.blockSignals(True)
            self.speed_box.setValue(speed)
            self.speed_box.blockSignals(blocked)
        self.speed.emit(speed)

    def _typed(self, speed: float) -> None:
        group = self.speeds.group
        preset = next((index for index, value in enumerate(SPEEDS) if value == speed), None)
        blocked = group.blockSignals(True)  # the box already holds the speed: no ``_preset``
        if preset is None:
            checked = group.checkedButton()
            if checked is not None:
                group.setExclusive(False)  # an exclusive group cannot be left without a choice
                checked.setChecked(False)
                group.setExclusive(True)
        else:
            self.speeds.segments[preset].setChecked(True)
        group.blockSignals(blocked)
        self.speeds.update()
        self.speed.emit(speed)

    def set_playing(self, playing: bool) -> None:
        self.toggle.glyph = "pause" if playing else "play"
        word = "Pause" if playing else "Play"
        self.toggle.setToolTip(word)
        self.toggle.setAccessibleName(word)
        self.toggle.update()

    def set_time(self, position: float, length: float, note: str = "") -> None:
        """``position`` and ``length`` in simulation seconds from the timeline's start."""
        self.position, self.length, self.note = position, length, note
        self.track.set_fraction(position / length if length > 0 else 1.0)
        self._layout()
        self.update()

    def time_text(self) -> tuple[str, str]:
        rest = f" of {clock(self.length)} · sim"
        return clock(self.position), rest + (f" · {self.note}" if self.note else "")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def _layout(self) -> None:
        top = (47 - 32) / 2
        x = 12.0
        for button in (self.toggle, self.back, self.forward):
            width = button.natural_width()
            button.setGeometry(round(x), round(top), round(width), 32)
            x += width + 8
        value, rest = self.time_text()
        self.text_left = x + 4
        x = self.text_left + TIME_VALUE.advance(value) + TIME_TEXT.advance(rest) + 6
        seg = self.speeds.natural_width()
        box = self.speed_box.width()
        self.speed_box.move(round(self.width() - 16 - box), round((47 - 32) / 2))
        right = self.width() - 16 - box - 8 - seg
        self.speeds.setGeometry(round(right), round((47 - 32) / 2), round(seg), 32)
        self.track.setGeometry(
            round(x + 8), round((47 - 28) / 2), max(40, round(right - 16 - x)), 28
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.fillRect(
            QtCore.QRect(0, self.height() - 1, self.width(), 1), theme.color("border.line")
        )
        value, rest = self.time_text()
        top = (47 - 18) / 2
        advance = draw_text(
            painter, self.text_left, top, value, TIME_VALUE, theme.color("text.ink")
        )
        draw_text(
            painter, self.text_left + advance, top, rest, TIME_TEXT, theme.color("text.secondary")
        )
