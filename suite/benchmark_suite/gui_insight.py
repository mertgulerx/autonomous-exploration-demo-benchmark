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

"""Explorer insight on the Monitor map (docs/suite-gui-design/MONITOR_INSIGHT.md, screens 18–21).

The layer system (two groups, Run and Explorer analysis, each layer with a switch; the choice remembered per
viewer, the first time only today's three run layers), the analysis marks in ``data.analysis`` (frontier
candidates, the target ring, the planned route with numbered badges, the DP pool, the chunk cache) and the goal
history, hover with the inverted tooltip and click to pin, the status chip on the map and the legend of the
visible layers. Positions arrive in world metres (the worker placed them with the frozen map-to-world
alignment, an estimate); the marks show the explorer's analysis, never a judgement.
"""

import base64
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import json
import math
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_controls import paint_switch_track
from .gui_mapview import LiveMap, MapButton, MapControls, MapLegend, snapped
from .gui_menus import INSET, ITEM_HEIGHT, ITEM_TEXT, REASON_TEXT, SEPARATOR_HEIGHT
from .gui_maps import Placement
from .gui_style import fixed, number
from .gui_theme import Theme, ThemeManager, font
from .gui_widgets import (
    PopupHost,
    TextStyle,
    Themed,
    draw_text,
    rounded_path,
)

# ---------- layers ----------


@dataclass(frozen=True, slots=True)
class LayerSpec:
    key: str
    group: str  # "run" or "analysis"
    title: str  # the Layers menu
    legend: str  # the legend line


LAYERS = (
    LayerSpec("path", "run", "Ground-truth path", "Path (Gazebo)"),
    LayerSpec("explored", "run", "Explored cells", "Explored"),
    LayerSpec("navigation", "run", "Nav2 goal and plan", "Nav2 goal and plan (estimate)"),
    LayerSpec("goals", "run", "Goal history", "Goal history"),
    LayerSpec("candidates", "analysis", "Frontier candidates", "Frontiers (analysis)"),
    LayerSpec("route", "analysis", "Planned route", "Route"),
    LayerSpec("pool", "analysis", "DP pool", "DP pool"),
    LayerSpec("chunks", "analysis", "Chunk cache", "Chunk cache"),
)
LAYER_KEYS = tuple(spec.key for spec in LAYERS)
DEFAULT_VISIBLE = frozenset({"path", "explored", "navigation"})  # D7: today's three layers first
SETTINGS_KEY = "monitor/layers"
# Engine layers served by workers since the explorer insight (older engine images reject them).
INSIGHT_ENGINE_LAYERS = ("decisions", "chunks", "goals")


def load_visible() -> frozenset[str]:
    """The viewer's layer choice (QSettings), else the three run layers of today."""
    stored = QtCore.QSettings().value(SETTINGS_KEY)
    try:
        keys = json.loads(str(stored)) if stored is not None else None
    except ValueError:
        keys = None
    if not isinstance(keys, list):
        return DEFAULT_VISIBLE
    return frozenset(key for key in keys if key in LAYER_KEYS)


def save_visible(visible: Iterable[str]) -> None:
    QtCore.QSettings().setValue(
        SETTINGS_KEY, json.dumps([key for key in LAYER_KEYS if key in set(visible)])
    )


def engine_layers(
    visible: frozenset[str], decisions_view: bool, insight: bool, analysis: bool = True
) -> list[str]:
    """The ``monitor.get`` layers the visible map layers and the Decisions view need (hidden layers are not
    requested); ``insight``: the engine serves the explorer insight layers; ``analysis``: the run's explorer
    has the frontier insight (external explorers do not; their goal history is still served)."""
    layers = [name for name in ("explored", "navigation") if name in visible]
    if insight:
        if analysis and (decisions_view or visible & {"candidates", "route", "pool"}):
            layers.append("decisions")
        if analysis and "chunks" in visible:
            layers.append("chunks")
        if "goals" in visible:
            layers.append("goals")
    return layers


# ---------- layer data ----------


def _rank(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


@dataclass(frozen=True, slots=True)
class Candidate:
    """One frontier candidate of the shown tick (world metres)."""

    id: int
    x: float
    y: float
    size_cells: int
    gain: float | None
    path_cost: float | None
    time_cost: float | None
    start_cost: float | None
    distance_m: float | None
    score_order: int | None
    pool_rank: int | None
    route_position: int | None
    greedy_position: int | None


@dataclass(frozen=True, slots=True)
class Decisions:
    """The ``decisions`` layer of one analysis tick."""

    tick: int
    simulation_ns: int
    mode: str | None
    robot: tuple[float, float, float]
    candidates: tuple[Candidate, ...]
    route: tuple[int, ...]
    route_cost: float | None
    target: int | None
    greedy_first: int | None

    @classmethod
    def from_layer(cls, data: Mapping[str, Any]) -> "Decisions":
        candidates = []
        for item in data["candidates"]:
            x, y = number(item["world_x"]), number(item["world_y"])
            if x is None or y is None:
                raise ValueError("candidate without a world position")
            candidates.append(
                Candidate(
                    id=int(item["id"]),
                    x=x,
                    y=y,
                    size_cells=int(item["size_cells"]),
                    gain=number(item["gain"]),
                    path_cost=number(item["path_cost"]),
                    time_cost=number(item["time_cost"]),
                    start_cost=number(item["start_cost"]),
                    distance_m=number(item["distance_m"]),
                    score_order=_rank(item["score_order"]),
                    pool_rank=_rank(item["pool_rank"]),
                    route_position=_rank(item["route_position"]),
                    greedy_position=_rank(item["greedy_position"]),
                )
            )
        robot = tuple(float(value) for value in data["robot"])
        if len(robot) != 3 or not all(math.isfinite(value) for value in robot):
            raise ValueError("analysis without a robot pose")
        known = {candidate.id for candidate in candidates}
        route = tuple(int(identity) for identity in data["route"])
        if any(identity not in known for identity in route):
            raise ValueError("route names an unknown candidate")
        target, greedy = data.get("target"), data.get("greedy_first")
        return cls(
            tick=int(data["tick"]),
            simulation_ns=int(data["simulation_ns"]),
            mode=str(data["mode"]) if data.get("mode") else None,
            robot=(robot[0], robot[1], robot[2]),
            candidates=tuple(candidates),
            route=route,
            route_cost=number(data.get("route_cost")),
            target=int(target) if isinstance(target, int) else None,
            greedy_first=int(greedy) if isinstance(greedy, int) else None,
        )

    def candidate(self, identity: int | None) -> Candidate | None:
        return next((c for c in self.candidates if c.id == identity), None)

    @property
    def pool_size(self) -> int:
        return sum(c.pool_rank is not None for c in self.candidates)


CHUNK_HIT, CHUNK_REBUILT, CHUNK_RESET = 0, 1, 2


@dataclass(frozen=True, slots=True)
class ChunkGrid:
    """The ``chunks`` layer: the insight node's chunk states over the decision map, placed in the world."""

    origin: tuple[float, float, float]
    resolution: float
    size_cells: int
    map_width: int
    map_height: int
    columns: int
    rows: int
    states: bytes

    @classmethod
    def from_layer(cls, data: Mapping[str, Any]) -> "ChunkGrid":
        origin = tuple(float(value) for value in data["origin"])
        resolution = float(data["resolution"])
        columns, rows = int(data["columns"]), int(data["rows"])
        states = base64.b64decode(data["states"])
        if (
            len(origin) != 3
            or not all(math.isfinite(value) for value in (*origin, resolution))
            or resolution <= 0
            or len(states) != columns * rows
        ):
            raise ValueError("invalid chunk grid")
        return cls(
            (origin[0], origin[1], origin[2]),
            resolution,
            int(data["size_cells"]),
            int(data["map_width"]),
            int(data["map_height"]),
            columns,
            rows,
            states,
        )

    def cells(self) -> Iterable[tuple[int, list[tuple[float, float]]]]:
        """Each chunk's state and its four world corners (clipped to the decision map)."""
        x0, y0, yaw = self.origin
        c, s = math.cos(yaw), math.sin(yaw)
        size, resolution = self.size_cells, self.resolution
        for row in range(self.rows):
            for column in range(self.columns):
                left, bottom = column * size, row * size
                right = min(self.map_width, left + size)
                top = min(self.map_height, bottom + size)
                corners = []
                for cx, cy in ((left, bottom), (right, bottom), (right, top), (left, top)):
                    mx, my = cx * resolution, cy * resolution
                    corners.append((x0 + c * mx - s * my, y0 + s * mx + c * my))
                yield self.states[row * self.columns + column], corners


@dataclass(frozen=True, slots=True)
class Goal:
    """One goal Nav2 initialized: its number, world position, simulation start and end, reached or not."""

    number: int
    x: float | None
    y: float | None
    start_ns: int
    end_ns: int | None
    reached: bool | None

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> "Goal":
        end = entry.get("end_ns")
        reached = entry.get("reached")
        return cls(
            int(entry["goal"]),
            number(entry.get("x")),
            number(entry.get("y")),
            int(entry["start_ns"]),
            int(end) if isinstance(end, int) else None,
            reached if isinstance(reached, bool) else None,
        )


PIN_KEEP_M = 0.10  # a pinned candidate is kept while one lies this close to its dispatch point


@dataclass(slots=True)
class InsightLayers:
    """What the Monitor map shows of the explorer's analysis and the goal history."""

    decisions: Decisions | None = None
    chunks: ChunkGrid | None = None
    goals: dict[int, Goal] = field(default_factory=dict)
    # None: not known yet; False: the run has no analysis (``reason`` says why).
    available: bool | None = None
    reason: str = ""
    hovered: int | None = None
    pinned: int | None = None
    pinned_at: tuple[float, float] | None = None  # the pinned candidate's dispatch point

    def set_decisions(self, decisions: Decisions | None) -> None:
        """A new tick: keep the pin on the candidate at the same dispatch point (within 0.10 m), else clear it."""
        self.decisions = decisions
        self.hovered = None
        if self.pinned_at is None:
            return
        px, py = self.pinned_at
        match = None
        if decisions is not None:
            match = next(
                (c for c in decisions.candidates if math.hypot(c.x - px, c.y - py) <= PIN_KEEP_M),
                None,
            )
        self.pinned = match.id if match is not None else None
        self.pinned_at = (match.x, match.y) if match is not None else None

    def pin(self, identity: int | None) -> None:
        candidate = self.decisions.candidate(identity) if self.decisions is not None else None
        self.pinned = candidate.id if candidate is not None else None
        self.pinned_at = (candidate.x, candidate.y) if candidate is not None else None

    def apply_goals(self, page: Mapping[str, Any]) -> None:
        for entry in page["goals"]:
            goal = Goal.from_entry(entry)
            self.goals[goal.number] = goal

    def clear(self) -> None:
        self.decisions = self.chunks = None
        self.goals = {}
        self.hovered = self.pinned = None
        self.pinned_at = None


# ---------- marks (``maps.js``; sizes in screen pixels) ----------


def analysis_color(theme: Theme, alpha: float = 1.0) -> QtGui.QColor:
    color = QtGui.QColor(theme.color("data.analysis"))
    color.setAlphaF(alpha)
    return color


def _pen(
    color: QtGui.QColor, width: float, *, cap: QtCore.Qt.PenCapStyle | None = None
) -> QtGui.QPen:
    pen = QtGui.QPen(color, width)
    pen.setCapStyle(cap or QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    return pen


# Chunk states in the explorer's RViz colours (debug observer: hit green, rebuilt red, reset yellow), as tokens.
CHUNK_TOKENS = {
    CHUNK_HIT: "data.completed",
    CHUNK_REBUILT: "state.bad",
    CHUNK_RESET: "data.timeLimit",
}
HIT_OPACITY, CHANGED_FILL = 0.55, 0.10


def chunk_color(theme: Theme, state: int, alpha: float = 1.0) -> QtGui.QColor:
    color = QtGui.QColor(theme.color(CHUNK_TOKENS.get(state, "state.bad")))
    color.setAlphaF(alpha)
    return color


def paint_chunks(
    painter: QtGui.QPainter, grid: ChunkGrid, placement: Placement, theme: Theme
) -> None:
    """Chunk states as the explorer's RViz overlay colours them: hit chunks as thin green outlines, rebuilt
    chunks red and full resets yellow (1.4 px outline, 10 % fill; resets also dashed). Clipped to the world
    image; changed chunks are drawn over the hit grid."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setClipRect(placement.rect)
    changed: list[tuple[int, QtGui.QPolygonF]] = []
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.setPen(QtGui.QPen(chunk_color(theme, CHUNK_HIT, HIT_OPACITY), 1))
    for state, corners in grid.cells():
        polygon = QtGui.QPolygonF([placement.point(x, y) for x, y in corners])
        if state == CHUNK_HIT:
            painter.drawPolygon(polygon)
        else:
            changed.append((state, polygon))
    for state, polygon in changed:
        pen = _pen(chunk_color(theme, state), 1.4, cap=QtCore.Qt.PenCapStyle.FlatCap)
        if state == CHUNK_RESET:
            pen.setDashPattern([4 / 1.4, 3 / 1.4])
        painter.setPen(pen)
        painter.setBrush(chunk_color(theme, state, CHANGED_FILL))
        painter.drawPolygon(polygon)
    painter.restore()


GOAL_NUMBER = TextStyle(9.5, 500, 12)


def paint_goal_history(
    painter: QtGui.QPainter, goals: Iterable[Goal], placement: Placement, theme: Theme
) -> None:
    """Ended goals as small numbered × (6 px, halo): green when Nav2 reached them, else grey. The open goal is
    the large ink × of the Nav2 layer."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    halo = theme.color("map.halo")
    for goal in goals:
        if goal.end_ns is None or goal.x is None or goal.y is None:
            continue
        point = placement.point(goal.x, goal.y)
        d = 3.0
        cross = QtGui.QPainterPath()
        cross.moveTo(point.x() - d, point.y() - d)
        cross.lineTo(point.x() + d, point.y() + d)
        cross.moveTo(point.x() + d, point.y() - d)
        cross.lineTo(point.x() - d, point.y() + d)
        color = theme.color("data.completed" if goal.reached else "text.tertiary")
        for pen in (_pen(halo, 4.2), _pen(color, 1.7)):
            painter.setPen(pen)
            painter.drawPath(cross)
        # The number's baseline 4.5 px above the goal, outlined in the halo (SVG ``paint-order: stroke``).
        text = str(goal.number)
        x, baseline_y = point.x() + 5.5, point.y() - 4.5
        top = baseline_y - GOAL_NUMBER.baseline(0.0)
        outline = QtGui.QPainterPath()
        outline.addText(QtCore.QPointF(x, baseline_y), font(9.5, 500), text)
        painter.setPen(_pen(halo, 3))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(outline)
        draw_text(painter, x, top, text, GOAL_NUMBER, theme.color("text.secondary"))
    painter.restore()


def route_points(decisions: Decisions, placement: Placement) -> list[QtCore.QPointF]:
    points = [placement.point(decisions.robot[0], decisions.robot[1])]
    for identity in decisions.route:
        candidate = decisions.candidate(identity)
        if candidate is not None:
            points.append(placement.point(candidate.x, candidate.y))
    return points


def paint_route(
    painter: QtGui.QPainter, decisions: Decisions, placement: Placement, theme: Theme
) -> None:
    """The planned route: a dotted analysis line (2 px dots every 4.4 px) over a halo, from the analysed robot
    pose through the route order."""
    points = route_points(decisions, placement)
    if len(points) < 2:
        return
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.setOpacity(0.85)
    painter.setPen(_pen(theme.color("map.halo"), 4.2))
    painter.drawPolyline(QtGui.QPolygonF(points))
    painter.setOpacity(1.0)
    dotted = _pen(theme.color("data.analysis"), 2)
    dotted.setDashPattern([0.005, 4.4 / 2])
    painter.setPen(dotted)
    painter.drawPolyline(QtGui.QPolygonF(points))
    painter.restore()


def paint_candidates(
    painter: QtGui.QPainter,
    decisions: Decisions,
    placement: Placement,
    theme: Theme,
    *,
    pool: bool,
) -> None:
    """Candidates as analysis dots on a halo (the target larger, in a 1.5 px ring r 9). With the pool layer,
    pool members get a 17 % disc and the others turn hollow."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    analysis, halo = theme.color("data.analysis"), theme.color("map.halo")
    if pool:
        painter.setBrush(analysis_color(theme, 0.17))
        for candidate in decisions.candidates:
            if candidate.pool_rank is not None:
                painter.drawEllipse(placement.point(candidate.x, candidate.y), 8.5, 8.5)
    for candidate in decisions.candidates:
        point = placement.point(candidate.x, candidate.y)
        target = candidate.id == decisions.target
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setOpacity(0.95)
        painter.setBrush(halo)
        radius = 6.0 if target else 5.0
        painter.drawEllipse(point, radius, radius)
        painter.setOpacity(1.0)
        if pool and candidate.pool_rank is None:
            painter.setPen(QtGui.QPen(analysis, 1.4))
            painter.setBrush(halo)
            painter.drawEllipse(point, 3.1, 3.1)
        else:
            painter.setBrush(analysis)
            radius = 4.3 if target else 3.5
            painter.drawEllipse(point, radius, radius)
        if target:
            painter.setPen(QtGui.QPen(analysis, 1.5))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, 9.0, 9.0)
    painter.restore()


BADGE_NUMBER = TextStyle(9.5, 600, 12, tabular=True)


def paint_route_badges(
    painter: QtGui.QPainter,
    decisions: Decisions,
    placement: Placement,
    theme: Theme,
    bounds: QtCore.QRectF,
) -> None:
    """Route positions as badges (r 7, card fill, analysis ring and number) beside their candidates, placed
    clear of the other marks with a thin leader line."""
    taken = [
        (point.x(), point.y(), 6.0)
        for point in (placement.point(c.x, c.y) for c in decisions.candidates)
    ]
    robot = placement.point(decisions.robot[0], decisions.robot[1])
    taken.append((robot.x(), robot.y(), 10.0))

    def clear(x: float, y: float, radius: float) -> bool:
        return (
            bounds.left() < x - radius
            and x + radius < bounds.right()
            and bounds.top() < y - radius
            and y + radius < bounds.bottom()
            and all(math.hypot(x - tx, y - ty) >= radius + tr for tx, ty, tr in taken)
        )

    analysis = theme.color("data.analysis")
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    for position, identity in enumerate(decisions.route, start=1):
        candidate = decisions.candidate(identity)
        if candidate is None:
            continue
        point = placement.point(candidate.x, candidate.y)
        spot = None
        for reach in (12.0, 16.0, 21.0):
            for degrees in (45, 135, -45, -135, 90, -90, 0, 180):
                angle = math.radians(degrees)
                x = point.x() + reach * math.cos(angle)
                y = point.y() - reach * math.sin(angle)
                if clear(x, y, 7.5):
                    spot = QtCore.QPointF(x, y)
                    break
            if spot is not None:
                break
        if spot is None:
            spot = QtCore.QPointF(point.x() + 12, point.y() - 12)
        taken.append((spot.x(), spot.y(), 7.5))
        painter.setOpacity(0.5)
        painter.setPen(QtGui.QPen(analysis, 1))
        painter.drawLine(point, spot)
        painter.setOpacity(1.0)
        painter.setPen(QtGui.QPen(analysis, 1.2))
        painter.setBrush(theme.color("surface.card"))
        painter.drawEllipse(spot, 7.0, 7.0)
        text = str(position)
        draw_text(
            painter,
            spot.x() - BADGE_NUMBER.advance(text) / 2,
            spot.y() - BADGE_NUMBER.line / 2,
            text,
            BADGE_NUMBER,
            analysis,
        )
    painter.restore()


def paint_selection(painter: QtGui.QPainter, point: QtCore.QPointF, theme: Theme) -> None:
    """The dashed ink selection ring of the spawn maps: r 13, 6 % ink fill, 1 px dashed 3/2."""
    ink = theme.color("text.ink")
    fill = QtGui.QColor(ink)
    fill.setAlphaF(0.06)
    pen = QtGui.QPen(ink, 1)
    pen.setDashPattern([3, 2])
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(pen)
    painter.setBrush(fill)
    painter.drawEllipse(point, 13.0, 13.0)
    painter.restore()


def number_text(value: float | None, digits: int) -> str:
    """``gui_style.fixed`` with "—" for an unavailable value."""
    return fixed(value, digits) or "—"


def gain_text(candidate: Candidate) -> tuple[str, str]:
    """The gain and its unit: cells when the gain is the cluster size (``weight_gain_ws`` 1)."""
    if candidate.gain is None:
        return "—", ""
    if abs(candidate.gain - candidate.size_cells) < 1e-9:
        return f"{candidate.size_cells:,}", "cells"
    return f"{candidate.gain:,.1f}", ""


def candidate_facts(
    candidate: Candidate, decisions: Decisions, *, short: bool = False
) -> tuple[str, str]:
    """``Frontier #6`` and ``route 1 of 6 · score order 1 · pool 1 of 8``; ``short`` gives the tooltip's
    ``Route 2 · score order 2 · pool 2``. Absent parts are left out."""
    parts = []
    if candidate.route_position is not None:
        total = "" if short else f" of {len(decisions.route)}"
        parts.append(f"route {candidate.route_position}{total}")
    if candidate.score_order is not None:
        parts.append(f"score order {candidate.score_order}")
    if candidate.pool_rank is not None:
        total = "" if short else f" of {decisions.pool_size}"
        parts.append(f"pool {candidate.pool_rank}{total}")
    text = " · ".join(parts) or "off the route"
    return f"Frontier #{candidate.id}", text[:1].upper() + text[1:] if short else text


TIP_TITLE = TextStyle(12, 600, 16)
TIP_TEXT = TextStyle(12, 400, 16, tabular=True)
TIP_WIDTH = 214


def candidate_rows(candidate: Candidate) -> list[tuple[str, str]]:
    gain, unit = gain_text(candidate)
    return [
        ("Start cost", number_text(candidate.start_cost, 3)),
        ("Distance term", number_text(candidate.path_cost, 3)),
        ("Time term", number_text(candidate.time_cost, 3)),
        ("Gain (size)", gain if not unit else f"{gain} {unit}"),
        ("Distance", "—" if candidate.distance_m is None else f"{candidate.distance_m:.1f} m"),
    ]


def paint_candidate_tip(
    painter: QtGui.QPainter,
    candidate: Candidate,
    decisions: Decisions,
    anchor: QtCore.QPointF,
    bounds: QtCore.QRectF,
    theme: Theme,
) -> None:
    """The inverted tooltip (214 px): the frontier, its route, score order and pool, then its values; beside
    the candidate (14 px right, 10 px up), on the left near the right edge, kept inside ``bounds``."""
    title, sub = candidate_facts(candidate, decisions, short=True)
    rows = candidate_rows(candidate)
    height = 7 + 16 + 16 + 5 + 16 * len(rows) + 7
    x = anchor.x() + 14
    if x + TIP_WIDTH > bounds.right() - 4:
        x = anchor.x() - 14 - TIP_WIDTH
    y = min(max(anchor.y() - 10, bounds.top() + 4), bounds.bottom() - 4 - height)
    box = QtCore.QRectF(x, y, TIP_WIDTH, height)
    text = theme.color("overlay.tooltipText")
    muted = QtGui.QColor(text)
    muted.setAlphaF(0.72)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(theme.color("overlay.tooltip"))
    painter.drawPath(rounded_path(box, 6))
    draw_text(painter, x + 10, y + 7, title, TIP_TITLE, text)
    draw_text(painter, x + 10, y + 23, TIP_TEXT.elide(sub, TIP_WIDTH - 20), TIP_TEXT, muted)
    top = y + 7 + 32 + 5
    for label, value in rows:
        draw_text(painter, x + 10, top, label, TIP_TEXT, muted)
        draw_text(painter, x + TIP_WIDTH - 10 - TIP_TEXT.advance(value), top, value, TIP_TEXT, text)
        top += 16
    painter.restore()


# ---------- legend swatches ----------


SWATCH_WIDTHS = {
    "path": 22.0,
    "explored": 12.0,
    "navigation": 30.0,
    "goals": 10.0,
    "candidates": 8.0,
    "route": 22.0,
    "pool": 14.0,
    "chunks": 26.0,
}


def paint_swatch(painter: QtGui.QPainter, key: str, x: float, top: float, theme: Theme) -> None:
    """A layer's swatch in an 18 px line from ``top`` (the legend and the Layers menu)."""
    if key in ("path", "explored", "navigation"):
        MapLegend.paint_swatch(painter, key, x, top, theme)
        return
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    middle = top + 9
    analysis = theme.color("data.analysis")
    if key == "goals":
        d = 3.5
        cross = QtGui.QPainterPath()
        cross.moveTo(x + 5 - d, middle - d)
        cross.lineTo(x + 5 + d, middle + d)
        cross.moveTo(x + 5 + d, middle - d)
        cross.lineTo(x + 5 - d, middle + d)
        painter.setPen(_pen(theme.color("text.secondary"), 1.5))
        painter.drawPath(cross)
    elif key == "candidates":
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(analysis)
        painter.drawEllipse(QtCore.QPointF(x + 4, middle), 3.5, 3.5)
    elif key == "route":
        dotted = _pen(analysis, 2)
        dotted.setDashPattern([0.005, 4.4 / 2])
        painter.setPen(dotted)
        painter.drawLine(QtCore.QPointF(x + 1, middle), QtCore.QPointF(x + 21, middle))
    elif key == "pool":
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(analysis_color(theme, 0.17))
        painter.drawEllipse(QtCore.QPointF(x + 7, middle), 7, 7)
        painter.setBrush(analysis)
        painter.drawEllipse(QtCore.QPointF(x + 7, middle), 3.5, 3.5)
    elif key == "chunks":  # three small squares: hit, rebuilt, reset
        for index, state in enumerate((CHUNK_HIT, CHUNK_REBUILT, CHUNK_RESET)):
            painter.setPen(QtGui.QPen(chunk_color(theme, state), 1.2))
            painter.setBrush(chunk_color(theme, state, CHANGED_FILL))
            painter.drawRect(QtCore.QRectF(x + 1 + index * 8.5, middle - 3.5, 7, 7))
    painter.restore()


LEGEND_TEXT = TextStyle(12.5, 400, 18)


class LayerLegend(Themed):
    """``.legend`` of the visible layers that draw (40 px with a top rule, 13 px apart, 6 px after each
    swatch); the status is a chip on the map. Items that no longer fit are left out (the tooltip names all)."""

    GAP, SWATCH_GAP = 13.0, 6.0

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.visible: frozenset[str] = DEFAULT_VISIBLE
        self.setFixedHeight(40)

    def set_visible(self, visible: frozenset[str]) -> None:
        """``visible``: the switched-on layers less those the run cannot show."""
        if visible != self.visible:
            self.visible = visible
            self.setToolTip("\n".join(s.legend for s in LAYERS if s.key in visible))
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), theme.color("border.line"))
        top = 1 + (39 - 18) / 2
        x, limit = 20.0, self.width() - 20.0
        style = LEGEND_TEXT
        for spec in LAYERS:
            if spec.key not in self.visible:
                continue
            swatch = SWATCH_WIDTHS[spec.key]
            width = swatch + self.SWATCH_GAP + style.advance(spec.legend)
            if x + width > limit:
                break
            paint_swatch(painter, spec.key, x, top, theme)
            draw_text(
                painter,
                x + swatch + self.SWATCH_GAP,
                top,
                spec.legend,
                style,
                theme.color("text.secondary"),
            )
            x += width + self.GAP


# ---------- the Layers menu ----------


# The menu component's text, item and inset sizes (``gui_menus``); the group heads are the design's 28 px.
GROUP_TEXT = TextStyle(12, 600, 18)
MENU_WIDTH, GROUP_HEIGHT = 272, 28


class LayerMenu(Themed):
    """The Layers menu (``.laymenu``, 272 px, the menu component's look and sizes): Run and Explorer analysis,
    each layer with its swatch, its name and a switch. Unavailable layers are disabled with their reason.
    Unlike ``gui_menus.Menu`` it has group heads and keeps open: choosing an item flips its switch; the arrow
    keys, Home and End move, Enter or Space flips, Esc or a click outside closes it."""

    changed = QtCore.Signal(object)  # frozenset[str]

    def __init__(
        self,
        themes: ThemeManager,
        visible: frozenset[str],
        disabled: Mapping[str, str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.visible, self.disabled = set(visible), dict(disabled)
        self.hovered = -1
        self.host: PopupHost | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "none")
        self.setAccessibleName("Map layers")

    def entries(self) -> list[tuple[QtCore.QRectF, str, LayerSpec | None]]:
        """Rows: ("group", title), ("rule", ""), ("item", spec)."""
        rows: list[tuple[QtCore.QRectF, str, LayerSpec | None]] = []
        y, inner = float(INSET), self.width() - 2.0 * INSET
        for group, title in (("run", "Run"), ("analysis", "Explorer analysis")):
            if rows:
                rows.append((QtCore.QRectF(INSET, y, inner, SEPARATOR_HEIGHT), "rule", None))
                y += SEPARATOR_HEIGHT
            rows.append((QtCore.QRectF(INSET, y, inner, GROUP_HEIGHT), title, None))
            y += GROUP_HEIGHT
            for spec in LAYERS:
                if spec.group == group:
                    rows.append((QtCore.QRectF(INSET, y, inner, ITEM_HEIGHT), "item", spec))
                    y += ITEM_HEIGHT
        return rows

    def sizeHint(self) -> QtCore.QSize:
        rows = self.entries()
        bottom = rows[-1][0].bottom() if rows else 0.0
        return QtCore.QSize(MENU_WIDTH, math.ceil(bottom + INSET))

    def popup(
        self,
        host: PopupHost,
        anchor: QtWidgets.QWidget,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Open beside ``anchor`` (the Layers button), its top at the button's top, to the left of it."""
        window = host.window_widget
        origin = anchor.mapTo(window, QtCore.QPoint(0, 0))
        self.resize(MENU_WIDTH, 100)
        size = self.sizeHint()
        x = max(8, origin.x() - 4 - size.width())
        y = max(8, min(origin.y(), window.height() - 8 - size.height()))
        self.host = host
        self.hovered = -1
        host.open_popup(
            self,
            QtCore.QRect(x, y, size.width(), size.height()),
            opener=anchor,
            on_close=on_close,
            radius=8,
            delete_on_close=True,
        )
        self.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def _items(self) -> list[int]:
        return [
            index
            for index, (_, kind, spec) in enumerate(self.entries())
            if kind == "item" and spec is not None and spec.key not in self.disabled
        ]

    def _index_at(self, position: QtCore.QPointF) -> int:
        for index, (rect, kind, spec) in enumerate(self.entries()):
            if kind == "item" and rect.contains(position):
                return index
        return -1

    def _toggle(self, index: int) -> None:
        _, kind, spec = self.entries()[index]
        if kind != "item" or spec is None or spec.key in self.disabled:
            return
        self.visible ^= {spec.key}
        self.changed.emit(frozenset(self.visible))
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self._index_at(event.position())
        if index != self.hovered:
            self.hovered = index
            self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hovered = -1
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            index = self._index_at(event.position())
            if index >= 0:
                self._toggle(index)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        items = self._items()
        if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up) and items:
            step = 1 if key == QtCore.Qt.Key.Key_Down else -1
            position = items.index(self.hovered) if self.hovered in items else -1
            self.hovered = items[(position + step) % len(items)]
            self.update()
        elif key in (QtCore.Qt.Key.Key_Home, QtCore.Qt.Key.Key_End) and items:
            self.hovered = items[0] if key == QtCore.Qt.Key.Key_Home else items[-1]
            self.update()
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space):
            if self.hovered >= 0:
                self._toggle(self.hovered)
        elif key == QtCore.Qt.Key.Key_Escape and self.host is not None:
            self.host.close_popup()
        else:
            super().keyPressEvent(event)

    def focusNextPrevChild(self, next: bool) -> bool:
        return True

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
        painter.setBrush(theme.color("surface.card"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7.5)
        )
        for index, (rect, kind, spec) in enumerate(self.entries()):
            if kind == "rule":
                painter.fillRect(
                    QtCore.QRectF(rect.x() + 6, rect.y() + 4, rect.width() - 12, 1),
                    theme.color("border.line"),
                )
            elif spec is None:
                draw_text(
                    painter,
                    rect.x() + 10,
                    rect.y() + (GROUP_HEIGHT - 18) / 2 + 2,
                    kind,
                    GROUP_TEXT,
                    theme.color("text.secondary"),
                )
            else:
                self._paint_item(painter, rect, spec, index == self.hovered)

    def _paint_item(
        self, painter: QtGui.QPainter, rect: QtCore.QRectF, spec: LayerSpec, hovered: bool
    ) -> None:
        theme = self.theme
        reason = self.disabled.get(spec.key)
        if hovered and reason is None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("surface.hover"))
            painter.drawPath(rounded_path(rect, 6))
        top = rect.y() + (ITEM_HEIGHT - 18) / 2
        swatch = SWATCH_WIDTHS[spec.key]
        painter.setOpacity(0.4 if reason is not None else 1.0)
        paint_swatch(painter, spec.key, rect.x() + 10 + (26 - swatch) / 2, top, theme)
        painter.setOpacity(1.0)
        label = theme.color("text.disabled" if reason is not None else "text.ink")
        draw_text(painter, rect.x() + 10 + 26 + 10, top, spec.title, ITEM_TEXT, label)
        if reason is not None:
            x = rect.right() - 10 - REASON_TEXT.advance(reason)
            draw_text(painter, x, top, reason, REASON_TEXT, theme.color("text.tertiary"))
            return
        paint_switch_track(
            painter,
            QtCore.QPointF(rect.right() - 10 - 32, rect.y() + (ITEM_HEIGHT - 18) / 2),
            spec.key in self.visible,
            theme,
        )


# ---------- the Monitor map ----------


class InsightControls(MapControls):
    """The map's control column: Layers (pressed while its menu is open) over the reset button."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget) -> None:
        super().__init__(themes, parent)
        self.layers = MapButton(themes, "layers", "Map layers", self)
        self.layers.setAutoRepeat(False)
        self.layers.setGeometry(1, 1, 30, 30)
        self.layers.clip = rounded_path(QtCore.QRectF(0, 0, 30, 30), 5)
        self.fit.setGeometry(1, 31, 30, 30)
        self.fit.rule = True
        self.fit.clip = rounded_path(QtCore.QRectF(0, -30, 30, 60), 5)
        self.opened = False
        self.setFixedSize(32, 62)

    def set_opened(self, opened: bool) -> None:
        self.opened = opened
        self.layers.update()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self.opened:  # ``span.on``: the selected fill
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setClipPath(rounded_path(QtCore.QRectF(1, 1, 30, 60), 5))
            painter.fillRect(QtCore.QRectF(1, 1, 30, 30), self.color("surface.selected"))


CHIP_TEXT = TextStyle(12, 400, 16, tabular=True)


class InsightMap(LiveMap):
    """The Monitor's map: the layer switches, the goal history and analysis marks, hover with the candidate
    tooltip, click to pin (a click on the map elsewhere unpins), and the status chip in the box's bottom-right
    corner."""

    hovered = QtCore.Signal(object)  # candidate id or None
    pinned = QtCore.Signal(object)  # candidate id or None
    layers_clicked = QtCore.Signal()

    HIT_RADIUS = 9.0

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.controls.deleteLater()
        self.layer_controls = InsightControls(themes, self)
        self.controls = self.layer_controls
        self.layer_controls.fit.clicked.connect(self.fit_view)
        self.layer_controls.layers.clicked.connect(self.layers_clicked)
        self.layer_controls.hide()
        self.visible: frozenset[str] = DEFAULT_VISIBLE
        self.insight = InsightLayers()
        self.chip = ""
        self.chip_live = False
        self._press: QtCore.QPointF | None = None

    # data

    def set_visible(self, visible: frozenset[str]) -> None:
        if visible != self.visible:
            self.visible = visible
            self._base = None
            self.path_layers.invalidate()
            self.update()

    def set_chip(self, text: str, live: bool) -> None:
        if (text, live) != (self.chip, self.chip_live):
            self.chip, self.chip_live = text, live
            self.setToolTip(text)
            self.update()

    def shows(self, layer: str) -> bool:
        return layer in self.visible

    def decisions(self) -> Decisions | None:
        return self.insight.decisions

    # painting

    def paint_under(self, painter: QtGui.QPainter, placement: Placement) -> None:
        if "chunks" in self.visible and self.insight.chunks is not None:
            paint_chunks(painter, self.insight.chunks, placement, self.theme)

    def paint_marks(self, painter: QtGui.QPainter, placement: Placement) -> None:
        theme = self.theme
        if "goals" in self.visible:
            paint_goal_history(painter, self.insight.goals.values(), placement, theme)
        decisions = self.insight.decisions
        if decisions is None:
            return
        if "route" in self.visible:
            paint_route(painter, decisions, placement, theme)
        if "candidates" in self.visible or "pool" in self.visible:
            paint_candidates(painter, decisions, placement, theme, pool="pool" in self.visible)

    def paint_over(self, painter: QtGui.QPainter, placement: Placement) -> None:
        theme = self.theme
        decisions = self.insight.decisions
        bounds = self.visible_box() or QtCore.QRectF(self.rect())
        if decisions is not None:
            if "route" in self.visible:
                paint_route_badges(painter, decisions, placement, theme, bounds)
            for identity in (self.insight.pinned, self.insight.hovered):
                candidate = decisions.candidate(identity)
                if candidate is not None and self.marks_shown():
                    paint_selection(painter, placement.point(candidate.x, candidate.y), theme)
        self._paint_chip(painter, theme)
        hovered = decisions.candidate(self.insight.hovered) if decisions is not None else None
        if hovered is not None and decisions is not None and self.marks_shown():
            anchor = placement.point(hovered.x, hovered.y)
            paint_candidate_tip(painter, hovered, decisions, anchor, bounds, theme)

    def marks_shown(self) -> bool:
        return bool(self.visible & {"candidates", "pool"})

    def _paint_chip(self, painter: QtGui.QPainter, theme: Theme) -> None:
        """``.chip-float.stat``: 26 px, padding 0 10, ``map.float`` with a 1 px rule, radius 6, 12 px
        ``text.secondary``; a 7 px dot (``data.completed`` live, ``text.tertiary`` otherwise); 10 px inside
        the fitted box's bottom-right corner."""
        fit = self.fit_rect()
        visible = None if fit is None else fit.intersected(self.stage())
        if visible is None or visible.isEmpty() or not self.chip:
            return
        room = max(60.0, visible.width() - 160)
        text = CHIP_TEXT.elide(self.chip, room - 20 - 13)
        width = 10 + 7 + 6 + CHIP_TEXT.advance(text) + 10
        box = QtCore.QRectF(visible.right() - 10 - width, visible.bottom() - 10 - 26, width, 26)
        frame = QtCore.QRectF(snapped(box))
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
        painter.setBrush(theme.color("map.float"))
        painter.drawPath(rounded_path(frame.adjusted(0.5, 0.5, -0.5, -0.5), 5.5))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("data.completed" if self.chip_live else "text.tertiary"))
        painter.drawEllipse(QtCore.QRectF(frame.x() + 10, frame.center().y() - 3.5, 7, 7))
        painter.restore()
        draw_text(
            painter,
            frame.x() + 23,
            frame.y() + (26 - 16) / 2,
            text,
            CHIP_TEXT,
            theme.color("text.secondary"),
        )

    # input

    def candidate_at(self, position: QtCore.QPointF) -> int | None:
        decisions, placement = self.insight.decisions, self.placement()
        if decisions is None or placement is None or not self.marks_shown():
            return None
        best, best_distance = None, self.HIT_RADIUS
        for candidate in decisions.candidates:
            point = placement.point(candidate.x, candidate.y)
            distance = math.hypot(point.x() - position.x(), point.y() - position.y())
            if distance <= best_distance:
                best, best_distance = candidate.id, distance
        return best

    def set_hovered(self, identity: int | None) -> None:
        if identity != self.insight.hovered:
            self.insight.hovered = identity
            self.hovered.emit(identity)
            self.update()

    def set_pinned(self, identity: int | None) -> None:
        if identity != self.insight.pinned:
            self.insight.pin(identity)
            self.pinned.emit(self.insight.pinned)
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._press = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag is None:
            self.set_hovered(self.candidate_at(event.position()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        press, self._press = self._press, None
        super().mouseReleaseEvent(event)
        if (
            press is not None
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and math.hypot(event.position().x() - press.x(), event.position().y() - press.y()) < 4
        ):
            identity = self.candidate_at(event.position())
            self.set_pinned(None if identity == self.insight.pinned else identity)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.set_hovered(None)
        super().leaveEvent(event)

    def _update_cursor(self) -> None:
        if self.insight.hovered is not None:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            super()._update_cursor()


def disabled_layers(
    insight: InsightLayers, engine_insight: bool, explorer_insight: bool = True
) -> dict[str, str]:
    """Layers the run cannot show and why (the Layers menu's reasons); ``explorer_insight`` False: the run's
    explorer has no frontier insight (an external explorer)."""
    reason = ""
    if not engine_insight:
        reason = "Engine too old"
    elif not explorer_insight:
        reason = "No insight"
    elif insight.available is False:
        reason = "No analysis"
    disabled = {spec.key: reason for spec in LAYERS if spec.group == "analysis"} if reason else {}
    if not engine_insight:
        disabled["goals"] = "Engine too old"
    return disabled
