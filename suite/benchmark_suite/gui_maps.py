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

"""Map painting of the Main design (HANDOFF §7.11, ``mockups/maps.js``) for small fitted maps.

The world image in the theme's map colours, explored cells, the ground-truth path with its halo and start
mark, the Nav2 goal and plan, the robot and spawn arrows, inside hatched frames. Marks keep their screen size
at every scale, as the mockup's overlays do; the interactive maps (``gui_mapview.LiveMap`` and its subclasses)
use the same painters. Maps draw the data they are given: no smoothing, nothing joined across a segment start.
"""

import base64
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import struct
import time
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_tokens
from .gui_marks import paint_css_hatch
from .gui_theme import Theme, ThemeManager, css_color, font, snap
from .gui_widgets import BODY, CssBox, Themed, css_rect, draw_text, rounded_path
from .monitoring import IN_COLLISION, SEGMENT_START, PathCursor, PathSettings, accept_page

# Grey values of ``spawns.map`` (``monitoring.reference_pixels``); 170 elsewhere.
FREE, WALL = 245, 30


def decode_grey(image_png: str) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """A base64 PNG of a grey (or RGBA) layer as one uint8 channel, rows as the engine stores them."""
    image = QtGui.QImage.fromData(base64.b64decode(image_png))
    if image.isNull():
        raise ValueError("Unreadable map image")
    image = image.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
    rows = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )
    return np.array(rows[:, : image.width()], dtype=np.uint8)


@dataclass(frozen=True, slots=True, eq=False)
class WorldMap:
    """A world reference from ``spawns.map``; row 0 of ``pixels`` lies at the origin's y."""

    world_id: str
    pixels: np.ndarray[Any, np.dtype[np.uint8]]
    origin: tuple[float, float, float]
    resolution: float
    metadata_sha256: str | None
    _images: dict[tuple[str, bool], QtGui.QImage] = field(default_factory=dict, repr=False)

    @classmethod
    def from_reply(cls, reply: dict[str, Any]) -> "WorldMap":
        x, y, yaw = (float(value) for value in reply["origin"])
        resolution = float(reply["resolution"])
        if not all(math.isfinite(v) for v in (x, y, yaw, resolution)) or resolution <= 0:
            raise ValueError("Invalid map placement")
        return cls(
            str(reply["world_id"]),
            decode_grey(reply["image_png"]),
            (x, y, yaw),
            resolution,
            reply.get("metadata_sha256"),
        )

    @property
    def size(self) -> tuple[int, int]:
        rows, columns = self.pixels.shape
        return int(columns), int(rows)

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """World metres to continuous image coordinates (column, row from the origin's y)."""
        ox, oy, yaw = self.origin
        dx, dy = x - ox, y - oy
        c, s = math.cos(yaw), math.sin(yaw)
        return (c * dx + s * dy) / self.resolution, (-s * dx + c * dy) / self.resolution

    def image(self, theme: Theme, *, excluded: bool = False) -> QtGui.QImage:
        """Free and wall cells in the theme's map colours; elsewhere transparent (the frame shows)."""
        key = (theme.name, excluded)
        if key not in self._images:
            colors = gui_tokens.MAP[theme.name]
            free = css_color(colors["excludedFree" if excluded else "free"])
            wall = css_color(colors["excludedWall" if excluded else "wall"])
            rgba = np.zeros((*self.pixels.shape, 4), dtype=np.uint8)
            for value, color in ((FREE, free), (WALL, wall)):
                mask = self.pixels == value
                rgba[mask] = (color.red(), color.green(), color.blue(), color.alpha())
            self._images[key] = rgba_image(rgba)
        return self._images[key]


def rgba_image(rgba: np.ndarray[Any, np.dtype[np.uint8]]) -> QtGui.QImage:
    """An RGBA array (rows from the origin's y) as a QImage flipped to screen order (row 0 at the top)."""
    flipped = np.ascontiguousarray(rgba[::-1])
    height, width = flipped.shape[:2]
    image = QtGui.QImage(
        flipped.data, width, height, width * 4, QtGui.QImage.Format.Format_RGBA8888
    )
    return image.copy()


def explored_image(pixels: np.ndarray[Any, np.dtype[np.uint8]], theme: Theme) -> QtGui.QImage:
    """Explored cells (255) in ``data.coverage`` at 22 % (light) or 24 % (dark); the rest transparent."""
    color = theme.color("data.coverage")
    rgba = np.zeros((*pixels.shape, 4), dtype=np.uint8)
    rgba[pixels == 255] = (
        color.red(),
        color.green(),
        color.blue(),
        round(255 * gui_tokens.EXPLORED_OPACITY[theme.name]),
    )
    return rgba_image(rgba)


@dataclass(frozen=True, slots=True)
class Placement:
    """Where a world image sits in a frame (``fill``: stretched over it; ``fitted`` centres and sizes the frame)
    and how world points map onto it."""

    world: WorldMap
    rect: QtCore.QRectF  # the drawn image in widget coordinates

    @classmethod
    def fill(cls, world: WorldMap, frame: QtCore.QRectF) -> "Placement":
        """Stretched over the frame, as CSS draws an image at 100 % × 100 % with its SVG overlay
        (``preserveAspectRatio="none"``) in a box sized from the image's rounded aspect."""
        return cls(world, QtCore.QRectF(frame))

    @property
    def scale(self) -> float:
        return self.rect.width() / self.world.size[0]

    @property
    def scale_y(self) -> float:
        return self.rect.height() / self.world.size[1]

    def point(self, x: float, y: float) -> QtCore.QPointF:
        column, row = self.world.to_pixel(x, y)
        return QtCore.QPointF(
            self.rect.x() + column * self.scale, self.rect.bottom() - row * self.scale_y
        )

    def heading(self, yaw: float) -> float:
        """Screen angle in degrees (clockwise, y down) of a world heading in radians (counter-clockwise)."""
        return -math.degrees(yaw - self.world.origin[2])


def fitted(world: WorldMap, frame: QtCore.QRectF, pad: float) -> Placement:
    """``maps.js`` ``data-fit="contain"``: the image scaled into the frame less ``pad`` on every side, its box
    floored to whole pixels and centred in the frame (``place-items: center``)."""
    width, height = world.size
    scale = min((frame.width() - 2 * pad) / width, (frame.height() - 2 * pad) / height)
    box_width, box_height = math.floor(width * scale), math.floor(height * scale)
    left = frame.x() + (frame.width() - box_width) / 2
    top = frame.y() + (frame.height() - box_height) / 2
    return Placement.fill(world, QtCore.QRectF(left, top, box_width, box_height))


# ---------- painters ----------


def paint_hatch(
    painter: QtGui.QPainter, rect: QtCore.QRectF, theme: Theme, *, gap: float = 5, line: float = 1
) -> None:
    """``map.outside`` with a 135° hatch of ``map.hatch`` lines (CSS ``transparent 0 gap, hatch gap
    gap+line``), anchored at the frame's top-left corner and evaluated per pixel like a CSS gradient."""
    painter.fillRect(rect, theme.color("map.outside"))
    box = rect.toAlignedRect()
    paint_css_hatch(painter, box, QtGui.QColor(0, 0, 0, 0), theme.color("map.hatch"), gap, line)


def round_pen(color: QtGui.QColor, width: float) -> QtGui.QPen:
    pen = QtGui.QPen(color, width)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    return pen


# ``maps.js``: the ground-truth path, a 2.2 px line over a 4.5 px ``map.halo`` stroke at 90 % opacity.
HALO_WIDTH, PATH_WIDTH, HALO_OPACITY = 4.5, 2.2, 0.9


def paint_start(painter: QtGui.QPainter, point: QtCore.QPointF, theme: Theme) -> None:
    """Open start mark: r 3.5, ``map.halo`` fill, 1.8 px ``data.path`` stroke."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(theme.color("data.path"), 1.8))
    painter.setBrush(theme.color("map.halo"))
    painter.drawEllipse(point, 3.5, 3.5)
    painter.restore()


def paint_goal(painter: QtGui.QPainter, point: QtCore.QPointF, theme: Theme) -> None:
    """Nav2 goal: a 10 px × in ink (2 px) over a 4.5 px ``map.halo`` stroke, round caps."""
    d = 5.0
    path = QtGui.QPainterPath()
    path.moveTo(point.x() - d, point.y() - d)
    path.lineTo(point.x() + d, point.y() + d)
    path.moveTo(point.x() + d, point.y() - d)
    path.lineTo(point.x() - d, point.y() + d)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    for color, width in ((theme.color("map.halo"), 4.5), (theme.color("text.ink"), 2.0)):
        pen = QtGui.QPen(color, width)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPath(path)
    painter.restore()


def paint_plan(painter: QtGui.QPainter, points: Sequence[QtCore.QPointF], theme: Theme) -> None:
    """Nav2 plan: ink 1.5 px, dashed 4/3 px."""
    if len(points) < 2:
        return
    pen = QtGui.QPen(theme.color("text.ink"), 1.5)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)  # SVG butt caps (square caps lengthen dashes)
    pen.setDashPattern([4 / 1.5, 3 / 1.5])  # Qt dash lengths are in pen widths
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(pen)
    painter.drawPolyline(QtGui.QPolygonF(list(points)))
    painter.restore()


def arrow_polygon(point: QtCore.QPointF, degrees: float, size: float) -> QtGui.QPolygonF:
    """``maps.js`` arrow: [[s, 0], [−0.7s, −0.62s], [−0.3s, 0], [−0.7s, 0.62s]] turned by ``degrees``
    (screen, clockwise)."""
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    corners = (
        (size, 0.0),
        (-0.7 * size, -0.62 * size),
        (-0.3 * size, 0.0),
        (-0.7 * size, 0.62 * size),
    )
    return QtGui.QPolygonF(
        [QtCore.QPointF(point.x() + x * c - y * s, point.y() + x * s + y * c) for x, y in corners]
    )


def paint_robot(
    painter: QtGui.QPainter, point: QtCore.QPointF, degrees: float, theme: Theme, size: float = 7
) -> None:
    """Robot: a halo circle (r 1.35·size, fixed orange tints) and the ``data.path`` arrow with a white
    1.6 px outline."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(css_color(gui_tokens.ROBOT_HALO_STROKE), 1))
    painter.setBrush(css_color(gui_tokens.ROBOT_HALO))
    painter.drawEllipse(point, 1.35 * size, 1.35 * size)
    outline = QtGui.QPen(css_color(gui_tokens.ROBOT_ARROW_OUTLINE), 1.6)
    outline.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(outline)
    painter.setBrush(theme.color("data.path"))
    painter.drawPolygon(arrow_polygon(point, degrees, size))
    painter.restore()


def paint_footprint(
    painter: QtGui.QPainter,
    placement: "Placement",
    pose: tuple[float, float, float],
    footprint: Sequence[tuple[float, float]],
    theme: Theme,
    size: float,
) -> None:
    """The robot to scale as the round robot mark: a disc at the footprint's centre with half its longer side
    as radius (at least 4 px), in the mark's orange halo, and the robot arrow inside it (at most ``size``; left
    out while the disc is too small to hold a 3.5 px arrow)."""
    x, y, yaw = pose
    xs, ys = [px for px, _ in footprint], [py for _, py in footprint]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2  # footprint centre, robot frame
    c, s = math.cos(yaw), math.sin(yaw)
    centre = placement.point(x + cx * c - cy * s, y + cx * s + cy * c)
    origin = placement.point(x, y)
    unit = placement.point(x + c, y + s)  # one metre ahead, for the screen scale
    scale = math.hypot(unit.x() - origin.x(), unit.y() - origin.y())
    radius = max(4.0, max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * scale)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(css_color(gui_tokens.ROBOT_HALO_STROKE), 1))
    painter.setBrush(css_color(gui_tokens.ROBOT_HALO))
    painter.drawEllipse(centre, radius, radius)
    arrow = min(size, 0.85 * radius)
    if arrow >= 3.5:
        outline = QtGui.QPen(css_color(gui_tokens.ROBOT_ARROW_OUTLINE), 1.6)
        outline.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(outline)
        painter.setBrush(theme.color("data.path"))
        painter.drawPolygon(arrow_polygon(centre, placement.heading(yaw), arrow))
    painter.restore()


def paint_spawn(
    painter: QtGui.QPainter,
    point: QtCore.QPointF,
    degrees: float,
    theme: Theme,
    *,
    picked: bool = False,
    selected: bool = False,
) -> None:
    """Spawn arrow in the robot colour (``data.path``): 6 px (7.5 when selected), ``map.free`` fill (solid when
    picked or selected), 1.4 px outline; a selected pose also gets a dashed circle (r 11, 1 px, 3/2 dashes, the
    colour at 8 %)."""
    ink = theme.color("data.path")
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    if selected:
        fill = QtGui.QColor(ink)
        fill.setAlphaF(0.08)
        dashed = QtGui.QPen(ink, 1)
        dashed.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)  # SVG butt caps
        dashed.setDashPattern([3, 2])
        painter.setPen(dashed)
        painter.setBrush(fill)
        painter.drawEllipse(point, 11, 11)
    outline = QtGui.QPen(ink, 1.4)
    outline.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(outline)
    painter.setBrush(ink if picked or selected else theme.map_color("free"))
    painter.drawPolygon(arrow_polygon(point, degrees, 7.5 if selected else 6))
    painter.restore()


# ``maps.js``: a label starts 11 px right of its pose with its baseline 11 px above it.
LABEL_GAP = 11.0


def paint_spawn_label(
    painter: QtGui.QPainter,
    placement: Placement,
    point: QtCore.QPointF,
    name: str,
    theme: Theme,
    *,
    size: float,
    bold: bool,
) -> None:
    """A pose label as ``maps.js`` draws it: ink text (600 for chosen or inspected poses, else 500) over a
    3 px ``map.halo`` stroke; it ends left of the pose when its estimated width would pass the map's right
    edge, and sits below the pose near the top edge (within 18 px, or 1.9 m of the map's top)."""
    rect = placement.rect
    x, y = point.x() - rect.x(), point.y() - rect.y()
    flip_left = x + 9 + len(name) * size * 0.555 + 4 > rect.width()
    below = y - 18 < 0 or y / placement.scale_y < 1.9 / placement.world.resolution
    label_font = font(size, 600 if bold else 500)
    advance = QtGui.QFontMetricsF(label_font).horizontalAdvance(name)
    left = point.x() - LABEL_GAP - advance if flip_left else point.x() + LABEL_GAP
    baseline = point.y() + LABEL_GAP + 9 if below else point.y() - LABEL_GAP
    path = QtGui.QPainterPath()
    path.addText(QtCore.QPointF(left, baseline), label_font, name)
    halo = QtGui.QPen(theme.color("map.halo"), 3)
    halo.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)  # SVG's default join
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.strokePath(path, halo)  # ``paint-order: stroke``
    painter.fillPath(path, theme.color("text.ink"))
    painter.restore()


# ---------- live runs: the Monitor's map and the Campaigns tiles ----------


class LivePath:
    """A run's ground-truth path as the GUI keeps it: world metres and segment starts, oldest first, at most the
    worker's history (``PathSettings.capacity``). Past it the oldest vertices go in blocks of ``TRIM``
    (``dropped``) and the first kept vertex starts a segment. ``missing`` counts vertices the worker evicted
    before they were read (the cursor's). ``generation`` changes whenever the kept path stops being an extension
    of what was drawn (``clear`` or a trim)."""

    TRIM = 4096
    CAPACITY = PathSettings().capacity

    def __init__(self) -> None:
        self.x: list[float] = []
        self.y: list[float] = []
        self.starts: list[bool] = []
        self.markers: deque[tuple[float, float]] = deque(maxlen=2000)  # samples in collision cells
        self.dropped = 0
        self.missing = 0
        self.generation = 0

    def __len__(self) -> int:
        return len(self.x)

    @property
    def start_known(self) -> bool:
        """Whether the first kept vertex is the path's first (the start mark is drawn only then)."""
        return self.dropped == 0 and self.missing == 0

    def clear(self) -> None:
        self.x, self.y, self.starts = [], [], []
        self.markers.clear()
        self.dropped = self.missing = 0
        self.generation += 1

    def extend(self, vertices: Iterable[tuple[float, float, int]]) -> None:
        for x, y, flags in vertices:
            self.x.append(x)
            self.y.append(y)
            self.starts.append(bool(flags & SEGMENT_START) or not self.starts)
            if flags & IN_COLLISION:
                self.markers.append((x, y))
        excess = len(self.x) - self.CAPACITY
        if excess > 0:
            drop = self.TRIM * math.ceil(excess / self.TRIM)
            del self.x[:drop], self.y[:drop], self.starts[:drop]
            self.starts[0] = True
            self.dropped += drop
            self.generation += 1


@dataclass(slots=True)
class LiveLayers:
    """What a live map shows of one run (world metres)."""

    path: LivePath = field(default_factory=LivePath)
    explored: np.ndarray[Any, np.dtype[np.uint8]] | None = None
    pose: tuple[float, float, float] | None = None
    goal: tuple[float, float] | None = None
    plan: list[tuple[float, float]] = field(default_factory=list)
    tail: tuple[float, float] | None = None
    spawn: tuple[float, float, float] | None = (
        None  # tiles: the accepted spawn, while there is no path
    )
    # The robot's footprint (robot frame, metres; the catalog's): drawn to scale at the pose. Empty: the
    # fixed robot mark.
    footprint: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ReplyFacts:
    """What a ``monitor.get`` reply says beyond the layers, for a status line."""

    status: str  # available, stale or unavailable
    reason: str
    pose_age: float | None  # wall seconds
    breaks: Mapping[str, int]
    collisions: int
    outside: int
    mismatch: bool  # the world map is not the attempt's reference: explored cells hidden
    notes: tuple[str, ...]


LAYER_TITLES = {"explored": "Explored cells", "navigation": "Nav2 goal and plan"}


def apply_monitor_reply(
    layers: LiveLayers,
    cursor: PathCursor,
    revisions: dict[str, int],
    reply: Mapping[str, Any],
    world: WorldMap | None,
) -> ReplyFacts:
    """Apply a ``monitor.get`` reply (one with ``history``): the path page through ``accept_page`` (a new history
    clears the path), the pose while it is available, the tail while the last page is applied, and each layer
    whose revision changed (−1 clears it; ``revisions`` follows). A world map whose hash is not the attempt's
    reference hides the explored cells. A malformed reply raises KeyError, TypeError or ValueError before anything
    changes; a malformed layer adds a note."""
    status, reason, pose = str(reply["status"]), str(reply["reason"]), reply["pose"]
    values = (
        None
        if pose is None
        else (
            float(pose["x"]),
            float(pose["y"]),
            float(pose["yaw"]),
            float(pose["age_wall_seconds"]),
        )
    )
    reference, replied = reply["reference"], dict(reply["layers"])
    breaks = {str(key): int(count) for key, count in reply["breaks"].items()}
    collisions, outside = int(reply["collision_samples"]), int(reply["outside_samples"])
    tail, last = reply["tail"], reply["last_sequence"]
    tail_point = (float(tail[0]), float(tail[1])) if tail else None
    reset, vertices = accept_page(cursor, reply)
    if reset:
        layers.path.clear()
    layers.path.extend(vertices)
    layers.path.missing = cursor.dropped
    layers.tail = tail_point if last == cursor.applied else None
    layers.pose = values[:3] if status == "available" and values is not None else None
    mismatch = bool(
        world is not None
        and world.metadata_sha256
        and isinstance(reference, dict)
        and reference.get("metadata_sha256") != world.metadata_sha256
    )
    if mismatch:
        layers.explored = None
        revisions.pop("explored", None)
    notes = []
    for name, layer in replied.items():
        title = LAYER_TITLES.get(name)
        if title is None or (name == "explored" and mismatch):
            continue
        try:
            if layer["revision"] == -1:
                _set_layer(layers, name, None)
                revisions.pop(name, None)
                if layer.get("reason"):
                    notes.append(f"{title} unavailable: {layer['reason']}")
            elif layer["data"] is not None:
                _set_layer(layers, name, layer["data"])
                revisions[name] = int(layer["revision"])
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"{title} rejected: {exc}")
    age = values[3] if values is not None else None
    return ReplyFacts(status, reason, age, breaks, collisions, outside, mismatch, tuple(notes))


def _set_layer(layers: LiveLayers, name: str, data: Mapping[str, Any] | None) -> None:
    if name == "explored":
        layers.explored = None if data is None else decode_grey(data["image_png"])
        return
    goal = data.get("goal") if data is not None else None
    plan = (data.get("plan") or []) if data is not None else []
    layers.goal = (float(goal[0]), float(goal[1])) if goal else None
    layers.plan = [(float(point[0]), float(point[1])) for point in plan]


_ELEMENTS = np.dtype([("kind", ">i4"), ("x", ">f8"), ("y", ">f8")])


def polyline_path(
    xs: np.ndarray[Any, np.dtype[np.float64]],
    ys: np.ndarray[Any, np.dtype[np.float64]],
    joined: np.ndarray[Any, np.dtype[np.bool_]],
) -> QtGui.QPainterPath:
    """A path through the points; point i joins point i − 1 where ``joined[i]``, else it starts a subpath. Built
    from Qt's serialised path (big-endian: the element count; each element's kind, 0 move or 1 line, x and y;
    then the fill data), as pyqtgraph's ``arrayToQPath`` does, so long paths need no loop per point."""
    elements = np.empty(len(xs), dtype=_ELEMENTS)
    elements["kind"] = joined
    elements["kind"][:1] = 0
    elements["x"], elements["y"] = xs, ys
    data = struct.pack(">i", len(xs)) + elements.tobytes() + struct.pack(">ii", 0, 0)
    path = QtGui.QPainterPath()
    stream = QtCore.QDataStream(QtCore.QByteArray(data))
    stream >> path
    return path


def _layer_image(size: QtCore.QSize, ratio: float) -> QtGui.QImage:
    image = QtGui.QImage(
        max(1, round(size.width() * ratio)),
        max(1, round(size.height() * ratio)),
        QtGui.QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.setDevicePixelRatio(ratio)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    return image


class PathLayers:
    """A live map's path in two device-pixel images for one view: every halo (opaque, shown at 90 % as one
    shape, as the SVG stroke) and every line. Each ``draw`` adds the vertices kept since the last one, then goes
    on drawing older vertices, newest first, for at most ``BUDGET_S`` seconds; the map paints again until all
    are drawn (a long path in a new view shows the part near the robot first). Pieces of ``PIECE`` vertices share
    their end vertex (round caps and joins make the same shape; the layers are opaque, so order does not matter),
    pieces outside the images are skipped, segment starts are never joined and nothing is smoothed."""

    PIECE = 128  # vertices per stroke: Qt strokes long, overlapping paths far more slowly
    BLOCK = 8192  # vertices mapped to the screen at once
    BUDGET_S = 0.025

    def __init__(self) -> None:
        self.source: LivePath | None = None
        self.key: tuple[object, ...] | None = None
        self.halo = self.line = QtGui.QImage()
        self.low = self.high = 0  # the vertices low to high (exclusive) are drawn

    def invalidate(self) -> None:
        self.key = None

    def draw(
        self, path: LivePath, placement: Placement, size: QtCore.QSize, ratio: float, theme: Theme
    ) -> bool:
        """Draw what is new, then older vertices for at most ``BUDGET_S``; True when all are drawn."""
        rect = placement.rect
        key = (
            (rect.x(), rect.y(), rect.width(), rect.height()),
            (size.width(), size.height()),
            ratio,
            theme.name,
            path.generation,
        )
        count = len(path)
        if path is not self.source or key != self.key:
            self.source, self.key = path, key
            self.halo, self.line = _layer_image(size, ratio), _layer_image(size, ratio)
            self.low = self.high = 0
        if count == 0 or (self.low == 0 and self.high == count):
            return True
        painters = [QtGui.QPainter(self.halo), QtGui.QPainter(self.line)]
        pens = (
            round_pen(theme.color("map.halo"), HALO_WIDTH),
            round_pen(theme.color("data.path"), PATH_WIDTH),
        )
        for painter, pen in zip(painters, pens, strict=True):
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(pen)
        bounds = QtCore.QRectF(0, 0, size.width(), size.height()).adjusted(-4, -4, 4, 4)
        deadline = time.perf_counter() + self.BUDGET_S
        if self.low == self.high:  # nothing drawn yet: start at the newest vertex
            self.low = self.high = count
            if count == 1:  # a lone vertex is a dot of the line's width, as SVG draws it
                point = placement.point(path.x[0], path.y[0])
                for painter in painters:
                    painter.drawPoint(point)
                self.low = 0
        elif self.high < count:  # new vertices, joined to the drawn path at its last vertex
            self._stroke(painters, path, placement, self.high - 1, count, bounds, None)
            self.high = count
        while self.low > 0 and time.perf_counter() <= deadline:
            first = max(0, self.low - self.BLOCK)
            end = min(count, self.low + 1)  # shares the oldest drawn vertex
            self.low = self._stroke(painters, path, placement, first, end, bounds, deadline)
        for painter in painters:
            painter.end()
        return self.low == 0 and self.high == count

    def _stroke(
        self,
        painters: Sequence[QtGui.QPainter],
        path: LivePath,
        placement: Placement,
        first: int,
        end: int,
        bounds: QtCore.QRectF,
        deadline: float | None,
    ) -> int:
        """Stroke the vertices ``first`` to ``end`` (exclusive), newest piece first; returns the oldest vertex
        drawn when the deadline passed (else ``first``)."""
        xs, ys = screen_points(placement, path, first, end)
        joined = ~np.asarray(path.starts[first:end], dtype=bool)
        stop = len(xs)
        while stop > 1:
            start = max(0, stop - self.PIECE - 1)
            px, py = xs[start:stop], ys[start:stop]
            if (
                px.max() >= bounds.left()
                and px.min() <= bounds.right()
                and py.max() >= bounds.top()
                and py.min() <= bounds.bottom()
            ):
                stroke = polyline_path(px, py, joined[start:stop])
                for painter in painters:
                    painter.drawPath(stroke)
            if start == 0:
                break
            stop = start + 1
            if deadline is not None and time.perf_counter() > deadline:
                return first + start
        return first

    def paint(
        self, painter: QtGui.QPainter, layers: LiveLayers, placement: Placement, theme: Theme
    ) -> None:
        """Composite the drawn path: the halos at 90 % (the tail's halo joined to them), the lines, the tail's
        line (last vertex to the pose) and the start mark (only while the path's start is kept)."""
        path = layers.path
        tail: QtGui.QPainterPath | None = None
        if layers.tail is not None and len(path):
            tail = QtGui.QPainterPath(placement.point(path.x[-1], path.y[-1]))
            tail.lineTo(placement.point(*layers.tail))
        saved: tuple[QtCore.QPoint, QtGui.QImage] | None = None
        if tail is not None:
            # Drawn into the halo image and taken out again, only around the tail.
            ratio = self.halo.devicePixelRatio()
            area = tail.boundingRect().adjusted(-HALO_WIDTH, -HALO_WIDTH, HALO_WIDTH, HALO_WIDTH)
            region = QtCore.QRect(
                math.floor(area.left() * ratio),
                math.floor(area.top() * ratio),
                math.ceil(area.width() * ratio) + 2,
                math.ceil(area.height() * ratio) + 2,
            ).intersected(self.halo.rect())
            saved = (region.topLeft(), self.halo.copy(region))
            layer = QtGui.QPainter(self.halo)
            layer.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            layer.setPen(round_pen(theme.color("map.halo"), HALO_WIDTH))
            layer.drawPath(tail)
            layer.end()
        painter.save()
        painter.setOpacity(HALO_OPACITY)
        painter.drawImage(QtCore.QPointF(0, 0), self.halo)
        painter.restore()
        if saved is not None:
            corner, image = saved
            layer = QtGui.QPainter(self.halo)
            layer.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
            ratio = self.halo.devicePixelRatio()
            layer.drawImage(QtCore.QPointF(corner.x() / ratio, corner.y() / ratio), image)
            layer.end()
        painter.drawImage(QtCore.QPointF(0, 0), self.line)
        if tail is not None:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(round_pen(theme.color("data.path"), PATH_WIDTH))
            painter.drawPath(tail)
            painter.restore()
        if len(path) and path.start_known:
            paint_start(painter, placement.point(path.x[0], path.y[0]), theme)


def screen_points(
    placement: Placement, path: LivePath, first: int, last: int
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Widget coordinates of the kept vertices ``first`` to ``last`` (exclusive)."""
    world = placement.world
    ox, oy, yaw = world.origin
    cos, sin = math.cos(yaw), math.sin(yaw)
    xs = np.asarray(path.x[first:last], dtype=np.float64) - ox
    ys = np.asarray(path.y[first:last], dtype=np.float64) - oy
    column = (cos * xs + sin * ys) / world.resolution
    row = (-sin * xs + cos * ys) / world.resolution
    rect = placement.rect
    return rect.x() + column * placement.scale, rect.bottom() - row * placement.scale_y


def paint_markers(
    painter: QtGui.QPainter,
    markers: Iterable[tuple[float, float]],
    placement: Placement,
    theme: Theme,
) -> None:
    """Samples in reference collision cells: a 7 px × in ``state.bad`` over a ``map.halo`` stroke (no design
    mark exists; the reference disagreements were marked before the redesign too)."""
    path = QtGui.QPainterPath()
    for x, y in markers:
        point = placement.point(x, y)
        path.moveTo(point.x() - 3.5, point.y() - 3.5)
        path.lineTo(point.x() + 3.5, point.y() + 3.5)
        path.moveTo(point.x() + 3.5, point.y() - 3.5)
        path.lineTo(point.x() - 3.5, point.y() + 3.5)
    if path.isEmpty():
        return
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    for color, width in ((theme.color("map.halo"), 4.0), (theme.color("state.bad"), 1.6)):
        pen = QtGui.QPen(color, width)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPath(path)
    painter.restore()


def paint_live(
    painter: QtGui.QPainter,
    layers: LiveLayers,
    placement: Placement,
    theme: Theme,
    path_layers: PathLayers,
    robot_size: float,
    *,
    path: bool = True,
    navigation: bool = True,
    marks: Callable[[], None] | None = None,
) -> None:
    """A live run's marks in ``maps.js`` order: the plan, the path (``path_layers`` already drawn), reference
    disagreements, ``marks`` (the Monitor's goal history and analysis marks), the goal and the robot.
    ``path`` and ``navigation`` are the Monitor's layer switches."""
    if navigation and layers.plan and layers.pose is not None:
        points = [placement.point(layers.pose[0], layers.pose[1])]
        points += [placement.point(x, y) for x, y in layers.plan[1:]]
        paint_plan(painter, points, theme)
    if path and len(layers.path):
        path_layers.paint(painter, layers, placement, theme)
    if path:
        paint_markers(painter, layers.path.markers, placement, theme)
    if marks is not None:
        marks()
    if navigation and layers.goal is not None:
        paint_goal(painter, placement.point(*layers.goal), theme)
    if layers.pose is not None and layers.footprint:
        paint_footprint(painter, placement, layers.pose, layers.footprint, theme, robot_size)
    elif layers.pose is not None:
        x, y, yaw = layers.pose
        paint_robot(painter, placement.point(x, y), placement.heading(yaw), theme, robot_size)


class MiniMap(Themed):
    """A live map filling a hatched frame with a 1 px ``border.strong`` ring (radius 2), as the worker
    tiles of Campaigns draw it (CSS ``.thumb .map``: hatch 4/5 px; the box takes the image's aspect and the
    image is drawn ``image-rendering: pixelated``). The widget is the map box plus the ring. The path is drawn
    by ``PathLayers``, as on the Monitor."""

    def __init__(
        self,
        themes: ThemeManager,
        *,
        hatch_gap: float = 4,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.hatch_gap = hatch_gap
        self.world: WorldMap | None = None
        self.layers = LiveLayers()
        self.path_layers = PathLayers()
        # The explored image with the array it was made from (compared by identity, kept alive here).
        self._explored: tuple[np.ndarray[Any, np.dtype[np.uint8]], str, QtGui.QImage] | None = None

    def restyle(self, *_: object) -> None:
        self._explored = None
        self.update()

    def set_world(self, world: WorldMap | None) -> None:
        if world is not self.world:
            self.world = world
            self.path_layers.invalidate()
            self.update()

    def set_layers(self, layers: LiveLayers) -> None:
        """Show ``layers`` (also after they changed in place)."""
        self.layers = layers
        self.update()

    def frame(self) -> QtCore.QRectF:
        """The map box inside the 1 px ring."""
        return QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        frame = self.frame()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.strong"), 1))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(rounded_path(frame.adjusted(-0.5, -0.5, 0.5, 0.5), 2.5))
        painter.save()
        painter.setClipPath(rounded_path(frame, 2))
        paint_hatch(painter, frame, theme, gap=self.hatch_gap)
        world = self.world
        if world is None:
            painter.restore()
            return
        placement = Placement.fill(world, frame)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)  # pixelated
        painter.drawImage(placement.rect, world.image(theme))
        layers = self.layers
        explored = layers.explored
        if explored is not None and explored.shape == world.pixels.shape:
            if (
                self._explored is None
                or self._explored[0] is not explored
                or self._explored[1] != theme.name
            ):
                self._explored = (explored, theme.name, explored_image(explored, theme))
            painter.drawImage(placement.rect, self._explored[2])
        complete = True
        if len(layers.path):
            complete = self.path_layers.draw(
                layers.path, placement, self.size(), self.devicePixelRatioF(), theme
            )
        paint_live(painter, layers, placement, theme, self.path_layers, 7)
        if layers.pose is None and not len(layers.path) and layers.spawn is not None:
            x, y, yaw = layers.spawn
            paint_spawn(painter, placement.point(x, y), placement.heading(yaw), theme, picked=True)
        painter.restore()
        if not complete:
            QtCore.QTimer.singleShot(0, self.update)  # a long path goes on at the next paint


# ---------- a fitted map with starting poses ----------


@dataclass(frozen=True, slots=True)
class SpawnMark:
    """A starting pose on a ``MapStage``: chosen poses (``picked``) are solid arrows with labels in 600, the
    inspected one (``selected``) is larger with a dashed circle."""

    name: str
    x: float
    y: float
    yaw: float
    picked: bool = False
    selected: bool = False
    labelled: bool = False


class MapStage(Themed, CssBox):
    """A hatched frame with a world fitted inside (``.wc .stage``, ``.sp-map``): ``map.outside`` with the 135°
    hatch, optionally radius 2 with an inset 1 px ``border.line`` ring; the world image ``pad`` px inside
    (``fitted``), drawn pixelated (excluded worlds in the excluded colours); then each pose's arrow and label.
    Mouse events pass through to the card under it."""

    def __init__(
        self,
        themes: ThemeManager,
        *,
        pad: float = 14,
        ring: bool = True,
        label_size: float = 11,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.pad, self.ring, self.label_size = pad, ring, label_size
        self.world: WorldMap | None = None
        self.excluded = False
        self.marks: tuple[SpawnMark, ...] = ()
        self.message = "Loading map…"
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_world(self, world: WorldMap | None) -> None:
        if world is not self.world:
            self.world = world
            self.update()

    def set_excluded(self, excluded: bool) -> None:
        if excluded != self.excluded:
            self.excluded = excluded
            self.update()

    def set_marks(self, marks: Sequence[SpawnMark]) -> None:
        marks = tuple(marks)
        if marks != self.marks:
            self.marks = marks
            self.update()

    def set_message(self, text: str) -> None:
        if text != self.message:
            self.message = text
            self.update()

    def placement(self) -> Placement | None:
        return None if self.world is None else fitted(self.world, css_rect(self), self.pad)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        frame = css_rect(self)
        left, top = snap(frame.left()), snap(frame.top())
        box = QtCore.QRectF(left, top, snap(frame.right()) - left, snap(frame.bottom()) - top)
        painter.save()
        if self.ring:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setClipPath(rounded_path(box, 2))
        paint_hatch(painter, box, theme)
        if self.ring:  # ``box-shadow: inset 0 0 0 1px`` following the 2 px corners
            ring = rounded_path(box, 2).subtracted(rounded_path(box.adjusted(1, 1, -1, -1), 1))
            painter.fillPath(ring, theme.color("border.line"))
        painter.restore()
        placement = self.placement()
        if placement is None or self.world is None:
            if self.message:
                style = BODY
                text = style.elide(self.message, max(0.0, box.width() - 24))
                x = box.x() + (box.width() - style.advance(text)) / 2
                y = box.y() + (box.height() - style.line) / 2
                draw_text(painter, x, y, text, style, theme.color("text.secondary"))
            return
        # Pixelated, into the image box snapped to whole pixels as Chrome draws it; the marks keep the fraction.
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        image_box = placement.rect
        left, top = snap(image_box.left()), snap(image_box.top())
        snapped = QtCore.QRect(
            left, top, snap(image_box.right()) - left, snap(image_box.bottom()) - top
        )
        painter.drawImage(snapped, self.world.image(theme, excluded=self.excluded))
        for mark in self.marks:
            point = placement.point(mark.x, mark.y)
            degrees = placement.heading(mark.yaw)
            paint_spawn(painter, point, degrees, theme, picked=mark.picked, selected=mark.selected)
            if mark.labelled:
                bold = mark.picked or mark.selected
                size = self.label_size
                paint_spawn_label(painter, placement, point, mark.name, theme, size=size, bold=bold)
