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

"""Worlds & Spawns (15): a world's starting positions, the pose form and the map.

D1: a saved pose (or the world's Default) is a template; every save creates a new spawn named "Spawn N" unless a
name is typed, Revert returns the form to the template, and the positions card's menu imports and exports spawn
YAML. The validity line checks the footprint against the ``spawns.map`` image (inside the world, clear of
walls); the engine stays authoritative when it saves.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .contracts import next_spawn_name
from .gui_cells import PaintedColumn, PaintedTable
from .gui_controls import KeySelect
from .gui_maps import FREE, WALL, Placement, WorldMap, paint_spawn, paint_spawn_label
from .gui_mapview import LiveMap, MapControls
from .gui_dialogs import Consequences, ListRow, Run
from .gui_menus import Menu
from .gui_style import fixed
from .gui_support import Transport, stale_engine_message
from .gui_theme import ThemeManager, font, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    BODY_TABULAR,
    CARD_TITLE,
    LABEL,
    PAGE_TITLE,
    Button,
    Card,
    CardHead,
    CardSplit,
    CssBox,
    FlexRow,
    ModalHost,
    PopupHost,
    Rule,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    place,
    shadow_image,
    shadow_margin,
)

SPAWN_DELETION_FOOTNOTE = (
    "Submitted campaigns, their results and reports keep the exact poses they recorded. "
    "Campaign YAML files that name its ID will no longer validate."
)
NAME_SCAN_PAGES = 40  # bound on the spawn-name scan per pass (250 names per page)
LIST_PAGE, LIST_LIMIT = 250, 2000
WORLD_TAG = TextStyle(11.5, 400, 18)
VALIDITY = TextStyle(12, 400, 16)
HINT = TextStyle(12.5, 400, 18)
TICK = TextStyle(11, 400, 12, tabular=True)
CURSOR = TextStyle(12, 400, 12, mono=True)
CURSOR_VALUE = TextStyle(12, 500, 12, mono=True)
HANDLE_LENGTH, HANDLE_RADIUS = 34.0, 5.0
DEFAULT_KEY = "default"


@dataclass(frozen=True, slots=True)
class Pose:
    """A starting position in the list: the world's Default or a saved spawn."""

    key: str  # DEFAULT_KEY or the saved spawn's ID
    name: str
    x: float
    y: float
    yaw: float  # radians, counter-clockwise from east
    family: str = ""  # the saved spawn's stable key (``family_id``); empty for the Default

    @property
    def default(self) -> bool:
        return self.key == DEFAULT_KEY


def heading_degrees(yaw: float) -> float:
    """A heading in degrees within (−180, 180]."""
    degrees = math.remainder(math.degrees(yaw), 360.0)
    return 180.0 if degrees <= -180.0 else degrees


def coordinate_text(value: float, digits: int) -> str:
    return fixed(value, digits) or "—"


def footprint_problem(
    world: WorldMap, footprint: Sequence[Sequence[float]], x: float, y: float, yaw: float
) -> str:
    """Why the footprint at (x, y, yaw) does not fit the map ("" when it fits): the cells it covers must lie
    inside the world and none may be a wall. The engine's check (``validate_spawn``) uses the same overlap
    test on the reference and decides on save."""
    try:
        points = np.asarray(footprint, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            return "The robot footprint is unavailable"
    except (TypeError, ValueError):
        return "The robot footprint is unavailable"
    c, s = math.cos(yaw), math.sin(yaw)
    points = points @ np.array([[c, s], [-s, c]]) + [x, y]
    ox, oy, angle = world.origin
    c, s = math.cos(angle), math.sin(angle)
    grid = (points - [ox, oy]) @ np.array([[c, -s], [s, c]]) / world.resolution
    low = np.floor(grid.min(axis=0)).astype(int)
    high = np.floor(grid.max(axis=0)).astype(int)
    height, width = world.pixels.shape
    if min(low) < 0 or high[0] >= width or high[1] >= height:
        return "The footprint reaches outside the world"
    yy, xx = np.mgrid[low[1] : high[1] + 1, low[0] : high[0] + 1]
    centres = np.stack((xx + 0.5, yy + 0.5), axis=-1)
    overlaps = np.ones(xx.shape, dtype=bool)
    edges = np.roll(grid, -1, axis=0) - grid
    for axis in np.concatenate((np.eye(2), np.stack((-edges[:, 1], edges[:, 0]), axis=1))):
        projection = grid @ axis
        middle = centres @ axis
        radius = 0.5 * np.abs(axis).sum()
        overlaps &= (middle + radius > projection.min()) & (middle - radius < projection.max())
    cells = world.pixels[low[1] : high[1] + 1, low[0] : high[0] + 1]
    if (cells[overlaps] == WALL).any():
        return "The footprint overlaps a wall"
    return ""


def footprint_size(footprint: Sequence[Sequence[float]]) -> str:
    """The footprint's bounding box: "0.44 × 0.38 m" ("" when unavailable)."""
    try:
        xs = [float(point[0]) for point in footprint]
        ys = [float(point[1]) for point in footprint]
    except (TypeError, ValueError, IndexError):
        return ""
    if not xs or not all(math.isfinite(v) for v in (*xs, *ys)):
        return ""
    return f"{max(xs) - min(xs):.2f} × {max(ys) - min(ys):.2f} m"


# ---------- the positions table ----------


class PositionsTable(PaintedTable):
    """``.saved .tbl``: Name, X · m, Y · m, Heading; the Default row dimmed with "world" and a lock after its
    name; the template row selected (fill and ink bar)."""

    chosen = QtCore.Signal(int)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(
            themes,
            [
                PaintedColumn("Name", None, padding=20, padding_right=10),
                PaintedColumn("X · m", 62, padding=10, padding_right=10, right=True),
                PaintedColumn("Y · m", 62, padding=10, padding_right=10, right=True),
                PaintedColumn("Heading", 84, padding=10, padding_right=20, right=True),
            ],
            parent,
            header_height=36,
        )
        self.poses: list[Pose] = []
        self.current = 0
        self.hovered = -1
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "none")
        self.setAccessibleName("Starting positions")

    def set_poses(self, poses: list[Pose], current: int) -> None:
        self.poses, self.current = poses, current
        self.rows_changed(len(poses))

    def row_height(self, index: int) -> int:
        return 40

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self.row_at(event.position().y())
        if index >= 0 and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.chosen.emit(index)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self.row_at(event.position().y())
        if index != self.hovered:
            self.hovered = index
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hovered = -1
        self.viewport().update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        step = {QtCore.Qt.Key.Key_Up: -1, QtCore.Qt.Key.Key_Down: 1}.get(QtCore.Qt.Key(event.key()))
        if step is not None and self.poses:
            self.chosen.emit(min(max(0, self.current + step), len(self.poses) - 1))
            return
        super().keyPressEvent(event)

    def paint_row(self, painter: QtGui.QPainter, index: int, rect: QtCore.QRect) -> None:
        theme = self.themes.current
        pose = self.poses[index]
        last = index == len(self.poses) - 1
        content = rect.height() - (0 if last else 1)
        if index == self.current:
            painter.fillRect(rect, theme.color("surface.selected"))
            painter.fillRect(QtCore.QRect(0, rect.top(), 2, content), theme.color("text.ink"))
        elif index == self.hovered:
            painter.fillRect(rect, theme.color("surface.hover"))
        if not last:
            painter.fillRect(
                QtCore.QRect(0, rect.bottom(), rect.width(), 1), theme.color("border.line")
            )
        spans = self.column_spans()
        line_top = rect.top() + (content - 18) / 2
        tertiary = theme.color("text.tertiary")
        number = tertiary if pose.default else theme.color("text.body")
        left, width = spans[0]
        room = width - 20 - 10
        if pose.default:
            x = 20 + draw_text(
                painter, 20, line_top, "Default", BODY_MEDIUM, theme.color("text.secondary")
            )
            x += 6 + draw_text(painter, x + 6, line_top, "world", WORLD_TAG, tertiary)
            origin = QtCore.QPointF(x + 6, snap(rect.top() + (content - 14) / 2))
            gui_icons.draw(painter, "lock", tertiary, origin, 14)
        else:
            name = BODY_MEDIUM.elide(pose.name, room)
            draw_text(painter, 20, line_top, name, BODY_MEDIUM, theme.color("text.ink"))
        values = (
            coordinate_text(pose.x, 2),
            coordinate_text(pose.y, 2),
            (fixed(heading_degrees(pose.yaw), 1) or "—") + "°",
        )
        for column, text in zip((1, 2, 3), values, strict=True):
            left, width = spans[column]
            x = left + width - self.columns[column].padding_right - BODY_TABULAR.advance(text)
            draw_text(painter, x, line_top, text, BODY_TABULAR, number)


class PositionsCard(Card):
    """``.saved``: "Starting positions" with Delete… (the selected saved spawn) and the ⋯ menu (Import YAML,
    Export YAML; D1) over the table."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "Starting positions", self)
        self.delete = Button(themes, "Delete…", icon="trash", variant="danger", small=True)
        self.delete.setEnabled(False)
        self.head.add(self.delete)
        self.more = Button(themes, icon="more", small=True, icon_size=14)
        self.more.setToolTip("Import YAML, Export YAML")
        self.more.setAccessibleName("Spawn YAML")
        self.head.add(self.more)
        self.table = PositionsTable(themes, self)

    def natural_height(self, rows: int) -> int:
        return 2 + 52 + 36 + 40 * rows

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.head.setGeometry(1, 1, self.width() - 2, 52)
        self.table.setGeometry(1, 53, self.width() - 2, max(0, self.height() - 54))


# ---------- the pose form ----------


class UnitField(QtWidgets.QLineEdit):
    """``.input.unitin``: the value at the left and the unit (``text.tertiary``) at the right. A value set by
    the program keeps its exact number while its rounded text is unchanged."""

    def __init__(self, themes: ThemeManager, unit: str, digits: int) -> None:
        super().__init__()
        self.themes, self.unit, self.digits = themes, unit, digits
        self.exact: tuple[str, float] | None = None
        self.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression("[-−]?[0-9]*([.][0-9]*)?"), self
            )
        )
        self.setFont(QtGui.QFont(self.font()))
        self.setTextMargins(0, 0, math.ceil(BODY.advance(unit)) + 8, 0)
        themes.changed.connect(self.update)

    def set_value(self, value: float) -> None:
        text = coordinate_text(value, self.digits)
        self.exact = (text, value)
        self.blockSignals(True)
        self.setText(text)
        self.blockSignals(False)

    def value(self) -> float:
        """The number; ValueError while the text is not a finite number."""
        text = self.text()
        if self.exact is not None and text == self.exact[0]:
            return self.exact[1]
        value = float(text.replace("−", "-"))
        if not math.isfinite(value):
            raise ValueError("Enter a finite number")
        return value

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        x = self.width() - 11 - BODY.advance(self.unit)
        color = self.themes.current.color("text.tertiary")
        draw_text(painter, x, (self.height() - 18) / 2, self.unit, BODY, color)


class HeadingDial(Themed):
    """``.dial``: 56 px circle (1 px ``border.strong``) with a centre dot, an ``accent.primary`` arm and a
    handle pointing at the heading; drag to turn, arrow keys ±1°, Shift ±15°."""

    turned = QtCore.Signal(float)  # degrees, counter-clockwise from east

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.degrees = 0.0
        self.setFixedSize(56, 56)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 30)
        self.setToolTip("Drag to turn")
        self.setAccessibleName("Heading dial")

    def set_degrees(self, degrees: float) -> None:
        if degrees != self.degrees:
            self.degrees = degrees
            self.update()

    def _turn_to(self, position: QtCore.QPointF) -> None:
        dx, dy = position.x() - 28, 28 - position.y()
        if dx or dy:
            self.turned.emit(round(math.degrees(math.atan2(dy, dx)), 1))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self._turn_to(event.position())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._turn_to(event.position())

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        step = {
            QtCore.Qt.Key.Key_Left: 1,
            QtCore.Qt.Key.Key_Up: 1,
            QtCore.Qt.Key.Key_Right: -1,
            QtCore.Qt.Key.Key_Down: -1,
        }.get(QtCore.Qt.Key(event.key()))
        if step is None:
            super().keyPressEvent(event)
            return
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            step *= 15
        self.turned.emit(heading_degrees(math.radians(self.degrees + step)))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.strong"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawEllipse(QtCore.QRectF(0.5, 0.5, 55, 55))
        angle = math.radians(self.degrees)
        c, s = math.cos(angle), -math.sin(angle)
        accent = self.color("data.path")  # the edited pose's colour on the map
        painter.setPen(QtGui.QPen(accent, 2))
        painter.drawLine(QtCore.QPointF(28, 28), QtCore.QPointF(28 + 22 * c, 28 + 22 * s))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.color("text.secondary"))
        painter.drawEllipse(QtCore.QPointF(28, 28), 2, 2)
        painter.setPen(QtGui.QPen(accent, 2))
        painter.setBrush(QtGui.QColor("#FFFFFF"))
        painter.drawEllipse(QtCore.QPointF(28 + 21 * c, 28 + 21 * s), 4, 4)


class ValidityLine(Themed):
    """``.range``: ``ccheck`` in ``state.ok`` and "Inside the world; footprint clear of walls.", or ``alert``
    in ``state.bad`` and the reason (12/16 ``text.secondary``)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.text, self.ok = "", True
        self.setFixedHeight(16)

    def set_state(self, text: str, ok: bool) -> None:
        if (text, ok) != (self.text, self.ok):
            self.text, self.ok = text, ok
            self.setToolTip(text)
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self.text:
            return
        painter = QtGui.QPainter(self)
        icon, token = ("ccheck", "state.ok") if self.ok else ("alert", "state.bad")
        gui_icons.draw(painter, icon, self.color(token), QtCore.QPointF(0, 1), 14)
        text = VALIDITY.elide(self.text, self.width() - 22)
        draw_text(painter, 22, 0, text, VALIDITY, self.color("text.secondary"))


class PoseForm(Card):
    """``.form``: "New pose" (D1); Name; X and Y in metres; Heading in degrees with the dial; the validity line;
    Revert (to the template) and "Save as new spawn"."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "New pose", self)
        self.labels = [
            TextLine(themes, text, LABEL, "text.body", self)
            for text in ("Name", "X", "Y", "Heading")
        ]
        self.heading_hint = TextLine(
            themes, "· 0° east, counter-clockwise", HINT, "text.tertiary", self
        )
        self.name = QtWidgets.QLineEdit(next_spawn_name(()), self)
        self.name.setAccessibleName("Spawn name")
        self.x_field = UnitField(themes, "m", 2)
        self.y_field = UnitField(themes, "m", 2)
        self.heading_field = UnitField(themes, "°", 1)
        for field, name in (
            (self.x_field, "X in metres"),
            (self.y_field, "Y in metres"),
            (self.heading_field, "Heading in degrees"),
        ):
            field.setParent(self)
            field.setAccessibleName(name)
        self.dial = HeadingDial(themes, self)
        self.validity = ValidityLine(themes, self)
        self.footer = FlexRow(self, gap=8, left=20, right=20)
        self.revert = Button(themes, "Revert", icon="reset", variant="ghost")
        self.revert.setToolTip("Back to the selected starting position")
        self.save = Button(themes, "Save as new spawn", icon="check", variant="primary")
        self.footer.add(self.revert)
        self.footer.add_stretch()
        self.footer.add(self.save)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width = self.width()
        inner = width - 42
        self.head.setGeometry(1, 1, width - 2, 52)
        half = (inner - 12) / 2
        place(self.labels[0], 21, 69, inner, 18)
        self.name.setGeometry(21, 93, inner, 32)
        place(self.labels[1], 21, 139, half, 18)
        place(self.labels[2], 21 + half + 12, 139, half, 18)
        self.x_field.setGeometry(21, 163, round(half), 32)
        place(self.y_field, 21 + half + 12, 163, half, 32)
        place(self.labels[3], 21, 209, LABEL.advance("Heading"), 18)
        place(self.heading_hint, 21 + LABEL.advance("Heading") + 4, 209, 200, 18)
        self.heading_field.setGeometry(21, 233, inner - 56 - 12, 32)
        self.dial.move(21 + inner - 56, 209)
        self.validity.setGeometry(21, 279, inner, 16)
        self.footer.setGeometry(1, self.height() - 1 - 44, width - 2, 32)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        QtGui.QPainter(self).fillRect(
            QtCore.QRect(1, self.height() - 1 - 57, self.width() - 2, 1), self.color("border.line")
        )


# ---------- the map ----------


def to_world(placement: Placement, point: QtCore.QPointF) -> tuple[float, float]:
    """Widget point to world metres (the inverse of ``Placement.point``)."""
    world = placement.world
    column = (point.x() - placement.rect.x()) / placement.scale
    row = (placement.rect.bottom() - point.y()) / placement.scale_y
    ox, oy, yaw = world.origin
    dx, dy = column * world.resolution, row * world.resolution
    c, s = math.cos(yaw), math.sin(yaw)
    return ox + c * dx - s * dy, oy + s * dx + c * dy


class SpawnMap(LiveMap):
    """The map of 15: the world fitted 40 px inside the stage (``border.strong`` ring), a 1 m grid, metre ticks
    every 2 m outside the map, the saved poses labelled, and the pose being edited as the robot's footprint
    with a heading handle. A click positions the pose, dragging the handle turns it, the wheel and the
    controls zoom (and fit), dragging pans while zoomed."""

    PAD = 40
    RING = "border.strong"

    positioned = QtCore.Signal(float, float)
    turned = QtCore.Signal(float)  # world yaw in radians
    cursor_moved = QtCore.Signal(object)  # (x, y) in metres, or None off the map

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.poses: list[Pose] = []
        self.current_key = ""
        self.pose: tuple[float, float, float] | None = None
        self.footprint: list[list[float]] = []
        self._press: QtCore.QPointF | None = None
        self._moved = False
        self._turning = False

    def set_world(self, world: WorldMap | None) -> None:
        """Free cells read as the Monitor shows them fully explored (``data.coverage`` at the explored
        opacity over the free colour)."""
        super().set_world(world)
        self.layers.explored = (
            None if world is None else np.where(world.pixels == FREE, 255, 0).astype(np.uint8)
        )
        self._base = None

    def set_scene(
        self,
        poses: list[Pose],
        current_key: str,
        pose: tuple[float, float, float] | None,
        footprint: list[list[float]],
    ) -> None:
        self.poses, self.current_key, self.pose, self.footprint = (
            poses,
            current_key,
            pose,
            footprint,
        )
        self.update()

    def handle_point(self, placement: Placement) -> QtCore.QPointF | None:
        if self.pose is None:
            return None
        x, y, yaw = self.pose
        centre = placement.point(x, y)
        angle = math.radians(placement.heading(yaw))
        return QtCore.QPointF(
            centre.x() + HANDLE_LENGTH * math.cos(angle),
            centre.y() + HANDLE_LENGTH * math.sin(angle),
        )

    # input

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        placement = self.placement()
        if event.button() != QtCore.Qt.MouseButton.LeftButton or placement is None:
            super().mousePressEvent(event)
            return
        handle = self.handle_point(placement)
        position = event.position()
        if (
            handle is not None
            and math.dist((handle.x(), handle.y()), (position.x(), position.y())) <= 9
        ):
            self._turning = True
        else:
            self._press, self._moved = position, False
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        placement = self.placement()
        position = event.position()
        if placement is not None and self._turning and self.pose is not None:
            centre = placement.point(self.pose[0], self.pose[1])
            screen = math.atan2(position.y() - centre.y(), position.x() - centre.x())
            self.turned.emit(self.world.origin[2] - screen if self.world else -screen)
            return
        if self._press is not None:
            delta = position - self._press
            if self._moved or abs(delta.x()) + abs(delta.y()) > 3:
                self._moved = True
                self._press = position
                self.pan_by(delta.x(), delta.y())
            return
        box = self.visible_box()
        if placement is not None and box is not None and box.contains(position):
            self.cursor_moved.emit(to_world(placement, position))
        else:
            self.cursor_moved.emit(None)
        handle = self.handle_point(placement) if placement is not None else None
        near = (
            handle is not None
            and math.dist((handle.x(), handle.y()), (position.x(), position.y())) <= 9
        )
        if near:
            self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        elif not self.fitted():
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        placement = self.placement()
        if self._turning:
            self._turning = False
            return
        if self._press is not None and not self._moved and placement is not None:
            box = self.visible_box()
            if box is not None and box.contains(event.position()):
                self.positioned.emit(*to_world(placement, event.position()))
        self._press, self._moved = None, False

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.cursor_moved.emit(None)
        super().leaveEvent(event)

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        box = self.box()
        placement = self.placement()
        if box is None or placement is None or self.world is None:
            painter.fillRect(self.rect(), theme.color("surface.card"))
            if self.message:
                x = (self.width() - BODY.advance(self.message)) / 2
                color = theme.color("text.secondary")
                draw_text(painter, x, (self.height() - 18) / 2, self.message, BODY, color)
            return
        painter.drawImage(QtCore.QPointF(0, 0), self._base_image(box))
        visible = box.intersected(self.stage())
        painter.save()
        painter.setClipRect(visible)
        self._paint_grid(painter, placement)
        for pose in self.poses:
            if pose.key == self.current_key and self.pose is not None:
                continue
            point = placement.point(pose.x, pose.y)
            paint_spawn(painter, point, placement.heading(pose.yaw), theme, picked=True)
            paint_spawn_label(painter, placement, point, pose.name, theme, size=11, bold=False)
        self._paint_pose(painter, placement)
        painter.restore()
        if self.controls.isVisible():
            controls = self.controls.geometry()
            layers = ((*MapControls.SHADOW, theme.tokens["overlay.shadowSmall"]),)
            margin = shadow_margin(layers)
            painter.drawImage(
                QtCore.QPointF(controls.x() - margin, controls.y() - margin),
                shadow_image(controls.width(), controls.height(), 6, layers),
            )
        self._paint_ticks(painter, placement, visible)

    def _extent(self) -> tuple[float, float, float, float]:
        assert self.world is not None
        ox, oy, _ = self.world.origin
        width, height = self.world.size
        return ox, oy, ox + width * self.world.resolution, oy + height * self.world.resolution

    def _paint_grid(self, painter: QtGui.QPainter, placement: Placement) -> None:
        left, bottom, right, top = self._extent()
        theme = self.theme
        for metre in range(math.ceil(left), math.floor(right) + 1):
            a, b = placement.point(metre, bottom), placement.point(metre, top)
            color = theme.color("map.gridAxis" if metre == 0 else "map.grid")
            painter.fillRect(QtCore.QRectF(a.x() - 0.5, b.y(), 1, a.y() - b.y()), color)
        for metre in range(math.ceil(bottom), math.floor(top) + 1):
            a, b = placement.point(left, metre), placement.point(right, metre)
            color = theme.color("map.gridAxis" if metre == 0 else "map.grid")
            painter.fillRect(QtCore.QRectF(a.x(), a.y() - 0.5, b.x() - a.x(), 1), color)

    def _paint_ticks(
        self, painter: QtGui.QPainter, placement: Placement, visible: QtCore.QRectF
    ) -> None:
        left, bottom, right, top = self._extent()
        color = self.theme.color("text.tertiary")
        for metre in range(math.ceil(left / 2) * 2, math.floor(right) + 1, 2):
            x = placement.point(metre, bottom).x()
            if visible.left() - 0.5 <= x <= visible.right() + 0.5:
                text = str(metre).replace("-", "−")
                draw_text(
                    painter, x - TICK.advance(text) / 2, visible.bottom() + 7, text, TICK, color
                )
        for metre in range(math.ceil(bottom / 2) * 2, math.floor(top) + 1, 2):
            y = placement.point(left, metre).y()
            if visible.top() - 0.5 <= y <= visible.bottom() + 0.5:
                text = str(metre).replace("-", "−")
                x = visible.left() - 8 - TICK.advance(text)
                draw_text(painter, x, y - 6, text, TICK, color)

    def _paint_pose(self, painter: QtGui.QPainter, placement: Placement) -> None:
        """The edited pose in the Monitor's robot colour (``data.path``): the footprint (18 % fill, 1.5 px
        outline), the heading line (34 px, 2 px) and the handle (r 5, card fill), and its name centred above."""
        if self.pose is None:
            return
        x, y, yaw = self.pose
        theme = self.theme
        accent = theme.color("data.path")
        centre = placement.point(x, y)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.footprint:
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [
                placement.point(x + px * c - py * s, y + px * s + py * c)
                for px, py in self.footprint
            ]
            fill = QtGui.QColor(accent)
            fill.setAlphaF(0.18)
            painter.setPen(QtGui.QPen(accent, 1.5))
            painter.setBrush(fill)
            painter.drawPolygon(QtGui.QPolygonF(corners))
        handle = self.handle_point(placement)
        if handle is not None:
            painter.setPen(QtGui.QPen(accent, 2))
            painter.drawLine(centre, handle)
            painter.setBrush(theme.color("surface.card"))
            painter.drawEllipse(handle, HANDLE_RADIUS, HANDLE_RADIUS)
        painter.restore()
        name = next((p.name for p in self.poses if p.key == self.current_key), "")
        if name:
            label_font = font(11, 600)
            path = QtGui.QPainterPath()
            width = QtGui.QFontMetricsF(label_font).horizontalAdvance(name)
            below = centre.y() - placement.rect.y() < 30
            baseline = centre.y() + 25 if below else centre.y() - 16
            path.addText(QtCore.QPointF(centre.x() - width / 2, baseline), label_font, name)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.strokePath(path, QtGui.QPen(theme.color("map.halo"), 3))
            painter.fillPath(path, theme.color("text.ink"))
            painter.restore()


class LegendLine(Themed):
    """``.legend`` (15): the footprint swatch and "Selected pose · <robot> footprint 0.44 × 0.38 m", the arrow
    swatch and "Other saved poses", and the interaction hint at the right (``text.tertiary``)."""

    HINT_TEXT = "Click to position · drag the arrow to turn · scroll to zoom"

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.selected = ""
        self.setFixedHeight(40)

    def set_selected(self, text: str) -> None:
        if text != self.selected:
            self.selected = text
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), theme.color("border.line"))
        top = 1 + (39 - 18) / 2
        secondary = theme.color("text.secondary")
        accent = theme.color("data.path")  # the edited pose's colour
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        y = round(1 + (39 - 14) / 2)
        fill = QtGui.QColor(accent)
        fill.setAlphaF(0.18)
        painter.setPen(QtGui.QPen(accent, 1.5))
        painter.setBrush(fill)
        painter.drawRect(QtCore.QRectF(20 + 2, y + 3, 11, 8))
        painter.setPen(QtGui.QPen(accent, 2))
        painter.drawLine(QtCore.QPointF(20 + 13, y + 7), QtCore.QPointF(20 + 19, y + 7))
        painter.setPen(QtGui.QPen(accent, 1.5))
        painter.setBrush(QtGui.QColor("#FFFFFF"))
        painter.drawEllipse(QtCore.QPointF(20 + 19, y + 7), 2.5, 2.5)
        painter.restore()
        x = 20.0 + 22 + 7
        x += draw_text(painter, x, top, self.selected, HINT, secondary) + 22
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        outline = QtGui.QPen(theme.color("data.path"), 1.3)  # as ``paint_spawn`` draws saved poses
        outline.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(outline)
        painter.setBrush(theme.color("data.path"))
        painter.drawPolygon(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(x + px, y + py)
                    for px, py in ((14, 7), (3, 1.5), (6, 7), (3, 12.5))
                ]
            )
        )
        painter.restore()
        x += 16 + 7
        x += draw_text(painter, x, top, "Other saved poses", HINT, secondary)
        hint = HINT.elide(self.HINT_TEXT, max(0.0, self.width() - 20 - x - 22))
        draw_text(
            painter,
            self.width() - 20 - HINT.advance(hint),
            top,
            hint,
            HINT,
            theme.color("text.tertiary"),
        )


class CursorReadout(Themed, CssBox):
    """``.cursor``: "cursor x 3.12 y −4.85 m · grid 1 m" (mono 12 px, values ink 500)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.point: tuple[float, float] | None = None
        self.setFixedHeight(12)

    def parts(self) -> list[tuple[str, TextStyle, str]]:
        x, y = (
            ("—", "—")
            if self.point is None
            else (coordinate_text(self.point[0], 2), coordinate_text(self.point[1], 2))
        )
        return [
            ("cursor", CURSOR, "text.secondary"),
            (f"x {x}", CURSOR_VALUE, "text.ink"),
            (f"y {y}", CURSOR_VALUE, "text.ink"),
            ("m · grid 1 m", CURSOR, "text.secondary"),
        ]

    def set_point(self, point: tuple[float, float] | None) -> None:
        if point != self.point:
            self.point = point
            self.updateGeometry()
            self.update()

    def natural_width(self) -> float:
        return sum(style.advance(text) for text, style, _ in self.parts()) + 30

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 12)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        x = self.shift + (self.width() - self.shift - self.natural_width())
        for text, style, token in self.parts():
            x += draw_text(painter, x, 0, text, style, self.color(token)) + 10


class MapCard(Card):
    """``.mapcard``: the world's name and size, the cursor readout, the map and the legend line."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = FlexRow(self, gap=12, left=20, right=16)
        self.title = TextLine(themes, "", CARD_TITLE, "text.ink")
        self.size_text = TextLine(themes, "", BODY, "text.secondary")
        self.readout = CursorReadout(themes)
        self.head.add(self.title)
        self.head.add(self.size_text)
        self.head.add_stretch()
        self.head.add(self.readout)
        self.map = SpawnMap(themes, self)
        self.legend = LegendLine(themes, self)
        self.map.cursor_moved.connect(self.readout.set_point)

    def set_world(self, name: str, size: str) -> None:
        self.title.setText(name)
        self.size_text.setText(size)
        self.head.relayout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.head.setGeometry(1, 1, width - 2, 51)
        self.map.setGeometry(1, 53, width - 2, max(0, height - 53 - 41))
        self.legend.setGeometry(1, height - 41, width - 2, 40)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        QtGui.QPainter(self).fillRect(
            QtCore.QRect(1, 52, self.width() - 2, 1), self.color("border.line")
        )


# ---------- the page ----------


class WorldsPage(QtWidgets.QWidget):
    """Worlds & Spawns: World and Robot footprint selects in the head, the positions table and pose form at the
    left, the map at the right. ``received``/``failed`` return True for its replies; ``changed`` tells the
    window that the spawn library changed."""

    changed = QtCore.Signal()
    deleted = QtCore.Signal(object)  # the spawns.delete reply
    notify = QtCore.Signal(str, str, int)

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport = themes, transport
        self.catalog: dict[str, Any] = {}
        self.tables: list[
            Any
        ] = []  # the window refreshes these when the page shows (none: it syncs itself)
        self.world: WorldMap | None = None
        self.poses: list[Pose] = []
        self.template = 0  # index in poses
        self.export_path = ""
        self.pending: tuple[str, int] | None = (
            None  # world and offset of the list request in flight
        )
        self.stale = False
        self._page: list[dict[str, Any]] = []
        self.engine_problem = ""  # the engine's reason when it rejected the last save
        self.exact_yaw = ("", 0.0)  # the template's heading text and its exact yaw
        self.spawn_names: dict[str, list[str]] = {}
        self.name_scan: tuple[str, str, bool] | None = None
        self.scan_buffer: list[str] = []
        self.scan_pages = 0

        self.head = FlexRow(self, gap=12)
        self.head.add(TextLine(themes, "Worlds & Spawns", PAGE_TITLE, "text.ink"))
        rule = Rule(themes, vertical=True)
        rule.setFixedHeight(24)
        self.head.add(rule, width=1, margin=8)
        self.spawn_world = KeySelect(themes, "World")
        self.spawn_robot = KeySelect(themes, "Robot footprint")
        self.head.add(self.spawn_world, width=210)
        self.head.add(self.spawn_robot, width=210)
        self.spawn_world.currentIndexChanged.connect(self.load_world_map)
        self.spawn_robot.currentIndexChanged.connect(self.update_pose)
        self.positions = PositionsCard(themes, self)
        self.positions.table.chosen.connect(self.choose)
        self.positions.more.clicked.connect(self.open_menu)
        self.positions.delete.clicked.connect(self.delete_spawn)
        self.form = PoseForm(themes, self)
        for field in (self.form.x_field, self.form.y_field, self.form.heading_field):
            field.textChanged.connect(self.update_pose)
        self.form.dial.turned.connect(self.set_heading)
        self.form.revert.clicked.connect(lambda: self.choose(self.template))
        self.form.save.clicked.connect(self.save_spawn)
        self.map_card = MapCard(themes, self)
        # The gaps between the cards can be dragged (remembered; a double click restores the design).
        self.side = CardSplit(themes, self, "worlds/side", self._layout, across=True)
        self.positions_split = CardSplit(
            themes, self, "worlds/positions", self._layout, across=False
        )
        self.map_card.map.positioned.connect(self.position)
        self.map_card.map.turned.connect(lambda yaw: self.set_heading(heading_degrees(yaw)))
        self.sync = QtCore.QTimer(self)
        self.sync.setInterval(2000)
        self.sync.timeout.connect(lambda: self.refresh() if self.isVisible() else None)
        self.sync.start()

    # catalog, world and robot

    def set_catalog(self, catalog: dict[str, Any]) -> None:
        """Fill the World and Robot footprint selects (keeping their choices) and load the world."""
        self.catalog = catalog
        for select, items in (
            (self.spawn_robot, catalog["robots"]),
            (self.spawn_world, catalog["worlds"]),
        ):
            chosen = select.currentData()
            select.blockSignals(True)
            select.clear()
            for item in items:
                select.addItem(item["label"], item["id"])
            select.setCurrentIndex(max(0, select.findData(chosen)))
            select.blockSignals(False)
        self.load_world_map()

    def world_info(self) -> dict[str, Any] | None:
        world = self.spawn_world.currentData()
        return next((w for w in self.catalog.get("worlds", []) if w["id"] == world), None)

    def robot_info(self) -> dict[str, Any] | None:
        robot = self.spawn_robot.currentData()
        return next((r for r in self.catalog.get("robots", []) if r["id"] == robot), None)

    def load_world_map(self, *args: Any) -> None:
        info = self.world_info()
        if info is None or not self.catalog:
            return
        self.world = None
        self.map_card.map.set_world(None)
        self.map_card.map.set_message("Loading the world map…")
        width = info["width"] * info["resolution"]
        height = info["height"] * info["resolution"]
        self.map_card.set_world(info["label"], f"{width:.1f} × {height:.1f} m")
        default = info["default_spawn"]
        self.poses = [Pose(DEFAULT_KEY, "Default", default["x"], default["y"], default["yaw"])]
        self._page = []
        self.stale = self.pending is not None  # the reply in flight is for the previous world
        self.template = 0
        self.positions.table.set_poses(self.poses, 0)
        self.choose(0)
        self.suggest_name(info["id"])
        self.refresh()
        self.transport.submit("spawn-map", "spawns.map", {"world_id": info["id"]})
        self._layout()

    # the list of saved spawns

    def refresh(self) -> None:
        info = self.world_info()
        if info is None:
            return
        if self.pending is not None:
            self.stale = True
            return
        self._request(info["id"], 0)

    def _request(self, world: str, offset: int) -> None:
        self.pending = (world, offset)
        self.transport.submit(
            "spawn-table", "spawns.list", {"world_id": world, "offset": offset, "limit": LIST_PAGE}
        )

    def _listed(self, result: dict[str, Any]) -> None:
        assert self.pending is not None
        world, offset = self.pending
        self.pending = None
        info = self.world_info()
        if info is None:
            return
        if self.stale or world != info["id"]:
            self.stale = False
            self._request(info["id"], 0)
            return
        self._page = ([] if offset == 0 else self._page) + list(result.get("rows") or [])
        following = result.get("next_offset")
        if following is not None and len(self._page) < LIST_LIMIT:
            self._request(world, int(following))
            return
        if following is not None:
            self.notify.emit(
                f"First {len(self._page):,} saved spawns of this world listed", "warn", 8000
            )
        saved = []
        for row in self._page:
            snapshot = row.get("snapshot") or {}
            try:
                x, y, yaw = (float(snapshot[key]) for key in ("x", "y", "yaw"))
            except (KeyError, TypeError, ValueError):
                continue
            saved.append(
                Pose(
                    str(row["id"]),
                    str(row.get("name", "")),
                    x,
                    y,
                    yaw,
                    str(row.get("family_id") or ""),
                )
            )
        saved.sort(key=lambda pose: (pose.name.casefold(), pose.key))  # by name, as 06 and 15
        poses = self.poses[:1] + saved
        if [p.key for p in poses] != [p.key for p in self.poses]:
            key = self.poses[self.template].key if self.template < len(self.poses) else DEFAULT_KEY
            self.poses = poses
            self.template = next((i for i, p in enumerate(poses) if p.key == key), 0)
            self.positions.table.set_poses(poses, self.template)
            self.update_delete()
            self._layout()
            self._update_map()

    def choose(self, index: int) -> None:
        """Load a starting position as the template of a new spawn (its name stays the suggestion)."""
        if not 0 <= index < len(self.poses):
            return
        self.template = index
        self.positions.table.current = index
        self.positions.table.viewport().update()
        pose = self.poses[index]
        self.engine_problem = ""
        self.form.x_field.set_value(pose.x)
        self.form.y_field.set_value(pose.y)
        self.form.heading_field.set_value(heading_degrees(pose.yaw))
        self.exact_yaw = (self.form.heading_field.text(), pose.yaw)
        self.update_pose()
        self.update_delete()

    # the pose

    def pose(self) -> tuple[float, float, float]:
        """The form's pose; ValueError while a field is not a number. An unchanged heading keeps the template's
        exact yaw."""
        x, y = self.form.x_field.value(), self.form.y_field.value()
        text = self.form.heading_field.text()
        yaw = (
            self.exact_yaw[1]
            if text == self.exact_yaw[0]
            else math.radians(self.form.heading_field.value())
        )
        return x, y, yaw

    def position(self, x: float, y: float) -> None:
        self.engine_problem = ""
        self.form.x_field.set_value(x)
        self.form.y_field.set_value(y)
        self.update_pose()

    def set_heading(self, degrees: float) -> None:
        self.engine_problem = ""
        self.form.heading_field.set_value(degrees)
        self.update_pose()

    def update_pose(self, *args: Any) -> None:
        robot = self.robot_info()
        footprint = list(robot.get("footprint") or []) if robot else []
        try:
            pose: tuple[float, float, float] | None = self.pose()
        except ValueError:
            pose = None
        if pose is not None:
            self.form.dial.set_degrees(heading_degrees(pose[2]))
        if pose is None:
            self.form.validity.set_state("Enter finite coordinates and heading.", False)
        elif self.engine_problem:
            self.form.validity.set_state(self.engine_problem, False)
        elif self.world is None:
            self.form.validity.set_state("", True)
        else:
            problem = footprint_problem(self.world, footprint, *pose)
            self.form.validity.set_state(
                problem or "Inside the world; footprint clear of walls.", not problem
            )
        label = robot["label"] if robot else "Robot"
        size = footprint_size(footprint)
        self.map_card.legend.set_selected(
            f"Selected pose · {label} footprint {size}" if size else "Selected pose"
        )
        self._update_map(pose, footprint)

    def _update_map(
        self,
        pose: tuple[float, float, float] | None = None,
        footprint: list[list[float]] | None = None,
    ) -> None:
        if pose is None:
            try:
                pose = self.pose()
            except ValueError:
                pose = None
        if footprint is None:
            robot = self.robot_info()
            footprint = list(robot.get("footprint") or []) if robot else []
        key = self.poses[self.template].key if self.template < len(self.poses) else ""
        self.map_card.map.set_scene(self.poses, key, pose, footprint)

    # saving, names, YAML

    def save_spawn(self) -> None:
        try:
            x, y, yaw = self.pose()
        except ValueError:
            self.form.validity.set_state(
                "Enter finite coordinates and heading before saving.", False
            )
            return
        value = {
            "name": self.form.name.text(),
            "world_id": self.spawn_world.currentData(),
            "robot_ids": [self.spawn_robot.currentData()],
            "x": x,
            "y": y,
            "yaw": yaw,
        }
        self.transport.submit("spawn-edit:save", "spawns.save", value)

    def suggest_name(self, world: str) -> None:
        """Propose the next "Spawn N" for the world unless the user typed a name, then rescan."""
        if not self.form.name.isModified():
            self.form.name.setText(next_spawn_name(self.spawn_names.get(world, [])))
        self.name_scan = (world, uuid.uuid4().hex, False)
        self.scan_buffer, self.scan_pages = [], 0
        self.scan_page(0)

    def scan_page(self, offset: int) -> None:
        assert self.name_scan is not None
        world, token, archived = self.name_scan
        self.scan_pages += 1
        # Both passes together read every saved name, whether or not an engine filters archived rows.
        self.transport.submit(
            "spawn-names:" + token,
            "spawns.list",
            {
                "world_id": world,
                "search": "spawn",
                "summary": True,
                "limit": LIST_PAGE,
                "offset": offset,
                "archived": archived,
            },
        )

    def scanned_names(self, result: dict[str, Any]) -> None:
        assert self.name_scan is not None
        world, token, archived = self.name_scan
        self.scan_buffer.extend(str(row["name"]) for row in result["rows"])
        more = result.get("next_offset")
        if more is not None and self.scan_pages < NAME_SCAN_PAGES:
            self.scan_page(more)
            return
        if more is not None:
            self.notify.emit(
                "Not every saved spawn name was read; check the name before saving.", "warn", 8000
            )
        if not archived:
            self.name_scan, self.scan_pages = (world, token, True), 0
            self.scan_page(0)
            return
        self.name_scan = None
        self.spawn_names[world] = self.scan_buffer
        if world == self.spawn_world.currentData() and not self.form.name.isModified():
            self.form.name.setText(next_spawn_name(self.spawn_names[world]))

    def open_menu(self) -> None:
        host = self.window().findChild(PopupHost)
        if host is None:
            return
        saved = not self.poses[self.template].default if self.template < len(self.poses) else False
        menu = Menu(self.themes)
        menu.add_item("Import YAML", self.import_spawn, icon="upload")
        menu.add_item(
            "Export YAML",
            self.export_spawn,
            icon="download",
            enabled=saved,
            reason="" if saved else "Default",
        )
        menu.popup(host, self.positions.more)

    def selected_saved(self) -> Pose | None:
        """The selected row when it is a saved spawn (the Default cannot be deleted)."""
        pose = self.poses[self.template] if self.template < len(self.poses) else None
        return None if pose is None or pose.default or not pose.family else pose

    def update_delete(self) -> None:
        pose = self.selected_saved()
        self.positions.delete.setEnabled(pose is not None)
        self.positions.delete.setToolTip(
            "" if pose else "Select a saved spawn; the world's Default cannot be deleted"
        )

    def delete_spawn(self) -> None:
        pose = self.selected_saved()
        host = self.window().findChild(ModalHost)
        if pose is None or host is None:
            return
        world = self.world_info()
        dialog = Consequences(self.themes, "Delete spawn")
        dialog.add_sentence([Run(f"Delete “{pose.name}”? This cannot be undone.")], 6)
        dialog.add_items([ListRow(pose.name, pose.key[:8], str(world["label"]) if world else "")])
        dialog.add_footnote(SPAWN_DELETION_FOOTNOTE, ruled=False)
        dialog.add_button("Cancel", False, default=True)
        dialog.add_button("Delete spawn", True, variant="danger-solid", icon="trash")
        if host.exec_card(dialog):
            self.transport.submit("spawn-delete", "spawns.delete", {"family_ids": [pose.family]})

    def import_spawn(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import spawn", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.transport.submit("spawn-edit:read", "local.read", {"path": path})

    def export_spawn(self) -> None:
        pose = self.poses[self.template] if self.template < len(self.poses) else None
        if pose is None or pose.default:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export spawn", "spawn.yaml", "YAML (*.yaml)"
        )
        if path:
            self.export_path = path
            self.transport.submit("spawn-edit:export", "spawns.export", {"id": pose.key})

    # replies

    def received(self, tag: str, result: Any) -> bool:
        info = self.world_info()
        if tag == "spawn-map" and info is not None and result.get("world_id") == info["id"]:
            try:
                self.world = WorldMap.from_reply(result)
            except (KeyError, TypeError, ValueError) as exc:
                self.map_card.map.set_message("Map unavailable: " + str(exc))
                return True
            self.map_card.map.set_world(self.world)
            self.update_pose()
        elif tag == "spawn-table":
            if self.pending is not None:
                self._listed(result)
        elif self.name_scan and tag == "spawn-names:" + self.name_scan[1]:
            self.scanned_names(result)
        elif tag == "spawn-edit:read":
            self.transport.submit(
                "spawn-edit:import",
                "spawns.import",
                {"yaml_text": result["text"], "robot_ids": [self.spawn_robot.currentData()]},
            )
        elif tag == "spawn-edit:export":
            self.transport.submit(
                "spawn-edit:written",
                "local.write",
                {"path": self.export_path, "text": result["yaml"]},
            )
        elif tag in ("spawn-edit:save", "spawn-edit:import"):
            self.notify.emit(
                f"Saved “{result['name']}” as a new spawn; submitted campaigns keep their poses",
                "ok",
                8000,
            )
            self.changed.emit()
            self.refresh()
            world = self.spawn_world.currentData()
            if world:
                if tag == "spawn-edit:save":
                    # The typed name has been used: the next save gets a fresh suggestion.
                    self.spawn_names.setdefault(world, []).append(str(result["name"]))
                    self.form.name.setModified(False)
                self.suggest_name(world)
        elif tag == "spawn-edit:written":
            self.notify.emit("Spawn YAML exported", "ok", 5000)
        elif tag == "spawn-delete":
            self.deleted.emit(result)
            names = sorted({str(r["name"]) for r in result.get("deleted", [])})
            self.notify.emit(
                f"Deleted {', '.join(f'“{n}”' for n in names)}; "
                f"{result.get('referenced_campaigns', 0)} submitted campaigns keep their poses",
                "ok",
                8000,
            )
            self.changed.emit()
            self.refresh()
        else:
            return tag == "spawn-map"  # a map reply for another world
        return True

    def failed(self, tag: str, message: str) -> bool:
        if tag == "spawn-table":
            self.pending = None
            if self.stale:
                self.stale = False
                self.refresh()
        elif self.name_scan and tag == "spawn-names:" + self.name_scan[1]:
            self.name_scan = None
            self.notify.emit(
                f"Could not read saved spawn names: {message}. Check the name before saving.",
                "bad",
                0,
            )
        elif tag == "spawn-edit:save":
            self.engine_problem = message
            self.update_pose()
        elif tag.startswith("spawn-edit:"):
            self.notify.emit(message, "bad", 0)
        elif tag == "spawn-delete":
            stale = stale_engine_message(message, "Unsupported library operation", "spawn deletion")
            self.notify.emit(stale or message, "bad", 0)
            self.refresh()
        elif tag == "spawn-map":
            self.map_card.map.set_message("Map unavailable: " + message)
        else:
            return tag.startswith("spawn-names:")  # a superseded name scan
        return True

    # layout

    def _layout(self) -> None:
        width, height = self.width(), self.height()
        self.head.setGeometry(0, 0, width, 36)
        body = height - 52
        side = self.side.resolve(440, 380, width - 16 - 400)
        room = max(1, int((body - 16 - 364 - 2 - 52 - 36) // 40))  # the form needs 364 px
        rows = max(1, min(len(self.poses), room))
        positions = self.positions_split.resolve(
            self.positions.natural_height(rows),
            self.positions.natural_height(1),
            body - 16 - 364,
        )
        place(self.positions, 0, 52, side, positions)
        place(self.form, 0, 52 + positions + 16, side, max(0, body - positions - 16))
        place(self.map_card, side + 16, 52, max(0, width - side - 16), body)
        self.side.place(side, 52, 16, body)
        self.positions_split.place(0, 52 + positions, side, 16)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()
