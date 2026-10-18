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

"""The Profiles page (13): robots, that robot's profiles with Default pinned, and the selected profile
compared with Default parameter by parameter.

The list shows every saved profile of the robot (oldest first, as the wizard lists them), read in pages of
250 and re-read every 2 s while the page is shown. Change counts come from each profile's snapshot
(``profiles.get``), fetched once per ID: an edit saves the profile under a new ID, so an ID's content never
changes, and the current and selected rows follow the profile's stable ``family_id``.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal
import uuid

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .contracts import PROFILE_DELETION_LIMIT
from .gui_cells import PAINTED_HEADER, PaintedColumn, PaintedTable
from .gui_controls import KeySelect, SearchField, Switch, UnderlineTabs
from .gui_dialogs import Consequences, ListRow, Run
from .gui_profile_editor import Mode
from .gui_profile_preview import (
    BASELINE,
    ComparedParameter,
    compared_parameters,
    editor_groups,
    profile_differences,
    signed_change,
    value_text,
)
from .gui_robots import photo_aspect, photo_image
from .gui_style import cell_text
from .gui_support import Transport
from .gui_theme import Theme, ThemeManager, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    PAGE_TITLE,
    Button,
    Card,
    CardHead,
    CardSplit,
    FlexRow,
    ModalHost,
    TextLine,
    TextStyle,
    draw_text,
    paint_scroll_handle,
    place,
    plural,
    rounded_path,
    wheel_step,
)
from .gui_wizard import search_text

ROBOT_ROW, PROFILE_ROW, MIN_PROFILE_ROW = 40, 52, 50
HEAD = 52
NAME_SELECTED = TextStyle(13, 600, 18)
META = TextStyle(12, 400, 18, tabular=True)
TITLE = TextStyle(18, 600, 24)
SECTION = TextStyle(12, 600, 18)
CHANGE = TextStyle(13, 400, 18, tabular=True)
LIST_PAGE, LIST_LIMIT = 250, 2000  # profiles per request; at most this many listed per robot
SNAPSHOTS = 64  # profile snapshots kept for the comparison (a saved ID's content never changes)
DEFAULT = "default"  # the current-row key of the robot's Default
UNCOUNTABLE = -1  # a profile whose snapshot could not be compared (shown as "—", not read again)


DELETION_FOOTNOTE = (
    "Submitted campaigns, their results and reports keep the exact parameters they recorded. "
    "Campaign YAML files that name a deleted profile's ID will no longer validate."
)


def deletion_question(count: int) -> str:
    """The confirmation's question for deleting saved profiles."""
    return f"Delete {count} profile{'s' if count != 1 else ''}? This cannot be undone."


def created_text(row: dict[str, Any]) -> str:
    """When the profile was created: "15 Oct 2026 · 22:43"."""
    created = row.get("created_ns")
    return cell_text("created_ns", created) if isinstance(created, int) else "—"


# ---------- robots ----------


@lru_cache(maxsize=8)
def thumb_glow(
    width: int, height: int, card: tuple[int, int, int], sunken: tuple[int, int, int]
) -> QtGui.QImage:
    """Dark theme's ``.rr .th``: ``radial-gradient(closest-side, rgba(255,255,255,.10), sunken)`` over the
    card, at device-pixel centres (premultiplied interpolation, as CSS gradients)."""
    ys = (np.arange(height, dtype=np.float64) + 0.5 - height / 2) / (height / 2)
    xs = (np.arange(width, dtype=np.float64) + 0.5 - width / 2) / (width / 2)
    t = np.clip(np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2), 0.0, 1.0)[:, :, None]
    alpha = 0.10 * (1 - t) + t
    color = 255 * 0.10 * (1 - t) + np.array(sunken, dtype=np.float64) * t
    result = color + (1 - alpha) * np.array(card, dtype=np.float64)
    data = np.empty((height, width, 4), dtype=np.uint8)
    data[:, :, :3] = np.clip(np.round(result), 0, 255).astype(np.uint8)
    data[:, :, 3] = 255
    return QtGui.QImage(
        np.ascontiguousarray(data).tobytes(),
        width,
        height,
        width * 4,
        QtGui.QImage.Format.Format_RGBA8888,
    ).copy()


def paint_thumbnail(
    painter: QtGui.QPainter, robot_id: str, x: float, y: float, theme: Theme, ratio: float
) -> None:
    """``.rr .th``: a 40×32 sunken box (radius 4) with the robot's photo 28 px high, centred and clipped."""
    box = QtCore.QRectF(snap(x), snap(y), 40, 32)
    path = rounded_path(box, 4)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setClipPath(path)
    if theme.name == "dark":
        card, sunken = theme.color("surface.card"), theme.color("surface.sunken")
        image = thumb_glow(
            round(40 * ratio),
            round(32 * ratio),
            (card.red(), card.green(), card.blue()),
            (sunken.red(), sunken.green(), sunken.blue()),
        )
        painter.drawImage(box, image)
    else:
        painter.fillRect(box, theme.color("surface.sunken"))
    aspect = photo_aspect(robot_id)
    if aspect is not None:
        found = photo_image(robot_id, box.x() + (40 - 28 * aspect) / 2, box.y() + 2, 28.0, ratio)
        if found is not None:
            image, origin = found
            painter.drawImage(origin, image)
    painter.restore()


class RobotsCard(Card):
    """``.robots`` (13): "Robots" with "Profiles" at the right, then one 40 px row per robot: thumbnail, name
    (600 when chosen), the number of its profiles (saved profiles and Default). The chosen robot has the
    selected fill and the ink bar."""

    chosen = QtCore.Signal(str)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.robots: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.current = ""
        self.head = CardHead(themes, "Robots", self)
        self.head.add(TextLine(themes, "Profiles", BODY, "text.secondary"))
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "none")
        self.setAccessibleName("Robots")

    def set_robots(self, robots: list[dict[str, Any]]) -> None:
        self.robots = list(robots)
        if self.current not in {robot["id"] for robot in robots}:
            self.current = robots[0]["id"] if robots else ""
        self.update()

    def set_count(self, robot_id: str, count: int) -> None:
        if self.counts.get(robot_id) != count:
            self.counts[robot_id] = count
            self.update()

    def set_current(self, robot_id: str) -> None:
        if robot_id != self.current:
            self.current = robot_id
            self.update()
            self.chosen.emit(robot_id)

    def natural_height(self) -> int:
        return 2 + HEAD + ROBOT_ROW * len(self.robots)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.head.setGeometry(1, 1, self.width() - 2, HEAD)

    def index_at(self, y: float) -> int:
        index = int((y - 1 - HEAD) // ROBOT_ROW)
        return index if y >= 1 + HEAD and 0 <= index < len(self.robots) else -1

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self.index_at(event.position().y())
        if index >= 0 and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.set_current(self.robots[index]["id"])

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        step = {QtCore.Qt.Key.Key_Up: -1, QtCore.Qt.Key.Key_Down: 1}.get(QtCore.Qt.Key(event.key()))
        identities = [robot["id"] for robot in self.robots]
        if step is None or not identities:
            super().keyPressEvent(event)
            return
        index = identities.index(self.current) if self.current in identities else 0
        self.set_current(identities[min(max(0, index + step), len(identities) - 1)])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        theme = self.theme
        width = self.width() - 2
        ratio = self.devicePixelRatioF()
        for index, robot in enumerate(self.robots):
            top = 1 + HEAD + index * ROBOT_ROW
            last = index == len(self.robots) - 1
            content = ROBOT_ROW - (0 if last else 1)
            current = robot["id"] == self.current
            if current:
                painter.fillRect(
                    QtCore.QRect(1, top, width, ROBOT_ROW - (1 if last else 0)),
                    theme.color("surface.selected"),
                )
                painter.fillRect(QtCore.QRectF(1, top, 2, content), theme.color("text.ink"))
            if not last:
                painter.fillRect(
                    QtCore.QRect(1, top + content, width, 1), theme.color("border.line")
                )
            paint_thumbnail(painter, robot["id"], 15, top + (content - 32) / 2, theme, ratio)
            line_top = top + (content - 18) / 2
            count = self.counts.get(robot["id"])
            text = "—" if count is None else f"{count:,}"
            right = 1 + width - 20
            x = right - BODY.advance(text)
            draw_text(painter, x, line_top, text, BODY, theme.color("text.secondary"))
            style = NAME_SELECTED if current else BODY
            name = style.elide(robot["label"], x - 12 - 67)
            draw_text(painter, 67, line_top, name, style, theme.color("text.ink"))


# ---------- profiles ----------


class ProfilesCard(Card):
    """``.plist`` (13): "<robot> profiles" with a search button (it opens a search field in the head), Default
    pinned first ("Robot-aware baseline", lock, a ``border.strong`` rule) and the saved profiles, oldest
    first: name (600 when current), when it was created ("15 Oct 2026 · 22:43") and the change count. Rows are 52 px; when
    they nearly fit (at least 50 px each) they share the card's height, otherwise the saved rows scroll under
    Default. A click selects one row; Ctrl toggles a row and Shift selects a range (the rows Delete acts on)."""

    current_changed = QtCore.Signal()
    selection_changed = QtCore.Signal()
    search_changed = QtCore.Signal(str)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.rows: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.current = DEFAULT
        self.selected: set[str] = set()
        self.anchor = DEFAULT
        self.offset = 0.0
        self.message = ""  # loading or error text under Default
        self.head = CardHead(themes, "Profiles", self)
        self.search_button = Button(themes, icon="search", variant="ghost", small=True)
        self.search_button.setToolTip("Search profiles")
        self.search_button.setAccessibleName("Search profiles")
        self.search_button.clicked.connect(self.toggle_search)
        self.head.add(self.search_button)
        self.search = SearchField(themes, "Search profiles", self.head)
        self.search.hide()
        self.search.textChanged.connect(self.search_changed.emit)
        self.search.installEventFilter(self)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "none")
        self.setAccessibleName("Profiles")

    # state

    def set_title(self, title: str) -> None:
        self.head.title.setText(title)

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        identities = {row["id"] for row in rows}
        # An edit saves the profile under a new ID: the current and selected rows follow its family.
        families: dict[str, str] = {
            row["family_id"]: str(row["id"])
            for row in rows
            if isinstance(row.get("family_id"), str)
        }
        previous = {row["id"]: row.get("family_id") for row in self.rows}

        def follow(identity: str) -> str:
            family = previous.get(identity)
            if identity in identities or not isinstance(family, str):
                return identity
            return families.get(family, identity)

        self.rows = rows
        self.selected = {follow(identity) for identity in self.selected} & identities
        self.anchor = follow(self.anchor)
        if self.current != DEFAULT and follow(self.current) != self.current:
            self.current = follow(self.current)
            self.current_changed.emit()
        if self.current != DEFAULT and self.current not in identities:
            self.current = DEFAULT
            self.current_changed.emit()
        self.scroll_to(self.offset)
        self.update()

    def set_count(self, identity: str, count: int) -> None:
        self.counts[identity] = count
        self.update()

    def set_message(self, message: str) -> None:
        if message != self.message:
            self.message = message
            self.update()

    def current_row(self) -> dict[str, Any] | None:
        return next((row for row in self.rows if row["id"] == self.current), None)

    def selected_rows(self) -> list[dict[str, Any]]:
        """The rows Delete acts on, in list order (the current row when nothing else is selected)."""
        chosen = self.selected or ({self.current} if self.current != DEFAULT else set())
        return [row for row in self.rows if row["id"] in chosen]

    def choose(self, key: str, modifiers: QtCore.Qt.KeyboardModifier) -> None:
        keys = [DEFAULT] + [row["id"] for row in self.rows]
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier and self.anchor in keys:
            low, high = sorted((keys.index(self.anchor), keys.index(key)))
            self.selected = {k for k in keys[low : high + 1] if k != DEFAULT}
        elif modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            if key != DEFAULT:
                self.selected ^= {key}
            self.anchor = key
        else:
            self.selected = set() if key == DEFAULT else {key}
            self.anchor = key
        changed = key != self.current
        self.current = key
        self.ensure_shown(key)
        self.update()
        self.selection_changed.emit()
        if changed:
            self.current_changed.emit()

    # search

    def toggle_search(self) -> None:
        if self.search.isVisible():
            self.close_search()
        else:
            self.open_search("")

    def open_search(self, text: str) -> None:
        self.head.title.hide()
        self.search.show()
        self._place_search()
        if text:
            self.search.setText(self.search.text() + text)
        self.search.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def close_search(self) -> None:
        self.search.clear()
        self.search.hide()
        self.head.title.show()
        self.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            watched is self.search
            and event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() == QtCore.Qt.Key.Key_Escape
        ):
            self.close_search()
            return True
        return super().eventFilter(watched, event)

    def _place_search(self) -> None:
        right = self.search_button.x() - 8
        self.search.setGeometry(20, (HEAD - 1 - 32) // 2, max(80, right - 20), 32)

    # geometry

    def row_height(self) -> float:
        room = self.height() - 2 - HEAD
        count = 1 + len(self.rows)
        if count * PROFILE_ROW <= room:
            return float(PROFILE_ROW)
        return room / count if room / count >= MIN_PROFILE_ROW else float(PROFILE_ROW)

    def list_top(self) -> float:
        """Top of the saved rows (under the pinned Default)."""
        return 1 + HEAD + self.row_height()

    def max_offset(self) -> float:
        visible = self.height() - 1 - self.list_top()
        return max(0.0, len(self.rows) * self.row_height() - visible)

    def scroll_to(self, offset: float) -> None:
        offset = min(max(0.0, offset), self.max_offset())
        if offset != self.offset:
            self.offset = offset
            self.update()

    def ensure_shown(self, key: str) -> None:
        index = next((i for i, row in enumerate(self.rows) if row["id"] == key), -1)
        if index < 0:
            return
        height = self.row_height()
        visible = self.height() - 1 - self.list_top()
        top = index * height
        if top < self.offset:
            self.scroll_to(top)
        elif top + height > self.offset + visible:
            self.scroll_to(top + height - visible)

    def key_at(self, y: float) -> str | None:
        height = self.row_height()
        if 1 + HEAD <= y < self.list_top():
            return DEFAULT
        if y < self.list_top() or y >= self.height() - 1:
            return None
        index = int((y - self.list_top() + self.offset) // height)
        return self.rows[index]["id"] if 0 <= index < len(self.rows) else None

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.head.setGeometry(1, 1, self.width() - 2, HEAD)
        if self.search.isVisible():
            self._place_search()
        self.scroll_to(self.offset)

    # input

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        key = self.key_at(event.position().y())
        if key is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.choose(key, event.modifiers())

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self.scroll_to(self.offset - wheel_step(event, self.row_height()).y())
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        keys = [DEFAULT] + [row["id"] for row in self.rows]
        step = {QtCore.Qt.Key.Key_Up: -1, QtCore.Qt.Key.Key_Down: 1}.get(QtCore.Qt.Key(event.key()))
        if step is not None:
            index = keys.index(self.current) if self.current in keys else 0
            self.choose(keys[min(max(0, index + step), len(keys) - 1)], event.modifiers())
            return
        text = search_text(event)
        if text:
            self.open_search(text)  # type to search
            return
        super().keyPressEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()  # the scroll handle shows while the pointer is over the list
        return super().event(event)

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        height = self.row_height()
        self._paint_row(painter, None, 1 + HEAD, height)
        top = self.list_top()
        painter.setClipRect(QtCore.QRectF(1, top, self.width() - 2, self.height() - 1 - top))
        if self.message:
            line_top = top + (PROFILE_ROW - 1 - 18) / 2
            draw_text(painter, 21, line_top, self.message, BODY, self.color("text.secondary"))
        for index, row in enumerate(self.rows):
            y = top + index * height - self.offset
            if y + height <= top or y >= self.height():
                continue
            self._paint_row(painter, row, y, height)
        painter.setClipping(False)
        visible = self.height() - 1 - top
        paint_scroll_handle(painter, self, top, visible, self.offset, len(self.rows) * height)

    def _paint_row(
        self, painter: QtGui.QPainter, row: dict[str, Any] | None, top: float, height: float
    ) -> None:
        theme = self.theme
        width = self.width() - 2
        key = DEFAULT if row is None else row["id"]
        current = key == self.current
        content = height - 1
        if current or key in self.selected:
            painter.fillRect(QtCore.QRectF(1, top, width, height), theme.color("surface.selected"))
            painter.fillRect(QtCore.QRectF(1, top, 2, content), theme.color("text.ink"))
        rule = "border.strong" if row is None else "border.line"
        painter.fillRect(QtCore.QRectF(1, top + content, width, 1), theme.color(rule))
        right = 1 + width - 20
        room: float
        if row is None:
            gui_icons.draw(
                painter,
                "lock",
                theme.color("text.tertiary"),
                QtCore.QPointF(right - 16, snap(top + (content - 16) / 2)),
            )
            room = right - 16 - 12
            name, meta = "Default", BASELINE
        else:
            count = self.counts.get(key)
            text = "" if count is None else "—" if count == UNCOUNTABLE else plural(count, "change")
            x = right - BODY.advance(text)
            draw_text(
                painter, x, top + (content - 18) / 2, text, BODY, theme.color("text.secondary")
            )
            room = x - 12 if text else right
            name, meta = str(row.get("name", "")), created_text(row)
        style = NAME_SELECTED if current else BODY_MEDIUM
        line_top = top + (content - 37) / 2
        draw_text(
            painter, 21, line_top, style.elide(name, room - 21), style, theme.color("text.ink")
        )
        draw_text(
            painter,
            21,
            line_top + 19,
            META.elide(meta, room - 21),
            META,
            theme.color("text.secondary"),
        )


# ---------- comparison ----------


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    kind: Literal["section", "parameter", "empty"]
    text: str = ""  # the section title or the empty message
    parameter: ComparedParameter | None = None
    group: str = ""  # blank when the row above has the same group


def comparison_rows(
    parameters: list[ComparedParameter], group: str | None, changed_only: bool, empty: str
) -> list[ComparisonRow]:
    """Changed parameters first, then those equal to Default (left out with ``changed_only``), each in a
    titled section; a group name only on the first row of its run."""
    shown = [p for p in parameters if group is None or p.group == group]
    rows: list[ComparisonRow] = []
    for title, members in (
        ("Changed", [p for p in shown if p.changed]),
        ("Same as Default", [] if changed_only else [p for p in shown if not p.changed]),
    ):
        if not members:
            continue
        rows.append(ComparisonRow("section", title))
        previous = ""
        for parameter in members:
            rows.append(
                ComparisonRow(
                    "parameter",
                    parameter=parameter,
                    group="" if parameter.group == previous else parameter.group,
                )
            )
            previous = parameter.group
    return rows or [ComparisonRow("empty", empty)]


class ComparisonTable(PaintedTable):
    """``.ctab .tbl`` (13): Parameter, Group, "<robot> Default", the profile, Change; 36 px header, 32 px section
    rows and 40 px parameter rows. Changed rows have the ink dot, the profile's value in 600 ink and the
    signed change; unchanged values are secondary."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(
            themes,
            self.make_columns("Default", "Profile"),
            parent,
            header_height=36,
            key="profiles.comparison",
        )
        self.rows: list[ComparisonRow] = []

    @staticmethod
    def make_columns(default: str, profile: str) -> list[PaintedColumn]:
        return [
            PaintedColumn("Parameter", None, padding=20),
            PaintedColumn("Group", 170),
            PaintedColumn(PAINTED_HEADER.elide(default, 106), 130, right=True),
            PaintedColumn(PAINTED_HEADER.elide(profile, 106), 130, right=True),
            PaintedColumn("Change", 96, padding_right=20, right=True),
        ]

    def set_rows(self, rows: list[ComparisonRow], default: str, profile: str) -> None:
        self.columns = self.make_columns(default, profile)
        self.header.update()
        self.rows = rows
        self.verticalScrollBar().setValue(0)
        self.rows_changed(len(rows))

    def row_height(self, index: int) -> int:
        return 32 if self.rows[index].kind == "section" else 40

    def tooltip_at(self, index: int, position: QtCore.QPoint) -> str:
        parameter = self.rows[index].parameter
        spans = self.column_spans()
        if parameter is None or position.x() >= spans[1][0]:
            return ""
        return "\n".join(part for part in (parameter.key, parameter.description) if part)

    def paint_row(self, painter: QtGui.QPainter, index: int, rect: QtCore.QRect) -> None:
        theme = self.themes.current
        row = self.rows[index]
        last = index == len(self.rows) - 1
        content = rect.height() - (0 if last else 1)
        if not last:
            painter.fillRect(
                QtCore.QRect(0, rect.bottom(), rect.width(), 1), theme.color("border.line")
            )
        line_top = rect.top() + (content - 18) / 2
        secondary = theme.color("text.secondary")
        if row.kind == "section":
            draw_text(painter, 20, line_top, row.text, SECTION, secondary)
            return
        if row.kind == "empty":
            draw_text(painter, 20, line_top, row.text, BODY, secondary)
            return
        parameter = row.parameter
        assert parameter is not None
        spans = self.column_spans()
        ink = theme.color("text.ink")
        changed = parameter.changed
        if changed:
            dot_top = BODY.baseline(line_top) + BODY.middle_offset(7)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(ink)
            painter.drawEllipse(QtCore.QRectF(6, dot_top, 6, 6))
            painter.restore()
        name_style = BODY_MEDIUM if changed else BODY
        room = spans[1][0] - 12 - 20
        unit_room = BODY.advance(" ") + 2 + BODY.advance(parameter.unit) if parameter.unit else 0.0
        name = name_style.elide(parameter.name, room - unit_room)
        x = 20 + draw_text(
            painter, 20, line_top, name, name_style, ink if changed else theme.color("text.body")
        )
        if parameter.unit:
            x += BODY.advance(" ") + 2
            draw_text(painter, x, line_top, parameter.unit, BODY, theme.color("text.tertiary"))
        left, width = spans[1]
        draw_text(
            painter,
            left + 12,
            line_top,
            BODY.elide(row.group, width - 24),
            BODY,
            theme.color("text.body"),
        )
        for column, text, style, color in (
            (2, value_text(parameter.default), BODY, secondary),
            (
                3,
                value_text(parameter.value),
                NAME_SELECTED if changed else BODY,
                ink if changed else secondary,
            ),
            (
                4,
                signed_change(parameter.default, parameter.value) if changed else "",
                CHANGE,
                secondary,
            ),
        ):
            left, width = spans[column]
            padding = self.columns[column].padding_right
            text = style.elide(text, width - 12 - padding)
            draw_text(
                painter, left + width - padding - style.advance(text), line_top, text, style, color
            )


class ComparisonCard(Card):
    """``.cmp`` (13): the profile's name (18/24 600) and description, Edit, Export
    profile and Delete…; the group tabs ("All parameters", then the editor's groups with change pills)
    and "Changed only"; the comparison table."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.name, self.description = "", ""
        self.empty = ""
        self.groups: list[str] = []
        self.parameters: list[ComparedParameter] = []
        self.robot_label = ""
        self.edit = Button(themes, "Edit", icon="pencil")
        self.export = Button(themes, "Export profile", icon="download")
        self.delete = Button(themes, "Delete…", icon="trash", variant="danger")
        self.buttons = FlexRow(self, gap=8, right=16)
        self.buttons.add_stretch()
        for button in (self.edit, self.export, self.delete):
            self.buttons.add(button)
        self.tabs = UnderlineTabs(themes, [("All parameters", None)], self, gap=20, rule=False)
        self.tabs.changed.connect(lambda _: self.show_rows())
        self.changed_only = Switch(
            themes, "Changed only", label_token="text.secondary", parent=self
        )
        self.changed_only.toggled.connect(lambda _: self.show_rows())
        self.group_select = KeySelect(themes, "Group", self)
        self.group_select.currentIndexChanged.connect(self._group_chosen)
        self.group_select.hide()
        self.table = ComparisonTable(themes, self)

    def show_profile(
        self,
        name: str,
        description: str,
        robot_label: str,
        groups: list[str],
        parameters: list[ComparedParameter],
        empty: str = "",
    ) -> None:
        """Show a profile; ``empty`` replaces the table while its parameters are unknown (loading or failed)."""
        self.empty = empty
        self.name, self.description = name, description
        self.robot_label = robot_label
        if groups != self.groups:
            self.groups = groups
            index = self.tabs.index()
            self.tabs.deleteLater()
            self.tabs = UnderlineTabs(
                self.themes,
                [("All parameters", None)] + [(g, None) for g in groups],
                self,
                gap=20,
                rule=False,
            )
            self.tabs.set_index(index if 0 <= index <= len(groups) else 0)
            self.tabs.changed.connect(lambda _: self.show_rows())
            self.tabs.show()
        self.parameters = parameters
        for index, group in enumerate(groups):
            self.tabs.set_changes(index + 1, sum(p.changed for p in parameters if p.group == group))
        self._fill_group_select()
        self._layout()
        self.show_rows()
        self.update()

    def show_rows(self) -> None:
        index = self.tabs.index()
        if self.group_select.currentIndex() != index:
            self.group_select.blockSignals(True)
            self.group_select.setCurrentIndex(max(0, index))
            self.group_select.blockSignals(False)
        group = self.groups[index - 1] if 1 <= index <= len(self.groups) else None
        rows = (
            [ComparisonRow("empty", self.empty)]
            if self.empty
            else comparison_rows(
                self.parameters,
                group,
                self.changed_only.isChecked(),
                f"No changes from {self.robot_label} Default",
            )
        )
        self.table.set_rows(rows, f"{self.robot_label} Default", self.name)

    def _layout(self) -> None:
        width = self.width()
        self.buttons.setGeometry(1, 21, width - 2, 32)
        switch = self.changed_only.natural_width()
        fits = 21 + self.tabs.natural_width() + 24 + switch + 21 <= width
        # A narrow card shows the groups in a select (with their change counts) instead of the tabs.
        self.tabs.setVisible(fits)
        self.group_select.setVisible(not fits)
        if fits:
            place(self.tabs, 21, 89, self.tabs.natural_width(), 39)
        else:
            place(self.group_select, 21, 89 + 3, min(280.0, width - 42 - switch - 24), 32)
        place(self.changed_only, width - 21 - switch, 99, switch, 18)
        self.table.setGeometry(1, 128, width - 2, max(0, self.height() - 129))

    def _fill_group_select(self) -> None:
        self.group_select.blockSignals(True)
        self.group_select.clear()
        self.group_select.addItem("All parameters", 0)
        for index, group in enumerate(self.groups, start=1):
            changes = self.tabs.tabs[index].changes
            self.group_select.addItem(f"{group} · {changes}" if changes else group, index)
        self.group_select.setCurrentIndex(max(0, self.tabs.index()))
        self.group_select.blockSignals(False)

    def _group_chosen(self, index: int) -> None:
        if index >= 0:
            self.tabs.set_index(index)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        theme = self.theme
        line = theme.color("border.line")
        painter.fillRect(QtCore.QRect(1, 88, self.width() - 2, 1), line)
        painter.fillRect(QtCore.QRect(1, 127, self.width() - 2, 1), line)
        right = self.edit.x() - 8
        name = TITLE.elide(self.name, right - 21)
        draw_text(painter, 21, 21, name, TITLE, theme.color("text.ink"))
        description = BODY.elide(self.description, right - 21)
        draw_text(painter, 21, 51, description, BODY, theme.color("text.body"))


# ---------- page ----------


class ProfilesPage(QtWidgets.QWidget):
    """The page: head (Import profile, New profile), robots and profiles cards at the left, the comparison at
    the right. It owns the list, snapshot and deletion requests (``received``/``failed`` return True for its
    replies); the window opens the editor (``open_requested``) and refreshes the libraries."""

    open_requested = QtCore.Signal(str)  # "new" or "edit"
    library_changed = QtCore.Signal()
    deleted = QtCore.Signal(object)  # the profiles.delete reply
    notify = QtCore.Signal(str, str, int)  # text, kind, timeout (ms)

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport = themes, transport
        self.catalog: dict[str, Any] = {}
        self.robot = ""
        self.library: dict[str, list[dict[str, Any]]] = {}  # the window's library-<robot> rows
        self.loaded: list[dict[str, Any]] = []
        self._page: list[dict[str, Any]] = []  # pages read so far for the request in flight
        self.pending: tuple[str, str, int] | None = (
            None  # robot, search, offset of the request in flight
        )
        self.stale = False
        self.snapshots: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.counting = ""  # the profile whose snapshot is being read for its count
        self.detail_tag = ""
        self.export_path = ""

        self.search_timer = QtCore.QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(250)
        self.search_timer.timeout.connect(self.refresh)
        self.head = FlexRow(self, gap=12)
        self.head.add(TextLine(themes, "Profiles", PAGE_TITLE, "text.ink"))
        self.head.add_stretch()
        self.import_button = Button(themes, "Import profile", icon="upload")
        self.import_button.clicked.connect(self.import_profile)
        self.head.add(self.import_button, gap=8)
        new = Button(themes, "New profile", icon="plus", variant="primary")
        new.clicked.connect(lambda: self.open_requested.emit("new"))
        self.head.add(new, gap=8)
        self.robots = RobotsCard(themes, self)
        self.robots.chosen.connect(self.choose_robot)
        self.profiles = ProfilesCard(themes, self)
        self.profiles.current_changed.connect(self.show_current)
        self.profiles.selection_changed.connect(self.update_actions)
        self.profiles.search_changed.connect(lambda _: self.search_timer.start())
        self.comparison = ComparisonCard(themes, self)
        # The gaps between the cards can be dragged (remembered; a double click restores the design).
        self.side = CardSplit(themes, self, "profiles/side", self._layout, across=True)
        self.robots_split = CardSplit(themes, self, "profiles/robots", self._layout, across=False)
        self.comparison.edit.clicked.connect(lambda: self.open_requested.emit("edit"))
        self.comparison.export.clicked.connect(self.export_profile)
        self.comparison.delete.clicked.connect(self.delete_profiles)
        self.sync = QtCore.QTimer(self)
        self.sync.setInterval(2000)
        self.sync.timeout.connect(lambda: self.refresh() if self.isVisible() else None)
        self.sync.start()

    # catalog and robots

    def set_catalog(self, catalog: dict[str, Any]) -> None:
        self.catalog = catalog
        self.robots.set_robots(catalog.get("robots", []))
        self._layout()
        self.choose_robot(self.robots.current)

    def robot_info(self) -> dict[str, Any] | None:
        return next((r for r in self.catalog.get("robots", []) if r["id"] == self.robot), None)

    def choose_robot(self, robot_id: str) -> None:
        if robot_id == self.robot and self.loaded:
            return
        self.robot = robot_id
        self.robots.current = robot_id
        self.robots.update()
        robot = self.robot_info()
        self.profiles.set_title(f"{robot['label']} profiles" if robot else "Profiles")
        self.loaded = []
        self.profiles.current, self.profiles.selected, self.profiles.offset = DEFAULT, set(), 0.0
        self.profiles.set_rows([])
        self.refresh()
        self.show_current()

    def set_library(self, robot_id: str, rows: list[dict[str, Any]], total: int) -> None:
        """The window's ``library-<robot>`` reply: the robot's count is its saved profiles and Default."""
        self.library[robot_id] = rows
        self.robots.set_count(robot_id, total + 1)

    # the list

    def refresh(self) -> None:
        if not self.robot:
            return
        if self.pending is not None:
            self.stale = True
            return
        self._request(0)

    def _request(self, offset: int) -> None:
        search = self.profiles.search.text().strip()
        self.pending = (self.robot, search, offset)
        params: dict[str, Any] = {
            "robot_id": self.robot,
            "offset": offset,
            "limit": LIST_PAGE,
            "summary": True,
        }
        if search:
            params["search"] = search
        self.transport.submit("profile-table", "profiles.list", params)

    def _list_received(self, result: dict[str, Any]) -> None:
        assert self.pending is not None
        robot, search, offset = self.pending
        self.pending = None
        if self.stale or robot != self.robot or search != self.profiles.search.text().strip():
            self.stale = False
            self._request(0)
            return
        rows = list(result.get("rows") or [])
        self._page = ([] if offset == 0 else self._page) + rows
        following = result.get("next_offset")
        if following is not None and len(self._page) < LIST_LIMIT:
            self._request(int(following))
            return
        rows = sorted(
            self._page,
            key=lambda row: (
                not isinstance(row.get("created_ns"), int),
                row.get("created_ns") or 0,
            ),
        )
        total = int(result.get("total") or len(rows))
        message = ""
        if not rows and search:
            message = "No matching profiles"
        elif total > len(rows):
            message = f"First {len(rows):,} of {total:,} profiles listed; search to find others"
        if [row["id"] for row in rows] != [
            row["id"] for row in self.loaded
        ] or message != self.profiles.message:
            self.loaded = rows
            self.profiles.set_rows(rows)
            self.profiles.set_message(message if not rows or total > len(rows) else "")
            self.update_actions()
        self._count_next()

    def _count_next(self) -> None:
        """Read one profile's snapshot whose change count is unknown (one request at a time)."""
        if self.counting:
            return
        missing = next(
            (row["id"] for row in self.loaded if row["id"] not in self.profiles.counts), None
        )
        if missing is None:
            return
        cached = self.snapshots.get(missing)
        if cached is not None:
            self._count(cached)
            self._count_next()
            return
        self.counting = missing
        self.transport.submit("profile-count:" + missing, "profiles.get", {"id": missing})

    def _count(self, snapshot: dict[str, Any]) -> None:
        robot = next(
            (r for r in self.catalog.get("robots", []) if r["id"] == snapshot.get("robot_id")), None
        )
        if robot is not None and isinstance(snapshot.get("effective"), dict):
            self.profiles.set_count(
                snapshot["id"],
                len(profile_differences(robot, self.catalog["parameters"], snapshot)),
            )
        else:
            self.profiles.set_count(str(snapshot.get("id")), UNCOUNTABLE)

    def _keep(self, snapshot: dict[str, Any]) -> None:
        self.snapshots[snapshot["id"]] = snapshot
        self.snapshots.move_to_end(snapshot["id"])
        while len(self.snapshots) > SNAPSHOTS:
            self.snapshots.popitem(last=False)
        self._count(snapshot)

    # the comparison

    def show_current(self) -> None:
        robot = self.robot_info()
        if robot is None or not robot.get("defaults"):
            return
        groups = editor_groups(self.catalog["parameters"], robot["id"])
        row = self.profiles.current_row()
        if row is None:
            parameters = compared_parameters(robot, self.catalog["parameters"], None)
            self.comparison.show_profile("Default", BASELINE, robot["label"], groups, parameters)
        else:
            snapshot = self.snapshots.get(row["id"])
            if snapshot is None:
                self.detail_tag = "profile-detail-" + uuid.uuid4().hex
                self.transport.submit(self.detail_tag, "profiles.get", {"id": row["id"]})
                self.comparison.show_profile(
                    row["name"],
                    "",
                    robot["label"],
                    groups,
                    [],
                    "Loading parameters…",
                )
            else:
                parameters = compared_parameters(robot, self.catalog["parameters"], snapshot)
                description = str(snapshot.get("description") or "")
                self.comparison.show_profile(
                    snapshot["name"],
                    description,
                    robot["label"],
                    groups,
                    parameters,
                )
        self.update_actions()

    def update_actions(self) -> None:
        saved = self.profiles.current_row() is not None
        self.comparison.edit.setEnabled(saved)
        self.comparison.edit.setToolTip(
            "" if saved else "Default is not editable; New profile starts from its values"
        )
        self.comparison.export.setEnabled(saved)
        self.comparison.export.setToolTip("" if saved else "Default is not exported")
        families = {row["family_id"] for row in self.profiles.selected_rows()}
        self.comparison.delete.setEnabled(bool(families))
        self.comparison.delete.setToolTip(
            f"Delete the {len(families)} selected profiles"
            if len(families) > 1
            else ""
            if families
            else "Default is not deletable"
        )

    # actions

    def delete_profiles(self) -> None:
        records = self.profiles.selected_rows()
        if not records:
            return
        # Deleted by their stable keys (``family_id``), one entry per profile.
        families = {record["family_id"]: record for record in records}
        if len(families) > PROFILE_DELETION_LIMIT:
            self.notify.emit(
                f"Select at most {PROFILE_DELETION_LIMIT} profiles to delete at once", "warn", 8000
            )
            return
        host = self.window().findChild(ModalHost)
        if host is None:
            return
        robots = {robot["id"]: robot["label"] for robot in self.catalog.get("robots", [])}
        count = len(families)
        dialog = Consequences(self.themes, "Delete profiles")
        dialog.add_sentence([Run(deletion_question(count))], 6)
        dialog.add_items(
            [
                ListRow(row["name"], str(row["id"])[:8], robots.get(row["owner"], row["owner"]))
                for row in families.values()
            ]
        )
        dialog.add_footnote(DELETION_FOOTNOTE, ruled=False)
        dialog.add_button("Cancel", False, default=True)
        dialog.add_button(
            f"Delete {plural(count, 'profile')}", True, variant="danger-solid", icon="trash"
        )
        if host.exec_card(dialog):
            self.transport.submit(
                "profile-delete", "profiles.delete", {"family_ids": list(families)}
            )

    def import_profile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import suite profile", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.transport.submit("profile-read", "local.read", {"path": path})

    def export_profile(self) -> None:
        row = self.profiles.current_row()
        if row is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export suite profile", "suite-profile.yaml", "YAML (*.yaml)"
        )
        if path:
            self.export_path = path
            self.transport.submit(
                "profile-export", "profiles.export", {"id": row["id"], "format": "suite"}
            )

    def open_request(self, operation: str) -> Mode | None:
        """Ask for what the editor opens with: Default's values for New (``editor-default``), the saved
        profile for Edit (``profile-open``). Returns the editor's mode (None: Default is not editable)."""
        if operation == "new":
            self.transport.submit("editor-default", "parameters.resolve", {"robot_id": self.robot})
            return "new"
        identity = self.current_id()
        if identity is None:
            return None
        self.transport.submit("profile-open", "profiles.get", {"id": identity})
        return "edit"

    def current_id(self) -> str | None:
        row = self.profiles.current_row()
        return None if row is None else str(row["id"])

    # replies

    def received(self, tag: str, result: Any) -> bool:
        if tag == "profile-table":
            if self.pending is not None:
                self._list_received(result)
            return True
        if tag.startswith("profile-count:"):
            identity = tag.removeprefix("profile-count:")
            if identity == self.counting:
                self.counting = ""
                if isinstance(result, dict) and result.get("id") == identity:
                    self._keep(result)
                else:
                    self.profiles.set_count(identity, UNCOUNTABLE)  # not retried
            self._count_next()
            return True
        if tag.startswith("profile-detail-"):
            if tag == self.detail_tag and isinstance(result, dict):
                self._keep(result)
                if result.get("id") == self.profiles.current:
                    self.show_current()
            return True
        if tag == "profile-delete":
            self.profiles.selected.clear()
            self.deleted.emit(result)
            count = len(result["family_ids"])
            self.notify.emit(
                f"Deleted {count} profile{'s' if count != 1 else ''}; "
                f"{result['referenced_campaigns']} campaigns keep their recorded parameters",
                "ok",
                10000,
            )
            self.library_changed.emit()
            return True
        if tag == "profile-read":
            self.transport.submit(
                "profile-import", "profiles.import", {"yaml_text": result["text"]}
            )
            return True
        if tag == "profile-import":
            self.library_changed.emit()
            self.notify.emit("Library saved", "ok", 6000)
            return True
        if tag == "profile-export":
            self.transport.submit(
                "file-saved", "local.write", {"path": self.export_path, "text": result["yaml"]}
            )
            return True
        if tag == "file-saved":
            self.notify.emit("Export saved", "ok", 5000)
            return True
        return False

    def failed(self, tag: str, error: str) -> bool:
        if tag == "profile-table":
            self.pending = None
            if self.stale:
                self.stale = False
                self._request(0)
            else:
                self.profiles.set_message("Cannot list profiles: " + error)
            return True
        if tag.startswith("profile-count:"):
            if tag.removeprefix("profile-count:") == self.counting:
                self.counting = ""
            return True  # the count stays empty; the next refresh tries again
        if tag.startswith("profile-detail-"):
            if tag == self.detail_tag:
                self.comparison.empty = "Parameters unavailable: " + error
                self.comparison.show_rows()
            return True
        return False

    # layout

    def _layout(self) -> None:
        width, height = self.width(), self.height()
        self.head.setGeometry(0, 0, width, 36)
        side = self.side.resolve(340, 280, width - 16 - 520)
        body = height - 52
        robots = self.robots_split.resolve(
            self.robots.natural_height(),
            2 + HEAD + ROBOT_ROW,
            body - 16 - (2 + HEAD + 2 * PROFILE_ROW),
        )
        place(self.robots, 0, 52, side, robots)
        top = 52 + robots + 16
        place(self.profiles, 0, top, side, max(HEAD + 2, height - top))
        place(self.comparison, side + 16, 52, max(0, width - side - 16), max(0, body))
        self.side.place(side, 52, 16, body)
        self.robots_split.place(0, 52 + robots, side, 16)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()
