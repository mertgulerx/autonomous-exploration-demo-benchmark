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

"""New campaign step 2 (05, 06; HANDOFF §8.5-§8.6): the world cards and the starting-poses popup.

``WorldSelector`` keeps the selection rules: Bookstore starts ticked (else the first world), the selection is never
empty (the last ticked world cannot be cleared), each ticked world keeps at
least one starting pose, every ticked world has a time limit (simulation seconds from the first movement; 0 is
unlimited), and the popup lists the first 250 saved spawns (a footer row says so when there are more). A saved
spawn gets a new ID with each edit; the selection follows it through the spawn's stable ``family_id``.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .contracts import DEFAULT_LIMIT
from .gui_cells import (
    CELL_ROLE,
    CHECK_COLUMN,
    Cell,
    CellDelegate,
    ColumnStyle,
    TableHeader,
    TableLook,
    style_table,
)
from .gui_controls import DoubleSpinBox
from .gui_maps import MapStage, SpawnMark, WorldMap
from .gui_menus import show_tooltip
from .gui_support import Transport
from .gui_theme import Mark, ThemeManager, paint_check, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    BODY_TABULAR,
    PopupCard,
    PopupHost,
    TextLine,
    TextStyle,
    css_rect,
    draw_text,
    place,
    rounded_path,
)
from .gui_wizard import (
    CardGrid,
    PopupButton,
    PopupHead,
    Reference,
    ToggleCard,
    search_text,
    seconds,
    toggle_reference,
    toggle_shown,
)

NAME = TextStyle(15, 600, 18)
AREA = TextStyle(12, 400, 18)
MORE_NOTE = "First 250 saved spawns shown. The Worlds & Spawns page lists all of them."
LAST_WORLD = "Select another world before deselecting this one."
LAST_SPAWN = "At least one starting position is required. Select another first."
LIMIT_TIP = "Simulation time from the first movement; 0 means no limit."

DEFAULT_WORLD = "bookstore"  # ticked when a new campaign starts (as MOGI on step 1)


def spawn_key(world: str, reference: Reference) -> str:
    if isinstance(reference, dict):
        return str(reference["id"])
    return "default-" + world if reference in ("Default", "default") else str(reference)


def signed(value: float, digits: int) -> str:
    """Fixed decimals with a true minus sign (U+2212); a value that rounds to zero has no sign."""
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


def world_size(world: dict[str, Any]) -> str:
    """The reference grid's extent: "15.7 × 14.8 m"."""
    try:
        width = float(world["width"]) * float(world["resolution"])
        height = float(world["height"]) * float(world["resolution"])
    except (KeyError, TypeError, ValueError):
        return "—"
    return f"{width:g} × {height:g} m" if math.isfinite(width * height) else "—"


def world_area(world: dict[str, Any]) -> str:
    """The coverage target and its cells at the grid resolution: "Reference area 170.4 m² · 68,141 cells"."""
    area = (world.get("reference") or {}).get("target_area_m2")
    resolution = world.get("resolution")
    if (
        isinstance(area, bool)
        or not isinstance(area, (int, float))
        or not isinstance(resolution, (int, float))
        or not math.isfinite(area)
        or not resolution > 0
    ):
        return "Reference area —"
    return f"Reference area {area:.1f} m² · {round(area / resolution**2):,} cells"


class TimeLimitBox(DoubleSpinBox):
    """The world's time limit in simulation seconds (0: unlimited), shown as stored ("1800", "1800.5");
    Enter commits the value and stays in the box."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        # ``.wc .r .v`` also reaches the spin box's value: 13 px 500 (text.tertiary when the world is out).
        super().__init__(themes, "s", parent, weight=500)
        self.setLocale(QtCore.QLocale.c())
        self.setDecimals(6)
        self.setRange(0, 1e12)
        self.setSingleStep(60)
        self.setFixedSize(112, 32)
        # The box hands its colours to its edit, and the edit's own rules apply after that: set both.
        self.setProperty("muted", True)
        self.style().unpolish(self)
        self.style().polish(self)
        themes.changed.connect(self._restyle_value)
        self._restyle_value()

    def _restyle_value(self, *_: object) -> None:
        muted = self.themes.current.qss("text.tertiary")
        self.lineEdit().setStyleSheet(f"QLineEdit:disabled {{ color: {muted}; }}")

    def textFromValue(self, value: float) -> str:
        return seconds(value)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.interpretText()
            event.accept()
            return
        super().keyPressEvent(event)


# ---------- world cards ----------


class WorldCard(ToggleCard):
    """``.wc`` (05): check, name (15 px 600) and size, the reference area, the hatched map with the chosen poses,
    then the "Starting poses" row (names, Spawns ⌄) and the "Time limit" row (spin box with "s"). Selected:
    2 px ink border; hover: ``border.strong``; a world left out shows its name muted, "Not included", the
    excluded map colours and disabled controls."""

    # Offsets from the card's CSS box: 1 px border plus padding 16 20 6; the rows block (two 52 px rows) ends
    # at the bottom padding, and the map takes the height between the area line and the rows.
    PAD, HEAD_TOP, AREA_TOP, STAGE_TOP, ROWS, KEY_WIDTH = 21, 17, 45, 77, 111, 96

    def __init__(
        self, themes: ThemeManager, world: dict[str, Any], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.world = deepcopy(world)
        self.world_id, self.label = str(world["id"]), str(world["label"])
        self.size_text, self.area_text = world_size(world), world_area(world)
        self.checked = self.guarded = False
        self.names = "Default"
        # Engine errors, shown as the tooltips of the map and of the poses.
        self.map_error = self.pose_error = ""
        self.stage = MapStage(themes, parent=self)
        self.stage.setAccessibleName(self.label + " map and starting positions")
        self.spawns_button = PopupButton(themes, "Spawns", self)
        self.spawns_button.setAccessibleName("Select starting positions for " + self.label)
        self.time_limit = TimeLimitBox(themes, self)
        self.time_limit.setValue(float(world.get("default_time_limit", DEFAULT_LIMIT)))
        self.time_limit.setAccessibleName(self.label + " time limit in simulation seconds")
        self.time_limit.setToolTip(LIMIT_TIP)
        self._sync()

    def set_state(self, checked: bool, guarded: bool, names: str) -> None:
        if (checked, guarded, names) != (self.checked, self.guarded, self.names):
            self.checked, self.guarded, self.names = checked, guarded, names
            self._sync()

    def _sync(self) -> None:
        self.set_selected(self.checked)
        self.stage.set_excluded(not self.checked)
        self.spawns_button.setEnabled(self.checked)
        self.time_limit.setEnabled(self.checked)
        state = "selected" if self.checked else "not selected"
        self.setAccessibleName(f"Select world: {self.label}, {state}")
        self.setAccessibleDescription(
            f"{self.size_text}. {self.area_text}. Starting poses: {self.names}"
        )
        self.update()

    def rows_top(self) -> float:
        return css_rect(self).bottom() - self.ROWS

    def _place_children(self) -> None:
        box = css_rect(self)
        x, inner, rows = box.x() + self.PAD, box.width() - 2 * self.PAD, self.rows_top()
        stage_top = box.y() + self.STAGE_TOP
        place(self.stage, x, stage_top, inner, max(0.0, rows - 8 - stage_top))
        button_width = self.spawns_button.natural_width()
        place(self.spawns_button, x + inner - button_width, rows + 11.5, button_width, 28)
        place(self.time_limit, x + inner - 112, rows + 52 + 10, 112, 32)

    def css_changed(self) -> None:
        self._place_children()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_children()

    # tooltips

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            show_tooltip(self, event.globalPos(), self.tooltip_at(event.pos()))
            return True
        return super().event(event)

    def tooltip_at(self, position: QtCore.QPoint) -> str:
        box, rows, y = css_rect(self), self.rows_top(), position.y()
        if y < box.y() + self.AREA_TOP:
            return LAST_WORLD if self.guarded else ""
        if y < rows:
            return self.map_error
        if rows + 52 <= y and position.x() < box.x() + self.PAD + self.KEY_WIDTH:
            return LIMIT_TIP
        if rows <= y < rows + 52 and self.checked:
            return self.pose_error or self.names
        return ""

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)  # fill and border (2 px ink when selected)
        painter = QtGui.QPainter(self)
        theme = self.theme
        if not self.checked and self.underMouse():
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtGui.QPen(theme.color("border.strong"), 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            outline = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.drawPath(rounded_path(outline, 7.5))
        box = css_rect(self)
        top = box.y()
        x, inner = box.x() + self.PAD, box.width() - 2 * self.PAD
        secondary = theme.color("text.secondary")
        muted = theme.color("text.tertiary")
        # Head: check, name and the size at the right.
        y = top + self.HEAD_TOP
        mark: Mark = "on" if self.checked else "off"
        paint_check(painter, QtCore.QRectF(x, y + 4, 16, 16), theme, mark)
        size_width = BODY_TABULAR.advance(self.size_text)
        size_color = secondary if self.checked else muted
        draw_text(painter, x + inner - size_width, y + 3, self.size_text, BODY_TABULAR, size_color)
        name = NAME.elide(self.label, max(0.0, inner - 26 - size_width - 10))
        draw_text(
            painter, x + 26, y + 3, name, NAME, theme.color("text.ink") if self.checked else muted
        )
        draw_text(painter, x, top + self.AREA_TOP, self.area_text, AREA, secondary)
        # Rows: starting poses (with Spawns), a rule, then the time limit.
        rows = self.rows_top()
        rule = QtCore.QRect(snap(x), snap(rows + 51), snap(x + inner) - snap(x), 1)
        painter.fillRect(rule, theme.color("border.line"))
        draw_text(painter, x, rows + 16.5, "Starting poses", BODY, secondary)
        value_x = x + self.KEY_WIDTH + 12
        room = x + inner - self.spawns_button.natural_width() - 12 - value_x
        value = BODY_MEDIUM.elide(self.names if self.checked else "Not included", max(0.0, room))
        value_color = theme.color("text.ink") if self.checked else muted
        draw_text(painter, value_x, rows + 16.5, value, BODY_MEDIUM, value_color)
        draw_text(painter, x, rows + 52 + 17, "Time limit", BODY, secondary)


# ---------- spawns popup ----------


@dataclass(frozen=True, slots=True)
class SpawnEntry:
    """One row of the starting-poses popup."""

    reference: Reference
    key: str
    name: str
    pose: dict[str, Any] | None
    detail: str  # "· world file" after the default's name
    search: str


class SpawnModel(QtCore.QAbstractTableModel):
    """The poses shown in the popup (after the search), with their check states."""

    HEADERS = ("", "Starting pose", "X · m", "Y · m", "Heading")

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[SpawnEntry] = []
        self.checked: set[str] = set()

    def set_entries(self, entries: list[SpawnEntry], checked: set[str]) -> None:
        self.beginResetModel()
        self.entries, self.checked = entries, set(checked)
        self.endResetModel()

    def set_checked(self, checked: set[str]) -> None:
        self.checked = set(checked)
        if self.entries:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.entries) - 1, 0))

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
            and 0 <= section < len(self.HEADERS)
        ):
            return self.HEADERS[section]
        return None

    def flags(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> QtCore.Qt.ItemFlag:
        return QtCore.Qt.ItemFlag.ItemIsEnabled

    def cell(self, entry: SpawnEntry, column: int) -> Cell:
        pose = entry.pose
        if column == 0:
            checked = entry.key in self.checked
            sole = checked and len(self.checked) == 1
            state = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
            return Cell(kind="check", check=state, tooltip=LAST_SPAWN if sole else "")
        if column == 1:
            return Cell(entry.name, token="text.ink", note=entry.detail, note_kind="detail")
        if column in (2, 3):
            value = pose.get("x" if column == 2 else "y") if pose else None
            return Cell(signed(float(value), 2) if value is not None else "—", "number")
        yaw = pose.get("yaw") if pose else None
        return Cell(signed(math.degrees(float(yaw)), 1) + "°" if yaw is not None else "—", "number")

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or index.row() >= len(self.entries):
            return None
        cell = self.cell(self.entries[index.row()], index.column())
        if role == CELL_ROLE:
            return cell
        if role == QtCore.Qt.ItemDataRole.DisplayRole and cell.kind != "check":
            return (cell.text + " " + cell.note).strip()
        return None


class SpawnDelegate(CellDelegate):
    """``.sp-tbl``: no selected rows; the hovered row is the inspected pose (``hovered``)."""

    hovered = QtCore.Signal(int)

    def row_selected(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, selected: bool
    ) -> bool:
        return False

    def _set_hover(self, row: int) -> None:
        changed = row != self.hover_row
        super()._set_hover(row)
        if changed:
            self.hovered.emit(row)


SPAWN_LOOK = TableLook(
    key="worlds.spawns",
    columns=[
        CHECK_COLUMN,
        ColumnStyle("text"),
        ColumnStyle("number", width=80),
        ColumnStyle("number", width=80),
        ColumnStyle("number", padding_right=20, width=100),
    ],
)


class SpawnsPopup(PopupCard):
    """``.pp`` (06): the head (56 px: title, search, "Done · N selected"), the poses table (580 px: check,
    name, X, Y, heading; 40 px rows) and the map with every pose (chosen ones ink, the hovered one larger,
    labels for both); under a rule, a 38 px footer row notes the 250-spawn limit when the library has more
    (D2)."""

    changed = QtCore.Signal(object)
    finished = QtCore.Signal()

    HEAD, TABLE, NOTE = 56, 580, 38  # HEAD and TABLE include their rules; NOTE sits under its own

    def __init__(
        self, themes: ThemeManager, world: dict[str, Any], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.world_id = str(world["id"])
        self.references: list[Reference] = ["Default"]
        self.options: list[SpawnEntry] = []
        self.has_more = False
        self.inspected = -1
        self.setAccessibleName("Starting poses · " + str(world["label"]))
        self.head = PopupHead(
            themes, "Starting poses · " + str(world["label"]), "Search poses", self
        )
        self.done = self.head.done
        self.search = self.head.search
        self.model = SpawnModel()
        self.view = QtWidgets.QTableView(self)
        self.view.setModel(self.model)
        delegate = style_table(self.view, themes, SPAWN_LOOK, delegate_class=SpawnDelegate)
        assert isinstance(delegate, SpawnDelegate)
        self.delegate = delegate
        header = self.view.horizontalHeader()
        assert isinstance(header, TableHeader)
        self.header = header
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.view.setAccessibleName("Starting poses")
        self.stage = MapStage(themes, pad=10, ring=False, label_size=10, parent=self)
        self.note = TextLine(themes, MORE_NOTE, BODY, "text.secondary", self)
        self.note.hide()
        self.done.clicked.connect(self.finished.emit)
        self.search.textChanged.connect(self.filter_rows)
        self.header.check_toggled.connect(self.toggle_shown)
        self.delegate.hovered.connect(self.inspect)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        self.hide()

    # contents

    def set_has_more(self, has_more: bool) -> None:
        if has_more != self.has_more:
            self.has_more = has_more
            self.note.setVisible(has_more)
            self._layout()
            self.update()

    def refresh(self, entries: list[SpawnEntry], selected: list[Reference]) -> None:
        self.options = list(entries)
        self.references = deepcopy(selected) or ["Default"]
        self.filter_rows(self.search.text())

    def _key(self, reference: Reference) -> str:
        return spawn_key(self.world_id, reference)

    def keys(self) -> set[str]:
        return {self._key(reference) for reference in self.references}

    def filter_rows(self, query: str) -> None:
        """Show the poses matching ``query``; the inspected and the current (keyboard) row stay on their
        poses across the model reset."""
        needle = query.casefold().strip()
        shown = [entry for entry in self.options if not needle or needle in entry.search]
        inspected = self._inspected_key()
        current = self.view.currentIndex()
        entries = self.model.entries
        current_key = entries[current.row()].key if 0 <= current.row() < len(entries) else None
        self.model.set_entries(shown, self.keys())
        self.delegate.hover_row = -1
        self.inspected = next((i for i, e in enumerate(shown) if e.key == inspected), -1)
        row = next((i for i, e in enumerate(shown) if e.key == current_key), -1)
        if row >= 0:
            self.view.setCurrentIndex(self.model.index(row, current.column()))
        self._sync()

    def _inspected_key(self) -> str | None:
        entries = self.model.entries
        return entries[self.inspected].key if 0 <= self.inspected < len(entries) else None

    def inspect(self, row: int) -> None:
        """Show the pose of ``row`` as the inspected one (hovered rows; −1: none)."""
        if row != self.delegate.hover_row:
            self.delegate.hover_row = row
            self.view.viewport().update()
        self.inspected = row
        self._sync_map()

    def _sync(self) -> None:
        keys = self.keys()
        self.model.set_checked(keys)
        shown = self.model.entries
        ticked = sum(entry.key in keys for entry in shown)
        self.header.set_check_state(
            QtCore.Qt.CheckState.Checked
            if shown and ticked == len(shown)
            else QtCore.Qt.CheckState.PartiallyChecked
            if ticked
            else QtCore.Qt.CheckState.Unchecked
        )
        self.head.set_count(len(self.references))
        self._sync_map()

    def _sync_map(self) -> None:
        keys = self.keys()
        inspected = self._inspected_key()
        marks = []
        for entry in self.options:
            if entry.pose is None:
                continue
            picked, selected = entry.key in keys, entry.key == inspected
            pose = entry.pose
            marks.append(
                SpawnMark(
                    entry.name,
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["yaw"]),
                    picked=picked,
                    selected=selected,
                    labelled=picked or selected,
                )
            )
        self.stage.set_marks(marks)

    # actions

    def toggle(self, row: int) -> None:
        if not 0 <= row < len(self.model.entries):
            return
        selected = toggle_reference(self.references, self.model.entries[row].reference, self._key)
        if selected is not None:  # None: the last starting pose stays selected
            self._set_selected(selected)

    def toggle_shown(self) -> None:
        shown = [entry.reference for entry in self.model.entries]
        if shown:
            self._set_selected(toggle_shown(self.references, shown, self._key))

    def _set_selected(self, selected: list[Reference]) -> None:
        self.references = selected
        self._sync()
        self.changed.emit(deepcopy(self.references))

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if (
            watched is self.view.viewport()
            and kind == QtCore.QEvent.Type.MouseButtonRelease
            and isinstance(event, QtGui.QMouseEvent)
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and not self.delegate.dragging  # a text selection, not a click
        ):
            index = self.view.indexAt(event.position().toPoint())
            if index.isValid():
                self.view.setCurrentIndex(index)
                self.toggle(index.row())
        elif watched is self.view and kind == QtCore.QEvent.Type.KeyPress:
            assert isinstance(event, QtGui.QKeyEvent)
            current = self.view.currentIndex()
            if event.key() == QtCore.Qt.Key.Key_Space and current.isValid():
                self.toggle(current.row())
                return True
            text = search_text(event)
            if text:
                self.search.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                self.search.insert(text)  # type to search
                return True
        elif watched is self.view and kind == QtCore.QEvent.Type.KeyRelease:
            current = self.view.currentIndex()
            if current.isValid() and current.row() != self.inspected:
                self.inspect(current.row())  # the arrow keys inspect the current row
        return super().eventFilter(watched, event)

    # layout

    def body_height(self) -> int:
        """The table and map height (above the body's rule when the footer shows)."""
        return self.height() - 1 - self.HEAD - (1 + self.NOTE if self.has_more else 0) - 1

    @classmethod
    def popup_height(cls, has_more: bool) -> int:
        return 1 + cls.HEAD + 235 + (1 + cls.NOTE if has_more else 0) + 1

    def _layout(self) -> None:
        width, top, body = self.width(), 1 + self.HEAD, self.body_height()
        self.head.setGeometry(1, 1, width - 2, self.HEAD - 1)
        self.view.setGeometry(1, top, self.TABLE - 1, body)
        self.stage.setGeometry(self.TABLE + 1, top, width - self.TABLE - 2, body)
        self.note.setGeometry(21, top + body + 1, width - 42, self.NOTE)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        line = self.color("border.line")
        width, body = self.width(), self.body_height()
        painter.fillRect(QtCore.QRect(1, self.HEAD, width - 2, 1), line)
        painter.fillRect(QtCore.QRect(self.TABLE, 1 + self.HEAD, 1, body), line)
        if self.has_more:
            painter.fillRect(QtCore.QRect(1, 1 + self.HEAD + body, width - 2, 1), line)


# ---------- selector ----------


def world_columns(width: float) -> int:
    return 3 if width >= 900 else 2 if width >= 590 else 1


class WorldSelector(QtWidgets.QWidget):
    """Step 2: the world card grid and one starting-poses popup per world (on the window's PopupHost)."""

    changed = QtCore.Signal()
    map_loaded = QtCore.Signal(str, object)  # world ID and its WorldMap, or None when it failed

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes = themes
        self.transport = transport
        self._worlds: dict[str, dict[str, Any]] = {}
        self._cards: dict[str, WorldCard] = {}
        self._selected: set[str] = set()
        self._references: dict[str, list[Reference]] = {}
        self._saved: dict[str, dict[str, dict[str, Any]]] = {}
        self._poses: dict[tuple[str, str], dict[str, Any]] = {}
        self._maps: dict[str, WorldMap] = {}
        self._map_errors: dict[str, str] = {}
        self._pending: dict[str, tuple[str, str | None]] = {}
        self._popups: dict[str, SpawnsPopup] = {}
        self._has_more: dict[str, bool] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.grid = CardGrid(world_columns, 520)
        outer.addWidget(self.grid)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    def set_catalog(self, worlds: list[dict[str, Any]]) -> None:
        limits = {world: card.time_limit.value() for world, card in self._cards.items()}
        accepted = {str(world["id"]): deepcopy(world) for world in worlds}
        for identity in tuple(self._cards):
            if identity not in accepted or self._worlds.get(identity) != accepted[identity]:
                self._cards.pop(identity).deleteLater()
                self._drop_popup(identity)
                self._maps.pop(identity, None)
                self._map_errors.pop(identity, None)
                self._pending = {
                    tag: pending for tag, pending in self._pending.items() if pending[0] != identity
                }
                self._poses = {key: pose for key, pose in self._poses.items() if key[0] != identity}
        self._worlds = accepted
        self._selected.intersection_update(accepted)
        if not self._selected and accepted:  # never empty: Bookstore, else the first world
            self._selected.add(DEFAULT_WORLD if DEFAULT_WORLD in accepted else next(iter(accepted)))
        for identity, world in accepted.items():
            self._references.setdefault(identity, ["Default"])
            if identity not in self._cards:
                card = WorldCard(self.themes, world)
                if identity in limits:
                    card.time_limit.setValue(limits[identity])
                card.time_limit.valueChanged.connect(lambda _value: self.changed.emit())
                card.toggled.connect(
                    lambda wid=identity: self.select_world(wid, wid not in self._selected)
                )
                card.spawns_button.clicked.connect(
                    lambda checked=False, wid=identity: self.show_spawns(wid)
                )
                self._cards[identity] = card
            self._sync(identity)
            if self.isVisible() and identity not in self._maps:
                self._request(identity)
        self.grid.set_cards([self._cards[identity] for identity in self._worlds])

    def selections(self) -> dict[str, list[Reference]]:
        return {
            world: deepcopy(self._references[world])
            for world in self._worlds
            if world in self._selected
        }

    def set_selections(self, selected: dict[str, list[Reference]]) -> None:
        accepted = set(selected).intersection(self._worlds)
        if not accepted and self._selected:
            return
        self._selected = accepted
        for world in self._selected:
            self._references[world] = deepcopy(selected[world]) or ["Default"]
        for world in self._worlds:
            self._sync(world)

    def time_limits(self) -> dict[str, float]:
        limits = {}
        for world in self._worlds:
            if world not in self._selected:
                continue
            limit = self._cards[world].time_limit.value()
            if not math.isfinite(limit) or limit < 0:
                raise ValueError(
                    f"{self._worlds[world]['label']}: enter a finite, nonnegative time limit in seconds."
                )
            limits[world] = limit
        return limits

    def set_time_limits(self, limits: dict[str, float]) -> None:
        for world, card in self._cards.items():
            card.time_limit.setValue(
                limits.get(world, self._worlds[world].get("default_time_limit", DEFAULT_LIMIT))
            )

    def set_spawns(self, world: str, rows: list[dict[str, Any]], has_more: bool = False) -> None:
        if world not in self._worlds:
            return
        previous = self._saved.get(world, {})
        current = {str(row["id"]): deepcopy(row) for row in rows}
        for key, row in current.items():
            pose = row.get("snapshot", row)
            if all(field in pose for field in ("x", "y", "yaw")):
                self._poses[world, key] = deepcopy(pose)
        # An edit saves the spawn under a new ID and deletes the old row: the selection follows its family.
        families = {
            row["family_id"]: key
            for key, row in current.items()
            if isinstance(row.get("family_id"), str)
        }
        references = self._references.get(world, [])
        followed = False
        for position, ref in enumerate(references):
            key = spawn_key(world, ref)
            if not isinstance(ref, str) or key in current:
                continue
            family = previous.get(key, {}).get("family_id")
            replacement = families.get(family) if isinstance(family, str) else None
            if replacement is not None and replacement not in references:
                references[position] = replacement
                followed = True
        for ref in references:
            key = spawn_key(world, ref)
            if key not in current and key in previous:
                current[key] = previous[key]
        self._saved[world] = current
        self._has_more[world] = has_more
        self._sync(world)
        if followed:
            self.changed.emit()

    def world_map(self, world: str) -> WorldMap | None:
        return self._maps.get(world)

    def ensure_map(self, world: str) -> None:
        """Ask for a world's map when nothing has (step 4 reached without showing step 2)."""
        if world in self._worlds and world not in self._maps:
            self._request(world)

    def spawn_label(self, world: str, reference: Reference) -> str:
        if isinstance(reference, dict):
            return str(reference["name"])
        if spawn_key(world, reference) == "default-" + world:
            return "Default"
        row = self._saved.get(world, {}).get(str(reference), {})
        return str(row.get("name") or str(reference)[:8])

    def spawn_options(self) -> list[tuple[str, str, Reference, str]]:
        """Every world's poses for the Results filter: world ID and label, reference, and name."""
        return [
            (world, info["label"], deepcopy(entry.reference), entry.name)
            for world, info in self._worlds.items()
            for entry in self._rows(world)
        ]

    def forget_spawns(self, world: str, spawn_ids: set[str]) -> None:
        """Drop deleted saved spawns from this world's selection and saved rows."""
        references = self._references.get(world, [])
        kept = [r for r in references if not (isinstance(r, str) and r in spawn_ids)]
        saved = self._saved.get(world, {})
        removed = len(kept) != len(references) or bool(spawn_ids & set(saved))
        for identity in spawn_ids:
            saved.pop(identity, None)
        if not removed:
            return
        self._references[world] = kept or ["Default"]
        if world in self._worlds:
            self._sync(world)
        self.changed.emit()

    def select_world(self, world: str, checked: bool) -> None:
        if world not in self._worlds:
            return
        if not checked and self._selected == {world}:
            self._sync(world)
            return
        if checked:
            self._selected.add(world)
        else:
            self._selected.discard(world)
            self._close_popup(world)
        self._sync(world)
        self.changed.emit()

    def _pose(self, world: str, reference: Reference) -> dict[str, Any] | None:
        if isinstance(reference, dict):
            return reference
        key = spawn_key(world, reference)
        return (
            self._worlds[world]["default_spawn"]
            if key == "default-" + world
            else self._poses.get((world, key))
        )

    def _references_of(self, world: str) -> list[Reference]:
        """Default first, then the library's spawns by name (05, 06, 15), then campaign snapshots."""
        saved = sorted(
            self._saved.get(world, {}).items(),
            key=lambda item: (str(item[1].get("name") or item[0]).casefold(), item[0]),
        )
        references: list[Reference] = ["Default", *(key for key, _ in saved)]
        known = {spawn_key(world, ref) for ref in references}
        for ref in self._references.get(world, []):
            if spawn_key(world, ref) not in known or isinstance(ref, dict):
                references = [r for r in references if spawn_key(world, r) != spawn_key(world, ref)]
                references.append(deepcopy(ref))
                known.add(spawn_key(world, ref))
        return references

    def _rows(self, world: str) -> list[SpawnEntry]:
        rows = []
        for ref in self._references_of(world):
            key = spawn_key(world, ref)
            default = key == "default-" + world
            name = self.spawn_label(world, ref)
            detail = "· world file" if default else ""
            rows.append(
                SpawnEntry(
                    deepcopy(ref),
                    key,
                    name,
                    self._pose(world, ref),
                    detail,
                    (name + " " + detail).casefold(),
                )
            )
        return rows

    def _request(self, world: str, identity: str | None = None) -> None:
        """Ask for a saved spawn's pose (``identity``) or, without one, the world's map."""
        if (world, identity) in self._pending.values():
            return
        tag = "world-selection-" + uuid.uuid4().hex
        self._pending[tag] = (world, identity)
        self.transport.submit(
            tag,
            "spawns.get" if identity else "spawns.map",
            {"id": identity} if identity else {"world_id": world},
        )

    def _sync(self, world: str, refresh_popup: bool = True) -> None:
        """Bring the card (and the open popup, unless it made the change) up to date with the world's
        selection, and ask for the poses it does not know yet."""
        card = self._cards[world]
        references = self._references[world]
        selected = world in self._selected
        names = ", ".join(self.spawn_label(world, ref) for ref in references)
        card.set_state(selected, self._selected == {world}, names)
        marks = []
        for ref in references:
            pose = self._pose(world, ref)
            if pose is not None:
                marks.append(
                    SpawnMark(
                        self.spawn_label(world, ref),
                        float(pose["x"]),
                        float(pose["y"]),
                        float(pose["yaw"]),
                        picked=True,
                        labelled=True,
                    )
                )
            elif selected:
                self._request(world, spawn_key(world, ref))
        card.stage.set_marks(marks if selected else [])
        popup = self._popups.get(world)
        if popup is not None:
            popup.set_has_more(self._has_more.get(world, False))
            if refresh_popup and popup.isVisible():
                popup.refresh(self._rows(world), references)
        for identity, other in self._cards.items():
            if identity != world:
                other.set_state(other.checked, self._selected == {identity}, other.names)

    def _popup_rect(self, world: str, host: PopupHost, has_more: bool) -> QtCore.QRect:
        """HANDOFF §8.6: two columns from the clicked card's column (ending at it when it is the last one).
        With its footer row the popup ends 5 px into the card's Starting poses row, above the Spawns button
        (06); without it (D2: the row only carries the 250-spawn note) it keeps that top edge."""
        index = list(self._worlds).index(world)
        row, column = self.grid.cell(index)
        span = min(2, self.grid.columns)
        start = max(0, min(column, self.grid.columns - span))
        box = self.grid.cell_box(row, start, span)
        origin = self.grid.content.mapTo(host.window_widget, QtCore.QPoint(0, 0))
        left = snap(origin.x() + box.x())
        bottom = snap(origin.y() + box.bottom() - WorldCard.ROWS + 5)
        top = max(8, bottom - SpawnsPopup.popup_height(True))
        return QtCore.QRect(left, top, math.floor(box.width()), SpawnsPopup.popup_height(has_more))

    def show_spawns(self, world: str) -> None:
        host = self.window().findChild(PopupHost)
        if world not in self._selected or host is None:
            return
        popup = self._popups.get(world)
        if popup is None:
            popup = SpawnsPopup(self.themes, self._worlds[world])
            popup.changed.connect(lambda refs, wid=world: self._set_spawns(wid, refs))
            popup.finished.connect(lambda wid=world: self._close_popup(wid))
            self._popups[world] = popup
        has_more = self._has_more.get(world, False)
        popup.stage.set_world(self._maps.get(world))
        if world in self._map_errors:
            popup.stage.set_message("Map unavailable")
        popup.set_has_more(has_more)
        popup.refresh(self._rows(world), self._references[world])
        card = self._cards[world]
        card.spawns_button.set_open(True)
        host.open_popup(
            popup,
            self._popup_rect(world, host, has_more),
            opener=card.spawns_button,
            on_close=lambda: card.spawns_button.set_open(False),
        )
        popup.view.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def _close_popup(self, world: str) -> None:
        popup = self._popups.get(world)
        host = self.window().findChild(PopupHost)
        if popup is not None and host is not None and host.popup is popup:
            host.close_popup()

    def _drop_popup(self, world: str) -> None:
        self._close_popup(world)
        popup = self._popups.pop(world, None)
        if popup is not None:
            popup.hide()
            popup.deleteLater()

    def _set_spawns(self, world: str, references: list[Reference]) -> None:
        if not references:
            return
        self._references[world] = deepcopy(references)
        self._sync(world, refresh_popup=False)  # the popup holds this selection already
        self.changed.emit()

    def received(self, tag: str, result: Any) -> None:
        pending = self._pending.pop(tag, None)
        if pending is None:
            return
        world, identity = pending
        if world not in self._worlds or result.get("world_id") != world:
            return
        card = self._cards[world]
        if identity:
            if result.get("id") != identity:
                return
            self._poses[world, identity] = deepcopy(result)
            card.pose_error = ""
            self._sync(world)
            return
        try:
            world_map = WorldMap.from_reply(result)
        except (KeyError, TypeError, ValueError) as exc:
            self._map_failed(world, str(exc) or "Unreadable world map")
            return
        card.map_error = ""
        self._map_errors.pop(world, None)
        self._maps[world] = world_map
        card.stage.set_world(world_map)
        popup = self._popups.get(world)
        if popup is not None:
            popup.stage.set_world(world_map)
        self.map_loaded.emit(world, world_map)

    def failed(self, tag: str, error: str) -> None:
        pending = self._pending.pop(tag, None)
        if pending and pending[0] in self._cards:
            if pending[1] is None:
                self._map_failed(pending[0], error)
            else:
                self._cards[pending[0]].pose_error = error

    def _map_failed(self, world: str, error: str) -> None:
        """Every map of the world says so (card, popup, the Accepted card through ``map_loaded``); the
        next showing asks again."""
        self._map_errors[world] = error
        card = self._cards[world]
        card.map_error = error
        card.stage.set_message("Map unavailable")
        popup = self._popups.get(world)
        if popup is not None:
            popup.stage.set_message("Map unavailable")
        self.map_loaded.emit(world, None)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        for world in self._worlds:
            if world not in self._maps:
                self._request(world)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        for world in self._popups:
            self._close_popup(world)
        super().hideEvent(event)
