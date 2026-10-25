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

"""New campaign step 4 (08, HANDOFF §8.8): the combinations to include, the parameter changes of the accepted
profiles (D14) and the campaign the engine accepted.

The combinations follow the form; the other two cards show only what the last successful ``campaign.preview``
accepted, so they never describe a design the engine has not checked.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_cells import (
    CELL_ROLE,
    Cell,
    ColumnStyle,
    TableFooter,
    TableHeader,
    TableLook,
    style_table,
)
from .gui_dialogs import Run, wrap
from .gui_maps import MapStage, SpawnMark, WorldMap
from .gui_menus import show_tooltip
from .gui_profile_preview import ParameterDifference, profile_differences, value_text
from .gui_theme import ThemeManager, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    BODY_TABULAR,
    Card,
    CardSplit,
    CardHead,
    TextStyle,
    Themed,
    css_rect,
    draw_text,
    paint_scroll_handle,
    place,
    wheel_step,
)
from .gui_wizard import GROUP_HEAD, seconds

VALIDATE_ACCEPTED = "Validate to see the accepted campaign."
VALIDATE_CHANGES = "Validate to see the profile changes."
HEADER_TEXT = TextStyle(12, 500, 18)
DESCRIPTION = TextStyle(12, 400, 18)
OLD_VALUE = BODY_TABULAR
NEW_VALUE = TextStyle(13, 600, 18, tabular=True)
CAMPAIGN_NAME = TextStyle(15, 600, 21)
BAND_LEAD = TextStyle(13, 700, 18)  # ``<b>`` in the band
VALUE = TextStyle(13, 500, 18, tabular=True)
VALUE_MONO = TextStyle(12, 400, 18, mono=True)
WORLD_NAME = TextStyle(13, 600, 18)


# ---------- combinations ----------


@dataclass(frozen=True, slots=True)
class Combination:
    """One robot, world and starting pose; robot and profiles stay blank after the robot's first row."""

    key: tuple[str, str, str]  # robot, world, spawn identity
    robot: str
    profiles: str
    world: str
    spawn: str
    experiments: int  # profiles × repetitions


class CombinationModel(QtCore.QAbstractTableModel):
    HEADERS = ("", "Robot", "Profiles", "World", "Starting pose", "Experiments")

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[Combination] = []
        self.excluded: set[tuple[str, str, str]] = set()

    def set_rows(self, rows: list[Combination], excluded: set[tuple[str, str, str]]) -> None:
        self.beginResetModel()
        self.rows, self.excluded = rows, set(excluded)
        self.endResetModel()

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.rows)

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

    def cell(self, row: Combination, column: int) -> Cell:
        out = row.key in self.excluded
        muted = "text.tertiary" if out else None
        if column == 0:
            state = QtCore.Qt.CheckState.Unchecked if out else QtCore.Qt.CheckState.Checked
            return Cell(kind="check", check=state)
        if column == 1:  # ``td.name``; a left-out row drops to 400 in text.tertiary
            return Cell(row.robot, "text", token=muted) if out else Cell(row.robot, "name")
        if column == 5:
            return Cell("Left out" if out else f"{row.experiments:,}", "number", token=muted)
        text = (row.profiles, row.world, row.spawn)[column - 2]
        return Cell(text, token=muted)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        cell = self.cell(self.rows[index.row()], index.column())
        if role == CELL_ROLE:
            return cell
        if role == QtCore.Qt.ItemDataRole.DisplayRole and cell.kind != "check":
            return cell.text
        return None


COMBINATION_LOOK = TableLook(
    key="review.combinations",
    columns=[
        ColumnStyle("check", padding_left=20, padding_right=12, width=60),
        ColumnStyle("name", width=200),
        ColumnStyle("text"),
        ColumnStyle("text", width=140),
        ColumnStyle("text", width=150),
        ColumnStyle("number", padding_right=20, width=130),
    ],
)


class CombinationsCard(Card):
    """``.combos`` (08): the ruled head, the table (check, Robot, Profiles, World, Starting pose, Experiments;
    40 px rows) and "N of M included". The card ends at its last row; more rows scroll inside it. A click on
    a check, or Space on the current row, asks to include or leave out that combination."""

    toggled = QtCore.Signal(object)  # the combination's key
    toggled_all = QtCore.Signal()

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "Combinations", self)
        self.model = CombinationModel()
        self.view = QtWidgets.QTableView(self)
        self.view.setModel(self.model)
        style_table(self.view, themes, COMBINATION_LOOK)
        header = self.view.horizontalHeader()
        assert isinstance(header, TableHeader)
        self.header = header
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.view.setAccessibleName("Combinations")
        self.footer = TableFooter(themes, self, padding=20)
        self.header.check_toggled.connect(self.toggled_all.emit)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)

    def set_rows(self, rows: list[Combination], excluded: set[tuple[str, str, str]]) -> None:
        self.model.set_rows(rows, excluded)
        out = sum(row.key in excluded for row in rows)
        self.header.set_check_state(
            QtCore.Qt.CheckState.Checked
            if not out
            else QtCore.Qt.CheckState.Unchecked
            if out == len(rows)
            else QtCore.Qt.CheckState.PartiallyChecked
        )
        self.footer.set_text(f"{len(rows) - out:,} of {len(rows):,} included")
        self.updateGeometry()

    def natural_height(self) -> int:
        """Head, header, every row and the footer (the card ends at its last row)."""
        return 1 + 52 + 36 + 40 * max(1, len(self.model.rows)) + 40 + 1

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            watched is self.view.viewport()
            and event.type() == QtCore.QEvent.Type.MouseButtonPress
            and isinstance(event, QtGui.QMouseEvent)
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            index = self.view.indexAt(event.position().toPoint())
            if index.isValid() and index.column() == 0:
                self.view.setCurrentIndex(index)
                self.toggled.emit(self.model.rows[index.row()].key)
                return True
        if (
            watched is self.view
            and event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() == QtCore.Qt.Key.Key_Space
        ):
            current = self.view.currentIndex()
            if current.isValid():
                self.toggled.emit(self.model.rows[current.row()].key)
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.head.setGeometry(1, 1, width - 2, 52)
        self.view.setGeometry(1, 53, width - 2, max(0, height - 1 - 40 - 53))
        self.footer.setGeometry(1, height - 1 - 40, width - 2, 40)


# ---------- profile changes ----------


@dataclass(frozen=True, slots=True)
class ChangeRow:
    """One parameter: its name, unit and description, the Default value and each profile's value."""

    key: str
    name: str
    unit: str
    description: str  # the first sentence
    full: str  # the whole description (tooltip)
    default: str
    values: tuple[str, ...]
    changed: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class ChangeBlock:
    """One robot's changed parameters, one column per non-Default profile (D14)."""

    robot: str
    columns: tuple[str, ...]  # "<robot> Default", then the profiles ("Low Threshold r2")
    rows: tuple[ChangeRow, ...]


def first_sentence(text: str) -> str:
    """The description line under a parameter (08): its first sentence; the tooltip has the rest."""
    head, separator, _ = text.strip().partition(". ")
    return head + "." if separator else head


def is_default_profile(robot_id: str, profile: dict[str, Any]) -> bool:
    return profile.get("id") in ("Default", "default", "default-" + robot_id)


def profile_title(profile: dict[str, Any]) -> str:
    """The accepted profile's name ("Low Threshold")."""
    return str(profile.get("name") or "Profile")


def change_blocks(
    specification: dict[str, Any],
    robots: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> list[ChangeBlock]:
    """The accepted specification's parameter changes from each robot's Default (``profile_differences``)."""
    blocks = []
    for robot_id, profiles in (specification.get("robots") or {}).items():
        robot = robots.get(robot_id)
        if robot is None or not robot.get("defaults"):
            continue
        changed_profiles = [
            p for p in profiles if isinstance(p, dict) and not is_default_profile(robot_id, p)
        ]
        per_profile: list[dict[str, ParameterDifference]] = []
        order: dict[str, ParameterDifference] = {}
        for profile in changed_profiles:
            differences = {d.key: d for d in profile_differences(robot, metadata, profile)}
            per_profile.append(differences)
            for key, difference in differences.items():
                order.setdefault(key, difference)
        if not order:
            continue
        rows = []
        for key, first in order.items():
            full = str(metadata.get(key, {}).get("description") or "").strip()
            values, changed = [], []
            for differences in per_profile:
                found = differences.get(key)
                values.append(value_text(found.value if found else first.default))
                changed.append(found is not None)
            rows.append(
                ChangeRow(
                    key,
                    first.name,
                    first.unit,
                    first_sentence(full),
                    full,
                    value_text(first.default),
                    tuple(values),
                    tuple(changed),
                )
            )
        label = str(robot.get("label") or robot_id)
        columns = (label + " Default", *(profile_title(p) for p in changed_profiles))
        blocks.append(ChangeBlock(label, columns, tuple(rows)))
    return blocks


class ChangesTable(Themed):
    """The body of ``.diff`` (08): per robot (a 32 px robot row when there are several) the header row
    (Parameter, the robot's Default, one 150 px column per profile) and 60 px rows: the name in 500 with its
    unit, the description in 12 px ``text.secondary`` under it, the Default value and each profile's value
    (600 ink where it changed). Wider than the card from four profiles on: it then scrolls sideways."""

    ROW, HEAD, ROBOT_ROW, COLUMN, PARAMETER = 60, 36, 32, 150, 354

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.blocks: list[ChangeBlock] = []
        self.message = ""
        self.offset = QtCore.QPointF(0, 0)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

    def set_blocks(self, blocks: list[ChangeBlock], message: str) -> None:
        self.blocks, self.message = blocks, message
        self.offset = QtCore.QPointF(0, 0)
        self.update()

    def content_width(self) -> float:
        columns = max((len(block.columns) for block in self.blocks), default=1)
        return max(float(self.width()), self.PARAMETER + self.COLUMN * columns)

    def content_height(self) -> float:
        heads = self.ROBOT_ROW if len(self.blocks) > 1 else 0
        return sum(heads + self.HEAD + self.ROW * len(block.rows) for block in self.blocks)

    def scroll_by(self, dx: float, dy: float) -> None:
        x = min(max(0.0, self.offset.x() + dx), self.content_width() - self.width())
        y = min(max(0.0, self.offset.y() + dy), max(0.0, self.content_height() - self.height()))
        if (x, y) != (self.offset.x(), self.offset.y()):
            self.offset = QtCore.QPointF(x, y)
            self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        step = wheel_step(event, self.ROW)
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            step = QtCore.QPointF(step.y(), step.x())
        self.scroll_by(-step.x(), -step.y())
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.scroll_by(0, 0)

    def event(self, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if kind in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()
        elif kind == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            show_tooltip(self, event.globalPos(), self.tooltip_at(QtCore.QPointF(event.pos())))
            return True
        return super().event(event)

    def tooltip_at(self, position: QtCore.QPointF) -> str:
        y = position.y() + self.offset.y()
        for block in self.blocks:
            y -= (self.ROBOT_ROW if len(self.blocks) > 1 else 0) + self.HEAD
            if 0 <= y < self.ROW * len(block.rows):
                row = block.rows[int(y // self.ROW)]
                return f"{row.key}\n{row.full}" if row.full else row.key
            y -= self.ROW * len(block.rows)
        return ""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        if not self.blocks:
            if self.message:
                color = theme.color("text.secondary")
                draw_text(painter, 20, 16, BODY.elide(self.message, self.width() - 40), BODY, color)
            return
        width = self.content_width()
        painter.translate(-self.offset.x(), -self.offset.y())
        line, secondary = theme.color("border.line"), theme.color("text.secondary")
        y = 0.0
        last_block = self.blocks[-1]
        for block in self.blocks:
            if len(self.blocks) > 1:
                painter.fillRect(QtCore.QRectF(0, y + self.ROBOT_ROW - 1, width, 1), line)
                top = y + (self.ROBOT_ROW - 1 - 18) / 2
                draw_text(painter, 20, top, block.robot, GROUP_HEAD, secondary)
                y += self.ROBOT_ROW
            self._paint_head(painter, block, y, width)
            y += self.HEAD
            for index, row in enumerate(block.rows):
                last = block is last_block and index == len(block.rows) - 1
                self._paint_row(painter, block, row, y, width, last)
                y += self.ROW
        painter.resetTransform()
        paint_scroll_handle(painter, self, 0, self.height(), self.offset.y(), self.content_height())

    def _columns(self, block: ChangeBlock, width: float) -> list[tuple[float, float]]:
        """Where each value column's text ends and how wide it may be (padding 0 12; the last column's
        right padding is 20, ``.tbl.pad``)."""
        count = len(block.columns)
        edges = []
        for index in range(count):
            padding = 20 if index == count - 1 else 12
            right = width - self.COLUMN * (count - 1 - index) - padding
            edges.append((right, float(self.COLUMN - 12 - padding)))
        return edges

    def _paint_head(
        self, painter: QtGui.QPainter, block: ChangeBlock, y: float, width: float
    ) -> None:
        theme = self.theme
        painter.fillRect(QtCore.QRectF(0, y, width, self.HEAD), theme.color("surface.sunken"))
        painter.fillRect(QtCore.QRectF(0, y + self.HEAD - 1, width, 1), theme.color("border.line"))
        top = y + HEADER_TEXT.middle_baseline(self.HEAD - 1) - HEADER_TEXT.baseline(0)
        secondary = theme.color("text.secondary")
        draw_text(painter, 20, top, "Parameter", HEADER_TEXT, secondary)
        for (right, room), text in zip(self._columns(block, width), block.columns, strict=True):
            text = HEADER_TEXT.elide(text, room)
            draw_text(painter, right - HEADER_TEXT.advance(text), top, text, HEADER_TEXT, secondary)

    def _paint_row(
        self,
        painter: QtGui.QPainter,
        block: ChangeBlock,
        row: ChangeRow,
        y: float,
        width: float,
        last: bool,
    ) -> None:
        theme = self.theme
        content = self.ROW - (0 if last else 1)
        if not last:
            painter.fillRect(QtCore.QRectF(0, y + content, width, 1), theme.color("border.line"))
        # ``td.name``: the name line (taller when the ``td > span`` unit aligns middle), then the description.
        if row.unit:
            line_height, line_baseline = BODY_MEDIUM.middle_line(18)
        else:
            line_height, line_baseline = BODY_MEDIUM.line, BODY_MEDIUM.baseline(0)
        total = line_height + (2 + DESCRIPTION.line if row.description else 0)
        top = y + (content - total) / 2
        baseline = top + line_baseline
        value_columns = self._columns(block, width)
        room = width - self.COLUMN * len(block.columns) - 20 - 12  # the Parameter column's text
        name = BODY_MEDIUM.elide(row.name, max(0.0, room))
        name_top = baseline - BODY_MEDIUM.baseline(0)
        draw_text(painter, 20, name_top, name, BODY_MEDIUM, theme.color("text.ink"))
        if row.unit:
            unit_x = 20 + BODY_MEDIUM.advance(name + " ") + 2
            unit_top = baseline + BODY_MEDIUM.middle_offset(18)
            draw_text(painter, unit_x, unit_top, row.unit, BODY, theme.color("text.tertiary"))
        if row.description:
            description = DESCRIPTION.elide(row.description, max(0.0, room))
            draw_text(
                painter,
                20,
                top + line_height + 2,
                description,
                DESCRIPTION,
                theme.color("text.secondary"),
            )
        # Values sit on the line centred in the cell (``td.n``).
        value_top = y + BODY.middle_baseline(content) - BODY.baseline(0)
        cells = [(row.default, False), *zip(row.values, row.changed, strict=True)]
        for (right, room), (text, changed) in zip(value_columns, cells, strict=True):
            style = NEW_VALUE if changed else OLD_VALUE
            color = theme.color("text.ink" if changed else "text.secondary")
            text = style.elide(text, room)
            draw_text(painter, right - style.advance(text), value_top, text, style, color)


class ChangesCard(Card):
    """``.diff`` (08): "Profile changes" over the changes table, filling the column below the combinations."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "Profile changes", self)
        self.table = ChangesTable(themes, self)
        self.set_blocks([], VALIDATE_CHANGES)

    def set_blocks(self, blocks: list[ChangeBlock], message: str = "") -> None:
        self.table.set_blocks(blocks, message)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.head.setGeometry(1, 1, width - 2, 52)
        self.table.setGeometry(1, 53, width - 2, max(0, height - 54))


# ---------- accepted campaign ----------


@dataclass(frozen=True, slots=True)
class AcceptedWorld:
    world_id: str
    label: str
    poses: str  # "Default, Centre"
    limit: str  # "1800 s", "Unlimited"
    marks: tuple[SpawnMark, ...]


@dataclass(frozen=True, slots=True)
class AcceptedFacts:
    """What ``campaign.preview`` accepted, as the card shows it."""

    name: str
    description: str
    execution: str  # "Sequential", "2 parallel workers"
    timing_retry: str
    worlds: tuple[AcceptedWorld, ...]
    # the study and recording rows (33): key, value, set in mono, a faint addition (the plan's hash)
    study: tuple[tuple[str, str, bool, str], ...] = ()
    time_rule: str = ""  # the campaign's ``time_limit_rule`` (39), empty without one

    def rows(self) -> list["KeyValue"]:
        rule = [KeyValue("Time limits", "Kinematic rule", tail=self.time_rule, identifier=True)]
        return [
            KeyValue("Execution", self.execution),
            KeyValue("Timing retry", self.timing_retry),
            *(rule if self.time_rule else []),
            *(KeyValue(key, value, mono, tail) for key, value, mono, tail in self.study),
        ]


@dataclass(frozen=True, slots=True)
class KeyValue:
    """A ``.kv`` row of the accepted card: the value (in mono for identifiers such as the plan), then an
    optional ``tail``: a faint note (the plan's hash) or, with ``identifier``, an ID in mono
    ``text.secondary`` one space and 6 px after the value (the time-limit rule, 39)."""

    key: str
    value: str
    mono: bool = False
    tail: str = ""
    identifier: bool = False


def execution_text(concurrency: Any) -> str:
    if isinstance(concurrency, int) and not isinstance(concurrency, bool) and concurrency > 1:
        return f"{concurrency} parallel workers"
    return "Sequential" if concurrency == 1 else "—"


def limit_text(limit: Any) -> str:
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit):
        return "—"
    return "Unlimited" if limit == 0 else f"{seconds(limit)} s"


class AcceptedCard(Card):
    """``.acc`` (08): after a successful ``campaign.preview``, the success band, the campaign's name and
    description, Execution and Timing retry, then "Worlds & starting poses": one tile per world (a 168×132
    map, its time limit and poses to the right) in a grid from the top, as many columns as fit, rows ruled
    apart. Before that a muted line says what to do; a failed validation shows its error in the band, in the
    bad colours."""

    # screen 33 (the study rows joined): 34 px key-value rows, 128×98 maps, 10 px around the name
    BAND, KV_ROW, MAP_W, MAP_H = 44, 34, 128, 98
    MAP_GAP = 18  # ``.wr``: the map, then the world's facts 18 px to its right
    TILE_PAD = 14  # above and below the map in a grid row
    TILE_GAP = 24  # between the grid's columns
    FACTS_MIN = 180  # a tile's narrowest facts column
    WORLD_HEAD = 1 + 30 + 4  # the section's top rule, its 30 px heading and the list's 4 px padding

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "Accepted campaign", self)
        self.facts: AcceptedFacts | None = None
        self.error = ""
        self.message = VALIDATE_ACCEPTED
        self.maps: dict[str, WorldMap | None] = {}
        self.stages: list[MapStage] = []
        self.offset = 0.0  # the world grid scrolls when its rows do not fit the card
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

    # state

    def set_message(self, text: str) -> None:
        """No accepted campaign (not validated yet, or validating)."""
        self.facts, self.error, self.message = None, "", text
        self._sync_stages()

    def set_error(self, text: str) -> None:
        self.facts, self.error, self.message = None, text, ""
        self._sync_stages()

    def set_accepted(self, facts: AcceptedFacts, maps: dict[str, WorldMap | None]) -> None:
        self.facts, self.error, self.message = facts, "", ""
        self.maps = dict(maps)
        self.offset = 0.0
        self._sync_stages()

    def set_map(self, world_id: str, world: WorldMap | None) -> None:
        """A world map that arrived after validation (None: it could not be loaded)."""
        self.maps[world_id] = world
        for stage, entry in zip(self.stages, self.facts.worlds if self.facts else (), strict=False):
            if entry.world_id == world_id:
                stage.set_world(world)
                if world is None:
                    stage.set_message("Map unavailable")

    def _sync_stages(self) -> None:
        worlds = self.facts.worlds if self.facts else ()
        while len(self.stages) > len(worlds):
            self.stages.pop().deleteLater()
        while len(self.stages) < len(worlds):
            stage = MapStage(self.themes, pad=8, parent=self)
            stage.show()
            self.stages.append(stage)
        for stage, world in zip(self.stages, worlds, strict=True):
            stage.set_world(self.maps.get(world.world_id))
            stage.set_marks(world.marks)
        self._layout()
        self.update()

    # geometry (card coordinates: the 1 px border included)

    def _description_lines(self, width: float) -> list[list[Run]]:
        if self.facts is None or not self.facts.description:
            return []
        return wrap([Run(self.facts.description, BODY, "text.secondary")], width - 40)

    def _band_lines(self, width: float) -> list[list[Run]]:
        if self.error:
            runs = [
                Run("Validation failed.", BAND_LEAD, "state.bad"),
                Run(" " + self.error, BODY, "state.bad"),
            ]
        else:
            runs = [
                Run("Validation passed.", BAND_LEAD, "state.ok"),
                Run(" Start it from Campaigns after queueing.", BODY, "state.okStrong"),
            ]
        return wrap(runs, width - 20 - 16 - 10 - 20)

    def band_height(self, width: float) -> float:
        return max(float(self.BAND), 26 + 18 * len(self._band_lines(width)))

    def sections(self) -> tuple[float, float, float, float]:
        """Tops of the band, the name block, the key-values and the worlds section."""
        box = css_rect(self)
        width = box.width()
        band = box.y() + 53
        who = band + self.band_height(width)
        lines = self._description_lines(width)
        who_height = 10 + 21 + (3 + 18 * len(lines) if lines else 0) + 10 + 1
        kv = who + who_height
        where = kv + len(self.facts.rows() if self.facts else ()) * self.KV_ROW
        return band, who, kv, where

    def rows_top(self) -> float:
        """Where the world rows start, under the section's rule and heading."""
        return self.sections()[3] + self.WORLD_HEAD

    def row_height(self) -> float:
        return float(self.MAP_H + 2 * self.TILE_PAD)

    def columns(self) -> int:
        """Grid columns: as many minimum-width tiles as the inner width holds, at most one per world."""
        count = len(self.facts.worlds) if self.facts else 0
        inner = css_rect(self).width() - 42
        tile = self.MAP_W + self.MAP_GAP + self.FACTS_MIN
        return max(1, min(count, int((inner + self.TILE_GAP) // (tile + self.TILE_GAP))))

    def world_tiles(self) -> list[QtCore.QRectF]:
        """Each world's tile (left, top, width and row height), in grid order, scrolled."""
        if self.facts is None or not self.facts.worlds:
            return []
        box = css_rect(self)
        columns, rows_top, each = self.columns(), self.rows_top(), self.row_height()
        inner = box.width() - 42
        width = (inner - (columns - 1) * self.TILE_GAP) / columns
        return [
            QtCore.QRectF(
                box.x() + 21 + (index % columns) * (width + self.TILE_GAP),
                rows_top + (index // columns) * (each + 1) - self.offset,
                width,
                each,
            )
            for index in range(len(self.facts.worlds))
        ]

    def list_height(self) -> tuple[float, float]:
        """The world grid's visible and total height."""
        if self.facts is None:
            return 0.0, 0.0
        rows = math.ceil(len(self.facts.worlds) / self.columns())
        total = rows * self.row_height() + max(0, rows - 1)
        return css_rect(self).bottom() - 1 - self.rows_top(), total

    def _layout(self) -> None:
        box = css_rect(self)
        self.head.setGeometry(1, 1, self.width() - 2, 52)
        rows_top = self.rows_top() if self.facts else 0.0
        for stage, tile in zip(self.stages, self.world_tiles(), strict=False):
            y = tile.y() + self.TILE_PAD
            place(stage, tile.x(), y, self.MAP_W, self.MAP_H)
            # A stage scrolled partly out of the list is hidden (widgets do not clip to the list).
            stage.setVisible(y >= rows_top - 1 and y + self.MAP_H <= box.bottom())

    def css_changed(self) -> None:
        self._layout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        visible, total = self.list_height()
        step = wheel_step(event, 60).y()
        offset = min(max(0.0, self.offset - step), max(0.0, total - visible))
        if offset != self.offset:
            self.offset = offset
            self._layout()
            self.update()
        event.accept()

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()
        return super().event(event)

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        theme = self.theme
        box = css_rect(self)
        left, top, width = box.x(), box.y(), box.width()
        x, inner = left + 21, width - 42
        band, who, kv, where = self.sections()
        if self.facts is None and not self.error:
            draw_text(painter, x, top + 53 + 16, self.message, BODY, theme.color("text.secondary"))
            return
        self._paint_band(painter, left, band, width)
        if self.facts is None:
            return
        facts = self.facts
        line = theme.color("border.line")
        ink, secondary = theme.color("text.ink"), theme.color("text.secondary")
        # Name and description, then a rule.
        name = CAMPAIGN_NAME.elide(facts.name, inner)
        draw_text(painter, x, who + 10, name, CAMPAIGN_NAME, ink)
        for index, runs in enumerate(self._description_lines(width)):
            text = "".join(run.text for run in runs)
            draw_text(painter, x, who + 10 + 21 + 3 + 18 * index, text, BODY, secondary)
        painter.fillRect(QtCore.QRectF(left + 1, kv - 1, width - 2, 1), line)
        # Execution, Timing retry, the time-limit rule and the study rows (``.kv``: 112 px keys, values 16 px
        # further, rules between).
        rows = facts.rows()
        for index, item in enumerate(rows):
            row = kv + index * self.KV_ROW
            last = index == len(rows) - 1
            content = self.KV_ROW - (0 if last else 1)
            if not last:
                painter.fillRect(QtCore.QRectF(x, row + content, inner, 1), line)
            line_top = row + (content - 18) / 2
            draw_text(painter, x, line_top, item.key, BODY, secondary)
            value_x = x + 112 + 16
            style = VALUE_MONO if item.mono else VALUE
            if item.identifier:
                tail_style, tail_text, gap = VALUE_MONO, item.tail, BODY.advance(" ") + 6
            else:
                tail_style, tail_text, gap = BODY, " " + item.tail, 0.0
            tail = gap + tail_style.advance(tail_text) if item.tail else 0.0
            shown = style.elide(item.value, x + inner - value_x - tail)
            advance = draw_text(painter, value_x, line_top, shown, style, ink)
            if item.tail:
                room = x + inner - value_x - advance - gap
                draw_text(
                    painter,
                    value_x + advance + gap,
                    line_top,
                    tail_style.elide(tail_text, max(0.0, room)),
                    tail_style,
                    theme.color("text.secondary" if item.identifier else "text.tertiary"),
                )
        # Worlds and starting poses.
        painter.fillRect(QtCore.QRectF(left + 1, where, width - 2, 1), line)
        draw_text(painter, x, where + 1 + 12, "Worlds & starting poses", GROUP_HEAD, secondary)
        rows_top = self.rows_top()
        painter.setClipRect(
            QtCore.QRectF(left + 1, rows_top, width - 2, box.bottom() - 1 - rows_top)
        )
        tiles, columns = self.world_tiles(), self.columns()
        for index, (tile, world) in enumerate(zip(tiles, facts.worlds, strict=True)):
            if index % columns == 0 and index + columns < len(
                tiles
            ):  # a rule under each full grid row
                painter.fillRect(QtCore.QRectF(x, tile.bottom(), inner, 1), line)
            facts_x = tile.x() + self.MAP_W + self.MAP_GAP
            self._paint_world(
                painter, world, facts_x, tile.y(), tile.height(), tile.right() - facts_x
            )
        painter.setClipping(False)
        visible, total = self.list_height()
        paint_scroll_handle(painter, self, rows_top, visible, self.offset, total)

    def _paint_band(self, painter: QtGui.QPainter, left: float, band: float, width: float) -> None:
        theme = self.theme
        bad = bool(self.error)
        height = self.band_height(width)
        fill = theme.color("state.badSoft" if bad else "state.okSoft")
        painter.fillRect(QtCore.QRectF(left + 1, band, width - 2, height - 1), fill)
        rule = theme.color("state.badLine" if bad else "state.okLine")
        painter.fillRect(QtCore.QRectF(left + 1, band + height - 1, width - 2, 1), rule)
        icon_color = theme.color("state.bad" if bad else "state.ok")
        icon = QtCore.QPointF(snap(left + 21), snap(band + (height - 1 - 16) / 2))  # centred
        gui_icons.draw(painter, "cx" if bad else "ccheck", icon_color, icon)
        lines = self._band_lines(width)
        first = band + (height - 1 - 18 * len(lines)) / 2
        for index, runs in enumerate(lines):
            x = left + 21 + 16 + 10
            for run in runs:
                x += draw_text(
                    painter, x, first + 18 * index, run.text, run.style, theme.color(run.token)
                )

    def _paint_world(
        self,
        painter: QtGui.QPainter,
        world: AcceptedWorld,
        x: float,
        row_top: float,
        each: float,
        width: float,
    ) -> None:
        theme = self.theme
        ink, secondary = theme.color("text.ink"), theme.color("text.secondary")
        # ``.wf`` (33): the name, then "Starting poses" and "Time limit", each a label with its value at the
        # right (a long pose list elided), 6 px apart, the block centred on the map.
        block = 18 * 3 + 6 * 2
        top = row_top + (each - block) / 2
        draw_text(painter, x, top, WORLD_NAME.elide(world.label, width), WORLD_NAME, ink)
        for index, (key, value) in enumerate(
            (("Starting poses", world.poses or "—"), ("Time limit", world.limit))
        ):
            line_top = top + 24 * (index + 1)
            key_width = draw_text(painter, x, line_top, key, BODY, secondary)
            shown = VALUE.elide(value, max(0.0, width - key_width - 12))
            draw_text(painter, x + width - VALUE.advance(shown), line_top, shown, VALUE, ink)


def study_rows(
    specification: Mapping[str, Any], existing: Callable[[str], bool]
) -> tuple[tuple[str, str, bool, str], ...]:
    """The study, its plan (in mono) with the hash the engine computed (faint), the study's repetition
    numbers and the debug streams (33)."""
    study = specification.get("study")
    repetitions = specification.get("repetitions")
    count = repetitions if isinstance(repetitions, int) and not isinstance(repetitions, bool) else 0
    debug = "On" if specification.get("debug_streams") else "Off"
    if not isinstance(study, Mapping) or not study.get("id"):
        return (
            ("Study", "Unassigned", False, ""),
            ("Repetitions", f"1–{count}" if count else "—", False, ""),
            ("Debug streams", debug, False, ""),
        )
    identity = str(study["id"])
    first = study.get("first_repetition")
    first = first if isinstance(first, int) and not isinstance(first, bool) else 0
    plan = study.get("plan")
    digest = str(study.get("plan_sha256") or "")
    return (
        (
            "Study",
            f"{identity} · {'existing' if existing(identity) else 'new'} · "
            f"{study.get('evidence_level', 'exploratory')}",
            False,
            "",
        ),
        (
            "Plan",
            str(plan).rsplit("/", 1)[-1] if plan else "None",
            bool(plan),
            f"· {digest[:8]}" if plan and digest else "",
        ),
        ("Repetitions", f"{first + 1}–{first + count}" if count else "—", False, ""),
        ("Debug streams", debug, False, ""),
    )


def accepted_facts(
    specification: dict[str, Any],
    timing_retry: Any,
    world_labels: dict[str, str],
    spawn_label: Callable[[str, dict[str, Any]], str],
    existing: Callable[[str], bool] = lambda _: False,
) -> AcceptedFacts:
    """The card's view of an accepted specification (``campaign.preview``)."""
    limits = specification.get("world_time_limits") or {}
    worlds = []
    for world_id, spawns in (specification.get("worlds") or {}).items():
        marks = []
        names = []
        for spawn in spawns:
            name = spawn_label(world_id, spawn)
            names.append(name)
            try:
                x, y, yaw = (float(spawn[field]) for field in ("x", "y", "yaw"))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in (x, y, yaw)):
                marks.append(SpawnMark(name, x, y, yaw, picked=True))
        worlds.append(
            AcceptedWorld(
                world_id,
                world_labels.get(world_id, world_id),
                ", ".join(names),
                limit_text(limits.get(world_id)),
                tuple(marks),
            )
        )
    return AcceptedFacts(
        str(specification.get("name") or ""),
        str(specification.get("description") or ""),
        execution_text(specification.get("concurrency")),
        str(timing_retry) if timing_retry else "—",
        tuple(worlds),
        study_rows(specification, existing),
        str(specification.get("time_limit_rule") or ""),
    )


class ReviewPage(QtWidgets.QWidget):
    """``.body`` of 08: the combinations card (as tall as its rows, at most leaving the changes card its
    head, header and one row) over the profile changes card, and the 420 px accepted campaign card. Both gaps
    can be dragged (the sizes are remembered; a double click restores the design's)."""

    GAP, SIDE = 16, 420

    def __init__(
        self,
        combinations: CombinationsCard,
        changes: ChangesCard,
        accepted: AcceptedCard,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.combinations, self.changes, self.accepted = combinations, changes, accepted
        for card in (combinations, changes, accepted):
            card.setParent(self)
        themes = accepted.themes
        self.side = CardSplit(themes, self, "review/side", self.relayout, across=True, sign=-1)
        self.top = CardSplit(themes, self, "review/combinations", self.relayout, across=False)

    def relayout(self) -> None:
        width, height = float(self.width()), float(self.height())
        side = self.side.resolve(self.SIDE, 320, width - self.GAP - 480)
        left_width = width - self.GAP - side
        # The changes card keeps its head, the table's header and one row.
        minimum_changes = 2 + 52 + ChangesTable.HEAD + ChangesTable.ROW
        natural = min(
            float(self.combinations.natural_height()), height - self.GAP - minimum_changes
        )
        combinations = self.top.resolve(
            natural, 2 + 52 + 36 + 40, height - self.GAP - minimum_changes
        )
        place(self.combinations, 0, 0, left_width, max(0.0, combinations))
        changes_top = combinations + self.GAP
        place(self.changes, 0, changes_top, left_width, max(0.0, height - changes_top))
        place(self.accepted, left_width + self.GAP, 0, side, height)
        self.side.place(left_width, 0, self.GAP, height)
        self.top.place(0, combinations, left_width, self.GAP)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()
