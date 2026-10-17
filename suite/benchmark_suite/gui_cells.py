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

"""Tables of the Main design (HANDOFF §7.8): the cell delegate, the header and the footer, and the painted table
(``PaintedTable``) of the Monitor's feeds and the Profiles comparison.

``CellDelegate`` subclasses ``TableTextDelegate``, so in-cell text selection and copying keep working; it hit-
tests with the same glyph advances it paints with. Models may answer ``CELL_ROLE`` with a ``Cell`` (badge, icon,
unit, mono, colour); otherwise the delegate paints the display text, right-aligned when the model asks for it.
Vertical positions follow the CSS table layout of the mockups: a 1 px bottom rule on every row but the last,
content centred in what remains, inline boxes aligned ``middle`` and snapped to whole pixels like Chrome.

Column widths are adjustable as in a regular table: dragging a column's right edge in the header resizes it (a
split cursor and a 2 px line mark the edge), the flexible column (or, for the flexible column itself, the next
one) takes up the difference so the table keeps its width, and a double click on an edge restores the design's
widths. Dragged widths are remembered per table (``ColumnWidths``).
"""

import bisect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_controls import BadgeLook, paint_badge
from .gui_menus import show_tooltip
from .gui_table import TableTextDelegate
from .gui_theme import Theme, ThemeManager, check_mark, paint_check, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    BODY_TABULAR,
    FocusRing,
    TextStyle,
    Themed,
    draw_text,
    rounded_path,
    wheel_step,
)

CellKind = Literal["text", "name", "number", "mono", "badge", "rep", "check", "empty"]
Align = Literal["left", "right"]
NoteKind = Literal["excluded", "limit", "detail"]

CELL_ROLE = QtCore.Qt.ItemDataRole.UserRole + 41
DIM_ROLE = QtCore.Qt.ItemDataRole.UserRole + 42  # the whole row in text.tertiary (excluded runs)

MONO_TEXT = TextStyle(12, 400, 18, mono=True)
# Style, colour and gap after the text (None: a space of the note's style).
NOTE_STYLES: dict[NoteKind, tuple[TextStyle, str, float | None]] = {
    "excluded": (TextStyle(11, 400, 18), "text.tertiary", 6.0),  # ``.excl`` after a badge
    "limit": (TextStyle(11.5, 400, 18), "state.warn", 4.0),  # censored durations: "30:00 limit"
    "detail": (BODY, "text.tertiary", None),  # "Default · world file": a space, then faint text
}
HEADER_TEXT = TextStyle(12, 500, 18)
NO_TEXT: tuple[CellKind, ...] = ("badge", "check", "empty")


@dataclass(frozen=True, slots=True)
class Cell:
    """What one cell shows. ``kind``: text, name (ink 500), number (tabular), mono, badge, rep (icon and
    text in ``text.secondary``), check or empty."""

    text: str = ""
    kind: CellKind = "text"
    badge: BadgeLook | None = None
    icon: str | None = None
    unit: str = ""
    token: str | None = None
    note: str = ""
    note_kind: NoteKind = "excluded"
    check: QtCore.Qt.CheckState | None = None
    tooltip: str = ""


@dataclass(frozen=True, slots=True)
class ColumnStyle:
    """Per column: the default cell kind, alignment and paddings (``th``/``td`` padding 0 12; the
    checkbox column 0 0 0 16)."""

    kind: CellKind = "text"
    align: Align | None = None  # None: numbers right, everything else left
    padding_left: int = 12
    padding_right: int = 12
    width: int | None = None  # initial width at 1440 px (None: stretches)

    def right(self, kind: CellKind | None = None) -> bool:
        """Whether a cell of ``kind`` (default: the column's) is right-aligned (``th.n``/``td.n``)."""
        if self.align is not None:
            return self.align == "right"
        return (kind or self.kind) == "number"


CHECK_COLUMN = ColumnStyle("check", padding_left=16, padding_right=0, width=44)


@dataclass(slots=True)
class TableLook:
    """Shared by a table's delegate and header."""

    columns: list[ColumnStyle] = field(default_factory=list)
    row_height: int = 40
    header_height: int = 36
    text_size: float = 13
    # Inner radius of the card corners the header's top and the viewport's bottom follow (0: none).
    round_top: float = 0.0
    round_bottom: float = 0.0
    key: str = ""  # remembers dragged column widths as ``table/<key>`` (empty: not remembered)

    def column(self, index: int) -> ColumnStyle:
        return self.columns[index] if 0 <= index < len(self.columns) else ColumnStyle()


def _style(kind: CellKind, size: float) -> TextStyle:
    if kind == "mono":
        return MONO_TEXT
    if size == 13:
        return BODY_MEDIUM if kind == "name" else BODY_TABULAR if kind == "number" else BODY
    return TextStyle(size, 500 if kind == "name" else 400, 18, tabular=kind == "number")


class CellDelegate(TableTextDelegate):
    """Paints rows as ``.tbl``: hover and selected fills, the 2 px ink bar of the selected row, the bottom
    rule, the keyboard-focus frame of the current row, and the cell content."""

    def __init__(self, view: QtWidgets.QTableView, themes: ThemeManager, look: TableLook) -> None:
        super().__init__(view)
        self.themes, self.look = themes, look
        self.hover_row = -1
        themes.changed.connect(self._restyle)

    def _restyle(self, *_: object) -> None:
        self.viewport.update()

    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        # Cell widgets (row actions) take the whole cell and lay out their own padding; the style's
        # item text rect would inset them by the item padding.
        styled: Any = option  # the stubs omit QStyleOptionViewItem.rect
        editor.setGeometry(styled.rect)

    # what a cell shows, and where its text sits (shared by painting, hit tests and copying)

    def cell(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> Cell:
        value = index.data(CELL_ROLE)
        if isinstance(value, Cell):
            return value
        column = self.look.column(index.column())
        if column.kind == "check":
            state = index.data(QtCore.Qt.ItemDataRole.CheckStateRole)
            return Cell(
                kind="check", check=QtCore.Qt.CheckState(state) if state is not None else None
            )
        text = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
        kind = column.kind
        alignment = index.data(QtCore.Qt.ItemDataRole.TextAlignmentRole)
        if kind == "text" and alignment is not None:
            if QtCore.Qt.AlignmentFlag(int(alignment)) & QtCore.Qt.AlignmentFlag.AlignRight:
                kind = "number"
        return Cell("" if text is None else str(text), kind)

    def text(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> str:
        cell = self.cell(index)
        return "" if cell.kind in NO_TEXT else cell.text

    def row_selected(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, selected: bool
    ) -> bool:
        """Whether the row shows the selected fill and ink bar (tables with ticked rows override it)."""
        return selected

    def is_last_row(self, row: int) -> bool:
        model = self.view.model()
        return model is None or row >= model.rowCount() - 1

    def content_height(self, rect: QtCore.QRect, row: int) -> int:
        return rect.height() - (0 if self.is_last_row(row) else 1)

    def text_frame(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, rect: QtCore.QRect
    ) -> tuple[TextStyle, float, float, float, str]:
        """Style, x, line top and room of the cell's main text, and the text itself."""
        cell = self.cell(index)
        column = self.look.column(index.column())
        style = _style(cell.kind, self.look.text_size)
        left = float(rect.x() + column.padding_left)
        room = float(rect.width() - column.padding_left - column.padding_right)
        content = self.content_height(rect, index.row())
        text = cell.text
        if cell.kind == "rep":
            left += 14 + 6
            room -= 14 + 6
        unit = (2 + style.advance(cell.unit)) if cell.unit else 0.0
        if cell.note:  # a right-aligned value ends with its note ("30:00 limit")
            note_style, _, gap = NOTE_STYLES[cell.note_kind]
            unit += (style.advance(" ") if gap is None else gap) + note_style.advance(cell.note)
        width = style.advance(text) + unit
        if column.right(cell.kind):
            left = max(left, rect.x() + rect.width() - column.padding_right - width)
        line_top = rect.y() + style.middle_baseline(content) - style.baseline(0)
        return style, left, line_top, max(0.0, room - unit), text

    def hit(
        self, position: QtCore.QPoint, *, dragging: bool = False
    ) -> tuple[QtCore.QModelIndex, int] | None:
        """The cell and UTF-16 cursor position under ``position``, measured with the painted advances."""
        index = self.view.indexAt(position)
        if not index.isValid() or not self.text(index):
            return None
        flags = index.flags()
        if (
            flags & QtCore.Qt.ItemFlag.ItemIsEditable
            or not flags & QtCore.Qt.ItemFlag.ItemIsEnabled
        ):
            return None
        rect = self.view.visualRect(index)
        style, left, _, room, text = self.text_frame(index, rect)
        x = float(position.x())
        if dragging and x >= rect.right():
            return index, self.length(text)
        if not dragging and not left <= x <= left + style.advance(style.elide(text, room)):
            return None
        best, distance = 0, abs(x - left)
        for count in range(1, len(text) + 1):
            edge = left + style.advance(text[:count])
            if abs(edge - x) < distance:
                best, distance = count, abs(edge - x)
            elif edge > x:
                break
        return index, self.length(text[:best])

    # painting

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        theme = self.themes.current
        styled: Any = option  # the stubs omit QStyleOptionViewItem.rect and .state
        rect: QtCore.QRect = styled.rect
        row = index.row()
        selected = self.row_selected(
            index, bool(styled.state & QtWidgets.QStyle.StateFlag.State_Selected)
        )
        last = self.is_last_row(row)
        painter.save()
        fill = (
            "surface.selected"
            if selected
            else "surface.hover"
            if row == self.hover_row
            else "surface.card"
        )
        cell_rect = QtCore.QRectF(rect)
        bottom = self.viewport.height()
        if self.look.round_bottom and rect.bottom() + 1 > bottom - self.look.round_bottom:
            self._clip_bottom_corners(painter, cell_rect, index.column(), bottom)
        painter.fillRect(cell_rect, theme.color(fill))
        if not last:
            painter.fillRect(
                QtCore.QRect(rect.x(), rect.bottom(), rect.width(), 1), theme.color("border.line")
            )
        if selected and index.column() == 0:
            painter.fillRect(
                QtCore.QRect(rect.x(), rect.y(), 2, rect.height()), theme.color("text.ink")
            )
        if self._focus_row() == row:
            ink = theme.color("text.ink")
            painter.fillRect(QtCore.QRect(rect.x(), rect.y(), rect.width(), 2), ink)
            painter.fillRect(QtCore.QRect(rect.x(), rect.bottom() - 1, rect.width(), 2), ink)
            if index.column() == 0:
                painter.fillRect(QtCore.QRect(rect.x(), rect.y(), 2, rect.height()), ink)
        self._paint_content(painter, rect, index, theme)
        painter.restore()

    def _clip_bottom_corners(
        self, painter: QtGui.QPainter, rect: QtCore.QRectF, column: int, bottom: float
    ) -> None:
        """Clip fills at the viewport's bottom corners, which follow the card's rounded corners."""
        model = self.view.model()
        radius = self.look.round_bottom
        path = QtGui.QPainterPath()
        path.addRect(rect)
        corners = []
        if column == 0:
            corners.append((rect.x(), rect.x()))
        if model is not None and column == model.columnCount() - 1:
            corners.append((rect.right() - radius, rect.right() - 2 * radius))  # QRectF: x + width
        for corner_x, round_x in corners:
            corner = QtGui.QPainterPath()
            corner.addRect(QtCore.QRectF(corner_x, bottom - radius, radius, radius))
            round_part = rounded_path(
                QtCore.QRectF(round_x, bottom - 2 * radius, 2 * radius, 2 * radius), radius
            )
            path = path.subtracted(corner.subtracted(round_part))
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setClipPath(path)

    def _focus_row(self) -> int:
        """The current row when the table has keyboard focus (CSS ``:focus-visible``), else −1."""
        if not self.view.hasFocus() or not FocusRing.keyboard_visible():
            return -1
        current = self.view.currentIndex()
        return current.row() if current.isValid() else -1

    def _paint_content(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        theme: Theme,
    ) -> None:
        cell = self.cell(index)
        column = self.look.column(index.column())
        content = self.content_height(rect, index.row())
        style = _style(cell.kind, self.look.text_size)
        if cell.kind == "empty":
            return
        if cell.kind == "check":
            if cell.check is not None:
                top = rect.y() + style.middle_top(content, 16)
                enabled = bool(index.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
                box = QtCore.QRectF(rect.x() + column.padding_left, top, 16, 16)
                paint_check(painter, box, theme, check_mark(cell.check), enabled)
            return
        if cell.kind == "badge":
            if cell.badge is not None:
                top = rect.y() + style.middle_top(content, 22)
                x = rect.x() + column.padding_left
                width = paint_badge(painter, x, top, cell.badge, theme)
                if cell.note:
                    baseline = rect.y() + style.middle_baseline(content, 22)
                    self._paint_note(painter, x + width, baseline, cell, theme)
            return
        style, left, line_top, room, text = self.text_frame(index, rect)
        token = cell.token or (
            "text.tertiary"
            if index.data(DIM_ROLE)
            else "text.ink"
            if cell.kind == "name"
            else "text.secondary"
            if cell.kind == "rep"
            else "text.body"
        )
        if cell.kind == "rep" and cell.icon:
            top = rect.y() + style.middle_top(content, 18)
            gui_icons.draw(
                painter,
                cell.icon,
                theme.color("text.tertiary"),
                QtCore.QPointF(snap(rect.x() + column.padding_left), snap(top + 2)),
                14,
            )
            line_top = top
        span = self.span(QtCore.QModelIndex(cast(Any, index)))
        if span is not None and span[0] != span[1]:
            start = style.advance(self.substring(text, 0, span[0]))
            end = style.advance(self.substring(text, 0, span[1]))
            painter.fillRect(
                QtCore.QRectF(left + start, line_top, end - start, style.line),
                theme.color("accent.soft"),
            )
        shown = style.elide(text, room)
        advance = draw_text(painter, left, line_top, shown, style, theme.color(token))
        if cell.unit:
            unit_color = theme.color("text.tertiary")
            draw_text(painter, left + advance + 2, line_top, cell.unit, style, unit_color)
        if cell.note:
            self._paint_note(painter, left + advance, style.baseline(line_top), cell, theme)

    @staticmethod
    def _paint_note(
        painter: QtGui.QPainter, x: float, baseline: float, cell: Cell, theme: Theme
    ) -> None:
        note_style, token, gap = NOTE_STYLES[cell.note_kind]
        line_top = baseline - note_style.baseline(0)
        space = note_style.advance(" ") if gap is None else gap
        draw_text(painter, x + space, line_top, cell.note, note_style, theme.color(token))

    # hover tracking and tooltips

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.viewport:
            kind = event.type()
            if kind == QtCore.QEvent.Type.MouseMove and isinstance(event, QtGui.QMouseEvent):
                self._set_hover(self.view.rowAt(int(event.position().y())))
            elif kind == QtCore.QEvent.Type.Leave:
                self._set_hover(-1)
            elif kind == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
                index = self.view.indexAt(event.pos())
                if index.isValid():
                    cell = self.cell(index)
                    tip = cell.tooltip or index.data(QtCore.Qt.ItemDataRole.ToolTipRole) or ""
                    show_tooltip(self.viewport, event.globalPos(), str(tip))
                    return True
        return super().eventFilter(watched, event)

    def _set_hover(self, row: int) -> None:
        if row != self.hover_row:
            previous, self.hover_row = self.hover_row, row
            for changed in (previous, row):
                if changed >= 0:
                    top = self.view.rowViewportPosition(changed)
                    self.viewport.update(
                        QtCore.QRect(0, top, self.viewport.width(), self.view.rowHeight(changed))
                    )


# ---------- Adjustable column widths ----------

COLUMN_MIN = 48  # px a dragged column keeps
GRIP = 4  # px either side of a column's right edge that take a drag


class ColumnWidths:
    """The widths a reader dragged in one table (column index → px), remembered in the application's settings
    as ``table/<key>`` (JSON with the column count; widths stored for another count are ignored). An empty key
    remembers nothing."""

    def __init__(self, key: str, count: int) -> None:
        self.key = "table/" + key if key else ""
        self.count = count
        self.widths: dict[int, float] = {}
        stored = QtCore.QSettings().value(self.key) if self.key else None
        if stored is None:
            return
        try:
            data = json.loads(str(stored))
            if data.get("count") == count:
                for name, value in data["widths"].items():
                    index, width = int(name), float(value)
                    if 0 <= index < count and math.isfinite(width):
                        self.widths[index] = max(COLUMN_MIN, width)
        except (AttributeError, KeyError, TypeError, ValueError):
            self.widths = {}

    def save(self) -> None:
        if not self.key:
            return
        settings = QtCore.QSettings()
        if self.widths:
            widths = {str(index): round(width) for index, width in sorted(self.widths.items())}
            settings.setValue(self.key, json.dumps({"count": self.count, "widths": widths}))
        else:
            settings.remove(self.key)


def column_drag(
    widths: Sequence[float], movable: Sequence[bool], flex: int | None, index: int, delta: float
) -> tuple[int, float] | None:
    """Moving the right edge of column ``index`` by ``delta`` px with the table's width kept: the column that
    gives or takes the difference (the flexible column when it lies right of the edge, else the next movable
    column) and the delta kept within both columns' minimum; None when that edge does not move."""
    if not movable[index]:
        return None
    if flex is not None and flex > index and movable[flex]:
        other = flex
    else:
        other = next((i for i in range(index + 1, len(widths)) if movable[i]), -1)
        if other < 0:
            return None
    low = min(0.0, COLUMN_MIN - widths[index])
    high = max(0.0, widths[other] - COLUMN_MIN)
    return other, max(low, min(delta, high))


def grip_at(
    edges: Sequence[tuple[int, float]],
    x: float,
    widths: Sequence[float],
    movable: Sequence[bool],
    flex: int | None,
) -> int:
    """The column whose right edge (``edges``: column and edge x, left to right, the table's own right edge
    excluded) is within ``GRIP`` of ``x`` and can move; -1 for none."""
    for index, edge in edges:
        if abs(x - edge) <= GRIP and column_drag(widths, movable, flex, index, 0.0) is not None:
            return index
    return -1


def paint_grip(
    painter: QtGui.QPainter, theme: Theme, x: float, height: int, dragging: bool
) -> None:
    """The 2 px line on a hovered (``border.strong``) or dragged (``text.ink``) column edge."""
    color = theme.color("text.ink" if dragging else "border.strong")
    painter.fillRect(QtCore.QRectF(round(x) - 1, 0, 2, height), color)


class TableHeader(QtWidgets.QHeaderView):
    """``.tbl th``: 36 px, sunken fill, bottom rule, 12 px 500 secondary; the sorted column in ink with a
    14 px chevron; the checkbox column shows the header check (off, mixed or on)."""

    check_toggled = QtCore.Signal()

    def __init__(
        self,
        themes: ThemeManager,
        look: TableLook,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.themes, self.look = themes, look
        self.check_state: QtCore.Qt.CheckState | None = None
        self.sizes = ColumnWidths(look.key, len(look.columns))
        self.grip = -1  # the column whose right edge is hovered or dragged
        self._drag: float | None = None  # the pointer's x while dragging
        self.flex = next((i for i, column in enumerate(look.columns) if column.width is None), None)
        self.setProperty("designTable", True)
        self.setFixedHeight(look.header_height)
        self.setSectionsClickable(True)
        self.setHighlightSections(False)
        self.viewport().setMouseTracking(True)
        self.setDefaultAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        themes.changed.connect(self._restyle)

    def _restyle(self, *_: object) -> None:
        self.viewport().update()

    def set_check_state(self, state: QtCore.Qt.CheckState | None) -> None:
        if state != self.check_state:
            self.check_state = state
            self.viewport().update()

    # column widths

    def _columns(self) -> tuple[list[float], list[bool]]:
        widths, movable = [], []
        for logical in range(self.count()):
            hidden = self.isSectionHidden(logical)
            widths.append(0.0 if hidden else float(self.sectionSize(logical)))
            movable.append(not hidden and self.look.column(logical).kind != "check")
        return widths, movable

    def _grip_at(self, x: float) -> int:
        visible = [i for i in range(self.count()) if not self.isSectionHidden(i)]
        edges = [
            (i, float(self.sectionViewportPosition(i) + self.sectionSize(i))) for i in visible[:-1]
        ]
        widths, movable = self._columns()
        return grip_at(edges, x, widths, movable, self.flex)

    def _set_grip(self, grip: int) -> None:
        if grip != self.grip:
            self.grip = grip
            if grip >= 0:
                self.viewport().setCursor(QtCore.Qt.CursorShape.SplitHCursor)
            else:
                self.viewport().unsetCursor()
            self.viewport().update()

    def _resize(self, index: int, width: float) -> None:
        if index != self.flex:  # the flexible column stretches over the room left
            self.resizeSection(index, round(width))
            self.sizes.widths[index] = float(self.sectionSize(index))

    def restore_widths(self) -> None:
        """The design's widths (the flexible column takes the room left)."""
        self.sizes.widths.clear()
        self.sizes.save()
        for index, column in enumerate(self.look.columns):
            if column.width is not None and index < self.count():
                self.resizeSection(index, column.width)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = event.position().x()
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._grip_at(x) >= 0:
            self._set_grip(self._grip_at(x))
            self._drag = x
            return
        section = self.logicalIndexAt(event.position().toPoint())
        if self.look.column(section).kind == "check" and self.check_state is not None:
            self.check_toggled.emit()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        x = event.position().x()
        if self._drag is None:
            self._set_grip(self._grip_at(x))
            super().mouseMoveEvent(event)
            return
        widths, movable = self._columns()
        move = column_drag(widths, movable, self.flex, self.grip, x - self._drag)
        if move is None:
            return
        other, delta = move
        if delta:
            self._resize(self.grip, widths[self.grip] + delta)
            self._resize(other, widths[other] - delta)
            self._drag += delta

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        self.sizes.save()
        self._set_grip(self._grip_at(event.position().x()))

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._grip_at(event.position().x()) >= 0:
            self.restore_widths()
            return
        super().mouseDoubleClickEvent(event)

    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Leave and self._drag is None:
            self._set_grip(-1)
        return super().viewportEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self.viewport())
        theme = self.themes.current
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        height = self.look.header_height
        fill = QtCore.QRectF(0, 0, self.viewport().width(), height)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("surface.sunken"))
        if self.look.round_top:
            path = QtGui.QPainterPath()
            path.addRoundedRect(
                fill.adjusted(0, 0, 0, self.look.round_top),
                self.look.round_top,
                self.look.round_top,
            )
            clip = QtGui.QPainterPath()
            clip.addRect(fill)
            painter.drawPath(path.intersected(clip))
        else:
            painter.drawRect(fill)
        painter.fillRect(
            QtCore.QRect(0, height - 1, self.viewport().width(), 1), theme.color("border.line")
        )
        model = self.model()
        if model is None:
            return
        sorted_section = self.sortIndicatorSection() if self.isSortIndicatorShown() else -1
        line_top = HEADER_TEXT.middle_baseline(height - 1) - HEADER_TEXT.baseline(0)
        for logical in range(self.count()):
            if self.isSectionHidden(logical):
                continue
            x = self.sectionViewportPosition(logical)
            width = self.sectionSize(logical)
            column = self.look.column(logical)
            if column.kind == "check":
                if self.check_state is not None:
                    top = HEADER_TEXT.middle_top(height - 1, 16)
                    box = QtCore.QRectF(x + column.padding_left, top, 16, 16)
                    paint_check(painter, box, theme, check_mark(self.check_state))
                continue
            text = str(model.headerData(logical, QtCore.Qt.Orientation.Horizontal) or "")
            sorted_here = logical == sorted_section
            color = theme.color("text.ink" if sorted_here else "text.secondary")
            space = HEADER_TEXT.advance(" ")
            chevron = (14 + space) if sorted_here else 0.0
            room = width - column.padding_left - column.padding_right
            left = float(x + column.padding_left)
            if column.right() and HEADER_TEXT.advance(text) + chevron > room:
                shown = text  # CSS: an overflowing right-aligned title starts at the left padding
            else:
                shown = HEADER_TEXT.elide(text, room - chevron)
                if column.right():
                    left = x + width - column.padding_right - HEADER_TEXT.advance(shown) - chevron
            advance = draw_text(painter, left, line_top, shown, HEADER_TEXT, color)
            if sorted_here:
                descending = self.sortIndicatorOrder() == QtCore.Qt.SortOrder.DescendingOrder
                # ``vertical-align: -2px``: the 14 px chevron ends 2 px below the baseline
                baseline = HEADER_TEXT.baseline(line_top)
                origin = QtCore.QPointF(left + advance + space, snap(baseline + 2 - 14))
                gui_icons.draw(painter, "cdown" if descending else "cup", color, origin, 14)
        if self.grip >= 0:
            edge = self.sectionViewportPosition(self.grip) + self.sectionSize(self.grip)
            paint_grip(painter, theme, edge, height, self._drag is not None)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            model = self.model()
            section = self.logicalIndexAt(event.pos())
            tip = (
                model.headerData(
                    section, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.ToolTipRole
                )
                if model is not None and section >= 0
                else None
            )
            show_tooltip(self, event.globalPos(), str(tip or ""))
            return True
        return super().event(event)


class TableFooter(Themed):
    """``.tbl-foot``: 40 px, top rule, padding 0 16 (``padding``), 12.5 px ``text.secondary``, gap 8."""

    STYLE = TextStyle(12.5, 400, 18)

    def __init__(
        self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None, *, padding: int = 16
    ) -> None:
        super().__init__(themes, parent)
        self.padding = padding
        self.setFixedHeight(40)
        self.row = QtWidgets.QHBoxLayout(self)
        self.row.setContentsMargins(padding, 1, padding, 0)
        self.row.setSpacing(8)
        self.message = ""

    def set_text(self, text: str) -> None:
        if text != self.message:
            self.message = text
            self.setAccessibleName(text)
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), self.color("border.line"))
        if self.message:
            color = self.color("text.secondary")
            draw_text(painter, self.padding, 1 + (39 - 18) / 2, self.message, self.STYLE, color)


QUIET_SCROLL = "quietScroll"  # an area property: the scroll in progress shows no handle


def scroll_quietly(area: QtWidgets.QAbstractScrollArea, value: int) -> None:
    """Scroll ``area`` without showing its overlay handle (a scroll the view makes itself, such as following
    new rows)."""
    area.setProperty(QUIET_SCROLL, True)
    area.verticalScrollBar().setValue(value)
    area.setProperty(QUIET_SCROLL, False)


class OverlayScrollBar(Themed):
    """A thin vertical scroll handle over a scroll area's right edge, shown while the content overflows
    and the pointer is over the area or it scrolls (the design draws no scroll bars; the native bar is off
    so the columns keep their widths). Dragging the handle scrolls."""

    WIDTH, MARGIN, MINIMUM = 6, 3, 28

    def __init__(self, themes: ThemeManager, area: QtWidgets.QAbstractScrollArea) -> None:
        super().__init__(themes, area)
        self.area = area
        self.bar = area.verticalScrollBar()
        self.drag: tuple[float, int] | None = None
        self.hovered = False
        area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.hide_timer = QtCore.QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.setInterval(1200)
        self.hide_timer.timeout.connect(self._idle)
        self.bar.valueChanged.connect(self._scrolled)
        self.bar.rangeChanged.connect(self._place)
        area.viewport().installEventFilter(self)
        area.installEventFilter(self)
        self.hide()

    def _overflows(self) -> bool:
        return self.bar.maximum() > self.bar.minimum()

    def _place(self, *_: object) -> None:
        viewport = self.area.viewport()
        origin = viewport.mapTo(self.area, QtCore.QPoint(0, 0))
        self.setGeometry(
            origin.x() + viewport.width() - self.WIDTH - self.MARGIN,
            origin.y(),
            self.WIDTH + self.MARGIN,
            viewport.height(),
        )
        if not self._overflows():
            self.hide()
        self.update()

    def _scrolled(self, *_: object) -> None:
        if self._overflows() and not self.area.property(QUIET_SCROLL):  # see ``scroll_quietly``
            self._place()
            self.show()
            self.raise_()
            self.hide_timer.start()

    def _idle(self) -> None:
        if not self.hovered and self.drag is None:
            self.hide()

    def handle(self) -> QtCore.QRectF:
        span = self.bar.maximum() - self.bar.minimum() + self.bar.pageStep()
        track = self.height() - 2 * self.MARGIN
        length = max(self.MINIMUM, track * self.bar.pageStep() / max(1, span))
        room = max(1.0, track - length)
        travel = (self.bar.value() - self.bar.minimum()) / max(
            1, self.bar.maximum() - self.bar.minimum()
        )
        return QtCore.QRectF(0, self.MARGIN + room * travel, self.WIDTH, length)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if kind in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self._place()
        elif kind == QtCore.QEvent.Type.Enter and self._overflows():
            self.hovered = True
            self._place()
            self.show()
            self.raise_()
        elif kind == QtCore.QEvent.Type.Leave:
            self.hovered = False
            self.hide_timer.start()
        return False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        handle = self.handle()
        y = event.position().y()
        if handle.contains(QtCore.QPointF(handle.center().x(), y)):
            self.drag = (y, self.bar.value())
        else:  # page towards the click, as a native bar does
            step = self.bar.pageStep()
            self.bar.setValue(self.bar.value() + (step if y > handle.center().y() else -step))
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.drag is not None:
            start_y, start_value = self.drag
            room = max(1.0, self.height() - 2 * self.MARGIN - self.handle().height())
            span = self.bar.maximum() - self.bar.minimum()
            self.bar.setValue(round(start_value + (event.position().y() - start_y) / room * span))

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.drag = None
        self.hide_timer.start()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self._overflows():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.color("border.strong" if self.drag is None else "text.tertiary"))
        painter.drawPath(rounded_path(self.handle(), self.WIDTH / 2))


# ---------- painted tables ----------


PAINTED_HEADER = TextStyle(12, 500, 18)


@dataclass(frozen=True, slots=True)
class PaintedColumn:
    title: str
    width: float | None  # None: the room left
    padding: float = 12  # left padding (the first column: 20)
    padding_right: float = 12
    right: bool = False  # right-aligned (``.n``: numbers)


class PaintedHeader(Themed):
    """``.lh`` / ``.tbl th``: ``surface.sunken``, 12 px 500 ``text.secondary`` titles (right-aligned for
    ``right`` columns), bottom rule; as tall as the table's ``header_height``."""

    def __init__(self, table: "PaintedTable") -> None:
        super().__init__(table.themes, table)
        self.table = table
        self.grip = -1  # the column whose right edge is hovered or dragged
        self._drag: float | None = None  # the pointer's x while dragging
        self.setMouseTracking(True)
        self.setFixedHeight(table.header_height)

    def _grip_at(self, x: float) -> int:
        spans = self.table.column_spans()
        edges = [(i, left + width) for i, (left, width) in enumerate(spans[:-1])]
        widths = [width for _, width in spans]
        return grip_at(edges, x, widths, [True] * len(spans), self.table.flex())

    def _set_grip(self, grip: int) -> None:
        if grip != self.grip:
            self.grip = grip
            if grip >= 0:
                self.setCursor(QtCore.Qt.CursorShape.SplitHCursor)
            else:
                self.unsetCursor()
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x = event.position().x()
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._grip_at(x) >= 0:
            self._set_grip(self._grip_at(x))
            self._drag = x

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        x = event.position().x()
        if self._drag is None:
            self._set_grip(self._grip_at(x))
            return
        widths = [width for _, width in self.table.column_spans()]
        flex = self.table.flex()
        move = column_drag(widths, [True] * len(widths), flex, self.grip, x - self._drag)
        if move is None:
            return
        other, delta = move
        if delta:
            for index, width in (
                (self.grip, widths[self.grip] + delta),
                (other, widths[other] - delta),
            ):
                if index != flex:
                    self.table.sizes.widths[index] = width
            self._drag += delta
            self.table.widths_changed()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag is not None:
            self._drag = None
            self.table.sizes.save()
            self._set_grip(self._grip_at(event.position().x()))

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._grip_at(event.position().x()) >= 0:
            self.table.sizes.widths.clear()
            self.table.sizes.save()
            self.table.widths_changed()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self._drag is None:
            self._set_grip(-1)
        super().leaveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.color("surface.sunken"))
        painter.fillRect(
            QtCore.QRect(0, self.height() - 1, self.width(), 1), self.color("border.line")
        )
        color = self.color("text.secondary")
        top = (self.height() - 1 - 18) / 2
        # Titles are not cut: a header cell's text may run past its padding (``white-space: nowrap``).
        for (left, width), column in zip(
            self.table.column_spans(), self.table.columns, strict=True
        ):
            if not column.title:
                continue
            x = left + column.padding
            advance = PAINTED_HEADER.advance(column.title)
            # A right-aligned title wider than its cell starts at the cell's left padding (CSS overflow).
            if column.right and advance <= width - column.padding - column.padding_right:
                x = left + width - column.padding_right - advance
            draw_text(painter, x, top, column.title, PAINTED_HEADER, color)
        if self.grip >= 0:
            left, width = self.table.column_spans()[self.grip]
            paint_grip(
                painter, self.table.themes.current, left + width, self.height(), bool(self._drag)
            )


class PaintedTable(QtWidgets.QAbstractScrollArea):
    """A painted, virtual table under a ``PaintedHeader``: rows of ``row_height(index)`` painted by
    ``paint_row``, a thin scroll handle while the pointer is over it, and ``follow`` (keep the last row in
    view, as the Monitor's "Follow newest"). Scrolled to its end it shows whole rows from the top.
    ``reached_end`` is emitted when the reader scrolls to the end."""

    reached_end = QtCore.Signal()
    menu_requested = QtCore.Signal(QtCore.QPoint)  # a context menu at this viewport position

    def __init__(
        self,
        themes: ThemeManager,
        columns: Sequence[PaintedColumn],
        parent: QtWidgets.QWidget | None = None,
        *,
        header_height: int = 32,
        key: str = "",
    ) -> None:
        super().__init__(parent)
        self.themes = themes
        self.columns = list(columns)
        self.sizes = ColumnWidths(key, len(self.columns))  # dragged widths by column index
        self.header_height = header_height
        self.follow = False
        self._count = 0
        self._tops: list[int] = [0]  # row tops; the last entry is the total height
        self.header = PaintedHeader(self)
        self.setViewportMargins(0, header_height, 0, 0)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setMouseTracking(True)
        self.viewport().setAutoFillBackground(False)
        OverlayScrollBar(themes, self)
        self.verticalScrollBar().valueChanged.connect(self._scrolled)
        themes.changed.connect(self.viewport().update)

    # rows

    def row_count(self) -> int:
        return self._count

    def row_height(self, index: int) -> int:
        raise NotImplementedError

    def paint_row(self, painter: QtGui.QPainter, index: int, rect: QtCore.QRect) -> None:
        raise NotImplementedError

    def rows_changed(self, count: int) -> None:
        """The rows were replaced: lay out ``count`` rows and keep the last in view when following."""
        self._count = count
        self.layout_rows()
        if self.follow:
            self.scroll_to_end()
        self.viewport().update()

    def layout_rows(self) -> None:
        """Row tops from the row heights (which may depend on the width)."""
        tops = [0]
        for index in range(self._count):
            tops.append(tops[-1] + self.row_height(index))
        self._tops = tops
        self._update_range()

    def scroll_to_end(self) -> None:
        scroll_quietly(self, self.verticalScrollBar().maximum())

    def _update_range(self) -> None:
        bar = self.verticalScrollBar()
        page = self.viewport().height()
        end = self._tops[-1] - page
        if end > 0:
            # Scrolled to its end the table shows whole rows from the top (as 10 and 11 draw them); the
            # room under the last row is less than a row.
            end = self._tops[min(bisect.bisect_left(self._tops, end), len(self._tops) - 1)]
        bar.setRange(0, max(0, end))
        bar.setPageStep(max(1, page))
        bar.setSingleStep(24)

    def row_at(self, y: float) -> int:
        """The row under viewport ``y`` (-1 for none)."""
        position = y + self.verticalScrollBar().value()
        if position < 0 or position >= self._tops[-1]:
            return -1
        low, high = 0, self.row_count() - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self._tops[middle] <= position:
                low = middle
            else:
                high = middle - 1
        return low

    def row_rect(self, index: int) -> QtCore.QRect:
        top = self._tops[index] - self.verticalScrollBar().value()
        return QtCore.QRect(
            0, top, self.viewport().width(), self._tops[index + 1] - self._tops[index]
        )

    def flex(self) -> int | None:
        """The flexible column (the one without a width), if any."""
        return next((i for i, column in enumerate(self.columns) if column.width is None), None)

    def column_spans(self) -> list[tuple[float, float]]:
        """Each column's left edge and width: the dragged width, else the column's own; the flexible column
        takes the room left."""
        dragged = self.sizes.widths
        widths = [
            dragged.get(i, column.width) if column.width is not None else None
            for i, column in enumerate(self.columns)
        ]
        rest = max(0.0, self.viewport().width() - sum(width or 0.0 for width in widths))
        spans, x = [], 0.0
        for width in widths:
            span = width if width is not None else rest
            spans.append((x, span))
            x += span
        return spans

    def widths_changed(self) -> None:
        """Lay the rows out again for new column widths (row heights may depend on them)."""
        self.layout_rows()
        self.header.update()
        self.viewport().update()

    # events

    def _scrolled(self, value: int) -> None:
        self.viewport().update()
        # The reader (wheel, keys or the handle) reached the end; following new rows does not count.
        if not self.property(QUIET_SCROLL) and value >= self.verticalScrollBar().maximum() > 0:
            self.reached_end.emit()

    def _check_end(self) -> None:
        bar = self.verticalScrollBar()
        if bar.value() >= bar.maximum():
            self.reached_end.emit()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        bar = self.verticalScrollBar()
        before = bar.value()
        bar.setValue(round(bar.value() - wheel_step(event, 48).y()))
        if bar.value() != before:
            event.accept()
        elif before >= bar.maximum():
            event.accept()
            self._check_end()  # turning the wheel at the end asks for more
        else:
            event.ignore()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.header.setGeometry(0, 0, self.width(), self.header_height)
        at_end = self.verticalScrollBar().value() >= self.verticalScrollBar().maximum()
        self.layout_rows()
        if self.follow or at_end:
            self.scroll_to_end()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), self.themes.current.color("surface.card"))
        count = self.row_count()
        if not count:
            return
        first = max(0, self.row_at(0))
        height = self.viewport().height()
        for index in range(first, count):
            rect = self.row_rect(index)
            if rect.top() >= height:
                break
            self.paint_row(painter, index, rect)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        # The scroll area receives its viewport's context menu events (viewport coordinates).
        self.menu_requested.emit(event.pos())
        event.accept()

    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            index = self.row_at(event.pos().y())
            tip = self.tooltip_at(index, event.pos()) if index >= 0 else ""
            show_tooltip(self.viewport(), event.globalPos(), tip)
            return True
        return super().viewportEvent(event)

    def tooltip_at(self, index: int, position: QtCore.QPoint) -> str:
        return ""


def style_table(
    view: QtWidgets.QTableView,
    themes: ThemeManager,
    look: TableLook,
    *,
    delegate_class: type[CellDelegate] = CellDelegate,
) -> CellDelegate:
    """Apply the design to a table view: header, delegate, rows, no grid, no zebra; returns the delegate."""
    header = TableHeader(themes, look, view)
    view.setProperty("designTable", True)
    view.style().unpolish(view)
    view.style().polish(view)
    view.setHorizontalHeader(header)
    view.setShowGrid(False)
    view.setAlternatingRowColors(False)
    view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    view.setWordWrap(False)
    view.verticalHeader().hide()
    view.verticalHeader().setDefaultSectionSize(look.row_height)
    view.verticalHeader().setMinimumSectionSize(look.row_height)
    view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    OverlayScrollBar(themes, view)
    header.setStretchLastSection(False)
    for index, column in enumerate(look.columns):
        if column.width is None:
            header.setSectionResizeMode(index, QtWidgets.QHeaderView.ResizeMode.Stretch)
        else:
            # The header resizes columns itself (``TableHeader``): Qt's own section dragging is off.
            header.setSectionResizeMode(index, QtWidgets.QHeaderView.ResizeMode.Fixed)
            header.resizeSection(index, round(header.sizes.widths.get(index, column.width)))
    previous = view.itemDelegate()
    delegate = delegate_class(view, themes, look)
    view.setItemDelegate(delegate)
    if previous is not None and previous.parent() is view:
        previous.deleteLater()  # its event filters go with it
    view.setMouseTracking(True)
    return delegate
