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

from typing import Any, cast

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from .gui_style import COLORS


class TableTextDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, view: QtWidgets.QTableView):
        super().__init__(view)
        self.view = view
        self.viewport = view.viewport()
        self.start: tuple[QtCore.QPersistentModelIndex, int] | None = None
        self.end: tuple[QtCore.QPersistentModelIndex, int] | None = None
        self.anchor: tuple[QtCore.QPersistentModelIndex, int] | None = None
        self.dragging = False
        self.snapshot = ""
        self.viewport.installEventFilter(self)
        view.installEventFilter(self)
        view.setMouseTracking(True)
        model = view.model()
        if model is not None:
            model.modelAboutToBeReset.connect(self.clear)
            model.layoutAboutToBeChanged.connect(self.clear)
            model.rowsAboutToBeRemoved.connect(self.clear)
            model.columnsAboutToBeRemoved.connect(self.clear)
            model.dataChanged.connect(self.data_changed)

    @staticmethod
    def length(text: str) -> int:
        return len(text.encode("utf-16-le")) // 2

    @staticmethod
    def substring(text: str, start: int, end: int) -> str:
        return text.encode("utf-16-le")[2 * start : 2 * end].decode("utf-16-le")

    def text(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> str:
        return self.displayText(index.data(), self.view.locale())

    def bounds(self) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        if self.start is None or self.end is None:
            return None
        if not self.start[0].isValid() or not self.end[0].isValid():
            return None
        first, last = sorted(
            ((index.row(), index.column(), position) for index, position in (self.start, self.end))
        )
        return (first, last) if first != last else None

    def span(self, index: QtCore.QModelIndex) -> tuple[int, int] | None:
        bounds = self.bounds()
        if bounds is None:
            return None
        first, last = bounds
        cell = index.row(), index.column()
        if not first[:2] <= cell <= last[:2]:
            return None
        start = first[2] if cell == first[:2] else 0
        end = last[2] if cell == last[:2] else self.length(self.text(index))
        return start, end

    def selected_text(self) -> str:
        bounds, model = self.bounds(), self.view.model()
        if bounds is None or model is None:
            return ""
        first, last = bounds
        rows = []
        for row in range(first[0], last[0] + 1):
            columns = range(
                first[1] if row == first[0] else 0,
                (last[1] + 1) if row == last[0] else model.columnCount(),
            )
            values = []
            for column in columns:
                index = model.index(row, column)
                span = self.span(index)
                if span is not None:
                    values.append(self.substring(self.text(index), *span))
            rows.append("\t".join(values))
        return "\n".join(rows)

    def clear(self, *args: Any) -> None:
        self.start = self.end = self.anchor = None
        self.dragging = False
        self.snapshot = ""
        if isValid(self.viewport):
            self.viewport.update()

    def data_changed(self, *args: Any) -> None:
        if self.bounds() is not None:
            try:
                changed = self.selected_text() != self.snapshot
            except UnicodeError:
                changed = True
            if changed:
                self.clear()

    def text_layout(
        self, index: QtCore.QModelIndex, option: QtWidgets.QStyleOptionViewItem | None = None
    ) -> tuple[QtGui.QTextLayout, QtCore.QPointF, QtCore.QRect]:
        item: Any = (
            QtWidgets.QStyleOptionViewItem(option)
            if option is not None
            else QtWidgets.QStyleOptionViewItem()
        )
        if option is None:
            item.initFrom(self.view)
            item.font = self.view.font()
            item.fontMetrics = QtGui.QFontMetrics(item.font)
        self.initStyleOption(item, index)
        if option is None:
            item.rect = self.view.visualRect(index)
        rect = self.view.style().subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText, item, self.view
        )
        margin = (
            self.view.style().pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_FocusFrameHMargin, item, self.view
            )
            + 1
        )
        rect.adjust(margin, 0, -margin, 0)
        layout = QtGui.QTextLayout(self.text(index), item.font)
        text_option = QtGui.QTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
        text_option.setTextDirection(self.view.layoutDirection())
        text_option.setAlignment(
            item.displayAlignment & QtCore.Qt.AlignmentFlag.AlignHorizontal_Mask
        )
        layout.setTextOption(text_option)
        layout.beginLayout()
        line = layout.createLine()
        line.setLineWidth(max(1, rect.width()))
        layout.endLayout()
        origin = QtCore.QPointF(rect.left(), rect.top() + (rect.height() - line.height()) / 2)
        return layout, origin, rect

    def hit(
        self, position: QtCore.QPoint, *, dragging: bool = False
    ) -> tuple[QtCore.QModelIndex, int] | None:
        index = self.view.indexAt(position)
        if not index.isValid() or not self.text(index):
            return None
        if index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
            return None
        if not index.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
            return None
        layout, origin, rect = self.text_layout(index)
        line = layout.lineAt(0)
        natural = line.naturalTextRect().translated(origin).intersected(QtCore.QRectF(rect))
        if not dragging and not natural.contains(QtCore.QPointF(position)):
            return None
        cursor = line.xToCursor(position.x() - origin.x())
        if dragging and position.x() >= rect.right():
            cursor = self.length(self.text(index))
        return index, cursor

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        index = QtCore.QModelIndex(cast(Any, index))
        span = self.span(index)
        if span is None or span[0] == span[1]:
            super().paint(painter, option, index)
            return
        item: Any = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(item, index)
        layout, origin, rect = self.text_layout(index, item)
        item.text = ""
        self.view.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_ItemViewItem, item, painter, self.view
        )
        selected: Any = QtGui.QTextLayout.FormatRange()
        selected.start, selected.length = span[0], span[1] - span[0]
        selected.format.setBackground(QtGui.QColor(COLORS["action.primary"]))
        selected.format.setForeground(QtGui.QColor(COLORS["text.on-accent"]))
        painter.save()
        painter.setClipRect(rect)
        role = (
            QtGui.QPalette.ColorRole.HighlightedText
            if item.state & QtWidgets.QStyle.StateFlag.State_Selected
            else QtGui.QPalette.ColorRole.Text
        )
        painter.setPen(item.palette.color(role))
        layout.draw(painter, origin, [selected])
        painter.restore()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.view and isinstance(event, QtGui.QKeyEvent):
            if event.type() == QtCore.QEvent.Type.KeyPress and (
                event.key()
                in {
                    QtCore.Qt.Key.Key_Left,
                    QtCore.Qt.Key.Key_Right,
                    QtCore.Qt.Key.Key_Up,
                    QtCore.Qt.Key.Key_Down,
                    QtCore.Qt.Key.Key_Home,
                    QtCore.Qt.Key.Key_End,
                    QtCore.Qt.Key.Key_PageUp,
                    QtCore.Qt.Key.Key_PageDown,
                    QtCore.Qt.Key.Key_Escape,
                }
                or event.matches(QtGui.QKeySequence.StandardKey.SelectAll)
            ):
                self.clear()
        if watched is self.view:
            return False
        if watched is not self.viewport:
            return super().eventFilter(watched, event)
        if not isinstance(event, QtGui.QMouseEvent):
            return False
        position = event.position().toPoint()
        if (
            event.type() == QtCore.QEvent.Type.MouseButtonPress
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self.clear()
            hit = self.hit(position)
            if hit is not None:
                self.anchor = QtCore.QPersistentModelIndex(hit[0]), hit[1]
        elif event.type() == QtCore.QEvent.Type.MouseMove:
            if event.buttons() & QtCore.Qt.MouseButton.LeftButton and self.anchor is not None:
                hit = self.hit(position, dragging=True)
                if hit is not None:
                    self.start = self.anchor
                    self.end = QtCore.QPersistentModelIndex(hit[0]), hit[1]
                    self.dragging = self.bounds() is not None
                    self.snapshot = self.selected_text()
                    self.viewport.update()
                return True
            elif not event.buttons():
                self.viewport.setCursor(
                    QtCore.Qt.CursorShape.IBeamCursor
                    if self.hit(position)
                    else QtCore.Qt.CursorShape.ArrowCursor
                )
        elif (
            event.type() == QtCore.QEvent.Type.MouseButtonRelease
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            dragging = self.dragging
            self.anchor = None
            self.dragging = False
            if dragging:
                return True
        elif (
            event.type() == QtCore.QEvent.Type.MouseButtonDblClick
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            hit = self.hit(position)
            if hit is not None:
                index, position_in_text = hit
                document = QtGui.QTextDocument()
                document.setPlainText(self.text(index))
                cursor = QtGui.QTextCursor(document)
                cursor.setPosition(position_in_text)
                cursor.select(QtGui.QTextCursor.SelectionType.WordUnderCursor)
                self.start = QtCore.QPersistentModelIndex(index), cursor.selectionStart()
                self.end = QtCore.QPersistentModelIndex(index), cursor.selectionEnd()
                self.anchor = None
                self.snapshot = self.selected_text()
                self.view.selectionModel().setCurrentIndex(
                    index,
                    QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                )
                self.viewport.update()
                return True
        return False
