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

"""Confirmation dialogs of the Main design (02 and 12, HANDOFF §8.2, §8.13): a title, a context line, the
consequences as a sentence or a plain bulleted list, an optional list of items and a footnote, then Cancel
(focused) and the action. Shown in the window's ``ModalHost``."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_theme import ThemeManager
from .gui_widgets import (
    BODY,
    BODY_TABULAR,
    DialogCard,
    TextStyle,
    Themed,
    draw_text,
    rounded_path,
)

BOLD = TextStyle(13, 600, 18, tabular=True)  # figures in a sentence (``b.tnum``)
STRONG = TextStyle(13, 600, 18)  # item names (``.strong``)
MONO = TextStyle(12, 400, 18, mono=True)
FOOTNOTE = TextStyle(12.5, 400, 17)
REASON = TextStyle(12.5, 400, 18)


@dataclass(frozen=True, slots=True)
class Run:
    """A piece of text in one style and colour token."""

    text: str
    style: TextStyle = BODY
    token: str = "text.secondary"


def wrap(runs: Sequence[Run], width: float) -> list[list[Run]]:
    """Runs broken into lines at spaces so that no line is wider than ``width`` (a word longer than the
    width keeps a line of its own)."""
    lines: list[list[Run]] = [[]]
    used = 0.0
    for run in runs:
        words = run.text.split(" ")
        for index, word in enumerate(words):
            piece = word if index == 0 else " " + word
            size = run.style.advance(piece)
            if lines[-1] and used + size > width and index > 0:
                lines.append([])
                used = 0.0
                piece = word
                size = run.style.advance(piece)
            if piece:
                lines[-1].append(Run(piece, run.style, run.token))
                used += size
    return lines


def wrap_text(text: str, style: TextStyle, width: float, *, limit: int | None = None) -> list[str]:
    """``text`` in one style as ``wrap`` breaks it; with ``limit``, the lines past it join the last one, which is
    elided to ``width``."""
    lines = ["".join(run.text for run in line) for line in wrap([Run(text, style)], width)]
    if limit is not None and len(lines) > limit:
        lines = [*lines[: limit - 1], style.elide(" ".join(lines[limit - 1 :]), width)]
    return lines or [""]


def bullet(*parts: str | Run) -> list[Run]:
    """A bullet item's runs: plain strings in ``text.body`` (the list's colour), runs as given."""
    return [Run(part, token="text.body") if isinstance(part, str) else part for part in parts]


class Paragraph(Themed):
    """Rich text in runs, wrapped to the widget's width; lines ``line`` px apart."""

    def __init__(
        self,
        themes: ThemeManager,
        runs: Sequence[Run],
        *,
        line: float = 18,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.runs, self.line = list(runs), line
        self.setAccessibleName("".join(run.text for run in self.runs))
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred
        )
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return math.ceil(self.line * len(wrap(self.runs, width)))

    def sizeHint(self) -> QtCore.QSize:
        width = sum(run.style.advance(run.text) for run in self.runs)
        return QtCore.QSize(math.ceil(width), math.ceil(self.line))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        for row, runs in enumerate(wrap(self.runs, self.width())):
            x = 0.0
            for run in runs:
                top = row * self.line + (self.line - run.style.line) / 2
                x += draw_text(painter, x, top, run.text, run.style, self.color(run.token))


class Bullets(Themed):
    """``ul``: padding-left 18, items 8 px apart, ``text.tertiary`` disc markers; items are ``text.body``
    (``bullet`` builds their runs)."""

    def __init__(
        self,
        themes: ThemeManager,
        items: Sequence[Sequence[Run]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.items = [list(item) for item in items]
        self.setAccessibleName(" ".join("".join(r.text for r in item) for item in self.items))
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred
        )
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def _lines(self, width: float) -> list[list[list[Run]]]:
        return [wrap(item, width - 18) for item in self.items]

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        lines = self._lines(width)
        return sum(18 * len(item) for item in lines) + 8 * max(0, len(lines) - 1)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(480, self.heightForWidth(480))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        y = 0.0
        for item in self._lines(self.width()):
            # The disc marker sits in the 18 px padding, centred on the first line's x-height.
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self.color("text.tertiary"))
            baseline = BODY.baseline(y)
            painter.drawEllipse(QtCore.QPointF(7.5, baseline - 3.4), 2.4, 2.4)
            for row, runs in enumerate(item):
                x = 18.0
                for run in runs:
                    top = y + 18 * row + (18 - run.style.line) / 2
                    x += draw_text(painter, x, top, run.text, run.style, self.color(run.token))
            y += 18 * len(item) + 8


class RuledNote(Themed):
    """``.fn``: a 1 px rule, 14 px padding, then 12.5/17 ``text.secondary`` (wrapped)."""

    def __init__(
        self, themes: ThemeManager, text: str, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.text = text
        self.setAccessibleName(text)
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred
        )
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return 15 + 17 * len(wrap([Run(self.text, FOOTNOTE)], width))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(480, self.heightForWidth(480))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), self.color("border.line"))
        for row, runs in enumerate(wrap([Run(self.text, FOOTNOTE)], self.width())):
            x = 0.0
            for run in runs:
                x += draw_text(
                    painter, x, 15 + 17 * row, run.text, FOOTNOTE, self.color("text.secondary")
                )


@dataclass(frozen=True, slots=True)
class ListRow:
    """One row of a dialog's item list: name (600 ink), ID (mono, tertiary), the detail at the right, and an
    optional reason in ``state.bad`` (items the action skips)."""

    name: str
    identity: str
    detail: str
    reason: str = ""


class ItemList(Themed):
    """``.dl``: rows of 38 px with rules (name ``.strong``, mono ID, ``.tnum`` detail); at most ``visible``
    rows, then it scrolls inside the dialog (wheel, arrows, Page Up and Down, Home, End) with a scroll handle
    always shown, so no row goes unseen before an irreversible action."""

    ROW = 38
    HANDLE, HANDLE_MARGIN = 6, 3

    def __init__(
        self,
        themes: ThemeManager,
        rows: Sequence[ListRow],
        *,
        visible: int = 6,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.rows, self.visible, self.offset = list(rows), visible, 0
        self.setFixedHeight(1 + self.ROW * min(len(self.rows), visible))
        self.setAccessibleName(
            "; ".join(
                f"{row.name} {row.identity} {row.detail} {row.reason}".strip() for row in rows
            )
        )
        self.setFocusPolicy(
            QtCore.Qt.FocusPolicy.WheelFocus
            if len(rows) > visible
            else QtCore.Qt.FocusPolicy.NoFocus
        )

    def scrolls(self) -> bool:
        return len(self.rows) > self.visible

    def scroll_to(self, offset: int) -> None:
        offset = min(max(0, len(self.rows) - self.visible), max(0, offset))
        if offset != self.offset:
            self.offset = offset
            self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self.scroll_to(self.offset + (-1 if event.angleDelta().y() > 0 else 1))
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        steps = {
            QtCore.Qt.Key.Key_Up: -1,
            QtCore.Qt.Key.Key_Down: 1,
            QtCore.Qt.Key.Key_PageUp: -self.visible,
            QtCore.Qt.Key.Key_PageDown: self.visible,
            QtCore.Qt.Key.Key_Home: -len(self.rows),
            QtCore.Qt.Key.Key_End: len(self.rows),
        }
        step = steps.get(QtCore.Qt.Key(event.key()))
        if step is None or not self.scrolls():
            super().keyPressEvent(event)
            return
        self.scroll_to(self.offset + step)
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        rule = self.color("border.line")
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), rule)
        # A scrolling list keeps the handle's strip free of text.
        width = self.width() - (self.HANDLE + 2 * self.HANDLE_MARGIN if self.scrolls() else 0)
        for slot, row in enumerate(self.rows[self.offset : self.offset + self.visible]):
            top = 1 + slot * self.ROW
            painter.fillRect(QtCore.QRect(0, top + self.ROW - 1, width, 1), rule)
            line_top = top + (self.ROW - 1 - 18) / 2
            detail = row.reason or row.detail
            detail_style = REASON if row.reason else BODY_TABULAR
            detail_token = "state.bad" if row.reason else "text.secondary"
            detail_width = detail_style.advance(detail)
            room = width - detail_width - 10 - MONO.advance(row.identity) - 10
            name = STRONG.elide(row.name, max(0.0, room))
            x = draw_text(painter, 0, line_top, name, STRONG, self.color("text.ink")) + 10
            draw_text(painter, x, line_top, row.identity, MONO, self.color("text.tertiary"))
            draw_text(
                painter,
                width - detail_width,
                line_top,
                detail,
                detail_style,
                self.color(detail_token),
            )
        if self.scrolls():
            track = self.height() - 1 - 2 * self.HANDLE_MARGIN
            length = max(28.0, track * self.visible / len(self.rows))
            travel = self.offset / (len(self.rows) - self.visible)
            handle = QtCore.QRectF(
                self.width() - self.HANDLE - self.HANDLE_MARGIN,
                1 + self.HANDLE_MARGIN + (track - length) * travel,
                self.HANDLE,
                length,
            )
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self.color("border.strong"))
            painter.drawPath(rounded_path(handle, self.HANDLE / 2))


class Consequences(DialogCard):
    """A ``DialogCard`` body built from the design's blocks, each with its CSS margin above it."""

    def add_block(self, widget: QtWidgets.QWidget, margin: int) -> QtWidgets.QWidget:
        self.body.addSpacing(margin)
        self.body.addWidget(widget)
        return widget

    def add_context(self, text: str) -> None:
        """``.ctx2``: the item the action applies to, 4 px under the title."""
        self.add_block(Paragraph(self.themes, [Run(text)]), 4)

    def add_sentence(self, runs: Sequence[Run], margin: int = 6) -> None:
        self.add_block(Paragraph(self.themes, runs), margin)

    def add_bullets(self, items: Sequence[Sequence[Run]]) -> None:
        self.add_block(Bullets(self.themes, items), 16)

    def add_items(self, rows: Sequence[ListRow]) -> None:
        self.add_block(ItemList(self.themes, rows), 16)

    def add_footnote(self, text: str, *, ruled: bool) -> None:
        if ruled:
            self.add_block(RuledNote(self.themes, text), 16)
        else:
            self.add_block(Paragraph(self.themes, [Run(text)]), 14)
