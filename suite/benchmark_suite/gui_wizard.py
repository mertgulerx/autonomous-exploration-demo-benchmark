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

"""What the New campaign steps of the Main design share (03–08, HANDOFF §8.3–§8.7).

- The frame: the head with the back link, the title and the step indicator, and the bottom bar with Import
  and Export YAML, the totals and the step buttons.
- Steps 1 and 2: the card grid, the toggling card, the popups' head and opening button, and the selection
  rules of their lists (``toggle_reference``, ``toggle_shown``).
- Step 3: the campaign settings card and the draft summary.
"""

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Literal

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_controls import CheckBox, KeySelect, SearchField, SpinBox, Switch
from .gui_theme import ThemeManager, border_width, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    Button,
    Card,
    CardHead,
    CssBox,
    FlexRow,
    LinkButton,
    Rule,
    SegmentedControl,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    place,
    press_on_enter,
)

StepState = Literal["current", "done", "future"]
# A profile or spawn as a campaign names it: "Default", a saved ID, or an inline snapshot.
Reference = str | dict[str, Any]

WIZARD_TITLE = TextStyle(22, 600, 28)  # the wizard's h1 has no page-head letter spacing
STEP_NUMBER = TextStyle(12, 600, 18, tabular=True)
TOTAL_NUMBER = TextStyle(15, 600, 18, tabular=True)
GRAND_NUMBER = TextStyle(20, 600, 18, tabular=True)
STEPS = ("Robots & profiles", "Worlds & spawns", "Execution", "Review")


class BackLink(QtWidgets.QAbstractButton, CssBox):
    """``.back``: the ``aleft`` icon, 6 px gap, 13 px 500 ``text.secondary``; Enter or a click activates."""

    OVERHANG = True

    def __init__(
        self, themes: ThemeManager, text: str, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes = themes
        self.setText(text)
        self.setProperty("focusRadius", 2)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        themes.changed.connect(self.update)

    def natural_width(self) -> float:
        return 16 + 6 + BODY_MEDIUM.advance(self.text())

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 18)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not press_on_enter(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        color = self.themes.current.color("text.secondary")
        top = (self.height() - 18) / 2
        gui_icons.draw(painter, "aleft", color, QtCore.QPointF(snap(self.shift), snap(top + 1)), 16)
        draw_text(painter, self.shift + 22, top, self.text(), BODY_MEDIUM, color)


class StepButton(QtWidgets.QAbstractButton, CssBox):
    """``.step``: a 22 px circle (number; a check when done) and the label 8 px after it, 13 px 500.

    Current: ink circle, ``text.onInk`` number, ink label. Done: card circle with an ink border and an ink
    check. Future: ``border.strong`` circle, ``text.secondary`` number and label.
    """

    OVERHANG = True

    def __init__(
        self,
        themes: ThemeManager,
        number: int,
        text: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes, self.number = themes, number
        self.state: StepState = "future"
        self.setText(text)
        self.setProperty("focusRadius", 11)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(f"Step {number}: {text}")
        themes.changed.connect(self.update)

    def set_state(self, state: StepState) -> None:
        if state != self.state:
            self.state = state
            self.setAccessibleDescription(
                {"current": "current step", "done": "done"}.get(state, "")
            )
            self.update()

    def natural_width(self) -> float:
        return 22 + 8 + BODY_MEDIUM.advance(self.text())

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 22)

    def painted_rect(self) -> QtCore.QRectF:
        """The circle, which the focus ring follows."""
        return QtCore.QRectF(snap(self.shift), (self.height() - 22) // 2, 22, 22)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not press_on_enter(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():  # a step a tuning run skips
            painter.setOpacity(0.4)
        theme = self.themes.current
        circle = self.painted_rect()
        ink = theme.color("text.ink")
        fill = {"current": ink, "done": theme.color("surface.card")}.get(
            self.state, theme.color("surface.card")
        )
        line = border_width(painter, 1.5)
        painter.setPen(
            QtGui.QPen(ink if self.state != "future" else theme.color("border.strong"), line)
        )
        painter.setBrush(fill)
        painter.drawEllipse(circle.adjusted(line / 2, line / 2, -line / 2, -line / 2))
        if self.state == "done":
            # ``.sn svg.i.sm`` (14 px) with a 2.4 px stroke, centred in the circle.
            origin = QtCore.QPointF(circle.x() + 4, circle.y() + 4)
            gui_icons.draw(painter, "check", ink, origin, 14, stroke=2.4)
        else:
            digit = str(self.number)
            color = theme.color("text.onInk" if self.state == "current" else "text.secondary")
            x = circle.x() + (22 - STEP_NUMBER.advance(digit)) / 2
            draw_text(painter, x, circle.y() + 2, digit, STEP_NUMBER, color)
        label = theme.color("text.secondary" if self.state == "future" else "text.ink")
        draw_text(
            painter, self.shift + 30, (self.height() - 18) / 2, self.text(), BODY_MEDIUM, label
        )


class StepIndicator(FlexRow):
    """``.steps``: the four steps with 28 px ``border.strong`` connectors, 10 px apart. Steps are clickable
    (as the wizard's step buttons were)."""

    step_selected = QtCore.Signal(int)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, gap=10)
        self.buttons: list[StepButton] = []
        for index, title in enumerate(STEPS):
            if index:
                connector = Rule(themes, token="border.strong")
                connector.setFixedSize(28, 1)
                self.add(connector, width=28)
            step = StepButton(themes, index + 1, title)
            step.clicked.connect(lambda checked=False, i=index: self.step_selected.emit(i))
            self.add(step)
            self.buttons.append(step)
        self.setFixedHeight(22)

    def natural_width(self) -> float:
        return sum(step.natural_width() for step in self.buttons) + 3 * (10 + 28 + 10)

    def set_step(self, current: int, done: Sequence[int]) -> None:
        for index, step in enumerate(self.buttons):
            step.set_state("current" if index == current else "done" if index in done else "future")


class WizardHead(FlexRow):
    """``.wiz-head`` (36 px, gap 16): back link, vertical rule, "New campaign", and the steps at the right."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, gap=16)
        self.setFixedHeight(36)
        self.back = BackLink(themes, "Campaigns")
        self.back.setAccessibleName("Back to campaigns")
        self.add(self.back)
        rule = Rule(themes, vertical=True)
        rule.setFixedHeight(28)
        self.add(rule, width=1, margin=4)
        self.add(TextLine(themes, "New campaign", WIZARD_TITLE, "text.ink"))
        self.add_stretch()
        self.steps = StepIndicator(themes)
        self.add(self.steps)


class Totals(Themed, CssBox):
    """``.totals``: a left rule, then "2 robots · 3 profiles · 2 worlds · 5 repetitions" (numbers 15 px 600
    ink on the words' baseline, 22 px apart) and, after its own rule, "40 planned experiments" (20 px)."""

    OVERHANG = True

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.setFixedHeight(32)

    def set_counts(
        self, robots: int, profiles: int, worlds: int, repetitions: int, experiments: int
    ) -> None:
        counts = (robots, profiles, worlds, repetitions, experiments)
        if counts != self.counts:
            self.counts = counts
            self.setAccessibleName(
                ", ".join(f"{number:,} {word}" for number, word, _ in self.groups())
            )
            self.updateGeometry()
            self.update()

    def groups(self) -> list[tuple[int, str, TextStyle]]:
        robots, profiles, worlds, repetitions, experiments = self.counts
        return [
            (robots, "robot" if robots == 1 else "robots", TOTAL_NUMBER),
            (profiles, "profile" if profiles == 1 else "profiles", TOTAL_NUMBER),
            (worlds, "world" if worlds == 1 else "worlds", TOTAL_NUMBER),
            (repetitions, "repetition" if repetitions == 1 else "repetitions", TOTAL_NUMBER),
            (
                experiments,
                "planned experiment" if experiments == 1 else "planned experiments",
                GRAND_NUMBER,
            ),
        ]

    @staticmethod
    def group_width(number: int, word: str, style: TextStyle) -> float:
        return style.advance(f"{number:,}") + 6 + BODY.advance(word)

    def natural_width(self) -> float:
        groups = self.groups()
        width = 1 + 20 + sum(self.group_width(*group) for group in groups)
        return width + 22 * (len(groups) - 1) + 1 + 22  # the grand total's rule and padding

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 32)

    @staticmethod
    def line_box(style: TextStyle) -> tuple[float, float]:
        """A group's baseline offset and height: the number and the words share a baseline
        (``align-items: baseline``), each in its 18 px line box."""
        number, word = style.baseline(0), BODY.baseline(0)
        above = max(number, word)
        return above, above + max(style.line - number, BODY.line - word)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        line = theme.color("border.line")
        painter.fillRect(QtCore.QRect(snap(self.shift), 0, 1, self.height()), line)
        x = self.shift + 1 + 20
        groups = self.groups()
        for index, (number, word, style) in enumerate(groups):
            if index:
                x += 22
            above, height = self.line_box(style)
            top = (self.height() - height) / 2
            if index == len(groups) - 1:
                painter.fillRect(
                    QtCore.QRect(snap(x), snap(top), 1, snap(top + height) - snap(top)), line
                )
                x += 1 + 22
            baseline = top + above
            text = f"{number:,}"
            x += draw_text(
                painter, x, baseline - style.baseline(0), text, style, theme.color("text.ink")
            )
            x += 6
            token = "text.body" if index == len(groups) - 1 else "text.secondary"
            x += draw_text(painter, x, baseline - BODY.baseline(0), word, BODY, theme.color(token))


class WizardFooter(Card):
    """``.wiz-foot`` (56 px card, padding 0 16): the Tuning run box, Import YAML and Export YAML, the totals
    (12 px further, after a rule), then the step buttons at the right; Start run replaces Continue on the
    worlds step of a tuning run."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.setFixedHeight(56)
        self.row = FlexRow(self, gap=8, left=16, right=16)
        self.tuning_box = CheckBox(themes, "Tuning run")
        self.tuning_box.setToolTip(
            "One run of one robot, profile, world and starting pose, opened in Monitor and not kept"
        )
        self.import_button = Button(themes, "Import YAML", icon="upload")
        self.export_button = Button(themes, "Export YAML", icon="download")
        self.totals = Totals(themes)
        self.back_button = Button(themes, "Back", icon="aleft")
        self.validate_button = Button(themes, "Validate", icon="ccheck")
        self.next_button = Button(
            themes, "Continue", icon="aright", variant="primary", trailing=True
        )
        self.submit_button = Button(themes, "Add to queue", icon="queue", variant="primary")
        self.start_button = Button(themes, "Start run", icon="play", variant="primary")
        self.row.add(self.tuning_box, gap=8 + 8)
        self.row.add(self.import_button)
        self.row.add(self.export_button)
        self.row.add(self.totals, gap=8 + 12)
        self.row.add_stretch()
        for button in (
            self.back_button,
            self.validate_button,
            self.next_button,
            self.submit_button,
            self.start_button,
        ):
            self.row.add(button)

    def relayout(self) -> None:
        self.row.relayout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.row.setGeometry(1, 1, self.width() - 2, self.height() - 2)


# ---------- cards, popups and lists of steps 1 and 2 ----------

POPUP_TITLE = TextStyle(15, 600, 22)


class ToggleCard(Card):
    """A robot or world card: a click anywhere outside its controls, or Space or Enter while it has focus,
    asks to tick or untick it (``toggled``; the selector applies its rules). A drag or the second click of a
    double click does not toggle; the hover repaints the border."""

    toggled = QtCore.Signal()

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self._press: QtCore.QPoint | None = None
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "ring")
        self.setProperty("focusRadius", 8)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        left = event.button() == QtCore.Qt.MouseButton.LeftButton
        self._press = event.position().toPoint() if left else None
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._press is not None
            and (event.position().toPoint() - self._press).manhattanLength()
            >= QtWidgets.QApplication.startDragDistance()
        ):
            self._press = None

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        clicked = self._press is not None and event.button() == QtCore.Qt.MouseButton.LeftButton
        self._press = None
        if clicked:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.toggled.emit()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self._press = None  # the first click of the pair already toggled

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # Keys a control on the card ignores (Enter in its time limit) reach the card too: only the
        # focused card toggles.
        if self.hasFocus() and event.key() in (
            QtCore.Qt.Key.Key_Space,
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        ):
            self.toggled.emit()
            return
        super().keyPressEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()
        return super().event(event)


def toggle_reference(
    selected: Sequence[Reference], reference: Reference, key: Callable[[Reference], str]
) -> list[Reference] | None:
    """``selected`` with ``reference`` added, or removed when it is there; None when removing it would leave
    nothing selected (a robot keeps one profile, a world one starting pose)."""
    identity = key(reference)
    kept = [deepcopy(item) for item in selected if key(item) != identity]
    if len(kept) == len(selected):
        return [*kept, deepcopy(reference)]
    return kept or None


def toggle_shown(
    selected: Sequence[Reference], shown: Sequence[Reference], key: Callable[[Reference], str]
) -> list[Reference]:
    """A list's header check: select every shown entry, or, when all are selected, clear them, keeping the
    first shown one when nothing else would stay selected."""
    keys = {key(item) for item in selected}
    shown_keys = {key(item) for item in shown}
    if shown and shown_keys <= keys:
        kept = [deepcopy(item) for item in selected if key(item) not in shown_keys]
        return kept or [deepcopy(shown[0])]
    added = [deepcopy(item) for item in shown if key(item) not in keys]
    return [deepcopy(item) for item in selected] + added


def seconds(value: float) -> str:
    """A time limit as entered and stored: "1800", "1800.5" (up to six decimals, never an exponent)."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def grouped_seconds(value: float) -> str:
    """A time limit as the world cards show it: "1,800", "1,800.5" (``seconds`` with grouped thousands)."""
    whole, _, fraction = seconds(value).partition(".")
    sign = "-" if whole.startswith("-") else ""
    grouped = sign + f"{int(whole.lstrip('-') or 0):,}"
    return grouped + ("." + fraction if fraction else "")


def search_text(event: QtGui.QKeyEvent) -> str:
    """The text a key press types into a list's search field ("" for keys that keep their own meaning:
    Space, shortcuts and keys without printable text)."""
    text = event.text()
    plain = not event.modifiers() & (
        QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.AltModifier
    )
    return text if text and text.isprintable() and text != " " and plain else ""


class PopupButton(Button):
    """A small secondary button that opens a wizard popup ("Profiles ⌄", "Spawns ⌄"); while the popup is open
    it is ``.pressed`` with an ink border and ⌃."""

    def __init__(
        self, themes: ThemeManager, text: str, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, text, icon="cdown", small=True, trailing=True, parent=parent)
        self.open = False

    def set_open(self, open: bool) -> None:
        if open != self.open:
            self.open = open
            self.set_icon("cup" if open else "cdown")
            self.update()

    def colors(self) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
        if self.open and self.isEnabled():
            theme = self.themes.current
            return theme.color("surface.selected"), theme.color("text.ink"), theme.color("text.ink")
        return super().colors()


class PopupHead(FlexRow):
    """``.pp-head``: the title (15/22 600), the search field (260 px) and "Done · N selected" (primary with
    ``check``); padding 0 12 0 20, 10 px gaps."""

    def __init__(
        self,
        themes: ThemeManager,
        title: str,
        placeholder: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent, gap=10, left=20, right=12)
        self.title = TextLine(themes, title, POPUP_TITLE, "text.ink")
        self.search = SearchField(themes, placeholder)
        self.search.setFixedWidth(260)
        self.done = Button(themes, "Done", icon="check", variant="primary")
        self.add(self.title)
        self.add_stretch()
        self.add(self.search, width=260)
        self.add(self.done)

    def set_count(self, count: int) -> None:
        self.done.setText(f"Done · {count} selected")


# ---------- card grids (steps 1 and 2) ----------


class CardGrid(QtWidgets.QScrollArea):
    """``.robots`` and ``.worlds``: equal columns 16 px apart at the browser's fractional positions and rows
    sharing the height (``grid-template-rows: 1fr``), each at least ``min_row`` px; the grid scrolls when the
    rows do not fit. ``columns_for`` maps the content width to the number of columns."""

    GAP = 16

    def __init__(
        self,
        columns_for: Callable[[float], int],
        min_row: float,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.columns_for, self.min_row = columns_for, min_row
        self.cards: list[QtWidgets.QWidget] = []
        self.columns, self.card_width, self.row_height = 1, 0.0, min_row
        self._laying_out = False
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)
        self.content = QtWidgets.QWidget()
        self.setWidget(self.content)

    def set_cards(self, cards: Sequence[QtWidgets.QWidget]) -> None:
        self.cards = list(cards)
        for card in self.cards:
            if card.parent() is not self.content:
                card.setParent(self.content)
            card.show()
        self.relayout()

    def cell(self, index: int) -> tuple[int, int]:
        """Row and column of the card at ``index``."""
        return divmod(index, max(1, self.columns))

    def cell_box(self, row: int, column: int, span: int = 1) -> QtCore.QRectF:
        """The CSS box of ``span`` cells from (row, column), in content coordinates."""
        return QtCore.QRectF(
            column * (self.card_width + self.GAP),
            row * (self.row_height + self.GAP),
            span * self.card_width + (span - 1) * self.GAP,
            self.row_height,
        )

    def relayout(self) -> None:
        if self._laying_out:
            return
        self._laying_out = True
        try:
            viewport = self.viewport()
            width, height = float(viewport.width()), float(viewport.height())
            count = len(self.cards)
            self.columns = max(1, self.columns_for(width))
            rows = max(1, math.ceil(count / self.columns))
            self.card_width = (width - (self.columns - 1) * self.GAP) / self.columns
            self.row_height = max(self.min_row, (height - (rows - 1) * self.GAP) / rows)
            total = rows * self.row_height + (rows - 1) * self.GAP
            self.content.resize(viewport.width(), math.ceil(total))
            for index, card in enumerate(self.cards):
                box = self.cell_box(*self.cell(index))
                place(card, box.x(), box.y(), box.width(), box.height())
        finally:
            self._laying_out = False

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            self.relayout()  # a scroll bar appeared or went away
        return super().viewportEvent(event)


# ---------- step 3: execution ----------

SETTING_LABEL = TextStyle(13, 600, 32)
NOTE_TEXT = TextStyle(12.5, 400, 17)
GROUP_HEAD = TextStyle(12, 600, 18)
CHANGE_TEXT = TextStyle(12.5, 500, 18)
COUNT_TEXT = TextStyle(13, 600, 18, tabular=True)
SUMMARY_NOTE = TextStyle(12, 400, 18)
SUMMARY_MONO = TextStyle(12, 400, 18, mono=True)


class NotesEdit(QtWidgets.QTextEdit):
    """``.input.area``: plain text on 18 px lines (padding 8 10), read and set like the line edit it
    replaces (``text``, ``setText``)."""

    LINE = 18

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("notes")
        self.setAcceptRichText(False)
        self.setTabChangesFocus(True)
        self.document().setDocumentMargin(0)
        self.apply_line_height()

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:
        self.setPlainText(text)
        self.apply_line_height()

    def apply_line_height(self) -> None:
        cursor = QtGui.QTextCursor(self.document())
        cursor.select(QtGui.QTextCursor.SelectionType.Document)
        block = QtGui.QTextBlockFormat()
        block.setLineHeight(self.LINE, QtGui.QTextBlockFormat.LineHeightTypes.FixedHeight.value)
        cursor.mergeBlockFormat(block)


class ChoiceSegments(SegmentedControl):
    """A segmented control with a value per segment; ``set_value`` turns no segment on for other values."""

    def __init__(
        self,
        themes: ThemeManager,
        choices: Sequence[tuple[str, int]],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, [(text, None) for text, _ in choices], parent=parent)
        self.values = [value for _, value in choices]

    def value(self) -> int | None:
        index = self.index()
        return self.values[index] if 0 <= index < len(self.values) else None

    def set_value(self, value: int) -> None:
        if value in self.values:
            self.set_index(self.values.index(value))
        else:
            self.clear()


class NoteLine(Themed):
    """``.note``: the ``info`` icon (``text.tertiary``, 1 px down) and 12.5/17 ``text.secondary``, 8 px apart."""

    def __init__(
        self, themes: ThemeManager, text: str, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.text = text
        self.setFixedHeight(17)
        self.setAccessibleName(text)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        gui_icons.draw(painter, "info", theme.color("text.tertiary"), QtCore.QPointF(0, 1), 16)
        room = self.width() - 24
        text = NOTE_TEXT.elide(self.text, room)
        draw_text(painter, 24, 0, text, NOTE_TEXT, theme.color("text.secondary"))


class SettingRow(Themed):
    """``.set``: padding 14 20 (screen 32), the label in a 240 px
    column (600 ink on a 32 px line; a note such as "· optional" in ``text.tertiary``) and the control 32 px
    after it; a bottom rule unless last. When less than ``STACK_BELOW`` px would stay beside the label (the
    window's narrowest widths), the control goes under the label at the row's full width."""

    LABEL_WIDTH, GAP, PAD_X, PAD_Y = 240, 32, 20, 14
    LABEL_LINE = 32  # the label's line, above a stacked control
    STACK_BELOW = 360

    def __init__(
        self,
        themes: ThemeManager,
        label: str,
        control: QtWidgets.QWidget,
        height: int,
        *,
        optional: bool = False,
        note: str = "",
        last: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.label, self.control, self.control_height = label, control, height
        self.note = note or ("· optional" if optional else "")  # drawn after the label when set
        self.last = last
        control.setParent(self)
        self.setFixedHeight(2 * self.PAD_Y + height + (0 if last else 1))
        self.setAccessibleName(label)

    def set_control_height(self, height: int) -> None:
        """A control that wraps onto more lines (the Study row at narrow widths)."""
        if height != self.control_height:
            self.control_height = height
            self.resizeEvent(QtGui.QResizeEvent(self.size(), self.size()))

    def stacked(self, width: int | None = None) -> bool:
        """The control goes under the label in a row ``width`` wide (the current one by default)."""
        width = self.width() if width is None else width
        return width - 2 * self.PAD_X - self.LABEL_WIDTH - self.GAP < self.STACK_BELOW

    def control_width(self, width: int | None = None) -> int:
        """The width the control gets in a row ``width`` wide (the current one by default)."""
        width = self.width() if width is None else width
        beside = 0 if self.stacked(width) else self.LABEL_WIDTH + self.GAP
        return width - 2 * self.PAD_X - beside

    def height_for(self, width: int) -> int:
        """The row's height when it is ``width`` wide."""
        label = self.LABEL_LINE if self.stacked(width) else 0
        return 2 * self.PAD_Y + label + self.control_height + (0 if self.last else 1)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width = self.width()
        if self.height_for(width) != self.height():
            self.setFixedHeight(self.height_for(width))
        stacked = self.stacked(width)
        left = self.PAD_X if stacked else self.PAD_X + self.LABEL_WIDTH + self.GAP
        top = self.PAD_Y + (self.LABEL_LINE if stacked else 0)
        place(self.control, left, top, self.control_width(width), self.control_height)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        x = self.PAD_X + draw_text(
            painter,
            self.PAD_X,
            self.PAD_Y,
            self.label + (" " if self.note else ""),
            SETTING_LABEL,
            theme.color("text.ink"),
        )
        if self.note:
            # The span sits on the label's baseline in its own 15 px inline box.
            top = SETTING_LABEL.baseline(self.PAD_Y) - BODY.baseline(0)
            draw_text(painter, x, top, self.note, BODY, theme.color("text.tertiary"))
        if not self.last:
            painter.fillRect(
                QtCore.QRect(0, self.height() - 1, self.width(), 1), theme.color("border.line")
            )


class ExecutionControl(QtWidgets.QWidget):
    """The Execution setting: Sequential to 4 parallel, then the replacement note 10 px below."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.concurrency = ChoiceSegments(
            themes,
            [("Sequential", 1), ("2 parallel", 2), ("3 parallel", 3), ("4 parallel", 4)],
            parent=self,
        )
        self.concurrency.setAccessibleName("Execution")
        self.note = NoteLine(
            themes,
            "Timing-rejected parallel attempts receive one final sequential replacement.",
            self,
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        place(self.concurrency, 0, 0, self.concurrency.natural_width(), 32)
        place(self.note, 0, 42, self.width(), 17)


class ExecutionCard(Card):
    """``.form``: "Campaign" with Name, Description (optional), Repetitions per combination (spin box and
    presets 1 · 10 · 100) and Execution (Sequential to 4 parallel, with the replacement note)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.name = QtWidgets.QLineEdit("Exploration study")
        self.name.setAccessibleName("Campaign name")
        self.description = NotesEdit()
        self.description.setPlaceholderText("Optional research notes")
        self.description.setAccessibleName("Description")
        self.repetitions = SpinBox(themes)
        self.repetitions.setRange(1, 1_000_000)
        self.repetitions.setValue(1)
        self.repetitions.setAccessibleName("Repetitions per combination")
        self.presets = ChoiceSegments(themes, [("1", 1), ("10", 10), ("100", 100)])
        self.presets.changed.connect(
            lambda index: self.repetitions.setValue(self.presets.values[index])
        )
        self.repetitions.valueChanged.connect(self.presets.set_value)
        self.presets.set_value(self.repetitions.value())
        repetitions = FlexRow(gap=12)
        repetitions.add(self.repetitions, width=112)
        repetitions.add(TextLine(themes, "Presets", BODY, "text.tertiary"))
        repetitions.add(self.presets)
        # the study's numbers for these repetitions ("· study repetitions 1–5"; hidden when unassigned)
        self.study_range = TextLine(themes, "", BODY, "text.tertiary")
        repetitions.add(self.study_range)
        self.repetition_row = repetitions
        execution = ExecutionControl(themes)
        self.concurrency = execution.concurrency
        # Debug streams (A7): a run condition, recorded per run and checked across a study
        self.debug = Switch(themes, "Off", gap=10)
        self.debug.setAccessibleName("Debug streams")
        self.debug_note = TextLine(themes, "", BODY, "text.tertiary")
        debug = FlexRow(gap=10)
        debug.add(self.debug)
        debug.add(self.debug_note)
        self.debug_row = debug
        self.debug.toggled.connect(self._debug_text)
        self._debug_text(False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(CardHead(themes, "Campaign"))
        self.rows = (
            SettingRow(themes, "Name", self.name, 32),
            SettingRow(themes, "Description", self.description, 56, optional=True),
            SettingRow(themes, "Repetitions per combination", repetitions, 32),
            SettingRow(themes, "Execution", execution, 59),
            SettingRow(themes, "Debug streams", debug, 32, last=True),
        )
        for row in self.rows:
            layout.addWidget(row)
        layout.addStretch(1)

    def _debug_text(self, on: bool) -> None:
        self.debug.setText("On" if on else "Off")
        self.debug_note.setText(
            "· records Nav2 feedback at 1 Hz per goal and command rows"
            if on
            else "· records Nav2 feedback and command rows when on"
        )
        self.debug_row.relayout()

    def set_study_range(self, text: str) -> None:
        self.study_range.setText(text)
        self.study_range.setVisible(bool(text))
        self.repetition_row.relayout()

    def natural_height(self, width: int | None = None) -> int:
        """Border, head and the rows' heights (at the card's ``width`` when given)."""
        if width is None:
            return 2 + 52 + sum(row.minimumHeight() for row in self.rows)
        return 2 + 52 + sum(row.height_for(width - 2) for row in self.rows)


STUDY_CHOICES = (("Unassigned", 0), ("Existing study", 1), ("New study", 2))
UNASSIGNED_STUDY, EXISTING_STUDY, NEW_STUDY = 0, 1, 2
HASH_TEXT = TextStyle(12, 400, 18, mono=True)


class StudyChoice(QtWidgets.QWidget):
    """The Study row's control: the kind segments, then the ID field or study select (``width`` px) beside
    them, or under them when ``wrapped``."""

    def __init__(self, kind: "ChoiceSegments", fields: tuple[QtWidgets.QWidget, ...]) -> None:
        super().__init__()
        self.kind, self.fields = kind, fields
        self.width_px, self.wrapped = 240.0, False
        for widget in (kind, *fields):
            widget.setParent(self)

    def relayout(self) -> None:
        kind = math.ceil(self.kind.natural_width())
        self.kind.setGeometry(0, 0, kind, 32)
        x, y = (0, 32 + 10) if self.wrapped else (kind + 10, 0)
        for field in self.fields:
            field.setGeometry(x, y, math.floor(self.width_px), 32)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()


class StudyForm(Card):
    """``.form.study`` (32): "Study" with the study (Unassigned, an existing study, or a new one by its ID),
    its evidence level as the researcher declares it (Exploratory or Confirmatory) and its plan (a Markdown
    file in the checkout, its SHA-256 above the deviations log as the engine computes it; required for
    confirmatory)."""

    choose_plan = QtCore.Signal()

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.kind = ChoiceSegments(themes, STUDY_CHOICES)
        self.kind.setAccessibleName("Study")
        self.new_id = QtWidgets.QLineEdit()
        self.new_id.setPlaceholderText("Study ID, e.g. S2-threshold")
        self.new_id.setAccessibleName("New study ID")
        self.existing = KeySelect(themes)
        self.existing.setAccessibleName("Existing study")
        study = StudyChoice(self.kind, (self.new_id, self.existing))
        self.study_row = study
        self.evidence = ChoiceSegments(themes, [("Exploratory", 0), ("Confirmatory", 1)])
        self.evidence.setAccessibleName("Evidence level")
        evidence = QtWidgets.QWidget()
        self.evidence.setParent(evidence)
        self.plan = QtWidgets.QLineEdit()
        self.plan.setPlaceholderText("docs/research/experiments/YYYY-MM-DD-slug.md")
        self.plan.setAccessibleName("Plan")
        self.choose = Button(themes, "Choose…", icon="folder")
        self.choose.clicked.connect(self.choose_plan.emit)
        self.hash = TextLine(themes, "", HASH_TEXT, "text.secondary")
        plan = FlexRow(gap=10)
        plan.add(self.plan)
        plan.add(self.choose)
        plan.add(self.hash)
        self.plan_row = plan
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(CardHead(themes, "Study"))
        self.rows = (
            SettingRow(themes, "Study", study, 32),
            SettingRow(themes, "Evidence level", evidence, 32),
            SettingRow(themes, "Plan", plan, 32, note="· required for confirmatory", last=True),
        )
        for row in self.rows:
            layout.addWidget(row)
        layout.addStretch(1)
        self.kind.set_index(UNASSIGNED_STUDY)
        self.evidence.set_index(0)
        self.kind.changed.connect(lambda _: self.sync())
        self.sync()

    def sync(self) -> None:
        kind = self.kind.index()
        self.new_id.setVisible(kind == NEW_STUDY)
        self.existing.setVisible(kind == EXISTING_STUDY)
        enabled = kind != UNASSIGNED_STUDY
        for widget in (self.evidence, self.plan, self.choose):
            widget.setEnabled(enabled)
        # the ID field or study select: 240 px beside the choice, less when the card is narrow, and on a line
        # of its own under it when even 120 px does not fit there (the window's narrowest width)
        width = self.rows[0].control_width() if self.rows[0].width() else self.study_row.width()
        room = width - self.kind.natural_width() - 10
        wrapped = room < 120 and kind != UNASSIGNED_STUDY
        self.study_row.wrapped = wrapped
        self.study_row.width_px = min(240.0, width) if wrapped else max(120.0, min(240.0, room))
        self.rows[0].set_control_height(32 + 10 + 32 if wrapped else 32)
        self.study_row.relayout()
        self._plan_width()

    def set_hash(self, text: str) -> None:
        self.hash.setText(text)
        self._plan_width()

    def _plan_width(self) -> None:
        """The plan field takes the room the button and the hash leave (``.input.grow``)."""
        room = (
            self.plan_row.width()
            - 10
            - self.choose.natural_width()
            - (10 + self.hash.natural_width() if self.hash.text() else 0)
        )
        self.plan_row.set_width(self.plan, max(60.0, room))  # the hash is read before the path
        self.evidence.setGeometry(0, 0, math.ceil(self.evidence.natural_width()), 32)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.sync()

    def natural_height(self, width: int | None = None) -> int:
        """Border, head and the rows' heights (at the card's ``width`` when given)."""
        if width is None:
            return 2 + 52 + sum(row.minimumHeight() for row in self.rows)
        return 2 + 52 + sum(row.height_for(width - 2) for row in self.rows)


@dataclass(frozen=True, slots=True)
class SummaryItem:
    """A ``.it`` row: the name (150 px, 500 ink), the detail (ellipsis) and an optional count."""

    name: str
    detail: str
    count: int | None = None
    mono: bool = False  # the detail is an identifier ("kinematic-time-limit-v2", 38)


@dataclass(frozen=True, slots=True)
class DraftFacts:
    robots: tuple[SummaryItem, ...] = ()
    worlds: tuple[SummaryItem, ...] = ()
    study: tuple[SummaryItem, ...] = ()  # "Study & recording" (32)
    combinations: tuple[SummaryItem, ...] = ()
    per_repetition: int = 0
    note: str = ""


class ChangeLink(LinkButton):
    """``.sum .gh .link``: "Change" in 12.5 px 500 ink with the underline, no arrow."""

    STYLE = CHANGE_TEXT


class DraftSummary(Card):
    """``.sum``: "Draft summary" with the robots and their profiles, the worlds with their starting poses
    and time limits (each with a Change link to its step), the combinations per robot, the total per
    repetition and the combinations left out in Review."""

    change_requested = QtCore.Signal(int)

    GROUP_TOP, GROUP_BOTTOM, HEAD, ITEM, TOTAL = 6, 10, 36, 30, 36

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.facts = DraftFacts()
        self.head = CardHead(themes, "Draft summary", self)
        self.links = [ChangeLink(themes, "Change", icon=None, parent=self) for _ in range(2)]
        for step, link in enumerate(self.links):
            link.setAccessibleName(("Change robots and profiles", "Change worlds and poses")[step])
            link.clicked.connect(lambda checked=False, s=step: self.change_requested.emit(s))

    def set_facts(self, facts: DraftFacts) -> None:
        if facts != self.facts:
            self.facts = facts
            self.relayout()
            self.update()

    def groups(self) -> list[tuple[str, tuple[SummaryItem, ...]]]:
        return [
            ("Robots & profiles", self.facts.robots),
            ("Worlds & starting poses", self.facts.worlds),
            ("Study & recording", self.facts.study),
            ("Combinations", self.facts.combinations),
        ]

    def natural_height(self) -> int:
        """Border, head, the groups with their rules, the per-repetition total and the note."""
        height = 2 + 52.0
        groups = self.groups()
        for index, (_, items) in enumerate(groups):
            last = index == len(groups) - 1
            height += self.GROUP_TOP + self.HEAD + self.ITEM * len(items)
            height += 4 + self.TOTAL + (20 if self.facts.note else 0) if last else 0
            height += self.GROUP_BOTTOM + (0 if last else 1)
        return math.ceil(height)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def relayout(self) -> None:
        self.head.setGeometry(1, 1, self.width() - 2, 52)
        y = 53.0
        right = self.width() - 1 - 20
        for index, (_, items) in enumerate(self.groups()[:2]):
            link = self.links[index]
            width = link.natural_width()
            place(link, right - width, y + self.GROUP_TOP + (self.HEAD - 18) / 2, width, 18)
            y += self.GROUP_TOP + self.HEAD + self.ITEM * len(items) + self.GROUP_BOTTOM + 1

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        theme = self.theme
        left, right = 1 + 20.0, self.width() - 1 - 20.0
        y = 53.0
        groups = self.groups()
        for index, (title, items) in enumerate(groups):
            last = index == len(groups) - 1
            top = y + self.GROUP_TOP
            draw_text(painter, left, top + 9, title, GROUP_HEAD, theme.color("text.secondary"))
            row = top + self.HEAD
            for item in items:
                self.paint_item(painter, item, left, right, row, self.ITEM)
                row += self.ITEM
            if last:
                # ``.it.tot``: 4 px margin, a 1 px top border inside its 36 px, the row centred below it.
                painter.fillRect(
                    QtCore.QRect(snap(left), snap(row + 4), snap(right) - snap(left), 1),
                    theme.color("border.line"),
                )
                total = SummaryItem("Per repetition", "", self.facts.per_repetition)
                self.paint_item(painter, total, left, right, row + 5, self.TOTAL - 1)
                row += 4 + self.TOTAL
                if self.facts.note:
                    note = SUMMARY_NOTE.elide(self.facts.note, right - left)
                    draw_text(
                        painter, left, row + 2, note, SUMMARY_NOTE, theme.color("text.secondary")
                    )
            else:
                y = row + self.GROUP_BOTTOM
                painter.fillRect(
                    QtCore.QRect(1, snap(y), self.width() - 2, 1), theme.color("border.line")
                )
                y += 1

    def paint_item(
        self,
        painter: QtGui.QPainter,
        item: SummaryItem,
        left: float,
        right: float,
        top: float,
        height: float,
    ) -> None:
        """``.it``: grid 150 px | 1fr | auto, 12 px gaps, centred on its row."""
        theme = self.theme
        line_top = top + (height - 18) / 2
        draw_text(painter, left, line_top, item.name, BODY_MEDIUM, theme.color("text.ink"))
        count_width = 0.0
        if item.count is not None:
            text = f"{item.count:,}"
            count_width = max(16.0, COUNT_TEXT.advance(text))
            x = right - COUNT_TEXT.advance(text)
            draw_text(painter, x, line_top, text, COUNT_TEXT, theme.color("text.ink"))
        detail_left = left + 150 + 12
        room = right - detail_left - (count_width + 12 if item.count is not None else 0)
        style = SUMMARY_MONO if item.mono else BODY
        detail = style.elide(item.detail, max(0.0, room))
        draw_text(painter, detail_left, line_top, detail, style, theme.color("text.secondary"))
