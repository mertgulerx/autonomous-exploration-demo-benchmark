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

"""Controls of the Main design (docs/suite-gui-design, HANDOFF §7.4-§7.7): badges, chips, pills, the
switch, underline tabs, selects, spin boxes and the search field.

Painted from the active theme like ``gui_widgets``. Geometry follows ``mockups/styles.css``: browsers place
text at fractional x and round baselines, and snap boxes and icons to whole pixels (half up).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons, gui_tokens
from .gui_style import humanize
from .gui_theme import Mark, Theme, ThemeManager, css_color, font, paint_check, paint_radio, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    CssBox,
    TextStyle,
    Themed,
    draw_text,
    press_on_enter,
    rounded_path,
    shadow_image,
    shadow_margin,
)

Tone = Literal["neutral", "running", "ok", "warn", "bad", "muted"]


# ---------- Badges ----------


BADGE_TEXT = TextStyle(12, 500, 12)
BADGE_TONES: Mapping[Tone, tuple[str, str]] = MappingProxyType(
    {
        "neutral": ("state.neutralSoft", "state.neutral"),
        "running": ("accent.soft", "accent.softText"),
        "ok": ("state.okSoft", "state.ok"),
        "warn": ("state.warnSoft", "state.warn"),
        "bad": ("state.badSoft", "state.bad"),
        "muted": ("state.neutralSoft", "text.secondary"),  # excluded outcomes in Results rows
    }
)


@dataclass(frozen=True, slots=True)
class BadgeLook:
    """``.badge``: 22 px, padding 0 8 0 6, gap 5, radius 4, 12 px 500, a 13 px icon with stroke 2."""

    word: str
    tone: Tone = "neutral"
    icon: str | None = None

    @property
    def width(self) -> float:
        return 6 + (13 + 5 if self.icon else 0) + BADGE_TEXT.advance(self.word) + 8


# One mapping for the whole application (HANDOFF §7.4): value -> (tone, icon).
CAMPAIGN_BADGES: Mapping[str, tuple[Tone, str]] = MappingProxyType(
    {
        "RUNNING": ("running", "cplay"),
        "RETRYING": ("running", "loader"),
        "REPORTING": ("running", "loader"),
        "QUEUED": ("neutral", "clock"),
        "PAUSING": ("warn", "loader"),
        "STOPPING": ("warn", "loader"),
        "PAUSED": ("warn", "cpause"),
        "STOPPED": ("neutral", "cstop"),
        "RECOVERY_REQUIRED": ("bad", "alert"),
        "BLOCKED": ("bad", "ban"),
        "FINISHED": ("ok", "ccheck"),
    }
)
ATTEMPT_BADGES: Mapping[str, tuple[Tone, str]] = MappingProxyType(
    {
        "STARTING": ("neutral", "loader"),
        "READY": ("neutral", "loader"),
        "FINALIZING": ("neutral", "loader"),
        "RUNNING": ("running", "cplay"),
        "STOPPING": ("warn", "loader"),
    }
)
OUTCOME_BADGES: Mapping[str, tuple[Tone, str]] = MappingProxyType(
    {
        "completed": ("ok", "ccheck"),
        "time_limit": ("warn", "hourglass"),
        "no_movement": ("neutral", "ban"),
        "manual_stop": ("neutral", "cstop"),
        "startup_failure": ("bad", "cx"),
        "runtime_failure": ("bad", "cx"),
        "storage_failure": ("bad", "cx"),
        "engine_interruption": ("bad", "cx"),
        "timing_rejection": ("bad", "clock"),
        "manual_discard": ("bad", "cx"),
    }
)
REPORT_BADGES: Mapping[str, tuple[Tone, str]] = MappingProxyType(
    {
        "QUEUED": ("neutral", "clock"),
        "COMPACTING": ("running", "loader"),
        "COMMITTING": ("running", "loader"),
        "RUNNING": ("running", "loader"),
        "INDEXING": ("running", "loader"),
        "FINISHED": ("ok", "ccheck"),
        "FAILED": ("bad", "cx"),
    }
)
HEALTH_BADGES: Mapping[str, tuple[Tone, str]] = MappingProxyType(
    {"healthy": ("ok", "ccheck"), "flags": ("warn", "alert"), "pending": ("neutral", "clock")}
)
EXPERIMENT_BADGES: Mapping[str, tuple[Tone, str]] = MappingProxyType(
    {"unresolved": ("bad", "cminus")}
)
BADGE_KINDS: Mapping[str, Mapping[str, tuple[Tone, str]]] = MappingProxyType(
    {
        "campaign": CAMPAIGN_BADGES,
        "attempt": ATTEMPT_BADGES,
        "outcome": OUTCOME_BADGES,
        "experiment": EXPERIMENT_BADGES,
        "report": REPORT_BADGES,
        "health": HEALTH_BADGES,
    }
)


def badge_look(
    kind: str, value: str | None, *, word: str | None = None, muted: bool = False
) -> BadgeLook | None:
    """The badge of a campaign state, attempt phase, outcome, experiment state, report state or health value;
    None for a kind or value the design does not draw as a badge. ``muted`` gives excluded outcomes the
    neutral fill and secondary text of Results rows."""
    found = BADGE_KINDS.get(kind, {}).get(value) if value is not None else None
    if found is None or value is None:
        return None
    tone, icon = found
    return BadgeLook(word or humanize(value), "muted" if muted else tone, icon)


def paint_badge(
    painter: QtGui.QPainter, x: float, y: float, look: BadgeLook, theme: Theme
) -> float:
    """Paint a badge with its top-left corner at (x, y); returns its width."""
    fill, text = BADGE_TONES[look.tone]
    width = look.width
    left, top = snap(x), snap(y)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(theme.color(fill))
    painter.drawPath(rounded_path(QtCore.QRectF(left, top, snap(x + width) - left, 22), 4))
    color = theme.color(text)
    text_x = x + 6
    if look.icon:
        gui_icons.draw(
            painter,
            look.icon,
            color,
            QtCore.QPointF(snap(x + 6), snap(y + 4.5)),
            13,
            stroke=gui_icons.BADGE_STROKE,
        )
        text_x += 13 + 5
    draw_text(painter, text_x, y + 5, look.word, BADGE_TEXT, color)
    painter.restore()
    return width


class StatusBadge(Themed, CssBox):
    """A badge as a widget (cards, heads); tables paint badges with ``paint_badge``."""

    def __init__(
        self,
        themes: ThemeManager,
        look: BadgeLook | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.look: BadgeLook | None = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.set_look(look)

    def natural_width(self) -> float:
        return self.look.width if self.look else 0.0

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 22)

    def set_look(self, look: BadgeLook | None) -> None:
        """Show ``look`` (None hides the badge)."""
        if look == self.look:
            return
        self.look = look
        self.setVisible(look is not None)
        self.setAccessibleName(look.word if look else "")
        self.updateGeometry()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self.look is not None:
            paint_badge(QtGui.QPainter(self), self.shift, 0, self.look, self.theme)


# ---------- Chips and pills ----------


COUNT_TEXT = TextStyle(12, 600, 18, tabular=True)
CHANGE_PILL_TEXT = TextStyle(10.5, 600, 18, tabular=True)


class CountChip(Themed, CssBox):
    """``.count``: 22 px, padding 0 7, radius 4, sunken fill, 1 px line, 12 px 600 tabular secondary."""

    def __init__(
        self, themes: ThemeManager, count: int | None = 0, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.text = ""
        self.setFixedHeight(22)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.set_count(count)

    def natural_width(self) -> float:
        return 2 + 14 + COUNT_TEXT.advance(self.text)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 22)

    def set_count(self, count: int | None) -> None:
        """The total, or "—" when unavailable."""
        text = "—" if count is None else f"{count:,}"
        if text != self.text:
            self.text = text
            self.setAccessibleName(text)
            self.updateGeometry()
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.sunken"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 3.5)
        )
        draw_text(painter, self.shift + 8, 2, self.text, COUNT_TEXT, self.color("text.secondary"))


def change_pill_width(count: int) -> float:
    return max(16.0, 8 + CHANGE_PILL_TEXT.advance(str(count)))


def paint_change_pill(
    painter: QtGui.QPainter, x: float, y: float, count: int, theme: Theme
) -> float:
    """``.pill``: the ink count of changed parameters on group tabs (16 px, radius 8, 10.5 px 600)."""
    width = change_pill_width(count)
    left = snap(x)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(theme.color("text.ink"))
    painter.drawPath(rounded_path(QtCore.QRectF(left, snap(y), snap(x + width) - left, 16), 8))
    text = str(count)
    draw_text(
        painter,
        x + (width - CHANGE_PILL_TEXT.advance(text)) / 2,
        y - 1,
        text,
        CHANGE_PILL_TEXT,
        theme.color("text.onInk"),
    )
    painter.restore()
    return width


# ---------- Switch ----------


def paint_switch_track(
    painter: QtGui.QPainter, origin: QtCore.QPointF, on: bool, theme: Theme
) -> None:
    """The ``.switch`` track (32×18: ink when on, ``control.switchOff`` off) and its 14 px knob with its
    shadow, at ``origin`` (also painted inside the Monitor's Layers menu)."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(theme.color("text.ink" if on else "control.switchOff"))
    painter.drawPath(rounded_path(QtCore.QRectF(origin.x(), origin.y(), 32, 18), 9))
    knob = QtCore.QRectF(origin.x() + (16 if on else 2), origin.y() + 2, 14, 14)
    margin = shadow_margin((gui_tokens.KNOB_SHADOW,))
    painter.drawImage(
        QtCore.QPointF(knob.x() - margin, knob.y() - margin),
        shadow_image(14, 14, 7, (gui_tokens.KNOB_SHADOW,)),
    )
    painter.setBrush(theme.color("text.onInk") if on else css_color(gui_tokens.ON_ACCENT))
    painter.drawEllipse(knob)
    painter.restore()


class Switch(QtWidgets.QAbstractButton, CssBox):
    """``.switch``: 32×18 track (off ``control.switchOff``, on ink) with a 14 px knob; label 13 px after it.
    The track snaps with the box; the label keeps the fraction."""

    LABEL = BODY
    OVERHANG = True

    def __init__(
        self,
        themes: ThemeManager,
        text: str = "",
        *,
        gap: int = 8,
        style: TextStyle | None = None,
        label_token: str = "text.body",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes, self.gap, self.label_token = themes, gap, label_token
        self.label_style = style or self.LABEL
        self.setCheckable(True)
        self.setText(text)
        self.setAccessibleName(text)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 9)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        themes.changed.connect(self.update)

    def natural_width(self) -> float:
        return 32 + (self.gap + self.label_style.advance(self.text()) if self.text() else 0)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 18)

    def painted_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, (self.height() - 18) // 2, 32, 18)

    def hitButton(self, pos: QtCore.QPoint) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        theme = self.themes.current
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        top = (self.height() - 18) // 2
        paint_switch_track(painter, QtCore.QPointF(0, top), self.isChecked(), theme)
        if self.text():
            color = theme.color(self.label_token if self.isEnabled() else "text.disabled")
            draw_text(
                painter,
                self.shift + 32 + self.gap,
                (self.height() - self.label_style.line) / 2,
                self.text(),
                self.label_style,
                color,
            )


# ---------- Check boxes and radio buttons ----------


class CheckBox(QtWidgets.QAbstractButton):
    """``.check`` with a label (16 px box, 1.5 px border; ink fill and mark when on, a bar when mixed)."""

    def __init__(
        self,
        themes: ThemeManager,
        text: str = "",
        *,
        gap: int = 8,
        style: TextStyle = BODY,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes, self.gap, self.text_style = themes, gap, style
        self.mixed = False
        self.setCheckable(True)
        self.setText(text)
        self.setAccessibleName(text)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 4)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.toggled.connect(self._clear_mixed)
        themes.changed.connect(self.update)

    def _clear_mixed(self, checked: bool) -> None:
        self.mixed = False

    def checkState(self) -> QtCore.Qt.CheckState:
        if self.mixed:
            return QtCore.Qt.CheckState.PartiallyChecked
        return QtCore.Qt.CheckState.Checked if self.isChecked() else QtCore.Qt.CheckState.Unchecked

    def setCheckState(self, state: QtCore.Qt.CheckState) -> None:
        self.setChecked(state == QtCore.Qt.CheckState.Checked)
        self.mixed = state == QtCore.Qt.CheckState.PartiallyChecked
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        width = 16 + (self.gap + self.text_style.advance(self.text()) if self.text() else 0)
        return QtCore.QSize(math.ceil(width), max(16, math.ceil(self.text_style.line)))

    def painted_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, (self.height() - 16) // 2, 16, 16)

    def hitButton(self, pos: QtCore.QPoint) -> bool:
        return self.rect().contains(pos)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)  # Space toggles; Enter is left to the dialog or form

    def paint_indicator(self, painter: QtGui.QPainter, theme: Theme) -> None:
        mark: Mark = "mixed" if self.mixed else "on" if self.isChecked() else "off"
        paint_check(painter, self.painted_rect(), theme, mark, self.isEnabled())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        theme = self.themes.current
        painter = QtGui.QPainter(self)
        self.paint_indicator(painter, theme)
        if self.text():
            color = theme.color("text.body" if self.isEnabled() else "text.disabled")
            line_top = (self.height() - self.text_style.line) / 2
            draw_text(painter, 16 + self.gap, line_top, self.text(), self.text_style, color)


class RadioButton(CheckBox):
    """``.radio``: 16 px circle; ink border and an 8 px ink dot when on. Exclusive within its parent."""

    def __init__(
        self,
        themes: ThemeManager,
        text: str = "",
        *,
        gap: int = 8,
        style: TextStyle = BODY,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, text, gap=gap, style=style, parent=parent)
        self.setAutoExclusive(True)
        self.setProperty("focusRadius", 8)

    def paint_indicator(self, painter: QtGui.QPainter, theme: Theme) -> None:
        paint_radio(painter, self.painted_rect(), theme, self.isChecked(), self.isEnabled())


# ---------- Underline tabs ----------


TAB_TEXT = BODY_MEDIUM
TAB_COUNT = TextStyle(11.5, 400, 18, tabular=True)


class Tab(QtWidgets.QAbstractButton):
    """One ``.tabs a``: 38 px, icon, label, optional count (11.5 px tertiary) and change pill, gap 7."""

    def __init__(self, bar: "UnderlineTabs", text: str, icon: str | None) -> None:
        super().__init__(bar)
        self.bar, self.glyph = bar, icon
        self.count: int | None = None
        self.changes = 0
        self.setText(text)
        self.setAccessibleName(text)
        self.setCheckable(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 4)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not press_on_enter(self, event):
            super().keyPressEvent(event)

    def natural_width(self) -> float:
        width = (16 + 7 if self.glyph else 0) + TAB_TEXT.advance(self.text())
        if self.count is not None:
            width += 7 + TAB_COUNT.advance(f"{self.count:,}")
        if self.changes:
            width += 7 + change_pill_width(self.changes)
        return width

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 38)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        theme = self.bar.themes.current
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        active = self.isChecked()
        hovered = self.underMouse() and not active
        color = theme.color("text.ink" if active else "text.body" if hovered else "text.secondary")
        x = 0.0
        if self.glyph:
            icon_color = theme.color("text.ink" if active else "text.tertiary")
            gui_icons.draw(painter, self.glyph, icon_color, QtCore.QPointF(0, 11))
            x += 16 + 7
        x += draw_text(painter, x, 10, self.text(), TAB_TEXT, color)
        if self.count is not None:
            x += 7
            x += draw_text(
                painter, x, 10, f"{self.count:,}", TAB_COUNT, theme.color("text.tertiary")
            )
        if self.changes:
            paint_change_pill(painter, x + 7, 11, self.changes, theme)
        if active:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("text.ink"))
            painter.drawPath(rounded_path(QtCore.QRectF(0, 37, self.width(), 2), 1))


class UnderlineTabs(Themed):
    """``.tabs``: 38 px tabs ``gap`` px apart (24; 20 on Profiles, 22 in the editor) over a 1 px
    ``border.line`` rule (``rule=False``: the container draws it); the active tab is ink with a 2 px underline
    over the rule. ``changed`` carries the new index; arrow keys move between tabs."""

    changed = QtCore.Signal(int)

    def __init__(
        self,
        themes: ThemeManager,
        tabs: Sequence[tuple[str, str | None]],
        parent: QtWidgets.QWidget | None = None,
        *,
        gap: float = 24,
        rule: bool = True,
    ) -> None:
        super().__init__(themes, parent)
        self.gap, self.rule = gap, rule
        self.setFixedHeight(39)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.tabs: list[Tab] = []
        for index, (text, icon) in enumerate(tabs):
            tab = Tab(self, text, icon)
            self.group.addButton(tab, index)
            self.tabs.append(tab)
        self.group.idToggled.connect(self._toggled)
        if self.tabs:
            self.tabs[0].setChecked(True)
        self.relayout()

    def relayout(self) -> None:
        x = 0.0
        for tab in self.tabs:
            width = tab.natural_width()
            tab.setGeometry(snap(x), 0, snap(x + width) - snap(x), 39)
            tab.update()
            x += width + self.gap
        self.setMinimumWidth(math.ceil(max(0.0, x - self.gap)))

    def natural_width(self) -> float:
        widths = [tab.natural_width() for tab in self.tabs]
        return sum(widths) + self.gap * max(0, len(widths) - 1)

    def _toggled(self, index: int, checked: bool) -> None:
        if checked:
            self.changed.emit(index)

    def index(self) -> int:
        return self.group.checkedId()

    def set_index(self, index: int) -> None:
        if 0 <= index < len(self.tabs) and index != self.index():
            self.tabs[index].setChecked(True)

    def set_count(self, index: int, count: int | None) -> None:
        tab = self.tabs[index]
        if tab.count != count:
            tab.count = count
            self.relayout()

    def set_changes(self, index: int, changes: int) -> None:
        tab = self.tabs[index]
        if tab.changes != changes:
            tab.changes = changes
            self.relayout()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        step = {QtCore.Qt.Key.Key_Left: -1, QtCore.Qt.Key.Key_Right: 1}.get(
            QtCore.Qt.Key(event.key()), 0
        )
        if step and self.tabs:
            target = self.tabs[(max(0, self.index()) + step) % len(self.tabs)]
            target.setChecked(True)
            target.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self.rule:
            QtGui.QPainter(self).fillRect(
                QtCore.QRect(0, 38, self.width(), 1), self.color("border.line")
            )


# ---------- Selects ----------


SELECT_TEXT = BODY


class KeySelect(QtWidgets.QComboBox):
    """``.select``: 32 px field showing an optional key in ``text.secondary`` before the value in ink, and a
    16 px ``cdown`` chevron in ``text.secondary``; the list opens in the menu style."""

    def __init__(
        self,
        themes: ThemeManager,
        key: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes, self.key = themes, key
        self.setFixedHeight(32)
        self.setProperty("focusRadius", 6)
        self.setAccessibleName(key)
        themes.changed.connect(self.update)

    def sizeHint(self) -> QtCore.QSize:
        widest = max(
            (SELECT_TEXT.advance(self.itemText(i)) for i in range(self.count())), default=0.0
        )
        key = SELECT_TEXT.advance(self.key) + 2 + SELECT_TEXT.advance(" ") if self.key else 0.0
        return QtCore.QSize(math.ceil(2 + 10 + key + widest + 8 + 16 + 10), 32)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(2 + 10 + 16 + 10, 32)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        theme = self.themes.current
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        painter.setPen(QtGui.QPen(theme.color("border.strong" if enabled else "border.line"), 1))
        painter.setBrush(theme.color("surface.card" if enabled else "surface.sunken"))
        box = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawPath(rounded_path(box, 5.5))
        chevron = theme.color("text.secondary" if enabled else "text.disabled")
        gui_icons.draw(painter, "cdown", chevron, QtCore.QPointF(self.width() - 11 - 16, 8))
        x, room = 11.0, self.width() - 11 - 16 - 8 - 11.0
        if self.key:
            key_color = theme.color("text.secondary" if enabled else "text.disabled")
            x += draw_text(painter, x, 7, self.key, SELECT_TEXT, key_color) + 2
            room -= x - 11
            value = " " + self.currentText()
        else:
            value = self.currentText()
        value_color = theme.color("text.ink" if enabled else "text.disabled")
        draw_text(painter, x, 7, SELECT_TEXT.elide(value, max(0.0, room)), SELECT_TEXT, value_color)


# ---------- Text fields ----------


class SearchField(QtWidgets.QLineEdit):
    """``.input`` with a leading 16 px ``search`` icon in ``text.tertiary`` and an 8 px gap."""

    def __init__(
        self,
        themes: ThemeManager,
        placeholder: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes = themes
        self.setPlaceholderText(placeholder)
        self.setAccessibleName(placeholder)
        self.setFixedHeight(32)
        self.setTextMargins(16 + 8, 0, 0, 0)
        self.setClearButtonEnabled(False)
        themes.changed.connect(self.update)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        gui_icons.draw(
            painter, "search", self.themes.current.color("text.tertiary"), QtCore.QPointF(11, 8)
        )


UNIT_TEXT = TextStyle(13, 400, 18, tabular=True)


# QLineEdit keeps 2 px between its frame and the text, and after right-aligned text 1 px plus the cursor;
# edits placed by hand are widened by these so the text lands on the CSS positions.
EDIT_MARGIN, CURSOR_ROOM = 2, 2


class _UnitEdit(QtWidgets.QLineEdit):
    """The spin box's line edit: right-aligned tabular value with the unit painted 6 px after it."""

    def __init__(self, box: "SpinBox | DoubleSpinBox", weight: int) -> None:
        super().__init__()
        self.box = box
        self.setObjectName("unitEdit")
        self.setFrame(False)
        self.setFont(font(13, weight, tabular=True))
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    def unit_room(self) -> int:
        return round(6 + UNIT_TEXT.advance(self.box.unit)) if self.box.unit else 0

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self.box.unit:
            # ``.unit`` keeps its colour when the spin box is disabled.
            color = self.box.themes.current.color("text.tertiary")
            draw_text(
                QtGui.QPainter(self),
                self.width() - EDIT_MARGIN - CURSOR_ROOM - UNIT_TEXT.advance(self.box.unit),
                (self.height() - 18) / 2,
                self.box.unit,
                UNIT_TEXT,
                color,
            )


class _Steppers:
    """The 24 px stepper column of ``.spin``: a left rule, ``cup`` and ``cdown`` halves split by a rule."""

    WIDTH = 24

    @staticmethod
    def paint(widget: QtWidgets.QAbstractSpinBox, theme: Theme) -> None:
        painter = QtGui.QPainter(widget)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        enabled = widget.isEnabled()
        painter.setPen(QtGui.QPen(theme.color("border.strong" if enabled else "border.line"), 1))
        painter.setBrush(theme.color("surface.card" if enabled else "surface.sunken"))
        box = QtCore.QRectF(widget.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawPath(rounded_path(box, 5.5))
        left = widget.width() - 1 - _Steppers.WIDTH
        height = widget.height() - 2
        rule = theme.color("border.line")
        painter.fillRect(QtCore.QRectF(left, 1, 1, height), rule)
        half = (height - 1) / 2
        painter.fillRect(QtCore.QRect(left + 1, snap(1 + half), _Steppers.WIDTH - 1, 1), rule)
        color = theme.color("text.secondary")  # ``.spin .st span`` keeps its colour when disabled
        center = left + 1 + (_Steppers.WIDTH - 1 - 14) / 2
        gui_icons.draw(
            painter, "cup", color, QtCore.QPointF(snap(center), snap(1 + (half - 14) / 2)), 14
        )
        gui_icons.draw(
            painter,
            "cdown",
            color,
            QtCore.QPointF(snap(center), snap(2 + half + (half - 14) / 2)),
            14,
        )

    @staticmethod
    def hit(widget: QtWidgets.QAbstractSpinBox, position: QtCore.QPoint) -> int:
        """+1 on the upper half, −1 on the lower half, 0 elsewhere."""
        if position.x() < widget.width() - 1 - _Steppers.WIDTH:
            return 0
        return 1 if position.y() < widget.height() / 2 else -1


class _SpinMixin:
    """Shared look, stepping and events of ``SpinBox`` and ``DoubleSpinBox`` (``.spin``)."""

    themes: ThemeManager
    unit: str

    def _setup(self, themes: ThemeManager, unit: str, weight: int = 400) -> None:
        box = self._box()
        self.themes, self.unit = themes, unit
        edit = _UnitEdit(self, weight)  # type: ignore[arg-type]
        box.setLineEdit(edit)
        box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        # The wheel steps only a focused box (``wheelEvent``); it neither takes focus nor changes a value
        # the pointer merely passes over while a page scrolls.
        box.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        box.setFixedHeight(32)
        box.setProperty("focusRadius", 6)
        box.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        # A press steps once, then repeats every 80 ms after a 400 ms hold.
        self._delay = QtCore.QTimer(box)
        self._delay.setSingleShot(True)
        self._delay.setInterval(400)
        self._repeat = QtCore.QTimer(box)
        self._repeat.setInterval(80)
        self._delay.timeout.connect(self._repeat.start)
        self._repeat.timeout.connect(self._step)
        self._direction = 0
        themes.changed.connect(box.update)
        self._place_edit()

    def _box(self) -> QtWidgets.QAbstractSpinBox:
        assert isinstance(self, QtWidgets.QAbstractSpinBox)
        return self

    def _place_edit(self) -> None:
        """Value area of ``.spin .v``: 1 px border, padding 0 10, then the stepper column."""
        box = self._box()
        edit = box.lineEdit()
        if isinstance(edit, _UnitEdit):
            left = 11 - EDIT_MARGIN
            right = box.width() - 1 - _Steppers.WIDTH - 10 + EDIT_MARGIN + CURSOR_ROOM
            edit.setTextMargins(0, 0, edit.unit_room(), 0)
            edit.setGeometry(left, 1, max(0, right - left), box.height() - 2)

    def _step(self) -> None:
        if self._direction:
            self._box().stepBy(self._direction)

    def _stop(self) -> None:
        self._direction = 0
        if hasattr(self, "_delay"):  # hide events may arrive before _setup ran
            self._delay.stop()
            self._repeat.stop()

    # QAbstractSpinBox lays its line edit out again on these events; the edit is placed by hand.

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        QtWidgets.QAbstractSpinBox.resizeEvent(self._box(), event)
        self._place_edit()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        QtWidgets.QAbstractSpinBox.showEvent(self._box(), event)
        self._place_edit()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        QtWidgets.QAbstractSpinBox.changeEvent(self._box(), event)
        self._place_edit()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        _Steppers.paint(self._box(), self.themes.current)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        box = self._box()
        direction = _Steppers.hit(box, event.position().toPoint())
        if direction and box.isEnabled() and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._direction = direction
            box.stepBy(direction)
            self._delay.start()
            return
        QtWidgets.QAbstractSpinBox.mousePressEvent(box, event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._stop()
        QtWidgets.QAbstractSpinBox.mouseReleaseEvent(self._box(), event)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._stop()
        QtWidgets.QAbstractSpinBox.hideEvent(self._box(), event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        box = self._box()
        if box.hasFocus():
            QtWidgets.QAbstractSpinBox.wheelEvent(box, event)
        else:
            event.ignore()  # the scroll area under it scrolls


class SpinBox(_SpinMixin, QtWidgets.QSpinBox):
    """``.spin`` for integers: value right-aligned with tabular figures, the unit inside in
    ``text.tertiary``, and a 24 px stepper column."""

    def __init__(
        self, themes: ThemeManager, unit: str = "", parent: QtWidgets.QWidget | None = None
    ) -> None:
        QtWidgets.QSpinBox.__init__(self, parent)
        self._setup(themes, unit)


class DoubleSpinBox(_SpinMixin, QtWidgets.QDoubleSpinBox):
    """``.spin`` for decimals (same look as ``SpinBox``); ``weight`` is the value's font weight."""

    def __init__(
        self,
        themes: ThemeManager,
        unit: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        weight: int = 400,
    ) -> None:
        QtWidgets.QDoubleSpinBox.__init__(self, parent)
        self._setup(themes, unit, weight)
