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

"""Top bar of the Main design (HANDOFF §7.2): wordmark, centred navigation, engine state, path colours, theme
switch."""

from functools import partial
import math

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons, gui_tokens
from .gui_menus import Menu
from .gui_theme import PATH_COLOURS, Theme, ThemeManager, css_color
from .gui_widgets import (
    Button,
    PopupHost,
    SegmentedControl,
    TextStyle,
    Themed,
    draw_text,
    press_on_enter,
    rounded_path,
)

NAVIGATION = (
    ("campaigns", "grid", "Campaigns"),
    ("monitor", "activity", "Monitor"),
    ("profiles", "sliders", "Profiles"),
    ("worlds", "map", "Worlds & Spawns"),
    ("results", "chart", "Results"),
)
WORDMARK = TextStyle(15, 600, 18, spacing_em=-0.005)
SUBTITLE = TextStyle(13, 400, 18)
NAV_TEXT = TextStyle(13, 500, 18)
NAV_ACTIVE = TextStyle(13, 600, 18)
PILL = TextStyle(11, 600, 18, tabular=True)
ENGINE = TextStyle(12.5, 400, 18)


class Brand(Themed):
    """28 px ink mark with the ``nav`` icon, "Exploration" and the ruled "Benchmark Suite" (left out when
    ``compact``: the centred navigation would cover it in a narrow window)."""

    compact = False

    def natural_width(self, compact: bool) -> float:
        width = 28 + 10 + WORDMARK.advance("Exploration")
        if not compact:
            width += 10 + 2 + 1 + 10 + SUBTITLE.advance("Benchmark Suite")
        return width

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width(False)), 28)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.color("text.ink"))
        painter.drawPath(rounded_path(QtCore.QRectF(0, 0, 28, 28), 6))
        gui_icons.draw(painter, "nav", self.color("text.onInk"), QtCore.QPointF(6, 6))
        line_top = (self.height() - 18) / 2
        x = 38 + draw_text(painter, 38, line_top, "Exploration", WORDMARK, self.color("text.ink"))
        if self.compact:
            return
        x += 12  # flex gap 10 plus the subtitle's 2 px margin
        painter.fillRect(QtCore.QRectF(x, line_top, 1, 18), self.color("border.line"))
        draw_text(
            painter, x + 11, line_top, "Benchmark Suite", SUBTITLE, self.color("text.tertiary")
        )


class NavItem(QtWidgets.QAbstractButton):
    """34 px item: 14 px padding, 16 px icon, 7 px gaps, optional live-worker pill (Monitor)."""

    def __init__(self, bar: "NavBar", key: str, icon: str, text: str) -> None:
        super().__init__(bar)
        self.bar, self.key, self.glyph = bar, key, icon
        self.live = 0
        self.shift = 0.0  # fractional x of the item in the window; painting moves by it
        self.setText(text)
        self.setAccessibleName(text)
        self.setCheckable(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 8)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)

    def pill_width(self) -> float:
        return 6 + 6 + 4 + PILL.advance(str(self.live)) + 6 if self.live else 0.0

    def natural_width(self) -> float:
        style = NAV_ACTIVE if self.isChecked() else NAV_TEXT
        width = 14 + 16 + 7 + style.advance(self.text()) + 14
        return width + (7 + self.pill_width() if self.live else 0)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 34)

    def painted_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.shift, 0, self.natural_width(), 34)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # Enter navigates like Space; arrows move focus only (the bar handles them), never navigate.
        if press_on_enter(self, event):
            return
        step = {QtCore.Qt.Key.Key_Left: -1, QtCore.Qt.Key.Key_Right: 1}.get(
            QtCore.Qt.Key(event.key()), 0
        )
        if step:
            self.bar.move_focus(self, step)
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.translate(self.shift, 0)
        theme = self.bar.themes.current
        active = self.isChecked()
        width = self.natural_width()
        if active or self.underMouse():
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("accent.soft" if active else "surface.hover"))
            painter.drawPath(rounded_path(QtCore.QRectF(0, 0, width, 34), 8))
        icon_color = theme.color("accent.softText" if active else "text.tertiary")
        text_color = theme.color("accent.softText" if active else "text.secondary")
        gui_icons.draw(painter, self.glyph, icon_color, QtCore.QPointF(14, 9))
        style = NAV_ACTIVE if active else NAV_TEXT
        x = 37 + draw_text(painter, 37, 8, self.text(), style, text_color)
        if self.live:
            self._paint_pill(painter, x + 7, active)

    def _paint_pill(self, painter: QtGui.QPainter, x: float, active: bool) -> None:
        theme = self.bar.themes.current
        white = css_color(gui_tokens.ON_ACCENT)
        fill = theme.color("accent.primary" if active else "accent.soft")
        dot = white if active else theme.color("accent.primary")
        text = white if active else theme.color("accent.softText")
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPath(rounded_path(QtCore.QRectF(x, 8, self.pill_width(), 18), 9))
        painter.setBrush(dot)
        painter.drawEllipse(QtCore.QRectF(x + 6, 14, 6, 6))
        draw_text(painter, x + 16, 8, str(self.live), PILL, text)


class NavBar(Themed):
    """The five destinations, 4 px apart; left and right arrows move between them."""

    navigate = QtCore.Signal(str)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.items: dict[str, NavItem] = {}
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        for key, icon, text in NAVIGATION:
            item = NavItem(self, key, icon, text)
            item.clicked.connect(lambda checked=False, k=key: self.navigate.emit(k))
            self.group.addButton(item)
            self.items[key] = item
        self.relayout()

    def restyle(self, *_: object) -> None:
        for item in self.items.values():
            item.update()

    def set_current(self, key: str | None) -> None:
        """Mark ``key`` active; None (a page outside the navigation) leaves no item active."""
        self.group.setExclusive(False)
        for name, item in self.items.items():
            if (name == key) != item.isChecked():
                item.setChecked(name == key)
        self.group.setExclusive(True)
        self.relayout()

    def set_live(self, count: int) -> None:
        item = self.items["monitor"]
        if item.live != count:
            item.live = count
            item.setAccessibleName(f"Monitor, {count} active" if count else "Monitor")
            self.relayout()

    def natural_width(self) -> float:
        return sum(item.natural_width() for item in self.items.values()) + 4 * (len(self.items) - 1)

    def relayout(self, offset: float = 0.0) -> None:
        """Place the items at their CSS positions; ``offset`` is the bar's fractional x in the window."""
        x = offset
        for item in self.items.values():
            width = item.natural_width()
            item.shift = x - math.floor(x)
            item.setGeometry(math.floor(x), 0, math.ceil(item.shift + width), 34)
            item.update()
            x += width + 4
        width = math.ceil(x - 4) + 1
        if self.width() != width:
            self.setFixedSize(width, 34)
        parent = self.parentWidget()
        if isinstance(parent, TopBar) and offset == 0.0:
            parent.place()

    def move_focus(self, item: NavItem, step: int) -> None:
        items = list(self.items.values())
        target = items[(items.index(item) + step) % len(items)]
        target.setFocus(QtCore.Qt.FocusReason.TabFocusReason)


class EngineState(Themed):
    """8 px dot with a 3 px halo and the engine state in 12.5 px ``text.secondary``."""

    STATES = {
        "connected": ("Engine connected", "data.completed", "state.okHalo"),
        "connecting": ("Connecting…", "state.neutral", "state.neutralSoft"),
        "unavailable": ("Engine unavailable", "state.bad", "state.badSoft"),
    }

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.state = "connecting"
        self.shift = 0.0  # fractional x in the window; painting moves by it
        self.setFixedHeight(18)
        self.set_state("connecting")

    def natural_width(self) -> float:
        return 8 + 7 + ENGINE.advance(self.STATES[self.state][0])

    def painted_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.shift, 0, self.natural_width(), 18)

    def set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        self.setToolTip(detail)
        self.setAccessibleName(self.STATES[state][0] + (": " + detail if detail else ""))
        width = math.ceil(self.natural_width()) + 1
        if self.width() != width:
            self.setFixedWidth(width)
            parent = self.parentWidget()
            if isinstance(parent, TopBar):
                parent.place()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.translate(self.shift, 0)
        text, dot, halo = self.STATES[self.state]
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.color(halo))
        painter.drawEllipse(QtCore.QRectF(-3, 2, 14, 14))
        painter.setBrush(self.color(dot))
        painter.drawEllipse(QtCore.QRectF(0, 5, 8, 8))
        draw_text(painter, 15, 0, text, ENGINE, self.color("text.secondary"))


class TopBar(Themed):
    """56 px ``surface.card`` bar with a bottom rule; the navigation is centred on the window."""

    navigate = QtCore.Signal(str)
    theme_requested = QtCore.Signal(str)
    path_colours_requested = QtCore.Signal(str)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.ready = False  # children ask for placement while they are still being built
        self.setFixedHeight(56)
        self.brand = Brand(themes, self)
        self.brand.setFixedSize(self.brand.sizeHint())
        self.nav = NavBar(themes, self)
        self.nav.navigate.connect(self.navigate)
        self.engine = EngineState(themes, self)
        self.switch = SegmentedControl(
            themes,
            [("", "sun"), ("", "moon")],
            segment_width=30,
            idle_token="text.tertiary",
            hover_token="text.body",
            tooltips=("Light theme", "Dark theme"),
            parent=self,
        )
        # The maps' path colours (gui_maps.PATH_SCALES), chosen from a menu.
        self.colours = Button(
            themes, icon="route", variant="ghost", small=True, icon_size=16, parent=self
        )
        self.colours.setToolTip("Path colours")
        self.colours.setAccessibleName("Path colours")
        self.colours.clicked.connect(self._colours_menu)
        self.switch.setAccessibleName("Theme")
        self.switch.setFixedSize(68, 32)
        self.switch.set_index(1 if themes.current.dark else 0)
        self.switch.changed.connect(
            lambda index: self.theme_requested.emit(("light", "dark")[index])
        )
        themes.changed.connect(self._theme_changed)
        self.ready = True
        self.place()

    def _colours_menu(self) -> None:
        host = self.window().findChild(PopupHost)
        if host is None:
            return
        menu = Menu(self.themes)
        for name in PATH_COLOURS:
            current = name == self.themes.path_colours
            menu.add_item(
                name.capitalize(),
                partial(self.path_colours_requested.emit, name),
                enabled=not current,
                reason="Current" if current else "",
            )
        menu.popup(host, self.colours)

    def _theme_changed(self, theme: Theme) -> None:
        self.switch.set_index(1 if theme.dark else 0)

    def place(self) -> None:
        # Vertical positions follow the mockup's line boxes snapped like Chrome (x.5 rounds up).
        if not self.ready:
            return
        width = self.width()
        self.brand.move(24, 14)
        left = (width - self.nav.natural_width()) / 2
        compact = 24 + self.brand.natural_width(False) + 16 > left
        if compact != self.brand.compact:
            self.brand.compact = compact
            self.brand.update()
        self.nav.move(math.floor(left), 11)
        self.nav.relayout(left - math.floor(left))
        self.switch.move(width - 24 - 68, 12)
        self.colours.move(width - 24 - 68 - 8 - 28, 14)
        engine = width - 24 - 68 - 8 - 28 - 16 - self.engine.natural_width()
        self.engine.shift = engine - math.floor(engine)
        self.engine.move(math.floor(engine), 19)
        self.engine.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.place()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.color("surface.card"))
        painter.fillRect(
            QtCore.QRect(0, self.height() - 1, self.width(), 1), self.color("border.line")
        )
