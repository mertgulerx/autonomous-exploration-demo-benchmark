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

"""Menus and tooltips of the Main design (HANDOFF §7.12).

Menus are in-window popups on the window's ``PopupHost`` (exact card, shadow and keyboard handling, and they
appear in window captures). Tooltips are small top-level windows, because they may extend past the window.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
import math

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from . import gui_icons, gui_tokens
from .gui_theme import Theme, ThemeManager
from .gui_widgets import (
    PopupHost,
    TextStyle,
    Themed,
    draw_text,
    rounded_path,
    shadow_image,
    shadow_margin,
)

# ---------- Menus ----------


ITEM_TEXT = TextStyle(13, 400, 18)
REASON_TEXT = TextStyle(12, 400, 18)
ITEM_HEIGHT, SEPARATOR_HEIGHT, INSET = 32, 9, 5  # INSET: 1 px border plus 4 px padding


@dataclass(frozen=True, slots=True)
class MenuItem:
    text: str = ""
    icon: str | None = None
    callback: Callable[[], None] | None = None
    enabled: bool = True
    reason: str = ""  # why the item is disabled, right-aligned in 12 px tertiary
    danger: bool = False
    separator: bool = False


def menu_item_width(item: MenuItem) -> float:
    width = 10 + (16 + 10 if item.icon else 0) + ITEM_TEXT.advance(item.text) + 10
    return width + (10 + REASON_TEXT.advance(item.reason) if item.reason else 0)


def paint_menu_item(
    painter: QtGui.QPainter, rect: QtCore.QRectF, item: MenuItem, theme: Theme, hovered: bool
) -> None:
    """``.menu .mi``: 32 px, padding 0 10, gap 10, radius 6; hover fill, danger red, disabled grey with the
    reason at the right. Icons keep ``text.secondary`` unless the item is a danger item (as in the CSS)."""
    if item.separator:
        painter.fillRect(
            QtCore.QRectF(rect.x() + 6, rect.y() + 4, rect.width() - 12, 1),
            theme.color("border.line"),
        )
        return
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    if hovered and item.enabled:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("surface.hover"))
        painter.drawPath(rounded_path(rect, 6))
    text_token = "text.disabled" if not item.enabled else "state.bad" if item.danger else "text.ink"
    x = rect.x() + 10
    if item.icon:
        icon_token = "state.bad" if item.danger and item.enabled else "text.secondary"
        gui_icons.draw(
            painter,
            item.icon,
            theme.color(icon_token),
            QtCore.QPointF(x, rect.y() + (ITEM_HEIGHT - 16) // 2),
        )
        x += 16 + 10
    draw_text(painter, x, rect.y() + 7, item.text, ITEM_TEXT, theme.color(text_token))
    if item.reason:
        draw_text(
            painter,
            rect.right() - 10 - REASON_TEXT.advance(item.reason),
            rect.y() + 7,
            item.reason,
            REASON_TEXT,
            theme.color("text.tertiary"),
        )
    painter.restore()


class Menu(Themed):
    """``.menu``: card fill, 1 px ``border.line``, radius 8, padding 4, minimum width 200; 32 px items.

    Open it with ``popup`` (or ``popup_at``) on the window's ``PopupHost``. Arrow keys, Home and End move
    between enabled items; Enter or Space chooses; Esc or a click outside closes it.
    """

    def __init__(
        self, themes: ThemeManager, *, min_width: int = 200, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.min_width = min_width
        self.items: list[MenuItem] = []
        self.hovered = -1
        self.host: PopupHost | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "none")

    # building

    def add_item(
        self,
        text: str,
        callback: Callable[[], None] | None = None,
        *,
        icon: str | None = None,
        enabled: bool = True,
        reason: str = "",
        danger: bool = False,
    ) -> int:
        self.items.append(MenuItem(text, icon, callback, enabled, reason, danger))
        return len(self.items) - 1

    def add_separator(self) -> None:
        self.items.append(MenuItem(separator=True))

    def set_enabled(self, index: int, enabled: bool, reason: str = "") -> None:
        self.items[index] = replace(self.items[index], enabled=enabled, reason=reason)
        self.update()

    # geometry

    def rows(self) -> list[tuple[QtCore.QRectF, MenuItem]]:
        y, inner = float(INSET), self.width() - 2 * INSET
        rows = []
        for item in self.items:
            height = SEPARATOR_HEIGHT if item.separator else ITEM_HEIGHT
            rows.append((QtCore.QRectF(INSET, y, inner, height), item))
            y += height
        return rows

    def sizeHint(self) -> QtCore.QSize:
        width = max(
            [float(self.min_width)]
            + [menu_item_width(item) + 2 * INSET for item in self.items if not item.separator]
        )
        height = 2 * INSET + sum(
            SEPARATOR_HEIGHT if item.separator else ITEM_HEIGHT for item in self.items
        )
        return QtCore.QSize(math.ceil(width), height)

    # opening

    def popup(self, host: PopupHost, anchor: QtWidgets.QWidget, *, align: str = "right") -> None:
        """Open 4 px below ``anchor`` (above it when there is no room), aligned to its right or left edge."""
        window = host.window_widget
        origin = anchor.mapTo(window, QtCore.QPoint(0, 0))
        self.popup_at(host, QtCore.QRect(origin, anchor.size()), align=align, opener=anchor)

    def popup_at(
        self,
        host: PopupHost,
        anchor: QtCore.QRect,
        *,
        align: str = "right",
        opener: QtWidgets.QWidget | None = None,
    ) -> None:
        size = self.sizeHint()
        window = host.window_widget
        x = anchor.right() + 1 - size.width() if align == "right" else anchor.left()
        x = max(8, min(x, window.width() - 8 - size.width()))
        y = anchor.bottom() + 1 + 4
        if y + size.height() > window.height() - 8:
            y = max(8, anchor.top() - 4 - size.height())
        self.host = host
        self.hovered = -1
        rect = QtCore.QRect(x, y, size.width(), size.height())
        # A menu is built for one opening: the host deletes it when it closes.
        host.open_popup(self, rect, opener=opener, radius=8, delete_on_close=True)
        self.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def dismiss(self) -> None:
        if self.host is not None and self.host.popup is self:
            self.host.close_popup()

    # interaction

    def _index_at(self, position: QtCore.QPointF) -> int:
        for index, (rect, item) in enumerate(self.rows()):
            if rect.contains(position) and not item.separator:
                return index
        return -1

    def _choose(self, index: int) -> None:
        item = self.items[index]
        if not item.enabled or item.separator:
            return
        self.dismiss()
        if item.callback is not None:
            item.callback()

    def _move(self, step: int) -> None:
        enabled = [i for i, item in enumerate(self.items) if item.enabled and not item.separator]
        if not enabled:
            return
        if self.hovered not in enabled:
            self.hovered = enabled[0] if step > 0 else enabled[-1]
        else:
            self.hovered = enabled[(enabled.index(self.hovered) + step) % len(enabled)]
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self._index_at(event.position())
        if index != self.hovered:
            self.hovered = index
            self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hovered = -1
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            index = self._index_at(event.position())
            if index >= 0:
                self._choose(index)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Tab):
            self._move(1)
        elif key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Backtab):
            self._move(-1)
        elif key == QtCore.Qt.Key.Key_Home:
            self.hovered = -1
            self._move(1)
        elif key == QtCore.Qt.Key.Key_End:
            self.hovered = -1
            self._move(-1)
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space):
            if self.hovered >= 0:
                self._choose(self.hovered)
        else:
            super().keyPressEvent(event)

    def focusNextPrevChild(self, next: bool) -> bool:
        self._move(1 if next else -1)
        return True

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7.5)
        )
        for index, (rect, item) in enumerate(self.rows()):
            paint_menu_item(painter, rect, item, self.theme, index == self.hovered)


# ---------- Tooltips ----------


TOOLTIP_TEXT = TextStyle(12, 400, 16)
TOOLTIP_MAX_WIDTH = 260


def tooltip_lines(text: str) -> list[str]:
    """Words wrapped to the 240 px text width of a 260 px tooltip (padding 7 10)."""
    room = TOOLTIP_MAX_WIDTH - 20
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split(" "):
            candidate = f"{line} {word}" if line else word
            if line and TOOLTIP_TEXT.advance(candidate) > room:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def paint_tooltip(
    painter: QtGui.QPainter,
    x: float,
    y: float,
    lines: list[str],
    theme: Theme,
    width: float | None = None,
) -> QtCore.QRectF:
    """``.tooltip``: inverted fill, radius 6, padding 7 10, 12/16 text; as wide as its longest line unless
    ``width`` is given; returns the box."""
    if width is None:
        width = math.ceil(max((TOOLTIP_TEXT.advance(line) for line in lines), default=0.0) + 20)
    box = QtCore.QRectF(x, y, width, 14 + 16 * len(lines))
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(theme.color("overlay.tooltip"))
    painter.drawPath(rounded_path(box, 6))
    for index, line in enumerate(lines):
        draw_text(
            painter,
            x + 10,
            y + 7 + 16 * index,
            line,
            TOOLTIP_TEXT,
            theme.color("overlay.tooltipText"),
        )
    painter.restore()
    return box


class TooltipWindow(QtWidgets.QWidget):
    """A frameless, translucent top-level window that paints one tooltip and its shadow."""

    LAYERS = (gui_tokens.TOOLTIP_SHADOW,)

    def __init__(self, themes: ThemeManager) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.ToolTip
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowTransparentForInput,
        )
        self.themes = themes
        self.lines: list[str] = []
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def show_text(self, text: str, near: QtCore.QPoint) -> None:
        self.lines = tooltip_lines(text)
        margin = shadow_margin(self.LAYERS)
        width = math.ceil(max(TOOLTIP_TEXT.advance(line) for line in self.lines) + 20)
        height = 14 + 16 * len(self.lines)
        self.resize(width + 2 * margin, height + 2 * margin)
        position = near + QtCore.QPoint(0, 20) - QtCore.QPoint(margin, margin)
        screen = QtGui.QGuiApplication.screenAt(near)
        if screen is not None:
            area = screen.availableGeometry()
            position.setX(
                max(area.left() - margin, min(position.x(), area.right() + margin - self.width()))
            )
            if position.y() + self.height() - margin > area.bottom():
                position.setY(near.y() - 8 - height - margin)
        self.move(position)
        self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        margin = shadow_margin(self.LAYERS)
        width, height = self.width() - 2 * margin, self.height() - 2 * margin
        painter.drawImage(QtCore.QPoint(0, 0), shadow_image(width, height, 6, self.LAYERS))
        paint_tooltip(painter, margin, margin, self.lines, self.themes.current)


class Tooltips(QtCore.QObject):
    """Application-wide tooltips in the design's style instead of ``QToolTip`` (HANDOFF §7.12)."""

    HIDE_EVENTS = (
        QtCore.QEvent.Type.Leave,
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QEvent.Type.KeyPress,
        QtCore.QEvent.Type.Wheel,
        QtCore.QEvent.Type.Hide,
        QtCore.QEvent.Type.WindowDeactivate,
    )

    def __init__(self, themes: ThemeManager, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.window = TooltipWindow(themes)
        self.target: QtWidgets.QWidget | None = None
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(10000)
        self.timer.timeout.connect(self.hide)
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.installEventFilter(self)

    def show(self, widget: QtWidgets.QWidget, global_pos: QtCore.QPoint, text: str) -> None:
        if not text:
            self.hide()
            return
        self.target = widget
        self.window.show_text(text, global_pos)
        self.timer.start()

    def hide(self) -> None:
        self.target = None
        self.timer.stop()
        if self.window.isVisible():
            self.window.hide()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if kind == QtCore.QEvent.Type.ToolTip:
            if (
                isinstance(watched, QtWidgets.QWidget)
                and isinstance(event, QtGui.QHelpEvent)
                and watched.toolTip()
            ):
                self.show(watched, event.globalPos(), watched.toolTip())
                return True
            return False
        if kind in self.HIDE_EVENTS and self.target is not None:
            if kind in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.MouseButtonPress) or (
                watched is self.target
            ):
                self.hide()
        return False


_TOOLTIPS: list[Tooltips] = []


def install_tooltips(themes: ThemeManager, parent: QtCore.QObject) -> Tooltips:
    """Install the design's tooltips for the application (one at a time; it lives as long as ``parent``)."""
    current = _active_tooltips()
    if current is not None:
        return current
    tooltips = Tooltips(themes, parent)
    _TOOLTIPS[:] = [tooltips]
    tooltips.destroyed.connect(_TOOLTIPS.clear)
    return tooltips


def _active_tooltips() -> Tooltips | None:
    if _TOOLTIPS and isValid(_TOOLTIPS[0]):
        return _TOOLTIPS[0]
    _TOOLTIPS.clear()
    return None


def show_tooltip(widget: QtWidgets.QWidget, global_pos: QtCore.QPoint, text: str) -> None:
    """Show ``text`` for a region of ``widget`` (widgets that compute tooltips per row or cell)."""
    tooltips = _active_tooltips()
    if tooltips is not None:
        tooltips.show(widget, global_pos, text)
    elif text:
        QtWidgets.QToolTip.showText(global_pos, text, widget)
    else:
        QtWidgets.QToolTip.hideText()
