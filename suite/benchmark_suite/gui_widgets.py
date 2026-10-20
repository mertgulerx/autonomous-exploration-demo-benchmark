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

"""Painted foundation components of the Main design (docs/suite-gui-design, HANDOFF §6-§7).

Every colour comes from the active theme; each widget repaints when ``ThemeManager.changed`` fires. Text that
must match the mockups to the pixel is painted: ``draw_text`` shapes it at 64 times its size (Qt rounds
fractional font sizes to whole pixels and advances to 1/64 px) and draws the glyphs from a raw font at the
exact size. Widgets at a browser's fractional positions are placed with ``place`` (``CssBox``).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from . import gui_icons, gui_tokens
from .gui_theme import Theme, ThemeManager, baseline, css_color, font, snap

Shadow = tuple[int, int, int, int, str]
Variant = Literal["secondary", "primary", "ghost", "danger", "danger-solid"]
ToastKind = Literal["ok", "warn", "bad", "neutral"]


def plural(count: int, word: str) -> str:
    """The count (grouped) and the word, with an s unless the count is 1: "1 run", "2 runs"."""
    return f"{count:,} {word if count == 1 else word + 's'}"


def press_on_enter(button: QtWidgets.QAbstractButton, event: QtGui.QKeyEvent) -> bool:
    """Click ``button`` on Return or Enter, as browsers activate a focused button; True when handled."""
    if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and button.isEnabled():
        button.click()
        return True
    return False


def alive(widget: QtWidgets.QWidget | None) -> bool:
    """Whether ``widget`` still exists (its C++ object may be gone while Python holds a reference)."""
    return widget is not None and isValid(widget)


class CssBox:
    """A widget that ``place`` puts at a browser's fractional CSS box.

    Its geometry snaps to whole pixels; it paints its content ``shift`` (x) and ``vshift`` (y) from the snapped
    corner and lays its children out in the exact ``css_width`` and ``css_height`` (None: the widget's size).
    ``OVERHANG`` widgets end in text that may pass the snapped right edge, so they extend to the next whole
    pixel.
    """

    OVERHANG: ClassVar[bool] = False
    shift: float = 0.0
    vshift: float = 0.0
    css_width: float | None = None
    css_height: float | None = None

    def css_changed(self) -> None:
        """Called by ``place`` when the CSS box changed; widgets that lay children out override it."""


@runtime_checkable
class NaturalWidth(Protocol):
    """A painted component that knows its CSS width (the size hint rounds it up)."""

    def natural_width(self) -> float: ...


def css_rect(widget: QtWidgets.QWidget) -> QtCore.QRectF:
    """The CSS box ``place`` gave ``widget``, in the widget's own coordinates (its rect when never placed)."""
    if not isinstance(widget, CssBox):
        return QtCore.QRectF(widget.rect())
    width = float(widget.width()) if widget.css_width is None else widget.css_width
    height = float(widget.height()) if widget.css_height is None else widget.css_height
    return QtCore.QRectF(widget.shift, widget.vshift, width, height)


def place(widget: QtWidgets.QWidget, x: float, y: float, width: float, height: float) -> None:
    """Give ``widget`` the box a browser draws for the CSS box (x, y, width, height): edges snapped half up,
    the fractions kept on ``CssBox`` widgets."""
    left, top = snap(x), snap(y)
    right = snap(x + width)
    changed = False
    if isinstance(widget, CssBox):
        box = (x - left, y - top, width, height)
        changed = box != (widget.shift, widget.vshift, widget.css_width, widget.css_height)
        widget.shift, widget.vshift, widget.css_width, widget.css_height = box
        if widget.OVERHANG:
            right = math.ceil(x + width)
    widget.setGeometry(left, top, max(1, right - left), max(1, snap(y + height) - top))
    if isinstance(widget, CssBox) and changed:
        widget.css_changed()
        widget.update()


# ---------- Text ----------


@dataclass(frozen=True, slots=True)
class TextStyle:
    size: float = 13
    weight: int = 400
    line: float = 18
    mono: bool = False
    tabular: bool = False
    spacing_em: float = 0.0

    def advance(self, text: str) -> float:
        return _advance(self, text)

    def baseline(self, line_top: float) -> float:
        ascent, descent, _ = _vertical_metrics(self)
        return baseline(line_top, self.line, ascent, descent)

    def middle_top(self, content_height: float, box_height: float) -> float:
        """Top of an inline box with ``vertical-align: middle`` (its centre at the baseline raised by half
        the x-height) in a line of this style centred in ``content_height``, as CSS lays out table cells."""
        ascent, descent, x_height = _vertical_metrics(self)
        above = baseline(0, self.line, ascent, descent)  # strut: line top to baseline
        box_top = -x_height / 2 - box_height / 2  # relative to the baseline
        line_top = min(-above, box_top)
        line_bottom = max(self.line - above, box_top + box_height)
        first = (content_height - (line_bottom - line_top)) / 2
        return first - line_top + box_top

    def middle_line(self, box_height: float) -> tuple[float, float]:
        """A line of this style holding an inline box of ``box_height`` aligned ``middle`` (``td > span``): the
        line box's height (taller than ``line`` when the box reaches past the strut) and its baseline from the
        line's top."""
        ascent, descent, x_height = _vertical_metrics(self)
        above = baseline(0, self.line, ascent, descent)
        box_top = -x_height / 2 - box_height / 2
        line_top = min(-above, box_top)
        line_bottom = max(self.line - above, box_top + box_height)
        return line_bottom - line_top, -line_top

    def middle_offset(self, box_height: float) -> float:
        """Top of such a middle-aligned box relative to the line's baseline."""
        return -_vertical_metrics(self)[2] / 2 - box_height / 2

    def middle_baseline(self, content_height: float, box_height: float = 0.0) -> float:
        """Baseline of this style's text in the same line as such a box (0: text alone)."""
        ascent, descent, x_height = _vertical_metrics(self)
        above = baseline(0, self.line, ascent, descent)
        box_top = -x_height / 2 - box_height / 2
        line_top = min(-above, box_top) if box_height else -above
        line_bottom = (
            max(self.line - above, box_top + box_height) if box_height else self.line - above
        )
        return (content_height - (line_bottom - line_top)) / 2 - line_top

    def elide(self, text: str, width: float) -> str:
        if self.advance(text) <= width:
            return text
        layout_font, scale = _layout_font(self)
        return QtGui.QFontMetricsF(layout_font).elidedText(
            text, QtCore.Qt.TextElideMode.ElideRight, width / scale
        )


LAYOUT_SCALE = 64  # text is shaped at this multiple of its size
LAYOUT_UNIT = 64  # Blink lays text out in 1/64 px


@lru_cache(maxsize=64)
def _layout_font(style: TextStyle) -> tuple[QtGui.QFont, float]:
    """Shared, never modified: the font text is shaped with, at 64 times the style's size. Qt keeps
    advances in 1/64 px at the size it shapes, which at 13 px makes runs about 0.1 % narrower than a
    browser's; at this scale they agree within 0.01 px."""
    size = style.size * LAYOUT_SCALE
    shared = font(
        size, style.weight, mono=style.mono, tabular=style.tabular, spacing_em=style.spacing_em
    )
    return shared, 1 / LAYOUT_SCALE


@lru_cache(maxsize=8192)
def _advance(style: TextStyle, text: str) -> float:
    """The width a browser lays a run of text out with: its shaped width rounded up to 1/64 px (Blink's
    layout unit); the glyphs inside keep their exact positions."""
    layout_font, scale = _layout_font(style)
    width = QtGui.QFontMetricsF(layout_font).horizontalAdvance(text) * scale
    return math.ceil(width * LAYOUT_UNIT - 1e-6) / LAYOUT_UNIT


@lru_cache(maxsize=64)
def _vertical_metrics(style: TextStyle) -> tuple[float, float, float]:
    """Ascent, descent and x-height at the style's size (integer sizes as they are, fractional ones
    measured at four times the size)."""
    size = style.size if float(style.size).is_integer() else style.size * 4
    metrics_font = font(
        size, style.weight, mono=style.mono, tabular=style.tabular, spacing_em=style.spacing_em
    )
    metrics, scale = QtGui.QFontMetricsF(metrics_font), style.size / size
    return metrics.ascent() * scale, metrics.descent() * scale, metrics.xHeight() * scale


BODY = TextStyle()
BODY_MEDIUM = TextStyle(13, 500, 18)
BODY_TABULAR = TextStyle(13, 400, 18, tabular=True)
BUTTON = TextStyle(13, 500, 13)
BUTTON_SMALL = TextStyle(12.5, 500, 12.5)
CARD_TITLE = TextStyle(14, 600, 20)
PAGE_TITLE = TextStyle(22, 600, 28, spacing_em=-0.01)
DIALOG_TITLE = TextStyle(16, 600, 22)
LABEL = TextStyle(12.5, 500, 18)
SMALL = TextStyle(12, 400, 16)


@lru_cache(maxsize=4096)
def glyph_runs(text: str, style: TextStyle) -> tuple[QtGui.QGlyphRun, ...]:
    """The shaped glyphs of one line at the style's exact pixel size, positioned from the baseline
    origin (shaping, kerning, features and tracking come from the layout font)."""
    layout_font, scale = _layout_font(style)
    layout = QtGui.QTextLayout(text, layout_font)
    layout.beginLayout()
    line = layout.createLine()
    layout.endLayout()
    runs = []
    for run in line.glyphRuns():
        raw = run.rawFont()
        raw.setPixelSize(style.size)
        run.setRawFont(raw)
        run.setPositions([QtCore.QPointF(point.x() * scale, 0.0) for point in run.positions()])
        runs.append(run)
    return tuple(runs)


def paint_glyphs(
    painter: QtGui.QPainter, x: float, baseline: float, text: str, style: TextStyle
) -> None:
    """One line of glyphs from the baseline origin (x, baseline), in the painter's pen colour.

    Glyphs are rasterised at quarter-pixel x positions: Skia rounds a glyph's x to the nearest quarter,
    Qt floors it to the quarter below. Starting an eighth of a device pixel further right makes Qt pick
    the rendering Skia picks.
    """
    device = painter.device()
    ratio = device.devicePixelRatioF() if device is not None else 1.0
    origin = QtCore.QPointF(x + 0.125 / ratio, baseline)
    for run in glyph_runs(text, style):
        painter.drawGlyphRun(origin, run)


def draw_text(
    painter: QtGui.QPainter,
    x: float,
    line_top: float,
    text: str,
    style: TextStyle,
    color: QtGui.QColor,
) -> float:
    """Paint one line of text in a CSS line box starting at ``line_top``; returns its advance."""
    painter.save()
    painter.setPen(color)
    paint_glyphs(painter, x, style.baseline(line_top), text, style)
    painter.restore()
    return style.advance(text)


# ---------- Themed base ----------


class Themed(QtWidgets.QWidget):
    """A widget that paints from the active theme and repaints when it changes."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.themes = themes
        themes.changed.connect(self.restyle)

    @property
    def theme(self) -> Theme:
        return self.themes.current

    def color(self, token: str) -> QtGui.QColor:
        return self.themes.current.color(token)

    def restyle(self, *_: object) -> None:
        self.update()


class TextLine(Themed, CssBox):
    """One line of painted text (exact fractional sizes); elided to the width it is given."""

    OVERHANG = True

    def __init__(
        self,
        themes: ThemeManager,
        text: str = "",
        style: TextStyle = BODY,
        token: str = "text.body",
        parent: QtWidgets.QWidget | None = None,
        *,
        alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignLeft,
    ) -> None:
        super().__init__(themes, parent)
        self._text, self.text_style, self.token, self.alignment = text, style, token, alignment
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.setAccessibleName(text)

    def text(self) -> str:
        return self._text

    def natural_width(self) -> float:
        return self.text_style.advance(self._text)

    def setText(self, text: str) -> None:
        if text != self._text:
            self._text = text
            self.setAccessibleName(text)
            self.updateGeometry()
            self.update()

    def set_token(self, token: str) -> None:
        if token != self.token:
            self.token = token
            self.update()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            math.ceil(self.text_style.advance(self._text)), math.ceil(self.text_style.line)
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, math.ceil(self.text_style.line))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        room = self.width() - self.shift if self.css_width is None else self.css_width
        text = self.text_style.elide(self._text, room)
        width = self.text_style.advance(text)
        x = (
            self.width() - width
            if self.alignment & QtCore.Qt.AlignmentFlag.AlignRight
            else (self.width() - width) / 2
            if self.alignment & QtCore.Qt.AlignmentFlag.AlignHCenter
            else self.shift
        )
        top = self.vshift + (self.height() - self.text_style.line) / 2
        draw_text(painter, x, top, text, self.text_style, self.color(self.token))


class FlexRow(QtWidgets.QWidget, CssBox):
    """A CSS flex row (``align-items: center``): children at the browser's fractional positions.

    Items before ``add_stretch`` run from the row's left edge, items after it end at its right edge (CSS
    ``margin-left: auto``). Each item's width is its ``natural_width()`` or its size hint, and ``place`` gives
    it the browser's box. ``left`` and ``right`` are the row's paddings; a row placed at a fractional box lays
    its items out in that box.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        gap: float = 8,
        left: float = 0,
        right: float = 0,
    ) -> None:
        super().__init__(parent)
        self.gap, self.left, self.right = gap, left, right
        self.items: list[tuple[QtWidgets.QWidget, float | None, float, float]] = []
        self.stretch_at: int | None = None
        # Items are centred in this height from the top (None: the row's height), e.g. above a rule.
        self.inner_height: float | None = None

    def add(
        self,
        widget: QtWidgets.QWidget,
        *,
        gap: float | None = None,
        width: float | None = None,
        margin: float = 0,
    ) -> QtWidgets.QWidget:
        """Append ``widget``; ``gap`` overrides the row gap before it, ``margin`` adds CSS margins on
        both sides (``.vr``: 4)."""
        widget.setParent(self)
        self.items.append((widget, width, self.gap if gap is None else gap, margin))
        widget.show()
        self.relayout()
        return widget

    def add_stretch(self) -> None:
        self.stretch_at = len(self.items)

    def set_width(self, widget: QtWidgets.QWidget, width: float | None) -> None:
        """Give an item a fixed CSS width (None: its natural width) and lay the row out again."""
        self.items = [
            (item, width if item is widget else size, gap, margin)
            for item, size, gap, margin in self.items
        ]
        self.relayout()

    @staticmethod
    def width_of(widget: QtWidgets.QWidget, width: float | None) -> float:
        if width is not None:
            return width
        if isinstance(widget, NaturalWidth):
            return widget.natural_width()
        return float(widget.sizeHint().width())

    def relayout(self) -> None:
        split = len(self.items) if self.stretch_at is None else self.stretch_at
        height = float(self.height()) if self.inner_height is None else self.inner_height
        x = self.shift + self.left
        for index, (widget, width, gap, margin) in enumerate(self.items[:split]):
            if not widget.isVisibleTo(self):
                continue
            x += (gap if index else 0) + margin
            size = self.width_of(widget, width)
            self._place(widget, x, size, height)
            x += size + margin
        row_width = self.width() if self.css_width is None else self.css_width
        x = self.shift + row_width - self.right
        for widget, width, gap, margin in reversed(self.items[split:]):
            if not widget.isVisibleTo(self):
                continue
            size = self.width_of(widget, width)
            x -= margin + size
            self._place(widget, x, size, height)
            x -= margin + gap

    def css_changed(self) -> None:
        self.relayout()

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.LayoutRequest:
            self.relayout()  # an item changed its natural width (``updateGeometry``) or visibility
        return super().event(event)

    @staticmethod
    def _place(widget: QtWidgets.QWidget, x: float, width: float, height: float) -> None:
        """``align-items: center``: the item's own height (a fixed height, else its size hint), centred."""
        hint = widget.sizeHint().height()
        box_height: float
        if widget.minimumHeight() == widget.maximumHeight():
            box_height = widget.minimumHeight()
        else:
            box_height = hint if hint > 0 else widget.height()
        box_height = min(box_height, height) if height > 0 else box_height
        place(widget, x, (height - box_height) / 2, width, box_height)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()


class Rule(Themed):
    """A 1 px line in ``border.line`` (horizontal by default)."""

    def __init__(
        self,
        themes: ThemeManager,
        vertical: bool = False,
        token: str = "border.line",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.vertical, self.token = vertical, token
        if vertical:
            self.setFixedWidth(1)
        else:
            self.setFixedHeight(1)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        QtGui.QPainter(self).fillRect(self.rect(), self.color(self.token))


# ---------- Canvas and cards ----------


class DotCanvas(Themed):
    """Page canvas: ``surface.canvas`` with a 24 px dot grid, one cached texture brush (DESIGN §1, §5)."""

    PITCH, RADIUS = 24, 1.3  # 1.3 px matches the CSS stops (solid to 1.2 px, clear at 1.4 px)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self._tile: QtGui.QPixmap | None = None

    def restyle(self, *_: object) -> None:
        self._tile = None
        self.update()

    def _texture(self) -> QtGui.QPixmap:
        ratio = self.devicePixelRatioF()
        if self._tile is None or self._tile.devicePixelRatio() != ratio:
            size = round(self.PITCH * ratio)
            tile = QtGui.QPixmap(size, size)
            tile.fill(self.color("surface.canvas"))
            painter = QtGui.QPainter(tile)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self.color("canvas.dot"))
            radius = self.RADIUS * ratio
            painter.drawEllipse(QtCore.QPointF(size / 2, size / 2), radius, radius)
            painter.end()
            tile.setDevicePixelRatio(ratio)
            self._tile = tile
        return self._tile

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        # CSS: background-position calc(50% + 12px): tiles start at the page centre, so a dot column sits in
        # the middle of each 24 px margin at any width.
        painter.setBrushOrigin(
            QtCore.QPointF((self.width() / 2) % self.PITCH, (self.height() / 2) % self.PITCH)
        )
        painter.fillRect(event.rect(), QtGui.QBrush(self._texture()))


def rounded_path(rect: QtCore.QRectF, radius: float) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


SCROLL_HANDLE, SCROLL_MARGIN = 6, 3  # the design draws no scroll bars: a thin handle while hovered


def paint_scroll_handle(
    painter: QtGui.QPainter, widget: Themed, top: float, height: float, offset: float, total: float
) -> None:
    """A ``border.strong`` handle at the widget's right edge for ``total`` px of content scrolled by
    ``offset`` in a ``height`` px window starting at ``top``; drawn only while the pointer is over the widget
    and the content overflows."""
    maximum = total - height
    if maximum <= 0 or not widget.underMouse():
        return
    track = height - 2 * SCROLL_MARGIN
    length = max(28.0, track * height / total)
    handle = QtCore.QRectF(
        widget.width() - SCROLL_HANDLE - SCROLL_MARGIN,
        top + SCROLL_MARGIN + (track - length) * offset / maximum,
        SCROLL_HANDLE,
        length,
    )
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(widget.color("border.strong"))
    painter.drawPath(rounded_path(handle, SCROLL_HANDLE / 2))
    painter.restore()


def wheel_step(event: QtGui.QWheelEvent, row: float) -> QtCore.QPointF:
    """Pixels a wheel event scrolls (x, y): pixel deltas as given, one ``row`` per notch otherwise."""
    pixels = event.pixelDelta()
    if not pixels.isNull():
        return QtCore.QPointF(pixels)
    angle = event.angleDelta()
    return QtCore.QPointF(angle.x() / 120 * row, angle.y() / 120 * row)


class Card(Themed, CssBox):
    """``.card``: ``surface.card``, 1 px ``border.line``, radius 8; a selected card gets a 2 px ink edge. A card
    placed at a fractional box paints on its snapped box and lays its children out in the CSS box."""

    def __init__(
        self,
        themes: ThemeManager,
        parent: QtWidgets.QWidget | None = None,
        *,
        radius: float = 8,
        selected: bool = False,
    ) -> None:
        super().__init__(themes, parent)
        self.radius, self._selected = radius, selected

    def box_width(self) -> float:
        return css_rect(self).width()

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        outer = QtCore.QRectF(self.rect())
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(rounded_path(outer, self.radius))
        if self._selected:
            # 1 px ink border plus a 1 px ink inset ring
            painter.setPen(QtGui.QPen(self.color("text.ink"), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(rounded_path(outer.adjusted(1, 1, -1, -1), self.radius - 1))
        else:
            painter.setPen(QtGui.QPen(self.color("border.line"), 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(rounded_path(outer.adjusted(0.5, 0.5, -0.5, -0.5), self.radius - 0.5))


class CardHead(Themed):
    """``.card-head.ruled``: 52 px, padding 0 16 0 20, title 14/20 600 ink, bottom rule; ``add`` puts
    actions or an aside at the right. Items centre on the 51 px above the rule, at fractional positions."""

    def __init__(
        self, themes: ThemeManager, title: str = "", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.setFixedHeight(52)
        self.row = FlexRow(self, gap=8, left=20, right=16)
        self.title = TextLine(themes, title, CARD_TITLE, "text.ink")
        self.row.add(self.title)
        self.row.add_stretch()

    def add(self, widget: QtWidgets.QWidget, *, gap: float | None = None) -> QtWidgets.QWidget:
        return self.row.add(widget, gap=gap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.row.setGeometry(0, 0, self.width(), self.height() - 1)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(
            QtCore.QRect(0, self.height() - 1, self.width(), 1), self.color("border.line")
        )


# ---------- Shadows ----------


def _gaussian(sigma: float) -> np.ndarray[Any, np.dtype[np.float64]]:
    radius = max(1, math.ceil(3 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xs**2) / (2 * sigma * sigma))
    return np.asarray(kernel / kernel.sum(), dtype=np.float64)


@lru_cache(maxsize=64)
def shadow_image(
    width: int, height: int, radius: float, layers: tuple[Shadow, ...]
) -> QtGui.QImage:
    """CSS ``box-shadow`` layers around a ``width``×``height`` rounded box, as an image with margin
    ``shadow_margin(layers)`` on every side (Gaussian blur with sigma = blur / 2, like browsers)."""
    margin = shadow_margin(layers)
    size_w, size_h = width + 2 * margin, height + 2 * margin
    result = np.zeros((size_h, size_w, 4), dtype=np.float64)
    for dx, dy, blur, spread, color in layers:
        mask = QtGui.QImage(size_w, size_h, QtGui.QImage.Format.Format_Grayscale8)
        mask.fill(0)
        painter = QtGui.QPainter(mask)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255))
        box = QtCore.QRectF(
            margin + dx - spread, margin + dy - spread, width + 2 * spread, height + 2 * spread
        )
        painter.drawPath(rounded_path(box, radius + spread))
        painter.end()
        buffer = np.frombuffer(mask.constBits(), dtype=np.uint8).reshape(
            size_h, mask.bytesPerLine()
        )[:, :size_w]
        alpha = buffer.astype(np.float64) / 255.0
        if blur > 0:
            kernel = _gaussian(blur / 2)
            alpha = np.apply_along_axis(lambda row: np.convolve(row, kernel, "same"), 1, alpha)
            alpha = np.apply_along_axis(lambda col: np.convolve(col, kernel, "same"), 0, alpha)
        shade = css_color(color)
        layer_alpha = alpha * shade.alphaF()
        # Composite "over" in premultiplied space, later layers under earlier ones (CSS order).
        for channel, value in enumerate((shade.red(), shade.green(), shade.blue())):
            result[:, :, channel] += value * layer_alpha * (1 - result[:, :, 3])
        result[:, :, 3] += layer_alpha * (1 - result[:, :, 3])
    premultiplied = np.empty((size_h, size_w, 4), dtype=np.uint8)
    premultiplied[:, :, 0] = np.clip(np.round(result[:, :, 2]), 0, 255)  # ARGB32 is BGRA in memory
    premultiplied[:, :, 1] = np.clip(np.round(result[:, :, 1]), 0, 255)
    premultiplied[:, :, 2] = np.clip(np.round(result[:, :, 0]), 0, 255)
    premultiplied[:, :, 3] = np.clip(np.round(result[:, :, 3] * 255), 0, 255)
    image = QtGui.QImage(
        premultiplied.tobytes(),
        size_w,
        size_h,
        size_w * 4,
        QtGui.QImage.Format.Format_ARGB32_Premultiplied,
    )
    return image.copy()


def shadow_margin(layers: Sequence[Shadow]) -> int:
    return (
        max(
            (
                math.ceil(3 * blur / 2) + abs(spread) + max(abs(dx), abs(dy))
                for dx, dy, blur, spread, _ in layers
            ),
            default=0,
        )
        + 1
    )


def paint_shadow(
    painter: QtGui.QPainter, rect: QtCore.QRect, radius: float, layers: tuple[Shadow, ...]
) -> None:
    """The shadow of a rounded box at ``rect``. Large boxes are drawn in nine slices of one cached image
    of a box just large enough that its middle row and column lie beyond every corner's blur."""
    margin = shadow_margin(layers)
    side = 2 * (margin + math.ceil(radius)) + 1
    if rect.width() <= side or rect.height() <= side:
        painter.drawImage(
            QtCore.QPoint(rect.x() - margin, rect.y() - margin),
            shadow_image(rect.width(), rect.height(), radius, layers),
        )
        return
    tile = shadow_image(side, side, radius, layers)
    full = side + 2 * margin
    head = full // 2  # columns (rows) copied from the start; the middle one is stretched
    tail = full - head - 1
    columns = (
        (0, head, rect.x() - margin, head),
        (head, 1, rect.x() - margin + head, rect.width() + 2 * margin - head - tail),
        (head + 1, tail, rect.right() + 1 + margin - tail, tail),
    )
    rows = (
        (0, head, rect.y() - margin, head),
        (head, 1, rect.y() - margin + head, rect.height() + 2 * margin - head - tail),
        (head + 1, tail, rect.bottom() + 1 + margin - tail, tail),
    )
    for source_y, source_h, target_y, target_h in rows:
        for source_x, source_w, target_x, target_w in columns:
            painter.drawImage(
                QtCore.QRect(target_x, target_y, target_w, target_h),
                tile,
                QtCore.QRect(source_x, source_y, source_w, source_h),
            )


def paint_popup_shadow(
    painter: QtGui.QPainter, rect: QtCore.QRect, radius: float, theme: Theme
) -> None:
    paint_shadow(painter, rect, radius, gui_tokens.POPUP_SHADOW[theme.name])


# ---------- Buttons ----------


VARIANTS: tuple[Variant, ...] = ("secondary", "primary", "ghost", "danger", "danger-solid")


class Button(QtWidgets.QAbstractButton, CssBox):
    """``.btn``: 32 px (small 28), radius 6, 16 px icon (14 in small text buttons), 6 px gap; the icon
    leads the text, or follows it with ``trailing`` ("Continue →", "Profiles ⌄").

    Variants: secondary, primary, ghost, danger (outline) and danger-solid (dialogs only). Focus is drawn by
    the window's FocusRing.
    """

    def __init__(
        self,
        themes: ThemeManager,
        text: str = "",
        *,
        icon: str | None = None,
        variant: Variant = "secondary",
        small: bool = False,
        icon_size: int | None = None,
        trailing: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if variant not in VARIANTS:
            raise ValueError(f"Unknown button variant: {variant}")
        self.themes, self.variant, self.small, self.glyph = themes, variant, small, icon
        self.trailing = trailing
        self.opened = False
        self.icon_px = icon_size or (14 if small and text else 16)
        self.setText(text)
        self.setProperty("focusRadius", 6)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        themes.changed.connect(self.update)
        if text:
            self.setAccessibleName(text)

    @property
    def text_style(self) -> TextStyle:
        return BUTTON_SMALL if self.small else BUTTON

    def set_variant(self, variant: Variant) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown button variant: {variant}")
        if variant != self.variant:
            self.variant = variant
            self.update()

    def set_opened(self, opened: bool) -> None:
        """The look of a button whose dialog is open (``.btn.pressed`` under 12's dialog): the
        ``surface.selected`` fill with the variant's border and text."""
        if opened != self.opened:
            self.opened = opened
            self.update()

    def set_icon(self, name: str | None) -> None:
        if name != self.glyph:
            self.glyph = name
            self.updateGeometry()
            self.update()

    def setText(self, text: str | None) -> None:
        super().setText(text or "")
        self.updateGeometry()

    def natural_width(self) -> float:
        """The CSS width; the size hint rounds it up."""
        height = 28 if self.small else 32
        if not self.text():
            return float(height)
        padding = 10 if self.small else 12
        width = 2 * padding + 2 + self.text_style.advance(self.text())
        if self.glyph:
            width += self.icon_px + 6
        return width

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 28 if self.small else 32)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not press_on_enter(self, event):
            super().keyPressEvent(event)

    def colors(self) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
        """Fill, border and foreground for the current state."""
        theme = self.themes.current
        enabled = self.isEnabled()
        pressed = self.isDown()
        hovered = self.underMouse() and enabled
        transparent = QtGui.QColor(0, 0, 0, 0)
        if not enabled:
            if self.variant == "primary":
                disabled = theme.color("accent.disabled")
                return disabled, disabled, theme.color("accent.disabledText")
            if self.variant == "ghost":
                return transparent, transparent, theme.color("text.disabled")
            return (
                theme.color("surface.sunken"),
                theme.color("border.line"),
                theme.color("text.disabled"),
            )
        if self.variant == "primary":
            fill = theme.color(
                "accent.pressed" if pressed else "accent.hover" if hovered else "accent.primary"
            )
            return fill, fill, css_color(gui_tokens.ON_ACCENT)
        if self.variant == "danger-solid":
            fill = theme.color("state.badSolid")
            if pressed or hovered:
                # Not drawn in the mockups: darker by 18 % (pressed) or 10 % (hover) black, as HANDOFF §7.3.
                fill = _over(fill, QtGui.QColor(0, 0, 0, round(255 * (0.18 if pressed else 0.10))))
            return fill, fill, css_color(gui_tokens.ON_ACCENT)
        if self.variant == "ghost":
            fill = (
                theme.color("surface.selected")
                if pressed
                else theme.color("surface.hover")
                if hovered
                else transparent
            )
            return fill, transparent, theme.color("text.body")
        if self.variant == "danger":
            if self.opened:
                return (
                    theme.color("surface.selected"),
                    theme.color("state.badLine"),
                    theme.color("state.bad"),
                )
            if pressed:  # 00-system draws the pressed danger button solid
                fill = theme.color("state.badSolid")
                return fill, fill, css_color(gui_tokens.ON_ACCENT)
            fill = theme.color("state.badSoft") if hovered else theme.color("surface.card")
            return fill, theme.color("state.badLine"), theme.color("state.bad")
        fill = theme.color(
            "surface.selected"
            if pressed or self.opened
            else "surface.hover"
            if hovered
            else "surface.card"
        )
        return fill, theme.color("border.strong"), theme.color("text.ink")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        fill, border, foreground = self.colors()
        box = QtCore.QRectF(self.rect())
        if border.alpha():
            painter.setPen(QtGui.QPen(border, 1))
            painter.setBrush(fill)
            painter.drawPath(rounded_path(box.adjusted(0.5, 0.5, -0.5, -0.5), 5.5))
        else:  # ghost: the fill covers the transparent border too (CSS background-clip: border-box)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawPath(rounded_path(box, 6))
        text = self.text()
        style = self.text_style
        content = (self.icon_px + (6 if text else 0) if self.glyph else 0) + (
            style.advance(text) if text else 0
        )
        # Centred in the CSS box (the text keeps its fraction, icons snap); in a layout, in the widget.
        span = float(self.width()) if self.css_width is None else self.css_width
        x = self.shift + (span - content) / 2
        icon_top = snap(self.vshift + (self.height() - self.icon_px) / 2)
        if self.glyph and not self.trailing:
            gui_icons.draw(
                painter, self.glyph, foreground, QtCore.QPointF(snap(x), icon_top), self.icon_px
            )
            x += self.icon_px + 6
        if text:
            top = self.vshift + (self.height() - style.line) / 2
            x += draw_text(painter, x, top, text, style, foreground) + 6
        if self.glyph and self.trailing:
            gui_icons.draw(
                painter, self.glyph, foreground, QtCore.QPointF(snap(x), icon_top), self.icon_px
            )


def _over(base: QtGui.QColor, overlay: QtGui.QColor) -> QtGui.QColor:
    alpha = overlay.alphaF()
    return QtGui.QColor(
        round(overlay.red() * alpha + base.red() * (1 - alpha)),
        round(overlay.green() * alpha + base.green() * (1 - alpha)),
        round(overlay.blue() * alpha + base.blue() * (1 - alpha)),
        base.alpha(),
    )


class LinkButton(QtWidgets.QAbstractButton, CssBox):
    """``.link``: ink 13 px 500, ``border.strong`` underline 3 px below the baseline, optional arrow."""

    STYLE = BODY_MEDIUM
    OVERHANG = True

    def __init__(
        self,
        themes: ThemeManager,
        text: str,
        *,
        icon: str | None = "aright",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes, self.glyph = themes, icon
        self.setText(text)
        self.setAccessibleName(text)
        self.setProperty("focusRadius", 2)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        themes.changed.connect(self.update)

    def natural_width(self) -> float:
        return self.STYLE.advance(self.text()) + (4 + 14 if self.glyph else 0)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 18)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not press_on_enter(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.themes.current
        top = (self.height() - self.STYLE.line) / 2
        x = self.shift
        width = draw_text(painter, x, top, self.text(), self.STYLE, theme.color("text.ink"))
        baseline = self.STYLE.baseline(top)
        painter.fillRect(QtCore.QRectF(x, baseline + 3, width, 1), theme.color("border.strong"))
        if self.glyph:
            gui_icons.draw(
                painter,
                self.glyph,
                theme.color("text.secondary"),
                QtCore.QPointF(snap(x + width + 4), (self.height() - 14) / 2),
                14,
            )


# ---------- Segmented control ----------


class Segment(QtWidgets.QAbstractButton):
    def __init__(self, control: "SegmentedControl", text: str, icon: str | None) -> None:
        super().__init__(control)
        self.control, self.glyph = control, icon
        self.shift = 0.0
        self.setText(text)
        self.setCheckable(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 4)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not press_on_enter(self, event):
            super().keyPressEvent(event)

    def natural_width(self) -> float:
        if self.control.segment_width:
            return float(self.control.segment_width)
        width = 24 + self.control.text_style.advance(self.text())
        if self.glyph:
            width += self.control.icon_size + (6 if self.text() else 0)
        return width

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(math.ceil(self.natural_width()), 26)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.control.themes.current
        box = QtCore.QRectF(self.rect())
        if self.isChecked():
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("surface.card"))
            painter.drawPath(rounded_path(box, 4))
            foreground = theme.color("text.ink")
        else:
            foreground = theme.color(
                self.control.hover_token if self.underMouse() else self.control.idle_token
            )
        text, style, size = self.text(), self.control.text_style, self.control.icon_size
        content = (size + (6 if text else 0) if self.glyph else 0) + (
            style.advance(text) if text else 0
        )
        # Content centred in the unsnapped box; ``shift`` is that box's offset from this widget's left.
        x = self.shift + (self.natural_width() - content) / 2
        if self.glyph:
            gui_icons.draw(
                painter,
                self.glyph,
                foreground,
                QtCore.QPointF(x, (self.height() - size) / 2),
                size,
            )
            x += size + 6
        if text:
            draw_text(painter, x, (self.height() - style.line) / 2, text, style, foreground)


class SegmentedControl(Themed, CssBox):
    """``.seg``: sunken track, 1 px ``border.line``, radius 6, padding 2, gap 2; raised active segment.
    Icons lead the text (16 px; ``icon_size`` 14 for ``.i.sm``).

    Exclusive; arrow keys move between segments. ``changed`` carries the new index.
    """

    changed = QtCore.Signal(int)

    def __init__(
        self,
        themes: ThemeManager,
        segments: Sequence[tuple[str, str | None]],
        *,
        segment_width: int = 0,
        icon_size: int = 16,
        idle_token: str = "text.secondary",
        hover_token: str = "text.body",
        tooltips: Sequence[str] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.segment_width, self.icon_size = segment_width, icon_size
        self.idle_token, self.hover_token = idle_token, hover_token
        # ``.seg span``: the body's 18 px line, centred in the 26 px segment.
        self.text_style = TextStyle(13, 500, 18)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.segments: list[Segment] = []
        for index, (text, icon) in enumerate(segments):
            segment = Segment(self, text, icon)
            if index < len(tooltips):
                segment.setToolTip(tooltips[index])
                segment.setAccessibleName(tooltips[index])
            elif text:
                segment.setAccessibleName(text)
            self.group.addButton(segment, index)
            self.segments.append(segment)
        self.group.idToggled.connect(self._toggled)
        if self.segments:
            self.segments[0].setChecked(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.relayout()

    def natural_width(self) -> float:
        widths = [segment.natural_width() for segment in self.segments]
        return 2 + 4 + sum(widths) + 2 * max(0, len(widths) - 1)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            math.ceil(self.shift + self.natural_width()) - math.floor(self.shift), 32
        )

    def css_changed(self) -> None:
        self.relayout()

    def set_origin_x(self, x: float) -> None:
        """Place the control at a fractional x of its parent (segment edges then snap like a browser's)."""
        self.shift = x - math.floor(x)
        self.move(math.floor(x), self.y())
        self.relayout()

    def relayout(self) -> None:
        """Segments at their CSS boxes (1 px border, 2 px padding, 2 px gaps), edges snapped half up."""
        x = self.shift + 3
        for segment in self.segments:
            width = segment.natural_width()
            left, right = math.floor(x + 0.5), math.floor(x + width + 0.5)
            segment.shift = x - left
            segment.setGeometry(left, 3, right - left, 26)
            x += width + 2
        self.updateGeometry()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def _toggled(self, index: int, checked: bool) -> None:
        self.update()
        if checked:
            self.changed.emit(index)

    def index(self) -> int:
        return self.group.checkedId()

    def set_index(self, index: int) -> None:
        if 0 <= index < len(self.segments) and index != self.index():
            self.segments[index].setChecked(True)

    def clear(self) -> None:
        """No segment on (repetition presets when the value is none of them)."""
        if self.index() < 0:
            return
        self.group.setExclusive(False)
        for segment in self.segments:
            segment.setChecked(False)
        self.group.setExclusive(True)
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        step = {QtCore.Qt.Key.Key_Left: -1, QtCore.Qt.Key.Key_Right: 1}.get(
            QtCore.Qt.Key(event.key()), 0
        )
        if step and self.segments:
            target = (max(0, self.index()) + step) % len(self.segments)
            self.segments[target].setChecked(True)
            self.segments[target].setFocus(QtCore.Qt.FocusReason.TabFocusReason)
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        box = QtCore.QRectF(self.rect())
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.sunken"))
        painter.drawPath(rounded_path(box.adjusted(0.5, 0.5, -0.5, -0.5), 5.5))
        for segment in self.segments:
            if segment.isChecked():
                # Raised segment: 0 0 0 1px border.strong, 0 1px 2px overlay.shadowSmall (outside it).
                rect = QtCore.QRectF(segment.geometry())
                layers: tuple[Shadow, ...] = (
                    (0, 1, 2, 0, self.theme.tokens["overlay.shadowSmall"]),
                )
                margin = shadow_margin(layers)
                painter.drawImage(
                    QtCore.QPointF(rect.x() - margin, rect.y() - margin),
                    shadow_image(int(rect.width()), int(rect.height()), 4, layers),
                )
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(self.color("border.strong"))
                painter.drawPath(rounded_path(rect.adjusted(-1, -1, 1, 1), 5))


# ---------- Focus ring ----------


KEYBOARD_REASONS = (
    QtCore.Qt.FocusReason.TabFocusReason,
    QtCore.Qt.FocusReason.BacktabFocusReason,
    QtCore.Qt.FocusReason.ShortcutFocusReason,
)


class FocusRing(Themed):
    """Keyboard focus for the whole window (HANDOFF §7.13), drawn on top of every page and overlay.

    Buttons, segments, tabs, check boxes and focusable cards get a 2 px ink ring outside a 2 px card-coloured
    gap; inputs and selects get a 1 px ink ring outside their border. Mouse focus shows nothing, like CSS
    ``:focus-visible``. Widgets choose with the dynamic properties ``focusShape`` (``ring``, ``input``,
    ``none``) and ``focusRadius``; a widget painted at a fractional position reports its drawn box with
    ``painted_rect()``.
    """

    WATCHED = (
        QtCore.QEvent.Type.Move,
        QtCore.QEvent.Type.Resize,
        QtCore.QEvent.Type.Hide,
        QtCore.QEvent.Type.Show,
    )
    # Keyboard mode (CSS :focus-visible), application-wide; painters of focus states (table rows) read it.
    _keyboard = False

    @classmethod
    def keyboard_visible(cls) -> bool:
        return cls._keyboard

    @property
    def keyboard(self) -> bool:
        return FocusRing._keyboard

    def __init__(self, themes: ThemeManager, window: QtWidgets.QWidget) -> None:
        super().__init__(themes, window)
        self.window_widget = window
        self.target: QtWidgets.QWidget | None = None
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.focusChanged.connect(self._focus_changed)
            application.installEventFilter(self)
        window.installEventFilter(self)
        self.setGeometry(window.rect())
        self.raise_()

    @staticmethod
    def shape(widget: QtWidgets.QWidget) -> str:
        declared = widget.property("focusShape")
        if isinstance(declared, str):
            return declared
        if isinstance(
            widget,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QComboBox,
                QtWidgets.QAbstractSpinBox,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QTextEdit,
            ),
        ):
            return "input"
        if isinstance(widget, QtWidgets.QAbstractButton):
            return "ring"
        return "none"

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if kind == QtCore.QEvent.Type.KeyPress:
            FocusRing._keyboard = True
        elif kind == QtCore.QEvent.Type.MouseButtonPress:
            if FocusRing._keyboard:
                FocusRing._keyboard = False
                self.update()
        elif kind == QtCore.QEvent.Type.FocusIn and isinstance(event, QtGui.QFocusEvent):
            if event.reason() in KEYBOARD_REASONS:
                FocusRing._keyboard = True
        if watched is self.window_widget and kind == QtCore.QEvent.Type.Resize:
            self.setGeometry(self.window_widget.rect())
        elif (
            kind in self.WATCHED
            and self.target is not None
            and isinstance(watched, QtWidgets.QWidget)
            and (watched is self.target or watched.isAncestorOf(self.target))
        ):
            self.update()
        return False

    def _focus_changed(self, old: QtWidgets.QWidget | None, new: QtWidgets.QWidget | None) -> None:
        self.target = new if new is not None and self.window_widget.isAncestorOf(new) else None
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        target = self.target
        if target is None or not target.isVisible():
            return
        shape = self.shape(target)
        # Text fields show their ring on any focus, as browsers match :focus-visible on them.
        if shape == "none" or (shape != "input" and not self.keyboard):
            return
        painted = getattr(target, "painted_rect", None)
        rect = painted() if callable(painted) else QtCore.QRectF(target.rect())
        rect.translate(QtCore.QPointF(target.mapTo(self.window_widget, QtCore.QPoint(0, 0))))
        radius_value = target.property("focusRadius")
        radius = float(radius_value) if isinstance(radius_value, (int, float)) else 6.0
        paint_focus_ring(QtGui.QPainter(self), rect, radius, shape, self.theme)


def paint_focus_ring(
    painter: QtGui.QPainter, rect: QtCore.QRectF, radius: float, shape: str, theme: Theme
) -> None:
    """Focus outside ``rect``: ``input`` is a 1 px ink ring (CSS ``0 0 0 1px ink``); ``ring`` is a 2 px
    card-coloured gap and a 2 px ink ring (``0 0 0 2px card, 0 0 0 4px ink``)."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    if shape == "input":
        painter.setPen(QtGui.QPen(theme.color("text.ink"), 1))
        painter.drawPath(rounded_path(rect.adjusted(-0.5, -0.5, 0.5, 0.5), radius + 0.5))
    else:
        painter.setPen(QtGui.QPen(theme.color("surface.card"), 2))
        painter.drawPath(rounded_path(rect.adjusted(-1, -1, 1, 1), radius + 1))
        painter.setPen(QtGui.QPen(theme.color("text.ink"), 2))
        painter.drawPath(rounded_path(rect.adjusted(-3, -3, 3, 3), radius + 3))
    painter.restore()


# ---------- Dialogs ----------


class ModalCard(Themed):
    """What ``ModalHost`` shows: a ``.dialog`` card (card fill, 1 px ``border.line``, radius 10) that ends
    with ``finish`` and keeps the chosen ``result``; ``default_button`` answers Enter."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.result = False  # the chosen button's result (True: the action, False: Cancel or Esc)
        self.default_button: QtWidgets.QAbstractButton | None = None

    def finish(self, result: bool) -> None:
        self.result = result
        host = self.parent()
        if isinstance(host, ModalHost):
            host.close_card(self)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        box = QtCore.QRectF(self.rect())
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(rounded_path(box.adjusted(0.5, 0.5, -0.5, -0.5), 9.5))


class DialogCard(ModalCard):
    """``.dialog``: 560 px, radius 10, body padding 24 24 20, title 16/22 600, ruled footer 14 24."""

    def __init__(
        self,
        themes: ThemeManager,
        title: str,
        *,
        width: int = 560,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.setFixedWidth(width)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)  # the 1 px border
        outer.setSpacing(0)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setContentsMargins(24, 24, 24, 20)
        self.body.setSpacing(0)
        self.title = TextLine(themes, title, DIALOG_TITLE, "text.ink")
        self.body.addWidget(self.title)
        outer.addLayout(self.body)
        self.footer = QtWidgets.QWidget()
        self.footer.setFixedHeight(61)  # the 1 px rule, then 14 + 32 + 14
        footer = QtWidgets.QVBoxLayout(self.footer)
        footer.setContentsMargins(0, 15, 0, 14)
        # Right-aligned at the browser's fractional positions (8 px gaps, 24 px padding).
        self.buttons = FlexRow(gap=8, left=24, right=24)
        self.buttons.setFixedHeight(32)
        self.buttons.add_stretch()
        footer.addWidget(self.buttons)
        outer.addWidget(self.footer)

    def add_button(
        self,
        text: str,
        result: bool,
        *,
        variant: Variant = "secondary",
        icon: str | None = None,
        default: bool = False,
    ) -> Button:
        control = Button(self.themes, text, icon=icon, variant=variant)
        control.clicked.connect(lambda: self.finish(result))
        self.buttons.add(control)
        if default:
            self.default_button = control
        return control

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        QtGui.QPainter(self).fillRect(
            QtCore.QRectF(1, self.footer.y(), self.width() - 2, 1), self.color("border.line")
        )


@dataclass(slots=True)
class _Modal:
    card: ModalCard
    escape_result: bool
    opener: QtWidgets.QWidget | None
    loop: QtCore.QEventLoop | None = None


class ModalHost(Themed):
    """Window-wide overlay: the scrim over everything (top bar included) and a centred card.

    ``exec_card`` runs a local event loop like ``QMessageBox.exec`` so call sites keep their flow. A card opened
    while another is shown (an error during a confirmation) covers it until it closes. Esc chooses the card's
    ``escape_result``; Enter or Return press the focused button, else the card's ``default_button``; Tab stays
    inside the card; focus returns to the opener on close.
    """

    closed = QtCore.Signal()

    def __init__(self, themes: ThemeManager, window: QtWidgets.QWidget) -> None:
        super().__init__(themes, window)
        self.window_widget = window
        self.stack: list[_Modal] = []
        self.hide()
        window.installEventFilter(self)

    @property
    def card(self) -> ModalCard | None:
        return self.stack[-1].card if self.stack else None

    @property
    def escape_result(self) -> bool:
        return self.stack[-1].escape_result if self.stack else False

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.window_widget and event.type() == QtCore.QEvent.Type.Resize:
            self.setGeometry(self.window_widget.rect())
            self._place()
        return False

    def _place(self) -> None:
        for entry in self.stack:
            entry.card.adjustSize()
            size = entry.card.size()
            entry.card.move(
                (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
            )

    def open_card(
        self,
        card: ModalCard,
        *,
        escape_result: bool = False,
        position: QtCore.QPoint | None = None,
    ) -> None:
        if self.stack:
            self.stack[-1].card.hide()
        self.stack.append(_Modal(card, escape_result, QtWidgets.QApplication.focusWidget()))
        card.setParent(self)
        self.setGeometry(self.window_widget.rect())
        if position is None:
            self._place()
        else:
            card.adjustSize()
            card.move(position)
        card.show()
        self.show()
        self.raise_()
        for ring in self.window_widget.findChildren(FocusRing):
            ring.raise_()
        default = card.default_button
        if isinstance(default, QtWidgets.QWidget):
            # The dialog opens with its default button focused and ringed (Cancel in destructive
            # dialogs, as 02 and 12 draw it), so Enter and Space say what they will do.
            FocusRing._keyboard = True
            default.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
        else:
            card.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def exec_card(self, card: ModalCard, *, escape_result: bool = False) -> bool:
        """Show the card and wait until it closes; returns its ``result``."""
        self.open_card(card, escape_result=escape_result)
        loop = QtCore.QEventLoop(self)
        self.stack[-1].loop = loop
        loop.exec()
        loop.deleteLater()
        return card.result

    def close_card(self, card: ModalCard) -> None:
        entry = next((item for item in self.stack if item.card is card), None)
        if entry is None:
            return
        self.stack.remove(entry)
        card.hide()
        card.deleteLater()
        if self.stack:
            self.stack[-1].card.show()
            self._place()
            self.update()
        else:
            self.hide()
        if alive(entry.opener) and entry.opener is not None and entry.opener.isVisible():
            entry.opener.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        if entry.loop is not None:
            entry.loop.quit()
        self.closed.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        card = self.card
        key = event.key()
        if card is not None and key == QtCore.Qt.Key.Key_Escape:
            card.finish(self.escape_result)
            return
        if card is not None and key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            focused = QtWidgets.QApplication.focusWidget()
            default = card.default_button
            target = (
                focused
                if isinstance(focused, QtWidgets.QAbstractButton) and card.isAncestorOf(focused)
                else default
            )
            if isinstance(target, QtWidgets.QAbstractButton) and target.isEnabled():
                target.click()
            return
        super().keyPressEvent(event)

    def focusNextPrevChild(self, next: bool) -> bool:
        # Tab cycles inside the card.
        card = self.card
        if card is None:
            return super().focusNextPrevChild(next)
        chain = [
            widget
            for widget in card.findChildren(QtWidgets.QWidget)
            if widget.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus
            and widget.isVisible()
            and widget.isEnabled()
        ]
        if not chain:
            return True
        current = QtWidgets.QApplication.focusWidget()
        index = chain.index(current) if current in chain else -1
        target = chain[(index + (1 if next else -1)) % len(chain)]
        target.setFocus(
            QtCore.Qt.FocusReason.TabFocusReason
            if next
            else QtCore.Qt.FocusReason.BacktabFocusReason
        )
        return True

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()  # the scrim swallows clicks; Esc and the buttons close the dialog

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.color("overlay.scrim"))
        card = self.card
        if card is not None and card.isVisible():
            paint_popup_shadow(painter, card.geometry(), 10, self.theme)


# ---------- Popups anchored to the page ----------


class PopupHost(Themed):
    """Transparent window-wide layer for in-window popups (wizard popups, menus drawn by the app).

    It paints the popup shadow, closes the popup on Esc, a click outside it or a window resize (popups are
    placed against the page as it was) and returns focus to the widget that opened it.
    """

    closed = QtCore.Signal()

    def __init__(self, themes: ThemeManager, window: QtWidgets.QWidget) -> None:
        super().__init__(themes, window)
        self.window_widget = window
        self.popup: QtWidgets.QWidget | None = None
        self.radius = 10.0
        self.delete_on_close = False
        self.opener: QtWidgets.QWidget | None = None
        self.on_close: Callable[[], None] | None = None
        self.hide()
        window.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.window_widget and event.type() == QtCore.QEvent.Type.Resize:
            self.setGeometry(self.window_widget.rect())
            if self.popup is not None:
                self.close_popup()
        return False

    def open_popup(
        self,
        popup: QtWidgets.QWidget,
        rect: QtCore.QRect,
        *,
        opener: QtWidgets.QWidget | None = None,
        on_close: Callable[[], None] | None = None,
        radius: float = 10,
        delete_on_close: bool = False,
    ) -> None:
        """Show ``popup`` at ``rect`` (window coordinates); ``radius`` is its corner radius (menus 8). The
        caller owns the popup unless ``delete_on_close`` (popups built for one opening, such as menus)."""
        if self.popup is not None:
            self.close_popup()
        self.opener = opener or QtWidgets.QApplication.focusWidget()
        self.on_close = on_close
        self.popup, self.radius, self.delete_on_close = popup, radius, delete_on_close
        popup.setParent(self)
        popup.setGeometry(rect)
        self.setGeometry(self.window_widget.rect())
        popup.show()
        self.show()
        self.raise_()
        for ring in self.window_widget.findChildren(FocusRing):
            ring.raise_()
        focused = QtWidgets.QApplication.focusWidget()
        if focused is None or not popup.isAncestorOf(focused):
            focusable = [
                child
                for child in popup.findChildren(QtWidgets.QWidget)
                if child.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus and child.isVisible()
            ]
            target = focusable[0] if focusable else popup
            if target is popup:
                popup.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            target.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def close_popup(self) -> None:
        popup, self.popup = self.popup, None
        if popup is None:
            return
        popup.hide()
        if self.delete_on_close:
            popup.deleteLater()
        self.hide()
        callback, self.on_close = self.on_close, None
        if alive(self.opener) and self.opener is not None and self.opener.isVisible():
            self.opener.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self.opener = None
        if callback is not None:
            callback()
        self.closed.emit()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.popup is not None and not self.popup.geometry().contains(
            event.position().toPoint()
        ):
            self.close_popup()
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close_popup()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self.popup is not None and self.popup.isVisible():
            paint_popup_shadow(QtGui.QPainter(self), self.popup.geometry(), self.radius, self.theme)


class PopupCard(Themed):
    """``.popup``: ``surface.card``, 1 px ``border.line``, radius 10 (the host paints the shadow)."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 9.5)
        )


# ---------- Toasts ----------


TOAST_ICONS = {
    "ok": ("ccheck", "state.ok"),
    "warn": ("alert", "state.warn"),
    "bad": ("cx", "state.bad"),
    "neutral": ("info", "text.secondary"),
}


class Toast(Themed):
    """One message (HANDOFF §7.14): card fill, radius 8, padding 10 10 10 14, state icon, 13 px body text and a
    dismiss button."""

    dismissed = QtCore.Signal(object)

    def __init__(self, themes: ThemeManager, text: str, kind: ToastKind) -> None:
        super().__init__(themes)
        self.text, self.kind = text, kind
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self.dismissed.emit(self))
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(14 + 16 + 10, 10, 10, 10)
        row.setSpacing(8)
        self.label = QtWidgets.QLabel(text)
        self.label.setWordWrap(True)
        self.label.setFont(font(13))
        self.label.setMaximumWidth(480 - 14 - 16 - 10 - (10 + 28 + 8))
        row.addWidget(self.label, 1)
        close = Button(themes, icon="x", variant="ghost", small=True)
        close.setToolTip("Dismiss")
        close.setAccessibleName("Dismiss message")
        close.clicked.connect(lambda: self.dismissed.emit(self))
        row.addWidget(close, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.setAccessibleName(text)
        self.restyle()

    def restart(self, timeout_ms: int) -> None:
        self.timer.start(timeout_ms)

    def restyle(self, *_: object) -> None:
        self.label.setStyleSheet(f"color: {self.theme.qss('text.body')}; background: transparent;")
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7.5)
        )
        name, token = TOAST_ICONS.get(self.kind, TOAST_ICONS["neutral"])
        gui_icons.draw(painter, name, self.color(token), QtCore.QPointF(14, 11))


class ToastArea(Themed):
    """Top-right stack of toasts, newest first: 24 px from the window's right edge, ``top`` px from its top (under
    the top bar), 8 px apart. Every toast, errors included, fades after ``TIMEOUT_MS`` and has a
    dismiss button; a repeated message restarts the shown one instead of stacking a copy. The area only paints
    the toasts' popup shadows and lets clicks through; the toasts are the window's children, so they take
    clicks."""

    LIMIT = 5
    TIMEOUT_MS = 3000  # maintainer, 20 October 2026

    def __init__(self, themes: ThemeManager, window: QtWidgets.QWidget, *, top: int = 24) -> None:
        super().__init__(themes, window)
        self.window_widget = window
        self.top = top
        self.toasts: list[Toast] = []  # newest first
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        window.installEventFilter(self)
        self.hide()

    def margin(self) -> int:
        return shadow_margin(gui_tokens.POPUP_SHADOW[self.theme.name])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        for toast in self.toasts:
            if toast.isVisible():
                paint_popup_shadow(painter, toast.geometry().translated(-self.pos()), 8, self.theme)

    def restyle(self, *_: object) -> None:
        self._layout()
        self.update()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.window_widget and event.type() == QtCore.QEvent.Type.Resize:
            self._layout()
        return False

    def show_message(self, text: str, kind: ToastKind = "neutral") -> Toast:
        shown = next((t for t in self.toasts if t.text == text and t.kind == kind), None)
        if (
            shown is not None
        ):  # the same message again: keep one, newest first, with a fresh timeout
            self.toasts.remove(shown)
            self.toasts.insert(0, shown)
            shown.restart(self.TIMEOUT_MS)
            self._layout()
            return shown
        toast = Toast(self.themes, text, kind)
        toast.setParent(self.window_widget)
        toast.dismissed.connect(self.dismiss)
        self.toasts.insert(0, toast)
        while len(self.toasts) > self.LIMIT:
            self.dismiss(self.toasts[-1])
        toast.restart(self.TIMEOUT_MS)
        toast.show()
        self._layout()
        return toast

    def dismiss(self, toast: object) -> None:
        if isinstance(toast, Toast) and toast in self.toasts:
            self.toasts.remove(toast)
            toast.timer.stop()
            toast.hide()
            toast.deleteLater()
            self._layout()

    def _layout(self) -> None:
        if not self.toasts:
            self.hide()
            return
        width = 0
        sizes = []
        for toast in self.toasts:
            toast.adjustSize()
            size = toast.sizeHint().expandedTo(toast.minimumSizeHint())
            size.setWidth(min(480, max(size.width(), 240)))
            size.setHeight(
                max(
                    38,
                    toast.heightForWidth(size.width())
                    if toast.hasHeightForWidth()
                    else size.height(),
                )
            )
            sizes.append(size)
            width = max(width, size.width())
        height = sum(size.height() for size in sizes) + 8 * (len(sizes) - 1)
        window = self.window_widget.rect()
        margin = self.margin()
        left = window.right() - 24 - width + 1
        self.setGeometry(left - margin, self.top - margin, width + 2 * margin, height + 2 * margin)
        self.show()
        self.raise_()
        y = self.top
        for toast, size in zip(self.toasts, sizes, strict=True):
            toast.setGeometry(left + width - size.width(), y, size.width(), size.height())
            toast.raise_()
            y += size.height() + 8
        self.update()
        for ring in self.window_widget.findChildren(FocusRing):
            ring.raise_()


# ---------- Adjustable splits between cards ----------


class SplitSetting:
    """One adjustable size between cards (a column width or a card height) that the user dragged, remembered
    in the application's settings (``layout/<key>``); None keeps the design's size."""

    def __init__(self, key: str) -> None:
        self.key = "layout/" + key
        stored = QtCore.QSettings().value(self.key)
        try:
            self.value: float | None = None if stored is None else float(stored)
        except (TypeError, ValueError):
            self.value = None

    def resolve(self, default: float, low: float, high: float) -> float:
        """The size to lay out: the dragged one (else ``default``) kept between ``low`` and ``high``."""
        size = default if self.value is None else self.value
        return max(low, min(size, high)) if high >= low else low

    def drag(self, current: float, delta: float, low: float, high: float) -> None:
        self.value = max(low, min(current + delta, high)) if high >= low else low

    def save(self) -> None:
        settings = QtCore.QSettings()
        if self.value is None:
            settings.remove(self.key)
        else:
            settings.setValue(self.key, round(self.value))

    def reset(self) -> None:
        self.value = None
        self.save()


class Gutter(Themed):
    """The gap between two cards made draggable, as split views in an IDE: a split cursor over it, a 2 px
    line while it is hovered or dragged; dragging emits the pointer's movement along the split (``moved``),
    releasing it ``released``, a double click ``reset`` (back to the design's size). ``across``: the gap runs
    top to bottom between a left and a right card (dragged in x); otherwise it runs left to right."""

    moved = QtCore.Signal(float)
    released = QtCore.Signal()
    reset = QtCore.Signal()

    def __init__(self, themes: ThemeManager, *, across: bool, parent: QtWidgets.QWidget) -> None:
        super().__init__(themes, parent)
        self.across = across
        self._last: float | None = None
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.setCursor(
            QtCore.Qt.CursorShape.SplitHCursor if across else QtCore.Qt.CursorShape.SplitVCursor
        )
        self.setToolTip("Drag to resize · double-click to restore")

    def _position(self, event: QtGui.QMouseEvent) -> float:
        point = event.globalPosition()
        return point.x() if self.across else point.y()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last = self._position(event)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last is not None:
            position = self._position(event)
            if position != self._last:
                self.moved.emit(position - self._last)
                self._last = position

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last is not None:
            self._last = None
            self.update()
            self.released.emit()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.reset.emit()

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self._last is None and not self.underMouse():
            return
        painter = QtGui.QPainter(self)
        color = self.color("text.ink" if self._last is not None else "border.strong")
        if self.across:
            painter.fillRect(QtCore.QRect(self.width() // 2 - 1, 0, 2, self.height()), color)
        else:
            painter.fillRect(QtCore.QRect(0, self.height() // 2 - 1, self.width(), 2), color)


class CardSplit:
    """A ``Gutter`` and its ``SplitSetting`` for one size of a page's layout. The page asks ``resolve`` for the
    size on each layout (with its bounds) and ``place``s the gutter over the gap; a drag changes the size
    (``sign`` -1: dragging right or down makes it smaller, for a card on the right or bottom) and lays the page
    out again."""

    def __init__(
        self,
        themes: ThemeManager,
        parent: QtWidgets.QWidget,
        key: str,
        relayout: Callable[[], None],
        *,
        across: bool,
        sign: int = 1,
    ) -> None:
        self.setting = SplitSetting(key)
        self.gutter = Gutter(themes, across=across, parent=parent)
        self.size, self.low, self.high = 0.0, 0.0, 0.0
        self.relayout, self.sign = relayout, sign
        self.gutter.moved.connect(self._moved)
        self.gutter.released.connect(self.setting.save)
        self.gutter.reset.connect(self._reset)

    def _moved(self, delta: float) -> None:
        self.setting.drag(self.size, self.sign * delta, self.low, self.high)
        self.relayout()

    def _reset(self) -> None:
        self.setting.reset()
        self.relayout()

    def resolve(self, default: float, low: float, high: float) -> float:
        self.low, self.high = low, high
        self.size = self.setting.resolve(default, low, high)
        return self.size

    def place(self, x: float, y: float, width: float, height: float) -> None:
        self.gutter.setGeometry(round(x), round(y), max(1, round(width)), max(1, round(height)))
        self.gutter.raise_()
