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

"""Data marks of the Main design (HANDOFF §7.9, §7.10, §12.4): progress ring, outcome bar, meter, legend
dots, key-value lists and the coverage chart. They draw values they are given and never invent data:
unavailable values are "—" and a chart without samples stays empty."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .gui_menus import show_tooltip
from .gui_style import duration
from .gui_theme import Theme, ThemeManager, snap
from .gui_widgets import CssBox, TextStyle, Themed, draw_text, paint_glyphs, rounded_path


def _hatch_pixels(
    width: int,
    height: int,
    rgba_a: int,
    rgba_b: int,
    length_a: float,
    length_b: float,
    ratio: float,
    offset: int,
) -> QtGui.QImage:
    rows, columns = np.mgrid[0:height, 0:width]
    # A 135° gradient line starts at the box's top-left corner; browsers sample it at pixel centres.
    # ``offset``: the image's distance from that corner in device pixels, right plus down.
    distance = (rows + columns + offset + 1) / math.sqrt(2)
    band = np.mod(distance, length_a + length_b) >= length_a
    pixels = np.where(band, np.uint32(rgba_b), np.uint32(rgba_a)).astype(np.uint32)
    image = QtGui.QImage(
        pixels.tobytes(), width, height, width * 4, QtGui.QImage.Format.Format_ARGB32
    ).copy()
    image.setDevicePixelRatio(ratio)
    return image


# Whole boxes repeat (cards, frames); windows of a zoomed map change with every pan, so they keep only two.
_hatch_image = lru_cache(maxsize=64)(_hatch_pixels)
_hatch_window = lru_cache(maxsize=2)(_hatch_pixels)


def paint_css_hatch(
    painter: QtGui.QPainter,
    rect: QtCore.QRect,
    color_a: QtGui.QColor,
    color_b: QtGui.QColor,
    length_a: float,
    length_b: float,
) -> None:
    """CSS ``repeating-linear-gradient(135deg, a 0 length_a, b length_a length_a+length_b)`` over the
    whole-pixel box ``rect``, evaluated per device pixel as a browser does (Qt's gradients place a few
    band edges a pixel off)."""
    if rect.width() <= 0 or rect.height() <= 0:
        return
    device = painter.device()
    ratio = device.devicePixelRatioF() if device is not None else 1.0
    image = _hatch_image(
        round(rect.width() * ratio),
        round(rect.height() * ratio),
        color_a.rgba(),
        color_b.rgba(),
        length_a * ratio,
        length_b * ratio,
        ratio,
        0,
    )
    painter.drawImage(rect.topLeft(), image)


def paint_css_hatch_window(
    painter: QtGui.QPainter,
    box: QtCore.QRect,
    window: QtCore.QRect,
    color_a: QtGui.QColor,
    color_b: QtGui.QColor,
    length_a: float,
    length_b: float,
) -> None:
    """The hatch of ``paint_css_hatch`` over ``box`` painted only where it meets ``window`` (a zoomed map's
    box is far larger than its stage): the bands keep their phase from the box's top-left corner."""
    visible = box.intersected(window)
    if visible == box:
        paint_css_hatch(painter, box, color_a, color_b, length_a, length_b)
        return
    if visible.isEmpty():
        return
    device = painter.device()
    ratio = device.devicePixelRatioF() if device is not None else 1.0
    offset = round((visible.left() - box.left() + visible.top() - box.top()) * ratio)
    image = _hatch_window(
        round(visible.width() * ratio),
        round(visible.height() * ratio),
        color_a.rgba(),
        color_b.rgba(),
        length_a * ratio,
        length_b * ratio,
        ratio,
        offset,
    )
    painter.drawImage(visible.topLeft(), image)


def paint_legend_dot(
    painter: QtGui.QPainter, x: float, y: float, theme: Theme, token: str | None
) -> None:
    """``.legend-dot``: 8×8, radius 2; ``None`` draws the remaining hatch (``data.legendRemainingA/B``)."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    path = rounded_path(QtCore.QRectF(x, y, 8, 8), 2)
    if token is None:
        painter.setClipPath(path, QtCore.Qt.ClipOperation.IntersectClip)
        paint_css_hatch(
            painter,
            QtCore.QRect(snap(x), snap(y), 8, 8),
            theme.color("data.legendRemainingA"),
            theme.color("data.legendRemainingB"),
            2,
            2,
        )
    else:
        painter.setBrush(theme.color(token))
        painter.drawPath(path)
    painter.restore()


# ---------- Worker tag ----------


WORKER_TAG = TextStyle(11, 600, 11, mono=True)


def worker_tag_width(tag: str) -> float:
    """``.wtag``: the text with 4 px padding inside a 1 px ring."""
    return 2 + 8 + WORKER_TAG.advance(tag)


def paint_worker_tag(
    painter: QtGui.QPainter, left: float, top: float, tag: str, theme: Theme
) -> float:
    """``.wtag`` ("W1"): mono 11 px 600 ``text.secondary`` in a 19 px box with a 1 px ``border.line`` ring
    (radius 3) whose top-left corner is (left, top); returns its width."""
    width = worker_tag_width(tag)
    box_left = snap(left)
    box = QtCore.QRectF(box_left, snap(top), snap(left + width) - box_left, 19)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(rounded_path(box.adjusted(0.5, 0.5, -0.5, -0.5), 2.5))
    painter.restore()
    draw_text(painter, left + 5, top + 4, tag, WORKER_TAG, theme.color("text.secondary"))
    return width


# ---------- Explorer tag ----------


EXPLORER_TAG = TextStyle(11, 500, 11)


def explorer_tag_width(tag: str) -> float:
    """The explorer tag: the text with 5 px padding inside a 1 px ring (``.wtag``'s box)."""
    return 2 + 10 + EXPLORER_TAG.advance(tag)


def paint_explorer_tag(
    painter: QtGui.QPainter, left: float, top: float, tag: str, theme: Theme
) -> float:
    """An external explorer's name on a profile row ("m-explore-ros2"): 11 px 500 ``text.secondary`` in the
    worker tag's 19 px ``border.line`` ring (radius 3) whose top-left corner is (left, top); returns its width."""
    width = explorer_tag_width(tag)
    box_left = snap(left)
    box = QtCore.QRectF(box_left, snap(top), snap(left + width) - box_left, 19)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(rounded_path(box.adjusted(0.5, 0.5, -0.5, -0.5), 2.5))
    painter.restore()
    draw_text(painter, left + 6, top + 4, tag, EXPLORER_TAG, theme.color("text.secondary"))
    return width


# ---------- Progress ring ----------


@dataclass(frozen=True, slots=True)
class RingGeometry:
    size: int
    radius: float
    stroke: float
    value: TextStyle
    caption: TextStyle | None = None


CAMPAIGN_RING = RingGeometry(
    92, 39, 8, TextStyle(19, 600, 22, tabular=True), TextStyle(11, 400, 13, tabular=True)
)
SMALL_RING = RingGeometry(64, 27, 6, TextStyle(13, 600, 18, tabular=True))


class ProgressRing(Themed):
    """``.ring``: ``data.remaining`` track and an ``accent.primary`` arc from 12 o'clock, clockwise, with round
    caps (as the mockup's SVG); the share and an optional caption in the centre. A complete ring is
    ``data.completed`` and reads "100 %" (02)."""

    def __init__(
        self,
        themes: ThemeManager,
        geometry: RingGeometry = CAMPAIGN_RING,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.geometry_spec = geometry
        self.share: float | None = None
        self.text = "—"
        self.caption = ""
        self.setFixedSize(geometry.size, geometry.size)

    def set_value(self, share: float | None, caption: str = "", *, decimals: int = 1) -> None:
        """``share`` in [0, 1] shown as "52.1 %"; None shows "—" and no arc."""
        if share is not None and not math.isfinite(share):
            share = None
        share = None if share is None else min(1.0, max(0.0, share))
        if share is None:
            text = "—"
        elif share == 1:
            text = "100 %"
        else:  # rounded, but an incomplete share never reads as complete
            text = f"{min(100 * share, 100 - 10**-decimals):.{decimals}f} %"
        if (share, text, caption) != (self.share, self.text, self.caption):
            self.share, self.text, self.caption = share, text, caption
            self.setAccessibleName(text + (f", {caption}" if caption else ""))
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        spec = self.geometry_spec
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        center = spec.size / 2
        box = QtCore.QRectF(
            center - spec.radius, center - spec.radius, 2 * spec.radius, 2 * spec.radius
        )
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(self.color("data.remaining"), spec.stroke))
        painter.drawEllipse(box)
        if self.share:
            arc = "data.completed" if self.share == 1 else "accent.primary"
            pen = QtGui.QPen(self.color(arc), spec.stroke)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(box, 90 * 16, -round(self.share * 360 * 16))
        caption = spec.caption if self.caption else None
        block = spec.value.line + (1 + caption.line if caption else 0)
        top = (spec.size - block) / 2
        x = (spec.size - spec.value.advance(self.text)) / 2
        draw_text(painter, x, top, self.text, spec.value, self.color("text.ink"))
        if caption is not None:
            draw_text(
                painter,
                (spec.size - caption.advance(self.caption)) / 2,
                top + spec.value.line + 1,
                self.caption,
                caption,
                self.color("text.secondary"),
            )


# ---------- Outcome bar and meter ----------


BAR_SEGMENTS = (
    ("completed", "data.completed"),
    ("time_limit", "data.timeLimit"),
    ("failure", "data.failure"),
    # D9: no-movement and stopped runs join as neutral segments when there are any.
    ("no_movement", "state.neutral"),
    ("stopped", "state.neutral"),
)


class OutcomeBar(Themed, CssBox):
    """``.bar``: 8 px, radius 4, 2 px gaps; completed, time-limit and failure shares of the planned
    experiments (and no-movement and stopped ones, neutral, D9), then the remaining share hatched at 135°
    (zero-count segments take no gap)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.counts: dict[str, int] = {}
        self.total = 0
        self.setFixedHeight(8)

    def set_counts(self, counts: dict[str, int], total: int) -> None:
        if (counts, total) != (self.counts, self.total):
            self.counts, self.total = dict(counts), total
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        width = float(self.width()) if self.css_width is None else self.css_width
        clip = rounded_path(QtCore.QRectF(0, 0, self.width(), 8), 4)
        painter.setClipPath(clip)
        painter.fillRect(self.rect(), self.color("data.remaining"))
        if self.total <= 0:
            return
        shown = [(key, token) for key, token in BAR_SEGMENTS if self.counts.get(key, 0) > 0]
        resolved = sum(self.counts[key] for key, _ in shown)
        remaining = resolved < self.total
        # CSS: each segment is its share of the bar's width with 2 px gaps between the present elements
        # (the hatched remainder too). When segments and gaps overflow, the segments shrink in proportion
        # (flex-shrink); the remainder has no basis and takes what is left.
        widths = [width * self.counts[key] / self.total for key, _ in shown]
        gaps = 2 * max(0, len(shown) + int(remaining) - 1)
        if widths and sum(widths) + gaps > width:
            scale = max(0.0, width - gaps) / sum(widths)
            widths = [segment * scale for segment in widths]
        x = 0.0
        for (_, token), segment in zip(shown, widths, strict=True):
            left, right = snap(x), snap(x + segment)  # backgrounds snap like a browser's
            painter.fillRect(QtCore.QRect(left, 0, right - left, 8), self.color(token))
            x += segment + 2
        left = snap(x)
        if remaining and left < width:
            paint_css_hatch(
                painter,
                QtCore.QRect(left, 0, self.width() - left, 8),
                self.color("data.remainingHatchA"),
                self.color("data.remainingHatchB"),
                4,
                4,
            )


class Meter(Themed):
    """``.meter``: 4 px, radius 2, ``data.remaining`` track, ``accent.primary`` fill."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.share: float | None = None
        self.setFixedHeight(4)

    def set_share(self, share: float | None) -> None:
        share = None if share is None or not math.isfinite(share) else min(1.0, max(0.0, share))
        if share != self.share:
            self.share = share
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setClipPath(rounded_path(QtCore.QRectF(self.rect()), 2))
        painter.fillRect(self.rect(), self.color("data.remaining"))
        if self.share:
            painter.fillRect(
                QtCore.QRectF(0, 0, self.width() * self.share, 4), self.color("accent.primary")
            )


# ---------- Key-value lists ----------


KV_LABEL = TextStyle(13, 400, 18)
KV_VALUE = TextStyle(13, 500, 18, tabular=True)


@dataclass(frozen=True, slots=True)
class KeyValue:
    label: str
    value: str
    unit: str = ""
    tooltip: str = ""
    value_token: str = "text.ink"
    value_style: TextStyle | None = None  # this row's value style (the list's by default)
    mark: str | None = None  # a 7 px dot in this colour token before the value (``.okd``, ``.wd``)
    action: str = ""  # a link after the value ("Show log"); a click emits ``KeyValueList.acted``


class KeyValueList(Themed):
    """``.kv``: rows with a bottom rule each; label ``text.secondary`` left, value ink 500 right-aligned with
    tabular figures and a muted unit. ``row_height`` 30 (32 on Monitor, 31 in the Results run view, with a 12.5 px
    label); ``last_open`` leaves the last row without its rule. ``value_style`` sets the values' size (the
    units follow it at weight 400)."""

    acted = QtCore.Signal(int)  # the row whose action was clicked

    def __init__(
        self,
        themes: ThemeManager,
        *,
        row_height: int = 30,
        label_style: TextStyle = KV_LABEL,
        value_style: TextStyle = KV_VALUE,
        last_open: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.row_height, self.label_style, self.last_open = row_height, label_style, last_open
        self.value_style = value_style
        self.unit_style = TextStyle(value_style.size, 400, value_style.line)
        self.rows: list[KeyValue] = []
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def set_rows(self, rows: Sequence[KeyValue]) -> None:
        rows = list(rows)
        if rows != self.rows:
            resized = len(rows) != len(self.rows)
            self.rows = rows
            self.setAccessibleDescription(
                "; ".join(
                    f"{row.label}: {row.value}{(' ' + row.unit) if row.unit else ''}"
                    for row in rows
                )
            )
            if resized:
                self.setFixedHeight(self.row_height * len(rows))
                self.updateGeometry()
            self.update()

    def sizeHint(self) -> QtCore.QSize:
        width = max(
            (
                self.label_style.advance(row.label)
                + 16
                + (row.value_style or self.value_style).advance(row.value)
                + (2 + self.unit_style.advance(row.unit) if row.unit else 0)
                for row in self.rows
            ),
            default=0.0,
        )
        return QtCore.QSize(math.ceil(width), self.row_height * len(self.rows))

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        row = int(event.position().y() // self.row_height)
        if 0 <= row < len(self.rows) and self.rows[row].action:
            action = self.rows[row]
            link = TextStyle((action.value_style or self.value_style).size, 500, 18)
            if event.position().x() >= self.width() - link.advance(action.action) - 4:
                self.acted.emit(row)
                return
        super().mouseReleaseEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            row = event.pos().y() // self.row_height
            tip = self.rows[row].tooltip if 0 <= row < len(self.rows) else ""
            show_tooltip(self, event.globalPos(), tip)
            return True
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        width = self.width()
        rule = self.color("border.line")
        for index, row in enumerate(self.rows):
            top = index * self.row_height
            open_row = self.last_open and index == len(self.rows) - 1
            # The line box is centred above the 1 px rule (in the whole row when it has none).
            top_offset = (self.row_height - (0 if open_row else 1) - 18) / 2
            draw_text(
                painter,
                0,
                top + top_offset,
                row.label,
                self.label_style,
                self.color("text.secondary"),
            )
            style = row.value_style or self.value_style
            right = float(width)
            if row.action:  # the link at the right, 8 px after the value
                link = TextStyle(style.size, 500, style.line)
                link_width = link.advance(row.action)
                draw_text(
                    painter,
                    right - link_width,
                    top + top_offset,
                    row.action,
                    link,
                    self.color("text.ink"),
                )
                baseline = link.baseline(top + top_offset) + 2
                painter.fillRect(
                    QtCore.QRectF(right - link_width, baseline, link_width, 1),
                    self.color("text.ink"),
                )
                right -= link_width + 8
            unit_width = (2 + self.unit_style.advance(row.unit)) if row.unit else 0.0
            # a long value is cut to the room the label leaves (``.kv .v`` padding-left 16)
            room = right - self.label_style.advance(row.label) - 16 - unit_width
            value = style.elide(row.value, room)
            value_width = style.advance(value)
            x = right - unit_width - value_width
            draw_text(painter, x, top + top_offset, value, style, self.color(row.value_token))
            if row.mark:  # ``vertical-align: 1px`` on the text's baseline, 6 px before it
                baseline = style.baseline(top + top_offset)
                painter.save()
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(self.color(row.mark))
                painter.drawEllipse(QtCore.QRectF(x - 6 - 7, baseline - 1 - 7, 7, 7))
                painter.restore()
            if row.unit:
                draw_text(
                    painter,
                    right - unit_width + 2,
                    top + top_offset,
                    row.unit,
                    self.unit_style,
                    self.color("text.tertiary"),
                )
            if not open_row:
                painter.fillRect(QtCore.QRect(0, top + self.row_height - 1, width, 1), rule)


# ---------- Coverage chart ----------


CHART_LABEL = TextStyle(10.5, 400, 13, tabular=True)


@dataclass(frozen=True, slots=True)
class CoverageSample:
    seconds: float  # simulation time since the attempt's first movement
    percent: float
    start: bool = False  # the line breaks before this sample (a gap in the data)


def tick_step(span: float, ticks: int = 5) -> float:
    """The first of the chart steps that puts at most ``ticks`` intervals on ``span`` seconds."""
    for step in (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200):
        if span / step <= ticks:
            return float(step)
    return 14400.0


class CoverageChart(Themed):
    """Coverage over simulation time (HANDOFF §12.4): 0-100 % with grid lines every 25 %, the area tinted at
    10 %, a 2 px line, milestone circles with dashed drop lines and a filled latest point."""

    PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 26, 6, 6, 18

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.samples: list[CoverageSample] = []
        self.milestones: list[CoverageSample] = []
        self.x_end: float | None = None
        self.setMinimumHeight(self.PAD_TOP + self.PAD_BOTTOM + 40)

    def set_data(
        self,
        samples: Sequence[CoverageSample],
        milestones: Sequence[CoverageSample] = (),
        *,
        x_end: float | None = None,
    ) -> None:
        """``x_end`` is the end of the time axis (default: 10 % after the last sample)."""
        finite = [s for s in samples if math.isfinite(s.seconds) and math.isfinite(s.percent)]
        self.samples = sorted(finite, key=lambda s: s.seconds)
        self.milestones = [
            m for m in milestones if math.isfinite(m.seconds) and math.isfinite(m.percent)
        ]
        self.x_end = x_end
        self.update()

    def _x_range(self) -> float:
        if self.x_end is not None and self.x_end > 0:
            return self.x_end
        last = self.samples[-1].seconds if self.samples else 0.0
        return max(60.0, last * 1.1)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        left, top = self.PAD_LEFT, self.PAD_TOP
        right, bottom = self.width() - self.PAD_RIGHT, self.height() - self.PAD_BOTTOM
        plot_w, plot_h = right - left, bottom - top
        tertiary = self.color("text.tertiary")
        for percent in (0, 25, 50, 75, 100):
            y = bottom - plot_h * percent / 100
            token = "chart.axis" if percent == 0 else "chart.grid"
            painter.fillRect(QtCore.QRectF(left, y - 0.5, plot_w, 1), self.color(token))
            if percent in (0, 50, 100):
                text = str(percent)
                # SVG text: right-aligned 6 px left of the plot, baseline 3.5 px below the line.
                self._label(painter, left - 6 - CHART_LABEL.advance(text), y + 3.5, text, tertiary)
        span = self._x_range()

        def to_point(sample: CoverageSample) -> QtCore.QPointF:
            return QtCore.QPointF(
                left + plot_w * sample.seconds / span, bottom - plot_h * sample.percent / 100
            )

        step = tick_step(span)
        tick = 0.0
        while tick <= span + 1e-9:
            text = duration(tick)
            x = left + plot_w * tick / span
            width = CHART_LABEL.advance(text)
            label_x = min(x - width / 2, self.width() - width)
            self._label(painter, label_x, self.height() - 4, text, tertiary)
            tick += step
        if not self.samples:
            return
        coverage = self.color("data.coverage")
        # Each run of samples without a gap is one area and one line; nothing joins across a gap.
        segments: list[list[QtCore.QPointF]] = []
        for sample in self.samples:
            if sample.start or not segments:
                segments.append([])
            segments[-1].append(to_point(sample))
        # Samples outside the axes are clipped, never moved onto the edge.
        painter.setClipRect(QtCore.QRectF(left, 0, plot_w + self.PAD_RIGHT, bottom + 4))
        tint = QtGui.QColor(coverage)
        tint.setAlphaF(0.10)
        pen = QtGui.QPen(coverage, 2)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        for points in segments:
            area = QtGui.QPolygonF(
                [
                    QtCore.QPointF(points[0].x(), bottom),
                    *points,
                    QtCore.QPointF(points[-1].x(), bottom),
                ]
            )
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(tint)
            painter.drawPolygon(area)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QtGui.QPolygonF(points))
        card = self.color("surface.card")
        for milestone in self.milestones:
            point = to_point(milestone)
            drop = QtGui.QColor(coverage)
            drop.setAlphaF(0.55)
            dashed = QtGui.QPen(drop, 1)
            dashed.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)  # SVG butt caps
            dashed.setDashPattern([2, 2])
            painter.setPen(dashed)
            painter.drawLine(point, QtCore.QPointF(point.x(), bottom))
            painter.setPen(QtGui.QPen(coverage, 1.6))
            painter.setBrush(card)
            painter.drawEllipse(point, 3.5, 3.5)
        painter.setPen(QtGui.QPen(card, 1.5))
        painter.setBrush(coverage)
        painter.drawEllipse(segments[-1][-1], 3.5, 3.5)

    @staticmethod
    def _label(
        painter: QtGui.QPainter, x: float, baseline: float, text: str, color: QtGui.QColor
    ) -> None:
        """SVG ``<text>``: placed by its baseline."""
        painter.save()
        painter.setPen(color)
        paint_glyphs(painter, x, baseline, text, CHART_LABEL)
        painter.restore()
