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

"""The painted map stage (HANDOFF §7.11, §8.9), as the Monitor shows a live run: the world fitted 20 px inside the
``surface.card`` stage, in a
hatched box with a 1 px ``border.line`` ring (``.stage .map``), with the run's ``LiveLayers`` drawn as ``maps.js``
draws them (``gui_maps.paint_live``: explored cells, the Nav2 plan and goal, the ground-truth path with its halo
and start mark, reference disagreements, the robot), in screen pixels at every zoom; zoom controls in the box's
top-right corner (D3: zoom in and zoom out), a scale bar in its bottom-left corner, and the legend line under
the stage.

The wheel and the controls zoom (up to 16 times the fitted scale), dragging pans a zoomed map, and zooming out
to the fitted scale centres the map again. The path is drawn by ``gui_maps.PathLayers``: cached for the view,
extended by each update, drawn again progressively after a new view.
"""

import math
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_maps import (
    LiveLayers,
    PathLayers,
    Placement,
    WorldMap,
    explored_image,
    fitted,
    paint_live,
)
from .gui_marks import paint_css_hatch_window
from .gui_theme import Theme, ThemeManager, snap
from .gui_widgets import (
    BODY,
    TextStyle,
    Themed,
    draw_text,
    press_on_enter,
    rounded_path,
    shadow_image,
    shadow_margin,
)

PAD = 20  # ``data-pad="20"``
ROBOT_SIZE = 10  # ``data-robot="10"`` on 09
ZOOM_STEP, MIN_ZOOM, MAX_ZOOM = 1.25, 0.25, 16.0
SCALE_TEXT = TextStyle(11.5, 400, 18, tabular=True)
SCALE_MAX_PX = 100.0  # the longest scale bar
Explored = np.ndarray[Any, np.dtype[np.uint8]]


def nice_distance(pixels_per_metre: float) -> float | None:
    """The scale bar's distance: the longest 1, 2 or 5 × 10ⁿ m that is at most ``SCALE_MAX_PX`` long."""
    if not math.isfinite(pixels_per_metre) or pixels_per_metre <= 0:
        return None
    exponent = math.floor(math.log10(SCALE_MAX_PX / pixels_per_metre))
    best = None
    for power in (exponent - 1, exponent, exponent + 1):
        for step in (1.0, 2.0, 5.0):
            distance = step * 10.0**power
            if distance * pixels_per_metre <= SCALE_MAX_PX + 1e-9 and (
                best is None or distance > best
            ):
                best = distance
    return best


def snapped(rect: QtCore.QRectF) -> QtCore.QRect:
    """The whole-pixel box a browser draws a fractional box on (edges rounded half up)."""
    left, top = snap(rect.left()), snap(rect.top())
    return QtCore.QRect(left, top, snap(rect.right()) - left, snap(rect.bottom()) - top)


class MapButton(QtWidgets.QAbstractButton):
    """One ``.map-ctl span``: 30×30 with a 16 px ``text.body`` icon; ``surface.hover`` under the pointer
    (``surface.selected`` pressed), clipped to the column's rounded corners."""

    def __init__(
        self, themes: ThemeManager, icon: str, tooltip: str, parent: QtWidgets.QWidget
    ) -> None:
        super().__init__(parent)
        self.themes, self.glyph = themes, icon
        self.clip = QtGui.QPainterPath()
        self.rule = False  # a 1 px rule along the top (``span + span``)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 6)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.setAutoRepeat(True)
        themes.changed.connect(self.update)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if not press_on_enter(self, event):
            super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.themes.current
        top = 1 if self.rule else 0
        if self.isDown() or self.underMouse():
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setClipPath(self.clip)
            painter.fillRect(
                QtCore.QRect(0, top, self.width(), self.height() - top),
                theme.color("surface.selected" if self.isDown() else "surface.hover"),
            )
            painter.setClipping(False)
        if self.rule:
            painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), theme.color("border.line"))
        # The icon is centred in the content box (below the rule).
        origin = QtCore.QPointF(7, top + (self.height() - top - 16) / 2)
        gui_icons.draw(painter, self.glyph, theme.color("text.body"), origin)


class MapControls(Themed):
    """``.map-ctl``: a 30×30 button on ``surface.card`` in a 1 px ``border.line`` ring, radius 6 (its small
    shadow is painted by the map under it). The maintainer's choice (17 October 2026): one button that resets
    the view to fit; zoom is the wheel's."""

    SHADOW = (0, 1, 2, 0)  # ``0 1px 2px overlay.shadowSmall``

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget) -> None:
        super().__init__(themes, parent)
        self.fit = MapButton(themes, "fit", "Reset the view to fit", self)
        self.fit.setAutoRepeat(False)
        self.fit.setGeometry(1, 1, 30, 30)
        self.fit.clip = rounded_path(QtCore.QRectF(0, 0, 30, 30), 5)
        self.setFixedSize(32, 32)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 5.5)
        )


class LiveMap(Themed):
    """The map stage (see the module docstring); the widget is the stage (``.stage``). It shows a run's
    ``LiveLayers``; call ``update`` after changing them in place. Worlds & Spawns (``gui_spawns.SpawnMap``) and the
    Results run view (``gui_run_detail.RunMapView``) change the class attributes below."""

    PAD = PAD  # ``data-pad``: the fitted world's margin in the stage
    RING = "border.line"  # the 1 px ring around the world box
    CONTROLS = True  # the zoom controls in the box's top-right corner
    SCALE_BAR = True
    ROBOT = ROBOT_SIZE  # the robot mark's size (``data-robot``)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.world: WorldMap | None = None
        self.layers = LiveLayers()
        self.message = ""
        self.zoom = 1.0
        self.focus = (0.5, 0.5)  # the box fraction at the stage centre
        self._drag: QtCore.QPointF | None = None
        self.path_layers = PathLayers()
        # The base image with its view key and the explored cells it shows (kept alive, compared by identity).
        self._base: tuple[tuple[object, ...], Explored | None, QtGui.QImage] | None = None
        self._explored: tuple[Explored, str, QtGui.QImage] | None = None
        self.controls = MapControls(themes, self)
        self.controls.fit.clicked.connect(self.fit_view)
        self.controls.hide()
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )

    # ---------- data ----------

    def set_world(self, world: WorldMap | None) -> None:
        """Show ``world`` fitted (a new world starts unzoomed)."""
        if world is self.world:
            return
        self.world = world
        self.zoom, self.focus = 1.0, (0.5, 0.5)
        self._base = self._explored = None
        self.path_layers.invalidate()
        self.controls.setVisible(world is not None and self.CONTROLS)
        self.relayout()
        self.update()

    def set_layers(self, layers: LiveLayers) -> None:
        self.layers = layers
        self.update()

    def set_message(self, text: str) -> None:
        """Shown centred while no world is set."""
        if text != self.message:
            self.message = text
            self.update()

    # ---------- view ----------

    def stage(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.rect())

    def fit_rect(self) -> QtCore.QRectF | None:
        return None if self.world is None else fitted(self.world, self.stage(), self.PAD).rect

    def box(self) -> QtCore.QRectF | None:
        """The world box in widget coordinates (the fitted box until the view is zoomed or panned)."""
        fit = self.fit_rect()
        if fit is None or self.fitted():
            return fit
        width, height = fit.width() * self.zoom, fit.height() * self.zoom
        centre = self.stage().center()
        return QtCore.QRectF(
            centre.x() - self.focus[0] * width, centre.y() - self.focus[1] * height, width, height
        )

    def placement(self) -> Placement | None:
        box = self.box()
        return None if box is None or self.world is None else Placement.fill(self.world, box)

    def visible_box(self) -> QtCore.QRectF | None:
        box = self.box()
        return None if box is None else box.intersected(self.stage())

    def zoom_by(self, factor: float, anchor: QtCore.QPointF | None = None) -> None:
        """Zoom about ``anchor`` (default: the stage centre), from a quarter of the fitted scale to 16 times it."""
        box, fit = self.box(), self.fit_rect()
        if box is None or fit is None:
            return
        zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        if abs(zoom - self.zoom) < 1e-9:
            return
        point = anchor if anchor is not None else self.stage().center()
        fraction_x = (point.x() - box.left()) / box.width()
        fraction_y = (point.y() - box.top()) / box.height()
        width, height = fit.width() * zoom, fit.height() * zoom
        left, top = point.x() - fraction_x * width, point.y() - fraction_y * height
        centre = self.stage().center()
        self.zoom = zoom
        self.focus = (
            min(1.0, max(0.0, (centre.x() - left) / width)),
            min(1.0, max(0.0, (centre.y() - top) / height)),
        )
        self.view_changed()

    def fitted(self) -> bool:
        return self.zoom == 1.0 and self.focus == (0.5, 0.5)

    def fit_view(self) -> None:
        """Back to the fitted view (the reset button)."""
        if not self.fitted():
            self.zoom, self.focus = 1.0, (0.5, 0.5)
            self.view_changed()

    def pan_by(self, dx: float, dy: float) -> None:
        """Drag the map (at any zoom); the stage centre stays on the map."""
        box = self.box()
        if box is None:
            return
        self.focus = (
            min(1.0, max(0.0, self.focus[0] - dx / box.width())),
            min(1.0, max(0.0, self.focus[1] - dy / box.height())),
        )
        self.view_changed()

    def view_changed(self) -> None:
        self._base = None
        self.relayout()
        self.update()

    def relayout(self) -> None:
        """The reset button 10 px inside the fitted box's top-right corner (where it stays at every zoom)."""
        fit = self.fit_rect()
        visible = None if fit is None else fit.intersected(self.stage())
        if visible is None or visible.isEmpty():
            return
        x = snap(visible.right() - 10 - self.controls.width())
        self.controls.move(x, snap(visible.top() + 10))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._base = None
        self.relayout()

    def restyle(self, *_: object) -> None:
        self._base = self._explored = None
        self.update()

    # ---------- input ----------

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120
        if self.world is None or not steps:
            event.ignore()
            return
        self.zoom_by(ZOOM_STEP**steps, event.position())
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.world is not None:
            self._drag = event.position()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag is not None:
            delta = event.position() - self._drag
            self._drag = event.position()
            self.pan_by(delta.x(), delta.y())
            return
        self._update_cursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = None
            self._update_cursor()
            return
        super().mouseReleaseEvent(event)

    def _update_cursor(self) -> None:
        if self.world is not None:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    # ---------- layer hooks (the Monitor's insight map overrides them) ----------

    def shows(self, layer: str) -> bool:
        """Whether a run layer (``path``, ``explored``, ``navigation``) is switched on."""
        return True

    def paint_under(self, painter: QtGui.QPainter, placement: Placement) -> None:
        """Marks under the run's marks, over the world and explored cells."""

    def paint_marks(self, painter: QtGui.QPainter, placement: Placement) -> None:
        """Marks over the path, under the goal and the robot."""

    def paint_over(self, painter: QtGui.QPainter, placement: Placement) -> None:
        """Marks over everything else on the map."""

    # ---------- painting ----------

    def _base_image(self, box: QtCore.QRectF) -> QtGui.QImage:
        """Stage, hatched box, ring, world and explored cells for the current view."""
        explored = self.layers.explored if self.shows("explored") else None
        world = self.world
        if world is not None and explored is not None and explored.shape != world.pixels.shape:
            explored = None
        ratio = self.devicePixelRatioF()
        key = (box.getRect(), self.width(), self.height(), ratio, self.theme.name)
        if self._base is not None and self._base[0] == key and self._base[1] is explored:
            return self._base[2]
        theme = self.theme
        image = QtGui.QImage(
            max(1, round(self.width() * ratio)),
            max(1, round(self.height() * ratio)),
            QtGui.QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.setDevicePixelRatio(ratio)
        painter = QtGui.QPainter(image)
        painter.fillRect(self.stage(), theme.color("surface.card"))  # ``.stage``
        world_box = snapped(box)
        # ``box-shadow: 0 0 0 1px border.line``: a ring just outside the box.
        painter.fillRect(world_box.adjusted(-1, -1, 1, 1), theme.color(self.RING))
        painter.fillRect(world_box, theme.color("map.outside"))
        paint_css_hatch_window(
            painter,
            world_box,
            self.rect(),
            QtGui.QColor(0, 0, 0, 0),
            theme.color("map.hatch"),
            5,
            1,
        )
        if world is not None:
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(world_box, world.image(theme))
            if explored is not None:
                cached = self._explored
                if cached is None or cached[0] is not explored or cached[1] != theme.name:
                    cached = (explored, theme.name, explored_image(explored, theme))
                    self._explored = cached
                painter.drawImage(world_box, cached[2])
        painter.end()
        self._base = (key, explored, image)
        return image

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        box = self.box()
        placement = self.placement()
        if box is None or placement is None:
            painter.fillRect(self.rect(), theme.color("surface.card"))
            if self.message:
                text = BODY.elide(self.message, max(0.0, self.width() - 48.0))
                x = (self.width() - BODY.advance(text)) / 2
                color = theme.color("text.secondary")
                draw_text(painter, x, (self.height() - 18) / 2, text, BODY, color)
            return
        painter.drawImage(QtCore.QPointF(0, 0), self._base_image(box))
        if self.controls.isVisible():  # the controls' small shadow, under them
            controls = self.controls.geometry()
            layers = ((*MapControls.SHADOW, theme.tokens["overlay.shadowSmall"]),)
            margin = shadow_margin(layers)
            painter.drawImage(
                QtCore.QPointF(controls.x() - margin, controls.y() - margin),
                shadow_image(controls.width(), controls.height(), 6, layers),
            )
        self.paint_under(painter, placement)
        path = self.layers.path
        complete = True
        show_path = self.shows("path")
        if show_path and len(path):
            complete = self.path_layers.draw(
                path, placement, self.size(), self.devicePixelRatioF(), theme
            )
        paint_live(
            painter,
            self.layers,
            placement,
            theme,
            self.path_layers,
            self.ROBOT,
            path=show_path,
            navigation=self.shows("navigation"),
            marks=lambda: self.paint_marks(painter, placement),
        )
        self.paint_over(painter, placement)
        if self.SCALE_BAR:
            self._paint_scale(painter, placement, theme)
        if not complete:
            QtCore.QTimer.singleShot(0, self.update)  # a long path goes on at the next paint

    def _paint_scale(self, painter: QtGui.QPainter, placement: Placement, theme: Theme) -> None:
        """``.scale``: 10 px inside the fitted box's bottom-left corner; ``map.float``, 1 px ``border.line``,
        radius 4, padding 5 8 6; the label above a 5 px bracket (1.5 px ``text.secondary``, no top edge) as
        long as the label's distance."""
        fit = self.fit_rect()
        visible = None if fit is None else fit.intersected(self.stage())  # fixed at every zoom
        if visible is None or visible.isEmpty():
            return
        distance = nice_distance(placement.scale / placement.world.resolution)
        if distance is None:
            return
        length = distance * placement.scale / placement.world.resolution
        label = f"{distance:g} m"
        inner = max(length, SCALE_TEXT.advance(label))
        width, height = 2 + 16 + inner, 39.0
        left, bottom = visible.left() + 10, visible.bottom() - 10
        box = QtCore.QRectF(left, bottom - height, width, height)
        frame = QtCore.QRectF(snapped(box))
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
        painter.setBrush(theme.color("map.float"))
        painter.drawPath(rounded_path(frame.adjusted(0.5, 0.5, -0.5, -0.5), 3.5))
        painter.restore()
        text_left, text_top = box.left() + 9, box.top() + 6
        draw_text(painter, text_left, text_top, label, SCALE_TEXT, theme.color("text.secondary"))
        # The bracket: CSS borders of 1.5 px drawn on whole device pixels (1 px at ratio 1).
        ratio = self.devicePixelRatioF()
        stroke = max(1.0, math.floor(1.5 * ratio)) / ratio
        bracket = QtCore.QRectF(snapped(QtCore.QRectF(text_left, text_top + 18 + 3, length, 5)))
        color = theme.color("text.secondary")
        painter.fillRect(QtCore.QRectF(bracket.left(), bracket.top(), stroke, 5), color)
        painter.fillRect(QtCore.QRectF(bracket.right() - stroke, bracket.top(), stroke, 5), color)
        painter.fillRect(
            QtCore.QRectF(bracket.left(), bracket.bottom() - stroke, bracket.width(), stroke),
            color,
        )


# ---------- Legend ----------


LEGEND_TEXT = TextStyle(12.5, 400, 18)
LEGEND_ITEMS = (
    ("path", 22.0, "Ground-truth path (Gazebo)"),
    ("explored", 12.0, "Explored cells (coverage)"),
    ("navigation", 30.0, "Nav2 goal and plan (SLAM estimate)"),
)


def _mix(color: QtGui.QColor, share: float, base: QtGui.QColor) -> QtGui.QColor:
    """CSS ``color-mix(in srgb, color share, base)``."""
    return QtGui.QColor(
        round(color.red() * share + base.red() * (1 - share)),
        round(color.green() * share + base.green() * (1 - share)),
        round(color.blue() * share + base.blue() * (1 - share)),
    )


class MapLegend(Themed):
    """``.legend``: 40 px with a top rule, padding 0 20, items 20 px apart (swatch, 7 px, 12.5 px
    ``text.secondary`` label); the status at the right with a ``data.completed`` dot while the pose is live.
    When the line is too narrow, the status keeps its room and the items that no longer fit are left out
    (the tooltip has everything)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.status = ""
        self.live = False
        self.setFixedHeight(40)

    def set_status(self, text: str, live: bool) -> None:
        if (text, live) != (self.status, self.live):
            self.status, self.live = text, live
            self.setToolTip(
                "\n".join([*(label for _, _, label in LEGEND_ITEMS), text] if text else [])
            )
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), theme.color("border.line"))
        top = 1 + (39 - 18) / 2
        secondary = theme.color("text.secondary")
        dot = 14.0 if self.live else 0.0
        status_width = dot + LEGEND_TEXT.advance(self.status) if self.status else 0.0
        # The status may run into the padding (as 09 draws it) up to 8 px before the edge; items that would
        # leave it less than 120 px are left out.
        limit = self.width() - 8.0
        reserve = 20 + min(status_width, 120.0) if self.status else 0.0
        x = 20.0
        for kind, swatch, label in LEGEND_ITEMS:
            width = swatch + 7 + LEGEND_TEXT.advance(label)
            if x + width + reserve > limit:
                break
            self.paint_swatch(painter, kind, x, top, theme)
            draw_text(painter, x + swatch + 7, top, label, LEGEND_TEXT, secondary)
            x += width + 20
        if not self.status:
            return
        # ``margin-left: auto``: at the right padding, or right after the items when they reach it.
        start = max(x, self.width() - 20.0 - status_width)
        text = self.status
        if start + status_width > limit:
            text = LEGEND_TEXT.elide(self.status, max(0.0, limit - start - dot))
        if self.live:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("data.completed"))
            painter.drawPath(rounded_path(QtCore.QRectF(start, top + 5.5, 7, 7), 3.5))
            painter.restore()
            start += 14
        draw_text(painter, start, top, text, LEGEND_TEXT, secondary)

    @staticmethod
    def paint_swatch(
        painter: QtGui.QPainter, kind: str, x: float, top: float, theme: Theme
    ) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        ink = theme.color("text.ink")
        # Inline SVG and the dot's box are painted on whole pixels.
        x = snap(x)
        if kind == "path":  # 22×10: a 5 px halo and a 2.4 px path, round caps
            y = snap(top + 4) + 5
            for color, width in ((theme.color("map.halo"), 5.0), (theme.color("data.path"), 2.4)):
                pen = QtGui.QPen(color, width)
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(x + 1, y), QtCore.QPointF(x + 21, y))
        elif kind == "explored":  # 12×12, radius 2: coverage at 22 % with a 45 % inset ring
            box = QtCore.QRectF(x, snap(top + 3), 12, 12)
            free, coverage = theme.color("map.free"), theme.color("data.coverage")
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(_mix(coverage, 0.45, free))
            painter.drawPath(rounded_path(box, 2))
            painter.setBrush(_mix(coverage, 0.22, free))
            painter.drawPath(rounded_path(box.adjusted(1, 1, -1, -1), 1))
        else:  # 30×12: the goal's × (2 px, round caps) and a dashed plan (1.5 px, 4/3)
            y0 = snap(top + 3)
            cross = QtGui.QPainterPath()
            cross.moveTo(x + 2, y0 + 2)
            cross.lineTo(x + 10, y0 + 10)
            cross.moveTo(x + 10, y0 + 2)
            cross.lineTo(x + 2, y0 + 10)
            pen = QtGui.QPen(ink, 2)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPath(cross)
            dashed = QtGui.QPen(ink, 1.5)
            dashed.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
            dashed.setDashPattern([4 / 1.5, 3 / 1.5])
            painter.setPen(dashed)
            painter.drawLine(QtCore.QPointF(x + 14, y0 + 6), QtCore.QPointF(x + 29, y0 + 6))
        painter.restore()
