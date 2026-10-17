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

"""Line icons of the Main design: SVG files recoloured and rasterised per colour, size and stroke."""

from functools import lru_cache
import math
from pathlib import Path

from PySide6 import QtCore, QtGui, QtSvg

ICONS = Path(__file__).with_name("icons")
STROKE = 1.75  # stroke width in the 24-unit view box
BADGE_STROKE = 2.0


@lru_cache(maxsize=128)
def _source(name: str) -> str:
    return (ICONS / f"{name}.svg").read_text()


@lru_cache(maxsize=1024)
def _rendered(
    name: str, rgba: int, size: int, stroke: float, ratio: float, dx: float = 0.0, dy: float = 0.0
) -> QtGui.QPixmap:
    color = QtGui.QColor.fromRgba(rgba)
    text = (
        _source(name)
        .replace("currentColor", color.name(QtGui.QColor.NameFormat.HexRgb))
        .replace('stroke-width="STROKE"', f'stroke-width="{stroke:g}"')
    )
    if color.alpha() < 255:
        text = text.replace("<svg ", f'<svg opacity="{color.alphaF():.4f}" ', 1)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(text.encode()))
    width = max(1, math.ceil((size + dx) * ratio))
    height = max(1, math.ceil((size + dy) * ratio))
    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QtCore.QRectF(dx * ratio, dy * ratio, size * ratio, size * ratio))
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def pixmap(
    name: str,
    color: QtGui.QColor,
    size: int = 16,
    *,
    stroke: float = STROKE,
    ratio: float = 1.0,
) -> QtGui.QPixmap:
    """The icon ``name`` in ``color`` at ``size`` logical pixels for a device pixel ratio."""
    return _rendered(name, color.rgba(), size, stroke, ratio)


def draw(
    painter: QtGui.QPainter,
    name: str,
    color: QtGui.QColor,
    origin: QtCore.QPointF,
    size: int = 16,
    *,
    stroke: float = STROKE,
) -> None:
    """Paint the icon with its top-left corner at ``origin``, anti-aliased at fractional positions
    like a browser draws inline SVG."""
    ratio = painter.device().devicePixelRatioF() if painter.device() else 1.0
    transform = painter.worldTransform()
    if transform.isScaling() or transform.isRotating():
        painter.drawPixmap(origin, pixmap(name, color, size, stroke=stroke, ratio=ratio))
        return
    point = transform.map(origin)
    x, y = round(point.x() * 8) / 8, round(point.y() * 8) / 8  # 1/8 px keeps the cache small
    left, top = math.floor(x), math.floor(y)
    image = _rendered(name, color.rgba(), size, stroke, ratio, x - left, y - top)
    painter.save()
    painter.resetTransform()
    painter.drawPixmap(QtCore.QPointF(left, top), image)
    painter.restore()
