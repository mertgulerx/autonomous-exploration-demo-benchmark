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

"""The produced PDFs on white pages (Results: the Study tab's outputs, the Files tab's PDFs; RESULTS_STUDY.md
decisions 5 and 7, RO-19): rendered with QtPdf at one scale, never recoloured in the dark theme."""

from dataclasses import dataclass, field
import math

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtPdf import QPdfDocument
from shiboken6 import isValid

from .gui_cells import OverlayScrollBar
from .gui_theme import ThemeManager, snap
from .gui_widgets import BODY, Themed, draw_text

# a double-column page (182 mm) is 840 px wide on the sheet (screens 25, 26)
PX_PER_PT = 840 / 515.9
SHEET_PAD = 30  # the white page's margin around the produced PDF


@dataclass
class Sheet:
    """One produced PDF on the page: its pages rendered for the current width and pixel ratio."""

    document: QPdfDocument
    buffer: QtCore.QBuffer
    images: list[QtGui.QImage] = field(default_factory=list)
    key: tuple[float, float] = (0.0, 0.0)  # the width and ratio the images were rendered for


class PdfPages(Themed):
    """``.well``: the produced PDFs one under the other, each on a white page with a 1 px edge (never
    recoloured in the dark theme), centred, at one scale (a double-column page 840 px wide, smaller when the
    well is narrower)."""

    GAP = 16

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.sheets: list[Sheet] = []
        self.message = ""

    def clear(self, message: str = "") -> None:
        for sheet in self.sheets:
            if isValid(sheet.document):
                sheet.document.statusChanged.disconnect()
                sheet.document.close()
                sheet.document.deleteLater()
            sheet.buffer.deleteLater()
        self.sheets, self.message = [], message
        self.fit()

    def add(self, data: bytes) -> None:
        buffer = QtCore.QBuffer(self)
        buffer.setData(QtCore.QByteArray(data))
        buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
        document = QPdfDocument(self)
        sheet = Sheet(document, buffer)
        document.statusChanged.connect(self.fit)
        document.load(buffer)
        self.sheets.append(sheet)
        self.fit()

    def ready(self) -> list[Sheet]:
        """The sheets whose PDF has loaded."""
        return [
            s
            for s in self.sheets
            if isValid(s.document) and s.document.status() == QPdfDocument.Status.Ready
        ]

    def scale(self) -> float:
        widest = max(
            (s.document.pagePointSize(0).width() for s in self.ready() if s.document.pageCount()),
            default=515.9,
        )
        parent = self.parentWidget()
        room = parent.width() - 40 if parent is not None else self.width()
        return min(PX_PER_PT, max(0.2, (room - 2 * SHEET_PAD) / max(1.0, widest)))

    def page_sizes(self, sheet: Sheet) -> list[QtCore.QSizeF]:
        scale = self.scale()
        return [
            QtCore.QSizeF(
                sheet.document.pagePointSize(page).width() * scale,
                sheet.document.pagePointSize(page).height() * scale,
            )
            for page in range(sheet.document.pageCount())
        ]

    def fit(self, *_: object) -> None:
        if any(
            isValid(s.document) and s.document.status() == QPdfDocument.Status.Error
            for s in self.sheets
        ):
            self.message = "A produced PDF could not be read (damaged or not a PDF)"
        height = 18.0 + (28.0 if self.message and self.ready() else 0.0)
        for sheet in self.ready():
            sizes = self.page_sizes(sheet)
            if sizes:
                height += sum(size.height() for size in sizes) + 2 * SHEET_PAD + self.GAP
        parent = self.parentWidget()
        width = parent.width() if parent is not None else self.width()
        self.resize(width, math.ceil(height + 18))
        self.update()

    def _images(self, sheet: Sheet) -> list[QtGui.QImage]:
        ratio = self.devicePixelRatioF()
        key = (self.scale(), ratio)
        if sheet.key != key:
            sheet.images = []
            for page, size in enumerate(self.page_sizes(sheet)):
                pixels = QtCore.QSize(
                    math.ceil(size.width() * ratio), math.ceil(size.height() * ratio)
                )
                image = sheet.document.render(page, pixels)
                image.setDevicePixelRatio(ratio)
                sheet.images.append(image)
            sheet.key = key
        return sheet.images

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        top = 18.0
        if self.message:  # above the readable sheets, if any
            draw_text(
                painter,
                20,
                18,
                BODY.elide(self.message, self.width() - 40),
                BODY,
                self.color("text.secondary"),
            )
            if not self.ready():
                return
            top += 28.0
        for sheet in self.ready():
            sizes = self.page_sizes(sheet)
            if not sizes:
                continue
            width = max(size.width() for size in sizes) + 2 * SHEET_PAD
            height = sum(size.height() for size in sizes) + 2 * SHEET_PAD
            left = snap((self.width() - width) / 2)
            page = QtCore.QRectF(left, top, width, height)
            painter.fillRect(page.adjusted(-1, -1, 1, 1), self.color("border.line"))
            painter.fillRect(page, QtGui.QColor("#FFFFFF"))
            y = top + SHEET_PAD
            for image, size in zip(self._images(sheet), sizes, strict=True):
                painter.drawImage(QtCore.QPointF(left + SHEET_PAD, y), image)
                y += size.height()
            top += height + self.GAP


class Well(QtWidgets.QScrollArea):
    """``.well``: the sunken, scrolling area under the sheets."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.themes = themes
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)
        self.pages = PdfPages(themes)
        self.setWidget(self.pages)
        OverlayScrollBar(themes, self)
        themes.changed.connect(self._restyle)
        self._restyle()

    def _restyle(self, *_: object) -> None:
        sunken = self.themes.current.qss("surface.sunken")
        self.setStyleSheet(
            f"QScrollArea, QScrollArea > QWidget > QWidget {{ background: {sunken}; }}"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.pages.fit()
