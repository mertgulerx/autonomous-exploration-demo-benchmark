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

"""Theme of the Main design: colours, fonts, palette, style sheet and the proxy style.

Every colour comes from the active ``Theme`` (light or dark tokens of ``gui_tokens``). ``ThemeManager`` stores
the choice in the application's QSettings, applies palette, style and style sheet, and emits ``changed`` so
painted widgets and caches repaint without a restart.
"""

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_tokens

THEME_KEY = "appearance/theme"
UI_FAMILIES = ("Ubuntu Sans", "Segoe UI Variable Text", "Segoe UI", "SF Pro Text")
MONO_FAMILIES = ("Ubuntu Sans Mono", "Cascadia Mono", "SF Mono")
LOG = logging.getLogger(__name__)


def css_color(value: str) -> QtGui.QColor:
    """``#RRGGBB`` or ``#RRGGBBAA`` (CSS order, alpha last) as a QColor."""
    digits = value.removeprefix("#")
    if len(digits) not in (6, 8) or not all(c in "0123456789abcdefABCDEF" for c in digits):
        raise ValueError(f"Not a CSS hex colour: {value!r}")
    alpha = int(digits[6:8], 16) if len(digits) == 8 else 255
    return QtGui.QColor(int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16), alpha)


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    tokens: Mapping[str, str]

    @property
    def dark(self) -> bool:
        return self.name == "dark"

    def color(self, token: str) -> QtGui.QColor:
        """A copy the caller may change (parsed once per theme and token)."""
        return QtGui.QColor(_parsed(self.tokens[token]))

    def qss(self, token: str) -> str:
        """The token as a style-sheet colour (``rgba(...)`` when translucent)."""
        color = self.color(token)
        if color.alpha() == 255:
            return color.name()
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()})"

    def map_color(self, role: str) -> QtGui.QColor:
        return css_color(gui_tokens.MAP[self.name][role])


@lru_cache(maxsize=512)
def _parsed(value: str) -> QtGui.QColor:
    return css_color(value)


THEMES: Mapping[str, Theme] = MappingProxyType(
    {"light": Theme("light", gui_tokens.LIGHT), "dark": Theme("dark", gui_tokens.DARK)}
)


# ---------- Fonts ----------


@lru_cache(maxsize=2)
def family(mono: bool = False) -> str:
    """First installed family of the design's stack; the fallback is logged once."""
    available = set(QtGui.QFontDatabase.families())
    stack = MONO_FAMILIES if mono else UI_FAMILIES
    for name in stack:
        if name in available:
            return name
    fallback = QtGui.QFontDatabase.systemFont(
        QtGui.QFontDatabase.SystemFont.FixedFont
        if mono
        else QtGui.QFontDatabase.SystemFont.GeneralFont
    ).family()
    LOG.warning("Fonts %s are not installed; using %s", ", ".join(stack), fallback)
    return fallback


WEIGHTS = {
    400: QtGui.QFont.Weight.Normal,
    500: QtGui.QFont.Weight.Medium,
    600: QtGui.QFont.Weight.DemiBold,
    700: QtGui.QFont.Weight.Bold,  # ``<b>`` (the success band of 08)
}


def font(
    size: float,
    weight: int = 400,
    *,
    mono: bool = False,
    tabular: bool = False,
    spacing_em: float = 0.0,
) -> QtGui.QFont:
    """A font in CSS terms: pixel size (fractional sizes allowed), weight, tabular figures, tracking.
    Callers get their own copy and may change it."""
    return QtGui.QFont(_font(size, weight, mono, tabular, spacing_em))


@lru_cache(maxsize=256)
def _font(size: float, weight: int, mono: bool, tabular: bool, spacing_em: float) -> QtGui.QFont:
    result = QtGui.QFont(family(mono))
    if float(size).is_integer():
        result.setPixelSize(int(size))
    else:
        screen = QtGui.QGuiApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() if screen else 96.0
        result.setPointSizeF(size * 72.0 / dpi)
    result.setWeight(WEIGHTS[weight])
    # Chrome renders with slight (vertical-only) hinting and unhinted, fractional advances.
    result.setHintingPreference(QtGui.QFont.HintingPreference.PreferVerticalHinting)
    if tabular:
        for tag in ("tnum", "lnum"):
            feature = QtGui.QFont.Tag.fromString(tag)
            if feature is not None:
                result.setFeature(feature, 1)
    if spacing_em:
        result.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, spacing_em * size)
    return result


def snap(value: float) -> int:
    """A box edge or icon position as a browser snaps it to whole pixels (half up)."""
    return math.floor(value + 0.5)


def baseline(top: float, line_height: float, ascent: float, descent: float) -> float:
    """Baseline of one line in a CSS line box starting at ``top``, placed as Chrome places it: the
    font's ascent and descent are rounded to whole pixels and the half-leading above is floored.
    Painting then rounds the baseline half up, in Qt as in Chrome."""
    rounded_ascent, rounded_descent = snap(ascent), snap(descent)
    return top + rounded_ascent + math.floor((line_height - rounded_ascent - rounded_descent) / 2)


# ---------- Palette and style sheet ----------


def palette(theme: Theme) -> QtGui.QPalette:
    result = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    for role, token in (
        (roles.Window, "surface.canvas"),
        (roles.Base, "surface.card"),
        (roles.AlternateBase, "surface.sunken"),
        (roles.WindowText, "text.body"),
        (roles.Text, "text.body"),
        (roles.Button, "surface.card"),
        (roles.ButtonText, "text.ink"),
        (roles.Highlight, "surface.selected"),
        (roles.HighlightedText, "text.ink"),
        (roles.Link, "text.ink"),
        (roles.LinkVisited, "text.ink"),
        (roles.ToolTipBase, "overlay.tooltip"),
        (roles.ToolTipText, "overlay.tooltipText"),
        (roles.PlaceholderText, "text.tertiary"),
        (roles.Light, "surface.card"),
        (roles.Midlight, "surface.sunken"),
        (roles.Mid, "border.line"),
        (roles.Dark, "border.strong"),
        (roles.Shadow, "text.ink"),
        (roles.BrightText, "text.onInk"),
        (roles.Accent, "accent.primary"),
    ):
        result.setColor(role, theme.color(token))
    for role in (roles.WindowText, roles.Text, roles.ButtonText, roles.PlaceholderText):
        result.setColor(QtGui.QPalette.ColorGroup.Disabled, role, theme.color("text.disabled"))
    return result


def stylesheet(theme: Theme) -> str:
    """Rules for plain widgets; painted components draw themselves (see ``gui_widgets``)."""
    t = theme.qss
    return f"""
    * {{ outline: 0; }}
    QMainWindow, QDialog {{ background: {t("surface.canvas")}; }}
    QToolTip {{ background: {t("overlay.tooltip")}; color: {t("overlay.tooltipText")}; border: 0;
        padding: 7px 10px; font-size: 12px; }}
    QScrollArea {{ border: 0; background: transparent; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle {{ background: {t("border.strong")}; border-radius: 3px; min-height: 28px;
        min-width: 28px; }}
    QScrollBar::handle:hover {{ background: {t("text.tertiary")}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    """ + _component_stylesheet(theme)


def _component_stylesheet(theme: Theme) -> str:
    """Inputs and tables of the design."""
    t = theme.qss
    return f"""
    QLineEdit, QPlainTextEdit, QTextEdit {{ background: {t("surface.card")}; color: {t("text.ink")};
        border: 1px solid {t("border.strong")}; border-radius: 6px;
        placeholder-text-color: {t("text.tertiary")};
        selection-background-color: {t("accent.soft")}; selection-color: {t("text.ink")}; }}
    QLineEdit {{ min-height: 30px; max-height: 30px; padding: 0 8px; }}
    QPlainTextEdit, QTextEdit {{ padding: 6px 8px; }}
    QTextEdit#notes {{ padding: 7px 10px 8px 10px; }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {t("text.ink")}; }}
    QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{ background: {t("surface.sunken")};
        color: {t("text.disabled")}; border-color: {t("border.line")}; }}
    QLineEdit[valueField="true"] {{ padding: 0 6px 0 8px; }}
    QLineEdit#unitEdit {{ border: 0; padding: 0; background: transparent; min-height: 0;
        max-height: 16777215px; }}
    QAbstractSpinBox[muted="true"]:disabled {{ color: {t("text.tertiary")}; }}
    QComboBox QAbstractItemView {{ background: {t("surface.card")}; color: {t("text.ink")};
        border: 1px solid {t("border.line")}; padding: 4px; outline: 0;
        selection-background-color: {t("surface.hover")}; selection-color: {t("text.ink")}; }}
    QComboBox QAbstractItemView::item {{ min-height: 32px; padding: 0 10px; border-radius: 6px; }}
    QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected {{
        background: {t("surface.hover")}; color: {t("text.ink")}; }}
    QTableView[designTable="true"] {{ background: transparent; border: 0; border-radius: 0; }}
    QHeaderView[designTable="true"] {{ background: transparent; border: 0; border-radius: 0;
        padding: 0; }}
    """


# ---------- Proxy style ----------


class MainStyle(QtWidgets.QProxyStyle):
    """Fusion with the design's check boxes and radio buttons and without Qt focus rectangles."""

    def __init__(self, themes: "ThemeManager") -> None:
        super().__init__("Fusion")
        self.themes = themes

    def pixelMetric(
        self,
        metric: QtWidgets.QStyle.PixelMetric,
        option: QtWidgets.QStyleOption | None = None,
        widget: QtWidgets.QWidget | None = None,
    ) -> int:
        metrics = QtWidgets.QStyle.PixelMetric
        if metric in (
            metrics.PM_IndicatorWidth,
            metrics.PM_IndicatorHeight,
            metrics.PM_ExclusiveIndicatorWidth,
            metrics.PM_ExclusiveIndicatorHeight,
        ):
            return 16
        if metric in (metrics.PM_CheckBoxLabelSpacing, metrics.PM_RadioButtonLabelSpacing):
            return 8
        return int(super().pixelMetric(metric, option, widget))

    def drawPrimitive(
        self,
        element: QtWidgets.QStyle.PrimitiveElement,
        option: QtWidgets.QStyleOption,
        painter: QtGui.QPainter,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        primitives = QtWidgets.QStyle.PrimitiveElement
        styled: Any = option  # the stubs omit QStyleOption.state and .rect
        if element == primitives.PE_FrameFocusRect:
            return  # the FocusRing overlay draws keyboard focus
        if element in (primitives.PE_IndicatorCheckBox, primitives.PE_IndicatorItemViewItemCheck):
            state = styled.state
            mark: Mark = (
                "mixed"
                if state & QtWidgets.QStyle.StateFlag.State_NoChange
                else "on"
                if state & QtWidgets.QStyle.StateFlag.State_On
                else "off"
            )
            enabled = bool(state & QtWidgets.QStyle.StateFlag.State_Enabled)
            paint_check(painter, QtCore.QRectF(styled.rect), self.themes.current, mark, enabled)
            return
        if element == primitives.PE_IndicatorRadioButton:
            state = styled.state
            paint_radio(
                painter,
                QtCore.QRectF(styled.rect),
                self.themes.current,
                bool(state & QtWidgets.QStyle.StateFlag.State_On),
                bool(state & QtWidgets.QStyle.StateFlag.State_Enabled),
            )
            return
        super().drawPrimitive(element, option, painter, widget)


Mark = Literal["off", "on", "mixed"]


def check_mark(state: QtCore.Qt.CheckState) -> Mark:
    if state == QtCore.Qt.CheckState.Checked:
        return "on"
    return "mixed" if state == QtCore.Qt.CheckState.PartiallyChecked else "off"


def border_width(painter: QtGui.QPainter, width: float) -> float:
    """A CSS border width as browsers draw it: wider than one device pixel, it is floored to whole
    device pixels (the 1.5 px borders of checks and radios are 1 px at a device pixel ratio of 1)."""
    device = painter.device()
    ratio = device.devicePixelRatioF() if device is not None else 1.0
    return max(1.0, math.floor(width * ratio)) / ratio


def paint_check(
    painter: QtGui.QPainter, rect: QtCore.QRectF, theme: Theme, mark: Mark, enabled: bool = True
) -> None:
    """The 16 px check box (``.check``) centred in ``rect``: off, on (check mark), mixed (bar) or disabled.
    As in a browser the box and the bar snap to whole pixels, while the rotated mark keeps the fraction."""
    cx, cy = rect.center().x(), rect.center().y()
    box = QtCore.QRectF(snap(cx - 8), snap(cy - 8), 16, 16)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    filled = mark != "off" and enabled
    fill = (
        theme.color("text.ink")
        if filled
        else theme.color("surface.card" if enabled else "surface.sunken")
    )
    border = (
        theme.color("text.ink")
        if filled
        else theme.color("control.checkBorder" if enabled else "control.checkBorderDisabled")
    )
    if mark != "off" and not enabled:
        fill = border = theme.color("control.checkBorderDisabled")
    line = border_width(painter, 1.5)
    painter.setPen(QtGui.QPen(border, line))
    painter.setBrush(fill)
    half = line / 2
    painter.drawRoundedRect(box.adjusted(half, half, -half, -half), 4 - half, 4 - half)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(theme.color("text.onInk") if enabled else theme.color("surface.card"))
    if mark == "on":
        # CSS: a 9×5 box with 2 px left and bottom borders (11×7 in all: ``* { box-sizing }`` does not
        # reach ``::after``), centred in the 13 px content box and snapped like any box, then raised 1 px
        # and turned −45° about its centre; one L-shaped polygon.
        mark_left, mark_top = math.floor(cx - 5), math.floor(cy - 3)
        painter.translate(mark_left + 5.5, mark_top + 3.5 - 1)
        painter.rotate(-45)
        corners = ((-5.5, -3.5), (-3.5, -3.5), (-3.5, 1.5), (5.5, 1.5), (5.5, 3.5), (-5.5, 3.5))
        painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in corners]))
    elif mark == "mixed":
        painter.drawRoundedRect(QtCore.QRectF(snap(cx - 4), snap(cy - 1), 8, 2), 1, 1)
    painter.restore()


def paint_radio(
    painter: QtGui.QPainter, rect: QtCore.QRectF, theme: Theme, on: bool, enabled: bool = True
) -> None:
    """The 16 px radio button (``.radio``): ink border and an 8 px ink dot when on."""
    box = QtCore.QRectF(rect.center().x() - 8, rect.center().y() - 8, 16, 16)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    border = (
        theme.color("text.ink")
        if on
        else theme.color("control.checkBorder" if enabled else "control.checkBorderDisabled")
    )
    line = border_width(painter, 1.5)
    half = line / 2
    painter.setPen(QtGui.QPen(border, line))
    painter.setBrush(theme.color("surface.card" if enabled else "surface.sunken"))
    painter.drawEllipse(box.adjusted(half, half, -half, -half))
    if on:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("text.ink"))
        painter.drawEllipse(box.center(), 4, 4)
    painter.restore()


# ---------- Theme manager ----------


class ThemeManager(QtCore.QObject):
    """Owns the active theme: stored choice, application palette, style and style sheet."""

    changed = QtCore.Signal(object)

    def __init__(self, settings: QtCore.QSettings, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        stored = settings.value(THEME_KEY)
        if stored not in THEMES:
            # First launch: follow the desktop; afterwards the stored choice wins.
            hints = QtGui.QGuiApplication.styleHints()
            stored = "dark" if hints.colorScheme() == QtCore.Qt.ColorScheme.Dark else "light"
        self.current: Theme = THEMES[str(stored)]
        self.style = MainStyle(self)

    def set(self, name: str) -> None:
        if name not in THEMES:
            raise ValueError(f"Unknown theme: {name}")
        if name != self.current.name:
            self.current = THEMES[name]
            self.settings.setValue(THEME_KEY, name)
            self.apply()

    def apply(self) -> None:
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.styleHints().setColorScheme(
                QtCore.Qt.ColorScheme.Dark if self.current.dark else QtCore.Qt.ColorScheme.Light
            )
            if application.style() is not self.style:
                application.setStyle(self.style)
            application.setFont(font(13))
            application.setPalette(palette(self.current))
            application.setStyleSheet(stylesheet(self.current))
        self.changed.emit(self.current)
