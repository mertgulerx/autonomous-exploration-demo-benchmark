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

"""New campaign step 1 (03, 04; HANDOFF §8.3-§8.4): the robot cards and the profiles popup.

``RobotSelector`` keeps the selection rules: the last ticked available robot cannot be cleared, each ticked
robot keeps at least one profile, the popup lists the first 250 saved profiles (a footer row says so when the
library has more) and Edit opens the profile editor. A saved profile gets a new ID with each edit; the selection
follows it through the profile's stable ``family_id``.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_controls import BadgeLook, paint_badge
from .gui_menus import show_tooltip
from .gui_profile_preview import BASELINE, ProfilePreview, profile_key as _key
from .gui_style import humanize
from .gui_support import Transport
from .gui_theme import Mark, ThemeManager, paint_check, snap
from .gui_widgets import (
    BODY,
    Button,
    PopupCard,
    PopupHost,
    TextLine,
    TextStyle,
    Themed,
    css_rect,
    draw_text,
    paint_glyphs,
    paint_scroll_handle,
    place,
    plural,
    rounded_path,
    wheel_step,
)
from .gui_wizard import (
    CardGrid,
    PopupButton,
    PopupHead,
    Reference,
    ToggleCard,
    search_text,
    toggle_reference,
    toggle_shown,
)

ROBOT_IMAGES = Path(__file__).with_name("assets") / "robots"

NAME = TextStyle(14, 600, 18)
SPEC_KEY = TextStyle(11.5, 400, 18)
SPEC_VALUE = TextStyle(14, 600, 18, tabular=True)
SPEC_UNIT = TextStyle(12, 400, 18)
DRIVE = TextStyle(12, 400, 18)
PILL = TextStyle(12, 600, 18, tabular=True)
REASON = TextStyle(12.5, 400, 18)
BAR_TEXT = TextStyle(12, 500, 18)
CURRENT_NAME = TextStyle(13, 600, 18)
UNAVAILABLE = BadgeLook("Unavailable")
MORE_NOTE = "First 250 saved profiles shown. Reference others by ID in an imported campaign YAML."
LAST_PROFILE = "At least one profile is required. Select another profile first."
LAST_ROBOT = "Select another robot before deselecting this one."


def robot_columns(width: float) -> int:
    """4 columns from 1240 px of content, then 3, 2 and 1 (HANDOFF §8.3)."""
    return 1 if width < 620 else 2 if width < 940 else 3 if width < 1240 else 4


def _label(robot_id: str, reference: Reference) -> str:
    if _key(robot_id, reference) == "default":
        return "Default"
    if isinstance(reference, dict):
        return str(reference.get("name") or "Campaign snapshot")
    return str(reference)


def _created(row: dict[str, Any]) -> tuple[bool, int]:
    """Sort key: oldest profile first; rows without a creation time last."""
    created = row.get("created_ns")
    return (False, created) if isinstance(created, int) else (True, 0)


def number_text(value: Any) -> str:
    """A catalog figure as the cards show it ("10", "0.8"); "—" when unavailable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{float(value):g}"


def drive_text(robot: dict[str, Any]) -> str:
    """The drive type (D5: engines before E2 do not send it) and the size of the footprint's bounding box:
    "Differential · 0.44 × 0.38 m"."""
    parts = []
    kinematics = robot.get("kinematics")
    if isinstance(kinematics, str) and kinematics:
        parts.append(humanize(kinematics))
    try:
        xs = [float(point[0]) for point in robot.get("footprint") or ()]
        ys = [float(point[1]) for point in robot.get("footprint") or ()]
    except (TypeError, ValueError, IndexError):
        xs = ys = []
    if xs and ys and all(math.isfinite(value) for value in (*xs, *ys)):
        parts.append(f"{max(xs) - min(xs):.2f} × {max(ys) - min(ys):.2f} m")
    return " · ".join(parts)


# ---------- photos ----------


@lru_cache(maxsize=16)
def faded_photo(robot_id: str) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """The packaged render with its clipped floor shadow faded out, as ``tools/fade_robots.py`` fades the
    mockup photos (alpha times a ramp from the body's solid bounding box to the image edges, to the power
    1.6), in premultiplied float RGBA; None without an image."""
    image = QtGui.QImage(str(ROBOT_IMAGES / (robot_id + ".png")))
    if image.isNull():
        return None
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(height, image.bytesPerLine())
    rgba = pixels[:, : width * 4].reshape(height, width, 4).astype(np.int64)
    alpha = rgba[:, :, 3]
    solid = np.argwhere(alpha > 235)
    if solid.size:
        left, right = int(solid[:, 1].min()), int(solid[:, 1].max()) + 1
        bottom = int(solid[:, 0].max()) + 1
        fade_y, fade_left, fade_right = max(6, height - bottom), max(6, left), max(6, width - right)
        ys = np.arange(height, dtype=np.float64)[:, None]
        xs = np.arange(width, dtype=np.float64)[None, :]
        down = np.where(ys <= bottom, 1.0, np.maximum(0.0, 1 - (ys - bottom) / fade_y))
        across = np.where(
            xs < left, xs / fade_left, np.where(xs > right, (width - 1 - xs) / fade_right, 1.0)
        )
        mask = np.round(255 * np.minimum(down, across) ** 1.6).astype(np.int64)
        alpha = alpha * mask // 255
    result = np.empty((height, width, 4), dtype=np.float64)
    result[:, :, :3] = rgba[:, :, :3] * (alpha[:, :, None] / 255.0)
    result[:, :, 3] = alpha
    return result


def photo_aspect(robot_id: str) -> float | None:
    rgba = faded_photo(robot_id)
    return None if rgba is None else rgba.shape[1] / rgba.shape[0]


def _area_weights(
    texels: int, start: float, length: float, first: int, count: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Box-filter weights (count × texels): device pixel ``first + i`` averages the texels it covers when
    ``texels`` texels span [start, start + length); partly covered edge pixels get part of the weight."""
    scale = texels / length
    edges = np.arange(first, first + count + 1, dtype=np.float64)
    low = np.clip((edges[:-1] - start) * scale, 0.0, texels)
    high = np.clip((edges[1:] - start) * scale, 0.0, texels)
    index = np.arange(texels, dtype=np.float64)
    overlap = np.minimum(high[:, None], index + 1) - np.maximum(low[:, None], index)
    return np.asarray(np.clip(overlap, 0.0, None) / scale, dtype=np.float64)


PHOTO_STEPS = 16  # photos are resampled at 1/16 device-pixel offsets: at most 256 images per robot


def photo_image(
    robot_id: str, x: float, y: float, height: float, ratio: float
) -> tuple[QtGui.QImage, QtCore.QPointF] | None:
    """The faded photo ``height`` px high (width by aspect) with its corner at the fractional (x, y),
    area-sampled into whole device pixels: the image and where to draw it. The image depends only on the
    position's fraction of a device pixel (rounded to 1/16), so moving a card reuses it."""
    left, top = x * ratio, y * ratio
    first_x, first_y = math.floor(left), math.floor(top)
    fraction_x = round((left - first_x) * PHOTO_STEPS) / PHOTO_STEPS
    fraction_y = round((top - first_y) * PHOTO_STEPS) / PHOTO_STEPS
    image = _photo_at(robot_id, fraction_x, fraction_y, height, ratio)
    return None if image is None else (image, QtCore.QPointF(first_x / ratio, first_y / ratio))


@lru_cache(maxsize=128)
def _photo_at(
    robot_id: str, left: float, top: float, height: float, ratio: float
) -> QtGui.QImage | None:
    """The scaled photo starting ``left`` and ``top`` device pixels into its first pixel."""
    rgba = faded_photo(robot_id)
    if rgba is None:
        return None
    texel_rows, texel_columns = rgba.shape[:2]
    width = height * texel_columns / texel_rows * ratio
    columns, rows = math.ceil(left + width), math.ceil(top + height * ratio)
    across = _area_weights(texel_columns, left, width, 0, columns)
    down = _area_weights(texel_rows, top, height * ratio, 0, rows)
    scaled = np.tensordot(np.tensordot(down, rgba, axes=(1, 0)), across, axes=(1, 1))
    data = np.ascontiguousarray(
        np.clip(np.round(scaled.transpose(0, 2, 1)), 0, 255).astype(np.uint8)
    )
    image = QtGui.QImage(
        data.tobytes(),
        columns,
        rows,
        columns * 4,
        QtGui.QImage.Format.Format_RGBA8888_Premultiplied,
    ).copy()
    image.setDevicePixelRatio(ratio)
    return image


@lru_cache(maxsize=16)
def glow_image(width: int, height: int) -> QtGui.QImage:
    """The dark theme's light behind a photo (DESIGN §4): ``radial-gradient(closest-side, rgba(255, 255,
    255, .10), transparent 92%)`` over a ``width``×``height`` device-pixel box, at pixel centres."""
    ys = (np.arange(height, dtype=np.float64) + 0.5 - height / 2) / (height / 2)
    xs = (np.arange(width, dtype=np.float64) + 0.5 - width / 2) / (width / 2)
    distance = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
    value = np.round(255 * 0.10 * np.clip(1 - distance / 0.92, 0.0, 1.0)).astype(np.uint8)
    data = np.ascontiguousarray(np.repeat(value[:, :, None], 4, axis=2))  # premultiplied white
    return QtGui.QImage(
        data.tobytes(), width, height, width * 4, QtGui.QImage.Format.Format_RGBA8888_Premultiplied
    ).copy()


# ---------- robot cards ----------


class RobotCard(ToggleCard):
    """``.rc`` (03): the check and the name (14 px 600), the photo centred in a 132 px area (dark: on a soft
    light), Lidar, Linear and Angular, the drive line and, after a rule, the footer: the profile names
    (unticked), Profiles with the count pill and the names (ticked) or the engine's reason (unavailable).
    Selected: 1 px ink border; hover: ``border.strong``."""

    # Offsets from the card's CSS box (1 px border plus padding 16 20 14); FOOTER runs from the footer's
    # rule to the card's bottom edge (41 px, the 14 px padding and the border).
    PAD, HEAD_TOP, IMAGE_TOP, IMAGE_AREA, PHOTO = 21, 17, 49, 132, 124
    SPECS_TOP, DRIVE_TOP, FOOTER = 193, 236, 56

    def __init__(
        self, themes: ThemeManager, robot: dict[str, Any], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.robot = deepcopy(robot)
        self.robot_id, self.label = str(robot["id"]), str(robot["label"])
        self.available = bool(robot.get("available", True))
        self.reason = str(robot.get("reason") or "Required simulation packages are unavailable.")
        self.specs = (
            ("Lidar", number_text(robot.get("lidar_range")), "m"),
            ("Linear", number_text(robot.get("movement_speed")), "m/s"),
            ("Angular", number_text(robot.get("turning_speed")), "rad/s"),
        )
        self.drive = drive_text(robot)
        self.checked = self.guarded = False
        self.count, self.names = 1, "Default"
        self.button = PopupButton(themes, "Profiles", self)
        self.button.setAccessibleName("Select explorer profiles for " + self.label)
        self.button.hide()
        self._describe()

    def set_state(self, checked: bool, guarded: bool, count: int, names: str) -> None:
        """Ticked or not, whether it is the last ticked robot, and its profiles."""
        state = (checked and self.available, guarded, count, names)
        if state == (self.checked, self.guarded, self.count, self.names):
            return
        self.checked, self.guarded, self.count, self.names = state
        self.button.setVisible(self.checked)
        self._place_button()
        self._describe()
        self.update()

    def _describe(self) -> None:
        self.setAccessibleName(
            f"Select robot: {self.label}, {'selected' if self.checked else 'not selected'}"
        )
        detail = " · ".join(f"{key} {value} {unit}" for key, value, unit in self.specs)
        if self.drive:
            detail += ". " + self.drive
        if not self.available:
            detail += ". Unavailable: " + self.reason
        self.setAccessibleDescription(detail)

    def footer_top(self) -> float:
        return css_rect(self).bottom() - self.FOOTER

    def _place_button(self) -> None:
        left = css_rect(self).x() + self.PAD
        place(self.button, left, self.footer_top() + 13, self.button.natural_width(), 28)

    def css_changed(self) -> None:
        self._place_button()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_button()

    def pill_box(self) -> QtCore.QRectF:
        x = css_rect(self).x() + self.PAD + self.button.natural_width() + 8
        width = max(22.0, 12 + PILL.advance(str(self.count)))
        return QtCore.QRectF(x, self.footer_top() + 13 + 3, width, 22)

    # tooltips

    def tooltip_at(self, position: QtCore.QPoint) -> str:
        box = css_rect(self)
        left, top, width = box.x(), box.y(), box.width()
        footer = self.footer_top()
        if position.y() < top + self.IMAGE_TOP:
            return LAST_ROBOT if self.guarded else ""
        if position.y() < footer:
            return ""
        if not self.available:
            return self.reason
        if self.checked and self.pill_box().contains(QtCore.QPointF(position)):
            return plural(self.count, "profile") + " selected"
        names_x = self.pill_box().right() + 10 if self.checked else left + self.PAD
        room = left + width - self.PAD - names_x
        if position.x() >= names_x and BODY.advance(self.names) > room:
            return self.names
        return ""

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            show_tooltip(self, event.globalPos(), self.tooltip_at(event.pos()))
            return True
        return super().event(event)

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.theme
        box = css_rect(self)
        left, top, width = box.x(), box.y(), box.width()
        hovered = self.underMouse() and self.available
        border = "text.ink" if self.checked else "border.strong" if hovered else "border.line"
        painter.setPen(QtGui.QPen(theme.color(border), 1))
        painter.setBrush(theme.color("surface.card"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7.5)
        )
        x, inner = left + self.PAD, width - 2 * self.PAD
        ink = theme.color("text.ink" if self.available else "text.tertiary")
        # Head: check, name and, for an unavailable robot, the chip at the right.
        y = top + self.HEAD_TOP
        mark: Mark = "on" if self.checked else "off"
        paint_check(painter, QtCore.QRectF(x, y + 3, 16, 16), theme, mark, self.available)
        name_right = x + inner
        if not self.available:
            badge_x = x + inner - UNAVAILABLE.width
            paint_badge(painter, badge_x, y, UNAVAILABLE, theme)
            name_right = badge_x - 10
        name_x = x + 16 + 10
        name = NAME.elide(self.label, max(0.0, name_right - name_x))
        draw_text(painter, name_x, y + 2, name, NAME, ink)
        self._paint_photo(painter, x, top + self.IMAGE_TOP, inner)
        # Lidar, Linear, Angular: key above the value and its muted unit.
        column = inner / 3
        for index, (key, value, unit) in enumerate(self.specs):
            column_x = x + index * column
            secondary = theme.color("text.secondary")
            draw_text(painter, column_x, top + self.SPECS_TOP, key, SPEC_KEY, secondary)
            value_top = top + self.SPECS_TOP + 18 + 1
            advance = draw_text(painter, column_x, value_top, value, SPEC_VALUE, ink)
            if value != "—":
                painter.save()
                painter.setPen(theme.color("text.tertiary"))
                baseline = SPEC_VALUE.baseline(value_top)
                paint_glyphs(painter, column_x + advance + 2, baseline, unit, SPEC_UNIT)
                painter.restore()
        if self.drive:
            drive = DRIVE.elide(self.drive, inner)
            draw_text(painter, x, top + self.DRIVE_TOP, drive, DRIVE, theme.color("text.secondary"))
        self._paint_footer(painter, x, inner)

    def _paint_photo(self, painter: QtGui.QPainter, x: float, top: float, inner: float) -> None:
        theme = self.theme
        ratio = self.devicePixelRatioF()
        if theme.name == "dark":
            box = QtCore.QRect(
                snap(x),
                snap(top),
                snap(x + inner) - snap(x),
                snap(top + self.IMAGE_AREA) - snap(top),
            )
            painter.drawImage(
                box, glow_image(round(box.width() * ratio), round(box.height() * ratio))
            )
        aspect = photo_aspect(self.robot_id)
        if aspect is None:
            return
        photo_x = x + (inner - self.PHOTO * aspect) / 2
        photo_y = top + (self.IMAGE_AREA - self.PHOTO) / 2
        found = photo_image(self.robot_id, photo_x, photo_y, float(self.PHOTO), ratio)
        if found is None:
            return
        image, origin = found
        painter.save()
        if not self.available:
            painter.setOpacity(0.38)
        painter.drawImage(origin, image)
        painter.restore()

    def _paint_footer(self, painter: QtGui.QPainter, x: float, inner: float) -> None:
        theme = self.theme
        footer = self.footer_top()
        rule = QtCore.QRect(snap(x), snap(footer), snap(x + inner) - snap(x), 1)
        painter.fillRect(rule, theme.color("border.line"))
        row = footer + 13  # the 28 px content row under the rule and the 12 px padding
        secondary = theme.color("text.secondary")
        if not self.available:
            gui_icons.draw(
                painter, "alert", theme.color("state.warn"), QtCore.QPointF(snap(x), snap(row + 6))
            )
            text_x = x + 16 + 8
            reason = REASON.elide(self.reason, max(0.0, x + inner - text_x))
            draw_text(painter, text_x, row + 5, reason, REASON, secondary)
            return
        names_x = x
        if self.checked:
            pill = self.pill_box()
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("accent.soft"))
            pill_left = snap(pill.left())
            painter.drawPath(
                rounded_path(
                    QtCore.QRectF(pill_left, snap(pill.top()), snap(pill.right()) - pill_left, 22),
                    11,
                )
            )
            count = str(self.count)
            count_x = pill.left() + (pill.width() - PILL.advance(count)) / 2
            draw_text(painter, count_x, pill.top() + 2, count, PILL, theme.color("accent.softText"))
            names_x = pill.right() + 10
        names = BODY.elide(self.names, max(0.0, x + inner - names_x))
        draw_text(painter, names_x, row + 5, names, BODY, secondary)


# ---------- profiles popup ----------


@dataclass(frozen=True, slots=True)
class ProfileEntry:
    """A profile a robot can run: its Default, a saved profile, or a campaign's inline snapshot."""

    reference: Reference
    key: str
    name: str
    origin: Literal["default", "library", "snapshot"]

    @property
    def detail(self) -> str:
        """What the profile is, as searched: "Robot-aware baseline", "Saved profile", "Campaign snapshot"."""
        if self.origin == "default":
            return BASELINE
        if self.origin == "snapshot":
            return "Campaign snapshot"
        return "Saved profile"

    @property
    def search(self) -> str:
        return (self.name + " " + self.detail).casefold()


class ProfileList(Themed):
    """``.pp-list`` (04): the sunken header bar (the check of the shown rows, "Profile") over 40 px rows:
    check and name (Default adds "baseline"). The inspected row has the selected
    fill, a 2 px ink bar, its name in 600 and Edit (ghost, 26 px). Hovering or the arrow keys inspect a row; a
    click or Space toggles it; the rows scroll under the bar."""

    toggled = QtCore.Signal(int)
    inspected = QtCore.Signal(int)
    header_clicked = QtCore.Signal()
    edit_requested = QtCore.Signal(int)
    typed = QtCore.Signal(str)

    BAR, ROW = 36, 40

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.entries: list[ProfileEntry] = []
        self.shown: list[int] = []  # entries matching the search, in order
        self.checked: set[str] = set()
        self.current = -1
        self.offset = 0.0
        self.edit = Button(themes, "Edit", icon="pencil", variant="ghost", small=True, parent=self)
        self.edit.setFixedHeight(26)
        self.edit.clicked.connect(lambda: self.edit_requested.emit(self.current))
        self.edit.hide()
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "none")
        self.setAccessibleName("Profiles")

    # state

    def set_entries(self, entries: list[ProfileEntry], query: str) -> None:
        key = self.entries[self.current].key if 0 <= self.current < len(self.entries) else None
        self.entries = entries
        self.current = next((i for i, entry in enumerate(entries) if entry.key == key), -1)
        self.set_filter(query)

    def set_filter(self, query: str) -> None:
        needle = query.casefold().strip()
        self.shown = [i for i, e in enumerate(self.entries) if not needle or needle in e.search]
        self.scroll_to(self.offset)
        self._place_edit()
        self.update()

    def set_checked(self, keys: set[str]) -> None:
        self.checked = set(keys)
        self.update()

    def set_current(self, index: int) -> None:
        if index != self.current:
            self.current = index
            self.ensure_shown(index)
            self._place_edit()
            self.update()

    def header_mark(self) -> Mark:
        on = sum(self.entries[i].key in self.checked for i in self.shown)
        return "off" if not on else "on" if on == len(self.shown) else "mixed"

    # geometry

    def rows_height(self) -> float:
        return float(self.height() - self.BAR)

    def max_offset(self) -> float:
        return max(0.0, len(self.shown) * self.ROW - self.rows_height())

    def scroll_to(self, offset: float) -> None:
        offset = min(max(0.0, offset), self.max_offset())
        if offset != self.offset:
            self.offset = offset
            self._place_edit()
            self.update()

    def ensure_shown(self, index: int) -> None:
        if index not in self.shown or not self.isVisible():
            return  # a hidden list has no height yet; showEvent scrolls
        top = self.shown.index(index) * self.ROW
        if top < self.offset:
            self.scroll_to(top)
        elif top + self.ROW > self.offset + self.rows_height():
            self.scroll_to(top + self.ROW - self.rows_height())

    def row_top(self, slot: int) -> float:
        return self.BAR + slot * self.ROW - self.offset

    def row_content(self, slot: int) -> float:
        """The row's height above its rule (the last row has none)."""
        return self.ROW - (0 if slot == len(self.shown) - 1 else 1)

    def index_at(self, y: float) -> int:
        if y < self.BAR:
            return -1
        slot = int((y - self.BAR + self.offset) // self.ROW)
        return self.shown[slot] if 0 <= slot < len(self.shown) else -1

    def _place_edit(self) -> None:
        if self.current not in self.shown:
            self.edit.hide()
            return
        slot = self.shown.index(self.current)
        top = self.row_top(slot)
        content = self.row_content(slot)
        y = top + (content - 26) / 2
        if y < self.BAR or y + 26 > self.height():
            self.edit.hide()
            return
        width = self.edit.natural_width()
        place(self.edit, self.width() - 20 - width, y, width, 26)
        self.edit.show()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.scroll_to(self.offset)
        self._place_edit()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.ensure_shown(self.current)
        self._place_edit()

    # input

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self.index_at(event.position().y())
        if index >= 0 and index != self.current:
            self.set_current(index)
            self.inspected.emit(index)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        position = event.position()
        if position.y() < self.BAR:
            if position.x() < 20 + 16 + 4 and self.shown:
                self.header_clicked.emit()
            return
        index = self.index_at(position.y())
        if index >= 0:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.toggled.emit(index)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self.scroll_to(self.offset - wheel_step(event, self.ROW).y())
        index = self.index_at(event.position().y())  # the row now under the pointer
        if index >= 0 and index != self.current:
            self.set_current(index)
            self.inspected.emit(index)
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space and self.current in self.shown:
            self.toggled.emit(self.current)
            return
        page = max(1, int(self.rows_height() // self.ROW))
        moves = {
            QtCore.Qt.Key.Key_Up: -1,
            QtCore.Qt.Key.Key_Down: 1,
            QtCore.Qt.Key.Key_PageUp: -page,
            QtCore.Qt.Key.Key_PageDown: page,
            QtCore.Qt.Key.Key_Home: -len(self.shown),
            QtCore.Qt.Key.Key_End: len(self.shown),
        }
        move = moves.get(QtCore.Qt.Key(key))
        if move is not None and self.shown:
            slot = self.shown.index(self.current) if self.current in self.shown else -1
            target = self.shown[min(max(0, slot + move), len(self.shown) - 1)]
            if target != self.current:
                self.set_current(target)
                self.inspected.emit(target)
            return
        text = search_text(event)
        if text:
            self.typed.emit(text)  # type to search
            return
        super().keyPressEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if kind in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()  # the scroll handle shows while the pointer is over the list
        elif kind == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            index = self.index_at(event.pos().y())
            sole = len(self.checked) == 1
            text = ""
            if index >= 0 and sole and self.entries[index].key in self.checked:
                text = LAST_PROFILE if event.pos().x() < 46 else ""
            show_tooltip(self, event.globalPos(), text)
            return True
        return super().event(event)

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        width = self.width()
        line, secondary = theme.color("border.line"), theme.color("text.secondary")
        painter.fillRect(QtCore.QRect(0, 0, width, self.BAR), theme.color("surface.sunken"))
        painter.fillRect(QtCore.QRect(0, self.BAR - 1, width, 1), line)
        bar_top = (self.BAR - 1 - 18) / 2
        paint_check(
            painter, QtCore.QRectF(20, (self.BAR - 1 - 16) / 2, 16, 16), theme, self.header_mark()
        )
        draw_text(painter, 46, bar_top, "Profile", BAR_TEXT, secondary)
        painter.setClipRect(QtCore.QRect(0, self.BAR, width, self.height() - self.BAR))
        if not self.shown:
            draw_text(painter, 20, self.BAR + 10.5, "No matching profiles", BODY, secondary)
        for slot, index in enumerate(self.shown):
            top = self.row_top(slot)
            if top >= self.height() or top + self.ROW <= self.BAR:
                continue
            self._paint_row(painter, index, top, self.row_content(slot))
        painter.setClipping(False)
        total = len(self.shown) * self.ROW
        paint_scroll_handle(painter, self, self.BAR, self.rows_height(), self.offset, total)

    def _paint_row(self, painter: QtGui.QPainter, index: int, top: float, content: float) -> None:
        theme = self.theme
        entry = self.entries[index]
        width = self.width()
        current = index == self.current
        if current:
            painter.fillRect(
                QtCore.QRectF(0, top, width, self.ROW), theme.color("surface.selected")
            )
            painter.fillRect(QtCore.QRectF(0, top, 2, content), theme.color("text.ink"))
        if content < self.ROW:
            painter.fillRect(QtCore.QRectF(0, top + content, width, 1), theme.color("border.line"))
        mark: Mark = "on" if entry.key in self.checked else "off"
        paint_check(painter, QtCore.QRectF(20, top + (content - 16) / 2, 16, 16), theme, mark)
        line_top = top + (content - 18) / 2
        edit_shown = current and self.edit.isVisible()
        right = (self.edit.x() + self.edit.shift - 10) if edit_shown else width - 20.0
        style = CURRENT_NAME if current else BODY
        baseline = entry.origin == "default"
        extra = BODY.advance("baseline") + 10 if baseline else 0.0
        name = style.elide(entry.name, max(0.0, right - 46 - extra))
        x = 46 + draw_text(painter, 46, line_top, name, style, theme.color("text.ink"))
        if baseline:
            draw_text(painter, x + 10, line_top, "baseline", BODY, theme.color("text.tertiary"))


class ProfilesPopup(PopupCard):
    """``.pp`` (04): the head (title, search, "Done · N selected"), the profile list (320 px) and the preview
    of the inspected profile; under a rule, a 38 px footer row notes the 250-profile limit when the library
    has more."""

    changed = QtCore.Signal(object)
    edit_profile = QtCore.Signal(object)
    finished = QtCore.Signal()

    HEAD, LIST, NOTE = 53, 320, 38  # HEAD and LIST include their rules; NOTE sits under its own

    def __init__(
        self,
        themes: ThemeManager,
        robot: dict[str, Any],
        metadata: dict[str, Any],
        transport: Transport,
    ) -> None:
        super().__init__(themes)
        self.robot_id, robot_label = str(robot["id"]), str(robot["label"])
        self._options: list[ProfileEntry] = []
        self._selected: list[Reference] = ["Default"]
        self.has_more = False
        self.setAccessibleName("Profiles · " + robot_label)
        self.head = PopupHead(themes, "Profiles · " + robot_label, "Search profiles", self)
        self.search, self.done = self.head.search, self.head.done
        self.list = ProfileList(themes, self)
        self.preview = ProfilePreview(themes, robot, metadata, transport, self)
        self.note = TextLine(themes, MORE_NOTE, BODY, "text.secondary", self)
        self.note.hide()
        self.done.clicked.connect(self.finished.emit)
        self.search.textChanged.connect(self.filter_rows)
        self.list.toggled.connect(self._toggle)
        self.list.inspected.connect(self.inspect)
        self.list.header_clicked.connect(self._toggle_shown)
        self.list.edit_requested.connect(self._edit)
        self.list.typed.connect(self._type)
        self.hide()

    # contents

    def set_has_more(self, has_more: bool) -> None:
        if has_more != self.has_more:
            self.has_more = has_more
            self.note.setVisible(has_more)
            self._layout()
            self.update()

    def refresh(self, options: list[ProfileEntry], selected: list[Reference]) -> None:
        self._options = list(options)
        self._selected = deepcopy(selected) or ["Default"]
        self._populate()

    def _populate(self) -> None:
        entries = list(self._options)
        self.list.set_entries(entries, self.search.text())
        self._sync_selection()
        if self.list.current < 0 or not self.preview.matches(
            self.list.entries[self.list.current].reference
        ):
            selected = self._keys()
            first = next((i for i, e in enumerate(entries) if e.key in selected), -1)
            self.inspect(self.list.current if self.list.current >= 0 else first)
        self.filter_rows(self.search.text())

    def _keys(self) -> set[str]:
        return {self._key(reference) for reference in self._selected}

    def _key(self, reference: Reference) -> str:
        return _key(self.robot_id, reference)

    def _sync_selection(self) -> None:
        self.list.set_checked(self._keys())
        self.head.set_count(len(self._selected))

    def inspect(self, index: int) -> None:
        """Show the profile of row ``index`` in the preview (the row becomes the inspected one)."""
        if 0 <= index < len(self.list.entries):
            entry = self.list.entries[index]
            self.list.set_current(index)
            self.preview.inspect(entry.reference, entry.name)

    def filter_rows(self, query: str) -> None:
        self.list.set_filter(query)
        shown = self.list.shown
        if shown and self.list.current not in shown:
            self.inspect(shown[0])

    # actions

    def _toggle(self, index: int) -> None:
        self.inspect(index)
        selected = toggle_reference(self._selected, self.list.entries[index].reference, self._key)
        if selected is not None:  # None: the last profile stays selected
            self._set_selected(selected)

    def _toggle_shown(self) -> None:
        shown = [self.list.entries[i].reference for i in self.list.shown]
        self._set_selected(toggle_shown(self._selected, shown, self._key))

    def _set_selected(self, selected: list[Reference]) -> None:
        self._selected = selected
        self._sync_selection()
        self.changed.emit(deepcopy(self._selected))

    def _edit(self, index: int) -> None:
        if 0 <= index < len(self.list.entries):
            reference = deepcopy(self.list.entries[index].reference)
            self.finished.emit()
            self.edit_profile.emit(reference)

    def _type(self, text: str) -> None:
        self.search.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self.search.insert(text)

    # layout

    def body_height(self) -> int:
        """The list and preview height (above the footer's rule when the footer shows)."""
        return self.height() - 1 - self.HEAD - (1 + self.NOTE if self.has_more else 0) - 1

    def _layout(self) -> None:
        width, top, body = self.width(), 1 + self.HEAD, self.body_height()
        self.head.setGeometry(1, 1, width - 2, self.HEAD - 1)
        self.list.setGeometry(1, top, self.LIST - 1, body)
        self.preview.setGeometry(self.LIST + 1, top, width - self.LIST - 2, body)
        self.note.setGeometry(21, top + body + 1, width - 42, self.NOTE)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        line = self.color("border.line")
        width = self.width()
        painter.fillRect(QtCore.QRect(1, self.HEAD, width - 2, 1), line)
        painter.fillRect(QtCore.QRect(self.LIST, 1 + self.HEAD, 1, self.body_height()), line)
        if self.has_more:
            painter.fillRect(
                QtCore.QRect(1, 1 + self.HEAD + self.body_height(), width - 2, 1), line
            )


# ---------- selector ----------


class RobotSelector(QtWidgets.QWidget):
    """Step 1: the robot card grid and one profiles popup per robot (opened on the window's PopupHost).

    Only available robots are ticked: a robot the catalog reports unavailable, or an imported campaign names
    while it is unavailable, is left out (its profiles are kept) and ``notice`` says so."""

    changed = QtCore.Signal()
    edit_profile = QtCore.Signal(str, object)
    notice = QtCore.Signal(str)

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes = themes
        self.transport = transport
        self._metadata: dict[str, Any] = {}
        self._catalog: list[dict[str, Any]] = []
        self._cards: dict[str, RobotCard] = {}
        self._selected: dict[str, bool] = {}
        self._references: dict[str, list[Reference]] = {}
        self._saved: dict[str, dict[str, dict[str, Any]]] = {}
        self._has_more: dict[str, bool] = {}
        self._initialized = False
        self._open_popups: dict[str, ProfilesPopup] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.grid = CardGrid(robot_columns, 330)
        outer.addWidget(self.grid)

    def set_catalog(
        self, robots: list[dict[str, Any]], metadata: dict[str, Any] | None = None
    ) -> None:
        initial = not self._initialized
        if metadata is not None and metadata != self._metadata:
            self._metadata = deepcopy(metadata)
            for identity in tuple(self._open_popups):
                self._drop_popup(identity)
        self._catalog = deepcopy(robots)
        identities = {str(robot["id"]) for robot in robots}
        left_out = []
        for robot in robots:
            identity = str(robot["id"])
            available = bool(robot.get("available", True))
            self._references.setdefault(identity, ["Default"])
            if identity not in self._selected:
                self._selected[identity] = initial and identity == "mogi" and available
            elif self._selected[identity] and not available:
                self._selected[identity] = False
                left_out.append(str(robot["label"]))
        for identity in tuple(self._cards):
            if identity not in identities:
                self._cards.pop(identity).deleteLater()
                self._drop_popup(identity)
        for robot in robots:
            identity = str(robot["id"])
            card = self._cards.get(identity)
            if card is not None and card.robot != robot:
                card.deleteLater()
                self._drop_popup(identity)
                card = None
            if card is None:
                card = RobotCard(self.themes, robot)
                card.toggled.connect(lambda rid=identity: self._toggle_robot(rid))
                card.button.clicked.connect(
                    lambda checked=False, rid=identity: self._show_profiles(rid)
                )
                self._cards[identity] = card
            self._sync_card(identity)
        self._initialized = True
        self._layout_cards()
        if left_out:
            self.notice.emit(self._left_out_text(left_out))
            self.changed.emit()

    def selections(self) -> dict[str, list[Reference]]:
        return {
            identity: deepcopy(self._references.get(identity, ["Default"]))
            for identity in (str(robot["id"]) for robot in self._catalog)
            if self._selected.get(identity, False)
        }

    def set_selections(self, mapping: dict[str, list[Reference]]) -> None:
        values = deepcopy(mapping)
        robots = {str(robot["id"]): robot for robot in self._catalog}
        if not set(robots).intersection(values) and self.selections():
            return
        left_out = []
        for identity, robot in robots.items():
            references = values.get(identity)
            if references is None:
                self._selected[identity] = False
                continue
            self._references[identity] = deepcopy(references) or ["Default"]
            self._selected[identity] = bool(robot.get("available", True))
            if not self._selected[identity]:
                left_out.append(str(robot["label"]))
        for identity in self._cards:
            self._sync_card(identity)
        if left_out:
            self.notice.emit(self._left_out_text(left_out))

    @staticmethod
    def _left_out_text(labels: list[str]) -> str:
        names = ", ".join(labels)
        verb = "is" if len(labels) == 1 else "are"
        return f"{names} {verb} unavailable in this engine and left out of the campaign."

    def set_profiles(
        self, robot_id: str, rows: list[dict[str, Any]], has_more: bool = False
    ) -> None:
        previous = self._saved.get(robot_id, {})
        current: dict[str, dict[str, Any]] = {}
        for row in rows:
            identity = str(row.get("id", ""))
            if not identity or identity in ("Default", "default-" + robot_id):
                continue
            current[identity] = deepcopy(row)
        selected_ids = {
            reference
            for reference in self._references.get(robot_id, [])
            if isinstance(reference, str)
            and reference not in ("Default", "default", "default-" + robot_id)
        }
        # An edit saves the profile under a new ID and deletes the old row: the selection follows its family.
        families = {
            row["family_id"]: identity
            for identity, row in current.items()
            if isinstance(row.get("family_id"), str)
        }
        references = self._references.get(robot_id, [])
        followed = False
        for position, reference in enumerate(references):
            if not isinstance(reference, str) or reference not in selected_ids - set(current):
                continue
            family = previous.get(reference, {}).get("family_id")
            replacement = families.get(family) if isinstance(family, str) else None
            if replacement is not None and replacement not in references:
                references[position] = replacement
                selected_ids.discard(reference)
                followed = True
        for identity in selected_ids.difference(current):
            if identity in previous:
                current[identity] = deepcopy(previous[identity])
        self._saved[robot_id] = current
        self._has_more[robot_id] = has_more
        self._sync_card(robot_id)
        if followed:
            self.changed.emit()

    def add_profile(self, robot_id: str, reference: Reference, name: str) -> None:
        profile_reference = deepcopy(reference)
        if isinstance(profile_reference, str) and profile_reference not in (
            "Default",
            "default",
            "default-" + robot_id,
        ):
            self._saved.setdefault(robot_id, {}).setdefault(
                profile_reference, {"id": profile_reference, "name": name}
            )
        elif isinstance(profile_reference, dict):
            identity = _key(robot_id, profile_reference)
            if identity.startswith("profile:"):
                key = identity.removeprefix("profile:")
                self._saved.setdefault(robot_id, {}).setdefault(
                    key,
                    {"id": key, "name": name or profile_reference.get("name", "Campaign snapshot")},
                )
        self._selected[robot_id] = True
        self._append_reference(robot_id, profile_reference)
        self._sync_card(robot_id)
        self.changed.emit()

    def forget_profiles(self, robot_id: str, profile_ids: set[str]) -> None:
        """Drop deleted saved profiles from this robot's selection and saved rows."""
        references = self._references.get(robot_id, [])
        kept = [r for r in references if not (isinstance(r, str) and r in profile_ids)]
        saved = self._saved.get(robot_id, {})
        removed = len(kept) != len(references) or bool(profile_ids & set(saved))
        for identity in profile_ids:
            saved.pop(identity, None)
        if not removed:
            return
        self._references[robot_id] = kept or ["Default"]
        self._sync_card(robot_id)
        self.changed.emit()

    def replace_profile(
        self, robot_id: str, old_reference: Reference, profile: dict[str, Any]
    ) -> None:
        old_key = _key(robot_id, old_reference)
        self._references[robot_id] = [
            reference
            for reference in self._references.get(robot_id, ["Default"])
            if _key(robot_id, reference) != old_key
        ]
        profile_id = profile.get("id")
        profile_name = str(profile.get("name", "Profile"))
        if isinstance(profile_id, str) and profile_id:
            self._saved.setdefault(robot_id, {}).setdefault(
                profile_id,
                {
                    "id": profile_id,
                    "name": profile_name,
                    "family_id": profile.get("family_id"),
                    "created_ns": profile.get("created_ns"),
                },
            )
            self.add_profile(robot_id, profile_id, profile_name)
        else:
            self.add_profile(robot_id, deepcopy(profile), profile_name)

    def profile_label(self, robot_id: str, reference: Reference) -> str:
        """The profile's name as the summary and review show it."""
        key = _key(robot_id, reference)
        entry = next((e for e in self._profile_entries(robot_id) if e.key == key), None)
        return entry.name if entry is not None else _label(robot_id, reference)

    def profile_names(self, robot_id: str) -> str:
        """The selected profiles' names as the card's footer lists them ("Default, Low Threshold")."""
        names = {entry.key: entry.name for entry in self._profile_entries(robot_id)}
        return ", ".join(
            names.get(_key(robot_id, reference), _label(robot_id, reference))
            for reference in self._references.get(robot_id, ["Default"])
        )

    def profile_options(self) -> list[tuple[str, str, Reference, str]]:
        """Every robot's profiles for the Results filter: robot ID and label, reference, and a label (a
        campaign's inline snapshot adds "· campaign snapshot")."""
        options: list[tuple[str, str, Reference, str]] = []
        for robot in self._catalog:
            identity = str(robot["id"])
            for entry in self._profile_entries(identity):
                label = entry.name
                if entry.origin == "snapshot" and not label.endswith(" · campaign snapshot"):
                    label += " · campaign snapshot"
                options.append((identity, str(robot["label"]), deepcopy(entry.reference), label))
        return options

    def _toggle_robot(self, robot_id: str) -> None:
        self._set_robot(robot_id, not self._selected.get(robot_id, False))

    def _set_robot(self, robot_id: str, checked: bool) -> None:
        robot = next((entry for entry in self._catalog if entry["id"] == robot_id), None)
        if robot is None:
            return
        if not checked and self._last_selected(robot_id):
            self._sync_card(robot_id)
            return
        if checked and not robot.get("available", True):
            self._sync_card(robot_id)
            return
        if self._selected.get(robot_id, False) == checked:
            self._sync_card(robot_id)
            return
        self._selected[robot_id] = checked
        if checked and not self._references.get(robot_id):
            self._references[robot_id] = ["Default"]
        self._sync_card(robot_id)
        self.changed.emit()

    def _last_selected(self, robot_id: str) -> bool:
        selected = {
            str(robot["id"])
            for robot in self._catalog
            if self._selected.get(str(robot["id"]), False) and robot.get("available", True)
        }
        return selected == {robot_id}

    def _sync_card(self, robot_id: str, refresh_popup: bool = True) -> None:
        card = self._cards.get(robot_id)
        if card is None:
            return
        references = self._references.get(robot_id, ["Default"])
        checked = self._selected.get(robot_id, False)
        card.set_state(
            checked, self._last_selected(robot_id), len(references), self.profile_names(robot_id)
        )
        popup = self._open_popups.get(robot_id)
        if popup is not None:
            popup.set_has_more(self._has_more.get(robot_id, False))
            if popup.isVisible() and not (card.available and checked):
                self._close_popup(robot_id)
            elif refresh_popup and popup.isVisible():
                popup.refresh(self._profile_entries(robot_id), references)
        for identity, other in self._cards.items():
            if identity != robot_id:
                other.set_state(
                    self._selected.get(identity, False),
                    self._last_selected(identity),
                    other.count,
                    other.names,
                )

    def _append_reference(self, robot_id: str, reference: Reference) -> None:
        references = self._references.setdefault(robot_id, ["Default"])
        identity = _key(robot_id, reference)
        if all(_key(robot_id, existing) != identity for existing in references):
            references.append(deepcopy(reference))

    def _profile_entries(self, robot_id: str) -> list[ProfileEntry]:
        """Default, the saved profiles, then selected profiles the library does not list (campaign
        snapshots, saved profiles past the first 250)."""
        entries = [ProfileEntry("Default", "default", "Default", "default")]
        saved = self._saved.get(robot_id, {})
        # The engine lists profiles newest first; the design lists them oldest first (04, 13).
        for identity, row in sorted(saved.items(), key=lambda item: _created(item[1])):
            key = _key(robot_id, identity)
            if key == "default":
                continue
            name = str(row.get("name") or identity)
            entries.append(ProfileEntry(identity, key, name, "library"))
        known = {entry.key for entry in entries}
        for reference in self._references.get(robot_id, []):
            key = _key(robot_id, reference)
            if key in known:
                continue
            name = _label(robot_id, reference)
            if isinstance(reference, dict):
                if name.casefold() == "default":
                    name = "Default · campaign snapshot"
                entries.append(ProfileEntry(deepcopy(reference), key, name, "snapshot"))
            else:
                entries.append(ProfileEntry(reference, key, name, "library"))
            known.add(key)
        return entries

    def _popup_rect(self, robot_id: str, host: PopupHost) -> QtCore.QRect:
        """HANDOFF §8.4: cover the other card row, three columns from the clicked card's column (ending at
        it when it is too far right), one pixel shorter than the row."""
        index = [str(robot["id"]) for robot in self._catalog].index(robot_id)
        row, column = self.grid.cell(index)
        rows = max(1, math.ceil(len(self._catalog) / self.grid.columns))
        other = row + 1 if row + 1 < rows or row == 0 else row - 1
        span = min(3, self.grid.columns)
        start = max(0, min(column, self.grid.columns - span))
        box = self.grid.cell_box(other, start, span)
        origin = self.grid.content.mapTo(host.window_widget, QtCore.QPoint(0, 0))
        left, top = snap(origin.x() + box.x()), snap(origin.y() + box.y())
        width = snap(origin.x() + box.right()) - left
        height = snap(origin.y() + box.bottom()) - top - 1
        window = host.window_widget.rect()
        top = max(8, min(top, window.height() - 8 - height))
        return QtCore.QRect(left, top, width, height)

    def _show_profiles(self, robot_id: str) -> None:
        robot = next((entry for entry in self._catalog if entry["id"] == robot_id), None)
        host = self.window().findChild(PopupHost)
        if (
            robot is None
            or host is None
            or not robot.get("available", True)
            or not self._selected.get(robot_id, False)
        ):
            return
        card = self._cards[robot_id]
        popup = self._open_popups.get(robot_id)
        if popup is None:
            popup = ProfilesPopup(self.themes, robot, self._metadata, self.transport)
            popup.changed.connect(lambda refs, rid=robot_id: self._set_profiles(rid, refs))
            popup.finished.connect(lambda rid=robot_id: self._close_popup(rid))
            popup.edit_profile.connect(
                lambda reference, rid=robot_id: self.edit_profile.emit(rid, reference)
            )
            self._open_popups[robot_id] = popup
        popup.set_has_more(self._has_more.get(robot_id, False))
        popup.refresh(self._profile_entries(robot_id), self._references.get(robot_id, []))
        card.button.set_open(True)
        host.open_popup(
            popup,
            self._popup_rect(robot_id, host),
            opener=card.button,
            on_close=lambda: card.button.set_open(False),
        )
        popup.list.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def _close_popup(self, robot_id: str) -> None:
        popup = self._open_popups.get(robot_id)
        host = self.window().findChild(PopupHost)
        if popup is not None and host is not None and host.popup is popup:
            host.close_popup()

    def _drop_popup(self, robot_id: str) -> None:
        self._close_popup(robot_id)
        popup = self._open_popups.pop(robot_id, None)
        if popup is not None:
            popup.hide()
            popup.deleteLater()

    def _set_profiles(self, robot_id: str, references: list[Reference]) -> None:
        if not references:
            return
        self._references[robot_id] = deepcopy(references)
        self._sync_card(robot_id, refresh_popup=False)
        self.changed.emit()

    def _layout_cards(self) -> None:
        cards = [
            self._cards[str(robot["id"])]
            for robot in self._catalog
            if str(robot["id"]) in self._cards
        ]
        self.grid.set_cards(cards)
