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

"""New campaign step 2 (35–37, 06; HANDOFF §8.5-§8.6): the world sections, their cards and the starting-poses
popup.

``WorldSelector`` keeps the selection rules: Bookstore starts ticked (else the first world), the selection is never
empty (the last ticked world cannot be cleared), each ticked world keeps at least one starting pose, every ticked
world has a time limit (simulation seconds from the first movement; 0 is unlimited), and the popup lists the first
250 saved spawns (a footer row says so when there are more). A saved spawn gets a new ID with each edit; the
selection follows it through the spawn's stable ``family_id``.

Worlds sit in titled sections (``SECTION_ORDER``: a generated world's family, else its tier); a chip per section
shows or hides it, and Select all · Unselect all act on the shown worlds only. Collection worlds
(``spawn_policy: predefined``) always run all their predefined spawns (no popup; a tuning run picks one). The
"Kinematic time limits" switch shows each selected world's limit from the catalog's ``rule_time_limits`` for the
selected robots, read-only; the GUI never computes the rule.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .contracts import DEFAULT_LIMIT, TIME_LIMIT_RULE
from .gui_cells import (
    CELL_ROLE,
    CHECK_COLUMN,
    Cell,
    CellDelegate,
    ColumnStyle,
    OverlayScrollBar,
    TableHeader,
    TableLook,
    style_table,
)
from .gui_controls import UNIT_TEXT, DoubleSpinBox, Switch
from .gui_maps import MapStage, SpawnMark, WorldMap
from .gui_menus import show_tooltip
from .gui_support import Transport
from .gui_theme import Mark, ThemeManager, paint_check, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    CARD_TITLE,
    LABEL,
    Button,
    CssBox,
    PopupCard,
    PopupHost,
    SegmentedControl,
    Shadow,
    TextLine,
    TextStyle,
    Themed,
    css_rect,
    draw_text,
    place,
    rounded_path,
    shadow_image,
    shadow_margin,
)
from .gui_wizard import (
    PopupButton,
    PopupHead,
    Reference,
    ToggleCard,
    grouped_seconds,
    search_text,
    toggle_reference,
    toggle_shown,
)

NAME = TextStyle(14, 600, 18)
AREA = TextStyle(12, 400, 18, tabular=True)
VALUE = TextStyle(13, 500, 18, tabular=True)
MONO = TextStyle(12, 400, 18, mono=True)
COUNT = TextStyle(12.5, 400, 18, tabular=True)
SECTION_COUNT = TextStyle(13, 400, 18, tabular=True)
MORE_NOTE = "First 250 saved spawns shown. The Worlds & Spawns page lists all of them."
LAST_WORLD = "Select another world before deselecting this one."
LAST_SPAWN = "At least one starting position is required. Select another first."
LIMIT_TIP = "Simulation time from the first movement; 0 means no limit."
PREDEFINED_TIP = "Collection worlds run all their predefined spawns."
UNAVAILABLE_TIP = "The catalog lists no predefined spawns for this world; it cannot run."
RULE_TIP = (
    "On: each selected world's limit comes from the rule for the selected robots, read-only. "
    "Off: limits are set by hand."
)
RULE_UNAVAILABLE = (
    "The engine's catalog has no rule limits (an older suite image, or no world has a version-2 "
    "reference)."
)

DEFAULT_WORLD = "bookstore"  # ticked when a new campaign starts (as the default robot on step 1)
# The campaign key's value is contracts.TIME_LIMIT_RULE (plan I4); the engine computes the limits.
# The rule's formula as the tooltip writes it: it belongs to TIME_LIMIT_RULE (I4 §3.1: factor 30, floor 600 s,
# cap 14,400 s; exploration_benchmark_runner.time_limits); change both together. The GUI only shows it; the
# values always come from the catalog.
RULE_FLOOR_S = 600
RULE_CAP_S = 14400
RULE_FORMULA = f"min({RULE_CAP_S}, max({RULE_FLOOR_S}, ⌈30 A / (2 R v)⌉))"

# Sections of the world step, in order, with their titles: a generated world's family (F1–F5), else its tier.
# The analysis package groups worlds the same way (benchmark_analysis/worlds.py GROUP_ORDER, GROUP_TITLES), but
# takes any family as a group; here a family outside F1–F5 falls back to its tier (a new family needs a title).
SECTION_ORDER = ("A", "SC", "P", "F1", "F2", "F3", "F4", "F5", "RP", "MO")
SECTION_TITLES = {
    "A": "Original worlds",
    "SC": "Single challenge",
    "P": "Mixed worlds",
    "F1": "F1 Mazes",
    "F2": "F2 Offices",
    "F3": "F3 Warehouse aisles",
    "F4": "F4 Forests",
    "F5": "F5 Urban blocks",
    "RP": "Real places",
    "MO": "Moving objects",
}
FAMILIES = ("F1", "F2", "F3", "F4", "F5")


# ---------- catalog facts ----------


def section_of(world: Mapping[str, Any]) -> str:
    """The world's section: its family for a generated world, else its tier (an engine without the registry
    facts puts every world under the original worlds)."""
    family, tier = world.get("family"), world.get("tier")
    if isinstance(family, str) and family in FAMILIES:
        return family
    return tier if isinstance(tier, str) and tier else "A"


def section_title(section: str) -> str:
    return SECTION_TITLES.get(section, section)


def is_predefined(world: Mapping[str, Any]) -> bool:
    """A collection world: it runs all its predefined spawns and takes no custom ones."""
    return world.get("spawn_policy") == "predefined"


def world_code(world: Mapping[str, Any]) -> str:
    """The collection's short world ID ("SC2a", "F1-01") from the catalog's ``code``; empty without one."""
    code = world.get("code")
    return code if isinstance(code, str) else ""


def world_name(world: Mapping[str, Any]) -> str:
    """The short name the summaries and tables use: the collection ID, else the label."""
    return world_code(world) or str(world.get("label") or world.get("id") or "")


def world_title(world: Mapping[str, Any]) -> str:
    """ID and label ("SC4 · Office strip"), or the label alone."""
    code, label = world_code(world), str(world.get("label") or world.get("id") or "")
    return f"{code} · {label}" if code and label else code or label


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def predefined_spawns(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A collection world's predefined spawns as campaign snapshots (``id``, ``name``, ``x``, ``y``, ``yaw``)
    from the catalog's ``spawns``, in registry order; empty when the engine does not list them."""
    rows = world.get("spawns")
    result: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            continue
        pose = [_finite(row.get(field)) for field in ("x", "y", "yaw")]
        if any(value is None for value in pose):
            continue
        x, y, yaw = pose
        name = row.get("name")
        result.append(
            {
                "id": row["id"],
                "name": name if isinstance(name, str) and name else row["id"],
                "x": x,
                "y": y,
                "yaw": yaw,
            }
        )
    return result


def spawn_range(names: Sequence[str]) -> str:
    """Predefined spawns as one range ("S1–S3"), or the single name ("S1"); ``names`` is never empty."""
    return names[0] if len(names) == 1 else f"{names[0]}–{names[-1]}"


def spawn_key(world: str, reference: Reference) -> str:
    if isinstance(reference, dict):
        return str(reference["id"])
    return "default-" + world if reference in ("Default", "default") else str(reference)


def signed(value: float, digits: int) -> str:
    """Fixed decimals with a true minus sign (U+2212); a value that rounds to zero has no sign."""
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


def world_size(world: Mapping[str, Any]) -> str:
    """The reference grid's extent: "15.7 × 14.8 m"."""
    try:
        width = float(world["width"]) * float(world["resolution"])
        height = float(world["height"]) * float(world["resolution"])
    except (KeyError, TypeError, ValueError):
        return "—"
    if not math.isfinite(width * height):
        return "—"
    return f"{round(width, 1):g} × {round(height, 1):g} m"


def target_area(world: Mapping[str, Any]) -> float | None:
    area = _finite((world.get("reference") or {}).get("target_area_m2"))
    return area if area is not None and area >= 0 else None


def world_extent(world: Mapping[str, Any]) -> str:
    """Size and coverage target area: "15.7 × 14.8 m · 157.1 m²" (35)."""
    area = target_area(world)
    size = world_size(world)
    return size if area is None else f"{size} · {area:,.1f} m²"


def decimal(value: float) -> str:
    """At least one decimal and at most three, trailing zeros dropped: "10.0", "0.26", "1.0"."""
    text = f"{value:.3f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


# ---------- the kinematic rule ----------


@dataclass(frozen=True, slots=True)
class RuleLimit:
    """A world's limit under the rule for the robots that run it: the seconds, or None when a robot has no
    value or the robots disagree (the box shows "—"); the tooltip gives the inputs or the reason."""

    seconds: float | None
    tooltip: str


def rule_available(worlds: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the catalog carries rule limits (an engine with the rule and at least one world with a
    version-2 reference)."""
    return any(
        isinstance(w.get("rule_time_limits"), dict) and w["rule_time_limits"] for w in worlds
    )


def rule_limit(world: Mapping[str, Any], robots: Sequence[tuple[str, str]]) -> RuleLimit:
    """The catalog's ``rule_time_limits`` of ``world`` for ``robots`` (ID and label)."""
    values = world.get("rule_time_limits")
    values = values if isinstance(values, dict) else {}
    name = world_name(world)
    if not robots:
        return RuleLimit(None, f"{name}: select a robot on step 1.")
    lines: list[str] = []
    found: set[float] = set()
    missing: list[str] = []
    for robot, label in robots:
        entry = values.get(robot)
        entry = entry if isinstance(entry, dict) else {}
        seconds = _finite(entry.get("seconds"))
        if seconds is None:
            missing.append(label)
            continue
        found.add(seconds)
        uncapped = _finite(entry.get("uncapped_seconds"))
        capped = entry.get("capped") is True
        inputs = [
            ("Target area A", entry.get("area_m2"), "{:,.2f} m²"),
            ("Lidar range R", entry.get("range_m"), "{} m"),
            ("Top speed v", entry.get("speed_mps"), "{} m/s"),
        ]
        if lines:
            lines.append("")
        lines.append(f"{name} · {label} · {TIME_LIMIT_RULE}")
        for key, raw, pattern in inputs:
            number = _finite(raw)
            text = (
                "—"
                if number is None
                else pattern.format(number if "," in pattern else decimal(number))
            )
            lines.append(f"{key}: {text}")
        result = f"{grouped_seconds(seconds)} s, " + (
            f"capped (uncapped {grouped_seconds(uncapped)} s)"
            if capped and uncapped is not None
            else "capped"
            if capped
            # The uncapped value is max(600, ⌈…⌉): at 600 the floor applied (or ⌈…⌉ is exactly 600).
            else f"floor {RULE_FLOOR_S} s applied"
            if uncapped == RULE_FLOOR_S
            else "not capped"
        )
        lines.append(f"{RULE_FORMULA}: {result}")
    if missing:
        return RuleLimit(
            None,
            f"{name}: the catalog has no rule limit for {', '.join(missing)} in this world.",
        )
    if len(found) > 1:
        lines += ["", "The robots' limits differ: split the campaign by robot."]
        return RuleLimit(None, "\n".join(lines))
    return RuleLimit(found.pop(), "\n".join(lines))


# ---------- time limit boxes ----------


class TimeLimitBox(DoubleSpinBox):
    """The world's time limit in simulation seconds (0: unlimited), shown with grouped thousands ("1,800",
    "1,800.5"); Enter commits the value and stays in the box."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        # ``.wc .r .v`` also reaches the spin box's value: 13 px 500 (text.tertiary when the world is out).
        super().__init__(themes, "s", parent, weight=500)
        self.setLocale(QtCore.QLocale.c())
        self.setDecimals(6)
        self.setRange(0, 1e12)
        self.setSingleStep(60)
        self.setFixedSize(104, 32)
        # The box hands its colours to its edit, and the edit's own rules apply after that: set both.
        self.setProperty("muted", True)
        self.style().unpolish(self)
        self.style().polish(self)
        themes.changed.connect(self._restyle_value)
        self._restyle_value()

    def _restyle_value(self, *_: object) -> None:
        muted = self.themes.current.qss("text.tertiary")
        self.lineEdit().setStyleSheet(f"QLineEdit:disabled {{ color: {muted}; }}")

    def textFromValue(self, value: float) -> str:
        return grouped_seconds(value)

    def valueFromText(self, text: str) -> float:
        return float(super().valueFromText(text.replace(",", "")))

    def validate(self, text: str, position: int) -> tuple[QtGui.QValidator.State, str, int]:
        """The thousands separators are accepted and kept as typed."""
        result: Any = super().validate(text.replace(",", ""), position)
        state = result[0] if isinstance(result, tuple) else result
        return QtGui.QValidator.State(state), text, position

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.interpretText()
            event.accept()
            return
        super().keyPressEvent(event)


class RuleLimitBox(Themed, CssBox):
    """``.spin.ro`` (36): the rule's limit, read-only: sunken fill, ``border.line``, no steppers, the value
    (13 px 500 ink; tertiary on an unselected card) and "s" 12 px from the right edge; "—" when the robots
    disagree or have no value. Hovering shows the rule's inputs."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.seconds: float | None = None
        self.muted = False
        self.tip = ""
        self.setFixedSize(104, 32)

    def set_limit(self, limit: RuleLimit, muted: bool) -> None:
        if (limit.seconds, limit.tooltip, muted) != (self.seconds, self.tip, self.muted):
            self.seconds, self.tip, self.muted = limit.seconds, limit.tooltip, muted
            shown = "no single value" if limit.seconds is None else grouped_seconds(limit.seconds)
            self.setAccessibleName(f"Time limit from the kinematic rule: {shown} seconds")
            self.update()

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            show_tooltip(self, event.globalPos(), self.tip)
            return True
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.theme
        painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
        painter.setBrush(theme.color("surface.sunken"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 5.5)
        )
        top = (self.height() - 18) / 2
        right = self.width() - 1 - 12.0
        if self.seconds is None:
            dash = "—"
            draw_text(
                painter, right - VALUE.advance(dash), top, dash, VALUE, self.color("text.secondary")
            )
            return
        right -= draw_text(
            painter,
            right - UNIT_TEXT.advance("s"),
            top,
            "s",
            UNIT_TEXT,
            self.color("text.tertiary"),
        )
        value = grouped_seconds(self.seconds)
        ink = self.color("text.tertiary" if self.muted else "text.ink")
        draw_text(painter, right - 6 - VALUE.advance(value), top, value, VALUE, ink)


class PoseChoice(SegmentedControl):
    """``.wc .seg`` (37): a tuning run's single choice among a collection world's predefined spawns, with
    22 px segments padded 0 9 in a 28 px track."""

    def __init__(
        self, themes: ThemeManager, names: Sequence[str], parent: QtWidgets.QWidget | None = None
    ) -> None:
        style = TextStyle(13, 500, 18)
        width = max((style.advance(name) for name in names), default=0.0)
        super().__init__(
            themes,
            [(name, None) for name in names],
            segment_width=math.ceil(18 + width),
            parent=parent,
        )
        self.setAccessibleName("Starting pose of the tuning run")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            math.ceil(self.shift + self.natural_width()) - math.floor(self.shift), 28
        )

    def relayout(self) -> None:
        x = self.shift + 3
        for segment in self.segments:
            width = segment.natural_width()
            left, right = math.floor(x + 0.5), math.floor(x + width + 0.5)
            segment.shift = x - left
            segment.setGeometry(left, 3, right - left, 22)
            x += width + 2
        self.updateGeometry()


# ---------- world cards ----------


class WorldCard(ToggleCard):
    """``.wc`` (35): 292 px; check, name (14 px 600; a collection world's ID, then its title in
    ``text.secondary``), "size · area", the 128 px hatched map with the chosen poses, then the "Starting poses"
    row (original worlds: names and Spawns ⌄; collection worlds: "S1–S3 · predefined", or a tuning run's single
    choice) and the "Time limit" row (spin box with "s", or the rule's read-only value). Selected: 2 px ink
    border; hover: ``border.strong``; a world left out shows its name muted, "Not included", the excluded map
    colours and disabled controls."""

    # Offsets from the card's CSS box: 1 px border plus padding 14 16 2; the 18 px head line, the area line
    # 2 px under it, the map 10 px further, then the rows 4 px under the map: the first 52 px (with its rule),
    # the time limit row in the remaining 45 px (its box 10 px from the row's top, as the mockup's CSS lays
    # out the 52 px row that the card's height cuts short).
    HEIGHT, PAD, HEAD_TOP, AREA_TOP, STAGE_TOP, STAGE_HEIGHT = 292, 17, 15, 35, 63, 128
    ROWS_TOP, ROW, KEY_WIDTH = 195, 52, 84

    def __init__(
        self, themes: ThemeManager, world: dict[str, Any], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.world = deepcopy(world)
        self.world_id, self.label = str(world["id"]), str(world["label"])
        self.code = world_code(world)
        self.collection = is_predefined(world)
        self.extent_text = world_extent(world)
        self.checked = self.guarded = False
        self.unavailable = False  # a collection world whose catalog lists no predefined spawns
        self.names = "Default"
        # Engine errors, shown as the tooltips of the map and of the poses.
        self.map_error = self.pose_error = ""
        self.stage = MapStage(themes, pad=8, parent=self)
        self.stage.setAccessibleName(self.label + " map and starting positions")
        self.spawns_button = PopupButton(themes, "Spawns", self)
        self.spawns_button.setAccessibleName("Select starting positions for " + self.label)
        self.spawns_button.setVisible(not self.collection)
        names = [spawn["name"] for spawn in predefined_spawns(world)]
        self.choice: PoseChoice | None = None
        if self.collection and len(names) > 1:
            self.choice = PoseChoice(themes, names, self)
            self.choice.hide()
        self.time_limit = TimeLimitBox(themes, self)
        self.time_limit.setValue(float(world.get("default_time_limit", DEFAULT_LIMIT)))
        self.time_limit.setAccessibleName(self.label + " time limit in simulation seconds")
        self.time_limit.setToolTip(LIMIT_TIP)
        self.rule_box = RuleLimitBox(themes, self)
        self.rule_box.hide()
        self._sync()

    def set_state(self, checked: bool, guarded: bool, names: str) -> None:
        if (checked, guarded, names) != (self.checked, self.guarded, self.names):
            self.checked, self.guarded, self.names = checked, guarded, names
            self._sync()

    def set_choice(self, index: int | None) -> None:
        """A tuning run's single choice among the predefined spawns (None: the fixed line)."""
        if self.choice is None:
            return
        if index is not None:
            self.choice.set_index(index)
        self.choice.setVisible(index is not None)
        self.update()

    def set_rule(self, limit: RuleLimit | None) -> None:
        """The rule's read-only value (None: the editable limit)."""
        self.time_limit.setVisible(limit is None)
        self.rule_box.setVisible(limit is not None)
        if limit is not None:
            self.rule_box.set_limit(limit, not self.checked)

    def choosing(self) -> bool:
        return self.choice is not None and self.choice.isVisible()

    def _sync(self) -> None:
        self.set_selected(self.checked)
        self.stage.set_excluded(not self.checked)
        self.spawns_button.setEnabled(self.checked)
        self.time_limit.setEnabled(self.checked)
        if self.choice is not None:
            self.choice.setEnabled(self.checked)
        if self.rule_box.muted == self.checked:
            self.rule_box.muted = not self.checked
            self.rule_box.update()
        state = "selected" if self.checked else "not selected"
        self.setAccessibleName(f"Select world: {world_title(self.world)}, {state}")
        self.setAccessibleDescription(f"{self.extent_text}. Starting poses: {self.names}")
        self.update()

    def rows_top(self) -> float:
        return css_rect(self).y() + self.ROWS_TOP

    def _place_children(self) -> None:
        box = css_rect(self)
        x, inner, rows = box.x() + self.PAD, box.width() - 2 * self.PAD, self.rows_top()
        place(self.stage, x, box.y() + self.STAGE_TOP, inner, self.STAGE_HEIGHT)
        button_width = self.spawns_button.natural_width()
        place(self.spawns_button, x + inner - button_width, rows + 11.5, button_width, 28)
        if self.choice is not None:
            self.choice.set_origin_x(x + self.KEY_WIDTH + 10)
            self.choice.move(self.choice.x(), snap(rows + 11.5))
            self.choice.resize(self.choice.sizeHint())
        for box_widget in (self.time_limit, self.rule_box):
            place(box_widget, x + inner - 104, rows + self.ROW + 10, 104, 32)

    def css_changed(self) -> None:
        self._place_children()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_children()

    # tooltips

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            show_tooltip(self, event.globalPos(), self.tooltip_at(event.pos()))
            return True
        return super().event(event)

    def tooltip_at(self, position: QtCore.QPoint) -> str:
        box, rows, y = css_rect(self), self.rows_top(), position.y()
        if y < box.y() + self.AREA_TOP:
            if self.guarded:
                return LAST_WORLD
            return world_title(self.world) if self.code else ""
        if y < rows:
            return self.map_error
        if rows + self.ROW <= y and position.x() < box.x() + self.PAD + self.KEY_WIDTH:
            return LIMIT_TIP
        if rows <= y < rows + self.ROW:
            if self.collection:
                return UNAVAILABLE_TIP if self.unavailable else PREDEFINED_TIP
            if self.checked:
                return self.pose_error or self.names
        return ""

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)  # fill and border (2 px ink when selected)
        painter = QtGui.QPainter(self)
        theme = self.theme
        if not self.checked and self.underMouse():
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtGui.QPen(theme.color("border.strong"), 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            outline = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.drawPath(rounded_path(outline, 7.5))
        box = css_rect(self)
        top = box.y()
        x, inner = box.x() + self.PAD, box.width() - 2 * self.PAD
        ink, secondary = theme.color("text.ink"), theme.color("text.secondary")
        muted = theme.color("text.tertiary")
        # Head: check, the name (a collection world's ID) and its title.
        y = top + self.HEAD_TOP
        mark: Mark = "on" if self.checked else "off"
        paint_check(painter, QtCore.QRectF(x, y + 1, 16, 16), theme, mark)
        name_x = x + 16 + 8
        if self.code:
            name_width = draw_text(
                painter,
                name_x,
                y,
                NAME.elide(self.code, inner - 24),
                NAME,
                ink if self.checked else muted,
            )
            title_x = name_x + name_width + 8
            title = BODY.elide(self.label, max(0.0, x + inner - title_x))
            draw_text(painter, title_x, y, title, BODY, secondary if self.checked else muted)
        else:
            name = NAME.elide(self.label, max(0.0, inner - 24))
            draw_text(painter, name_x, y, name, NAME, ink if self.checked else muted)
        draw_text(painter, x, top + self.AREA_TOP, self.extent_text, AREA, secondary)
        # Rows: starting poses (with Spawns, the fixed line or the choice), a rule, then the time limit.
        rows = self.rows_top()
        rule = QtCore.QRect(snap(x), snap(rows + self.ROW - 1), snap(x + inner) - snap(x), 1)
        painter.fillRect(rule, theme.color("border.line"))
        line = rows + (self.ROW - 18) / 2
        draw_text(painter, x, line, "Starting poses", BODY, secondary)
        value_x = x + self.KEY_WIDTH + 10
        if self.collection and self.unavailable:
            draw_text(painter, value_x, line, self.names, BODY, muted)
        elif self.collection:
            if not self.choosing():
                fixed = draw_text(
                    painter, value_x, line, self.names, BODY_MEDIUM, ink if self.checked else muted
                )
                draw_text(painter, value_x + fixed, line, " · predefined", BODY, muted)
        else:
            room = x + inner - self.spawns_button.natural_width() - 10 - value_x
            value = BODY_MEDIUM.elide(
                self.names if self.checked else "Not included", max(0.0, room)
            )
            draw_text(painter, value_x, line, value, BODY_MEDIUM, ink if self.checked else muted)
        draw_text(painter, x, rows + self.ROW + 17, "Time limit", BODY, secondary)


# ---------- section heads and chips ----------


class SectionHead(Themed, CssBox):
    """``.sect .sh`` (35): the title (14/20 600) at the top left, "k of n selected" 10 px after it, and Select
    all · Unselect all (small ghost buttons, 4 px apart) at the right; 28 px high."""

    def __init__(
        self, themes: ThemeManager, section: str, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.section = section
        self.title = section_title(section)
        self.count_text = ""
        self.select_all = Button(themes, "Select all", variant="ghost", small=True, parent=self)
        self.unselect_all = Button(themes, "Unselect all", variant="ghost", small=True, parent=self)
        self.select_all.setAccessibleName("Select all worlds of " + self.title)
        self.unselect_all.setAccessibleName("Unselect all worlds of " + self.title)

    def set_count(self, selected: int, total: int) -> None:
        text = f"{selected} of {total} selected"
        if text != self.count_text:
            self.count_text = text
            self.update()

    def css_changed(self) -> None:
        self._layout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def _layout(self) -> None:
        box = css_rect(self)
        right = box.right()
        for button in (self.unselect_all, self.select_all):
            width = button.natural_width()
            place(button, right - width, box.y(), width, 28)
            right -= width + 4

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        box = css_rect(self)
        advance = draw_text(
            painter, box.x(), box.y() + 4, self.title, CARD_TITLE, self.color("text.ink")
        )
        draw_text(
            painter,
            box.x() + advance + 10,
            box.y() + 5,
            self.count_text,
            SECTION_COUNT,
            self.color("text.secondary"),
        )


class Chip(QtWidgets.QAbstractButton, CssBox):
    """``.chip`` (35): a section's filter, 28 px, padding 0 10, its title (12.5 px 500) and "k/n" 6 px after
    it (tertiary, tabular). Shown sections are raised like a segment (card fill, ``border.strong`` ring, small
    shadow); hidden ones are flat with a ``border.line`` ring and secondary text. The widget keeps ``MARGIN``
    px around the chip for its ring and shadow."""

    MARGIN = 3

    def __init__(
        self, themes: ThemeManager, section: str, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes, self.section = themes, section
        self.title = section_title(section)
        self.count_text = "0/0"
        self.setText(self.title)
        self.setCheckable(True)
        self.setChecked(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 6)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.toggled.connect(self._toggled)
        themes.changed.connect(self.update)
        self._toggled(True)

    def _toggled(self, shown: bool) -> None:
        self.setToolTip(("Hide " if shown else "Show ") + self.title)
        self.setAccessibleName(
            f"{self.title}, {self.count_text} selected, {'shown' if shown else 'hidden'}"
        )
        self.update()

    def set_count(self, selected: int, total: int) -> None:
        text = f"{selected}/{total}"
        if text != self.count_text:
            self.count_text = text
            self._toggled(self.isChecked())

    def natural_width(self) -> float:
        return 20 + LABEL.advance(self.title) + 6 + COUNT.advance(self.count_text)

    def painted_rect(self) -> QtCore.QRectF:
        box = css_rect(self)
        return QtCore.QRectF(
            box.x() + self.MARGIN, box.y() + self.MARGIN, box.width() - 2 * self.MARGIN, 28
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.click()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.themes.current
        chip = self.painted_rect()
        if self.isChecked():
            layers: tuple[Shadow, ...] = ((0, 1, 2, 0, theme.tokens["overlay.shadowSmall"]),)
            margin = shadow_margin(layers)
            painter.drawImage(
                QtCore.QPointF(snap(chip.x()) - margin, snap(chip.y()) - margin),
                shadow_image(round(chip.width()), round(chip.height()), 6, layers),
            )
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("border.strong"))
            painter.drawPath(rounded_path(chip.adjusted(-1, -1, 1, 1), 7))
            fill = theme.color("surface.hover" if self.underMouse() else "surface.card")
            painter.setBrush(fill)
            painter.drawPath(rounded_path(chip, 6))
            label = theme.color("text.ink")
        else:
            if self.underMouse():
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(theme.color("surface.hover"))
                painter.drawPath(rounded_path(chip, 6))
            painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(rounded_path(chip.adjusted(0.5, 0.5, -0.5, -0.5), 5.5))
            label = theme.color("text.secondary")
        top = chip.y() + 5
        x = chip.x() + 10
        x += draw_text(painter, x, top, self.title, LABEL, label) + 6
        draw_text(painter, x, top, self.count_text, COUNT, theme.color("text.tertiary"))


class ChipRow(QtWidgets.QWidget):
    """``.chips``: the section chips in a wrapping row, 8 px apart, the first 1 px in (its ring)."""

    GAP, LINE = 8, 28

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.chips: list[Chip] = []

    def set_chips(self, chips: Sequence[Chip]) -> None:
        for chip in self.chips:
            if chip not in chips:
                chip.hide()
        self.chips = list(chips)
        for chip in self.chips:
            chip.setParent(self)
            chip.show()
        self.relayout()

    def _positions(self, width: float) -> list[tuple[float, float]]:
        positions, x, y = [], 1.0, 0.0
        for chip in self.chips:
            size = chip.natural_width()
            if x > 1 and x + size > width:
                x, y = 1.0, y + self.LINE + self.GAP
            positions.append((x, y))
            x += size + self.GAP
        return positions

    def height_for(self, width: float) -> int:
        positions = self._positions(width)
        return math.ceil(positions[-1][1] + self.LINE) if positions else 0

    def relayout(self) -> None:
        margin = Chip.MARGIN
        for chip, (x, y) in zip(self.chips, self._positions(self.width()), strict=True):
            place(
                chip,
                x - margin,
                y - margin,
                chip.natural_width() + 2 * margin,
                self.LINE + 2 * margin,
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()


class StepBar(QtWidgets.QWidget):
    """``.stepbar`` (35, 36): the "Kinematic time limits" switch with the rule ID (mono, secondary) and, with the
    switch on, a faint note; the global Select all · Unselect all (small, 8 px apart) at the right. 32 px."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.themes = themes
        self.rule = Switch(
            themes,
            "Kinematic time limits",
            gap=10,
            style=BODY_MEDIUM,
            label_token="text.ink",
            parent=self,
        )
        self.rule.setToolTip(RULE_TIP)
        self.rule_id = TextLine(themes, TIME_LIMIT_RULE, MONO, "text.secondary", self)
        self.note = TextLine(themes, "", BODY, "text.tertiary", self)
        self.select_all = Button(themes, "Select all", small=True, parent=self)
        self.unselect_all = Button(themes, "Unselect all", small=True, parent=self)
        self.select_all.setAccessibleName("Select all shown worlds")
        self.unselect_all.setAccessibleName("Unselect all shown worlds")

    def set_note(self, text: str) -> None:
        self.note.setText(text)
        self.note.setVisible(bool(text))
        self.relayout()

    def relayout(self) -> None:
        width = float(self.width())
        switch = self.rule.natural_width()
        place(self.rule, 0, 7, switch, 18)
        x = switch + 10
        place(self.rule_id, x, 7, self.rule_id.natural_width(), 18)
        x += self.rule_id.natural_width() + 10
        right = width
        for button in (self.unselect_all, self.select_all):
            button_width = button.natural_width()
            right -= button_width
            place(button, right, 2, button_width, 28)
            right -= 8
        place(self.note, x, 7, max(0.0, min(self.note.natural_width(), right - 8 - x)), 18)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()


# ---------- spawns popup ----------


@dataclass(frozen=True, slots=True)
class SpawnEntry:
    """One row of the starting-poses popup."""

    reference: Reference
    key: str
    name: str
    pose: dict[str, Any] | None
    detail: str  # "· world file" after the default's name
    search: str


class SpawnModel(QtCore.QAbstractTableModel):
    """The poses shown in the popup (after the search), with their check states."""

    HEADERS = ("", "Starting pose", "X · m", "Y · m", "Heading")

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[SpawnEntry] = []
        self.checked: set[str] = set()

    def set_entries(self, entries: list[SpawnEntry], checked: set[str]) -> None:
        self.beginResetModel()
        self.entries, self.checked = entries, set(checked)
        self.endResetModel()

    def set_checked(self, checked: set[str]) -> None:
        self.checked = set(checked)
        if self.entries:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.entries) - 1, 0))

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
            and 0 <= section < len(self.HEADERS)
        ):
            return self.HEADERS[section]
        return None

    def flags(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> QtCore.Qt.ItemFlag:
        return QtCore.Qt.ItemFlag.ItemIsEnabled

    def cell(self, entry: SpawnEntry, column: int) -> Cell:
        pose = entry.pose
        if column == 0:
            checked = entry.key in self.checked
            sole = checked and len(self.checked) == 1
            state = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
            return Cell(kind="check", check=state, tooltip=LAST_SPAWN if sole else "")
        if column == 1:
            return Cell(entry.name, token="text.ink", note=entry.detail, note_kind="detail")
        if column in (2, 3):
            value = pose.get("x" if column == 2 else "y") if pose else None
            return Cell(signed(float(value), 2) if value is not None else "—", "number")
        yaw = pose.get("yaw") if pose else None
        return Cell(signed(math.degrees(float(yaw)), 1) + "°" if yaw is not None else "—", "number")

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or index.row() >= len(self.entries):
            return None
        cell = self.cell(self.entries[index.row()], index.column())
        if role == CELL_ROLE:
            return cell
        if role == QtCore.Qt.ItemDataRole.DisplayRole and cell.kind != "check":
            return (cell.text + " " + cell.note).strip()
        return None


class SpawnDelegate(CellDelegate):
    """``.sp-tbl``: no selected rows; the hovered row is the inspected pose (``hovered``)."""

    hovered = QtCore.Signal(int)

    def row_selected(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, selected: bool
    ) -> bool:
        return False

    def _set_hover(self, row: int) -> None:
        changed = row != self.hover_row
        super()._set_hover(row)
        if changed:
            self.hovered.emit(row)


SPAWN_LOOK = TableLook(
    key="worlds.spawns",
    columns=[
        CHECK_COLUMN,
        ColumnStyle("text"),
        ColumnStyle("number", width=80),
        ColumnStyle("number", width=80),
        ColumnStyle("number", padding_right=20, width=100),
    ],
)


class SpawnsPopup(PopupCard):
    """``.pp`` (06): the head (56 px: title, search, "Done · N selected"), the poses table (580 px: check,
    name, X, Y, heading; 40 px rows) and the map with every pose (chosen ones ink, the hovered one larger,
    labels for both); under a rule, a 38 px footer row notes the 250-spawn limit when the library has more
    (D2)."""

    changed = QtCore.Signal(object)
    finished = QtCore.Signal()

    HEAD, TABLE, NOTE = 56, 580, 38  # HEAD and TABLE include their rules; NOTE sits under its own

    def __init__(
        self, themes: ThemeManager, world: dict[str, Any], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.world_id = str(world["id"])
        self.references: list[Reference] = ["Default"]
        self.options: list[SpawnEntry] = []
        self.has_more = False
        self.inspected = -1
        self.setAccessibleName("Starting poses · " + str(world["label"]))
        self.head = PopupHead(
            themes, "Starting poses · " + str(world["label"]), "Search poses", self
        )
        self.done = self.head.done
        self.search = self.head.search
        self.model = SpawnModel()
        self.view = QtWidgets.QTableView(self)
        self.view.setModel(self.model)
        delegate = style_table(self.view, themes, SPAWN_LOOK, delegate_class=SpawnDelegate)
        assert isinstance(delegate, SpawnDelegate)
        self.delegate = delegate
        header = self.view.horizontalHeader()
        assert isinstance(header, TableHeader)
        self.header = header
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.view.setAccessibleName("Starting poses")
        self.stage = MapStage(themes, pad=10, ring=False, label_size=10, parent=self)
        self.note = TextLine(themes, MORE_NOTE, BODY, "text.secondary", self)
        self.note.hide()
        self.done.clicked.connect(self.finished.emit)
        self.search.textChanged.connect(self.filter_rows)
        self.header.check_toggled.connect(self.toggle_shown)
        self.delegate.hovered.connect(self.inspect)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        self.hide()

    # contents

    def set_has_more(self, has_more: bool) -> None:
        if has_more != self.has_more:
            self.has_more = has_more
            self.note.setVisible(has_more)
            self._layout()
            self.update()

    def refresh(self, entries: list[SpawnEntry], selected: list[Reference]) -> None:
        self.options = list(entries)
        self.references = deepcopy(selected) or ["Default"]
        self.filter_rows(self.search.text())

    def _key(self, reference: Reference) -> str:
        return spawn_key(self.world_id, reference)

    def keys(self) -> set[str]:
        return {self._key(reference) for reference in self.references}

    def filter_rows(self, query: str) -> None:
        """Show the poses matching ``query``; the inspected and the current (keyboard) row stay on their
        poses across the model reset."""
        needle = query.casefold().strip()
        shown = [entry for entry in self.options if not needle or needle in entry.search]
        inspected = self._inspected_key()
        current = self.view.currentIndex()
        entries = self.model.entries
        current_key = entries[current.row()].key if 0 <= current.row() < len(entries) else None
        self.model.set_entries(shown, self.keys())
        self.delegate.hover_row = -1
        self.inspected = next((i for i, e in enumerate(shown) if e.key == inspected), -1)
        row = next((i for i, e in enumerate(shown) if e.key == current_key), -1)
        if row >= 0:
            self.view.setCurrentIndex(self.model.index(row, current.column()))
        self._sync()

    def _inspected_key(self) -> str | None:
        entries = self.model.entries
        return entries[self.inspected].key if 0 <= self.inspected < len(entries) else None

    def inspect(self, row: int) -> None:
        """Show the pose of ``row`` as the inspected one (hovered rows; −1: none)."""
        if row != self.delegate.hover_row:
            self.delegate.hover_row = row
            self.view.viewport().update()
        self.inspected = row
        self._sync_map()

    def _sync(self) -> None:
        keys = self.keys()
        self.model.set_checked(keys)
        shown = self.model.entries
        ticked = sum(entry.key in keys for entry in shown)
        self.header.set_check_state(
            QtCore.Qt.CheckState.Checked
            if shown and ticked == len(shown)
            else QtCore.Qt.CheckState.PartiallyChecked
            if ticked
            else QtCore.Qt.CheckState.Unchecked
        )
        self.head.set_count(len(self.references))
        self._sync_map()

    def _sync_map(self) -> None:
        keys = self.keys()
        inspected = self._inspected_key()
        marks = []
        for entry in self.options:
            if entry.pose is None:
                continue
            picked, selected = entry.key in keys, entry.key == inspected
            pose = entry.pose
            marks.append(
                SpawnMark(
                    entry.name,
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["yaw"]),
                    picked=picked,
                    selected=selected,
                    labelled=picked or selected,
                )
            )
        self.stage.set_marks(marks)

    # actions

    def toggle(self, row: int) -> None:
        if not 0 <= row < len(self.model.entries):
            return
        selected = toggle_reference(self.references, self.model.entries[row].reference, self._key)
        if selected is not None:  # None: the last starting pose stays selected
            self._set_selected(selected)

    def toggle_shown(self) -> None:
        shown = [entry.reference for entry in self.model.entries]
        if shown:
            self._set_selected(toggle_shown(self.references, shown, self._key))

    def _set_selected(self, selected: list[Reference]) -> None:
        self.references = selected
        self._sync()
        self.changed.emit(deepcopy(self.references))

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if (
            watched is self.view.viewport()
            and kind == QtCore.QEvent.Type.MouseButtonRelease
            and isinstance(event, QtGui.QMouseEvent)
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and not self.delegate.dragging  # a text selection, not a click
        ):
            index = self.view.indexAt(event.position().toPoint())
            if index.isValid():
                self.view.setCurrentIndex(index)
                self.toggle(index.row())
        elif watched is self.view and kind == QtCore.QEvent.Type.KeyPress:
            assert isinstance(event, QtGui.QKeyEvent)
            current = self.view.currentIndex()
            if event.key() == QtCore.Qt.Key.Key_Space and current.isValid():
                self.toggle(current.row())
                return True
            text = search_text(event)
            if text:
                self.search.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                self.search.insert(text)  # type to search
                return True
        elif watched is self.view and kind == QtCore.QEvent.Type.KeyRelease:
            current = self.view.currentIndex()
            if current.isValid() and current.row() != self.inspected:
                self.inspect(current.row())  # the arrow keys inspect the current row
        return super().eventFilter(watched, event)

    # layout

    def body_height(self) -> int:
        """The table and map height (above the body's rule when the footer shows)."""
        return self.height() - 1 - self.HEAD - (1 + self.NOTE if self.has_more else 0) - 1

    @classmethod
    def popup_height(cls, has_more: bool) -> int:
        return 1 + cls.HEAD + 235 + (1 + cls.NOTE if has_more else 0) + 1

    def _layout(self) -> None:
        width, top, body = self.width(), 1 + self.HEAD, self.body_height()
        self.head.setGeometry(1, 1, width - 2, self.HEAD - 1)
        self.view.setGeometry(1, top, self.TABLE - 1, body)
        self.stage.setGeometry(self.TABLE + 1, top, width - self.TABLE - 2, body)
        self.note.setGeometry(21, top + body + 1, width - 42, self.NOTE)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        line = self.color("border.line")
        width, body = self.width(), self.body_height()
        painter.fillRect(QtCore.QRect(1, self.HEAD, width - 2, 1), line)
        painter.fillRect(QtCore.QRect(self.TABLE, 1 + self.HEAD, 1, body), line)
        if self.has_more:
            painter.fillRect(QtCore.QRect(1, 1 + self.HEAD + body, width - 2, 1), line)


# ---------- sections ----------


def section_columns(width: float) -> int:
    """Cards per row: four at the design's 1440 px window, fewer in narrower ones."""
    return 4 if width >= 1100 else 3 if width >= 820 else 2 if width >= 540 else 1


class WorldSections(QtWidgets.QScrollArea):
    """``.scroller`` (35): the shown sections top to bottom, 24 px apart, each its 28 px head, 10 px, then its
    cards (292 px, 16 px gaps, ``section_columns`` per row); the cards end 14 px before the right edge, where the
    thin scroll handle runs. ``scrolled`` fires when the visible cards may have changed."""

    GAP, SECTION_GAP, HEAD, HEAD_GAP, RIGHT = 16, 24, 28, 10, 14
    scrolled = QtCore.Signal()

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)
        self.viewport().setAutoFillBackground(False)
        self.content = QtWidgets.QWidget()
        self.content.setAutoFillBackground(False)
        self.setWidget(self.content)
        OverlayScrollBar(themes, self)
        self.sections: list[tuple[SectionHead, list[WorldCard]]] = []
        self.hidden: set[str] = set()
        self.columns, self.card_width = 4, 0.0
        self.boxes: dict[str, QtCore.QRectF] = {}  # card CSS boxes by world, content coordinates
        self.heads: dict[str, float] = {}  # section head tops
        self._laying_out = False
        self.verticalScrollBar().valueChanged.connect(lambda _value: self.scrolled.emit())

    def set_sections(self, sections: Sequence[tuple[SectionHead, list[WorldCard]]]) -> None:
        self.sections = list(sections)
        for head, cards in self.sections:
            for widget in (head, *cards):
                if widget.parent() is not self.content:
                    widget.setParent(self.content)
        self.relayout()

    def set_hidden(self, hidden: set[str]) -> None:
        self.hidden = set(hidden)
        self.relayout()

    def relayout(self) -> None:
        if self._laying_out:
            return
        self._laying_out = True
        try:
            width = max(0.0, float(self.viewport().width()) - self.RIGHT)
            self.columns = section_columns(width)
            self.card_width = (width - (self.columns - 1) * self.GAP) / self.columns
            self.boxes, self.heads = {}, {}
            y, first = 0.0, True
            for head, cards in self.sections:
                shown = head.section not in self.hidden and bool(cards)
                head.setVisible(shown)
                for card in cards:
                    card.setVisible(shown)
                if not shown:
                    continue
                y += 0 if first else self.SECTION_GAP
                first = False
                self.heads[head.section] = y
                place(head, 0, y, width, self.HEAD)
                y += self.HEAD + self.HEAD_GAP
                for index, card in enumerate(cards):
                    row, column = divmod(index, self.columns)
                    box = QtCore.QRectF(
                        column * (self.card_width + self.GAP),
                        y + row * (WorldCard.HEIGHT + self.GAP),
                        self.card_width,
                        WorldCard.HEIGHT,
                    )
                    self.boxes[card.world_id] = box
                    place(card, box.x(), box.y(), box.width(), box.height())
                rows = math.ceil(len(cards) / self.columns)
                y += rows * WorldCard.HEIGHT + (rows - 1) * self.GAP
            self.content.resize(self.viewport().width(), max(1, math.ceil(y)))
        finally:
            self._laying_out = False
        self.scrolled.emit()

    def column_of(self, world: str) -> int:
        box = self.boxes.get(world)
        return 0 if box is None else round(box.x() / (self.card_width + self.GAP))

    def visible_worlds(self, margin: float = WorldCard.HEIGHT + 16) -> list[str]:
        """Worlds whose cards are in view or within ``margin`` px of it (their maps load first)."""
        top = float(self.verticalScrollBar().value()) - margin
        bottom = top + self.viewport().height() + 2 * margin
        return [w for w, box in self.boxes.items() if box.bottom() >= top and box.top() <= bottom]

    def scroll_to_section(self, section: str) -> None:
        """Bring a shown section's head to the top (used by the offscreen render harness)."""
        if section in self.heads:
            self.verticalScrollBar().setValue(round(self.heads[section]))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            self.relayout()
        return super().viewportEvent(event)


# ---------- selector ----------


class WorldSelector(QtWidgets.QWidget):
    """Step 2 (35–37): the step bar (the rule switch, the global Select all · Unselect all), the section chips,
    the sections of world cards, and one starting-poses popup per original world (on the window's
    PopupHost)."""

    changed = QtCore.Signal()
    map_loaded = QtCore.Signal(str, object)  # world ID and its WorldMap, or None when it failed

    BAR, CHIPS_TOP, GAP = 32, 46, 16  # the chips sit 2 px into the 16 px gap under the step bar

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes = themes
        self.transport = transport
        self._worlds: dict[str, dict[str, Any]] = {}
        self._cards: dict[str, WorldCard] = {}
        self._heads: dict[str, SectionHead] = {}
        self._chips: dict[str, Chip] = {}
        self._selected: set[str] = set()
        self._references: dict[str, list[Reference]] = {}
        self._saved: dict[str, dict[str, dict[str, Any]]] = {}
        self._poses: dict[tuple[str, str], dict[str, Any]] = {}
        self._maps: dict[str, WorldMap] = {}
        self._map_errors: dict[str, str] = {}
        self._pending: dict[str, tuple[str, str | None]] = {}
        self._popups: dict[str, SpawnsPopup] = {}
        self._has_more: dict[str, bool] = {}
        self._hidden: set[str] = set()
        self._single = False  # a tuning run: one world with one starting pose
        # The rule's scope: the selected robots (ID, label) and the combinations left out in Review.
        self._robots: list[tuple[str, str]] = []
        self._excluded: set[tuple[str, str, str]] = set()
        self.bar = StepBar(themes, self)
        self.rule = self.bar.rule
        self.rule.setEnabled(False)
        self.rule.setToolTip(RULE_UNAVAILABLE)
        self.rule.toggled.connect(self._rule_toggled)
        self.bar.select_all.clicked.connect(lambda: self.select_shown(None, True))
        self.bar.unselect_all.clicked.connect(lambda: self.select_shown(None, False))
        self.chip_row = ChipRow(self)
        self.sections = WorldSections(themes, self)
        self.sections.scrolled.connect(self._request_visible)
        self.sections.verticalScrollBar().valueChanged.connect(lambda _value: self._close_popups())
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    # layout

    def _layout(self) -> None:
        width, height = self.width(), self.height()
        place(self.bar, 0, 0, width, self.BAR)
        chips = self.chip_row.height_for(width)
        place(self.chip_row, 0, self.CHIPS_TOP, width, chips)
        top = self.CHIPS_TOP + chips + self.GAP
        place(self.sections, 0, top, width, max(0, height - top))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    # catalog

    def set_catalog(self, worlds: list[dict[str, Any]]) -> None:
        limits = {world: card.time_limit.value() for world, card in self._cards.items()}
        accepted = {str(world["id"]): deepcopy(world) for world in worlds}
        reset = False
        for identity in tuple(self._cards):
            if identity not in accepted or self._worlds.get(identity) != accepted[identity]:
                self._cards.pop(identity).deleteLater()
                self._drop_popup(identity)
                self._maps.pop(identity, None)
                self._map_errors.pop(identity, None)
                self._pending = {
                    tag: pending for tag, pending in self._pending.items() if pending[0] != identity
                }
                self._poses = {key: pose for key, pose in self._poses.items() if key[0] != identity}
                # Custom spawns are kept; a collection world's references follow its predefined spawns.
                old = self._worlds.get(identity, {})
                new = accepted.get(identity, {})
                facts = ("spawn_policy", "spawns")
                collection = is_predefined(old) or is_predefined(new)
                if identity not in accepted or (
                    collection and any(old.get(k) != new.get(k) for k in facts)
                ):
                    reset = reset or (identity in self._selected and identity in self._references)
                    self._references.pop(identity, None)
        self._worlds = accepted
        self._selected.intersection_update(accepted)
        self._selected = {world for world in self._selected if self.available(world)}
        if not self._selected and accepted:  # never empty: Bookstore, else the first world
            usable = [world for world in accepted if self.available(world)] or list(accepted)
            self._selected.add(DEFAULT_WORLD if DEFAULT_WORLD in usable else usable[0])
        for identity, world in accepted.items():
            if is_predefined(world):
                self._references.setdefault(identity, self._predefined_references(identity))
            else:
                self._references.setdefault(identity, ["Default"])
            if identity not in self._cards:
                card = WorldCard(self.themes, world)
                if identity in limits:
                    card.time_limit.setValue(limits[identity])
                card.time_limit.valueChanged.connect(lambda _value: self.changed.emit())
                card.toggled.connect(
                    lambda wid=identity: self.select_world(wid, wid not in self._selected)
                )
                card.spawns_button.clicked.connect(
                    lambda checked=False, wid=identity: self.show_spawns(wid)
                )
                if card.choice is not None:
                    card.choice.changed.connect(
                        lambda index, wid=identity: self._choose_predefined(wid, index)
                    )
                self._cards[identity] = card
        self._build_sections()
        available = rule_available(worlds)
        self.rule.setEnabled(available or self.rule.isChecked())
        self.rule.setToolTip(RULE_TIP if available else RULE_UNAVAILABLE)
        for identity in accepted:
            self._sync(identity, counts=False)
        self._sync_counts()
        if self._single:
            self.set_single(True)
        self._sync_rule()
        self._request_visible()
        if reset:
            self.changed.emit()

    def _build_sections(self) -> None:
        grouped: dict[str, list[WorldCard]] = {}
        for identity, world in self._worlds.items():
            grouped.setdefault(section_of(world), []).append(self._cards[identity])
        order = [key for key in SECTION_ORDER if key in grouped]
        order += [key for key in grouped if key not in SECTION_ORDER]
        for key in tuple(self._heads):
            if key not in grouped:
                self._heads.pop(key).deleteLater()
                self._chips.pop(key).deleteLater()
        for key in order:
            if key not in self._heads:
                head = SectionHead(self.themes, key)
                head.select_all.clicked.connect(
                    lambda checked=False, k=key: self.select_shown(k, True)
                )
                head.unselect_all.clicked.connect(
                    lambda checked=False, k=key: self.select_shown(k, False)
                )
                self._heads[key] = head
                chip = Chip(self.themes, key)
                chip.toggled.connect(lambda shown, k=key: self.show_section(k, shown))
                self._chips[key] = chip
        self._hidden.intersection_update(order)
        self.chip_row.set_chips([self._chips[key] for key in order])
        self.sections.hidden = set(self._hidden)
        self.sections.set_sections([(self._heads[key], grouped[key]) for key in order])
        self._layout()

    def _predefined_references(self, world: str) -> list[Reference]:
        """A collection world's predefined spawns as a campaign names them: their ID strings in registry
        order (never inline copies; EXECUTION.md "Catalog ``spawns``"); empty when the catalog lists none
        (the world is then unavailable)."""
        return [spawn["id"] for spawn in predefined_spawns(self._worlds[world])]

    def available(self, world: str) -> bool:
        """False for a collection world whose catalog lists no valid predefined spawns: it cannot run."""
        return not self.is_collection(world) or bool(predefined_spawns(self._worlds[world]))

    def _predefined(self, world: str, reference: Reference) -> dict[str, Any] | None:
        """The catalog's registry spawn a reference names: a collection world's predefined spawns, or an
        original world's fixed registry choices (decision 60: Bookstore and Corridor S2, S3; Warehouse S2)."""
        if world not in self._worlds or not isinstance(reference, str):
            return None
        key = spawn_key(world, reference)
        return next((s for s in predefined_spawns(self._worlds[world]) if s["id"] == key), None)

    def is_collection(self, world: str) -> bool:
        """A collection world: it runs its predefined spawns and has no popup."""
        return world in self._worlds and is_predefined(self._worlds[world])

    # sections and the filter

    def section_worlds(self, section: str | None) -> list[str]:
        """The worlds of ``section``, or of every shown section (None)."""
        return [
            identity
            for identity, world in self._worlds.items()
            if (section_of(world) == section if section else section_of(world) not in self._hidden)
        ]

    def show_section(self, section: str, shown: bool) -> None:
        """A chip: show or hide its section; the selection of hidden worlds is kept."""
        if shown:
            self._hidden.discard(section)
        else:
            self._hidden.add(section)
            for world in self.section_worlds(section):
                self._close_popup(world)
        if self._chips[section].isChecked() != shown:
            self._chips[section].setChecked(shown)
        self.sections.set_hidden(self._hidden)
        self._sync_counts()

    def hidden_sections(self) -> set[str]:
        """The sections hidden by their chips (used by the offscreen render harness)."""
        return set(self._hidden)

    def _select_result(self, scope: list[str], checked: bool) -> set[str]:
        """The selection after Select all (``checked``) or Unselect all on ``scope``; unselecting keeps the
        first world of the scope when nothing else would stay selected."""
        if checked:
            return self._selected | {world for world in scope if self.available(world)}
        kept = self._selected - set(scope)
        if not kept:
            first = next((w for w in scope if w in self._selected), None)
            kept = {first} if first else set(self._selected)
        return kept

    def select_shown(self, section: str | None, checked: bool) -> None:
        """Select all or Unselect all on a section's worlds, or on every shown world (None)."""
        if self._single:
            return
        selected = self._select_result(self.section_worlds(section), checked)
        if selected == self._selected:
            return
        removed = self._selected - selected
        self._selected = selected
        for world in removed:
            self._close_popup(world)
        for world in self._worlds:
            self._sync(world, counts=False)
        self._sync_counts()
        self.changed.emit()

    def _sync_counts(self) -> None:
        for key, chip in self._chips.items():
            worlds = self.section_worlds(key)
            selected = sum(world in self._selected for world in worlds)
            chip.set_count(selected, len(worlds))
            head = self._heads[key]
            head.set_count(selected, len(worlds))
            for button, checked in ((head.select_all, True), (head.unselect_all, False)):
                button.setEnabled(
                    not self._single and self._select_result(worlds, checked) != self._selected
                )
        self.chip_row.relayout()
        self._layout()
        shown = self.section_worlds(None)
        for button, checked in ((self.bar.select_all, True), (self.bar.unselect_all, False)):
            button.setEnabled(
                not self._single
                and bool(shown)
                and self._select_result(shown, checked) != self._selected
            )

    # the selection

    def selections(self) -> dict[str, list[Reference]]:
        return {
            world: deepcopy(self._references[world])
            for world in self._worlds
            if world in self._selected
        }

    def set_selections(self, selected: dict[str, list[Reference]]) -> None:
        accepted = {
            world for world in set(selected).intersection(self._worlds) if self.available(world)
        }
        if not accepted and self._selected:
            return
        self._selected = accepted
        for world in self._selected:
            if self.is_collection(world):
                self._references[world] = self._predefined_choice(world, selected[world])
            else:
                self._references[world] = deepcopy(selected[world]) or ["Default"]
        for world in self._worlds:
            self._sync(world, counts=False)
        self._sync_counts()

    def _predefined_choice(self, world: str, references: list[Reference]) -> list[Reference]:
        """A collection world's references: all its predefined spawns, or in a tuning run the one named first
        (S1 when none matches)."""
        spawns = self._predefined_references(world)
        if not self._single or not spawns:
            return spawns
        wanted = {spawn_key(world, ref) for ref in references[:1]}
        return [next((s for s in spawns if spawn_key(world, s) in wanted), spawns[0])]

    def set_single(self, single: bool) -> None:
        """A tuning run allows one world with one starting pose: the first selected world and pose stay; a
        collection world offers its predefined spawns as a single choice."""
        was = self._single
        self._single = single
        changed = False
        if not single:
            for world in self._worlds:
                predefined = self._predefined_references(world)
                if self.is_collection(world) and self._references[world] != predefined:
                    self._references[world] = predefined
                    changed = True
                self._sync(world, counts=False)
            self._sync_counts()
            if changed and was:
                self.changed.emit()
            return
        kept = next((world for world in self._worlds if world in self._selected), None)
        for world in self._worlds:
            if world in self._selected and world != kept:
                self._selected.discard(world)
                self._close_popup(world)
                changed = True
            references = self._references.get(world, ["Default"])
            if len(references) > 1:
                self._references[world] = references[:1]
                changed = True
            self._sync(world, counts=False)
        self._sync_counts()
        if changed:
            self.changed.emit()

    def _choose_predefined(self, world: str, index: int) -> None:
        """A tuning run's choice on a collection card."""
        spawns = self._predefined_references(world)
        if not self._single or not 0 <= index < len(spawns):
            return
        if self._references.get(world) != [spawns[index]]:
            self._references[world] = [spawns[index]]
            self._sync(world)
            self.changed.emit()

    def select_world(self, world: str, checked: bool) -> None:
        if world not in self._worlds or (checked and not self.available(world)):
            return
        if not checked and self._selected == {world}:
            self._sync(world)
            return
        if checked:
            if self._single:  # a tuning run: choosing a world deselects the others
                for other in tuple(self._selected):
                    self._selected.discard(other)
                    self._close_popup(other)
                    self._sync(other, counts=False)
            self._selected.add(world)
        else:
            self._selected.discard(world)
            self._close_popup(world)
        self._sync(world)
        self.changed.emit()

    # time limits

    def time_limits(self) -> dict[str, float]:
        """The selected worlds' limits: the rule's values with the switch on (ValueError naming a world
        without one), else the boxes."""
        limits = {}
        for world in self._worlds:
            if world not in self._selected:
                continue
            if self.rule_on():
                rule = self._rule_limit(world)
                if rule.seconds is None:
                    raise ValueError(
                        f"{world_name(self._worlds[world])}: the kinematic rule gives no single limit "
                        "for the selected robots."
                    )
                limits[world] = rule.seconds
                continue
            limit = self._cards[world].time_limit.value()
            if not math.isfinite(limit) or limit < 0:
                raise ValueError(
                    f"{self._worlds[world]['label']}: enter a finite, nonnegative time limit in seconds."
                )
            limits[world] = limit
        return limits

    def limit_fields(self) -> dict[str, Any]:
        """The specification's time-limit keys: ``world_time_limits`` from the boxes, or with the switch on
        ``time_limit_rule`` and the rule's values (left to the engine when a world has none, so its
        validation names the world)."""
        if not self.rule_on():
            return {"world_time_limits": self.time_limits()}
        try:
            return {"time_limit_rule": TIME_LIMIT_RULE, "world_time_limits": self.time_limits()}
        except ValueError:
            return {"time_limit_rule": TIME_LIMIT_RULE}

    def shown_limit(self, world: str) -> float | None:
        """The limit a selected world's card shows (None: "—" under the rule, or an invalid entry)."""
        if world not in self._cards:
            return None
        if self.rule_on():
            return self._rule_limit(world).seconds
        limit = self._cards[world].time_limit.value()
        return limit if math.isfinite(limit) and limit >= 0 else None

    def set_time_limits(self, limits: dict[str, float]) -> None:
        for world, card in self._cards.items():
            card.time_limit.setValue(
                limits.get(world, self._worlds[world].get("default_time_limit", DEFAULT_LIMIT))
            )

    def rule_on(self) -> bool:
        return self.rule.isChecked()

    def set_rule(self, on: bool) -> None:
        """Turn the rule switch on or off (an imported campaign with ``time_limit_rule`` turns it on); it
        stays usable only while it is on or the catalog has rule values."""
        self.rule.setEnabled(on or rule_available(list(self._worlds.values())))
        self.rule.setChecked(on)
        self._sync_rule()

    def set_rule_scope(
        self, robots: Sequence[tuple[str, str]], excluded: set[tuple[str, str, str]]
    ) -> None:
        """The selected robots (ID, label) and the combinations left out: a world's rule limit is that of the
        robots with at least one combination in it."""
        robots, excluded = list(robots), set(excluded)
        if (robots, excluded) != (self._robots, self._excluded):
            self._robots, self._excluded = robots, excluded
            self._sync_rule()

    def _rule_robots(self, world: str) -> list[tuple[str, str]]:
        references = self._references.get(world, [])
        kept = [
            (robot, label)
            for robot, label in self._robots
            if any(
                (robot, world, spawn_key(world, ref)) not in self._excluded for ref in references
            )
        ]
        return kept or list(self._robots)

    def _rule_limit(self, world: str) -> RuleLimit:
        return rule_limit(self._worlds[world], self._rule_robots(world))

    def _rule_toggled(self, on: bool) -> None:
        self._sync_rule()
        self.changed.emit()

    def _sync_rule(self) -> None:
        on = self.rule_on()
        for world, card in self._cards.items():
            card.set_rule(self._rule_limit(world) if on else None)
        if on:
            labels = [label for _, label in self._robots]
            who = ", ".join(labels) if labels else "the selected robots"
            self.bar.set_note(f"· computed for {who}; hover a limit for its inputs")
        else:
            self.bar.set_note("")

    # spawns

    def set_spawns(self, world: str, rows: list[dict[str, Any]], has_more: bool = False) -> None:
        if world not in self._worlds:
            return
        previous = self._saved.get(world, {})
        current = {str(row["id"]): deepcopy(row) for row in rows}
        for key, row in current.items():
            pose = row.get("snapshot", row)
            if all(field in pose for field in ("x", "y", "yaw")):
                self._poses[world, key] = deepcopy(pose)
        # An edit saves the spawn under a new ID and deletes the old row: the selection follows its family.
        families = {
            row["family_id"]: key
            for key, row in current.items()
            if isinstance(row.get("family_id"), str)
        }
        references = self._references.get(world, [])
        followed = False
        for position, ref in enumerate(references):
            key = spawn_key(world, ref)
            if not isinstance(ref, str) or key in current:
                continue
            family = previous.get(key, {}).get("family_id")
            replacement = families.get(family) if isinstance(family, str) else None
            if replacement is not None and replacement not in references:
                references[position] = replacement
                followed = True
        for ref in references:
            key = spawn_key(world, ref)
            if key not in current and key in previous:
                current[key] = previous[key]
        self._saved[world] = current
        self._has_more[world] = has_more
        self._sync(world)
        if followed:
            self.changed.emit()

    def world_map(self, world: str) -> WorldMap | None:
        return self._maps.get(world)

    def ensure_map(self, world: str) -> None:
        """Ask for a world's map when nothing has (step 4 reached without showing step 2)."""
        if world in self._worlds and world not in self._maps:
            self._request(world)

    def name_of(self, world: str) -> str:
        """The collection ID or the label (summaries, Review)."""
        return world_name(self._worlds[world]) if world in self._worlds else world

    def spawn_label(self, world: str, reference: Reference) -> str:
        if isinstance(reference, dict):
            return str(reference["name"])
        predefined = self._predefined(world, reference)
        if predefined is not None:
            return str(predefined["name"])
        if spawn_key(world, reference) == "default-" + world:
            return "Default"
        row = self._saved.get(world, {}).get(str(reference), {})
        return str(row.get("name") or str(reference)[:8])

    def poses_text(self, world: str, references: Sequence[Reference]) -> str:
        """A world's starting poses in one line: a collection world running all its predefined spawns gives
        their range ("S1–S3"), otherwise the names."""
        if self.is_collection(world):
            spawns = self._predefined_references(world)
            keys = [spawn_key(world, ref) for ref in references]
            if keys == [spawn_key(world, spawn) for spawn in spawns]:
                return spawn_range([self.spawn_label(world, spawn) for spawn in spawns])
        return ", ".join(self.spawn_label(world, ref) for ref in references)

    def spawn_options(self) -> list[tuple[str, str, Reference, str]]:
        """Every world's poses for the Results filter: world ID and label, reference, and name."""
        return [
            (world, info["label"], deepcopy(entry.reference), entry.name)
            for world, info in self._worlds.items()
            for entry in self._rows(world)
        ]

    def forget_spawns(self, world: str, spawn_ids: set[str]) -> None:
        """Drop deleted saved spawns from this world's selection and saved rows."""
        references = self._references.get(world, [])
        kept = [r for r in references if not (isinstance(r, str) and r in spawn_ids)]
        saved = self._saved.get(world, {})
        removed = len(kept) != len(references) or bool(spawn_ids & set(saved))
        for identity in spawn_ids:
            saved.pop(identity, None)
        if not removed:
            return
        self._references[world] = kept or ["Default"]
        if world in self._worlds:
            self._sync(world)
        self.changed.emit()

    def _pose(self, world: str, reference: Reference) -> dict[str, Any] | None:
        if isinstance(reference, dict):
            return reference
        predefined = self._predefined(world, reference)
        if predefined is not None:
            return predefined
        key = spawn_key(world, reference)
        return (
            self._worlds[world]["default_spawn"]
            if key == "default-" + world
            else self._poses.get((world, key))
        )

    def _references_of(self, world: str) -> list[Reference]:
        """Default first, the registry's fixed spawns, then the library's spawns by name (05, 06, 15), then
        campaign snapshots; a collection world's predefined spawns."""
        if self.is_collection(world):
            return self._predefined_references(world)
        saved = sorted(
            self._saved.get(world, {}).items(),
            key=lambda item: (str(item[1].get("name") or item[0]).casefold(), item[0]),
        )
        # Default, the registry's fixed choices (decision 60), then the saved custom spawns.
        registry = [spawn["id"] for spawn in predefined_spawns(self._worlds[world])]
        references: list[Reference] = ["Default", *registry, *(key for key, _ in saved)]
        known = {spawn_key(world, ref) for ref in references}
        for ref in self._references.get(world, []):
            if spawn_key(world, ref) not in known or isinstance(ref, dict):
                references = [r for r in references if spawn_key(world, r) != spawn_key(world, ref)]
                references.append(deepcopy(ref))
                known.add(spawn_key(world, ref))
        return references

    def _rows(self, world: str) -> list[SpawnEntry]:
        rows = []
        for ref in self._references_of(world):
            key = spawn_key(world, ref)
            default = key == "default-" + world
            name = self.spawn_label(world, ref)
            fixed = self._predefined(world, ref) is not None
            detail = "· world file" if default else "· registry" if fixed else ""
            rows.append(
                SpawnEntry(
                    deepcopy(ref),
                    key,
                    name,
                    self._pose(world, ref),
                    detail,
                    (name + " " + detail).casefold(),
                )
            )
        return rows

    def _request(self, world: str, identity: str | None = None) -> None:
        """Ask for a saved spawn's pose (``identity``) or, without one, the world's map."""
        if (world, identity) in self._pending.values():
            return
        tag = "world-selection-" + uuid.uuid4().hex
        self._pending[tag] = (world, identity)
        self.transport.submit(
            tag,
            "spawns.get" if identity else "spawns.map",
            {"id": identity} if identity else {"world_id": world},
        )

    def _request_visible(self) -> None:
        """Maps load lazily: only the cards in view (or about to be) ask for theirs."""
        if not self.isVisible():
            return
        for world in self.sections.visible_worlds():
            if world not in self._maps and world not in self._map_errors:
                self._request(world)

    def _sync(self, world: str, refresh_popup: bool = True, counts: bool = True) -> None:
        """Bring the card (and the open popup, unless it made the change) up to date with the world's
        selection, and ask for the poses it does not know yet; ``counts`` also updates the chips and section
        heads (loops over every world pass False and update them once)."""
        card = self._cards[world]
        references = self._references[world]
        selected = world in self._selected
        keys = {spawn_key(world, ref) for ref in references}
        if self.is_collection(world):
            spawns = self._predefined_references(world)
            card.unavailable = not spawns
            names = (
                spawn_range([self.spawn_label(world, spawn) for spawn in spawns])
                if spawns
                else "No predefined spawns"
            )
            choosing = self._single and selected and len(spawns) > 1
            index = next((i for i, s in enumerate(spawns) if spawn_key(world, s) in keys), 0)
            card.set_choice(index if choosing else None)
            shown: list[Reference] = list(spawns) if choosing else list(references)
        else:
            names = ", ".join(self.spawn_label(world, ref) for ref in references)
            shown = list(references)
        card.set_state(selected, self._selected == {world}, names)
        marks = []
        for ref in shown:
            pose = self._pose(world, ref)
            picked = spawn_key(world, ref) in keys
            if pose is not None:
                marks.append(
                    SpawnMark(
                        self.spawn_label(world, ref),
                        float(pose["x"]),
                        float(pose["y"]),
                        float(pose["yaw"]),
                        picked=picked,
                        labelled=True,
                    )
                )
            # Saved spawns are asked for; predefined ones come with the catalog.
            elif selected and not self.is_collection(world):
                self._request(world, spawn_key(world, ref))
        card.stage.set_marks(marks if selected else [])
        if self.rule_on():
            card.set_rule(self._rule_limit(world))
        popup = self._popups.get(world)
        if popup is not None:
            popup.set_has_more(self._has_more.get(world, False))
            if refresh_popup and popup.isVisible():
                popup.refresh(self._rows(world), references)
        for identity, other in self._cards.items():
            if identity != world:
                other.set_state(other.checked, self._selected == {identity}, other.names)
        if counts:
            self._sync_counts()

    # the starting-poses popup (original worlds)

    def _popup_rect(self, world: str, host: PopupHost, has_more: bool) -> QtCore.QRect:
        """HANDOFF §8.6: as wide as in 06 (two columns of three; three of four), from the clicked card's
        column (ending at the last one). With its footer row the popup ends 5 px into the card's Starting
        poses row, above the Spawns button (06); without it (D2: the row only carries the 250-spawn note) it
        keeps that top edge."""
        sections = self.sections
        card = sections.boxes[world]
        column = sections.column_of(world)
        span = min(3 if sections.columns >= 4 else 2, sections.columns)
        start = max(0, min(column, sections.columns - span))
        step = sections.card_width + sections.GAP
        origin = sections.content.mapTo(host.window_widget, QtCore.QPoint(0, 0))
        left = snap(origin.x() + card.x() - (column - start) * step)
        width = span * sections.card_width + (span - 1) * sections.GAP
        bottom = snap(origin.y() + card.y() + WorldCard.ROWS_TOP + 5)
        top = max(8, bottom - SpawnsPopup.popup_height(True))
        return QtCore.QRect(left, top, math.floor(width), SpawnsPopup.popup_height(has_more))

    def show_spawns(self, world: str) -> None:
        host = self.window().findChild(PopupHost)
        if (
            world not in self._selected
            or host is None
            or self.is_collection(world)
            or world not in self.sections.boxes
        ):
            return
        popup = self._popups.get(world)
        if popup is None:
            popup = SpawnsPopup(self.themes, self._worlds[world])
            popup.changed.connect(lambda refs, wid=world: self._set_spawns(wid, refs))
            popup.finished.connect(lambda wid=world: self._close_popup(wid))
            self._popups[world] = popup
        has_more = self._has_more.get(world, False)
        popup.stage.set_world(self._maps.get(world))
        if world in self._map_errors:
            popup.stage.set_message("Map unavailable")
        popup.set_has_more(has_more)
        popup.refresh(self._rows(world), self._references[world])
        card = self._cards[world]
        card.spawns_button.set_open(True)
        host.open_popup(
            popup,
            self._popup_rect(world, host, has_more),
            opener=card.spawns_button,
            on_close=lambda: card.spawns_button.set_open(False),
        )
        popup.view.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def _close_popup(self, world: str) -> None:
        popup = self._popups.get(world)
        host = self.window().findChild(PopupHost)
        if popup is not None and host is not None and host.popup is popup:
            host.close_popup()

    def _close_popups(self) -> None:
        for world in self._popups:
            self._close_popup(world)

    def _drop_popup(self, world: str) -> None:
        self._close_popup(world)
        popup = self._popups.pop(world, None)
        if popup is not None:
            popup.hide()
            popup.deleteLater()

    def _set_spawns(self, world: str, references: list[Reference]) -> None:
        if not references:
            return
        if self._single and len(references) > 1:  # a tuning run: the newly ticked pose only
            before = {spawn_key(world, r) for r in self._references.get(world, [])}
            added = [r for r in references if spawn_key(world, r) not in before]
            self._references[world] = [deepcopy((added or references)[-1])]
            self._sync(world)
            self.changed.emit()
            return
        self._references[world] = deepcopy(references)
        self._sync(world, refresh_popup=False)  # the popup holds this selection already
        self.changed.emit()

    # replies

    def received(self, tag: str, result: Any) -> None:
        pending = self._pending.pop(tag, None)
        if pending is None:
            return
        world, identity = pending
        if world not in self._worlds or result.get("world_id") != world:
            return
        card = self._cards[world]
        if identity:
            if result.get("id") != identity:
                return
            self._poses[world, identity] = deepcopy(result)
            card.pose_error = ""
            self._sync(world)
            return
        try:
            world_map = WorldMap.from_reply(result)
        except (KeyError, TypeError, ValueError) as exc:
            self._map_failed(world, str(exc) or "Unreadable world map")
            return
        card.map_error = ""
        self._map_errors.pop(world, None)
        self._maps[world] = world_map
        card.stage.set_world(world_map)
        popup = self._popups.get(world)
        if popup is not None:
            popup.stage.set_world(world_map)
        self.map_loaded.emit(world, world_map)

    def failed(self, tag: str, error: str) -> None:
        pending = self._pending.pop(tag, None)
        if pending and pending[0] in self._cards:
            if pending[1] is None:
                self._map_failed(pending[0], error)
            else:
                self._cards[pending[0]].pose_error = error

    def _map_failed(self, world: str, error: str) -> None:
        """Every map of the world says so (card, popup, the Accepted card through ``map_loaded``); the
        next showing asks again."""
        self._map_errors[world] = error
        card = self._cards[world]
        card.map_error = error
        card.stage.set_message("Map unavailable")
        popup = self._popups.get(world)
        if popup is not None:
            popup.stage.set_message("Map unavailable")
        self.map_loaded.emit(world, None)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._map_errors.clear()  # the next showing asks again
        self._request_visible()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._close_popups()
        super().hideEvent(event)
