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

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .contracts import HIDDEN_GROUPS
from .gui_menus import show_tooltip
from .gui_style import cell_text, humanize
from .gui_support import Transport
from .gui_theme import ThemeManager
from .gui_widgets import (
    BODY,
    TextStyle,
    Themed,
    draw_text,
    paint_scroll_handle,
    plural,
    wheel_step,
)

BASELINE = "Robot-aware baseline"  # the Default profile's description


def profile_key(robot_id: str, reference: Any) -> str:
    if isinstance(reference, dict):
        serialized = json.dumps(reference, sort_keys=True, ensure_ascii=False, default=str)
        return "snapshot:" + hashlib.sha256(serialized.encode()).hexdigest()
    if reference in ("Default", "default", "default-" + robot_id):
        return "default"
    return "profile:" + str(reference)


def _same_value(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return bool(left == right)


@dataclass(frozen=True)
class ParameterDifference:
    key: str
    name: str
    unit: str
    default: Any
    value: Any
    description: str


# HANDOFF §12.3: a trailing unit word becomes the muted unit, as written (13 draws "hz"), except where it
# belongs to the name.
UNIT_WORDS = frozenset({"cells", "m", "s", "hz", "deg"})
NAMES_WITH_UNIT_WORDS = frozenset({"Sigma s", "Sigma r"})
# MOGI's physical settings (``robot_overrides``): key, name, unit; the defaults are the catalog robot's.
PHYSICAL_SETTINGS = (
    ("movement_speed", "Movement speed", "m/s"),
    ("turning_speed", "Turning speed", "rad/s"),
    ("lidar_range", "Lidar range", "m"),
)
PHYSICAL_GROUP = "MOGI physical"
# The parameter schema's group order (``parameters.yaml``). Engine replies are canonical JSON with sorted keys, so
# the metadata arrives in key order and cannot give the groups their order.
GROUP_ORDER = ("Behavior", "MRTSP Scoring", "Navigation Safety", "Map Optimization", "Preemption")


def parameter_label(key: str) -> tuple[str, str]:
    """The humanized parameter name and its unit ("Min frontier size", "cells")."""
    name = humanize(key)
    words = name.split(" ")
    unit = words[-1].lower()
    if len(words) > 1 and unit in UNIT_WORDS and name not in NAMES_WITH_UNIT_WORDS:
        return " ".join(words[:-1]), unit
    return name, ""


def value_text(value: Any) -> str:
    """Numbers as stored (``2.0`` stays ``2.0``), booleans ``true``/``false``, choices as text."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def profile_differences(
    robot: dict[str, Any], metadata: dict[str, Any], profile: dict[str, Any]
) -> list[ParameterDifference]:
    defaults, effective = robot["defaults"], profile["effective"]
    differences = []
    for key in dict.fromkeys([*defaults, *effective]):
        information = metadata.get(key, {})
        if information.get("managed", False):
            continue
        before, after = defaults.get(key), effective.get(key)
        if not _same_value(before, after):
            description = f"{information.get('group', 'Explorer')} · {key}"
            if information.get("description"):
                description += "\n" + information["description"]
            name, unit = parameter_label(key)
            differences.append(ParameterDifference(key, name, unit, before, after, description))
    if robot["id"] == "mogi":
        for key, name, unit in PHYSICAL_SETTINGS:
            before, after = robot[key], physical_value(profile, key, robot[key])
            if before != after:
                differences.append(
                    ParameterDifference(
                        key,
                        "Robot · " + name,
                        unit,
                        before,
                        after,
                        "MOGI physical settings · " + key,
                    )
                )
    return differences


def physical_value(profile: dict[str, Any], key: str, default: Any) -> Any:
    """A MOGI physical setting of a profile: the resolved value, else the override, else ``default``."""
    physical = profile.get("robot", {}).get("benchmark_settings", {}).get("effective", {})
    return physical.get(key, profile.get("robot_overrides", {}).get(key, default))


def editor_groups(metadata: dict[str, Any], robot_id: str) -> list[str]:
    """The parameter groups the profile editor shows, in the schema's order (no managed parameters, no hidden
    groups); MOGI adds its physical settings last."""
    groups: list[str] = []
    for information in metadata.values():
        group = information.get("group", "")
        if information.get("managed") or group in HIDDEN_GROUPS or group in groups:
            continue
        groups.append(group)
    known = {group: index for index, group in enumerate(GROUP_ORDER)}
    groups.sort(key=lambda group: known.get(group, len(known)))  # unknown groups after, as found
    if robot_id == "mogi":
        groups.append(PHYSICAL_GROUP)
    return groups


@dataclass(frozen=True, slots=True)
class ComparedParameter:
    """One parameter of a profile against its robot's Default."""

    key: str
    name: str
    unit: str
    group: str
    default: Any
    value: Any
    description: str  # the catalog's text, may be empty

    @property
    def changed(self) -> bool:
        return not _same_value(self.default, self.value)


def compared_parameters(
    robot: dict[str, Any], metadata: dict[str, Any], profile: dict[str, Any] | None
) -> list[ComparedParameter]:
    """Every parameter the editor shows, in editor order (groups as ``editor_groups``, the catalog's order
    within a group), with the robot's Default and the profile's value (Default against itself for None),
    plus any other changed parameter outside the managed ones. Changed rows are exactly
    ``profile_differences``'."""
    defaults = robot["defaults"]
    effective = profile["effective"] if profile is not None else defaults
    groups = editor_groups(metadata, robot["id"])
    rows: list[ComparedParameter] = []
    for key in dict.fromkeys([*defaults, *effective]):
        information = metadata.get(key, {})
        if information.get("managed", False):
            continue
        group = information.get("group", "Explorer")
        before, after = defaults.get(key), effective.get(key)
        if group not in groups and _same_value(before, after):
            continue  # a hidden group's parameter is shown only when it differs
        name, unit = parameter_label(key)
        description = str(information.get("description", ""))
        rows.append(ComparedParameter(key, name, unit, group, before, after, description))
    if robot["id"] == "mogi":
        for key, name, unit in PHYSICAL_SETTINGS:
            after = robot[key] if profile is None else physical_value(profile, key, robot[key])
            rows.append(ComparedParameter(key, name, unit, PHYSICAL_GROUP, robot[key], after, ""))
    order = {group: index for index, group in enumerate(groups)}
    return sorted(rows, key=lambda row: order.get(row.group, len(order)))


def _decimals(value: float | int) -> int:
    if isinstance(value, int):
        return 0
    text = repr(value)
    return 0 if "e" in text or "." not in text else len(text.split(".")[1])


def signed_change(default: Any, value: Any) -> str:
    """A number's change from its Default with a sign (U+2212 for minus) in the values' precision ("−1.0",
    "−15", "+2"); empty for other values."""
    numbers = (int, float)
    if (
        isinstance(default, bool)
        or isinstance(value, bool)
        or not isinstance(default, numbers)
        or not isinstance(value, numbers)
    ):
        return ""
    places = max(_decimals(default), _decimals(value))
    delta = round(value - default, places)
    if delta == 0:
        return ""
    return ("+" if delta > 0 else "\u2212") + f"{abs(delta):.{places}f}"


BAR_TEXT = TextStyle(12, 500, 18)
BAR_NAME = TextStyle(12, 600, 18)
CREATED = TextStyle(12, 500, 18, tabular=True)
VALUE_TEXT = TextStyle(13, 400, 18, tabular=True)
NEW_VALUE = TextStyle(13, 600, 18, tabular=True)
OLD_COLUMN, NEW_COLUMN = 120, 140  # ``.pp-g``: grid-template-columns 1fr 120px 140px


class ProfilePreview(Themed):
    """``.pp-prev``: the inspected profile (04): a 36 px sunken bar with its name and creation time, its
    description, then "N changes from <robot> Default" over the two value columns (120 and 140 px) and one
    40 px row per changed parameter (name and muted unit, Default value, the profile's value in 600)."""

    BAR, ROW = 36, 40

    def __init__(
        self,
        themes: ThemeManager,
        robot: dict[str, Any],
        metadata: dict[str, Any],
        transport: Transport | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.robot, self.metadata, self.transport = deepcopy(robot), deepcopy(metadata), transport
        self.reference: Any = None
        self.name = ""
        self._key = ""
        self._pending: tuple[str, str, Any] | None = None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._loaded_key = ""
        self.profile: dict[str, Any] | None = None
        self.differences: list[ParameterDifference] = []
        self.message = ""  # loading or error text in place of the description
        self.error = False
        self.offset = 0.0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._load)
        if transport:
            transport.received.connect(self.received)
            transport.failed.connect(self.failed)

    def inspect(self, reference: Any, name: str) -> None:
        key = profile_key(self.robot["id"], reference)
        if self._key == key and (self._loaded_key == key or self.timer.isActive()):
            return
        self.reference, self._key, self.name = deepcopy(reference), key, name
        self.profile, self.differences, self.offset = None, [], 0.0
        self.message, self.error = "Loading changes from Default…", False
        self.update()
        self.timer.start()

    def matches(self, reference: Any) -> bool:
        return self._key == profile_key(self.robot["id"], reference)

    def _load(self) -> None:
        if not self.isVisible() or not self._key:
            return
        if self._key == "default":
            self._show({"effective": self.robot.get("defaults", {}), "description": BASELINE})
        elif self._key in self._cache:
            self._cache.move_to_end(self._key)
            self._show(self._cache[self._key])
        elif isinstance(self.reference, dict) and isinstance(self.reference.get("effective"), dict):
            self._show(self.reference)
        elif self._pending:
            return
        elif self.transport is None:
            self._error("Connect to the engine to view this profile's changes.")
        else:
            tag = "profile-preview-" + uuid.uuid4().hex
            self._pending = (tag, self._key, deepcopy(self.reference))
            if isinstance(self.reference, dict):
                self.transport.submit(
                    tag, "parameters.resolve", dict(self.reference, robot_id=self.robot["id"])
                )
            else:
                self.transport.submit(tag, "profiles.get", {"revision_id": self.reference})

    def _show(self, profile: dict[str, Any]) -> None:
        if not self.robot.get("defaults"):
            self._error("Default parameter values are unavailable. Refresh the robot catalog.")
            return
        self.profile = profile
        self.differences = profile_differences(self.robot, self.metadata, profile)
        self.message, self.error = "", False
        self._loaded_key = self._key
        self.update()

    def received(self, tag: str, result: Any) -> None:
        if self._pending is None or tag != self._pending[0]:
            return
        _, key, reference = self._pending
        self._pending = None
        if (
            result.get("robot_id") != self.robot["id"]
            or not isinstance(result.get("effective"), dict)
            or isinstance(reference, str)
            and result.get("id") != reference
        ):
            if key == self._key:
                self._error("The profile response does not match this robot or revision.")
            else:
                self._load()
            return
        self._cache[key] = deepcopy(result)
        self._cache.move_to_end(key)
        while len(self._cache) > 32:
            self._cache.popitem(last=False)
        if key == self._key:
            self._show(result)
        else:
            self._load()

    def failed(self, tag: str, error: str) -> None:
        if self._pending is None or tag != self._pending[0]:
            return
        _, key, _ = self._pending
        self._pending = None
        if key == self._key:
            self._error("Cannot load profile changes: " + error)
        else:
            self._load()

    def _error(self, message: str) -> None:
        self.profile, self.differences = None, []
        self.message, self.error = message, True
        self._loaded_key = ""
        self.update()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._loaded_key != self._key:
            self.timer.start()

    def summary(self) -> tuple[str, str]:
        """The header's bold count and the rest ("3 changes", " from MOGI Default")."""
        count = len(self.differences)
        if not count:
            return "", f"No changes from {self.robot['label']} Default"
        return plural(count, "change"), f" from {self.robot['label']} Default"

    # The bar, the description and the header stay; the difference rows scroll under them.

    def rows_top(self) -> int:
        return self.BAR + 2 * self.ROW

    def max_offset(self) -> float:
        return max(0.0, len(self.differences) * self.ROW - (self.height() - self.rows_top()))

    def scroll_to(self, offset: float) -> None:
        offset = min(max(0.0, offset), self.max_offset())
        if offset != self.offset:
            self.offset = offset
            self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self.scroll_to(self.offset - wheel_step(event, self.ROW).y())
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.scroll_to(self.offset)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        self._paint_bar(painter)
        self._paint_description(painter)
        if self.profile is None:
            return
        self._paint_header(painter)
        self._paint_rows(painter)
        total = len(self.differences) * self.ROW
        visible = self.height() - self.rows_top()
        paint_scroll_handle(painter, self, self.rows_top(), visible, self.offset, total)

    def _paint_bar(self, painter: QtGui.QPainter) -> None:
        """The sunken bar: the name and, at the right, when the revision was made."""
        theme, width = self.theme, self.width()
        left, right = 20.0, width - 20.0
        painter.fillRect(QtCore.QRect(0, 0, width, self.BAR), theme.color("surface.sunken"))
        painter.fillRect(QtCore.QRect(0, self.BAR - 1, width, 1), theme.color("border.line"))
        top = (self.BAR - 1 - 18) / 2
        created = (self.profile or {}).get("created_ns")
        when = "created " + cell_text("created_ns", created) if isinstance(created, int) else ""
        if when:
            x = right - CREATED.advance(when)
            draw_text(painter, x, top, when, CREATED, theme.color("text.secondary"))
        room = right - left - (CREATED.advance(when) + 10 if when else 0)
        name = BAR_NAME.elide(self.name, max(0.0, room))
        draw_text(painter, left, top, name, BAR_NAME, theme.color("text.ink"))

    def _paint_description(self, painter: QtGui.QPainter) -> None:
        """The profile's description, or the loading or error text."""
        theme, width = self.theme, self.width()
        description = self.message or str((self.profile or {}).get("description") or "")
        token = "state.bad" if self.error else "text.secondary"
        top = self.BAR + (self.ROW - 1 - 18) / 2
        text = BODY.elide(description, width - 40)
        draw_text(painter, 20, top, text, BODY, theme.color(token))
        rule = QtCore.QRect(0, self.BAR + self.ROW - 1, width, 1)
        painter.fillRect(rule, theme.color("border.line"))

    def _paint_header(self, painter: QtGui.QPainter) -> None:
        """The count of changes from the robot's Default and, when there are changes, the two value
        columns."""
        theme, width = self.theme, self.width()
        right = width - 20.0
        y = self.BAR + self.ROW
        top = y + (self.ROW - 1 - 18) / 2
        secondary = theme.color("text.secondary")
        count, rest = self.summary()
        x = 20 + draw_text(painter, 20, top, count, BAR_NAME, theme.color("text.ink"))
        draw_text(painter, x, top, rest, BAR_TEXT, secondary)
        if self.differences:
            for text, column_right, column in (
                ("Default", right - NEW_COLUMN, OLD_COLUMN),
                (self.name, right, NEW_COLUMN),
            ):
                text = BAR_TEXT.elide(text, column)
                x = column_right - BAR_TEXT.advance(text)
                draw_text(painter, x, top, text, BAR_TEXT, secondary)
        painter.fillRect(QtCore.QRect(0, y + self.ROW - 1, width, 1), theme.color("border.line"))

    def _paint_rows(self, painter: QtGui.QPainter) -> None:
        """One row per difference: name and muted unit, the Default value, the profile's value in 600;
        scrolled, under the header."""
        theme, width = self.theme, self.width()
        right = width - 20.0
        ink, secondary = theme.color("text.ink"), theme.color("text.secondary")
        painter.save()
        painter.setClipRect(
            QtCore.QRect(0, self.rows_top(), width, self.height() - self.rows_top())
        )
        row_y = self.rows_top() - self.offset
        for index, difference in enumerate(self.differences):
            last = index == len(self.differences) - 1
            if row_y + self.ROW > self.rows_top() and row_y < self.height():
                top = row_y + (self.ROW - (0 if last else 1) - 18) / 2
                label = BODY.elide(difference.name, right - 20 - OLD_COLUMN - NEW_COLUMN)
                draw_text(painter, 20, top, label, BODY, ink)
                if difference.unit:
                    unit_x = 20 + BODY.advance(label + " ") + 2
                    muted = theme.color("text.tertiary")
                    draw_text(painter, unit_x, top, difference.unit, BODY, muted)
                old, new = value_text(difference.default), value_text(difference.value)
                old_x = right - NEW_COLUMN - VALUE_TEXT.advance(old)
                draw_text(painter, old_x, top, old, VALUE_TEXT, secondary)
                draw_text(painter, right - NEW_VALUE.advance(new), top, new, NEW_VALUE, ink)
                if not last:
                    rule = QtCore.QRectF(0, row_y + self.ROW - 1, width, 1)
                    painter.fillRect(rule, theme.color("border.line"))
            row_y += self.ROW
        painter.restore()

    def event(self, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if kind in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self.update()  # the scroll handle shows while the pointer is over the preview
        elif kind == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            row = -1
            if event.pos().y() >= self.rows_top():
                row = int((event.pos().y() - self.rows_top() + self.offset) // self.ROW)
            text = self.differences[row].description if 0 <= row < len(self.differences) else ""
            show_tooltip(self, event.globalPos(), text)
            return True
        return super().event(event)
