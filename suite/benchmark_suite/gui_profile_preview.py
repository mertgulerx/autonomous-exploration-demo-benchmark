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

from .gui_style import humanize, label
from .gui_support import Transport


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
    name: str
    default: Any
    value: Any
    description: str


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
            differences.append(ParameterDifference(humanize(key), before, after, description))
    if robot["id"] == "mogi":
        physical = profile.get("robot", {}).get("benchmark_settings", {}).get("effective", {})
        overrides = profile.get("robot_overrides", {})
        for key, name in (
            ("movement_speed", "Movement speed (m/s)"),
            ("turning_speed", "Turning speed (rad/s)"),
            ("lidar_range", "Lidar range (m)"),
        ):
            before = robot[key]
            after = physical.get(key, overrides.get(key, before))
            if before != after:
                differences.append(
                    ParameterDifference(
                        "Robot · " + name, before, after, "MOGI physical settings · " + key
                    )
                )
    return differences


def _value_text(value: Any) -> str:
    return "Unavailable" if value is None else json.dumps(value, ensure_ascii=False)


class ProfilePreview(QtWidgets.QWidget):
    def __init__(
        self, robot: dict[str, Any], metadata: dict[str, Any], transport: Transport | None
    ) -> None:
        super().__init__()
        self.robot, self.metadata, self.transport = deepcopy(robot), deepcopy(metadata), transport
        self.reference: Any = None
        self._key = ""
        self._pending: tuple[str, str, Any] | None = None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._loaded_key = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.title = label("Changes from Default", "section")
        layout.addWidget(self.title)
        self.summary = label("", "muted")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Parameter", "Default", "Profile"])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        for column in (1, 2):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.Fixed
            )
            self.table.setColumnWidth(column, 100)
        self.table.verticalHeader().setDefaultSectionSize(38)
        layout.addWidget(self.table, 1)
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
        self.reference, self._key = deepcopy(reference), key
        self.title.setText(name)
        self.table.setRowCount(0)
        self.summary.setProperty("kind", "muted")
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)
        self.summary.setText("Loading changes from Default…")
        self.summary.setToolTip("")
        self.timer.start()

    def matches(self, reference: Any) -> bool:
        return self._key == profile_key(self.robot["id"], reference)

    def _load(self) -> None:
        if not self.isVisible() or not self._key:
            return
        if self._key == "default":
            self._show({"effective": self.robot.get("defaults", {})})
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
        differences = profile_differences(self.robot, self.metadata, profile)
        self.table.setRowCount(len(differences))
        for row, difference in enumerate(differences):
            for column, text in enumerate(
                (difference.name, _value_text(difference.default), _value_text(difference.value))
            ):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(difference.description if column == 0 else text)
                if column == 2:
                    font = item.font()
                    font.setWeight(QtGui.QFont.Weight.DemiBold)
                    item.setFont(font)
                self.table.setItem(row, column, item)
        count = len(differences)
        self.summary.setText(
            f"{count} {'change' if count == 1 else 'changes'} from {self.robot['label']} Default"
            if count
            else f"Matches {self.robot['label']} Default"
        )
        self._loaded_key = self._key

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
        self.table.setRowCount(0)
        self.summary.setProperty("kind", "error")
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)
        self.summary.setText(message)
        self.summary.setToolTip(message)
        self._loaded_key = ""

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._loaded_key != self._key:
            self.timer.start()
