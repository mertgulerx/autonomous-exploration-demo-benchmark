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

from copy import deepcopy
from typing import Any
import uuid

import yaml
from PySide6 import QtCore, QtWidgets

from .gui_style import DetailsPanel, label, surface
from .gui_support import FloatEdit, MapView, Transport, button
from .gui_robots import RobotSelector


class CampaignBuilder(QtWidgets.QWidget):
    edit_profile = QtCore.Signal(str, object)
    edit_spawn = QtCore.Signal(str)

    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self.catalog: dict[str, Any] = {}
        self.source_campaign_id: str | None = None
        self.excluded: set[tuple[str, str, str]] = set()
        self.validated: dict[str, Any] | None = None
        self.validation_tag = ""
        self.submission_tag = ""
        self.map_tag = ""
        self.pose_tag = ""
        self.import_tag = ""
        self.step = 0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        steps = QtWidgets.QHBoxLayout()
        self.step_buttons = []
        for index, title in enumerate(
            ("Robots & profiles", "Worlds & spawns", "Execution", "Review")
        ):
            control = button(
                f"{index + 1}  {title.replace('&', '&&')}",
                lambda checked=False, i=index: self.show_step(i),
                steps,
            )
            self.step_buttons.append(control)
        layout.addLayout(steps)
        self.summary = label("Choose your robots and explorer profiles", "muted")
        layout.addWidget(self.summary)
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.robot_selector = RobotSelector()
        self.robot_selector.edit_profile.connect(self.edit_profile.emit)
        self.stack.addWidget(self.robot_selector)
        worlds = QtWidgets.QWidget()
        world_layout = QtWidgets.QHBoxLayout(worlds)
        world_layout.setContentsMargins(0, 0, 0, 0)
        self.world_tree = self.tree(["World / spawn pose", "Position · heading"])
        self.world_tree.setMinimumWidth(340)
        panel = surface()
        pane = QtWidgets.QVBoxLayout(panel)
        pane.setContentsMargins(16, 16, 16, 16)
        pane.addWidget(label("Worlds and starting positions", "section"))
        pane.addWidget(self.world_tree, 1)
        button("Edit spawn poses", self.edit_selected_spawn, pane)
        split = QtWidgets.QSplitter()
        split.setObjectName("builder-world-split")
        split.addWidget(panel)
        self.world_map = MapView()
        split.addWidget(self.world_map)
        split.setStretchFactor(1, 2)
        world_layout.addWidget(split)
        self.stack.addWidget(worlds)
        execution = surface()
        execution_layout = QtWidgets.QVBoxLayout(execution)
        execution_layout.setContentsMargins(24, 24, 24, 24)
        execution_layout.addWidget(
            label("Name your study and choose execution settings", "section")
        )
        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(16)
        self.name = QtWidgets.QLineEdit("Exploration study")
        self.description = QtWidgets.QLineEdit()
        self.description.setPlaceholderText("Optional research notes")
        self.repetitions = QtWidgets.QSpinBox()
        self.repetitions.setRange(1, 1000000)
        self.repetitions.setValue(1)
        presets = QtWidgets.QWidget()
        preset_layout = QtWidgets.QHBoxLayout(presets)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(self.repetitions, 1)
        for count in (1, 10, 100):
            button(
                str(count),
                lambda checked=False, n=count: self.repetitions.setValue(n),
                preset_layout,
            )
        self.concurrency = QtWidgets.QComboBox()
        for count in (1, 2, 3, 4):
            self.concurrency.addItem(
                "Sequential" if count == 1 else f"{count} parallel workers", count
            )
        self.limit = FloatEdit(1800)
        form.addRow("Campaign name", self.name)
        form.addRow("Description", self.description)
        form.addRow("Repetitions per combination", presets)
        form.addRow("Execution", self.concurrency)
        form.addRow("Exploration time limit · seconds", self.limit)
        execution_layout.addLayout(form)
        execution_layout.addWidget(
            label("Time starts at first planar movement. Set 0 to disable the limit.", "muted")
        )
        execution_layout.addWidget(
            label(
                "Timing-rejected parallel attempts receive one final sequential replacement.",
                "muted",
            )
        )
        execution_layout.addStretch()
        self.stack.addWidget(execution)
        review = QtWidgets.QWidget()
        review_layout = QtWidgets.QVBoxLayout(review)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.addWidget(
            label("Review combinations before adding the campaign to the queue", "section")
        )
        review_layout.addWidget(
            label(
                "Uncheck a combination to exclude it explicitly. Validation checks every included spawn.",
                "muted",
            )
        )
        self.compatibility = QtWidgets.QTableWidget(0, 4)
        self.compatibility.setHorizontalHeaderLabels(["Include", "Robot", "World", "Spawn"])
        self.compatibility.verticalHeader().hide()
        self.compatibility.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.compatibility.setShowGrid(False)
        self.compatibility.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.compatibility.itemChanged.connect(self.change_exclusion)
        review_layout.addWidget(self.compatibility, 2)
        self.preview = DetailsPanel("Accepted campaign")
        review_layout.addWidget(self.preview, 1)
        self.stack.addWidget(review)
        self.error = label("", "error")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        footer = QtWidgets.QHBoxLayout()
        button("Import YAML", self.import_campaign, footer, role="quiet")
        button("Export YAML", self.export_campaign, footer, role="quiet")
        footer.addStretch()
        self.back_button = button("Back", lambda: self.show_step(self.step - 1), footer)
        self.next_button = button("Continue", self.next_step, footer, role="primary")
        self.submit_button = button(
            "Add to queue", lambda: self.submit_design(True), footer, role="primary"
        )
        layout.addLayout(footer)
        self.robot_selector.changed.connect(self.invalidate)
        self.world_tree.itemChanged.connect(self.invalidate)
        self.world_tree.currentItemChanged.connect(self.load_map)
        for field in (self.name, self.description):
            field.textChanged.connect(self.invalidate)
        self.repetitions.valueChanged.connect(self.invalidate)
        self.concurrency.currentIndexChanged.connect(self.invalidate)
        self.limit.textChanged.connect(self.invalidate)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)
        self.show_step(0)

    @staticmethod
    def tree(headers: list[str]) -> QtWidgets.QTreeWidget:
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(headers)
        tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setStretchLastSection(True)
        tree.setIndentation(24)
        tree.setAlternatingRowColors(True)
        return tree

    def selections(self, tree: QtWidgets.QTreeWidget) -> dict[str, list[Any]]:
        values: dict[str, list[Any]] = {}
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            assert item is not None
            if item.checkState(0) == QtCore.Qt.CheckState.Checked and not item.isDisabled():
                values[item.data(0, QtCore.Qt.ItemDataRole.UserRole)] = [
                    item.child(i).data(0, QtCore.Qt.ItemDataRole.UserRole)
                    for i in range(item.childCount())
                    if item.child(i).checkState(0) == QtCore.Qt.CheckState.Checked
                ]
        return values

    def design(self) -> dict[str, Any]:
        worlds = self.selections(self.world_tree)
        robots = self.robot_selector.selections()
        combinations = {
            (robot, world, self.spawn_identity(world, spawn))
            for robot in robots
            for world, spawns in worlds.items()
            for spawn in spawns
        }
        return {
            "name": self.name.text(),
            "description": self.description.text(),
            "robots": robots,
            "worlds": worlds,
            "repetitions": self.repetitions.value(),
            "concurrency": self.concurrency.currentData(),
            "time_limit": self.limit.value(),
            "exclusions": [list(v) for v in sorted(self.excluded & combinations)],
            **({"source_campaign_id": self.source_campaign_id} if self.source_campaign_id else {}),
        }

    def invalidate(self, *args: Any) -> None:
        self.validated = None
        self.validation_tag = ""
        self.submit_button.setEnabled(False)
        self.update_summary()

    def update_summary(self) -> None:
        robots, worlds = self.robot_selector.selections(), self.selections(self.world_tree)
        count = 0
        for robot, profiles in robots.items():
            for world, spawns in worlds.items():
                count += len(profiles) * sum(
                    (robot, world, self.spawn_identity(world, s)) not in self.excluded
                    for s in spawns
                )
        count *= self.repetitions.value()
        self.summary.setText(
            f"{len(robots)} robots  ·  {sum(map(len, robots.values()))} profiles  ·  {len(worlds)} worlds  ·  {self.repetitions.value():,} repetitions  ·  {count:,} planned experiments"
        )

    @staticmethod
    def spawn_identity(world: str, reference: Any) -> str:
        return (
            str(reference["id"])
            if isinstance(reference, dict)
            else "default-" + world
            if reference == "Default"
            else str(reference)
        )

    def show_step(self, index: int) -> None:
        self.step = max(0, min(index, 3))
        self.stack.setCurrentIndex(self.step)
        for i, control in enumerate(self.step_buttons):
            control.setProperty("role", "selected" if i == self.step else "quiet")
            control.style().unpolish(control)
            control.style().polish(control)
        self.back_button.setVisible(self.step > 0)
        self.next_button.setText("Validate campaign" if self.step == 3 else "Continue")
        self.next_button.setProperty(
            "role", "secondary" if self.step == 3 and self.validated else "primary"
        )
        self.next_button.style().unpolish(self.next_button)
        self.next_button.style().polish(self.next_button)
        self.submit_button.setVisible(self.step == 3)
        self.submit_button.setEnabled(self.validated is not None and not self.submission_tag)
        if self.step == 3:
            self.refresh_combinations()
        self.update_summary()

    def set_message(self, text: str, kind: str = "error") -> None:
        self.error.setProperty("kind", kind)
        self.error.style().unpolish(self.error)
        self.error.style().polish(self.error)
        self.error.setText(text)

    def next_step(self) -> None:
        if self.step == 0:
            selected = self.robot_selector.selections()
            if not selected or any(not values for values in selected.values()):
                self.set_message("Select at least one robot and a profile for each selected robot.")
                return
        if self.step == 1:
            selected = self.selections(self.world_tree)
            if not selected or any(not values for values in selected.values()):
                self.set_message(
                    "Select at least one world and a starting pose for each selected world."
                )
                return
        self.error.clear()
        if self.step == 3:
            self.submit_design(False)
        else:
            self.show_step(self.step + 1)

    def refresh_combinations(self) -> None:
        self.compatibility.blockSignals(True)
        self.compatibility.setRowCount(0)
        robots = {r["id"]: r for r in self.catalog.get("robots", [])}
        worlds = {w["id"]: w for w in self.catalog.get("worlds", [])}
        for robot in self.robot_selector.selections():
            for world, spawns in self.selections(self.world_tree).items():
                for spawn in spawns:
                    row = self.compatibility.rowCount()
                    self.compatibility.insertRow(row)
                    identity = self.spawn_identity(world, spawn)
                    include = QtWidgets.QTableWidgetItem()
                    include.setFlags(
                        QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    )
                    include.setData(QtCore.Qt.ItemDataRole.UserRole, (robot, world, identity))
                    include.setCheckState(
                        QtCore.Qt.CheckState.Unchecked
                        if (robot, world, identity) in self.excluded
                        else QtCore.Qt.CheckState.Checked
                    )
                    self.compatibility.setItem(row, 0, include)
                    values = [
                        robots.get(robot, {}).get("label", robot),
                        worlds.get(world, {}).get("label", world),
                        spawn.get("name", identity)
                        if isinstance(spawn, dict)
                        else "Default"
                        if spawn == "Default"
                        else self.spawn_label(world, spawn),
                    ]
                    for column, text in enumerate(values, 1):
                        item = QtWidgets.QTableWidgetItem(text)
                        item.setFlags(
                            QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
                        )
                        self.compatibility.setItem(row, column, item)
        self.compatibility.blockSignals(False)

    def spawn_label(self, world: str, identity: str) -> str:
        for i in range(self.world_tree.topLevelItemCount()):
            parent = self.world_tree.topLevelItem(i)
            if parent and parent.data(0, QtCore.Qt.ItemDataRole.UserRole) == world:
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    if child.data(0, QtCore.Qt.ItemDataRole.UserRole) == identity:
                        return child.text(0)
        return identity[:8]

    def change_exclusion(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() == 0:
            identity = tuple(item.data(QtCore.Qt.ItemDataRole.UserRole))
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                self.excluded.discard(identity)
            else:
                self.excluded.add(identity)
            self.invalidate()

    def submit_design(self, submit: bool) -> None:
        if self.import_tag:
            return
        if submit:
            if self.validated is None or self.submission_tag:
                return
            self.submission_tag = "builder-submit-" + uuid.uuid4().hex
            self.submit_button.setEnabled(False)
            self.transport.submit(
                self.submission_tag, "campaign.submit", {"specification": deepcopy(self.validated)}
            )
        else:
            self.validation_tag = "builder-preview-" + uuid.uuid4().hex
            try:
                design = self.design()
            except ValueError:
                self.set_message("Enter a finite, nonnegative exploration time limit in seconds.")
                return
            self.set_message("Validating included combinations…", "muted")
            self.transport.submit(
                self.validation_tag, "campaign.preview", {"specification": design}
            )

    def apply_design(self, value: dict[str, Any]) -> None:
        self.source_campaign_id = value.get("source_campaign_id")
        self.name.setText(value["name"])
        self.description.setText(value.get("description", ""))
        self.repetitions.setValue(value["repetitions"])
        self.concurrency.setCurrentIndex(self.concurrency.findData(value["concurrency"]))
        self.limit.setValue(value["time_limit"])
        self.excluded = {tuple(v) for v in value["exclusions"]}
        self.robot_selector.set_selections(value["robots"])
        for index in range(self.world_tree.topLevelItemCount()):
            parent = self.world_tree.topLevelItem(index)
            assert parent is not None
            selection = value["worlds"].get(parent.data(0, QtCore.Qt.ItemDataRole.UserRole), [])
            parent.setCheckState(
                0, QtCore.Qt.CheckState.Checked if selection else QtCore.Qt.CheckState.Unchecked
            )
            for child_index in range(parent.childCount()):
                parent.child(child_index).setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            for item in selection:
                child = QtWidgets.QTreeWidgetItem([item["name"]])
                child.setData(0, QtCore.Qt.ItemDataRole.UserRole, deepcopy(item))
                child.setCheckState(0, QtCore.Qt.CheckState.Checked)
                parent.addChild(child)
        self.invalidate()
        self.show_step(0)

    def edit_selected_spawn(self) -> None:
        child = self.world_tree.currentItem()
        if child:
            parent = child.parent() or child
            self.edit_spawn.emit(parent.data(0, QtCore.Qt.ItemDataRole.UserRole))

    def load_map(self, current: Any, previous: Any = None) -> None:
        if not current:
            return
        parent = current.parent() or current
        world = parent.data(0, QtCore.Qt.ItemDataRole.UserRole)
        self.map_tag = "builder-map-" + uuid.uuid4().hex
        self.pose_tag = ""
        self.world_map.clear_view()
        self.transport.submit(self.map_tag, "spawns.map", {"world_id": world})
        spawn = current.data(0, QtCore.Qt.ItemDataRole.UserRole) if current.parent() else "Default"
        if spawn == "Default":
            spawn = next(
                (w["default_spawn"] for w in self.catalog.get("worlds", []) if w["id"] == world),
                None,
            )
        if isinstance(spawn, dict):
            self.world_map.set_pose(spawn["x"], spawn["y"], spawn["yaw"])
        elif isinstance(spawn, str):
            self.pose_tag = "builder-pose-" + uuid.uuid4().hex
            self.transport.submit(self.pose_tag, "spawns.get", {"revision_id": spawn})

    def import_campaign(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import campaign", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.invalidate()
            self.import_tag = "campaign-read-" + uuid.uuid4().hex
            self.set_message("Reading campaign configuration…", "muted")
            self.transport.submit(self.import_tag, "local.read", {"path": path})

    def export_campaign(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export campaign", "campaign.yaml", "YAML (*.yaml)"
        )
        if path:
            try:
                design = self.design()
            except ValueError as exc:
                self.set_message(str(exc))
                return
            self.transport.submit(
                "file-saved",
                "local.write",
                {"path": path, "text": yaml.safe_dump(design, sort_keys=False)},
            )

    def received(self, tag: str, result: Any) -> None:
        if tag == self.validation_tag:
            if tag.startswith("campaign-import-"):
                self.import_tag = ""
                self.apply_design(result["specification"])
            self.validated = result["specification"]
            self.preview.set_record(
                {
                    "name": self.validated["name"],
                    "experiments": result["count"],
                    "minimum_attempts": result["expected_minimum_attempts"],
                    "workers": self.validated["concurrency"],
                    "time_limit_seconds": self.validated["time_limit"],
                    "timing_retry": result["timing_retry"],
                    "specification": self.validated,
                }
            )
            self.set_message(
                "Validation passed. Add this campaign to the queue, then explicitly Start it.",
                "success",
            )
            self.submit_button.setEnabled(True)
            self.show_step(self.step)
        elif tag == self.submission_tag:
            self.submission_tag = ""
            self.validated = None
            self.submit_button.setEnabled(False)
            self.set_message(
                "Campaign added to the queue. Open Campaigns and press Start to execute.",
                "success",
            )
            self.transport.received.emit("submit", result)
        elif tag == self.import_tag:
            from .yaml_io import load_mapping

            try:
                self.validation_tag = "campaign-import-" + uuid.uuid4().hex
                self.transport.submit(
                    self.validation_tag,
                    "campaign.preview",
                    {"specification": load_mapping(result["text"])},
                )
            except ValueError as exc:
                self.import_tag = ""
                self.set_message(str(exc))
        elif tag == self.map_tag:
            self.world_map.set_map(result)
        elif tag == self.pose_tag:
            self.world_map.set_pose(result["x"], result["y"], result["yaw"])

    def failed(self, tag: str, error: str) -> None:
        if tag in (self.validation_tag, self.submission_tag, self.import_tag, self.pose_tag):
            self.import_tag = ""
            self.submission_tag = ""
            self.set_message(error)
            self.submit_button.setEnabled(self.validated is not None)
