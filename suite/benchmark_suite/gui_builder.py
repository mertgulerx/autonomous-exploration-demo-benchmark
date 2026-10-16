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
from .gui_support import Transport, button
from .gui_robots import RobotSelector
from .gui_worlds import WorldSelector
from .contracts import resolve_world_time_limits


class CampaignBuilder(QtWidgets.QWidget):
    return_requested = QtCore.Signal()
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
        self.import_tag = ""
        self.step = 0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        heading = QtWidgets.QHBoxLayout()
        self.return_button = button(
            "Campaigns", self.return_requested.emit, heading, role="outlined", glyph="arrow-left"
        )
        self.return_button.setAccessibleName("Back to campaigns")
        self.return_button.setToolTip("Back to campaigns")
        heading.addWidget(label("New campaign", "title"))
        heading.addStretch()
        counters = QtWidgets.QWidget()
        counters.setObjectName("campaignCounters")
        counts_layout = QtWidgets.QHBoxLayout(counters)
        counts_layout.setContentsMargins(0, 0, 0, 0)
        counts_layout.setSpacing(24)
        self.counts: dict[str, QtWidgets.QLabel] = {}
        for key, title in (
            ("robots", "Robots"),
            ("profiles", "Profiles"),
            ("worlds", "Worlds"),
            ("repetitions", "Repetitions"),
            ("experiments", "Planned experiments"),
        ):
            if key == "experiments":
                divider = QtWidgets.QFrame()
                divider.setObjectName("campaignCounterDivider")
                divider.setFixedSize(1, 36)
                counts_layout.addWidget(divider)
            column = QtWidgets.QVBoxLayout()
            column.setSpacing(0)
            value = label("0", "counter")
            value.setAccessibleName(title)
            value.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            value.setMinimumHeight(32)
            value.setProperty("total", key == "experiments")
            caption = label(title, "muted")
            caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            column.addWidget(value)
            column.addWidget(caption)
            counts_layout.addLayout(column)
            self.counts[key] = value
        heading.addWidget(counters)
        layout.addLayout(heading)
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
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.robot_selector = RobotSelector(transport=transport)
        self.robot_selector.edit_profile.connect(self.edit_profile.emit)
        self.stack.addWidget(self.robot_selector)
        self.world_selector = WorldSelector(transport)
        self.world_selector.edit_spawn.connect(self.edit_spawn.emit)
        self.stack.addWidget(self.world_selector)
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
        form.addRow("Campaign name", self.name)
        form.addRow("Description", self.description)
        form.addRow("Repetitions per combination", presets)
        form.addRow("Execution", self.concurrency)
        execution_layout.addLayout(form)
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
        self.compatibility = QtWidgets.QTableWidget(0, 5)
        self.compatibility.setHorizontalHeaderLabels(
            ["Include", "Robot", "World", "Spawn", "Time limit"]
        )
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
        button("Import YAML", self.import_campaign, footer, role="outlined")
        button("Export YAML", self.export_campaign, footer, role="outlined")
        footer.addStretch()
        self.back_button = button("Back", lambda: self.show_step(self.step - 1), footer)
        self.next_button = button("Continue", self.next_step, footer, role="primary")
        self.submit_button = button(
            "Add to queue", lambda: self.submit_design(True), footer, role="primary"
        )
        layout.addLayout(footer)
        self.robot_selector.changed.connect(self.invalidate)
        self.world_selector.changed.connect(self.invalidate)
        for field in (self.name, self.description):
            field.textChanged.connect(self.invalidate)
        self.repetitions.valueChanged.connect(self.invalidate)
        self.concurrency.currentIndexChanged.connect(self.invalidate)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)
        self.show_step(0)

    def design(self) -> dict[str, Any]:
        worlds = self.world_selector.selections()
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
            "world_time_limits": self.world_selector.time_limits(),
            "exclusions": [list(v) for v in sorted(self.excluded & combinations)],
            **({"source_campaign_id": self.source_campaign_id} if self.source_campaign_id else {}),
        }

    def invalidate(self, *args: Any) -> None:
        self.validated = None
        self.validation_tag = ""
        self.submit_button.setEnabled(False)
        self.update_summary()

    def update_summary(self) -> None:
        robots, worlds = self.robot_selector.selections(), self.world_selector.selections()
        count = 0
        for robot, profiles in robots.items():
            for world, spawns in worlds.items():
                count += len(profiles) * sum(
                    (robot, world, self.spawn_identity(world, s)) not in self.excluded
                    for s in spawns
                )
        count *= self.repetitions.value()
        for key, value in {
            "robots": len(robots),
            "profiles": sum(map(len, robots.values())),
            "worlds": len(worlds),
            "repetitions": self.repetitions.value(),
            "experiments": count,
        }.items():
            self.counts[key].setText(f"{value:,}")

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
            control.setProperty("role", "selected" if i == self.step else "outlined")
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
            selected = self.world_selector.selections()
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
        try:
            limits = self.world_selector.time_limits()
        except ValueError as exc:
            self.set_message(str(exc))
            limits = {}
        for robot in self.robot_selector.selections():
            for world, spawns in self.world_selector.selections().items():
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
                        "Invalid"
                        if world not in limits
                        else "Unlimited"
                        if limits[world] == 0
                        else f"{limits[world]:g} s",
                    ]
                    for column, text in enumerate(values, 1):
                        item = QtWidgets.QTableWidgetItem(text)
                        item.setFlags(
                            QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
                        )
                        self.compatibility.setItem(row, column, item)
        self.compatibility.blockSignals(False)

    def spawn_label(self, world: str, identity: str) -> str:
        return self.world_selector.spawn_label(world, identity)

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
            except ValueError as exc:
                self.set_message(str(exc))
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
        self.excluded = {tuple(v) for v in value["exclusions"]}
        self.robot_selector.set_selections(value["robots"])
        self.world_selector.set_selections(value["worlds"])
        self.world_selector.set_time_limits(dict(resolve_world_time_limits(value, value["worlds"])))
        self.invalidate()
        self.show_step(0)

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
                    "world_time_limits": ", ".join(
                        f"{world.capitalize()}: {'unlimited' if limit == 0 else f'{limit:g} s'}"
                        for world, limit in resolve_world_time_limits(
                            self.validated, self.validated["worlds"]
                        )
                    ),
                    "timing_retry": result["timing_retry"],
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

    def failed(self, tag: str, error: str) -> None:
        if tag in (self.validation_tag, self.submission_tag, self.import_tag):
            self.import_tag = ""
            self.submission_tag = ""
            self.set_message(error)
            self.submit_button.setEnabled(self.validated is not None)
