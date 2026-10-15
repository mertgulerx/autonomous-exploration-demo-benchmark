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


import math
from typing import Any
from PySide6 import QtCore, QtWidgets
from .gui_support import FloatEdit, Transport, MapView, PagedTable, button
from .gui_style import label, surface


class SpawnEditor(QtWidgets.QWidget):
    changed = QtCore.Signal()
    return_requested = QtCore.Signal()

    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self.catalog: dict[str, Any] = {}
        self.spawn_row: dict[str, Any] | None = None
        self.export_path = ""
        self.loaded_yaw = 0.0
        self.loaded_heading_text = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(label("World", "muted"))
        self.spawn_world, self.spawn_robot = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.spawn_world.currentIndexChanged.connect(self.load_world_map)
        self.spawn_robot.currentIndexChanged.connect(self.update_spawn_pose)
        bar.addWidget(self.spawn_world)
        bar.addSpacing(16)
        bar.addWidget(label("Robot footprint", "muted"))
        bar.addWidget(self.spawn_robot)
        bar.addStretch()
        button("Back to campaign", self.return_requested.emit, bar, role="quiet")
        layout.addLayout(bar)
        split = QtWidgets.QSplitter()
        split.setObjectName("spawn-editor-split")
        map_panel = surface()
        map_layout = QtWidgets.QVBoxLayout(map_panel)
        map_layout.setContentsMargins(12, 12, 12, 12)
        instructions = QtWidgets.QHBoxLayout()
        hint = label("Click to position · drag the arrow to turn · scroll to zoom", "muted")
        hint.setWordWrap(True)
        instructions.addWidget(hint, 1)
        self.spawn_map = MapView()
        button("Fit map", self.spawn_map.fit_map, instructions, role="quiet")
        map_layout.addLayout(instructions)
        self.spawn_map.editable = True
        self.spawn_map.pose_selected.connect(self.map_pose)
        map_layout.addWidget(self.spawn_map, 1)
        split.addWidget(map_panel)
        panel = surface()
        pane = QtWidgets.QVBoxLayout(panel)
        pane.setContentsMargins(16, 16, 16, 16)
        pane.setSpacing(12)
        pane.addWidget(label("Saved starting positions", "section"))
        self.spawns = PagedTable(
            transport, "spawn-table", "library", [("name", "Spawn"), ("revision", "Revision")]
        )
        self.spawns.filters = {"kind": "spawn", "archived": 0}
        self.spawns.selected.connect(self.select_spawn)
        self.tables = [self.spawns]
        pane.addWidget(self.spawns, 1)
        row = QtWidgets.QHBoxLayout()
        button("New", self.new_spawn, row)
        button("Clone", self.clone_spawn, row)
        more = button("More", lambda: None, row, role="quiet")
        menu = QtWidgets.QMenu(more)
        menu.addAction("Import YAML", self.import_spawn)
        menu.addAction("Export YAML", self.export_spawn)
        menu.addSeparator()
        menu.addAction("Archive selected", self.archive_spawn)
        more.setMenu(menu)
        pane.addLayout(row)
        pane.addWidget(label("Position & heading", "section"))
        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(12)
        self.spawn_name = QtWidgets.QLineEdit("Spawn")
        form.addRow("Name", self.spawn_name)
        self.pose_fields = []
        for title in ("X · metres", "Y · metres", "Heading · degrees"):
            value = FloatEdit()
            value.valueChanged.connect(self.update_spawn_pose)
            self.pose_fields.append(value)
            form.addRow(title, value)
        pane.addLayout(form)
        button("Validate && save", self.save_spawn, pane, role="primary")
        self.message = label("", "muted")
        self.message.setWordWrap(True)
        pane.addWidget(self.message)
        panel.setMinimumWidth(320)
        split.addWidget(panel)
        split.setSizes([760, 350])
        split.setStretchFactor(0, 1)
        layout.addWidget(split, 1)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    def load_world_map(self, *args: Any) -> None:
        world = self.spawn_world.currentData()
        if world and self.catalog:
            self.spawn_row = None
            self.spawns.filters = {"kind": "spawn", "owner": world, "archived": 0}
            self.spawns.offset = 0
            self.spawns.refresh()
            pose = next(w["default_spawn"] for w in self.catalog["worlds"] if w["id"] == world)
            self.map_pose(pose["x"], pose["y"], pose["yaw"])
            self.transport.submit("spawn-map", "spawns.map", {"world_id": world})

    def map_pose(self, x: float, y: float, yaw: float) -> None:
        for field, value in zip(self.pose_fields, (x, y, math.degrees(yaw))):
            field.blockSignals(True)
            field.setValue(value)
            field.blockSignals(False)
        self.loaded_yaw = yaw
        self.loaded_heading_text = self.pose_fields[2].text()
        self.update_spawn_pose()

    def pose(self) -> tuple[float, float, float]:
        x, y, degrees = (f.value() for f in self.pose_fields)
        yaw = (
            self.loaded_yaw
            if self.pose_fields[2].text() == self.loaded_heading_text
            else math.radians(degrees)
        )
        return x, y, yaw

    def update_spawn_pose(self, *args: Any) -> None:
        robot = self.spawn_robot.currentData()
        if robot and self.catalog:
            footprint = next(r["footprint"] for r in self.catalog["robots"] if r["id"] == robot)
            try:
                x, y, yaw = self.pose()
            except ValueError:
                return
            self.spawn_map.set_pose(x, y, yaw, footprint)

    def save_spawn(self) -> None:
        try:
            x, y, yaw = self.pose()
        except ValueError:
            self.message.setText("Enter finite coordinates and heading before saving.")
            return
        value = {
            "name": self.spawn_name.text(),
            "world_id": self.spawn_world.currentData(),
            "robot_ids": [self.spawn_robot.currentData()],
            "x": x,
            "y": y,
            "yaw": yaw,
        }
        if self.spawn_row:
            value["family_id"] = self.spawn_row["family_id"]
        self.transport.submit(
            "spawn-edit:save",
            "spawns.save",
            value,
        )

    def select_spawn(self, row: dict[str, Any]) -> None:
        self.spawn_row = row
        self.transport.submit("spawn-edit:get", "spawns.get", {"revision_id": row["id"]})

    def received(self, tag: str, result: Any) -> None:
        if tag == "spawn-map" and result["world_id"] == self.spawn_world.currentData():
            self.spawn_map.set_map(result)
        elif tag == "spawn-edit:get" and self.spawn_row and result["id"] == self.spawn_row["id"]:
            row = self.spawn_row
            self.spawn_world.setCurrentIndex(self.spawn_world.findData(result["world_id"]))
            self.spawn_row = row
            self.spawn_name.setText(result["name"])
            self.map_pose(result["x"], result["y"], result["yaw"])
        elif tag == "spawn-edit:read":
            self.transport.submit(
                "spawn-edit:import",
                "spawns.import",
                {
                    "yaml_text": result["text"],
                    "robot_ids": [self.spawn_robot.currentData()],
                },
            )
        elif tag == "spawn-edit:export":
            self.transport.submit(
                "spawn-edit:written",
                "local.write",
                {"path": self.export_path, "text": result["yaml"]},
            )
        elif tag in ("spawn-edit:save", "spawn-edit:import", "spawn-edit:archive"):
            self.message.setText(
                "Spawn library saved; submitted campaigns keep their existing poses"
            )
            self.changed.emit()
        elif tag == "spawn-edit:written":
            self.message.setText("Spawn YAML exported")

    def new_spawn(self) -> None:
        self.spawn_row = None
        self.spawn_name.setText("Spawn")

    def clone_spawn(self) -> None:
        self.spawn_row = None
        self.spawn_name.setText(self.spawn_name.text() + " copy")

    def import_spawn(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import spawn", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.transport.submit("spawn-edit:read", "local.read", {"path": path})

    def export_spawn(self) -> None:
        if self.spawn_row:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export spawn", "spawn.yaml", "YAML (*.yaml)"
            )
            if path:
                self.export_path = path
                self.transport.submit(
                    "spawn-edit:export", "spawns.export", {"revision_id": self.spawn_row["id"]}
                )

    def archive_spawn(self) -> None:
        if self.spawn_row:
            self.transport.submit(
                "spawn-edit:archive", "spawns.archive", {"revision_id": self.spawn_row["id"]}
            )
            self.spawn_row = None

    def failed(self, tag: str, message: str) -> None:
        if tag.startswith("spawn-edit:") or tag == "spawn-map":
            self.message.setText(message)
