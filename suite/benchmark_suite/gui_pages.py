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
import uuid

from PySide6 import QtCore, QtWidgets

from .contracts import next_spawn_name
from .gui_support import FloatEdit, Transport, MapView, PagedTable, button
from .gui_style import label, surface

# Bound on the spawn-name scan per pass (250 names per page).
NAME_SCAN_PAGES = 40


class SpawnEditor(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self.catalog: dict[str, Any] = {}
        self.spawn_row: dict[str, Any] | None = None
        self.export_path = ""
        self.loaded_yaw = 0.0
        self.loaded_heading_text = ""
        # Saved spawn names per world, and the scan that reads them: (world, token, archived pass).
        self.spawn_names: dict[str, list[str]] = {}
        self.name_scan: tuple[str, str, bool] | None = None
        self.scan_buffer: list[str] = []
        self.scan_pages = 0
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
        layout.addLayout(bar)
        split = QtWidgets.QSplitter()
        split.setObjectName("spawn-editor-split")
        map_panel = surface()
        map_layout = QtWidgets.QVBoxLayout(map_panel)
        map_layout.setContentsMargins(12, 12, 12, 12)
        hint = label("Click to position · drag the arrow to turn · scroll to zoom", "muted")
        hint.setWordWrap(True)
        map_layout.addWidget(hint)
        self.spawn_map = MapView()
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
        self.spawns.filters = {"kind": "spawn"}
        self.spawns.selected.connect(self.select_spawn)
        self.tables = [self.spawns]
        pane.addWidget(self.spawns, 1)
        row = QtWidgets.QHBoxLayout()
        more = button("More", lambda: None, row, role="outlined")
        menu = QtWidgets.QMenu(more)
        menu.addAction("Import YAML", self.import_spawn)
        menu.addAction("Export YAML", self.export_spawn)
        more.setMenu(menu)
        row.addStretch()
        pane.addLayout(row)
        pane.addWidget(label("Position & heading", "section"))
        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(12)
        # Every save creates a new spawn; the name is the next "Spawn N" unless the user types one.
        self.spawn_name = QtWidgets.QLineEdit(next_spawn_name(()))
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
            self.suggest_name(world)
            self.spawns.filters = {"kind": "spawn", "owner": world}
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
        self.transport.submit(
            "spawn-edit:save",
            "spawns.save",
            value,
        )

    def select_spawn(self, row: dict[str, Any]) -> None:
        """Load a saved spawn as the template for a new one (and as the Export YAML target)."""
        self.spawn_row = row
        self.transport.submit("spawn-edit:get", "spawns.get", {"revision_id": row["id"]})

    def suggest_name(self, world: str) -> None:
        """Propose the next "Spawn N" for the world unless the user typed a name, then rescan."""
        if not self.spawn_name.isModified():
            self.spawn_name.setText(next_spawn_name(self.spawn_names.get(world, [])))
        self.name_scan = (world, uuid.uuid4().hex, False)
        self.scan_buffer, self.scan_pages = [], 0
        self.scan_page(0)

    def scan_page(self, offset: int) -> None:
        assert self.name_scan is not None
        world, token, archived = self.name_scan
        self.scan_pages += 1
        # Both passes together read every saved name, whether or not an engine filters archived rows.
        self.transport.submit(
            "spawn-names:" + token,
            "spawns.list",
            {
                "world_id": world,
                "search": "spawn",
                "summary": True,
                "limit": 250,
                "offset": offset,
                "archived": archived,
            },
        )

    def scanned_names(self, result: dict[str, Any]) -> None:
        assert self.name_scan is not None
        world, token, archived = self.name_scan
        self.scan_buffer.extend(str(row["name"]) for row in result["rows"])
        more = result.get("next_offset")
        if more is not None and self.scan_pages < NAME_SCAN_PAGES:
            self.scan_page(more)
            return
        if more is not None:
            self.message.setText(
                "Not every saved spawn name was read; check the name before saving."
            )
        if not archived:
            self.name_scan, self.scan_pages = (world, token, True), 0
            self.scan_page(0)
            return
        self.name_scan = None
        self.spawn_names[world] = self.scan_buffer
        if world == self.spawn_world.currentData() and not self.spawn_name.isModified():
            self.spawn_name.setText(next_spawn_name(self.spawn_names[world]))

    def received(self, tag: str, result: Any) -> None:
        if tag == "spawn-map" and result["world_id"] == self.spawn_world.currentData():
            self.spawn_map.set_map(result)
        elif self.name_scan and tag == "spawn-names:" + self.name_scan[1]:
            self.scanned_names(result)
        elif tag == "spawn-edit:get" and self.spawn_row and result["id"] == self.spawn_row["id"]:
            # A template: its pose is loaded, while the name stays the next "Spawn N" (or the typed one).
            row = self.spawn_row
            self.spawn_world.setCurrentIndex(self.spawn_world.findData(result["world_id"]))
            self.spawn_row = row
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
        elif tag in ("spawn-edit:save", "spawn-edit:import"):
            self.message.setText(
                f"Saved “{result['name']}” as a new spawn; submitted campaigns keep their poses"
            )
            self.changed.emit()
            world = self.spawn_world.currentData()
            if world:
                if tag == "spawn-edit:save":
                    # The typed name has been used: the next save gets a fresh suggestion.
                    self.spawn_names.setdefault(world, []).append(str(result["name"]))
                    self.spawn_name.setModified(False)
                self.suggest_name(world)
        elif tag == "spawn-edit:written":
            self.message.setText("Spawn YAML exported")

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

    def failed(self, tag: str, message: str) -> None:
        if self.name_scan and tag == "spawn-names:" + self.name_scan[1]:
            self.name_scan = None
            self.message.setText(
                f"Could not read saved spawn names: {message}. Check the name before saving."
            )
        elif tag.startswith("spawn-edit:") or tag == "spawn-map":
            self.message.setText(message)
