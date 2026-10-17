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
from PySide6 import QtCore, QtGui, QtWidgets

from .contracts import HIDDEN_GROUPS
from .gui_support import FloatEdit, Transport, button
from .gui_style import humanize, label


class ProfileEditor(QtWidgets.QDialog):
    saved = QtCore.Signal()
    accepted = QtCore.Signal(object)

    def __init__(
        self,
        transport: Transport,
        catalog: dict[str, Any],
        profile: dict[str, Any],
        clone: bool = False,
    ):
        super().__init__()
        self.transport, self.catalog, self.profile = transport, catalog, deepcopy(profile)
        self.tag = "editor-" + uuid.uuid4().hex
        self.export_path = ""
        self.setWindowTitle("Explorer Profile")
        self.resize(880, 700)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(label("Explorer profile", "title"))
        robot_label = next(r["label"] for r in catalog["robots"] if r["id"] == profile["robot_id"])
        layout.addWidget(
            label(f"{robot_label} · Saving creates an independent, immutable revision", "muted")
        )
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(profile["name"] + (" copy" if clone else ""))
        self.description = QtWidgets.QLineEdit(profile.get("description", ""))
        form.addRow("Name", self.name)
        form.addRow("Description", self.description)
        layout.addLayout(form)
        self.fields: dict[str, Any] = {}
        self.labels: dict[str, QtWidgets.QLabel] = {}
        self.defaults = next(
            r["defaults"] for r in catalog["robots"] if r["id"] == profile["robot_id"]
        )
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        groups: dict[str, QtWidgets.QFormLayout] = {}
        for key, metadata in catalog["parameters"].items():
            if metadata["managed"] or metadata["group"] in HIDDEN_GROUPS:
                continue
            group = metadata["group"]
            if group not in groups:
                content = QtWidgets.QWidget()
                groups[group] = QtWidgets.QFormLayout(content)
                groups[group].setContentsMargins(16, 16, 16, 16)
                groups[group].setVerticalSpacing(20)
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setWidget(content)
                self.tabs.addTab(scroll, group)
            default = self.defaults[key]
            field: Any
            if metadata.get("choices"):
                field = QtWidgets.QComboBox()
                field.addItems([str(v) for v in metadata["choices"]])
                field.currentIndexChanged.connect(self.changed)
            elif type(default) is bool:
                field = QtWidgets.QCheckBox()
                field.toggled.connect(self.changed)
            else:
                field = QtWidgets.QLineEdit()
                if type(default) is int:
                    field.setValidator(
                        QtGui.QRegularExpressionValidator(
                            QtCore.QRegularExpression("-?[0-9]+"), field
                        )
                    )
                elif type(default) is float:
                    validator = QtGui.QDoubleValidator(-1e308, 1e308, 17, field)
                    validator.setLocale(QtCore.QLocale.c())
                    validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
                    field.setValidator(validator)
                field.textChanged.connect(self.changed)
            field.setToolTip(metadata.get("description", ""))
            parameter_label = QtWidgets.QLabel(humanize(key))
            parameter_label.setToolTip(key)
            self.fields[key], self.labels[key] = field, parameter_label
            container = QtWidgets.QWidget()
            field_layout = QtWidgets.QVBoxLayout(container)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(5)
            field_layout.addWidget(field)
            description = label(metadata.get("description", ""), "muted")
            description.setWordWrap(True)
            field_layout.addWidget(description)
            groups[group].addRow(parameter_label, container)
        self.physical: dict[str, FloatEdit] = {}
        if profile["robot_id"] == "mogi":
            page = QtWidgets.QWidget()
            physical_form = QtWidgets.QFormLayout(page)
            for key, title, default in (
                ("movement_speed", "Movement speed (m/s)", 0.8),
                ("turning_speed", "Turning speed (rad/s)", 1.0),
                ("lidar_range", "Lidar range (m)", 10.0),
            ):
                field = FloatEdit(profile.get("robot_overrides", {}).get(key, default))
                self.physical[key] = field
                physical_form.addRow(title, field)
            self.tabs.addTab(page, "MOGI physical settings")
        self.error = QtWidgets.QLabel()
        self.error.setProperty("kind", "error")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        bar = QtWidgets.QHBoxLayout()
        button("Reset to Default", self.reset, bar)
        button("Import Explorer YAML", self.import_yaml, bar, role="outlined")
        button("Export Explorer YAML", self.export_yaml, bar, role="outlined")
        bar.addStretch()
        self.save_button = button("Save revision", self.save, bar, role="primary")
        button("Close", self.close, bar)
        layout.addLayout(bar)

        self.family_id = None if clone else profile.get("family_id")
        self.apply_values(profile["effective"])
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    def value(self, key: str) -> Any:
        field, default = self.fields[key], self.defaults[key]
        if isinstance(field, QtWidgets.QComboBox):
            return field.currentText()
        if isinstance(field, QtWidgets.QCheckBox):
            return field.isChecked()
        if isinstance(field, QtWidgets.QDoubleSpinBox):
            return field.value()
        return (
            int(field.text())
            if type(default) is int
            else float(field.text())
            if type(default) is float
            else field.text()
        )

    def apply_values(self, values: dict[str, Any]) -> None:
        for key, field in self.fields.items():
            value = values[key]
            if isinstance(field, QtWidgets.QComboBox):
                field.setCurrentText(str(value))
            elif isinstance(field, QtWidgets.QCheckBox):
                field.setChecked(value)
            elif isinstance(field, QtWidgets.QDoubleSpinBox):
                field.setValue(value)
            else:
                field.setText(str(value))
        self.changed()

    def changed(self, *args: Any) -> None:
        for key, parameter_label in self.labels.items():
            try:
                changed = self.value(key) != self.defaults[key]
            except ValueError:
                changed = True
            parameter_label.setText(humanize(key) + ("  •" if changed else ""))

    def configuration(self) -> dict[str, Any]:
        requested = deepcopy(self.profile.get("requested", {}))
        for key in self.fields:
            value = self.value(key)
            if value == self.defaults[key]:
                requested.pop(key, None)
            else:
                requested[key] = value
        return {
            "robot_id": self.profile["robot_id"],
            "name": self.name.text(),
            "description": self.description.text(),
            "requested": requested,
            "robot_overrides": {k: w.value() for k, w in self.physical.items()},
        }

    def reset(self) -> None:
        self.profile["requested"] = {}
        self.apply_values(self.defaults)
        for key, value in {
            "movement_speed": 0.8,
            "turning_speed": 1.0,
            "lidar_range": 10.0,
        }.items():
            if key in self.physical:
                self.physical[key].setValue(value)

    def save(self) -> None:
        try:
            value = self.configuration()
            if self.family_id:
                value["family_id"] = self.family_id
            self.save_button.setEnabled(False)
            self.transport.submit(self.tag + ":save", "profiles.save", value)
        except ValueError as exc:
            self.error.setText(str(exc))

    def import_yaml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import explorer YAML", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.transport.submit(self.tag + ":import", "local.read", {"path": path})

    def export_yaml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export explorer YAML", "profile.yaml", "YAML (*.yaml)"
        )
        if path:
            try:
                self.export_path = path
                self.transport.submit(
                    self.tag + ":export", "parameters.resolve", self.configuration()
                )
            except ValueError as exc:
                self.error.setText(str(exc))

    def received(self, tag: str, result: Any) -> None:
        if not tag.startswith(self.tag + ":"):
            return
        operation = tag.split(":")[-1]
        if operation == "save":
            self.accepted.emit(result)
            self.saved.emit()
            self.close()
        elif operation == "import":
            self.transport.submit(
                self.tag + ":resolved",
                "parameters.resolve",
                {"robot_id": self.profile["robot_id"], "yaml_text": result["text"]},
            )
        elif operation == "resolved":
            self.profile["requested"] = result["requested"]
            self.apply_values(result["effective"])
            self.error.setText("Imported values accepted")
        elif operation == "export":
            source = yaml.safe_dump(
                {"frontier_explorer": {"ros__parameters": result["effective"]}}, sort_keys=False
            )
            self.transport.submit(
                self.tag + ":written", "local.write", {"path": self.export_path, "text": source}
            )
        elif operation == "written":
            self.error.setText("Export saved")

    def failed(self, tag: str, error: str) -> None:
        if tag.startswith(self.tag + ":"):
            self.error.setText(error)
            self.save_button.setEnabled(True)
