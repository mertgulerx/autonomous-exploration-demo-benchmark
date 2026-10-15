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
from typing import Any, cast
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
        self.tags = QtWidgets.QLineEdit(", ".join(profile.get("tags", [])))
        form.addRow("Name", self.name)
        form.addRow("Description", self.description)
        form.addRow("Tags", self.tags)
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
        button("Import Explorer YAML", self.import_yaml, bar)
        button("Export Explorer YAML", self.export_yaml, bar)
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
            "tags": [s.strip() for s in self.tags.text().split(",") if s.strip()],
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


class SweepEditor(QtWidgets.QDialog):
    saved = QtCore.Signal()

    def __init__(
        self,
        transport: Transport,
        catalog: dict[str, Any],
        robot: str,
        parent_id: str | None = None,
    ):
        super().__init__()
        self.transport, self.catalog, self.robot, self.parent_id = (
            transport,
            catalog,
            robot,
            parent_id,
        )
        self.request_tag = ""
        self.save_pending = False
        self.setWindowTitle("Parameter sweep")
        self.resize(820, 520)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(label("Create a parameter sweep", "title"))
        layout.addWidget(
            label(
                "Choose axes and values. The engine expands and validates their Cartesian product.",
                "muted",
            )
        )
        self.axes = QtWidgets.QTableWidget(0, 3)
        self.axes.setHorizontalHeaderLabels(
            ["Parameter", "Definition", "Values / start, stop, step"]
        )
        self.axes.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.axes.setColumnWidth(0, 280)
        self.axes.setColumnWidth(1, 160)
        self.axes.verticalHeader().hide()
        self.axes.setShowGrid(False)
        self.axes.verticalHeader().setDefaultSectionSize(48)
        layout.addWidget(self.axes, 1)
        row = QtWidgets.QHBoxLayout()
        button("Add axis", self.add_axis, row)
        button("Remove selected", self.remove_axis, row, role="quiet")
        row.addStretch()
        button("Import YAML", self.import_yaml, row, role="quiet")
        button("Export YAML", self.export_yaml, row, role="quiet")
        layout.addLayout(row)
        self.message = label("", "muted")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        footer = QtWidgets.QHBoxLayout()
        button("Preview variants", lambda: self.generate(False), footer)
        footer.addStretch()
        self.save_button = button(
            "Save profiles", lambda: self.generate(True), footer, role="primary"
        )
        button("Close", self.close, footer)
        layout.addLayout(footer)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)
        self.add_axis()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    def add_axis(self, name: str | None = None, definition: dict[str, Any] | None = None) -> None:
        if isinstance(name, bool):
            name = None
        row = self.axes.rowCount()
        self.axes.insertRow(row)
        names = QtWidgets.QComboBox()
        names.currentTextChanged.connect(names.setToolTip)
        for key, metadata in self.catalog["parameters"].items():
            if not metadata["managed"] and metadata["group"] not in HIDDEN_GROUPS:
                names.addItem(humanize(key), key)
        if self.robot == "mogi":
            for key in ("movement_speed", "turning_speed", "lidar_range"):
                names.addItem("Robot · " + humanize(key), "robot." + key)
        if name:
            index = names.findData(name)
            if index < 0:
                raise ValueError("Unsupported sweep parameter: " + name)
            names.setCurrentIndex(index)
        mode = QtWidgets.QComboBox()
        mode.addItems(["Explicit values", "Numeric range"])
        values = QtWidgets.QLineEdit()
        values.setPlaceholderText("[0.4, 0.6, 0.8]")
        mode.currentIndexChanged.connect(
            lambda index: values.setPlaceholderText("0.4, 0.8, 0.1" if index else "[0.4, 0.6, 0.8]")
        )
        if definition:
            if "values" in definition:
                values.setText(
                    yaml.safe_dump(definition["values"], default_flow_style=True).strip()
                )
            else:
                mode.setCurrentIndex(1)
                values.setText(", ".join(str(definition[key]) for key in ("start", "stop", "step")))
        self.axes.setCellWidget(row, 0, names)
        self.axes.setCellWidget(row, 1, mode)
        self.axes.setCellWidget(row, 2, values)

    def remove_axis(self) -> None:
        row = self.axes.currentRow()
        if row >= 0:
            self.axes.removeRow(row)

    def definitions(self) -> dict[str, Any]:
        from .yaml_io import load_mapping

        result: dict[str, Any] = {}
        for row in range(self.axes.rowCount()):
            name = cast(QtWidgets.QComboBox, self.axes.cellWidget(row, 0)).currentData()
            text = cast(QtWidgets.QLineEdit, self.axes.cellWidget(row, 2)).text().strip()
            if name in result:
                raise ValueError("Choose each parameter only once: " + name)
            if cast(QtWidgets.QComboBox, self.axes.cellWidget(row, 1)).currentIndex() == 0:
                source = text if text.startswith("[") else "[" + text + "]"
                result[name] = load_mapping("values: " + source)
            else:
                parts = [v.strip() for v in text.split(",")]
                if len(parts) != 3:
                    raise ValueError(
                        "Numeric ranges require start, stop and step separated by commas"
                    )
                result[name] = load_mapping(
                    "\n".join(f"{key}: {v}" for key, v in zip(("start", "stop", "step"), parts))
                )
        return result

    def generate(self, save: bool) -> None:
        if self.save_pending:
            return
        try:
            params = {"robot_id": self.robot, "axes": self.definitions()}
            if self.parent_id:
                params["parent_id"] = self.parent_id
            self.request_tag = "sweep-editor-" + uuid.uuid4().hex
            self.save_pending = save
            self.save_button.setEnabled(not save)
            self.transport.submit(
                self.request_tag, "profiles.sweep" if save else "profiles.sweep_preview", params
            )
        except ValueError as error:
            self.message.setText(str(error))

    def import_yaml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import sweep axes", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.transport.submit("sweep-import-" + str(id(self)), "local.read", {"path": path})

    def export_yaml(self) -> None:
        try:
            source = yaml.safe_dump(self.definitions(), sort_keys=False)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export sweep axes", "sweep.yaml", "YAML (*.yaml)"
            )
            if path:
                self.transport.submit(
                    "sweep-export-" + str(id(self)), "local.write", {"path": path, "text": source}
                )
        except ValueError as error:
            self.message.setText(str(error))

    def received(self, tag: str, result: Any) -> None:
        if tag == self.request_tag:
            self.message.setText(
                f"{result['count']:,} {'profiles saved' if self.save_pending else 'validated variants'}"
            )
            if self.save_pending:
                self.saved.emit()
            self.save_pending = False
            self.save_button.setEnabled(True)
        elif tag == "sweep-import-" + str(id(self)):
            from .yaml_io import load_mapping

            try:
                axes = load_mapping(result["text"])
                if any(not isinstance(definition, dict) for definition in axes.values()):
                    raise ValueError(
                        "Each sweep axis must contain values or a start/stop/step range"
                    )
                self.axes.setRowCount(0)
                for key, definition in axes.items():
                    self.add_axis(key, definition)
            except (ValueError, KeyError, TypeError) as error:
                self.message.setText(str(error))

    def failed(self, tag: str, error: str) -> None:
        if tag == self.request_tag or tag.startswith("sweep-import-" + str(id(self))):
            self.save_pending = False
            self.save_button.setEnabled(True)
            self.message.setText(error)
