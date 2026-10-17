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

"""The profile editor (14): an overlay modal on the window's ``ModalHost``.

It keeps the rules of the earlier dialog: managed parameters and hidden groups are not shown (their requested
values are kept), integer and float fields validate as before, MOGI adds its physical settings, an edit saves
a new revision of the family while New starts a family, explorer YAML imports through
``parameters.resolve`` and exports the resolved values, and errors stay in the editor.
"""

from copy import deepcopy
from typing import Any, Literal
import uuid

import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .contracts import HIDDEN_GROUPS
from .gui_cells import OverlayScrollBar
from .gui_controls import KeySelect, Switch, UnderlineTabs
from .gui_menus import show_tooltip
from .gui_profile_preview import (
    PHYSICAL_GROUP,
    PHYSICAL_SETTINGS,
    editor_groups,
    parameter_label,
    physical_value,
    value_text,
)
from .gui_review import first_sentence
from .gui_support import FloatEdit, Transport
from .gui_theme import ThemeManager, font
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    DIALOG_TITLE,
    LABEL,
    Button,
    CssBox,
    FlexRow,
    ModalCard,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    place,
    plural,
    rounded_path,
)

Mode = Literal["edit", "new"]
TITLES: dict[Mode, str] = {"edit": "Edit profile", "new": "New profile"}

WIDTH, HEIGHT = 1040, 640
ROW = 62
DESCRIPTION = TextStyle(12, 400, 18)
HEADER = TextStyle(12, 500, 18)
ERROR = TextStyle(12.5, 400, 18)
SUMMARY_COUNT = TextStyle(13, 600, 18)
# Card coordinates (14): fields, tabs, column header, rows and footer.
FIELDS_TOP, TABS_TOP, HEADER_TOP, ROWS_TOP, FOOTER_TOP = 59, 133, 172, 206, 578
ERROR_STRIP = 30
# Row grid ``1fr 132px 132px 36px``, gap 16, padding 0 24, in the 1038 px rows area.
NAME_X, DEFAULT_X, VALUE_X, RESET_X, COLUMN = 24, 682, 830, 978, 132


ValueWidget = QtWidgets.QLineEdit | KeySelect


class LockedField(Themed):
    """``.input.lock``: the robot, which an edit cannot change: sunken, ``border.line``, a 14 px lock in
    ``text.tertiary`` and the value in ``text.secondary``."""

    def __init__(self, themes: ThemeManager, text: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(themes, parent)
        self.text = text
        self.setAccessibleName("Robot " + text)
        self.setToolTip("The robot of a profile cannot change")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.sunken"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 5.5)
        )
        gui_icons.draw(painter, "lock", self.color("text.tertiary"), QtCore.QPointF(11, 9), 14)
        room = self.width() - 33 - 11
        draw_text(painter, 33, 7, BODY.elide(self.text, room), BODY, self.color("text.secondary"))


class ChangeSummary(Themed, CssBox):
    """``.efoot .sum``: "<b>3 changes</b> from MOGI Default"."""

    def __init__(self, themes: ThemeManager, robot: str) -> None:
        super().__init__(themes)
        self.robot, self.count = robot, 0
        self.setFixedHeight(18)

    def parts(self) -> tuple[str, str]:
        if not self.count:
            return "", f"No changes from {self.robot} Default"
        return plural(self.count, "change"), f" from {self.robot} Default"

    def set_count(self, count: int) -> None:
        if count != self.count:
            self.count = count
            self.updateGeometry()
            self.update()

    def natural_width(self) -> float:
        bold, rest = self.parts()
        return SUMMARY_COUNT.advance(bold) + BODY.advance(rest)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(int(self.natural_width()) + 1, 18)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        bold, rest = self.parts()
        x = self.shift + draw_text(
            painter, self.shift, 0, bold, SUMMARY_COUNT, self.color("text.ink")
        )
        draw_text(painter, x, 0, rest, BODY, self.color("text.secondary"))


class ParameterRow(Themed):
    """``.er``: 62 px; the name (500 ink) with its muted unit and the description's first sentence (12 px
    secondary), the Default value (right, secondary), the value control (132 px) and, when the value differs
    from Default, the ink dot before the name and a reset button. The raw key and the whole description are
    in the tooltip."""

    def __init__(
        self,
        themes: ThemeManager,
        key: str,
        name: str,
        unit: str,
        description: str,
        default: Any,
        field: QtWidgets.QWidget,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(themes, parent)
        self.key, self.name, self.unit = key, name, unit
        self.description, self.default = description, default
        self.default_text = value_text(default)
        self.changed = False
        self.last = False
        self.field = field
        field.setParent(self)
        self.reset = Button(themes, icon="reset", variant="ghost", small=True, parent=self)
        self.reset.setToolTip("Reset to Default")
        self.reset.setAccessibleName("Reset " + name + " to Default")
        self.reset.hide()
        self.setFixedHeight(ROW)
        self.setAccessibleName(name)

    def set_changed(self, changed: bool) -> None:
        if changed != self.changed:
            self.changed = changed
            self.reset.setVisible(changed)
            self.update()

    def set_last(self, last: bool) -> None:
        if last != self.last:
            self.last = last
            self._place()
            self.update()

    def content(self) -> float:
        return ROW - (0 if self.last else 1)

    def _place(self) -> None:
        content = self.content()
        place(self.field, VALUE_X, (content - 32) / 2, COLUMN, 32)
        place(self.reset, RESET_X, (content - 28) / 2, 28, 28)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place()

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            inside = event.pos().x() < DEFAULT_X
            text = "\n".join(part for part in (self.key, self.description) if part)
            show_tooltip(self, event.globalPos(), text if inside else "")
            return True
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        content = self.content()
        line = first_sentence(self.description)
        block = 18 + (21 if line else 0)
        top = (content - block) / 2
        ink = self.color("text.ink")
        if self.changed:
            dot_top = BODY.baseline(top) + BODY.middle_offset(7)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(ink)
            painter.drawEllipse(QtCore.QRectF(NAME_X - 14, dot_top, 6, 6))
            painter.restore()
        room = DEFAULT_X - 16 - NAME_X
        unit_room = BODY.advance(" ") + 2 + BODY.advance(self.unit) if self.unit else 0.0
        name = BODY_MEDIUM.elide(self.name, room - unit_room)
        x = NAME_X + draw_text(painter, NAME_X, top, name, BODY_MEDIUM, ink)
        if self.unit:
            x += BODY.advance(" ") + 2
            draw_text(painter, x, top, self.unit, BODY, self.color("text.tertiary"))
        if line:
            text = DESCRIPTION.elide(line, room)
            draw_text(painter, NAME_X, top + 21, text, DESCRIPTION, self.color("text.secondary"))
        default = BODY.elide(self.default_text, COLUMN)
        x = DEFAULT_X + COLUMN - BODY.advance(default)
        draw_text(painter, x, (content - 18) / 2, default, BODY, self.color("text.secondary"))
        if not self.last:
            painter.fillRect(
                QtCore.QRect(0, self.height() - 1, self.width(), 1), self.color("border.line")
            )


class RowsView(QtWidgets.QScrollArea):
    """The rows of the open group, scrolled under the column header."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.themes = themes
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)
        self.content = QtWidgets.QWidget()
        self.setWidget(self.content)
        self.viewport().setAutoFillBackground(False)
        self.content.setAutoFillBackground(False)
        self.empty = TextLine(themes, "", BODY, "text.secondary", self.content)
        self.empty.hide()
        self.rows: list[ParameterRow] = []
        self.empty_text = ""
        OverlayScrollBar(themes, self)
        self.verticalScrollBar().setSingleStep(ROW // 2)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.show_rows(
            self.rows, self.empty_text
        )  # rows span the viewport, which is sized only now

    def show_rows(self, rows: list[ParameterRow], empty: str) -> None:
        self.rows, self.empty_text = rows, empty
        shown = set(map(id, rows))
        for child in self.content.findChildren(ParameterRow):
            if id(child) not in shown:
                child.hide()
        width = self.viewport().width()
        for index, row in enumerate(rows):
            row.set_last(index == len(rows) - 1)
            row.setGeometry(0, index * ROW, width, ROW)
            row.show()
        self.empty.setText(empty)
        self.empty.setVisible(not rows)
        if not rows:
            place(self.empty, NAME_X, (ROW - 1 - 18) / 2, width - 2 * NAME_X, 18)
        self.content.resize(width, max(ROW, len(rows) * ROW))


class ProfileEditor(ModalCard):
    """``.dialog.modal`` (14), 1040×640: title, Robot (locked), Name and Description; the editor's group tabs
    with change pills and "Changed only"; the column header (Parameter, "<robot> Default", Value); the rows;
    an error line when something failed; the footer with Import YAML, Export YAML, Reset to Default, the
    change count, Close and "Save revision N"."""

    saved = QtCore.Signal()
    accepted = QtCore.Signal(object)
    notify = QtCore.Signal(
        str, str, int
    )  # a message for the window's toasts: text, kind, timeout (ms)

    def __init__(
        self,
        themes: ThemeManager,
        transport: Transport,
        catalog: dict[str, Any],
        profile: dict[str, Any],
        *,
        mode: Mode = "edit",
        next_revision: int = 1,
    ) -> None:
        super().__init__(themes)
        self.setFixedSize(WIDTH, HEIGHT)
        self.transport, self.catalog, self.profile = transport, catalog, deepcopy(profile)
        self.tag = "editor-" + uuid.uuid4().hex
        self.export_path = ""
        robot_id = profile["robot_id"]
        robot = next(r for r in catalog["robots"] if r["id"] == robot_id)
        self.robot = robot
        self.defaults: dict[str, Any] = robot["defaults"]
        self.family_id = profile.get("family_id")
        self.error_text = ""

        self.title = TextLine(themes, TITLES[mode], DIALOG_TITLE, "text.ink", self)
        self.labels = [
            TextLine(themes, text, LABEL, "text.body", self)
            for text in ("Robot", "Name", "Description")
        ]
        self.robot_field = LockedField(themes, robot["label"], self)
        self.name = QtWidgets.QLineEdit(profile["name"], self)
        self.name.setAccessibleName("Profile name")
        self.description = QtWidgets.QLineEdit(profile.get("description", ""), self)
        self.description.setAccessibleName("Profile description")

        self.groups = editor_groups(catalog["parameters"], robot_id)
        self.tabs = UnderlineTabs(
            themes, [(g, None) for g in self.groups], self, gap=22, rule=False
        )
        self.tabs.changed.connect(lambda _: self.show_group())
        self.changed_only = Switch(
            themes, "Changed only", label_token="text.secondary", parent=self
        )
        self.changed_only.toggled.connect(lambda _: self.show_group())

        self.view = RowsView(themes, self)
        self.rows: dict[str, ParameterRow] = {}
        self.fields: dict[str, ValueWidget] = {}
        self.group_of: dict[str, str] = {}
        for key, metadata in catalog["parameters"].items():
            if metadata["managed"] or metadata["group"] in HIDDEN_GROUPS:
                continue
            self._add_row(key, metadata)
        self.physical: dict[str, FloatEdit] = {}
        if robot_id == "mogi":
            for key, name, unit in PHYSICAL_SETTINGS:
                field = FloatEdit(physical_value(profile, key, robot[key]))
                self._style_field(field)
                field.textChanged.connect(self.changed)
                self.physical[key] = field
                row = ParameterRow(
                    themes, key, name, unit, "", robot[key], field, self.view.content
                )
                row.reset.clicked.connect(lambda _=False, k=key: self.reset_value(k))
                self.rows[key], self.group_of[key] = row, PHYSICAL_GROUP

        self.footer = FlexRow(self, gap=8, left=24, right=24)
        self.import_button = Button(themes, "Import YAML", icon="upload")
        self.import_button.clicked.connect(self.import_yaml)
        self.export_button = Button(themes, "Export YAML", icon="download")
        self.export_button.clicked.connect(self.export_yaml)
        reset = Button(themes, "Reset to Default", icon="reset", variant="ghost")
        reset.clicked.connect(self.reset)
        self.summary = ChangeSummary(themes, robot["label"])
        close = Button(themes, "Close")
        close.clicked.connect(lambda: self.finish(False))
        self.save_button = Button(
            themes, f"Save revision {next_revision}", icon="check", variant="primary"
        )
        self.save_button.clicked.connect(self.save)
        for widget in (self.import_button, self.export_button, reset):
            self.footer.add(widget)
        self.footer.add_stretch()
        self.footer.add(self.summary)
        self.footer.add(close, gap=16)  # the summary's 8 px margin plus the gap
        self.footer.add(self.save_button)

        self.apply_values(profile["effective"])
        self._layout()
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    # building

    @staticmethod
    def _style_field(field: QtWidgets.QLineEdit) -> None:
        field.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        field.setProperty("valueField", True)
        field.setFont(font(13, 400, tabular=True))

    def _add_row(self, key: str, metadata: dict[str, Any]) -> None:
        default = self.defaults[key]
        field: ValueWidget
        if metadata.get("choices") or type(default) is bool:
            field = KeySelect(self.themes)
            values = (
                ["true", "false"]
                if type(default) is bool
                else [str(choice) for choice in metadata["choices"]]
            )
            field.addItems(values)
            field.currentIndexChanged.connect(self.changed)
        else:
            field = QtWidgets.QLineEdit()
            if type(default) is int:
                field.setValidator(
                    QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("-?[0-9]+"), field)
                )
                self._style_field(field)
            elif type(default) is float:
                validator = QtGui.QDoubleValidator(-1e308, 1e308, 17, field)
                validator.setLocale(QtCore.QLocale.c())
                validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
                field.setValidator(validator)
                self._style_field(field)
            field.textChanged.connect(self.changed)
        field.setAccessibleName(key)
        name, unit = parameter_label(key)
        row = ParameterRow(
            self.themes,
            key,
            name,
            unit,
            str(metadata.get("description", "")),
            default,
            field,
            self.view.content,
        )
        row.reset.clicked.connect(lambda _=False, k=key: self.reset_value(k))
        self.fields[key], self.rows[key] = field, row
        self.group_of[key] = metadata["group"]

    def _layout(self) -> None:
        place(self.title, 25, 21, 990, 22)
        for label, x, width in zip(self.labels, (25, 211, 487), (170, 260, 528), strict=True):
            place(label, x, FIELDS_TOP, width, 18)
        self.robot_field.setGeometry(25, FIELDS_TOP + 24, 170, 32)
        self.name.setGeometry(211, FIELDS_TOP + 24, 260, 32)
        self.description.setGeometry(487, FIELDS_TOP + 24, 528, 32)
        place(self.tabs, 25, TABS_TOP, self.tabs.natural_width(), 39)
        switch = self.changed_only.natural_width()
        place(self.changed_only, WIDTH - 25 - switch, TABS_TOP + 10, switch, 18)
        rows_height = FOOTER_TOP - ROWS_TOP - (ERROR_STRIP if self.error_text else 0)
        self.view.setGeometry(1, ROWS_TOP, WIDTH - 2, rows_height)
        self.footer.setGeometry(1, FOOTER_TOP + 15, WIDTH - 2, 32)
        self.show_group()

    # values

    def value(self, key: str) -> Any:
        """The field's value in the parameter's type; ValueError while the text is not a number."""
        field, default = self.fields[key], self.defaults[key]
        if isinstance(field, KeySelect):
            text = field.currentText()
            return text == "true" if type(default) is bool else text
        text = field.text()
        return (
            int(text) if type(default) is int else float(text) if type(default) is float else text
        )

    def apply_values(self, values: dict[str, Any]) -> None:
        for key, field in self.fields.items():
            value = values[key]
            field.blockSignals(True)
            if isinstance(field, KeySelect):
                field.setCurrentText(
                    ("true" if value else "false") if type(value) is bool else str(value)
                )
            else:
                field.setText(str(value))
            field.blockSignals(False)
        self.changed()

    def is_changed(self, key: str) -> bool:
        try:
            if key in self.physical:
                return bool(self.physical[key].value() != self.robot[key])
            return bool(self.value(key) != self.defaults[key])
        except ValueError:
            return True

    def changed(self, *args: Any) -> None:
        counts = dict.fromkeys(self.groups, 0)
        for key, row in self.rows.items():
            changed = self.is_changed(key)
            row.set_changed(changed)
            field = self.fields.get(key) or self.physical.get(key)
            if isinstance(field, QtWidgets.QLineEdit) and field.property("bold") != changed:
                field.setProperty("bold", changed)  # ``.er.chg .input``: 600
                field.setFont(font(13, 600 if changed else 400, tabular=True))
            if changed:
                counts[self.group_of[key]] = counts.get(self.group_of[key], 0) + 1
        for index, group in enumerate(self.groups):
            self.tabs.set_changes(index, counts.get(group, 0))
        place(self.tabs, 25, TABS_TOP, self.tabs.natural_width(), 39)
        self.summary.set_count(sum(counts.values()))
        self.footer.relayout()
        if self.changed_only.isChecked():
            self.show_group()

    def show_group(self) -> None:
        index = self.tabs.index()
        if not 0 <= index < len(self.groups):
            return
        group = self.groups[index]
        only = self.changed_only.isChecked()
        rows = [
            row
            for key, row in self.rows.items()
            if self.group_of[key] == group and (row.changed or not only)
        ]
        empty = f"No changes from {self.robot['label']} Default in {group}" if only else ""
        self.view.show_rows(rows, empty)

    def reset_value(self, key: str) -> None:
        """A row's reset button: the field shows its Default again (saving drops the key)."""
        if key in self.physical:
            self.physical[key].setValue(self.robot[key])
            return
        field, default = self.fields[key], self.defaults[key]
        if isinstance(field, KeySelect):
            field.setCurrentText(
                ("true" if default else "false") if type(default) is bool else str(default)
            )
        else:
            field.setText(str(default))

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
        for key, field in self.physical.items():
            field.setValue(self.robot[key])

    # engine

    def save(self) -> None:
        try:
            value = self.configuration()
            if self.family_id:
                value["family_id"] = self.family_id
            self.save_button.setEnabled(False)
            self.set_error("")
            self.transport.submit(self.tag + ":save", "profiles.save", value)
        except ValueError as exc:
            self.set_error(str(exc))

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
                self.set_error(str(exc))

    def received(self, tag: str, result: Any) -> None:
        if not tag.startswith(self.tag + ":"):
            return
        operation = tag.split(":")[-1]
        if operation == "save":
            self.accepted.emit(result)
            self.saved.emit()
            self.notify.emit(
                f"Saved “{result.get('name', '')}” revision {result.get('revision', '')}", "ok", 0
            )
            self.finish(True)
        elif operation == "import":
            self.transport.submit(
                self.tag + ":resolved",
                "parameters.resolve",
                {"robot_id": self.profile["robot_id"], "yaml_text": result["text"]},
            )
        elif operation == "resolved":
            self.profile["requested"] = result["requested"]
            self.apply_values(result["effective"])
            self.set_error("")
            self.notify.emit("Imported values accepted", "ok", 0)
        elif operation == "export":
            source = yaml.safe_dump(
                {"frontier_explorer": {"ros__parameters": result["effective"]}}, sort_keys=False
            )
            self.transport.submit(
                self.tag + ":written", "local.write", {"path": self.export_path, "text": source}
            )
        elif operation == "written":
            self.notify.emit("Export saved", "ok", 0)

    def failed(self, tag: str, error: str) -> None:
        if tag.startswith(self.tag + ":"):
            self.set_error(error)
            if tag == self.tag + ":save":
                self.save_button.setEnabled(True)

    def set_error(self, text: str) -> None:
        if text != self.error_text:
            self.error_text = text
            self._layout()
            self.update()

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        line = self.color("border.line")
        inner = WIDTH - 2
        painter.fillRect(QtCore.QRect(1, TABS_TOP + 38, inner, 1), line)
        header = QtCore.QRect(1, HEADER_TOP, inner, 34)
        painter.fillRect(header, self.color("surface.sunken"))
        painter.fillRect(QtCore.QRect(1, HEADER_TOP + 33, inner, 1), line)
        secondary = self.color("text.secondary")
        top = HEADER_TOP + (33 - 18) / 2
        draw_text(painter, 25, top, "Parameter", HEADER, secondary)
        default = f"{self.robot['label']} Default"
        for text, right in ((default, 1 + DEFAULT_X + COLUMN), ("Value", 1 + VALUE_X + COLUMN)):
            draw_text(painter, right - HEADER.advance(text), top, text, HEADER, secondary)
        if self.error_text:
            strip = FOOTER_TOP - ERROR_STRIP
            painter.fillRect(QtCore.QRect(1, strip, inner, 1), line)
            text = ERROR.elide(self.error_text, WIDTH - 50)
            draw_text(
                painter, 25, strip + (ERROR_STRIP - 18) / 2, text, ERROR, self.color("state.bad")
            )
        painter.fillRect(QtCore.QRect(1, FOOTER_TOP, inner, 1), line)
