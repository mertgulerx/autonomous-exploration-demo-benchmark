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
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets


@lru_cache(maxsize=32)
def _thumbnail(path: str) -> QtGui.QPixmap:
    return QtGui.QPixmap(path)


def _key(robot_id: str, reference: Any) -> str:
    if isinstance(reference, dict):
        serialized = json.dumps(reference, sort_keys=True, ensure_ascii=False, default=str)
        return "snapshot:" + hashlib.sha256(serialized.encode()).hexdigest()
    if reference in ("Default", "default", "default-" + robot_id):
        return "default"
    return "profile:" + str(reference)


def _label(robot_id: str, reference: Any) -> str:
    if _key(robot_id, reference) == "default":
        return "Default"
    if isinstance(reference, dict):
        return str(reference.get("name") or "Campaign snapshot")
    return str(reference)


def _selectable_label(target: QtWidgets.QLabel) -> None:
    target.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
    )


class _SelectableToggleLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def __init__(self, text: str):
        super().__init__(text)
        _selectable_label(self)
        self._press: QtCore.QPoint | None = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._press = (
            event.position().toPoint()
            if event.button() == QtCore.Qt.MouseButton.LeftButton
            else None
        )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._press is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
            and (event.position().toPoint() - self._press).manhattanLength()
            >= QtWidgets.QApplication.startDragDistance()
        ):
            self._press = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        clicked = self._press is not None and event.button() == QtCore.Qt.MouseButton.LeftButton
        self._press = None
        super().mouseReleaseEvent(event)
        if clicked:
            self.clicked.emit()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self._press = None
        super().mouseDoubleClickEvent(event)


class _ProfileRow(QtWidgets.QFrame):
    toggled = QtCore.Signal(object, bool)
    edit = QtCore.Signal(object)

    def __init__(
        self,
        robot_id: str,
        reference: Any,
        name: str,
        detail: str,
        checked: bool,
    ):
        super().__init__()
        self.reference = deepcopy(reference)
        self.setProperty("robotProfileRow", True)
        self.setProperty("searchText", (name + " " + detail).casefold())
        self.setProperty("selected", checked)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(8)
        self.check = QtWidgets.QCheckBox()
        self.check.setChecked(checked)
        self.check.setAccessibleName("Select profile " + name)
        self.check.setToolTip(name)
        layout.addWidget(self.check)
        text = QtWidgets.QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(0)
        title = QtWidgets.QLabel(name)
        title.setAccessibleName(name)
        title.setWordWrap(True)
        text.addWidget(title)
        if detail:
            subtitle = QtWidgets.QLabel(detail)
            subtitle.setProperty("kind", "muted")
            text.addWidget(subtitle)
            _selectable_label(subtitle)
        _selectable_label(title)
        layout.addLayout(text, 1)
        self.edit_button = QtWidgets.QToolButton()
        self.edit_button.setText("Edit")
        self.edit_button.setProperty("role", "quiet")
        self.edit_button.setAccessibleName("Edit profile " + name)
        layout.addWidget(self.edit_button)
        self.check.toggled.connect(lambda checked: self.toggled.emit(self.reference, checked))
        self.edit_button.clicked.connect(lambda: self.edit.emit(self.reference))


class _ProfilePopup(QtWidgets.QFrame):
    changed = QtCore.Signal(object)
    edit_profile = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        robot_id: str,
        robot_label: str,
        options: list[tuple[Any, str, str]],
        selected: list[Any],
        has_more: bool,
    ):
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.robot_id = robot_id
        self._options = options
        self._selected = deepcopy(selected)
        self._rows: list[_ProfileRow] = []
        self.setObjectName("robotProfilesPopup")
        self.setProperty("robotProfilesPopup", True)
        self.setMinimumWidth(350)
        self.setMaximumWidth(420)
        self.setMaximumHeight(460)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        heading = QtWidgets.QLabel("Profiles · " + robot_label)
        heading.setProperty("kind", "section")
        layout.addWidget(heading)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search profiles")
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self.list_content = QtWidgets.QWidget()
        self.list_layout = QtWidgets.QVBoxLayout(self.list_content)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(2)
        self._scroll_area.setWidget(self.list_content)
        layout.addWidget(self._scroll_area, 1)
        self.more = QtWidgets.QLabel(
            "First 250 revisions shown. Use Profiles to find older revisions."
        )
        self.more.setProperty("kind", "muted")
        self.more.setWordWrap(True)
        self.more.setVisible(has_more)
        layout.addWidget(self.more)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        self.done = QtWidgets.QPushButton("Done")
        self.done.setProperty("role", "primary")
        self.done.clicked.connect(self.hide)
        footer.addWidget(self.done)
        layout.addLayout(footer)
        self.search.textChanged.connect(self.filter_rows)
        escape = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        escape.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        escape.activated.connect(self.hide)
        self._populate()
        self.search.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def refresh(self, options: list[tuple[Any, str, str]], selected: list[Any]) -> None:
        self._options = deepcopy(options)
        self._selected = deepcopy(selected)
        self._populate()

    def _populate(self) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._rows.clear()
        selected_keys = {_key(self.robot_id, ref) for ref in self._selected}
        for reference, name, detail in self._options:
            row = _ProfileRow(
                self.robot_id,
                reference,
                name,
                detail,
                _key(self.robot_id, reference) in selected_keys,
            )
            row.toggled.connect(self._toggle)
            row.edit.connect(self._edit)
            self.list_layout.addWidget(row)
            self._rows.append(row)
        self.empty = QtWidgets.QLabel("No matching profiles")
        self.empty.setProperty("kind", "muted")
        self.list_layout.addWidget(self.empty)
        self.list_layout.addStretch()
        self._refresh_count()
        self.filter_rows(self.search.text())

    def _toggle(self, reference: Any, checked: bool) -> None:
        identity = _key(self.robot_id, reference)
        matches = [ref for ref in self._selected if _key(self.robot_id, ref) == identity]
        if checked and not matches:
            self._selected.append(deepcopy(reference))
        elif not checked and matches:
            self._selected = [ref for ref in self._selected if _key(self.robot_id, ref) != identity]
        selected_keys = {_key(self.robot_id, ref) for ref in self._selected}
        for row in self._rows:
            row.setProperty("selected", _key(self.robot_id, row.reference) in selected_keys)
            row.style().unpolish(row)
            row.style().polish(row)
        self._refresh_count()
        self.changed.emit(deepcopy(self._selected))

    def _edit(self, reference: Any) -> None:
        self.hide()
        self.edit_profile.emit(deepcopy(reference))

    def _refresh_count(self) -> None:
        self.done.setText(f"Done · {len(self._selected)} selected")

    def filter_rows(self, query: str) -> None:
        needle = query.casefold().strip()
        visible = 0
        for row in self._rows:
            match = not needle or needle in str(row.property("searchText"))
            row.setVisible(match)
            visible += int(match)
        self.empty.setVisible(visible == 0)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)


class _RobotCard(QtWidgets.QFrame):
    def __init__(self, robot: dict[str, Any], parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.robot = deepcopy(robot)
        self.robot_id = str(robot["id"])
        self.robot_label = str(robot["label"])
        self.setProperty("robotCard", True)
        self.setMinimumHeight(164)
        self.setMinimumWidth(235)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 9)
        layout.setSpacing(7)
        upper = QtWidgets.QHBoxLayout()
        upper.setSpacing(10)
        self.image = QtWidgets.QLabel()
        self.image.setFixedSize(128, 96)
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image.setAccessibleName(self.robot_label + " 3D model preview")
        self.image.setProperty("robotPreview", True)
        image_path = Path(__file__).with_name("assets") / "robots" / (self.robot_id + ".png")
        pixmap = _thumbnail(str(image_path))
        if not pixmap.isNull():
            scale = self.image.devicePixelRatioF()
            preview = pixmap.scaled(
                QtCore.QSize(
                    round(self.image.width() * scale),
                    round(self.image.height() * scale),
                ),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            preview.setDevicePixelRatio(scale)
            self.image.setPixmap(preview)
        upper.addWidget(self.image, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(5)
        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(6)
        self.name = _SelectableToggleLabel(self.robot_label)
        self.name.setProperty("kind", "section")
        self.name.setWordWrap(True)
        title_row.addWidget(self.name, 1)
        self.select = QtWidgets.QCheckBox()
        self.select.setAccessibleName("Select robot: " + self.robot_label)
        self.select.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        title_row.addWidget(self.select, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        info.addLayout(title_row)
        lidar = _number(robot.get("lidar_range"))
        movement = _number(robot.get("movement_speed"))
        turning = _number(robot.get("turning_speed"))
        specs = f"Lidar {lidar} m · {movement} m/s · {turning} rad/s"
        self.specs = _SelectableToggleLabel(specs)
        self.specs.setProperty("kind", "muted")
        self.specs.setWordWrap(True)
        info.addWidget(self.specs)
        self.reason = QtWidgets.QLabel()
        self.reason.setProperty("kind", "error")
        self.reason.setWordWrap(True)
        self.reason.setMaximumHeight(42)
        self.reason.setVisible(not bool(robot.get("available", True)))
        if not bool(robot.get("available", True)):
            detail = str(robot.get("reason") or "Required simulation packages are unavailable.")
            self.reason.setText("Unavailable: " + detail)
            self.reason.setToolTip(detail)
        info.addWidget(self.reason)
        info.addStretch(1)
        upper.addLayout(info, 1)
        layout.addLayout(upper, 1)
        self.profile_button = QtWidgets.QToolButton()
        self.profile_button.setProperty("role", "quiet")
        self.profile_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.profile_button.setText("Profiles · 1")
        self.profile_button.setAccessibleName("Select explorer profiles for " + self.robot_label)
        layout.addWidget(self.profile_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.select.toggled.connect(self._sync_state)
        self.profile_button.setEnabled(False)
        _selectable_label(self.reason)
        self.name.clicked.connect(self.select.click)
        self.specs.clicked.connect(self.select.click)
        self.image.installEventFilter(self)
        self.installEventFilter(self)
        self.select.installEventFilter(self)
        self._press: QtCore.QPoint | None = None

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.select and event.type() == QtCore.QEvent.Type.FocusIn:
            self._set_focus(True)
        elif watched is self.select and event.type() == QtCore.QEvent.Type.FocusOut:
            self._set_focus(False)
        elif (
            watched in (self, self.image)
            and event.type() == QtCore.QEvent.Type.MouseButtonPress
            and isinstance(event, QtGui.QMouseEvent)
        ):
            self._press = (
                event.position().toPoint()
                if event.button() == QtCore.Qt.MouseButton.LeftButton
                else None
            )
        elif (
            watched in (self, self.image)
            and event.type() == QtCore.QEvent.Type.MouseMove
            and self._press is not None
            and isinstance(event, QtGui.QMouseEvent)
        ):
            if (
                event.position().toPoint() - self._press
            ).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                self._press = None
        elif (
            watched in (self, self.image) and event.type() == QtCore.QEvent.Type.MouseButtonDblClick
        ):
            self._press = None
        elif (
            watched in (self, self.image)
            and event.type() == QtCore.QEvent.Type.MouseButtonRelease
            and isinstance(event, QtGui.QMouseEvent)
        ):
            if self._press is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._press = None
                self.select.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                self.select.click()
        return bool(super().eventFilter(watched, event))

    def _set_focus(self, focused: bool) -> None:
        if self.property("focused") == focused:
            return
        self.setProperty("focused", focused)
        self.style().unpolish(self)
        self.style().polish(self)

    def _sync_state(self, checked: bool) -> None:
        self.setProperty("selected", checked)
        available = bool(self.robot.get("available", True))
        self.select.setEnabled(available)
        self.profile_button.setEnabled(checked and available)
        self.select.setAccessibleName(
            f"Select robot: {self.robot_label}, {'selected' if checked else 'not selected'}"
        )
        self.select.setAccessibleDescription(
            self.specs.text() + (". " + self.reason.text() if self.reason.isVisible() else "")
        )
        self.style().unpolish(self)
        self.style().polish(self)


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:g}"


class RobotSelector(QtWidgets.QWidget):
    changed = QtCore.Signal()
    edit_profile = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._catalog: list[dict[str, Any]] = []
        self._cards: dict[str, _RobotCard] = {}
        self._selected: dict[str, bool] = {}
        self._references: dict[str, list[Any]] = {}
        self._saved: dict[str, dict[str, dict[str, Any]]] = {}
        self._has_more: dict[str, bool] = {}
        self._initialized = False
        self._open_popups: dict[str, _ProfilePopup] = {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.content = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.content)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(12)
        self.grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._scroll_area.setWidget(self.content)
        outer.addWidget(self._scroll_area)

    def set_catalog(self, robots: list[dict[str, Any]]) -> None:
        initial = not self._initialized
        self._catalog = deepcopy(robots)
        identities = {str(robot["id"]) for robot in robots}
        for robot in robots:
            identity = str(robot["id"])
            self._references.setdefault(identity, ["Default"])
            if identity not in self._selected:
                self._selected[identity] = bool(
                    initial and identity == "mogi" and robot.get("available", True)
                )
        for identity in tuple(self._cards):
            if identity not in identities:
                removed_card = self._cards.pop(identity)
                removed_card.deleteLater()
                popup = self._open_popups.pop(identity, None)
                if popup:
                    popup.hide()
                    popup.deleteLater()
        for robot in robots:
            identity = str(robot["id"])
            card = self._cards.get(identity)
            if card is None:
                card = _RobotCard(robot, self.content)
                card.select.toggled.connect(
                    lambda checked, rid=identity: self._set_robot(rid, checked)
                )
                card.profile_button.clicked.connect(
                    lambda checked=False, rid=identity: self._show_profiles(rid)
                )
                self._cards[identity] = card
            else:
                was_selected = self._selected.get(identity, False)
                old_robot = card.robot
                if old_robot != robot:
                    card.deleteLater()
                    popup = self._open_popups.pop(identity, None)
                    if popup:
                        popup.hide()
                        popup.deleteLater()
                    card = _RobotCard(robot, self.content)
                    card.select.toggled.connect(
                        lambda checked, rid=identity: self._set_robot(rid, checked)
                    )
                    card.profile_button.clicked.connect(
                        lambda checked=False, rid=identity: self._show_profiles(rid)
                    )
                    self._cards[identity] = card
                card.select.setChecked(was_selected)
            self._sync_card(identity)
        self._initialized = True
        self._layout_cards()

    def selections(self) -> dict[str, list[Any]]:
        return {
            identity: deepcopy(self._references.get(identity, ["Default"]))
            for identity in (str(robot["id"]) for robot in self._catalog)
            if self._selected.get(identity, False)
        }

    def set_selections(self, mapping: dict[str, list[Any]]) -> None:
        values = deepcopy(mapping)
        visible = {str(robot["id"]) for robot in self._catalog}
        for identity in visible:
            references = values.get(identity)
            if references is None:
                self._selected[identity] = False
            else:
                self._references[identity] = deepcopy(references)
                self._selected[identity] = True
        for identity, card in self._cards.items():
            blocker = QtCore.QSignalBlocker(card.select)
            card.select.setChecked(self._selected.get(identity, False))
            del blocker
            self._sync_card(identity)
            popup = self._open_popups.get(identity)
            if popup and popup.isVisible():
                popup.refresh(self._profile_entries(identity), self._references.get(identity, []))

    def set_profiles(
        self, robot_id: str, rows: list[dict[str, Any]], has_more: bool = False
    ) -> None:
        previous = self._saved.get(robot_id, {})
        current: dict[str, dict[str, Any]] = {}
        for row in rows:
            identity = str(row.get("id", ""))
            if not identity or identity in ("Default", "default-" + robot_id):
                continue
            current[identity] = deepcopy(row)
        selected_ids = {
            reference
            for reference in self._references.get(robot_id, [])
            if isinstance(reference, str)
            and reference not in ("Default", "default", "default-" + robot_id)
        }
        for identity in selected_ids.difference(current):
            if identity in previous:
                current[identity] = deepcopy(previous[identity])
        self._saved[robot_id] = current
        self._has_more[robot_id] = has_more
        self._sync_card(robot_id)
        popup = self._open_popups.get(robot_id)
        if popup and popup.isVisible():
            popup.refresh(self._profile_entries(robot_id), self._references.get(robot_id, []))

    def add_profile(self, robot_id: str, reference: Any, name: str) -> None:
        profile_reference = deepcopy(reference)
        if isinstance(profile_reference, str) and profile_reference not in (
            "Default",
            "default",
            "default-" + robot_id,
        ):
            self._saved.setdefault(robot_id, {}).setdefault(
                profile_reference, {"id": profile_reference, "name": name}
            )
        elif isinstance(profile_reference, dict):
            identity = _key(robot_id, profile_reference)
            if identity.startswith("profile:"):
                key = identity.removeprefix("profile:")
                self._saved.setdefault(robot_id, {}).setdefault(
                    key,
                    {
                        "id": key,
                        "name": name or profile_reference.get("name", "Campaign snapshot"),
                        "revision": profile_reference.get("revision"),
                    },
                )
        self._selected[robot_id] = True
        self._append_reference(robot_id, profile_reference)
        self._sync_card(robot_id)
        self.changed.emit()

    def replace_profile(self, robot_id: str, old_reference: Any, profile: dict[str, Any]) -> None:
        old_key = _key(robot_id, old_reference)
        self._references[robot_id] = [
            reference
            for reference in self._references.get(robot_id, ["Default"])
            if _key(robot_id, reference) != old_key
        ]
        profile_id = profile.get("id")
        profile_name = str(profile.get("name", "Profile"))
        if isinstance(profile_id, str) and profile_id:
            self._saved.setdefault(robot_id, {}).setdefault(
                profile_id,
                {
                    "id": profile_id,
                    "name": profile_name,
                    "revision": profile.get("revision"),
                },
            )
            self.add_profile(robot_id, profile_id, profile_name)
        else:
            self.add_profile(robot_id, deepcopy(profile), profile_name)

    def profile_options(self) -> list[tuple[str, str, Any, str]]:
        options: list[tuple[str, str, Any, str]] = []
        for robot in self._catalog:
            identity = str(robot["id"])
            robot_label = str(robot["label"])
            for reference, name, detail in self._profile_entries(identity):
                profile_label = name
                if detail.startswith("Revision "):
                    profile_label += " · r" + detail.removeprefix("Revision ")
                elif detail == "Campaign snapshot" and not profile_label.endswith(
                    " · campaign snapshot"
                ):
                    profile_label += " · campaign snapshot"
                options.append((identity, robot_label, deepcopy(reference), profile_label))
        return options

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._layout_cards)

    def _set_robot(self, robot_id: str, checked: bool) -> None:
        robot = next((entry for entry in self._catalog if entry["id"] == robot_id), None)
        if checked and robot is not None and not robot.get("available", True):
            card = self._cards.get(robot_id)
            if card:
                blocker = QtCore.QSignalBlocker(card.select)
                card.select.setChecked(False)
                del blocker
                self._sync_card(robot_id)
            return
        if self._selected.get(robot_id, False) == checked:
            self._sync_card(robot_id)
            return
        self._selected[robot_id] = checked
        if checked and not self._references.get(robot_id):
            self._references[robot_id] = ["Default"]
        self._sync_card(robot_id)
        self.changed.emit()

    def _sync_card(self, robot_id: str, refresh_popup: bool = True) -> None:
        card = self._cards.get(robot_id)
        if card is None:
            return
        blocker = QtCore.QSignalBlocker(card.select)
        card.select.setChecked(self._selected.get(robot_id, False))
        del blocker
        card._sync_state(self._selected.get(robot_id, False))
        count = len(self._references.get(robot_id, ["Default"]))
        card.profile_button.setText(f"Profiles · {count}")
        available = bool(card.robot.get("available", True))
        card.select.setEnabled(available)
        card.profile_button.setEnabled(self._selected.get(robot_id, False) and available)
        popup = self._open_popups.get(robot_id)
        if not available and popup and popup.isVisible():
            popup.hide()
        if refresh_popup and popup and popup.isVisible():
            popup.refresh(self._profile_entries(robot_id), self._references.get(robot_id, []))

    def _append_reference(self, robot_id: str, reference: Any) -> None:
        references = self._references.setdefault(robot_id, ["Default"])
        identity = _key(robot_id, reference)
        if all(_key(robot_id, existing) != identity for existing in references):
            references.append(deepcopy(reference))

    def _profile_entries(self, robot_id: str) -> list[tuple[Any, str, str]]:
        selected = self._references.get(robot_id, [])
        by_key = {_key(robot_id, reference): reference for reference in selected}
        entries: list[tuple[Any, str, str]] = []
        entries.append(("Default", "Default", "Robot-aware baseline"))
        saved = self._saved.get(robot_id, {})
        for identity, row in saved.items():
            if _key(robot_id, identity) == "default":
                continue
            reference = by_key.get("profile:" + identity, identity)
            revision = row.get("revision")
            name = str(row.get("name") or identity)
            detail = f"Revision {revision}" if revision is not None else "Saved profile"
            if isinstance(reference, dict):
                if reference.get("revision") is not None:
                    detail = f"Revision {reference['revision']}"
            entries.append((deepcopy(reference), name, detail))
        known = {_key(robot_id, reference) for reference, _, _ in entries}
        for reference in selected:
            identity = _key(robot_id, reference)
            if identity in known:
                continue
            name = _label(robot_id, reference)
            if identity.startswith("snapshot:") or isinstance(reference, dict):
                detail = "Campaign snapshot"
                if name.casefold() == "default":
                    name = "Default · campaign snapshot"
            else:
                detail = "Saved profile revision"
            entries.append((deepcopy(reference), name, detail))
            known.add(identity)
        return entries

    def _show_profiles(self, robot_id: str) -> None:
        robot = next((entry for entry in self._catalog if entry["id"] == robot_id), None)
        if (
            robot is None
            or not robot.get("available", True)
            or not self._selected.get(robot_id, False)
        ):
            return
        card = self._cards[robot_id]
        popup = self._open_popups.get(robot_id)
        if popup is None:
            popup = _ProfilePopup(
                self,
                robot_id,
                str(robot["label"]),
                self._profile_entries(robot_id),
                self._references.get(robot_id, ["Default"]),
                self._has_more.get(robot_id, False),
            )
            popup.changed.connect(lambda refs, rid=robot_id: self._set_profiles(rid, refs))
            popup.edit_profile.connect(
                lambda reference, rid=robot_id: self.edit_profile.emit(rid, reference)
            )
            self._open_popups[robot_id] = popup
        else:
            popup.refresh(self._profile_entries(robot_id), self._references.get(robot_id, []))
        popup.more.setVisible(self._has_more.get(robot_id, False))
        popup.adjustSize()
        anchor = card.profile_button.mapToGlobal(QtCore.QPoint(0, card.profile_button.height()))
        screen = card.profile_button.screen()
        if screen:
            bounds = screen.availableGeometry()
            if anchor.y() + popup.height() > bounds.bottom():
                anchor.setY(
                    card.profile_button.mapToGlobal(QtCore.QPoint(0, 0)).y() - popup.height()
                )
            anchor.setX(max(bounds.left(), min(anchor.x(), bounds.right() - popup.width())))
        popup.move(anchor)
        popup.show()

    def _set_profiles(self, robot_id: str, references: list[Any]) -> None:
        self._references[robot_id] = deepcopy(references)
        self._sync_card(robot_id, refresh_popup=False)
        self.changed.emit()

    def _layout_cards(self) -> None:
        width = max(self._scroll_area.viewport().width(), 1)
        columns = 1 if width < 620 else 2 if width < 940 else 3 if width < 1240 else 4
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                self.grid.removeWidget(widget)
        for index, robot in enumerate(self._catalog):
            identity = str(robot["id"])
            card = self._cards.get(identity)
            if card is not None:
                self.grid.addWidget(card, index // columns, index % columns)
        for column in range(4):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)
