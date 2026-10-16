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

from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_style import Badge, COLORS, humanize, icon, label
from .gui_support import PagedTable, Transport


class CampaignTable(PagedTable):
    action_requested = QtCore.Signal(str, str)
    permitted = {
        "start": {"QUEUED"},
        "pause": {"RUNNING", "RETRYING"},
        "resume": {"PAUSED", "STOPPED", "RECOVERY_REQUIRED", "BLOCKED"},
        "stop": {
            "QUEUED",
            "RUNNING",
            "RETRYING",
            "PAUSING",
            "PAUSED",
            "BLOCKED",
            "RECOVERY_REQUIRED",
        },
        "restart": {
            "RUNNING",
            "RETRYING",
            "PAUSING",
            "PAUSED",
            "STOPPED",
            "BLOCKED",
            "RECOVERY_REQUIRED",
            "FINISHED",
        },
    }

    def __init__(self, transport: Transport) -> None:
        super().__init__(
            transport,
            "campaign-table",
            "campaigns",
            [
                ("name", "Campaign"),
                ("state", "Status"),
                ("count", "Experiments"),
                ("report_state", "Reports"),
                ("created_ns", "Submitted"),
                ("actions", "Actions"),
            ],
        )
        self.pending: set[str] = set()
        self.row_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.row_menus: dict[str, QtWidgets.QToolButton] = {}
        self.row_actions: dict[str, dict[str, QtGui.QAction]] = {}
        self.action_rows: dict[int, str] = {}
        header = self.view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 100), (2, 100), (3, 100), (4, 190), (5, 112)):
            self.view.setColumnWidth(column, width)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(False)
        header.sectionClicked.connect(self.sort_column)
        self.model.pages_changed.connect(self.rebuild_actions)
        self.model.dataChanged.connect(self.update_actions)

    def sort_column(self, column: int) -> None:
        key = self.model.columns[column][0]
        if key == "actions":
            return
        self.sort_descending = not self.sort_descending if self.sort_by == key else False
        self.sort_by = key
        header = self.view.horizontalHeader()
        header.setSortIndicator(
            column,
            QtCore.Qt.SortOrder.DescendingOrder
            if self.sort_descending
            else QtCore.Qt.SortOrder.AscendingOrder,
        )
        header.setSortIndicatorShown(True)
        self.reset_page()

    def rebuild_actions(self) -> None:
        records = list(self.model.records())
        positions = {row: record["id"] for row, record in records}
        for row, identity in list(self.action_rows.items()):
            if (
                positions.get(row) != identity
                or self.view.indexWidget(self.model.index(row, 5)) is None
            ):
                self.view.setIndexWidget(self.model.index(row, 5), None)  # type: ignore[arg-type]
                self.row_buttons.pop(identity, None)
                self.row_menus.pop(identity, None)
                self.row_actions.pop(identity, None)
                del self.action_rows[row]
        for row, record in records:
            identity = record["id"]
            if self.action_rows.get(row) == identity:
                continue
            group = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(group)
            layout.setContentsMargins(8, 2, 8, 2)
            control = QtWidgets.QToolButton()
            control.setObjectName("campaignPlaybackButton")
            control.setProperty("tableAction", True)
            control.setIconSize(QtCore.QSize(16, 16))
            control.setFixedSize(32, 32)
            control.clicked.connect(lambda checked=False, selected=identity: self.trigger(selected))
            layout.addStretch()
            layout.addWidget(control)
            more = QtWidgets.QToolButton()
            more.setText("⋯")
            more.setProperty("tableAction", True)
            more.setFixedSize(32, 32)
            more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            more.setAccessibleName("Actions for " + record["name"])
            more.setToolTip("Campaign actions")
            menu = QtWidgets.QMenu(more)
            actions = {}
            for operation in ("stop", "restart"):
                action = menu.addAction(operation.capitalize() + " campaign")
                action.triggered.connect(
                    lambda checked=False, selected=identity, op=operation: self.trigger(
                        selected, op
                    )
                )
                actions[operation] = action
            more.setMenu(menu)
            layout.addWidget(more)
            layout.addStretch()
            self.row_buttons[identity] = control
            self.row_menus[identity] = more
            self.row_actions[identity] = actions
            self.action_rows[row] = identity
            self.view.setIndexWidget(self.model.index(row, 5), group)
        self.update_actions()

    @classmethod
    def operation_for_state(cls, state: str) -> str | None:
        return next((op for op in ("start", "pause", "resume") if state in cls.permitted[op]), None)

    def update_actions(self, *args: Any) -> None:
        for _, record in self.model.records():
            control = self.row_buttons.get(record["id"])
            if control is None:
                continue
            operation = self.operation_for_state(record["state"])
            pending = record["id"] in self.pending
            for menu_operation, menu_action in self.row_actions[record["id"]].items():
                menu_action.setEnabled(
                    not pending and record["state"] in self.permitted[menu_operation]
                )
            self.row_menus[record["id"]].setEnabled(
                any(action.isEnabled() for action in self.row_actions[record["id"]].values())
            )
            control.setIcon(
                icon(
                    "pause" if record["state"] in {"RUNNING", "RETRYING", "PAUSING"} else "play",
                    COLORS["text.heading"],
                )
            )
            control.setEnabled(operation is not None and not pending)
            control.setVisible(record["state"] != "FINISHED")
            action = operation.capitalize() if operation else humanize(record["state"])
            control.setAccessibleName(f"{action} {record['name']}")
            control.setToolTip(
                "Waiting for the engine"
                if pending
                else "Pause after active runs finish"
                if operation == "pause"
                else action
            )

    def trigger(self, identity: str, operation: str | None = None) -> None:
        if identity in self.pending:
            return
        for row, record in self.model.records():
            selected_operation = operation or self.operation_for_state(record["state"])
            if (
                record["id"] == identity
                and selected_operation is not None
                and record["state"] in self.permitted.get(selected_operation, set())
            ):
                self.view.setCurrentIndex(self.model.index(row, 0))
                self.pending.add(identity)
                self.update_actions()
                self.action_requested.emit(identity, selected_operation)
                return

    def finish_control(self, identity: str, state: str | None = None) -> None:
        self.pending.discard(identity)
        if state is not None:
            for _, record in self.model.records():
                if record["id"] == identity:
                    record["state"] = state
        self.update_actions()
        self.refresh()


class CampaignOverview(QtWidgets.QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setProperty("surface", True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)
        heading = QtWidgets.QHBoxLayout()
        self.name = label("Select a campaign", "section")
        self.name.setWordWrap(True)
        self.state = Badge("Loading")
        self.state.hide()
        self.progress_label = label("", "muted")
        heading.addWidget(self.name, 1)
        heading.addWidget(self.state)
        heading.addSpacing(16)
        heading.addWidget(self.progress_label)
        layout.addLayout(heading)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Resolved experiments")
        layout.addWidget(self.progress)
        self.values: dict[str, QtWidgets.QLabel] = {}
        counts = QtWidgets.QHBoxLayout()
        for key, title in (
            ("planned", "Planned experiments"),
            ("completed", "Completed"),
            ("limits", "Time limits"),
            ("failures", "Failures"),
        ):
            column = QtWidgets.QVBoxLayout()
            column.setSpacing(2)
            value = label("—", "metric")
            value.setAccessibleName(title)
            self.values[key] = value
            column.addWidget(value)
            column.addWidget(label(title, "muted"))
            counts.addLayout(column, 1)
        layout.addLayout(counts)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(32)
        grid.setVerticalSpacing(8)
        for index, (key, title, explanation) in enumerate(
            (
                (
                    "executed",
                    "Attempts",
                    "All executions, including retries and discarded attempts.",
                ),
                (
                    "rejections",
                    "Timing rejected",
                    "Attempts rejected by the campaign timing policy.",
                ),
                (
                    "replacements",
                    "Sequential replacements",
                    "Sequential retries, not extra repetitions.",
                ),
                (
                    "unresolved",
                    "Unresolved",
                    "Experiments without a timing-eligible result after retry.",
                ),
                ("not_done", "Not done", "Interrupted experiments waiting for a fresh execution."),
                (
                    "discards",
                    "Discards",
                    "Manually discarded attempts whose partial data was removed.",
                ),
            )
        ):
            row = QtWidgets.QHBoxLayout()
            title_label = label(title, "muted")
            value = label("—", "section")
            value.setAccessibleName(title)
            title_label.setToolTip(explanation)
            value.setToolTip(explanation)
            self.values[key] = value
            row.addWidget(title_label)
            row.addWidget(value, 1, QtCore.Qt.AlignmentFlag.AlignRight)
            grid.addLayout(row, index // 3, index % 3)
            grid.setColumnStretch(index % 3, 1)
        layout.addLayout(grid)

    def reset(self, name: str) -> None:
        self.name.setText(name)
        self.state.set_state("Loading")
        self.state.show()
        self.progress.setValue(0)
        self.progress_label.clear()
        for value in self.values.values():
            value.setText("—")

    def update_campaign(self, campaign: dict[str, Any], counters: dict[str, Any]) -> None:
        states, outcomes, attempts = counters["states"], counters["outcomes"], counters["attempts"]
        self.name.setText(campaign["name"])
        state = campaign["state"]
        tone = {
            "RUNNING": "active",
            "RETRYING": "active",
            "REPORTING": "active",
            "FINISHED": "success",
            "BLOCKED": "danger",
            "RECOVERY_REQUIRED": "warning",
            "PAUSED": "warning",
            "STOPPED": "warning",
        }.get(state, "neutral")
        self.state.set_state(humanize(state), tone)
        self.state.show()
        values = {
            "planned": campaign["count"],
            "completed": outcomes.get("completed", 0),
            "limits": outcomes.get("time_limit", 0),
            "failures": sum(
                n
                for key, n in outcomes.items()
                if key not in {"completed", "time_limit", "timing_rejection"}
            ),
            "executed": attempts["executed"],
            "rejections": attempts["timing_rejections"] or 0,
            "replacements": attempts["sequential_replacements"] or 0,
            "unresolved": states.get("UNRESOLVED", 0),
            "not_done": states.get("NOT_DONE", 0),
            "discards": attempts["discards"] or 0,
        }
        for key, count in values.items():
            self.values[key].setText(f"{count:,}")
        finished = sum(outcomes.values())
        progress = min(1000, round(1000 * finished / max(1, campaign["count"])))
        self.progress.setValue(progress)
        self.progress_label.setText(
            f"{finished:,} / {campaign['count']:,} resolved  ·  {progress / 10:g}%"
        )
        self.progress.setToolTip(
            "Resolved includes completion, time limits, failures and unresolved timing outcomes."
        )
