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

from PySide6 import QtCore, QtWidgets

from .gui_style import Badge, COLORS, humanize, icon, label
from .gui_support import PagedTable, Transport


class CampaignTable(PagedTable):
    action_requested = QtCore.Signal(str, str)
    permitted = {
        "start": {"QUEUED"},
        "pause": {"RUNNING", "RETRYING"},
        "resume": {"PAUSED", "STOPPED", "RECOVERY_REQUIRED", "BLOCKED"},
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
        self.row_buttons: dict[str, dict[str, QtWidgets.QToolButton]] = {}
        header = self.view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 100), (2, 100), (3, 100), (4, 190), (5, 120)):
            self.view.setColumnWidth(column, width)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.model.modelReset.connect(self.rebuild_actions)
        self.model.dataChanged.connect(self.update_actions)

    def rebuild_actions(self) -> None:
        self.row_buttons.clear()
        for row, record in enumerate(self.model.rows):
            identity = record["id"]
            group = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(group)
            layout.setContentsMargins(8, 2, 8, 2)
            layout.setSpacing(2)
            controls = {}
            for operation in self.permitted:
                control = QtWidgets.QToolButton()
                control.setProperty("tableAction", True)
                control.setIcon(
                    icon("pause" if operation == "pause" else "play", COLORS["text.heading"])
                )
                control.setIconSize(QtCore.QSize(16, 16))
                control.setFixedSize(32, 32)
                control.setAccessibleName(f"{operation.capitalize()} {record['name']}")
                control.setToolTip(operation.capitalize())
                control.clicked.connect(
                    lambda checked=False, op=operation, selected=identity: self.trigger(
                        selected, op
                    )
                )
                layout.addWidget(control)
                controls[operation] = control
            self.row_buttons[identity] = controls
            self.view.setIndexWidget(self.model.index(row, 5), group)
        self.update_actions()

    def update_actions(self, *args: Any) -> None:
        for record in self.model.rows:
            for operation, control in self.row_buttons.get(record["id"], {}).items():
                enabled = (
                    record["state"] in self.permitted[operation]
                    and record["id"] not in self.pending
                )
                control.setEnabled(enabled)
                control.setAccessibleName(f"{operation.capitalize()} {record['name']}")
                control.setToolTip(
                    operation.capitalize()
                    if enabled
                    else "Waiting for the engine"
                    if record["id"] in self.pending
                    else f"{operation.capitalize()} unavailable while {humanize(record['state']).lower()}"
                )

    def trigger(self, identity: str, operation: str) -> None:
        if identity in self.pending:
            return
        for row, record in enumerate(self.model.rows):
            if record["id"] == identity and record["state"] in self.permitted[operation]:
                self.view.setCurrentIndex(self.model.index(row, 0))
                self.pending.add(identity)
                self.update_actions()
                self.action_requested.emit(identity, operation)
                return

    def finish_control(self, identity: str, state: str | None = None) -> None:
        self.pending.discard(identity)
        if state is not None:
            self.model.replace(
                [
                    dict(record, state=state) if record["id"] == identity else record
                    for record in self.model.rows
                ]
            )
        else:
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
