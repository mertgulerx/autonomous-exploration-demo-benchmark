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

from .contracts import STOPPED_OUTCOMES, CampaignState, report_exclusion
from .gui_style import Badge, COLORS, humanize, icon, label
from .gui_support import PagedTable, Transport


def confirm_deletion(parent: QtWidgets.QWidget, campaigns: list[dict[str, Any]]) -> list[str]:
    """Ask before deleting; returns the IDs to delete (only deletable ones), or [] when cancelled."""
    deletable = [c for c in campaigns if c["deletable"]]
    blocked = [c for c in campaigns if not c["deletable"]]
    reasons = "; ".join(f"{c['name'] or c['campaign_id'][:8]} ({c['reason']})" for c in blocked)
    box = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Icon.Warning,
        "Delete campaigns",
        "",
        QtWidgets.QMessageBox.StandardButton.NoButton,
        parent,
    )
    if not deletable:
        box.setText("None of the selected campaigns can be deleted now.")
        box.setInformativeText(reasons)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()
        return []
    plural = "s" if len(deletable) != 1 else ""
    recorded = sum(c["recorded_bytes"] for c in deletable) / 2**20
    names = [f"{c['name']} · {c['campaign_id'][:8]}" for c in deletable]
    lines = [
        "\n".join(names[:10])
        + (f"\n… and {len(names) - 10} more (Show Details)" if len(names) > 10 else ""),
        f"{sum(c['experiments'] for c in deletable):,} experiments, "
        f"{sum(c['attempts'] for c in deletable):,} runs and "
        f"{sum(c['reports'] for c in deletable):,} report revisions; at least {recorded:.1f} MiB of "
        "recorded files.",
        "This cannot be undone. Profiles and spawn poses stay in the library.",
    ]
    if blocked:
        lines.append("Not deleted: " + reasons + ".")
    box.setText(f"Delete {len(deletable)} campaign{plural} and all their results?")
    box.setInformativeText("\n\n".join(lines))
    box.setDetailedText(
        "\n".join(
            f"{c['name']} · {c['campaign_id'][:8]} · {humanize(c['state'])} · {c['attempts']} runs · "
            f"at least {c['recorded_bytes'] / 2**20:.1f} MiB"
            for c in deletable
        )
    )
    delete = box.addButton(
        f"Delete {len(deletable)} campaign{plural}",
        QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
    )
    cancel = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.exec()
    return [c["campaign_id"] for c in deletable] if box.clickedButton() is delete else []


class CampaignTable(PagedTable):
    action_requested = QtCore.Signal(str, str)
    choice_changed = QtCore.Signal(int)
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
        # Restart runs a copy and never touches the source, so every state allows it.
        "restart": {state.value for state in CampaignState},
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
        # Rows reorder on every refresh, so the multi-selection is kept by campaign ID.
        self.chosen: dict[str, dict[str, Any]] = {}
        self.syncing = False
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.selectionModel().selectionChanged.connect(self.selection_changed)
        self.row_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.row_menus: dict[str, QtWidgets.QToolButton] = {}
        self.row_actions: dict[str, dict[str, QtGui.QAction]] = {}
        self.action_rows: dict[int, str] = {}
        self.row_widgets: dict[
            int, tuple[QtWidgets.QToolButton, QtWidgets.QToolButton, dict[str, QtGui.QAction]]
        ] = {}
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

    def selection_changed(self, *args: Any) -> None:
        if self.syncing:
            return
        selected = {index.row() for index in self.view.selectionModel().selectedRows()}
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if not modifiers & (
            QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            # A plain click replaces the choice, including campaigns on pages no longer loaded.
            self.chosen.clear()
        for row, record in self.model.records():
            if row in selected:
                self.chosen[record["id"]] = record
            else:
                self.chosen.pop(record["id"], None)
        self.choice_changed.emit(len(self.chosen))

    def _received(self, tag: str, result: Any) -> None:
        if tag != self.tag:
            return
        self.syncing = True
        try:
            super()._received(tag, result)
        finally:
            self.syncing = False
        self.restore_choice()

    def restore_choice(self) -> None:
        """Re-select the chosen campaigns wherever the refreshed page placed them."""
        selection = QtCore.QItemSelection()
        last = self.model.columnCount() - 1
        for row, record in self.model.records():
            if record["id"] in self.chosen:
                self.chosen[record["id"]] = record
                selection.select(self.model.index(row, 0), self.model.index(row, last))
        self.syncing = True
        try:
            self.view.selectionModel().select(
                selection, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
        finally:
            self.syncing = False
        self.choice_changed.emit(len(self.chosen))

    def clear_choice(self) -> None:
        self.chosen.clear()
        self.restore_choice()

    def forget(self, identities: list[str]) -> None:
        for identity in identities:
            self.chosen.pop(identity, None)
        self.restore_choice()

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
        records = dict(self.model.records())
        for row in list(self.row_widgets):
            if row not in records or self.view.indexWidget(self.model.index(row, 5)) is None:
                self.view.setIndexWidget(self.model.index(row, 5), None)  # type: ignore[arg-type]
                del self.row_widgets[row]
        for row, record in records.items():
            if row not in self.row_widgets:
                self.row_widgets[row] = self.action_widgets(row)
            if self.action_rows.get(row) != record["id"]:
                # A row's widgets are reused when another campaign moves into the row; the
                # handlers read the row's campaign at click time. Keyboard focus is dropped, as
                # when the previous campaign's widgets were replaced.
                control, more, _ = self.row_widgets[row]
                more.setAccessibleName("Actions for " + record["name"])
                for widget in (control, more):
                    if widget.hasFocus():
                        widget.clearFocus()
        self.action_rows = {row: record["id"] for row, record in records.items()}
        self.row_buttons = {self.action_rows[row]: w[0] for row, w in self.row_widgets.items()}
        self.row_menus = {self.action_rows[row]: w[1] for row, w in self.row_widgets.items()}
        self.row_actions = {self.action_rows[row]: w[2] for row, w in self.row_widgets.items()}
        self.update_actions()

    def action_widgets(
        self, row: int
    ) -> tuple[QtWidgets.QToolButton, QtWidgets.QToolButton, dict[str, QtGui.QAction]]:
        group = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(group)
        layout.setContentsMargins(8, 2, 8, 2)
        control = QtWidgets.QToolButton()
        control.setObjectName("campaignPlaybackButton")
        control.setProperty("tableAction", True)
        control.setIconSize(QtCore.QSize(16, 16))
        control.setFixedSize(32, 32)
        control.clicked.connect(lambda checked=False: self.trigger(self.action_rows[row]))
        layout.addStretch()
        layout.addWidget(control)
        more = QtWidgets.QToolButton()
        more.setText("⋯")
        more.setProperty("tableAction", True)
        more.setFixedSize(32, 32)
        more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        more.setToolTip("Campaign actions")
        menu = QtWidgets.QMenu(more)
        # The menu acts on the campaign of the row it was opened from, even if rows shift while
        # it is open.
        opened: dict[str, str] = {}
        menu.aboutToShow.connect(lambda: opened.update(campaign=self.action_rows[row]))
        actions = {}
        for operation in ("stop", "restart"):
            action = menu.addAction(operation.capitalize() + " campaign")
            action.triggered.connect(
                lambda checked=False, op=operation: self.trigger(opened["campaign"], op)
            )
            actions[operation] = action
        more.setMenu(menu)
        layout.addWidget(more)
        layout.addStretch()
        self.view.setIndexWidget(self.model.index(row, 5), group)
        return control, more, actions

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
            if key == "failures":
                value.setToolTip(
                    "Startup, runtime, storage, engine-interruption and unresolved timing outcomes; "
                    "excluded from campaign reports."
                )
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
                    "Discarded attempts whose partial data was removed (campaign Stop, "
                    "experiment stop or restart).",
                ),
                (
                    "no_movement",
                    "No movement",
                    "Final runs in which the activated explorer never moved the robot; reported.",
                ),
                (
                    "stopped",
                    "Stopped",
                    "Runs stopped from the Monitor; data kept, excluded from campaign reports.",
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
            "failures": sum(n for key, n in outcomes.items() if report_exclusion(key) == "failure"),
            "executed": attempts["executed"],
            "rejections": attempts["timing_rejections"] or 0,
            "replacements": attempts["sequential_replacements"] or 0,
            "unresolved": states.get("UNRESOLVED", 0),
            "not_done": states.get("NOT_DONE", 0),
            "discards": attempts["discards"] or 0,
            "no_movement": outcomes.get("no_movement", 0),
            "stopped": sum(n for key, n in outcomes.items() if key in STOPPED_OUTCOMES),
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
            "Resolved includes completion, time limits, failures, stopped runs and unresolved timing "
            "outcomes."
        )
