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

from pathlib import Path
import sys
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .contracts import CAMPAIGN_DELETION_LIMIT, PROFILE_DELETION_LIMIT
from .gui_support import PagedTable, TextSelection, Transport, button, stale_engine_message
from .gui_pages import SpawnEditor
from .gui_builder import CampaignBuilder
from .gui_campaigns import CampaignOverview, CampaignTable, confirm_deletion
from .gui_run_map import RunMap
from .gui_profiles import ProfileEditor
from .gui_style import (
    DetailsPanel,
    Badge,
    COLORS,
    cell_text,
    humanize,
    icon,
    label,
    stylesheet,
    surface,
)
from .gui_monitor import LiveMap, WorkerCards, RunMetrics
from .monitoring import MONITOR_LAYERS, PathCursor, accept_page
from .gui_reports import ReportBrowser
from .gui_logs import LogViewer

LAYER_TITLES = {"explored": "Explored cells", "navigation": "Nav2 goal and plan"}


def metric_number(value: Any, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, (float, int)) else "—"


def profile_deletion_message(rows: list[dict[str, Any]], robot_labels: dict[str, str]) -> str:
    """Confirmation text for deleting whole profiles (one row per revision family)."""
    count = len(rows)
    lines = [
        f"Delete {count} profile{'s' if count != 1 else ''} and all "
        f"{'their' if count != 1 else 'its'} saved revisions? This cannot be undone.",
        "",
        *(f"{row['name']} · {robot_labels.get(row['owner'], row['owner'])}" for row in rows[:10]),
    ]
    if count > 10:
        lines.append(f"and {count - 10} more")
    lines += [
        "",
        "Submitted campaigns, their results and reports keep the exact parameters they recorded. "
        "Campaign YAML files that name a deleted revision ID will no longer validate.",
    ]
    return "\n".join(lines)


class Window(QtWidgets.QMainWindow):
    def __init__(self, socket: Path):
        super().__init__()
        app = QtWidgets.QApplication.instance()
        if isinstance(app, QtWidgets.QApplication):
            self.text_selection = app.findChild(TextSelection) or TextSelection(app)
            app.installEventFilter(self.text_selection)
        self.setWindowTitle("Exploration · Benchmark Suite")
        self.resize(1440, 960)
        self.setMinimumSize(1100, 720)
        self.settings = QtCore.QSettings("Mert Güler", "Exploration Benchmark Suite")
        self.transport = Transport(socket)
        self.catalog: dict[str, Any] = {}
        self.catalog_pending = True
        self.status: dict[str, Any] = {}
        self.campaign_id: str | None = None
        self.profile_row: dict[str, Any] | None = None
        self.result_selection: str | None = None
        self.monitor_id: str | None = None
        self.monitor_requested: str | None = None
        self.selected_attempt: str | None = None
        self.profile_detail_tag = ""
        self.monitor_cursor = PathCursor()
        self.layer_revisions: dict[str, int] = {}
        self.reference_maps: dict[str, dict[str, Any]] = {}
        self.reference_mismatch = False
        self.event_sequence = 0
        self.status_pending, self.monitor_pending, self.events_pending = False, False, False
        self.campaign_status_pending = False
        self._campaign_page = "campaigns"
        self.page_widgets: dict[str, QtWidgets.QWidget] = {}
        self.pages = QtWidgets.QStackedWidget()
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        topbar = QtWidgets.QWidget()
        topbar.setObjectName("topbar")
        navigation_bar = QtWidgets.QHBoxLayout(topbar)
        navigation_bar.setContentsMargins(24, 4, 24, 4)
        navigation_bar.setSpacing(24)
        navigation_bar.addWidget(label("Exploration", "brand"))
        self.navigation = QtWidgets.QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setIconSize(QtCore.QSize(20, 20))
        self.navigation.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.navigation.setWrapping(False)
        self.navigation.setMovement(QtWidgets.QListView.Movement.Static)
        self.navigation.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.navigation.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.navigation.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.navigation.setFixedHeight(48)
        navigation_bar.addWidget(self.navigation, 1)
        root.addWidget(topbar)
        root.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.tables: list[PagedTable] = []
        self.make_dashboard()
        self.make_monitor()
        self.make_profiles()
        self.make_spawns()
        self.make_results()
        self.make_builder()
        self.navigation.currentRowChanged.connect(self.show_page)
        self.navigation.setCurrentRow(0)
        self.apply_style()
        geometry = self.settings.value("geometry")
        if isinstance(geometry, QtCore.QByteArray):
            self.restoreGeometry(geometry)
        for split in self.findChildren(QtWidgets.QSplitter):
            if split.objectName():
                state = self.settings.value("split/" + split.objectName())
                if isinstance(state, QtCore.QByteArray):
                    split.restoreState(state)
        self.transport.received.connect(self.received)
        self.transport.failed.connect(self.failed)
        self.transport.start()
        self.transport.submit("catalog", "catalog.get")
        self.poll = QtCore.QTimer(self)
        self.poll.setInterval(1000)
        self.poll.timeout.connect(self.refresh_status)
        self.poll.start()
        self.refresh_status()

    def page(
        self, name: str, key: str, *, navigation: bool = True, heading: bool = True
    ) -> QtWidgets.QVBoxLayout:
        widget = QtWidgets.QWidget()
        widget.setObjectName("page-" + key)
        self.pages.addWidget(widget)
        self.page_widgets[key] = widget
        if navigation:
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            self.navigation.addItem(item)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        if heading:
            layout.addWidget(label(name, "title"))
        return layout

    def go(self, key: str) -> None:
        if key in ("campaigns", "builder"):
            self._campaign_page = key
        self.pages.setCurrentWidget(self.page_widgets[key])
        navigation_key = "campaigns" if key == "builder" else key
        blocker = QtCore.QSignalBlocker(self.navigation)
        for index in range(self.navigation.count()):
            if self.navigation.item(index).data(QtCore.Qt.ItemDataRole.UserRole) == navigation_key:
                self.navigation.setCurrentRow(index)
                self.navigation.item(index).setSelected(True)
                break
        else:
            self.navigation.clearSelection()
        del blocker
        for table in self.tables:
            if table.isVisible():
                table.refresh()
        if key != "monitor":
            self.close_monitor()
        else:
            self.follow_single_worker()
            self.inspect_selected()

    def current_page(self) -> str:
        return self.pages.currentWidget().objectName().removeprefix("page-")

    def open_builder(self) -> None:
        # The engine or library may have changed since an earlier validation.
        self.builder.invalidate()
        self.go("builder")

    def make_dashboard(self) -> None:
        layout = self.page("Campaigns", "campaigns", heading=False)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label("Campaigns", "title"))
        self.campaign_total = Badge("—")
        self.campaign_total.setAccessibleName("Campaign count")
        heading.addWidget(self.campaign_total)
        heading.addSpacing(16)
        self.hide_finished = QtWidgets.QCheckBox("Hide finished campaigns")
        self.hide_finished.setChecked(
            bool(self.settings.value("hide_finished_campaigns", False, type=bool))
        )
        heading.addWidget(self.hide_finished)
        heading.addStretch()
        heading.addWidget(label("Active workers", "muted"))
        self.active_worker_count = label("—", "section")
        self.active_worker_count.setAccessibleName("Active workers")
        heading.addWidget(self.active_worker_count)
        heading.addSpacing(16)
        self.deletion_pending = False
        self.delete_campaigns_button = button(
            "Delete…", self.request_campaign_deletion, heading, role="danger"
        )
        self.delete_campaigns_button.setToolTip(
            "Delete the selected finished or stopped campaigns and all their results "
            "(Ctrl or Shift to select several)"
        )
        self.delete_campaigns_button.setEnabled(False)
        button("New campaign", self.open_builder, heading, glyph="plus")
        layout.addLayout(heading)
        self.summary = label("Connecting to the background engine…", "muted")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.campaign_overview = CampaignOverview()
        self.progress_bar = self.campaign_overview.progress
        layout.addWidget(self.campaign_overview)
        self.campaigns = CampaignTable(self.transport)
        self.campaigns.selected.connect(self.select_campaign)
        self.campaigns.action_requested.connect(self.control_campaign_row)
        self.campaigns.choice_changed.connect(self.update_delete_button)
        if self.hide_finished.isChecked():
            self.campaigns.filters = {"hide_finished": True}
        self.hide_finished.toggled.connect(self.filter_campaigns)
        self.tables.append(self.campaigns)
        layout.addWidget(self.campaigns, 1)

    def make_builder(self) -> None:
        layout = self.page("New campaign", "builder", navigation=False, heading=False)
        self.builder = CampaignBuilder(self.transport)
        self.builder.return_requested.connect(lambda: self.go("campaigns"))
        self.builder.edit_profile.connect(self.edit_builder_profile)
        layout.addWidget(self.builder, 1)

    def make_profiles(self) -> None:
        layout = self.page("Profiles", "profiles")
        search = QtWidgets.QHBoxLayout()
        self.profile_robot = QtWidgets.QComboBox()
        self.profile_robot.currentIndexChanged.connect(self.refresh_profile_list)
        self.profile_search = QtWidgets.QLineEdit()
        self.profile_search.setPlaceholderText("Search profiles…")
        for field in (self.profile_robot, self.profile_search):
            search.addWidget(field)
        button(
            "New profile", lambda: self.open_profile("new"), search, role="primary", glyph="plus"
        )
        self.profile_filter_timer = QtCore.QTimer(self)
        self.profile_filter_timer.setSingleShot(True)
        self.profile_filter_timer.setInterval(250)
        self.profile_filter_timer.timeout.connect(self.refresh_profile_list)
        self.profile_search.textChanged.connect(lambda _: self.profile_filter_timer.start())
        layout.addLayout(search)
        split = QtWidgets.QSplitter()
        split.setObjectName("profiles-split")
        self.profiles = PagedTable(
            self.transport,
            "profile-table",
            "library",
            [("name", "Profile"), ("revision", "Revision"), ("created_ns", "Created")],
            method="profiles.list",
            multiple=True,
        )
        self.profiles.selected.connect(self.select_profile)
        self.tables.append(self.profiles)
        split.addWidget(self.profiles)
        panel = surface()
        details = QtWidgets.QVBoxLayout(panel)
        details.setContentsMargins(20, 20, 20, 20)
        self.profile_details = DetailsPanel("Profile details")
        details.addWidget(self.profile_details, 1)
        bar = QtWidgets.QHBoxLayout()
        button("Edit", lambda: self.open_profile("edit"), bar)
        button("Clone", lambda: self.open_profile("clone"), bar)
        details.addLayout(bar)
        split.addWidget(panel)
        split.setSizes([650, 420])
        layout.addWidget(split, 1)
        bar = QtWidgets.QHBoxLayout()
        button("Import profile", self.import_profile, bar)
        button("Export profile", self.export_profile, bar)
        bar.addStretch()
        delete = button("Delete", self.delete_profiles, bar, role="danger")
        delete.setToolTip(
            "Delete the selected profiles and all their revisions (Ctrl or Shift to select several)"
        )
        layout.addLayout(bar)

    def make_spawns(self) -> None:
        layout = self.page("Worlds & Spawns", "worlds")
        self.spawn_editor = SpawnEditor(self.transport)
        self.spawn_editor.changed.connect(self.refresh_libraries)
        self.tables.extend(self.spawn_editor.tables)
        layout.addWidget(self.spawn_editor, 1)

    def make_monitor(self) -> None:
        layout = self.page("Monitor", "monitor")
        self.workers = WorkerCards()
        self.workers.chosen.connect(self.choose_worker)
        layout.addWidget(self.workers)
        title_row = QtWidgets.QHBoxLayout()
        self.focus_title = label("No worker selected", "section")
        title_row.addWidget(self.focus_title, 1)
        self.stop_run_button = button("Stop run", self.stop_selected_run, title_row, role="danger")
        self.stop_run_button.setToolTip(
            "End the selected run; its data is kept and campaign reports exclude it"
        )
        self.stop_run_button.setEnabled(False)
        self.stop_pending: set[str] = set()
        self.stop_requested_runs: set[str] = set()
        layout.addLayout(title_row)
        self.focus_tabs = QtWidgets.QTabWidget()
        inspection = QtWidgets.QSplitter()
        inspection.setObjectName("monitor-inspection-split")
        map_panel = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_panel)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self.monitor_legend = label(
            "Line: Gazebo ground-truth path · Shaded cells: explored (coverage metric) · "
            "× and dashed line: Nav2 goal and plan (SLAM-frame estimate)",
            "muted",
        )
        self.monitor_legend.setWordWrap(True)
        map_layout.addWidget(self.monitor_legend)
        self.live_map = LiveMap()
        map_layout.addWidget(self.live_map, 1)
        self.monitor_status = label("", "muted")
        self.monitor_status.setWordWrap(True)
        map_layout.addWidget(self.monitor_status)
        inspection.addWidget(map_panel)
        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.setInterval(100)
        self.monitor_timer.timeout.connect(self.poll_monitor)
        panel = surface()
        pane = QtWidgets.QVBoxLayout(panel)
        pane.setContentsMargins(16, 16, 16, 16)
        pane.addWidget(label("Run measurements", "section"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.run_metrics = RunMetrics()
        scroll.setWidget(self.run_metrics)
        pane.addWidget(scroll, 1)
        panel.setMinimumWidth(290)
        inspection.addWidget(panel)
        inspection.setSizes([760, 310])
        inspection.setStretchFactor(0, 1)
        self.focus_tabs.addTab(inspection, "Map && metrics")
        self.log_viewer = LogViewer(self.transport)
        self.focus_tabs.addTab(self.log_viewer, "Logs")
        events = QtWidgets.QWidget()
        event_layout = QtWidgets.QVBoxLayout(events)
        event_layout.setContentsMargins(0, 12, 0, 0)
        self.log_filter = QtWidgets.QLineEdit()
        self.log_filter.setPlaceholderText("Filter lifecycle events…")
        event_layout.addWidget(self.log_filter)
        self.logs = QtWidgets.QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.document().setMaximumBlockCount(2000)
        event_layout.addWidget(self.logs, 1)
        self.focus_tabs.addTab(events, "Events")
        layout.addWidget(self.focus_tabs, 1)

    def make_results(self) -> None:
        layout = self.page("Results", "results")
        self.results_context = label("Choose a campaign in Campaigns to narrow this view", "muted")
        layout.addWidget(self.results_context)
        tabs = QtWidgets.QTabWidget()
        runs = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(runs)
        run_layout.setContentsMargins(0, 12, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.result_kind = QtWidgets.QComboBox()
        self.result_kind.addItems(["Attempts", "Experiments"])
        self.result_robot, self.result_world = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.result_outcome = QtWidgets.QComboBox()
        self.result_outcome.addItem("All outcomes", None)
        for value in (
            "completed",
            "time_limit",
            "no_movement",
            "startup_failure",
            "runtime_failure",
            "timing_rejection",
            "engine_interruption",
            "manual_discard",
            "manual_stop",
            "storage_failure",
        ):
            self.result_outcome.addItem(humanize(value), value)
        for field in (self.result_kind, self.result_robot, self.result_world, self.result_outcome):
            bar.addWidget(field)
        run_layout.addLayout(bar)
        secondary = QtWidgets.QHBoxLayout()
        self.result_profile, self.result_spawn = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.result_profile.addItem("All profiles", None)
        self.result_spawn.addItem("All spawns", None)
        secondary.addWidget(self.result_profile)
        secondary.addWidget(self.result_spawn)
        secondary.addStretch()
        run_layout.addLayout(secondary)
        split = QtWidgets.QSplitter()
        split.setObjectName("results-split")
        self.results = PagedTable(
            self.transport, "result-table", "attempts", self.result_columns("attempts")
        )
        self.results.selected.connect(self.select_result)
        self.results.view.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.tables.append(self.results)
        split.addWidget(self.results)
        panel = surface()
        pane = QtWidgets.QVBoxLayout(panel)
        pane.setContentsMargins(20, 20, 20, 20)
        pane.addWidget(label("Run summary", "section"))
        summary = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        summary.setObjectName("results-summary-split")
        self.run_map = RunMap(self.transport)
        summary.addWidget(self.run_map)
        self.result_detail = DetailsPanel(None)
        summary.addWidget(self.result_detail)
        summary.setSizes([360, 300])
        pane.addWidget(summary, 1)
        split.addWidget(panel)
        split.setSizes([620, 470])
        run_layout.addWidget(split, 1)
        tabs.addTab(runs, "Runs")
        self.report_browser = ReportBrowser(self.transport)
        self.reports = self.report_browser.reports
        self.tables.append(self.reports)
        tabs.addTab(self.report_browser, "Reports")
        tabs.currentChanged.connect(lambda: self.refresh_visible_tables())
        layout.addWidget(tabs, 1)
        self.result_filter_timer = QtCore.QTimer(self)
        self.result_filter_timer.setSingleShot(True)
        self.result_filter_timer.setInterval(0)
        self.result_filter_timer.timeout.connect(self.filter_results)
        for field in (
            self.result_kind,
            self.result_robot,
            self.result_world,
            self.result_outcome,
            self.result_profile,
            self.result_spawn,
        ):
            field.currentIndexChanged.connect(lambda _: self.result_filter_timer.start())

    @staticmethod
    def result_columns(table: str) -> list[tuple[str, str]]:
        return [
            ("robot_id", "Robot"),
            ("outcome", "Outcome"),
            ("world_id", "World"),
            ("profile_name", "Profile"),
            ("spawn_name", "Spawn"),
            ("repetition", "Repeat"),
        ]

    def refresh_visible_tables(self) -> None:
        for table in self.tables:
            if table.isVisible():
                table.refresh()

    def show_page(self, index: int) -> None:
        if index >= 0:
            key = self.navigation.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
            self.go(self._campaign_page if key == "campaigns" else key)

    def refresh_status(self) -> None:
        if not self.status_pending:
            self.status_pending = True
            self.transport.submit("status", "engine.status")
        if self.current_page() == "monitor" and not self.events_pending:
            self.events_pending = True
            self.transport.submit(
                "events", "events.read", {"after_sequence": self.event_sequence, "limit": 100}
            )
        if self.campaign_id and not self.campaign_status_pending:
            self.campaign_status_pending = True
            self.transport.submit(
                "campaign-status", "campaign.status", {"campaign_id": self.campaign_id}
            )

    def received(self, tag: str, result: Any) -> None:
        if tag == "catalog":
            self.catalog_pending = False
            self.catalog = result
            self.fill_catalog()
        elif tag == "status":
            self.status_pending, self.status = False, result
            self.summary.hide()
            self.show_deletions(result.get("deletions"))
            self.active_worker_count.setText(str(len(result["active"])))
            self.campaign_total.setText(f"{result['campaigns']['total']:,}")
            if not self.catalog and not self.catalog_pending:
                self.catalog_pending = True
                self.transport.submit("catalog", "catalog.get")
            self.workers.update_runs(result["active"], self.catalog)
            self.stop_requested_runs |= {
                row["attempt"]["id"] for row in result["active"] if row.get("stop_requested")
            }
            if self.current_page() == "monitor":
                self.follow_single_worker()
            selected = next(
                (r for r in result["active"] if r["attempt"]["id"] == self.selected_attempt), None
            )
            if selected:
                self.run_metrics.update_metrics(selected["metrics"])
                if self.current_page() == "monitor":
                    self.inspect_selected()
            elif self.monitor_id:
                stopped = self.monitor_id in self.stop_requested_runs
                self.close_monitor()
                self.focus_title.setText(
                    ("Run stopped" if stopped else "Run ended") + " · retained live snapshot"
                )
                self.monitor_status.setText(
                    "Run ended: the last received path and pose stay on the map"
                )
            self.update_stop_button()
        elif tag.startswith("stop-run:"):
            self.stop_pending.discard(tag.removeprefix("stop-run:"))
            self.statusBar().showMessage(
                "Stop already requested; the run is finalizing"
                if result.get("already_requested")
                else "Stop requested; the run is finalizing",
                8000,
            )
            self.refresh_status()
        elif tag == "submit":
            self.select_campaign({"id": result["campaign_id"], "name": self.builder.name.text()})
            self.builder.show_step(0)
            self.campaigns.refresh()
            self.go("campaigns")
        elif tag.startswith("library-"):
            self.fill_revisions(tag, result)
        elif tag == "editor-default":
            self.show_editor(result)
        elif tag in ("profile-open:edit", "profile-open:clone"):
            self.show_editor(result, clone=tag.endswith("clone"))
        elif tag.startswith("builder-profile:") and tag == getattr(self, "builder_profile_tag", ""):
            editor = self.show_editor(result)
            owner, reference = self.builder_profile_context
            editor.accepted.connect(
                lambda profile, r=owner, old=reference: self.accept_builder_profile(r, old, profile)
            )
        elif tag == self.profile_detail_tag:
            if self.profile_row and result["id"] == self.profile_row["id"]:
                self.profile_details.set_record(
                    {
                        "name": result["name"],
                        "robot": self.results.model.names.get("robot_id", {}).get(
                            result["robot_id"], humanize(result["robot_id"])
                        ),
                        "revision": result.get("revision"),
                        "description": result.get("description", ""),
                        "changed_parameters": len(result.get("requested", {})),
                    }
                )
        elif tag == "result-detail" and result["id"] == self.result_selection:
            metrics = result.get("metrics") or {}
            summary = {
                key: value
                for key, value in self.result_row.items()
                if key
                in {
                    "robot_id",
                    "world_id",
                    "profile_name",
                    "spawn_name",
                    "repetition",
                    "outcome",
                    "phase",
                    "state",
                }
            }
            for key in (
                "simulation_elapsed",
                "wall_elapsed",
                "coverage_percent",
                "distance_m",
                "cpu_average_percent",
                "rss_average_mib",
                "rtf",
                "coverage_status",
            ):
                summary[key] = metrics.get(key)
            for key, title in (("robot_id", "robot"), ("world_id", "world")):
                if key in summary:
                    identity = summary.pop(key)
                    summary[title] = self.results.model.names.get(key, {}).get(
                        identity, humanize(identity)
                    )
            self.result_detail.set_record(summary)
        elif tag == "events":
            self.events_pending = False
            self.event_sequence = result["last_sequence"]
            for event in result["events"]:
                payload = event["payload"]
                parts = [
                    cell_text("utc_ns", event.get("utc_ns")),
                    humanize(event["kind"].replace(".", " ")),
                ]
                for key in ("phase", "state", "outcome", "detail", "direction"):
                    if isinstance(payload, dict) and payload.get(key):
                        parts.append(cell_text(key, payload[key]))
                line = " · ".join(parts)
                if self.log_filter.text().lower() in line.lower():
                    self.logs.appendPlainText(line)
        elif tag.startswith("monitor-map:"):
            self.reference_maps[result["world_id"]] = result
            selected = next(
                (
                    row
                    for row in self.status.get("active", [])
                    if row["attempt"]["id"] == self.selected_attempt
                ),
                None,
            )
            if (
                selected is not None
                and selected["experiment"]["world_id"] == result["world_id"]
                and self.live_map.reference_world is None
            ):
                self.live_map.set_reference(result)
        elif tag == "monitor":
            self.monitor_pending = False
            if self.monitor_id and result.get("attempt_id") == self.monitor_id:
                self.apply_monitor(result)
        elif tag == "profile-delete":
            self.profile_row = None
            self.profile_details.clear()
            deleted: dict[str, set[str]] = {}
            for revision in result["revisions"]:
                deleted.setdefault(revision["owner"], set()).add(revision["id"])
            for owner, revisions in deleted.items():
                self.builder.robot_selector.forget_profiles(owner, revisions)
            self.refresh_libraries()
            count = len(result["family_ids"])
            self.statusBar().showMessage(
                f"Deleted {count} profile{'s' if count != 1 else ''} "
                f"({result['deleted_revisions']} revisions); {result['referenced_campaigns']} "
                "campaigns keep their recorded parameters",
                10000,
            )
        elif tag == "profile-import":
            self.refresh_libraries()
            self.statusBar().showMessage("Library saved", 6000)
        elif tag.endswith("-read"):
            self.read_import(tag, result["text"])
        elif tag == "profile-export":
            self.transport.submit(
                "file-saved", "local.write", {"path": self.export_path, "text": result["yaml"]}
            )
        elif tag == "file-saved":
            self.statusBar().showMessage("Export saved", 5000)
        elif tag.startswith("campaign-control:"):
            self.campaigns.finish_control(
                tag.removeprefix("campaign-control:"), result.get("state", "PAUSING")
            )
        elif tag == "campaign-delete-preview":
            identities = confirm_deletion(self, result["campaigns"])
            if identities:
                self.transport.submit(
                    "campaign-delete", "campaign.delete", {"campaign_ids": identities}
                )
            else:
                self.deletion_pending = False
                self.update_delete_button()
        elif tag == "campaign-delete":
            self.deletion_pending = False
            gone = result["deleted"] + result["already_deleted"]
            self.campaigns.forget(gone)
            if self.campaign_id in gone:
                self.clear_campaign()
            self.campaigns.refresh()
            count = len(result["deleted"])
            self.statusBar().showMessage(
                f"Deleted {count} campaign{'s' if count != 1 else ''}; removing files…", 10000
            )
            self.update_delete_button()
        elif tag.startswith("campaign-restart:"):
            self.campaigns.finish_control(tag.removeprefix("campaign-restart:"))
            self.campaigns.view.clearSelection()
            self.campaigns.view.setCurrentIndex(QtCore.QModelIndex())
            self.select_campaign({"id": result["campaign_id"], "name": result["name"]})
            self.statusBar().showMessage(
                f"Started “{result['name']}”"
                if result["state"] == "RUNNING"
                else f"Queued “{result['name']}”; it runs after the current campaign",
                10000,
            )
            self.campaigns.refresh()
        elif tag == "campaign-status":
            self.campaign_status_pending = False
            campaign, counters = result["campaign"], result["counters"]
            if campaign["id"] != self.campaign_id:
                return
            self.campaign_overview.update_campaign(campaign, counters)

    def failed(self, tag: str, error: str) -> None:
        if tag.startswith("run-map:"):
            return  # the run map reports its own errors
        if tag.startswith("report-generate:") and tag != self.report_browser.generation_tag:
            return
        if tag == "catalog":
            self.catalog_pending = False
        if tag == "status":
            self.status_pending = False
            self.summary.setText("Engine connection unavailable: " + error)
            self.summary.show()
            self.active_worker_count.setText("—")
            self.campaign_total.setText("—")
        elif tag == "monitor":
            # The Monitor reports its own errors; polling at 10 Hz must not flood the status bar.
            self.monitor_pending = False
            stale = stale_engine_message(
                error, "Unsupported operation: monitor.get", "the ground-truth Monitor"
            )
            if stale:
                self.monitor_status.setText(stale)
                self.monitor_timer.stop()
            elif self.monitor_id and self.monitor_requested == self.monitor_id:
                self.monitor_status.setText(f"Monitor unavailable: {error}")
            return
        elif tag.startswith("monitor-map:"):
            self.monitor_status.setText(f"World map unavailable: {error}")
        elif tag == "events":
            self.events_pending = False
        elif tag == "campaign-status":
            self.campaign_status_pending = False
            if "Campaign does not exist" in error:
                self.clear_campaign()
        elif tag == "profile-delete":
            stale = stale_engine_message(error, "Unsupported library operation", "profile deletion")
            if stale:
                error = stale
            else:
                self.refresh_profile_list()
        elif tag in ("campaign-delete-preview", "campaign-delete"):
            self.deletion_pending = False
            self.update_delete_button()
            error = (
                stale_engine_message(
                    error, "Unsupported operation: campaign.delete", "campaign deletion"
                )
                or error
            )
        elif tag.startswith("campaign-control:"):
            self.campaigns.finish_control(tag.removeprefix("campaign-control:"))
        elif tag.startswith("campaign-restart:"):
            self.campaigns.finish_control(tag.removeprefix("campaign-restart:"))
            stale = stale_engine_message(
                error, "Unsupported operation: campaign.restart_copy", "Restart as a copy"
            )
            if stale:
                error = stale + " No campaign was changed."
        elif tag.startswith("stop-run:"):
            self.stop_pending.discard(tag.removeprefix("stop-run:"))
            self.update_stop_button()
            stale = stale_engine_message(error, "Unsupported operation: attempt.stop", "Stop run")
            if stale:
                # A dialog, not the status bar: the run keeps going.
                QtWidgets.QMessageBox.information(
                    self, "Stop run unavailable", "The run continues. " + stale
                )
                return
        self.statusBar().showMessage(error, 12000)

    def fill_catalog(self) -> None:
        self.spawn_editor.catalog = self.catalog
        self.builder.catalog = self.catalog
        self.results.model.set_names(
            {
                "robot_id": {r["id"]: r["label"] for r in self.catalog["robots"]},
                "world_id": {w["id"]: w["label"] for w in self.catalog["worlds"]},
            }
        )
        self.builder.robot_selector.set_catalog(self.catalog["robots"], self.catalog["parameters"])
        self.builder.world_selector.set_catalog(self.catalog["worlds"])
        self.result_robot.clear()
        self.result_world.clear()
        self.result_robot.addItem("All robots", None)
        self.result_world.addItem("All worlds", None)
        for combo in (
            self.profile_robot,
            self.spawn_editor.spawn_robot,
            self.spawn_editor.spawn_world,
        ):
            combo.blockSignals(True)
            combo.clear()
        for robot in self.catalog["robots"]:
            identity, label = robot["id"], robot["label"]
            self.profile_robot.addItem(label, identity)
            self.result_robot.addItem(label, identity)
            self.spawn_editor.spawn_robot.addItem(label, identity)
        for world in self.catalog["worlds"]:
            self.spawn_editor.spawn_world.addItem(world["label"], world["id"])
            self.result_world.addItem(world["label"], world["id"])
        for combo in (
            self.profile_robot,
            self.spawn_editor.spawn_robot,
            self.spawn_editor.spawn_world,
        ):
            combo.blockSignals(False)
        self.refresh_libraries()
        self.spawn_editor.load_world_map()
        self.builder.update_summary()

    def refresh_libraries(self) -> None:
        self.refresh_profile_list()
        self.spawn_editor.spawns.refresh()
        for robot in self.catalog.get("robots", []):
            owner = robot["id"]
            self.transport.submit(
                "library-" + owner,
                "profiles.list",
                {"robot_id": owner, "limit": 250, "summary": True},
            )
        for world in self.catalog.get("worlds", []):
            owner = world["id"]
            self.transport.submit(
                "library-" + owner,
                "spawns.list",
                {"world_id": owner, "limit": 250},
            )

    def fill_revisions(self, tag: str, result: Any) -> None:
        owner = tag[8:]
        if any(robot["id"] == owner for robot in self.catalog.get("robots", [])):
            self.builder.robot_selector.set_profiles(
                owner, result["rows"], result["next_offset"] is not None
            )
        elif any(world["id"] == owner for world in self.catalog.get("worlds", [])):
            self.builder.world_selector.set_spawns(
                owner, result["rows"], result["next_offset"] is not None
            )
        for options, combo in (
            (self.builder.robot_selector.profile_options(), self.result_profile),
            (self.builder.world_selector.spawn_options(), self.result_spawn),
        ):
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All profiles" if combo is self.result_profile else "All spawns", None)
            for owner_id, owner_label, reference, profile_label in options:
                identity = (
                    reference["id"]
                    if isinstance(reference, dict)
                    else "default-" + owner_id
                    if reference == "Default"
                    else reference
                )
                combo.addItem(owner_label + " · " + profile_label, identity)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
            combo.blockSignals(False)

    def select_campaign(self, row: dict[str, Any]) -> None:
        if self.campaign_id == row["id"]:
            return
        self.campaign_id = row["id"]
        self.campaign_overview.reset(row.get("name", "Selected campaign"))
        self.results_context.setText(row.get("name", "Selected campaign") + " · recorded outcomes")
        self.result_selection = None
        self.result_detail.clear()
        self.run_map.clear()
        self.report_browser.set_campaign(self.campaign_id)
        self.filter_results()

    def filter_results(self) -> None:
        self.result_selection = None
        self.result_row: dict[str, Any] = {}
        self.result_detail.clear()
        self.run_map.clear()
        filters: dict[str, Any] = {"campaign_id": self.campaign_id} if self.campaign_id else {}
        for key, value in (
            ("robot_id", self.result_robot.currentData()),
            ("world_id", self.result_world.currentData()),
            ("profile_id", self.result_profile.currentData()),
            ("spawn_id", self.result_spawn.currentData()),
        ):
            if value:
                filters[key] = value
        if self.result_outcome.currentIndex() > 0:
            filters["outcome"] = self.result_outcome.currentData()
        self.results.table = "experiments" if self.result_kind.currentIndex() == 1 else "attempts"
        self.results.filters, self.results.offset = filters, 0
        self.results.model.set_columns(self.result_columns(self.results.table))
        self.results.model.replace([])
        self.results.refresh()

    def update_delete_button(self, *args: Any) -> None:
        count = len(self.campaigns.chosen)
        self.delete_campaigns_button.setText(f"Delete {count}…" if count else "Delete…")
        self.delete_campaigns_button.setEnabled(bool(count) and not self.deletion_pending)

    def request_campaign_deletion(self) -> None:
        identities = list(self.campaigns.chosen)
        if not identities or self.deletion_pending:
            return
        if len(identities) > CAMPAIGN_DELETION_LIMIT:
            self.statusBar().showMessage(
                f"Select at most {CAMPAIGN_DELETION_LIMIT} campaigns to delete at once", 8000
            )
            return
        self.deletion_pending = True
        self.update_delete_button()
        self.transport.submit(
            "campaign-delete-preview", "campaign.delete_preview", {"campaign_ids": identities}
        )

    def show_deletions(self, deletions: dict[str, Any] | None) -> None:
        """State file removal of deleted campaigns; older engines send nothing."""
        if not deletions or not (deletions["removing"] or deletions["failed"]):
            return
        parts = []
        if deletions["removing"]:
            parts.append(f"Removing the files of {deletions['removing']} deleted campaign(s)…")
        if deletions["failed"]:
            parts.append(
                "File removal failed for "
                + "; ".join(f"{row['name']} ({row['detail']})" for row in deletions["failed"][:3])
                + ". The engine retries at its next start."
            )
        self.summary.setText(" ".join(parts))
        self.summary.show()

    def clear_campaign(self) -> None:
        self.campaign_id = None
        self.campaign_overview.reset("No campaign selected")
        self.results_context.setText("No campaign selected")
        self.result_selection = None
        self.result_detail.clear()
        self.run_map.clear()
        self.report_browser.set_campaign("")
        self.filter_results()

    def filter_campaigns(self, checked: bool) -> None:
        self.campaigns.clear_choice()
        self.settings.setValue("hide_finished_campaigns", checked)
        self.campaigns.filters = {"hide_finished": True} if checked else {}
        self.campaigns.reset_page()

    def control_campaign_row(self, identity: str, operation: str) -> None:
        if operation == "stop" and not self.confirm_discard(
            "Stop campaign",
            "Stop every active worker and leave remaining experiments Not done? Active partial "
            "measurements will be discarded. Completed results remain available.",
        ):
            self.campaigns.finish_control(identity)
            return
        if operation == "restart":
            name = next(
                (row["name"] for _, row in self.campaigns.model.records() if row["id"] == identity),
                "this campaign",
            )
            answer = QtWidgets.QMessageBox.question(
                self,
                "Restart campaign",
                f"Run a new copy of “{name}” with the same accepted settings? This campaign and its "
                "results stay unchanged. The copy starts now when no other campaign is running; "
                "otherwise it runs next in the queue.",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self.campaigns.finish_control(identity)
                return
            self.transport.submit(
                "campaign-restart:" + identity, "campaign.restart_copy", {"campaign_id": identity}
            )
            return
        self.transport.submit(
            "campaign-control:" + identity, "campaign." + operation, {"campaign_id": identity}
        )

    def refresh_profile_list(self, *args: Any) -> None:
        robot = self.profile_robot.currentData()
        if robot:
            self.profile_row = None
            self.profiles.filters = {
                "robot_id": robot,
                "search": self.profile_search.text().strip(),
            }
            self.profiles.offset = 0
            self.profiles.refresh()

    def open_profile(self, operation: str) -> None:
        if operation == "new":
            self.transport.submit(
                "editor-default",
                "parameters.resolve",
                {"robot_id": self.profile_robot.currentData()},
            )
        elif self.profile_row:
            self.transport.submit(
                "profile-open:" + operation, "profiles.get", {"revision_id": self.profile_row["id"]}
            )

    def show_editor(self, profile: dict[str, Any], clone: bool = False) -> ProfileEditor:
        editor = ProfileEditor(self.transport, self.catalog, profile, clone)
        editor.setParent(self, editor.windowFlags())
        editor.saved.connect(self.refresh_libraries)
        editor.show()
        return editor

    def delete_profiles(self) -> None:
        records = self.profiles.selected_records()
        if records is None:
            self.statusBar().showMessage("Scroll so every selected profile is loaded", 8000)
            return
        if not records:
            self.statusBar().showMessage("Select profiles to delete", 8000)
            return
        # One entry per profile (revision family); the last selected revision names it.
        families = {record["family_id"]: record for record in records}
        if len(families) > PROFILE_DELETION_LIMIT:
            self.statusBar().showMessage(
                f"Select at most {PROFILE_DELETION_LIMIT} profiles to delete at once", 8000
            )
            return
        robots = {robot["id"]: robot["label"] for robot in self.catalog.get("robots", [])}
        if self.confirm_discard(
            "Delete profiles", profile_deletion_message(list(families.values()), robots)
        ):
            self.transport.submit(
                "profile-delete", "profiles.delete", {"family_ids": list(families)}
            )

    def select_profile(self, row: dict[str, Any]) -> None:
        self.profile_row = row
        self.profile_details.set_record(row)
        self.profile_detail_tag = "profile-detail-" + uuid.uuid4().hex
        self.transport.submit(self.profile_detail_tag, "profiles.get", {"revision_id": row["id"]})

    def edit_builder_profile(self, robot: str, reference: Any) -> None:
        self.builder_profile_context = robot, reference
        self.builder_profile_tag = "builder-profile:" + uuid.uuid4().hex
        if isinstance(reference, dict):
            self.transport.submit(self.builder_profile_tag, "parameters.resolve", reference)
        elif reference == "Default":
            self.transport.submit(
                self.builder_profile_tag, "parameters.resolve", {"robot_id": robot}
            )
        else:
            self.transport.submit(
                self.builder_profile_tag, "profiles.get", {"revision_id": reference}
            )

    def accept_builder_profile(self, robot: str, reference: Any, profile: dict[str, Any]) -> None:
        self.builder.robot_selector.replace_profile(robot, reference, profile)
        self.refresh_libraries()

    def follow_single_worker(self) -> None:
        active = self.status.get("active", [])
        if len(active) == 1 and self.selected_attempt != active[0]["attempt"]["id"]:
            self.choose_worker(active[0]["attempt"]["id"], focus_map=False)

    def choose_worker(self, identity: str, *, focus_map: bool = True) -> None:
        active = next(
            (r for r in self.status.get("active", []) if r["attempt"]["id"] == identity), None
        )
        if not active:
            return
        self.selected_attempt = identity
        self.workers.select(identity)
        self.update_stop_button()
        experiment = active["experiment"]
        self.focus_title.setText(
            f"{humanize(experiment['world_id'])} · {experiment['profile_name']} · {experiment['spawn_name']} · repetition {experiment['repetition'] + 1}"
        )
        self.run_metrics.update_metrics(active["metrics"])
        self.log_viewer.select_attempt(identity, active["attempt"]["directory"])
        if focus_map:
            self.focus_tabs.setCurrentIndex(0)
        self.inspect_selected()

    def selected_run(self) -> dict[str, Any] | None:
        return next(
            (
                r
                for r in self.status.get("active", [])
                if r["attempt"]["id"] == self.selected_attempt
            ),
            None,
        )

    def update_stop_button(self) -> None:
        active = self.selected_run()
        self.stop_run_button.setEnabled(
            active is not None
            and active["attempt"]["phase"] in ("STARTING", "READY", "RUNNING")
            and not active.get("stop_requested")
            and not active["attempt"].get("discard_intent")
            and active["attempt"]["id"] not in self.stop_pending
        )

    def stop_run_text(self, active: dict[str, Any]) -> str:
        experiment, attempt = active["experiment"], active["attempt"]
        robot = next(
            (
                r["label"]
                for r in self.catalog.get("robots", [])
                if r["id"] == experiment["robot_id"]
            ),
            experiment["robot_id"],
        )
        return (
            f"{robot} · {humanize(experiment['world_id'])} · {experiment['profile_name']} · "
            f"{experiment['spawn_name']} · repetition {experiment['repetition'] + 1} · "
            f"attempt {attempt['number']}\n\n"
            "The run ends as Stopped. Its recorded data is kept for inspection, but campaign "
            "reports and statistics exclude it, and it is not run again. The campaign continues "
            "with the next experiment. If the run ends on its own before the stop takes effect, "
            "its own outcome is kept."
        )

    def stop_selected_run(self) -> None:
        active = self.selected_run()
        if active is None:
            return
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Warning,
            "Stop this run?",
            self.stop_run_text(active),
            QtWidgets.QMessageBox.StandardButton.NoButton,
            self,
        )
        stop = box.addButton("Stop run", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel))
        box.exec()
        if box.clickedButton() is not stop:
            return
        identity = active["attempt"]["id"]
        self.stop_pending.add(identity)
        self.update_stop_button()
        # A fixed request ID makes a repeated click return the stored answer.
        self.transport.submit(
            "stop-run:" + identity,
            "attempt.stop",
            {"attempt_id": identity},
            request_id="attempt-stop:" + identity,
        )

    def confirm_discard(self, title: str, explanation: str) -> bool:
        return (
            QtWidgets.QMessageBox.warning(
                self,
                title,
                explanation,
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            == QtWidgets.QMessageBox.StandardButton.Yes
        )

    def inspect_selected(self) -> None:
        if self.selected_attempt and self.monitor_id != self.selected_attempt:
            active = next(
                (
                    row
                    for row in self.status.get("active", [])
                    if row["attempt"]["id"] == self.selected_attempt
                ),
                None,
            )
            if active is None:
                return
            robot = next(
                (
                    row
                    for row in self.catalog.get("robots", [])
                    if row["id"] == active["experiment"]["robot_id"]
                ),
                None,
            )
            if robot is None:
                return
            # A request still in flight for the previous attempt clears monitor_pending on arrival
            # and its reply is ignored, so at most one request is outstanding.
            self.monitor_id = self.selected_attempt
            self.monitor_cursor = PathCursor()
            self.layer_revisions = {}
            self.reference_mismatch = False
            self.live_map.clear_view()
            self.live_map.set_footprint(robot["footprint"])
            world = active["experiment"]["world_id"]
            if world in self.reference_maps:
                self.live_map.set_reference(self.reference_maps[world])
            else:
                self.transport.submit("monitor-map:" + world, "spawns.map", {"world_id": world})
            self.monitor_status.setText("Waiting for the worker")
            self.monitor_timer.start()

    def close_monitor(self) -> None:
        self.monitor_id = None
        self.monitor_timer.stop()

    def poll_monitor(self) -> None:
        if not self.monitor_id or self.monitor_pending or self.current_page() != "monitor":
            return
        layers = [
            name for name in MONITOR_LAYERS if not (name == "explored" and self.reference_mismatch)
        ]
        self.monitor_pending = True
        self.monitor_requested = self.monitor_id
        self.transport.submit(
            "monitor",
            "monitor.get",
            {
                "attempt_id": self.monitor_id,
                "after_sequence": self.monitor_cursor.applied,
                "layers": layers,
                "revisions": {
                    name: self.layer_revisions[name]
                    for name in layers
                    if name in self.layer_revisions
                },
            },
        )

    def draw_layer(self, name: str, data: dict[str, Any] | None) -> None:
        if name == "explored":
            self.live_map.set_explored(data)
        elif name == "navigation":
            self.live_map.set_navigation(data)

    def apply_monitor(self, result: dict[str, Any]) -> None:
        if "history" not in result:
            self.monitor_status.setText(result.get("reason") or "Waiting for the worker")
            return
        try:
            status, reason, pose = result["status"], result["reason"], result["pose"]
            reference, layers, breaks = result["reference"], result["layers"], result["breaks"]
            tail, last = result["tail"], result["last_sequence"]
            collisions, outside = result["collision_samples"], result["outside_samples"]
            values = (
                None
                if pose is None
                else (
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["yaw"]),
                    float(pose["age_wall_seconds"]),
                )
            )
            reset, vertices = accept_page(self.monitor_cursor, result)
        except (KeyError, TypeError, ValueError) as exc:
            self.monitor_status.setText(f"Monitor reply rejected: {exc}")
            return
        if reset:
            self.live_map.clear_path()
        self.live_map.append_vertices(vertices)
        self.live_map.set_tail(tail if last == self.monitor_cursor.applied else None)
        if status == "available" and values is not None:
            self.live_map.set_pose(values[0], values[1], values[2])
        else:
            self.live_map.hide_pose()
        self.reference_mismatch = bool(
            self.live_map.reference_hash
            and isinstance(reference, dict)
            and reference.get("metadata_sha256") != self.live_map.reference_hash
        )
        if self.reference_mismatch:
            self.draw_layer("explored", None)
            self.layer_revisions.pop("explored", None)
        notes = []
        for name, layer in layers.items():
            title = LAYER_TITLES.get(name)
            if title is None or (name == "explored" and self.reference_mismatch):
                continue
            try:
                if layer["revision"] == -1:
                    self.draw_layer(name, None)
                    self.layer_revisions.pop(name, None)
                    if layer.get("reason"):
                        notes.append(f"{title} unavailable: {layer['reason']}")
                elif layer["data"] is not None:
                    self.draw_layer(name, layer["data"])
                    self.layer_revisions[name] = layer["revision"]
            except (KeyError, TypeError, ValueError) as exc:
                notes.append(f"{title} rejected: {exc}")
        if status == "available" and values is not None:
            text = (
                f"Gazebo ground truth · {self.monitor_cursor.applied:,} path vertices · "
                f"pose {values[3]:.1f} s old"
            )
        elif status == "stale":
            text = f"Ground-truth pose stale: {reason}"
        else:
            text = f"Ground-truth pose unavailable: {reason}"
        counted = {key: count for key, count in breaks.items() if count > 0}
        if counted:
            listed = ", ".join(f"{key} {count}" for key, count in counted.items())
            text += f" · {sum(counted.values())} breaks ({listed})"
        dropped = self.monitor_cursor.dropped + self.live_map.view_dropped
        if dropped > 0:
            text += f" · {dropped:,} older vertices dropped"
        if collisions > 0:
            text += f" · {collisions} samples in obstacle cells"
        if outside > 0:
            text += f" · {outside} samples outside the reference region"
        if self.reference_mismatch:
            text += " · World map differs from this attempt's reference; explored cells hidden"
        self.monitor_status.setText(" · ".join([text, *notes]))

    def select_result(self, row: dict[str, Any]) -> None:
        self.result_selection = row["id"]
        self.result_row = row
        identity = row["id"] if self.results.table == "attempts" else row.get("final_attempt_id")
        self.run_map.show_attempt(identity, "No finished attempt for this experiment")
        self.result_detail.set_record(
            {
                key.removesuffix("_id"): self.results.model.names.get(key, {}).get(value, value)
                for key, value in row.items()
                if key
                in {
                    "robot_id",
                    "world_id",
                    "outcome",
                    "phase",
                    "state",
                    "profile_name",
                    "spawn_name",
                    "repetition",
                }
            }
        )
        self.transport.submit(
            "result-detail", "results.get", {"table": self.results.table, "id": row["id"]}
        )
        if identity:
            self.log_viewer.select_attempt(identity, row.get("directory"))

    def import_profile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import suite profile", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.transport.submit("profile-read", "local.read", {"path": path})

    def export_profile(self) -> None:
        if self.profile_row:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export suite profile", "suite-profile.yaml", "YAML (*.yaml)"
            )
            if path:
                self.export_path = path
                self.transport.submit(
                    "profile-export",
                    "profiles.export",
                    {"revision_id": self.profile_row["id"], "format": "suite"},
                )

    def read_import(self, tag: str, source: str) -> None:
        if tag == "profile-read":
            self.transport.submit("profile-import", "profiles.import", {"yaml_text": source})

    def apply_style(self) -> None:
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.styleHints().setColorScheme(QtCore.Qt.ColorScheme.Light)
            application.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
            palette = QtGui.QPalette()
            for role, color in (
                (QtGui.QPalette.ColorRole.Window, "surface.base"),
                (QtGui.QPalette.ColorRole.Base, "surface.raised"),
                (QtGui.QPalette.ColorRole.AlternateBase, "surface.sunken"),
                (QtGui.QPalette.ColorRole.WindowText, "text.body"),
                (QtGui.QPalette.ColorRole.Text, "text.body"),
                (QtGui.QPalette.ColorRole.Button, "surface.raised"),
                (QtGui.QPalette.ColorRole.ButtonText, "text.heading"),
                (QtGui.QPalette.ColorRole.Highlight, "surface.selected"),
                (QtGui.QPalette.ColorRole.HighlightedText, "text.heading"),
                (QtGui.QPalette.ColorRole.Link, "text.accent"),
                (QtGui.QPalette.ColorRole.LinkVisited, "text.accent"),
                (QtGui.QPalette.ColorRole.ToolTipBase, "surface.raised"),
                (QtGui.QPalette.ColorRole.ToolTipText, "text.body"),
                (QtGui.QPalette.ColorRole.PlaceholderText, "text.muted"),
                (QtGui.QPalette.ColorRole.Light, "surface.raised"),
                (QtGui.QPalette.ColorRole.Midlight, "surface.base"),
                (QtGui.QPalette.ColorRole.Mid, "border.subtle"),
                (QtGui.QPalette.ColorRole.Dark, "border.control"),
                (QtGui.QPalette.ColorRole.Shadow, "text.strong"),
                (QtGui.QPalette.ColorRole.BrightText, "text.on-accent"),
                (QtGui.QPalette.ColorRole.Accent, "accent.default"),
            ):
                palette.setColor(role, QtGui.QColor(COLORS[color]))
            for role in (
                QtGui.QPalette.ColorRole.WindowText,
                QtGui.QPalette.ColorRole.Text,
                QtGui.QPalette.ColorRole.ButtonText,
                QtGui.QPalette.ColorRole.PlaceholderText,
            ):
                palette.setColor(
                    QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor(COLORS["text.disabled"])
                )
            application.setPalette(palette)
            application.setStyleSheet(stylesheet())
        for i in range(self.navigation.count()):
            item = self.navigation.item(i)
            item.setIcon(icon(item.data(QtCore.Qt.ItemDataRole.UserRole)))
            item.setSizeHint(
                QtCore.QSize(
                    self.navigation.fontMetrics().horizontalAdvance(item.text())
                    + self.navigation.iconSize().width()
                    + 64,
                    44,
                )
            )

    def closeEvent(self, event: Any) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        for split in self.findChildren(QtWidgets.QSplitter):
            if split.objectName():
                self.settings.setValue("split/" + split.objectName(), split.saveState())
        self.settings.sync()
        self.poll.stop()
        self.monitor_timer.stop()
        self.run_map.close_timers()
        if self.transport.close():
            event.accept()
        else:
            event.ignore()
            QtCore.QTimer.singleShot(200, self.close)


def run(socket: Path) -> int:
    application = QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("Exploration Benchmark Suite")
    application.setOrganizationName("Mert Güler")
    application.setStyle("Fusion")
    window = Window(socket)
    window.show()
    return application.exec()
