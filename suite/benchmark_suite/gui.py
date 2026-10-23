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

from PySide6 import QtCore, QtWidgets

from .contracts import CAMPAIGN_DELETION_LIMIT
from .gui_support import (
    PagedTable,
    TextSelection,
    Transport,
    stale_engine_message,
)
from .gui_spawns import WorldsPage
from .gui_builder import CampaignBuilder
from .gui_campaigns import (
    ATTENTION_REJECTED,
    CampaignCard,
    CampaignTable,
    LiveCard,
    OverviewRow,
    TableCard,
    confirm_deletion,
    confirm_restart,
    confirm_stop,
)
from .gui_controls import CountChip, Switch
from .gui_profile_editor import Mode, ProfileEditor
from .gui_profiles import ProfilesPage
from .gui_menus import install_tooltips
from .gui_theme import ThemeManager
from .gui_topbar import TopBar
from .gui_widgets import (
    PAGE_TITLE,
    Button,
    DotCanvas,
    FlexRow,
    FocusRing,
    ModalHost,
    PopupHost,
    Rule,
    TextLine,
    ToastArea,
    ToastKind,
)
from .gui_monitor import MonitorPage
from .gui_results import ResultsPage


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
        self.themes = ThemeManager(self.settings, self)
        # Before any page is built: until the event loop runs, Qt does not pass a new application font
        # (or palette) on to existing widgets, so inputs would keep the platform's default font.
        self.themes.apply()
        self.poll_errors: dict[str, str] = {}
        install_tooltips(self.themes, self)
        self.transport = Transport(socket)
        self.catalog: dict[str, Any] = {}
        self.catalog_pending = True
        self.status: dict[str, Any] = {}
        self.campaign_id: str | None = None
        self.status_pending = False
        self.campaign_status_pending = False
        self._campaign_page = "campaigns"
        self.page_widgets: dict[str, QtWidgets.QWidget] = {}
        self.pages = QtWidgets.QStackedWidget()
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.topbar = TopBar(self.themes)
        self.topbar.navigate.connect(self.navigate)
        self.topbar.theme_requested.connect(self.themes.set)
        root.addWidget(self.topbar)
        root.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.tables: list[PagedTable] = []
        self.make_dashboard()
        self.make_monitor()
        self.make_profiles()
        self.make_spawns()
        self.make_results()
        self.make_builder()
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
        # Window-wide layers, bottom to top: dialogs, popups, toasts, keyboard focus.
        self.modal = ModalHost(self.themes, self)
        self.popups = PopupHost(self.themes, self)
        self.toasts = ToastArea(self.themes, self, top=self.topbar.height() + 16)
        self.focus_ring = FocusRing(self.themes, self)
        self.go("campaigns")
        geometry = self.settings.value("geometry")
        if isinstance(geometry, QtCore.QByteArray):
            self.restoreGeometry(geometry)
        self.transport.received.connect(self.received)
        self.transport.failed.connect(self.failed)
        self.transport.start()
        self.transport.submit("catalog", "catalog.get")
        self.poll = QtCore.QTimer(self)
        self.poll.setInterval(1000)
        self.poll.timeout.connect(self.refresh_status)
        self.poll.start()
        self.refresh_status()

    def page(self, name: str, key: str) -> QtWidgets.QVBoxLayout:
        widget = DotCanvas(self.themes)
        widget.setObjectName("page-" + key)
        self.pages.addWidget(widget)
        self.page_widgets[key] = widget
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)
        widget.setAccessibleName(name)
        return layout

    def go(self, key: str) -> None:
        if key in ("campaigns", "builder"):
            self._campaign_page = key
        self.pages.setCurrentWidget(self.page_widgets[key])
        self.topbar.nav.set_current("campaigns" if key == "builder" else key)
        for table in self.tables:
            if table.isVisible():
                table.refresh()
        if key == "monitor":
            self.monitor.activate()
        else:
            self.monitor.deactivate()

    def current_page(self) -> str:
        return self.pages.currentWidget().objectName().removeprefix("page-")

    def open_builder(self) -> None:
        # The engine or library may have changed since an earlier validation.
        self.builder.invalidate()
        self.go("builder")

    def make_dashboard(self) -> None:
        layout = self.page("Campaigns", "campaigns")
        # Page head (``.page-head``): title, total, rule, Hide finished; Delete and New campaign at the right.
        head = FlexRow(gap=12)
        head.setFixedHeight(36)
        head.add(TextLine(self.themes, "Campaigns", PAGE_TITLE, "text.ink"))
        self.campaign_total = CountChip(self.themes, None)
        self.campaign_total.setAccessibleName("Campaign count")
        head.add(self.campaign_total)
        rule = Rule(self.themes, vertical=True)
        rule.setFixedHeight(28)  # ``.vr``: stretched in the 36 px head with 4 px margins
        head.add(rule, width=1, margin=4)
        self.hide_finished = Switch(self.themes, "Hide finished", label_token="text.secondary")
        self.hide_finished.setChecked(
            bool(self.settings.value("hide_finished_campaigns", False, type=bool))
        )
        head.add(self.hide_finished)
        head.add_stretch()
        self.deletion_pending = False
        self.deletion_message = ""
        self.delete_campaigns_button = Button(self.themes, "Delete", icon="trash")
        self.delete_campaigns_button.clicked.connect(self.request_campaign_deletion)
        head.add(self.delete_campaigns_button, gap=8)
        new_campaign = Button(self.themes, "New campaign", icon="plus", variant="primary")
        new_campaign.clicked.connect(self.open_builder)
        head.add(new_campaign, gap=8)
        layout.addWidget(head)
        self.campaigns = CampaignTable(self.transport, self.themes)
        self.campaign_card = CampaignCard(self.themes, self.campaigns.pending)
        self.campaign_card.action_requested.connect(self.control_selected_campaign)
        self.campaigns.pending_changed.connect(self.campaign_card.refresh_pending)
        self.live_card = LiveCard(self.themes, self.transport)
        self.live_card.open_monitor.connect(lambda: self.navigate("monitor"))
        self.live_card.open_worker.connect(self.open_worker)
        layout.addWidget(OverviewRow(self.campaign_card, self.live_card))
        self.campaigns.selected.connect(self.select_campaign)
        self.campaigns.action_requested.connect(self.control_campaign_row)
        self.campaigns.choice_changed.connect(self.update_delete_button)
        if self.hide_finished.isChecked():
            self.campaigns.filters = {"hide_finished": True}
        self.hide_finished.toggled.connect(self.filter_campaigns)
        self.tables.append(self.campaigns)
        layout.addWidget(TableCard(self.themes, self.campaigns), 1)
        self.update_delete_button()

    def make_builder(self) -> None:
        layout = self.page("New campaign", "builder")
        self.builder = CampaignBuilder(self.transport, self.themes)
        self.builder.message.connect(self.builder_message)
        self.builder.return_requested.connect(lambda: self.go("campaigns"))
        self.builder.tuning_started.connect(self.open_tuning_run)
        self.builder.edit_profile.connect(self.edit_builder_profile)
        layout.addWidget(self.builder, 1)

    def builder_message(self, text: str, kind: str) -> None:
        self.notify(text, "bad" if kind == "bad" else "ok" if kind == "ok" else "neutral")

    def make_profiles(self) -> None:
        layout = self.page("Profiles", "profiles")
        self.profiles_page = ProfilesPage(self.themes, self.transport)
        self.profiles_page.open_requested.connect(self.open_profile)
        self.profiles_page.library_changed.connect(self.refresh_libraries)
        self.profiles_page.deleted.connect(self.forget_deleted_profiles)
        self.profiles_page.notify.connect(self.page_notice)
        layout.addWidget(self.profiles_page, 1)

    def make_spawns(self) -> None:
        layout = self.page("Worlds & Spawns", "worlds")
        self.worlds_page = WorldsPage(self.themes, self.transport)
        self.worlds_page.changed.connect(self.refresh_libraries)
        self.worlds_page.deleted.connect(self.forget_deleted_spawns)
        self.worlds_page.notify.connect(self.page_notice)
        layout.addWidget(self.worlds_page, 1)

    def make_monitor(self) -> None:
        layout = self.page("Monitor", "monitor")
        self.monitor = MonitorPage(self.themes, self.transport)
        self.monitor.open_campaign.connect(self.open_campaign)
        self.monitor.notify.connect(self.page_notice)
        layout.addWidget(self.monitor, 1)

    def page_notice(self, text: str, kind: str) -> None:
        """A page's message as a toast (``ok``, ``warn``, ``bad`` or ``neutral``)."""
        tone: ToastKind = (
            "ok"
            if kind == "ok"
            else "warn"
            if kind == "warn"
            else "bad"
            if kind == "bad"
            else "neutral"
        )
        self.notify(text, tone)

    def open_worker(self, attempt_id: str) -> None:
        """A Live on the map tile: the Monitor with that run selected."""
        self.navigate("monitor")
        self.monitor.choose_worker(attempt_id)

    def open_replay(self, attempt_id: str) -> None:
        """Results' Replay: the Monitor replaying that finished run."""
        self.monitor.open_replay(attempt_id)
        self.navigate("monitor")

    def open_campaign(self, campaign_id: str, name: str) -> None:
        """The Monitor's Open campaign link: that campaign on Campaigns."""
        self.select_campaign({"id": campaign_id, "name": name or "Selected campaign"})
        self.navigate("campaigns")

    def make_results(self) -> None:
        layout = self.page("Results", "results")
        layout.setContentsMargins(0, 0, 0, 0)  # the run view scrolls over the whole page
        self.results_page = ResultsPage(self.themes, self.transport)
        self.results_page.attempt_selected.connect(self.monitor.log_view.select_attempt)
        self.results_page.replay_requested.connect(self.open_replay)
        self.results = self.results_page.results
        self.tables.append(self.results)
        layout.addWidget(self.results_page, 1)

    def refresh_visible_tables(self) -> None:
        for table in self.tables:
            if table.isVisible():
                table.refresh()

    def _poll_failed(self, tag: str, error: str) -> bool:
        """Whether a failed periodic request should toast: once per new message, never while the engine
        is unavailable (the engine state and its toast say so)."""
        if self.topbar.engine.state == "unavailable" or self.poll_errors.get(tag) == error:
            return False
        self.poll_errors[tag] = error
        return True

    def open_tuning_run(self) -> None:
        """A started tuning run: Monitor follows it as the sole active run once the status shows it."""
        self.navigate("monitor")
        self.refresh_status()

    def navigate(self, key: str) -> None:
        self.go(self._campaign_page if key == "campaigns" else key)

    def notify(self, text: str, kind: ToastKind = "neutral") -> None:
        """A toast (the design has no status bar); every toast fades after 3 s and is dismissible."""
        self.toasts.show_message(text, kind)

    def refresh_status(self) -> None:
        if not self.status_pending:
            self.status_pending = True
            self.transport.submit("status", "engine.status")
        if self.campaign_id and not self.campaign_status_pending:
            self.campaign_status_pending = True
            self.transport.submit(
                "campaign-status", "campaign.status", {"campaign_id": self.campaign_id}
            )

    def received(self, tag: str, result: Any) -> None:
        if tag.startswith("tile-") and self.live_card.received(tag, result):
            return
        if self.monitor.received(tag, result):
            if tag.startswith("stop-run:"):
                self.refresh_status()
            return
        if (
            self.profiles_page.received(tag, result)
            or self.worlds_page.received(tag, result)
            or self.results_page.received(tag, result)
        ):
            return
        if tag == "catalog":
            self.catalog_pending = False
            self.catalog = result
            self.fill_catalog()
        elif tag == "status":
            self.status_pending, self.status = False, result
            self.topbar.engine.set_state("connected")
            self.topbar.nav.set_live(len(result["active"]))
            self.show_deletions(result.get("deletions"))
            self.campaign_total.set_count(result["campaigns"]["total"])
            self.live_card.set_runs(result["active"], self.catalog)
            if not self.catalog and not self.catalog_pending:
                self.catalog_pending = True
                self.transport.submit("catalog", "catalog.get")
            self.monitor.update_status(result, self.catalog)
        elif tag == "submit":
            self.select_campaign({"id": result["campaign_id"], "name": self.builder.name.text()})
            self.builder.show_step(0)
            self.campaigns.refresh()
            self.go("campaigns")
        elif tag.startswith("library-"):
            self.fill_library(tag, result)
        elif tag == "editor-default":
            self.show_editor(result, "new")
        elif tag == "profile-open":
            self.show_editor(result, "edit")
        elif tag.startswith("builder-profile:") and tag == getattr(self, "builder_profile_tag", ""):
            editor = self.show_editor(result, "edit" if result.get("family_id") else "new")
            owner, reference = self.builder_profile_context
            editor.accepted.connect(
                lambda profile, r=owner, old=reference: self.accept_builder_profile(r, old, profile)
            )
        elif tag.startswith("campaign-control:"):
            self.campaigns.finish_control(
                tag.removeprefix("campaign-control:"), result.get("state", "PAUSING")
            )
        elif tag == "campaign-delete-preview":
            # Ticks on campaigns that are gone could not be cleared from the table any more.
            self.campaigns.forget(
                [
                    entry["campaign_id"]
                    for entry in result["campaigns"]
                    if entry.get("reason") in ("not found", "already deleted")
                ]
            )
            # The dialog is modal (the page under it takes no input), so Delete shows as it was (02).
            self.deletion_pending = False
            self.update_delete_button()
            identities = confirm_deletion(self.modal, self.themes, result["campaigns"])
            if identities:
                self.deletion_pending = True
                self.update_delete_button()
                self.transport.submit(
                    "campaign-delete", "campaign.delete", {"campaign_ids": identities}
                )
        elif tag == "campaign-delete":
            self.deletion_pending = False
            gone = result["deleted"] + result["already_deleted"]
            self.campaigns.forget(gone)
            if self.campaign_id in gone:
                self.clear_campaign()
            self.campaigns.refresh()
            count = len(result["deleted"])
            self.notify(
                f"Deleted {count} campaign{'s' if count != 1 else ''}; removing files…", "ok"
            )
            self.update_delete_button()
        elif tag.startswith("campaign-restart:"):
            self.campaigns.finish_control(tag.removeprefix("campaign-restart:"))
            self.campaigns.view.clearSelection()
            self.campaigns.view.setCurrentIndex(QtCore.QModelIndex())
            self.select_campaign({"id": result["campaign_id"], "name": result["name"]})
            self.notify(
                f"Started “{result['name']}”"
                if result["state"] == "RUNNING"
                else f"Queued “{result['name']}”; it runs after the current campaign",
                "ok",
            )
            self.campaigns.refresh()
        elif tag == "campaign-status":
            self.campaign_status_pending = False
            self.poll_errors.pop(tag, None)
            campaign, counters = result["campaign"], result["counters"]
            if campaign["id"] != self.campaign_id:
                return
            self.campaign_card.update_campaign(campaign, counters, self.catalog)

    def failed(self, tag: str, error: str) -> None:
        if tag.startswith("tile-") and self.live_card.failed(tag, error):
            return  # Live on the map keeps its last view; the status poll reports outages
        if self.monitor.failed(tag, error):
            return  # the Monitor reports its own errors
        if (
            self.profiles_page.failed(tag, error)
            or self.worlds_page.failed(tag, error)
            or self.results_page.failed(tag, error)
        ):
            return  # these pages show their own list, map, snapshot and save errors
        if tag.startswith("editor-") and ":" in tag:
            return  # an open profile editor shows its own errors
        if tag == self.campaigns.tag and ATTENTION_REJECTED in error:
            return  # an engine without the attention order: the table falls back silently (D6)
        if tag == "catalog":
            self.catalog_pending = False
        if tag == "status":
            self.status_pending = False
            if self.topbar.engine.state != "unavailable":
                self.notify("Engine connection unavailable: " + error, "bad")
            self.topbar.engine.set_state("unavailable", error)
            self.topbar.nav.set_live(0)
            self.campaign_total.set_count(None)
            return
        elif tag == "campaign-status":
            self.campaign_status_pending = False
            if "Campaign does not exist" in error:
                self.clear_campaign()
            if not self._poll_failed(tag, error):
                return
        elif tag == "profile-delete":
            stale = stale_engine_message(error, "Unsupported library operation", "profile deletion")
            if stale:
                error = stale
            else:
                self.profiles_page.refresh()
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
        self.notify(error, "bad")

    def fill_catalog(self) -> None:
        self.builder.catalog = self.catalog
        self.results_page.set_names(self.catalog)
        self.builder.robot_selector.set_catalog(self.catalog["robots"], self.catalog["parameters"])
        self.builder.world_selector.set_catalog(self.catalog["worlds"])
        self.profiles_page.set_catalog(self.catalog)
        self.worlds_page.set_catalog(self.catalog)
        self.refresh_libraries()
        self.builder.update_summary()

    def refresh_libraries(self) -> None:
        self.profiles_page.refresh()
        self.worlds_page.refresh()
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

    def fill_library(self, tag: str, result: Any) -> None:
        owner = tag[8:]
        if any(robot["id"] == owner for robot in self.catalog.get("robots", [])):
            self.builder.robot_selector.set_profiles(
                owner, result["rows"], result["next_offset"] is not None
            )
            self.profiles_page.set_library(owner, result["rows"], result["total"])
        elif any(world["id"] == owner for world in self.catalog.get("worlds", [])):
            self.builder.world_selector.set_spawns(
                owner, result["rows"], result["next_offset"] is not None
            )

        def identity(owner: str, reference: Any) -> Any:
            if isinstance(reference, dict):
                return reference["id"]
            return "default-" + owner if reference == "Default" else reference

        profiles, spawns = (
            [
                (f"{owner_label} · {label}", identity(owner_id, reference))
                for owner_id, owner_label, reference, label in options
            ]
            for options in (
                self.builder.robot_selector.profile_options(),
                self.builder.world_selector.spawn_options(),
            )
        )
        self.results_page.set_filter_options(profiles, spawns)

    def select_campaign(self, row: dict[str, Any]) -> None:
        if self.campaign_id == row["id"]:
            return
        self.campaign_id = row["id"]
        self.campaign_card.reset(row.get("name", "Selected campaign"))
        self.results_page.set_campaign(self.campaign_id)

    def update_delete_button(self, *args: Any) -> None:
        """Delete: disabled until rows are ticked; then "Delete N" in the danger style (01, 02)."""
        count = len(self.campaigns.chosen)
        button = self.delete_campaigns_button
        button.setText(f"Delete {count}" if count else "Delete")
        button.set_variant("danger" if count else "secondary")
        button.setEnabled(bool(count) and not self.deletion_pending)
        button.setToolTip("" if count else "Tick finished or stopped campaigns to delete them")
        button.setAccessibleName(
            f"Delete {count} ticked campaigns" if count else "Delete ticked campaigns"
        )
        parent = button.parentWidget()
        if isinstance(parent, FlexRow):
            parent.relayout()

    def request_campaign_deletion(self) -> None:
        identities = self.campaigns.chosen_in_order()
        if not identities or self.deletion_pending:
            return
        if len(identities) > CAMPAIGN_DELETION_LIMIT:
            self.notify(
                f"Select at most {CAMPAIGN_DELETION_LIMIT} campaigns to delete at once", "warn"
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
        message = " ".join(parts)
        if message != self.deletion_message:
            self.deletion_message = message
            self.notify(message, "bad" if deletions["failed"] else "neutral")

    def clear_campaign(self) -> None:
        self.campaign_id = None
        self.campaign_card.reset("No campaign selected")
        self.results_page.set_campaign(None)

    def filter_campaigns(self, checked: bool) -> None:
        self.campaigns.clear_choice()
        self.settings.setValue("hide_finished_campaigns", checked)
        self.campaigns.filters = {"hide_finished": True} if checked else {}
        self.campaigns.reset_page()

    def control_selected_campaign(self, identity: str, operation: str) -> None:
        """Start, Pause, Resume, Stop or Restart from the campaign card."""
        if self.campaigns.begin_control(identity):
            self.control_campaign_row(identity, operation)

    def control_campaign_row(self, identity: str, operation: str) -> None:
        name = next(
            (row["name"] for _, row in self.campaigns.model.records() if row["id"] == identity),
            self.campaign_card.facts.name if identity == self.campaign_id else "this campaign",
        )
        if operation == "stop" and not confirm_stop(self.modal, self.themes, name):
            self.campaigns.finish_control(identity)
            return
        if operation == "restart":
            if not confirm_restart(self.modal, self.themes, name):
                self.campaigns.finish_control(identity)
                return
            self.transport.submit(
                "campaign-restart:" + identity, "campaign.restart_copy", {"campaign_id": identity}
            )
            return
        self.transport.submit(
            "campaign-control:" + identity, "campaign." + operation, {"campaign_id": identity}
        )

    def open_profile(self, operation: str) -> None:
        self.profiles_page.open_request(operation)

    def show_editor(self, profile: dict[str, Any], mode: Mode = "edit") -> ProfileEditor:
        editor = ProfileEditor(
            self.themes,
            self.transport,
            self.catalog,
            profile,
            mode=mode,
        )
        editor.saved.connect(self.refresh_libraries)
        editor.notify.connect(self.page_notice)
        self.modal.open_card(editor)
        return editor

    def forget_deleted_profiles(self, result: dict[str, Any]) -> None:
        deleted: dict[str, set[str]] = {}
        for row in result["deleted"]:
            deleted.setdefault(row["owner"], set()).add(row["id"])
        for owner, identities in deleted.items():
            self.builder.robot_selector.forget_profiles(owner, identities)

    def forget_deleted_spawns(self, result: dict[str, Any]) -> None:
        deleted: dict[str, set[str]] = {}
        for row in result["deleted"]:
            deleted.setdefault(row["owner"], set()).add(row["id"])
        for world, identities in deleted.items():
            self.builder.world_selector.forget_spawns(world, identities)

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
            self.transport.submit(self.builder_profile_tag, "profiles.get", {"id": reference})

    def accept_builder_profile(self, robot: str, reference: Any, profile: dict[str, Any]) -> None:
        self.builder.robot_selector.replace_profile(robot, reference, profile)
        self.refresh_libraries()

    def closeEvent(self, event: Any) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()
        self.poll.stop()
        self.monitor.deactivate()
        self.monitor.viewers.stop_all()
        self.results_page.close_timers()
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
