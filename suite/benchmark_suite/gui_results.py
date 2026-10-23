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

"""Results by study (22–31; RESULTS_STUDY.md; RESULTS_RESEARCH RD-37 to RD-41).

The head selects a study (``studies.list``; "Unassigned · exploratory" holds the campaigns without one); the
tabs are Runs (``gui_runs``: every final run of the study, a row opens its run view), Study (the analysis
outputs) and Files (the export and run files). Export study and Run analysis sit in the head on the Study tab.
A campaign chosen on Campaigns selects its study and filters the runs to it. A run opened from the table shows
its run view (``gui_run_detail``) in place of the page until "Runs" goes back.
"""

from collections.abc import Mapping, Sequence
import math
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_controls import KeySelect, UnderlineTabs
from .gui_run_detail import RunView
from .gui_runs import OUTCOMES, Names, RunsCard, run_label
from .gui_analysis import Jobs, analysis_available
from .gui_files import FilesTab
from .gui_study import StudyTab, newest_export
from .gui_style import humanize
from .gui_support import POLL_MS, Transport
from .gui_theme import ThemeManager
from .gui_widgets import (
    BODY,
    PAGE_BOTTOM,
    PAGE_LEFT,
    PAGE_TITLE,
    PAGE_TOP,
    Button,
    FlexRow,
    Rule,
    TextLine,
    place,
)

UNASSIGNED = "unassigned"
RUNS, STUDY, FILES = 0, 1, 2


def study_label(row: Mapping[str, Any]) -> str:
    """ "Unassigned · exploratory", "S1-low-threshold · confirmatory" (both levels when campaigns differ)."""
    name = "Unassigned" if row["study_id"] == UNASSIGNED else str(row["study_id"])
    levels = row.get("evidence_levels") or ["exploratory"]
    return f"{name} · {' and '.join(str(level) for level in levels)}"


class ResultsPage(QtWidgets.QWidget):
    """Results: head (study select; Run analysis and Export study on the Study tab), Runs, Study and Files."""

    # The opened run's attempt and directory: the Monitor's Logs view reads it until a live run is chosen there.
    attempt_selected = QtCore.Signal(str, object)
    replay_requested = QtCore.Signal(str)  # attempt id

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport = themes, transport
        self.study_id: str | None = None
        self.studies: list[dict[str, Any]] = []
        self.campaigns: list[dict[str, Any]] = []
        self.detail: dict[str, Any] | None = None  # study.get of the selected study once ready
        self.counts_tag = ""  # the footer counts request in flight; earlier replies are dropped
        self.wanted_campaign: str | None = (
            None  # chosen on Campaigns, applied once its study is known
        )
        self.names = Names({}, {}, {}, {}, {})

        # The study page carries the page's paddings; the run view scrolls over the whole page area.
        self.study_page = QtWidgets.QWidget(self)
        self.head = FlexRow(self.study_page, gap=12)
        self.head.add(TextLine(themes, "Results", PAGE_TITLE, "text.ink"))
        rule = Rule(themes, vertical=True)
        rule.setFixedHeight(24)
        self.head.add(rule, width=1, margin=8)
        self.study = KeySelect(themes, "Study")
        self.study.addItem("No study yet", None)
        self.head.add(self.study, width=316)
        self.head.add_stretch()
        self.analyse = Button(themes, "Run analysis", icon="chart")
        self.export = Button(themes, "Export study", icon="download", variant="primary")
        self.head.add(self.analyse)
        self.head.add(self.export)
        self.tabs = UnderlineTabs(
            themes, [("Runs", "table"), ("Study", "chart"), ("Files", "folder")], self.study_page
        )
        self.stack = QtWidgets.QStackedWidget(self.study_page)
        self.runs = RunsCard(themes, transport, self.names)
        self.stack.addWidget(self.runs)
        self.study_view = StudyTab(themes, transport, self.name_of)
        self.study_view.opened.connect(self.open_run)
        self.stack.addWidget(self.study_view)
        self.files_view = FilesTab(themes, transport)
        self.stack.addWidget(self.files_view)
        self.results = self.runs.table
        self.tabs.changed.connect(self._tab)
        self.study.currentIndexChanged.connect(self._study_picked)
        self.runs.filters_changed.connect(self.filter_runs)
        self.results.loaded.connect(self._page_loaded)
        self.results.opened.connect(self.open_run)
        self.jobs = Jobs(themes, transport, self.window)
        self.study_view.jobs = self.jobs
        self.jobs.changed.connect(self._job_changed)
        self.jobs.exported.connect(lambda _: self.load_detail())
        self.jobs.analysed.connect(lambda _: self.study_view.reload_analysis())
        self.export.clicked.connect(self._export)
        self.analyse.clicked.connect(self._analyse)
        self.run_view = RunView(themes, transport, self.name_of, self)
        self.run_view.hide()
        self.run_view.back.connect(self.close_run)
        self.run_view.open_record.connect(self.open_run)
        self.run_view.replay_requested.connect(self.replay_requested.emit)
        self.run_view.files_requested.connect(self.show_run_files)
        self.sync = QtCore.QTimer(self)
        self.sync.setInterval(5000)
        self.sync.timeout.connect(self._sync)
        self.sync.start()
        # study.get answers "loading" until its read is built
        self.detail_timer = QtCore.QTimer(self)
        self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(POLL_MS)
        self.detail_timer.timeout.connect(self.load_detail)
        self._tab(RUNS)

    # studies and campaigns

    def load_studies(self) -> None:
        self.transport.submit("results-studies", "studies.list", {"offset": 0, "limit": 250})
        self.transport.submit(
            "results-campaigns",
            "results.query",
            {"table": "campaigns", "filters": {}, "offset": 0, "limit": 250, "include": ["study"]},
        )

    def _sync(self) -> None:
        if self.isVisible():
            self.load_studies()
            self.load_detail()

    def load_detail(self) -> None:
        if self.study_id is not None:
            self.transport.submit(
                "results-study:" + self.study_id, "study.get", {"study_id": self.study_id}
            )

    def _detail_received(self, tag: str, result: Mapping[str, Any]) -> None:
        if tag != "results-study:" + str(self.study_id):
            return  # a study chosen before
        if result.get("state") == "loading":
            self.detail_timer.start()
        elif result.get("state") == "ready":
            self.detail = dict(result)
            self.study_view.set_detail(self.detail)
            self.refresh_head()
            self.names.campaigns.update(
                {str(c["id"]): str(c.get("name") or str(c["id"])[:8]) for c in result["campaigns"]}
            )
            self._campaign_options()

    def _studies_received(self, result: Mapping[str, Any]) -> None:
        self.studies = [
            row for row in result.get("rows") or [] if row.get("final_runs") or row.get("campaigns")
        ]
        self.study.blockSignals(True)
        self.study.clear()
        if not self.studies:
            self.study.addItem("No study yet", None)
        for row in self.studies:
            self.study.addItem(study_label(row), row["study_id"])
        if self.study_id is None and self.studies:
            self.study_id = str(self.studies[0]["study_id"])
            self.files_view.set_study(self.study_id)
            self.filter_runs()
            self.load_detail()
        self.study.setCurrentIndex(max(0, self.study.findData(self.study_id)))
        self.study.blockSignals(False)
        self._size_select()

    def _campaigns_received(self, result: Mapping[str, Any]) -> None:
        self.campaigns = list(result.get("rows") or [])
        self.names.campaigns.clear()
        self.names.campaigns.update(
            {str(r["id"]): str(r.get("name") or str(r["id"])[:8]) for r in self.campaigns}
        )
        if self.wanted_campaign is not None:
            row = next((r for r in self.campaigns if r["id"] == self.wanted_campaign), None)
            if row is not None:
                self.select_study(
                    str(row.get("study_id") or UNASSIGNED), campaign=self.wanted_campaign
                )
                self.wanted_campaign = None
        self._campaign_options()

    def _campaign_options(self) -> None:
        """The study's campaigns (study.get; the Campaigns list's rows of the study until it is read)."""
        mine = (
            self.detail["campaigns"]
            if self.detail is not None and self.detail["study_id"] == self.study_id
            else [r for r in self.campaigns if (r.get("study_id") or UNASSIGNED) == self.study_id]
        )
        # a campaign a restarted copy replaces stays choosable, marked: the study's views leave it out
        self.runs.set_options(
            "campaign_id",
            [
                (
                    str(r.get("name") or r["id"])
                    + (" · replaced by its restart" if r.get("superseded_by_campaign_id") else ""),
                    r["id"],
                )
                for r in mine
            ],
        )
        self.results.view.viewport().update()

    def _study_picked(self, index: int) -> None:
        identity = self.study.itemData(index)
        if identity and identity != self.study_id:
            self.select_study(str(identity))

    def select_study(self, identity: str, *, campaign: str | None = None) -> None:
        if identity != self.study_id:
            self.detail = None
            self.detail_timer.stop()
        self.study_id = identity
        self.files_view.set_study(identity)
        self.study.blockSignals(True)
        self.study.setCurrentIndex(max(0, self.study.findData(identity)))
        self.study.blockSignals(False)
        self._campaign_options()
        if campaign is not None:
            select = self.runs.campaign
            select.blockSignals(True)
            select.setCurrentIndex(max(0, select.findData(campaign)))
            select.blockSignals(False)
        self._size_select()
        self.filter_runs()
        if self.detail is None:
            self.load_detail()

    def set_campaign(self, identity: str | None) -> None:
        """A campaign chosen on Campaigns: its study, and the runs filtered to it."""
        if identity is None:
            return
        row = next((r for r in self.campaigns if r["id"] == identity), None)
        if row is None:
            self.wanted_campaign = identity
            self.load_studies()
            return
        self.select_study(str(row.get("study_id") or UNASSIGNED), campaign=identity)

    def _size_select(self) -> None:
        """``.select.study``: at least 316 px, as wide as its key and current text (and its chevron)."""
        text = BODY.advance("Study") + 2 + BODY.advance(" " + self.study.currentText())
        self.head.set_width(self.study, max(316, math.ceil(2 + 10 + text + 8 + 16 + 10)))

    # names and filter options

    def name_of(self, kind: str, identity: Any) -> str:
        """A campaign's, robot's, world's, spawn's or profile's label (the run view's head and provenance)."""
        tables = {
            "campaign": self.names.campaigns,
            "robot": self.names.robots,
            "world": self.names.worlds,
            "spawn": self.names.spawns,
            "profile": self.names.profiles,
        }
        return self.names.get(tables[kind], identity)

    def set_names(self, catalog: Mapping[str, Any]) -> None:
        self.run_view.set_catalog(catalog)
        self.names.robots.update({r["id"]: r["label"] for r in catalog["robots"]})
        self.names.worlds.update({w["id"]: w["label"] for w in catalog["worlds"]})
        self.runs.set_options("robot_id", [(r["label"], r["id"]) for r in catalog["robots"]])
        self.runs.set_options("world_id", [(w["label"], w["id"]) for w in catalog["worlds"]])
        self.runs.set_options("outcome", [(humanize(o), o) for o in OUTCOMES])

    def set_filter_options(
        self, profiles: Sequence[tuple[str, Any]], spawns: Sequence[tuple[str, Any]]
    ) -> None:
        """The library's profiles and spawns ("Robot · Name", identity): the profile filter and the names the
        table shows (the profile's own name, the spawn's world and name)."""
        self.runs.set_options("profile_id", list(profiles))
        self.names.profiles.update(
            {str(value): text.split(" · ", 1)[-1] for text, value in profiles}
        )
        self.names.spawns.update({str(value): text.split(" · ", 1)[-1] for text, value in spawns})
        self.results.view.viewport().update()

    # runs

    def filter_runs(self) -> None:
        if self.study_id is None:
            return
        self.results.filters = {"study_id": self.study_id, "is_final": True, **self.runs.filters()}
        self.results.offset = 0
        self.results.model.replace([])
        self.results.refresh()
        # every attempt of the filter: the final runs by report class, the earlier ones replaced or discarded
        counted = {key: value for key, value in self.results.filters.items() if key != "is_final"}
        self.runs.footer.set_counts(None)  # never the previous filter's counts
        self.counts_tag = "results-run-counts:" + uuid.uuid4().hex
        self.transport.submit(
            self.counts_tag,
            "results.query",
            {
                "table": "runs",
                "filters": counted,
                "offset": 0,
                "limit": 1,
                "include": ["counts"],
            },
        )

    def _page_loaded(self, result: Mapping[str, Any]) -> None:
        total = result.get("total")
        if isinstance(total, int):
            self.tabs.set_count(RUNS, total)

    def open_run(self, record: Mapping[str, Any]) -> None:
        """The run's view in place of the page; the Monitor's Logs follow the run until a live one is chosen."""
        identity = str(record.get("attempt_id") or "")
        if not identity:
            return
        self.run_view.show_run(record)
        self.study_page.hide()
        self.run_view.show()
        self.run_view.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self.attempt_selected.emit(identity, None)

    def close_run(self) -> None:
        """Back to the runs as they were (filters, scroll and expanded attempts)."""
        self.run_view.close_timers()  # no polling for a view that is gone; opening a run asks again
        self.run_view.hide()
        self.study_page.show()
        self.results.view.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def show_run_files(self, attempt_id: str) -> None:
        """The run view's Files: the Files tab rooted at the run's directory."""
        self.files_view.show_run(attempt_id, run_label(self.run_view.record, self.name_of))
        self.close_run()
        self.tabs.set_index(FILES)
        self._tab(FILES)

    # export and analysis

    def _export(self) -> None:
        if self.study_id is not None:
            self.jobs.start_export(self.study_id)

    def _analyse(self) -> None:
        export = newest_export(self.detail)
        if self.study_id is not None and export is not None:
            self.jobs.start_analysis(self.study_id, export)

    def _job_changed(self, study_id: str) -> None:
        self.refresh_head()
        if study_id == self.study_id:
            self.study_view.refresh_card()

    def refresh_head(self) -> None:
        """Run analysis needs a finished export and the analysis environment; one analysis and one export run at
        a time (decision 6)."""
        reason = analysis_available()
        if self.detail is None or self.detail.get("study_id") != self.study_id:
            reason = reason or "Reading the study…"
        elif newest_export(self.detail) is None:
            reason = reason or "Export the study first: the analysis reads a finished export"
        if self.jobs.running():
            reason = "An analysis is running; Show log on the study card follows it"
        self.analyse.setEnabled(not reason)
        self.analyse.setToolTip(reason or "Analyse the newest export in the analysis environment")
        exporting = self.jobs.export is not None and self.jobs.export.state == "running"
        self.export.setEnabled(not exporting)
        self.export.setToolTip(
            "An export is running" if exporting else "Write a new export of the study's runs"
        )

    # replies

    def received(self, tag: str, result: Any) -> bool:
        if (
            self.run_view.received(tag, result)
            or self.study_view.received(tag, result)
            or self.files_view.received(tag, result)
            or self.jobs.received(tag, result)
        ):
            return True
        if tag == "results-studies":
            self._studies_received(result)
            return True
        if tag == "results-campaigns":
            self._campaigns_received(result)
            return True
        if tag.startswith("results-run-counts:"):
            if tag == self.counts_tag and result.get("total") is not None:
                self.runs.footer.set_counts(result.get("counts"))
            return True
        if tag.startswith("results-study:"):
            self._detail_received(tag, result)
            return True
        if tag.startswith("results-attempts:"):
            return self.results.attempts_received(tag, result)
        return False

    def failed(self, tag: str, error: str) -> bool:
        if (
            self.run_view.failed(tag, error)
            or self.study_view.failed(tag, error)
            or self.files_view.failed(tag, error)
            or self.jobs.failed(tag, error)
        ):
            return True
        if tag.startswith("results-run-counts:"):
            return True  # the footer stays empty: no counts rather than another filter's
        if tag in ("results-studies", "results-campaigns") or tag.startswith("results-study:"):
            return True  # the select keeps its studies
        if tag.startswith("results-attempts:"):
            self.results.expanding.pop(tag, None)
            return False  # the window shows the engine's reason
        return False

    # layout

    def _tab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        on_study = index == STUDY
        self.analyse.setVisible(on_study)
        self.export.setVisible(on_study)
        self.head.relayout()
        if index == RUNS:
            self.results.refresh()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.run_view.setGeometry(self.rect())
        self.study_page.setGeometry(
            PAGE_LEFT,
            PAGE_TOP,
            self.width() - 2 * PAGE_LEFT,
            max(0, self.height() - PAGE_TOP - PAGE_BOTTOM),
        )
        width, height = self.study_page.width(), self.study_page.height()
        self.head.setGeometry(0, 0, width, 36)
        place(self.tabs, 0, 48, width, 39)
        self.stack.setGeometry(0, 103, width, max(0, height - 103))

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.load_studies()

    def close_timers(self) -> None:
        self.run_view.close_timers()
        self.study_view.close_timers()
        self.files_view.close_timers()
        self.jobs.close()
        self.sync.stop()
        self.detail_timer.stop()
