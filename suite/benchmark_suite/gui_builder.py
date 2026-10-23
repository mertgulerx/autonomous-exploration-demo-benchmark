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
from pathlib import Path
from typing import Any
import uuid

import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from .gui_cells import OverlayScrollBar
from .gui_support import POLL_MS, ROOT, Transport
from .gui_review import (
    VALIDATE_ACCEPTED,
    VALIDATE_CHANGES,
    AcceptedCard,
    ChangesCard,
    Combination,
    CombinationsCard,
    ReviewPage,
    accepted_facts,
    change_blocks,
)
from .gui_robots import RobotSelector
from .gui_worlds import WorldSelector
from .gui_theme import ThemeManager
from .gui_widgets import CardSplit, place, plural
from .analysis_block import plan_parts
from .gui_wizard import (
    EXISTING_STUDY,
    NEW_STUDY,
    UNASSIGNED_STUDY,
    DraftFacts,
    DraftSummary,
    ExecutionCard,
    StudyForm,
    SummaryItem,
    WizardFooter,
    WizardHead,
    seconds,
)
from .contracts import resolve_world_time_limits


class StepCards(QtWidgets.QWidget):
    """Step 3's Campaign card over the Study card (16 px apart), and the 420 px draft summary, as tall as the
    taller side; the gap between them can be dragged (remembered; a double click restores the design's
    width)."""

    def __init__(
        self,
        themes: ThemeManager,
        execution: ExecutionCard,
        study: StudyForm,
        summary: DraftSummary,
    ) -> None:
        super().__init__()
        self.execution, self.study, self.summary = execution, study, summary
        for card in (execution, study, summary):
            card.setParent(self)
        self.side = CardSplit(themes, self, "builder/summary", self.relayout, across=True, sign=-1)

    def relayout(self) -> None:
        width = self.width()
        top = self.execution.natural_height()
        height = max(top + 16 + self.study.natural_height(), self.summary.natural_height())
        self.setMinimumHeight(height)
        side = self.side.resolve(420, 340, width - 16 - 520)
        place(self.execution, 0, 0, width - side - 16, top)
        place(self.study, 0, top + 16, width - side - 16, height - top - 16)
        place(self.summary, width - side, 0, side, height)
        self.side.place(width - side - 16, 0, 16, height)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()


class CampaignBuilder(QtWidgets.QWidget):
    """New campaign (03–08): the head with the steps, one page per step and the bottom bar. Messages go to
    toasts (``message``: text and kind, D8)."""

    return_requested = QtCore.Signal()
    tuning_started = QtCore.Signal()  # the window opens Monitor, which follows the sole active run
    edit_profile = QtCore.Signal(str, object)
    message = QtCore.Signal(str, str)

    def __init__(self, transport: Transport, themes: ThemeManager):
        super().__init__()
        self.transport = transport
        self.themes = themes
        self.catalog: dict[str, Any] = {}
        self.source_campaign_id: str | None = None
        self.excluded: set[tuple[str, str, str]] = set()
        self.validated: dict[str, Any] | None = None
        # Form content the engine accepted, and the content of the preview in flight.
        self.validated_design: dict[str, Any] | None = None
        self.pending_design: dict[str, Any] | None = None
        self.validation_tag = ""
        self.submission_tag = ""
        self.import_tag = ""
        self.export_tag = ""
        self.tuning_tag = ""
        self.step = 0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.head = WizardHead(themes)
        self.head.back.clicked.connect(self.return_requested.emit)
        self.head.steps.step_selected.connect(self.show_step)
        layout.addWidget(self.head)
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.robot_selector = RobotSelector(themes, transport)
        self.robot_selector.notice.connect(lambda text: self.set_message(text, "muted"))
        self.robot_selector.edit_profile.connect(self.edit_profile.emit)
        self.stack.addWidget(self.robot_selector)
        self.world_selector = WorldSelector(themes, transport)
        self.stack.addWidget(self.world_selector)
        # Step 3: the settings card and the draft summary side by side (``grid 1fr 420px``).
        self.execution = ExecutionCard(themes)
        self.study_form = StudyForm(themes)
        self.summary = DraftSummary(themes)
        self.summary.change_requested.connect(self.show_step)
        execution_page = StepCards(themes, self.execution, self.study_form, self.summary)
        # The studies a campaign can join (studies.list): ID -> row (evidence levels, plan, last repetition)
        self.studies: dict[str, dict[str, Any]] = {}
        self.studies_tag = ""
        self.plan_tag = ""
        self.plan_timer = QtCore.QTimer(self)
        self.plan_timer.setSingleShot(True)
        self.plan_timer.setInterval(POLL_MS)
        self.plan_timer.timeout.connect(self.read_plan)
        # A long draft (many robots or worlds) makes both cards taller; the step then scrolls.
        execution_scroll = QtWidgets.QScrollArea()
        execution_scroll.setWidgetResizable(True)
        execution_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        execution_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        execution_scroll.viewport().setAutoFillBackground(False)
        execution_page.setAutoFillBackground(False)
        execution_scroll.setWidget(execution_page)
        OverlayScrollBar(themes, execution_scroll)
        self.stack.addWidget(execution_scroll)
        self.fit_step_cards()
        self.name = self.execution.name
        self.description = self.execution.description
        self.repetitions = self.execution.repetitions
        self.concurrency = self.execution.concurrency
        self.concurrency.set_index(0)
        # Step 4: the combinations over the profile changes, and the accepted campaign (08).
        self.combinations = CombinationsCard(themes)
        self.combinations.toggled.connect(self.toggle_exclusion)
        self.combinations.toggled_all.connect(self.toggle_all_exclusions)
        self.changes = ChangesCard(themes)
        self.accepted = AcceptedCard(themes)
        self.world_selector.map_loaded.connect(self.accepted.set_map)
        self.review_page = ReviewPage(self.combinations, self.changes, self.accepted)
        self.stack.addWidget(self.review_page)
        self.footer = WizardFooter(themes)
        layout.addWidget(self.footer)
        self.footer.import_button.clicked.connect(self.import_campaign)
        self.export_button = self.footer.export_button
        self.export_button.clicked.connect(self.export_campaign)
        self.back_button = self.footer.back_button
        self.back_button.clicked.connect(lambda: self.show_step(self.step - 1))
        self.next_button = self.footer.next_button
        self.next_button.clicked.connect(self.next_step)
        self.footer.validate_button.clicked.connect(lambda: self.submit_design(False))
        self.submit_button = self.footer.submit_button
        self.submit_button.clicked.connect(lambda: self.submit_design(True))
        self.tuning_box = self.footer.tuning_box
        self.tuning_box.toggled.connect(self.set_tuning)
        self.footer.start_button.clicked.connect(self.start_tuning)
        self.robot_selector.changed.connect(self.invalidate)
        self.world_selector.changed.connect(self.invalidate)
        for field in (self.name, self.description):
            field.textChanged.connect(self.invalidate)
        self.repetitions.valueChanged.connect(self.invalidate)
        self.concurrency.changed.connect(self.invalidate)
        form = self.study_form
        form.kind.changed.connect(self.study_kind)
        form.evidence.changed.connect(self.invalidate)
        form.new_id.textChanged.connect(self.invalidate)
        form.existing.currentIndexChanged.connect(self.study_picked)
        form.plan.textChanged.connect(self.invalidate)
        form.plan.textChanged.connect(lambda _: self.plan_timer.start())
        form.choose_plan.connect(self.choose_plan)
        self.execution.debug.toggled.connect(self.invalidate)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)
        self.show_step(0)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.load_studies()  # the studies a campaign can join (Existing study, the repetition range)

    def design(self) -> dict[str, Any]:
        worlds = self.world_selector.selections()
        robots = self.robot_selector.selections()
        combinations = {
            (robot, world, self.spawn_identity(world, spawn))
            for robot in robots
            for world, spawns in worlds.items()
            for spawn in spawns
        }
        return {
            "name": self.name.text(),
            "description": self.description.text(),
            "robots": robots,
            "worlds": worlds,
            "repetitions": self.repetitions.value(),
            "concurrency": self.concurrency.value(),
            "world_time_limits": self.world_selector.time_limits(),
            "exclusions": [list(v) for v in sorted(self.excluded & combinations)],
            **({"source_campaign_id": self.source_campaign_id} if self.source_campaign_id else {}),
            **({"study": study} if (study := self.study_block()) is not None else {}),
            **({"debug_streams": True} if self.execution.debug.isChecked() else {}),
        }

    # the study (32)

    def study_id(self) -> str | None:
        form = self.study_form
        kind = form.kind.index()
        if kind == NEW_STUDY:
            return form.new_id.text().strip() or None
        if kind == EXISTING_STUDY:
            value = form.existing.currentData()
            return str(value) if value else None
        return None

    def study_block(self) -> dict[str, Any] | None:
        """The campaign's ``study`` block (contracts.Study): the engine adds the plan's hash and numbering."""
        form = self.study_form
        if form.kind.index() == UNASSIGNED_STUDY:
            return None
        identity = self.study_id()
        if identity is None:
            raise ValueError(
                "Name the new study, or choose Unassigned"
                if form.kind.index() == NEW_STUDY
                else "Choose an existing study, or choose Unassigned"
            )
        plan = form.plan.text().strip()
        return {
            "id": identity,
            "evidence_level": "confirmatory" if form.evidence.index() == 1 else "exploratory",
            "plan": plan or None,
        }

    def load_studies(self) -> None:
        self.studies_tag = "builder-studies-" + uuid.uuid4().hex
        self.transport.submit(self.studies_tag, "studies.list", {"offset": 0, "limit": 250})

    def studies_received(self, result: dict[str, Any]) -> None:
        rows = [r for r in result.get("rows") or [] if r.get("study_id") != "unassigned"]
        self.studies = {str(r["study_id"]): r for r in rows}
        select = self.study_form.existing
        chosen = select.currentData()
        select.blockSignals(True)
        select.clear()
        for row in rows:
            levels = " and ".join(row.get("evidence_levels") or ["exploratory"])
            select.addItem(f"{row['study_id']} · {levels}", row["study_id"])
        select.setCurrentIndex(max(0, select.findData(chosen)))
        select.blockSignals(False)
        if self.study_form.kind.index() == EXISTING_STUDY and select.currentData() != chosen:
            self.study_picked(select.currentIndex())  # chosen before the list arrived
        self.update_summary()

    def study_kind(self, kind: int) -> None:
        """Choosing Existing study takes the shown study's level and plan (its select may not change)."""
        if kind == EXISTING_STUDY:
            self.study_picked(self.study_form.existing.currentIndex())
        else:
            self.invalidate()

    def study_picked(self, index: int) -> None:
        """An existing study: its evidence level and plan as its newest campaign declared them."""
        row = self.studies.get(str(self.study_form.existing.itemData(index) or ""))
        if row is not None:
            levels = row.get("evidence_levels") or []
            self.study_form.evidence.set_index(1 if "confirmatory" in levels else 0)
            self.study_form.plan.setText(str(row.get("plan") or ""))
        self.invalidate()

    def study_range(self) -> str:
        """ "· study repetitions 1–5": the numbers this campaign's repetitions take in its study (RD-30)."""
        identity = self.study_id()
        if identity is None:
            return ""
        last = (self.studies.get(identity) or {}).get("last_repetition")
        first = (last + 1 if isinstance(last, int) else 0) + 1
        return f"· study repetitions {first}–{first + self.repetitions.value() - 1}"

    def choose_plan(self) -> None:
        start = ROOT / "docs/research/experiments"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Study plan", str(start if start.is_dir() else ROOT), "Markdown (*.md)"
        )
        if not path:
            return
        chosen = Path(path).resolve()
        if not chosen.is_relative_to(ROOT.resolve()):
            self.set_message("Choose a plan inside this checkout: the engine reads it from there.")
            return
        self.study_form.plan.setText(chosen.relative_to(ROOT.resolve()).as_posix())

    def read_plan(self) -> None:
        """The plan's SHA-256 above its deviations log, read off the GUI thread (the engine checks it again at
        submission, from its own checkout)."""
        plan = self.study_form.plan.text().strip()
        if not plan or self.study_form.kind.index() == UNASSIGNED_STUDY:
            self.study_form.set_hash("")
            return
        self.plan_tag = "builder-plan-" + uuid.uuid4().hex
        self.transport.submit(self.plan_tag, "local.read", {"path": str(ROOT / plan)})

    def invalidate(self, *args: Any) -> None:
        if self.import_tag and self.validation_tag.startswith("campaign-import-"):
            # Dropping the import's preview would otherwise leave validation blocked.
            self.import_tag = ""
            self.set_message("Import cancelled by an edit.", "muted")
        self.validated = None
        self.validated_design = None
        self.pending_design = None
        self.validation_tag = ""
        self.show_unvalidated(VALIDATE_ACCEPTED)
        self.sync_actions()
        self.update_summary()
        if self.step == 3:
            self.refresh_combinations()

    def show_unvalidated(self, message: str) -> None:
        """Step 4 without an accepted campaign: the cards that describe it say why."""
        self.accepted.set_message(message)
        self.changes.set_blocks([], VALIDATE_CHANGES)

    def labels(self, kind: str) -> dict[str, str]:
        """Catalog labels by ID of ``kind`` ("robots" or "worlds")."""
        return {
            entry["id"]: entry.get("label", entry["id"]) for entry in self.catalog.get(kind, [])
        }

    def sync_actions(self) -> None:
        exportable = (
            self.validated is not None and self.validated_design is not None and not self.import_tag
        )
        self.submit_button.setEnabled(self.validated is not None and not self.submission_tag)
        self.footer.start_button.setEnabled(not self.tuning_tag and not self.import_tag)
        self.export_button.setEnabled(exportable)
        self.export_button.setToolTip(
            "Save the accepted campaign as YAML" if exportable else "Available after validation"
        )

    def unchanged_since_validation(self) -> bool:
        """Whether the form still holds what the engine accepted; invalidates when not."""
        try:
            unchanged = self.design() == self.validated_design
        except ValueError:
            unchanged = False
        if not unchanged:
            self.invalidate()
            self.set_message("The campaign changed after validation; validate it again.")
        return unchanged

    def study_items(self) -> tuple[SummaryItem, ...]:
        form = self.study_form
        identity = self.study_id()
        kind = form.kind.index()
        if kind == UNASSIGNED_STUDY or identity is None:
            study = "Unassigned"
        else:
            study = f"{identity} · " + ("existing" if identity in self.studies else "new")
        evidence = (
            "—"
            if kind == UNASSIGNED_STUDY
            else form.evidence.segments[form.evidence.index()].text()
        )
        return (
            SummaryItem("Study", study),
            SummaryItem("Evidence level", evidence),
            SummaryItem("Debug streams", "On" if self.execution.debug.isChecked() else "Off"),
        )

    def update_summary(self) -> None:
        robots, worlds = self.robot_selector.selections(), self.world_selector.selections()
        count = 0
        for robot, profiles in robots.items():
            for world, spawns in worlds.items():
                count += len(profiles) * sum(
                    (robot, world, self.spawn_identity(world, s)) not in self.excluded
                    for s in spawns
                )
        repetitions = self.repetitions.value()
        if self.tuning_box.isChecked():  # one run, whatever Execution and Review hold
            repetitions, count = 1, int(bool(robots and worlds))
        else:
            count *= repetitions
        self.footer.totals.set_counts(
            len(robots),
            sum(map(len, robots.values())),
            len(worlds),
            repetitions,
            count,
        )
        self.summary.set_facts(self.draft_facts())
        self.execution.set_study_range(self.study_range())
        self.study_form.sync()
        self.fit_step_cards()

    def fit_step_cards(self) -> None:
        """Step 3's cards share one height: the settings card's or the draft summary's, the taller."""
        execution_page = self.execution.parentWidget()
        if isinstance(execution_page, StepCards):
            execution_page.relayout()

    def draft_facts(self) -> DraftFacts:
        """Step 3's summary of the form: robots and profiles, worlds and poses, combinations per robot."""
        robots, worlds = self.robot_selector.selections(), self.world_selector.selections()
        robot_labels, world_labels = self.labels("robots"), self.labels("worlds")
        try:
            limits: dict[str, float] = self.world_selector.time_limits()
        except ValueError:
            limits = {}
        world_items = []
        for world, spawns in worlds.items():
            names = ", ".join(self.world_selector.spawn_label(world, s) for s in spawns)
            limit = limits.get(world)
            if limit is not None:
                names += " · " + ("unlimited" if limit == 0 else f"{seconds(limit)} s")
            world_items.append(SummaryItem(world_labels.get(world, world), names))
        combinations, left_out = [], []
        for robot, profiles in robots.items():
            poses = 0
            for world, spawns in worlds.items():
                for spawn in spawns:
                    if (robot, world, self.spawn_identity(world, spawn)) in self.excluded:
                        left_out.append(
                            f"{robot_labels.get(robot, robot)} · {world_labels.get(world, world)}"
                            f" · {self.world_selector.spawn_label(world, spawn)}"
                        )
                    else:
                        poses += 1
            detail = f"{plural(len(profiles), 'profile')} × {plural(poses, 'starting pose')}"
            combinations.append(
                SummaryItem(robot_labels.get(robot, robot), detail, len(profiles) * poses)
            )
        if not left_out:
            note = ""
        elif len(left_out) == 1:
            note = left_out[0] + " is left out in Review."
        else:
            note = ", ".join(left_out[:-1]) + " and " + left_out[-1] + " are left out in Review."
        return DraftFacts(
            robots=tuple(
                SummaryItem(
                    robot_labels.get(robot, robot),
                    ", ".join(self.robot_selector.profile_label(robot, ref) for ref in references),
                )
                for robot, references in robots.items()
            ),
            worlds=tuple(world_items),
            study=self.study_items(),
            combinations=tuple(combinations),
            per_repetition=sum(item.count or 0 for item in combinations),
            note=note,
        )

    @staticmethod
    def spawn_identity(world: str, reference: Any) -> str:
        return (
            str(reference["id"])
            if isinstance(reference, dict)
            else "default-" + world
            if reference == "Default"
            else str(reference)
        )

    def set_tuning(self, tuning: bool) -> None:
        """A tuning run: one robot, profile, world and starting pose; Execution and Review are skipped and
        Start run on the worlds step starts it (plans/SUITE_TUNING_RUN.md)."""
        self.robot_selector.set_single(tuning)
        self.world_selector.set_single(tuning)
        for button in self.head.steps.buttons[2:]:
            button.setEnabled(not tuning)
        self.show_step(self.step)

    def show_step(self, index: int) -> None:
        tuning = self.tuning_box.isChecked()
        self.step = max(0, min(index, 1 if tuning else 3))
        self.stack.setCurrentIndex(self.step)
        self.head.steps.set_step(self.step, range(self.step))
        self.back_button.setEnabled(self.step > 0)
        self.next_button.setVisible(self.step < (1 if tuning else 3))
        self.footer.start_button.setVisible(tuning and self.step == 1)
        self.tuning_box.setVisible(self.step <= 1)
        self.footer.import_button.setVisible(not tuning)
        self.export_button.setVisible(not tuning)
        self.footer.validate_button.setVisible(self.step == 3)
        self.submit_button.setVisible(self.step == 3)
        self.sync_actions()
        if self.step == 3:
            self.refresh_combinations()
        self.update_summary()
        self.footer.relayout()

    def set_message(self, text: str, kind: str = "error") -> None:
        """A toast (D8): it fades after 3 s and is dismissible."""
        self.message.emit(text, {"error": "bad", "success": "ok"}.get(kind, "neutral"))

    def next_step(self) -> None:
        if self.step == 0:
            selected = self.robot_selector.selections()
            if not selected or any(not values for values in selected.values()):
                self.set_message("Select at least one robot and a profile for each selected robot.")
                return
        if self.step == 1:
            selected = self.world_selector.selections()
            if not selected or any(not values for values in selected.values()):
                self.set_message(
                    "Select at least one world and a starting pose for each selected world."
                )
                return
        self.show_step(self.step + 1)

    def start_tuning(self) -> None:
        """Start run: send the form as one tuning run; the window opens Monitor once it starts."""
        if self.tuning_tag or self.import_tag:
            return
        robots, worlds = self.robot_selector.selections(), self.world_selector.selections()
        if not robots or not worlds:
            self.set_message("Select a robot and a world for the tuning run.")
            return
        try:
            limits = self.world_selector.time_limits()
        except ValueError as exc:
            self.set_message(str(exc))
            return
        self.tuning_tag = "builder-tuning-" + uuid.uuid4().hex
        self.sync_actions()
        self.transport.submit(
            self.tuning_tag,
            "tuning.start",
            {
                "specification": {
                    "name": "Tuning run",  # required by the contract; the engine names every tuning run so
                    "robots": robots,
                    "worlds": worlds,
                    "repetitions": 1,
                    "concurrency": 1,
                    "world_time_limits": limits,
                }
            },
        )

    def combination_rows(self) -> list[Combination]:
        """Every robot, world and starting pose of the form; robot and profiles once per robot (08)."""
        robot_labels, world_labels = self.labels("robots"), self.labels("worlds")
        rows = []
        for robot, references in self.robot_selector.selections().items():
            profiles = ", ".join(self.robot_selector.profile_label(robot, r) for r in references)
            first = True
            for world, spawns in self.world_selector.selections().items():
                for spawn in spawns:
                    rows.append(
                        Combination(
                            (robot, world, self.spawn_identity(world, spawn)),
                            robot_labels.get(robot, robot) if first else "",
                            profiles if first else "",
                            world_labels.get(world, world),
                            self.spawn_label(world, spawn),
                            len(references) * self.repetitions.value(),
                        )
                    )
                    first = False
        return rows

    def refresh_combinations(self) -> None:
        self.combinations.set_rows(self.combination_rows(), self.excluded)
        self.review_page.relayout()

    def spawn_label(self, world: str, reference: Any) -> str:
        return self.world_selector.spawn_label(world, reference)

    def toggle_exclusion(self, key: tuple[str, str, str]) -> None:
        if key in self.excluded:
            self.excluded.discard(key)
        else:
            self.excluded.add(key)
        self.invalidate()

    def toggle_all_exclusions(self) -> None:
        """The header check: include every combination, or leave them all out."""
        keys = {row.key for row in self.combination_rows()}
        if keys & self.excluded:
            self.excluded -= keys
        else:
            self.excluded |= keys
        self.invalidate()

    def submit_design(self, submit: bool) -> None:
        if self.import_tag:
            return
        if submit:
            if self.validated is None or self.submission_tag:
                return
            if not self.unchanged_since_validation():
                return
            self.submission_tag = "builder-submit-" + uuid.uuid4().hex
            self.sync_actions()
            self.transport.submit(
                self.submission_tag, "campaign.submit", {"specification": deepcopy(self.validated)}
            )
        else:
            try:
                design = self.design()
            except ValueError as exc:
                self.set_message(str(exc))
                return
            # A new validation replaces the old result, so a failure leaves nothing enabled.
            self.validated = None
            self.validated_design = None
            self.pending_design = deepcopy(design)
            self.validation_tag = "builder-preview-" + uuid.uuid4().hex
            self.sync_actions()
            self.show_unvalidated("Validating included combinations…")
            self.transport.submit(
                self.validation_tag, "campaign.preview", {"specification": design}
            )

    def apply_design(self, value: dict[str, Any]) -> None:
        self.tuning_box.setChecked(False)  # an imported or repeated campaign is never a tuning run
        self.source_campaign_id = value.get("source_campaign_id")
        self.name.setText(value["name"])
        self.description.setText(value.get("description", ""))
        self.repetitions.setValue(value["repetitions"])
        self.concurrency.set_value(value["concurrency"])
        self.excluded = {tuple(v) for v in value["exclusions"]}
        self.robot_selector.set_selections(value["robots"])
        self.world_selector.set_selections(value["worlds"])
        self.world_selector.set_time_limits(dict(resolve_world_time_limits(value, value["worlds"])))
        self.execution.debug.setChecked(bool(value.get("debug_streams", False)))
        study = value.get("study")
        form = self.study_form
        if isinstance(study, dict) and study.get("id"):
            identity = str(study["id"])
            if identity in self.studies:
                form.kind.set_index(EXISTING_STUDY)
                form.existing.setCurrentIndex(max(0, form.existing.findData(identity)))
            else:
                form.kind.set_index(NEW_STUDY)
                form.new_id.setText(identity)
            form.evidence.set_index(1 if study.get("evidence_level") == "confirmatory" else 0)
            form.plan.setText(str(study.get("plan") or ""))
        else:
            form.kind.set_index(UNASSIGNED_STUDY)
            form.plan.setText("")
        self.invalidate()
        self.show_step(0)

    def import_campaign(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import campaign", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.invalidate()
            self.import_tag = "campaign-read-" + uuid.uuid4().hex
            self.set_message("Reading campaign configuration…", "muted")
            self.transport.submit(self.import_tag, "local.read", {"path": path})

    def export_campaign(self) -> None:
        """Save the specification the engine accepted for the current form."""
        if self.validated is None or self.validated_design is None or self.import_tag:
            return
        if not self.unchanged_since_validation():
            return
        # Serialized before the modal dialog: replies delivered while it is open cannot change it.
        text = yaml.safe_dump(deepcopy(self.validated), sort_keys=False, allow_unicode=True)
        if len(text.encode()) > 4 * 1024 * 1024:
            self.set_message(
                "The validated campaign exceeds the 4 MiB import limit; it cannot be re-imported."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export campaign", "campaign.yaml", "YAML (*.yaml)"
        )
        if path:
            self.export_tag = "builder-export-" + uuid.uuid4().hex
            self.transport.submit(self.export_tag, "local.write", {"path": path, "text": text})

    def received(self, tag: str, result: Any) -> None:
        if tag == self.studies_tag:
            self.studies_tag = ""
            self.studies_received(result)
            return
        if tag == self.plan_tag:
            self.plan_tag = ""
            try:
                digest = plan_parts(str(result.get("text") or "")).sha256
            except ValueError as exc:  # the engine refuses the same plan at submission
                self.study_form.set_hash("unreadable plan")
                self.study_form.hash.setToolTip(str(exc))
                return
            self.study_form.set_hash("SHA-256 " + digest[:8])
            self.study_form.hash.setToolTip(digest)
            return
        if tag == self.validation_tag:
            specification = result.get("specification") if isinstance(result, dict) else None
            if not isinstance(specification, dict):
                self.import_tag = ""
                self.invalidate()
                self.set_message(
                    "The engine reply has no accepted specification; rebuild the suite image."
                )
                return
            if tag.startswith("campaign-import-"):
                self.import_tag = ""
                self.apply_design(specification)
                try:
                    design: dict[str, Any] | None = self.design()
                except ValueError as exc:
                    self.set_message(str(exc))
                    return
            else:
                design = self.pending_design
            imported = tag.startswith("campaign-import-")
            self.validated = specification
            self.validated_design = design
            self.pending_design = None
            self.show_accepted(specification, result.get("timing_retry"))
            if imported:  # the import lands on step 1, away from the accepted campaign card
                self.set_message(
                    "Validation passed. Add this campaign to the queue, then explicitly Start it.",
                    "success",
                )
            self.show_step(self.step)
        elif tag == self.submission_tag:
            self.submission_tag = ""
            self.validated = None
            self.validated_design = None
            self.sync_actions()
            self.set_message(
                "Campaign added to the queue. Open Campaigns and press Start to execute.",
                "success",
            )
            self.transport.received.emit("submit", result)
        elif tag == self.import_tag:
            from .yaml_io import load_mapping

            try:
                self.validation_tag = "campaign-import-" + uuid.uuid4().hex
                self.transport.submit(
                    self.validation_tag,
                    "campaign.preview",
                    {"specification": load_mapping(result["text"])},
                )
            except ValueError as exc:
                self.import_tag = ""
                self.set_message(str(exc))
        elif tag == self.export_tag:
            self.export_tag = ""
            self.set_message(f"Validated campaign saved to {result['path']}", "success")
        elif tag == self.tuning_tag:
            self.tuning_tag = ""
            self.sync_actions()
            self.set_message("Tuning run started.", "success")
            self.tuning_started.emit()

    def failed(self, tag: str, error: str) -> None:
        if tag == self.studies_tag:
            self.studies_tag = ""
            return  # the Existing study list stays as it was
        if tag == self.plan_tag:
            self.plan_tag = ""
            self.study_form.set_hash("plan not found")
            return
        if tag == self.tuning_tag:
            self.tuning_tag = ""
            self.sync_actions()
            self.set_message(error)
            return
        if tag == self.export_tag:
            self.export_tag = ""
            self.set_message(error)
        elif tag in (self.validation_tag, self.submission_tag, self.import_tag):
            if tag == self.validation_tag:
                self.validated = None
                self.validated_design = None
                self.pending_design = None
            if tag == self.validation_tag and tag.startswith("builder-preview-"):
                self.accepted.set_error(error)  # 08: the error in the card's band
                self.changes.set_blocks([], VALIDATE_CHANGES)
                if self.step != 3:  # the band is out of sight
                    self.set_message(error)
            else:
                self.set_message(error)
            self.import_tag = ""
            self.submission_tag = ""
            self.sync_actions()

    def show_accepted(self, specification: dict[str, Any], timing_retry: Any) -> None:
        """Step 4's view of what the engine accepted: the campaign card and the profile changes."""
        facts = accepted_facts(
            specification,
            timing_retry,
            self.labels("worlds"),
            self.spawn_label,
            lambda identity: identity in self.studies,
        )
        maps = {}
        for world in facts.worlds:
            self.world_selector.ensure_map(world.world_id)
            maps[world.world_id] = self.world_selector.world_map(world.world_id)
        self.accepted.set_accepted(facts, maps)
        robots = {r["id"]: r for r in self.catalog.get("robots", [])}
        blocks = change_blocks(specification, robots, self.catalog.get("parameters", {}))
        self.changes.set_blocks(blocks, "" if blocks else "Every robot runs its Default profile.")
