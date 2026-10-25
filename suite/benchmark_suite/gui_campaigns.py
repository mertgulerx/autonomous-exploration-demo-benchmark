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

"""Campaigns page of the Main design (screens 01 and 02, HANDOFF §8.1-§8.2): the campaign table with ticks
for Delete, the selected campaign's card, Live on the map, and the delete, stop and restart dialogs."""

from collections.abc import Callable, Set as AbstractSet
from dataclasses import dataclass, field
from functools import partial
import json
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .contracts import STOPPED_OUTCOMES, CampaignState, report_exclusion
from .gui_cells import (
    CELL_ROLE,
    CHECK_COLUMN,
    Cell,
    CellDelegate,
    ColumnStyle,
    TableHeader,
    TableLook,
    style_table,
)
from .gui_controls import StatusBadge, badge_look, paint_badge
from .gui_dialogs import BOLD, Consequences, ListRow, Run, bullet
from .gui_maps import LiveLayers, MiniMap, WorldMap, apply_monitor_reply
from .gui_marks import (
    CAMPAIGN_RING,
    KeyValue,
    KeyValueList,
    OutcomeBar,
    ProgressRing,
    paint_legend_dot,
    paint_worker_tag,
)
from .gui_menus import Menu, show_tooltip
from .gui_style import cell_text, humanize, worker_values
from .gui_support import PagedTable, PagedTableModel, Transport
from .gui_theme import ThemeManager, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    CARD_TITLE,
    Button,
    Card,
    CssBox,
    FlexRow,
    LinkButton,
    ModalHost,
    PopupCard,
    PopupHost,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    place,
    plural,
    rounded_path,
)
from .monitoring import DRAWN_LAYERS, PathCursor

# ---------- operations ----------

PERMITTED: dict[str, set[str]] = {
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


def operation_for_state(state: str) -> str | None:
    """The row and card button's operation: Start, Pause or Resume."""
    return next((op for op in ("start", "pause", "resume") if state in PERMITTED[op]), None)


def control_look(state: str) -> tuple[str, str]:
    """Label and icon of the row and card button (hidden for finished campaigns)."""
    if state in {"RUNNING", "RETRYING", "PAUSING"}:
        return "Pause", "pause"
    return ("Start", "play") if state == "QUEUED" else ("Resume", "play")


# ---------- table ----------

COLUMNS = [
    ("check", ""),
    ("name", "Campaign"),
    ("state", "Status"),
    ("count", "Experiments"),
    ("study_id", "Study"),
    ("created_ns", "Submitted"),
    ("actions", "Actions"),
]
ACTIONS = len(COLUMNS) - 1
LOOK = TableLook(
    key="campaigns",
    columns=[
        CHECK_COLUMN,
        ColumnStyle("name"),
        ColumnStyle("badge", width=176),
        ColumnStyle("number", width=112),
        ColumnStyle("text", width=236, padding_left=36),
        ColumnStyle("number", align="left", width=164),
        ColumnStyle("empty", align="right", width=148),
    ],
    round_top=7,
    round_bottom=7,
)
SORT_KEYS = {1: "name", 2: "attention", 3: "count", 5: "created_ns"}
ATTENTION_REJECTED = "Choose a campaign table column"  # an engine without the attention sort


class CampaignModel(PagedTableModel):
    """Campaign pages with design cells; ``chosen`` (the ticked campaign IDs, in tick order) drives the
    checkbox column."""

    def __init__(self) -> None:
        super().__init__(COLUMNS)
        self.chosen: dict[str, None] = {}

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == CELL_ROLE and index.isValid():
            record = self.record(index.row())
            key = self.columns[index.column()][0]
            if record is None:
                return Cell("…") if key == "name" else Cell(kind="empty")
            return self.design_cell(record, key)
        return super().data(index, role)

    def cell(self, record: dict[str, Any], column: int, role: int) -> Any:
        if self.columns[column][0] == "check":
            return None
        return super().cell(record, column, role)

    def design_cell(self, record: dict[str, Any], key: str) -> Cell:
        if key == "check":
            ticked = record["id"] in self.chosen
            return Cell(
                kind="check",
                check=QtCore.Qt.CheckState.Checked if ticked else QtCore.Qt.CheckState.Unchecked,
            )
        if key == "name":
            return Cell(record["name"], "name")
        if key == "state":
            look = badge_look("campaign", record["state"])
            return Cell(kind="badge", badge=look) if look else Cell(humanize(record["state"]))
        if key == "count":
            return Cell(f"{record['count']:,}", "number")
        if (
            key == "study_id"
        ):  # (34) the study, or a muted "Unassigned"; a tag on a confirmatory one
            study = record.get("study_id")
            if not study or study == "unassigned":
                return Cell("Unassigned", token="text.tertiary")
            confirmatory = record.get("evidence_level") == "confirmatory"
            return Cell(str(study), note="confirmatory" if confirmatory else "", note_kind="tag")
        if key == "created_ns":
            return Cell(cell_text("created_ns", record.get("created_ns")), "number")
        return Cell(kind="empty")


class CampaignDelegate(CellDelegate):
    """Ticked rows take the selected fill and ink bar too (D7)."""

    def row_selected(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, selected: bool
    ) -> bool:
        model = self.view.model()
        if selected or not isinstance(model, CampaignModel):
            return selected
        record = model.record(index.row())
        return record is not None and record["id"] in model.chosen


class RowActions(QtWidgets.QWidget):
    """The Actions cell: Start, Pause or Resume (28 px) and the ⋯ ghost button, 6 px apart, right-aligned
    12 px inside the cell and centred above the row's rule. One widget per row slot, reused when campaigns
    move between rows."""

    def __init__(self, themes: ThemeManager) -> None:
        super().__init__()
        self.last = False  # the page's last row has no bottom rule
        self.row = FlexRow(self, gap=6, right=12)
        self.row.add_stretch()
        self.control = Button(themes, "Resume", icon="play", small=True)
        self.more = Button(themes, icon="more", variant="ghost", small=True, icon_size=16)
        self.more.setToolTip("Campaign actions")
        self.row.add(self.control)
        self.row.add(self.more)

    def relayout(self) -> None:
        place(self.row, 0, 0, self.width(), self.height() - (0 if self.last else 1))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()


class CampaignTable(PagedTable):
    """``.tablecard``: the campaign table (D6 attention order, D7 selection model).

    A plain click makes a row current; the checkbox, Ctrl+click, Shift+click (a range), Space and the header
    checkbox tick rows for Delete. Ticks are kept by campaign ID across refreshes.
    """

    action_requested = QtCore.Signal(str, str)
    choice_changed = QtCore.Signal(int)
    pending_changed = QtCore.Signal()

    def __init__(self, transport: Transport, themes: ThemeManager) -> None:
        model = CampaignModel()
        super().__init__(transport, "campaign-table", "campaigns", COLUMNS, model=model)
        self.include = ["study"]  # the Study column (34)
        self.campaign_model = model
        self.themes = themes
        self.label.hide()
        layout = self.layout()
        if layout is not None:
            layout.setSpacing(0)
        # Campaigns with a control request in flight: the one owner of "pending" (rows and card).
        self.pending: set[str] = set()
        self.chosen = self.campaign_model.chosen
        self.anchor: str | None = None
        self.attention_unsupported = False  # an engine image older than the attention order (D6)
        style_table(self.view, themes, LOOK, delegate_class=CampaignDelegate)
        header = self.view.horizontalHeader()
        assert isinstance(header, TableHeader)
        self.header = header
        header.check_toggled.connect(self.toggle_all)
        header.sectionClicked.connect(self.sort_column)
        self.sort_by, self.sort_descending = "attention", False
        header.setSortIndicator(2, QtCore.Qt.SortOrder.AscendingOrder)
        header.setSortIndicatorShown(True)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        self.row_widgets: dict[int, RowActions] = {}
        self.action_rows: dict[int, str] = {}
        self.model.pages_changed.connect(self.rebuild_actions)
        self.model.dataChanged.connect(self.update_actions)
        self.update_header_check()

    # ticking (D7)

    def toggle(self, row: int) -> None:
        record = self.model.record(row)
        if record is None:
            return
        if record["id"] in self.chosen:
            del self.chosen[record["id"]]
        else:
            self.chosen[record["id"]] = None
        self.anchor = record["id"]
        self.ticks_changed()

    def tick_range(self, row: int) -> None:
        rows = dict(self.model.records())
        anchor_row = next((r for r, rec in rows.items() if rec["id"] == self.anchor), None)
        if anchor_row is None:
            self.toggle(row)
            return
        for index in range(min(anchor_row, row), max(anchor_row, row) + 1):
            if index in rows:
                self.chosen[rows[index]["id"]] = None
        self.ticks_changed()

    def toggle_all(self) -> None:
        loaded = [record for _, record in self.model.records()]
        if loaded and all(record["id"] in self.chosen for record in loaded):
            for record in loaded:
                self.chosen.pop(record["id"], None)
        else:
            for record in loaded:
                self.chosen[record["id"]] = None
        self.ticks_changed()

    def ticks_changed(self) -> None:
        self.view.viewport().update()
        self.update_header_check()
        self.choice_changed.emit(len(self.chosen))

    def update_header_check(self) -> None:
        loaded = [record["id"] for _, record in self.model.records()]
        ticked = sum(identity in self.chosen for identity in loaded)
        self.header.set_check_state(
            QtCore.Qt.CheckState.Checked
            if loaded and ticked == len(loaded)
            else QtCore.Qt.CheckState.PartiallyChecked
            if ticked
            else QtCore.Qt.CheckState.Unchecked
        )

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.view.viewport() and event.type() in (
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonDblClick,
        ):
            assert isinstance(event, QtGui.QMouseEvent)
            index = self.view.indexAt(event.position().toPoint())
            if index.isValid() and event.button() == QtCore.Qt.MouseButton.LeftButton:
                modifiers = event.modifiers()
                if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.tick_range(index.row())
                    return True
                if index.column() == 0 or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                    self.toggle(index.row())
                    return True
        if (
            watched is self.view
            and event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() == QtCore.Qt.Key.Key_Space
        ):
            current = self.view.currentIndex()
            if current.isValid():
                self.toggle(current.row())
                return True
        return super().eventFilter(watched, event)

    def _received(self, tag: str, result: Any) -> None:
        if tag != self.tag:
            return
        super()._received(tag, result)
        self.update_header_check()  # loaded rows changed; ticks stay keyed by campaign ID

    def _failed(self, tag: str, message: str) -> None:
        if self.rejects_attention(tag, message):
            # An engine image older than the attention order: today's order, without an indicator.
            self.attention_unsupported = True
            self.sort_by = None
            self.header.setSortIndicatorShown(False)
            self.busy = False
            self.reset_page()
            return
        super()._failed(tag, message)

    def rejects_attention(self, tag: str, message: str) -> bool:
        """Whether a failure is the engine refusing the attention order (handled here, silently)."""
        return tag == self.tag and self.sort_by == "attention" and ATTENTION_REJECTED in message

    def chosen_in_order(self) -> list[str]:
        """The ticked campaigns in table order (the dialog lists them so), then ticked ones off the page."""
        shown = [record["id"] for _, record in self.model.records() if record["id"] in self.chosen]
        return shown + [identity for identity in self.chosen if identity not in shown]

    def clear_choice(self) -> None:
        self.chosen.clear()
        self.anchor = None
        self.ticks_changed()

    def forget(self, identities: list[str]) -> None:
        for identity in identities:
            self.chosen.pop(identity, None)
        self.ticks_changed()

    def sort_column(self, column: int) -> None:
        key = SORT_KEYS.get(column)
        if key == "attention" and self.attention_unsupported:
            key = "state"  # an older engine sorts the Status column by its name
        if key is None:
            # The header moves its indicator on any click: put it back where the order is.
            current = next((c for c, k in SORT_KEYS.items() if k == self.sort_by), None)
            if self.sort_by == "state":
                current = 2
            if current is None:
                self.header.setSortIndicatorShown(False)
            else:
                order = (
                    QtCore.Qt.SortOrder.DescendingOrder
                    if self.sort_descending
                    else QtCore.Qt.SortOrder.AscendingOrder
                )
                self.header.setSortIndicator(current, order)
            return
        self.sort_descending = not self.sort_descending if self.sort_by == key else False
        self.sort_by = key
        self.header.setSortIndicator(
            column,
            QtCore.Qt.SortOrder.DescendingOrder
            if self.sort_descending
            else QtCore.Qt.SortOrder.AscendingOrder,
        )
        self.header.setSortIndicatorShown(True)
        self.reset_page()

    # row actions

    def rebuild_actions(self) -> None:
        records = dict(self.model.records())
        for row in list(self.row_widgets):
            if row not in records or self.view.indexWidget(self.model.index(row, ACTIONS)) is None:
                self.view.setIndexWidget(self.model.index(row, ACTIONS), None)  # type: ignore[arg-type]
                del self.row_widgets[row]
        for row, record in records.items():
            if row not in self.row_widgets:
                self.row_widgets[row] = self.action_widget(row)
            if self.action_rows.get(row) != record["id"]:
                # A row's widgets are reused when another campaign moves into the row; the handlers read
                # the row's campaign at click time. Focus must not move to a different campaign.
                actions = self.row_widgets[row]
                actions.more.setAccessibleName("Actions for " + record["name"])
                for widget in (actions.control, actions.more):
                    if widget.hasFocus():
                        widget.clearFocus()
        self.action_rows = {row: record["id"] for row, record in records.items()}
        self.update_actions()
        self.update_header_check()

    def action_widget(self, row: int) -> RowActions:
        actions = RowActions(self.themes)
        actions.control.clicked.connect(lambda: self.trigger(self.action_rows[row]))
        actions.more.clicked.connect(lambda: self.open_menu(row, actions.more))
        self.view.setIndexWidget(self.model.index(row, ACTIONS), actions)
        return actions

    def open_menu(self, row: int, anchor: QtWidgets.QWidget) -> None:
        identity = self.action_rows.get(row)
        record = next((r for _, r in self.model.records() if r["id"] == identity), None)
        host = self.window().findChild(PopupHost)
        if record is None or identity is None or host is None:
            return
        menu = campaign_menu(
            self.themes,
            record["state"],
            identity in self.pending,
            lambda op: self.trigger(identity, op),
        )
        menu.popup(host, anchor)

    def update_actions(self, *args: Any) -> None:
        last = self.model.rowCount() - 1
        for row, record in self.model.records():
            actions = self.row_widgets.get(row)
            if actions is None:
                continue
            state = record["state"]
            operation = operation_for_state(state)
            pending = record["id"] in self.pending
            text, icon = control_look(state)
            control = actions.control
            if control.text() != text or control.glyph != icon or actions.last != (row == last):
                control.setText(text)
                control.set_icon(icon)
                actions.last = row == last
                actions.relayout()
            control.setEnabled(operation is not None and not pending)
            control.setVisible(state != "FINISHED")
            control.setAccessibleName(f"{text} {record['name']}")
            control.setToolTip(
                "Waiting for the engine"
                if pending
                else "Pause after active runs finish"
                if operation == "pause"
                else ""
            )
            actions.more.setEnabled(not pending)

    def trigger(self, identity: str, operation: str | None = None) -> None:
        if identity in self.pending:
            return
        for row, record in self.model.records():
            chosen = operation or operation_for_state(record["state"])
            if (
                record["id"] == identity
                and chosen is not None
                and record["state"] in PERMITTED.get(chosen, set())
            ):
                self.view.setCurrentIndex(self.model.index(row, 1))
                self.begin_control(identity)
                self.action_requested.emit(identity, chosen)
                return

    def begin_control(self, identity: str) -> bool:
        """Mark a control request for ``identity`` in flight; False when one already is."""
        if identity in self.pending:
            return False
        self.pending.add(identity)
        self.update_actions()
        self.pending_changed.emit()
        return True

    def finish_control(self, identity: str, state: str | None = None) -> None:
        self.pending.discard(identity)
        if state is not None:
            for _, record in self.model.records():
                if record["id"] == identity:
                    record["state"] = state
        self.update_actions()
        self.pending_changed.emit()
        self.refresh()


def campaign_menu(
    themes: ThemeManager,
    state: str,
    pending: bool,
    choose: Callable[[str], None],
    *,
    accounting: Callable[[], None] | None = None,
) -> Menu:
    """The ⋯ menu of a campaign row or card: [Attempt accounting,] Restart campaign, separator, Stop
    campaign (danger); a disabled item gives the campaign state as its reason."""
    menu = Menu(themes)
    if accounting is not None:
        menu.add_item("Attempt accounting", accounting, icon="list")
    for operation, text, icon, danger in (
        ("restart", "Restart campaign", "retry", False),
        ("stop", "Stop campaign", "cstop", True),
    ):
        if danger:
            menu.add_separator()
        allowed = state in PERMITTED[operation] and not pending
        reason = "" if allowed else "Waiting" if pending else humanize(state)
        menu.add_item(
            text,
            partial(choose, operation),
            icon=icon,
            enabled=allowed,
            reason=reason,
            danger=danger,
        )
    return menu


# ---------- the selected campaign's card ----------

EYEBROW = TextStyle(12, 500, 16)
CAMPAIGN_NAME = TextStyle(18, 600, 24, spacing_em=-0.005)
CHIP_NUMBER = TextStyle(13, 600, 18, tabular=True)
TOTAL = TextStyle(13, 600, 18, tabular=True)
STAT_VALUE = TextStyle(22, 600, 26, tabular=True)
STAT_KEY = TextStyle(12.5, 400, 18)
FAILURES_TIP = (
    "Startup, runtime, storage, engine-interruption and unresolved timing outcomes; left out of the "
    "study's statistics and counted."
)
ACCOUNTING = (
    ("executed", "Attempts", "All executions, including retries and discarded attempts."),
    ("rejections", "Timing rejected", "Attempts rejected by the campaign timing policy."),
    ("replacements", "Sequential replacements", "Sequential retries, not extra repetitions."),
    (
        "unresolved",
        "Unresolved",
        "Experiments without a timing-eligible result after retry.",
    ),
    ("not_done", "Not done", "Interrupted experiments waiting for a fresh execution."),
    (
        "discards",
        "Discards",
        "Discarded attempts whose partial data was removed (campaign Stop, experiment stop or "
        "restart).",
    ),
    (
        "no_movement",
        "No movement",
        "Final runs in which the activated explorer never moved the robot; reported.",
    ),
    (
        "stopped",
        "Stopped",
        "Runs stopped from the Monitor; data kept, left out of the study's statistics and counted.",
    ),
)


@dataclass(frozen=True, slots=True)
class Chip:
    word: str
    number: int | None = None

    def width(self) -> float:
        """``.ch``: 1 px border and 8 px padding each side; a number in 600 with 4 px after it."""
        number = CHIP_NUMBER.advance(f"{self.number:,}") + 4 if self.number is not None else 0
        return 18 + number + BODY_MEDIUM.advance(self.word)


def formula_chips(specification: dict[str, Any], catalog: dict[str, Any]) -> list[Chip]:
    """``.formula``: robots × profiles × worlds × spawns × repetitions. One robot, world or profile per robot
    (the same for every robot) shows its name; spawns and repetitions are always counted (01, 02)."""
    labels = {r["id"]: r.get("label", r["id"]) for r in catalog.get("robots", [])}
    world_labels = {w["id"]: w.get("label", w["id"]) for w in catalog.get("worlds", [])}
    robots: dict[str, list[dict[str, Any]]] = specification.get("robots", {})
    worlds: dict[str, list[dict[str, Any]]] = specification.get("worlds", {})
    chips: list[Chip] = []
    chips.append(
        Chip(labels.get(next(iter(robots)), next(iter(robots))))
        if len(robots) == 1
        else Chip("robots", len(robots))
    )
    profiles = [len(entries) for entries in robots.values()]
    names = {str(entries[0].get("name", "Default")) for entries in robots.values() if entries}
    if len(set(profiles)) == 1 and profiles[0] == 1 and len(names) == 1:
        chips.append(Chip(names.pop()))
    else:
        count = profiles[0] if len(set(profiles)) == 1 else sum(profiles)
        chips.append(Chip("profiles" if count != 1 else "profile", count))
    chips.append(
        Chip(world_labels.get(next(iter(worlds)), humanize(next(iter(worlds)))))
        if len(worlds) == 1
        else Chip("worlds", len(worlds))
    )
    spawns = [len(entries) for entries in worlds.values()]
    count = spawns[0] if len(set(spawns)) == 1 else sum(spawns)
    chips.append(Chip("spawns" if count != 1 else "spawn", count))
    repetitions = int(specification.get("repetitions", 1))
    chips.append(Chip("repetitions" if repetitions != 1 else "repetition", repetitions))
    return chips


def separator_width(sign: str) -> float:
    """``×`` or ``=`` between formula items, with the 6 px flex gaps on both sides."""
    return 12 + BODY.advance(sign)


def fit_chips(chips: list[Chip], room: float) -> list[Chip]:
    """The chips with their name chips (no number) elided, widest first, until the row fits ``room``."""
    fitted = list(chips)
    named = sorted(
        (index for index, chip in enumerate(fitted) if chip.number is None),
        key=lambda index: -fitted[index].width(),
    )
    for index in named:
        widths = sum(chip.width() for chip in fitted)
        overflow = widths + separator_width("×") * (len(fitted) - 1) - room
        if overflow <= 0:
            break
        chip = fitted[index]
        word = BODY_MEDIUM.elide(chip.word, max(0.0, BODY_MEDIUM.advance(chip.word) - overflow))
        fitted[index] = Chip(word)
    return fitted


@dataclass(frozen=True, slots=True)
class CampaignFacts:
    """What the card shows of ``campaign.status``."""

    name: str = "Select a campaign"
    eyebrow: str = "Selected campaign"
    state: str | None = None
    chips: tuple[Chip, ...] = ()
    count: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    resolved: int | None = None
    accounting: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Stat:
    """One of the card's numbers, the outcome bar's legend."""

    label: str
    value: int | None
    token: str | None  # the legend dot's colour; None draws the remaining hatch


def campaign_facts(
    campaign: dict[str, Any], counters: dict[str, Any], catalog: dict[str, Any]
) -> CampaignFacts:
    states, outcomes, attempts = counters["states"], counters["outcomes"], counters["attempts"]
    specification = campaign.get("specification")
    if isinstance(specification, str):
        specification = json.loads(specification)
    specification = specification if isinstance(specification, dict) else {}
    concurrency = int(specification.get("concurrency", 1))
    counts = {
        "completed": outcomes.get("completed", 0),
        "time_limit": outcomes.get("time_limit", 0),
        "failure": sum(n for key, n in outcomes.items() if report_exclusion(key) == "failure"),
        "no_movement": outcomes.get("no_movement", 0),
        "stopped": sum(n for key, n in outcomes.items() if key in STOPPED_OUTCOMES),
    }
    return CampaignFacts(
        name=campaign["name"],
        eyebrow="Selected campaign · "
        + (f"{concurrency} parallel workers" if concurrency > 1 else "sequential")
        + (
            f" · Study {study['id']}"
            if isinstance(study := specification.get("study"), dict)
            else ""
        ),
        state=campaign["state"],
        chips=tuple(formula_chips(specification, catalog)) if specification.get("robots") else (),
        count=int(campaign["count"]),
        counts=counts,
        resolved=sum(outcomes.values()),
        accounting={
            "executed": attempts["executed"],
            "rejections": attempts["timing_rejections"] or 0,
            "replacements": attempts["sequential_replacements"] or 0,
            "unresolved": states.get("UNRESOLVED", 0),
            "not_done": states.get("NOT_DONE", 0),
            "discards": attempts["discards"] or 0,
            "no_movement": counts["no_movement"],
            "stopped": counts["stopped"],
        },
    )


# ``.camp`` (CSS): padding 20 24 inside the 1 px border. Track rows are measured up from the card's bottom
# edge (20 px padding and the border): the rule 132 px up, the 92 px ring 113 px up, the numbers 100 px, their
# labels 72 px and the 8 px bar 42 px up; the numbers and the bar start 24 px after the ring.
CARD_LEFT = 25
TRACK_RULE, RING_TOP, STAT_VALUE_TOP, STAT_KEY_TOP, BAR_TOP = 132, 113, 100, 72, 42
TRACK_LEFT = CARD_LEFT + CAMPAIGN_RING.size + 24
STAT_GAP = 12


class CampaignCard(Card):
    """``.camp``: eyebrow, state badge, Start/Pause/Resume and ⋯; the name; the formula chips; then the
    ring, the outcome numbers (the bar's legend) and the outcome bar above a rule.

    ``pending_ids`` is the campaign table's set of campaigns with a control request in flight (its one
    owner); the card's controls wait while its campaign is in it.
    """

    action_requested = QtCore.Signal(str, str)

    def __init__(
        self,
        themes: ThemeManager,
        pending_ids: AbstractSet[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.facts = CampaignFacts()
        self.campaign_id: str | None = None
        self.pending_ids = pending_ids
        self.top = FlexRow(self, gap=8)
        self.eyebrow = TextLine(themes, self.facts.eyebrow, EYEBROW, "text.secondary")
        self.top.add(self.eyebrow)
        self.top.add_stretch()
        self.badge = StatusBadge(themes)
        self.control = Button(themes, "Pause", icon="pause", small=True)
        self.more = Button(themes, icon="more", variant="ghost", small=True, icon_size=16)
        self.more.setToolTip("Campaign actions")
        self.more.setAccessibleName("Campaign actions")
        for widget in (self.badge, self.control, self.more):
            self.top.add(widget)
        self.control.clicked.connect(self._control)
        self.more.clicked.connect(self._menu)
        self.name = TextLine(themes, self.facts.name, CAMPAIGN_NAME, "text.ink", self)
        self.ring = ProgressRing(themes, CAMPAIGN_RING, self)
        self.bar = OutcomeBar(themes, self)
        self.show_facts(self.facts)

    @property
    def pending(self) -> bool:
        return self.campaign_id is not None and self.campaign_id in self.pending_ids

    # data

    def reset(self, name: str) -> None:
        self.campaign_id = None
        self.show_facts(CampaignFacts(name=name))

    def update_campaign(
        self, campaign: dict[str, Any], counters: dict[str, Any], catalog: dict[str, Any]
    ) -> None:
        self.campaign_id = campaign["id"]
        self.show_facts(campaign_facts(campaign, counters, catalog))

    def show_facts(self, facts: CampaignFacts) -> None:
        self.facts = facts
        self.eyebrow.setText(facts.eyebrow)
        self.name.setText(facts.name)
        self.badge.set_look(badge_look("campaign", facts.state) if facts.state else None)
        state = facts.state or ""
        text, icon = control_look(state)
        self.control.setText(text)
        self.control.set_icon(icon)
        self.control.setVisible(bool(facts.state) and state != "FINISHED")
        self.more.setVisible(facts.state is not None)
        self.refresh_pending()
        self.top.relayout()
        share = None if not facts.count or facts.resolved is None else facts.resolved / facts.count
        self.ring.set_value(
            share, f"{facts.resolved:,} resolved" if facts.resolved is not None else ""
        )
        self.bar.set_counts(facts.counts, facts.count or 0)
        self.update()

    def refresh_pending(self) -> None:
        """Enable the controls unless the campaign's control request is in flight."""
        operation = operation_for_state(self.facts.state or "")
        self.control.setEnabled(operation is not None and not self.pending)

    def _control(self) -> None:
        operation = operation_for_state(self.facts.state or "")
        if self.campaign_id and operation:
            self.action_requested.emit(self.campaign_id, operation)

    def _menu(self) -> None:
        host = self.window().findChild(PopupHost)
        if host is None or not self.campaign_id or self.facts.state is None:
            return
        menu = campaign_menu(
            self.themes,
            self.facts.state,
            self.pending,
            partial(self._choose, self.campaign_id),
            accounting=self.show_accounting,
        )
        menu.popup(host, self.more)

    def _choose(self, identity: str, operation: str) -> None:
        """A menu choice, checked again against the campaign's state now (the menu may be old)."""
        if identity == self.campaign_id and (self.facts.state or "") in PERMITTED[operation]:
            self.action_requested.emit(identity, operation)

    def show_accounting(self) -> None:
        host = self.window().findChild(PopupHost)
        if host is None:
            return
        popup = AccountingPopup(self.themes, self.facts.accounting)
        size = popup.sizeHint()
        anchor = QtCore.QRect(
            self.more.mapTo(host.window_widget, QtCore.QPoint(0, 0)), self.more.size()
        )
        x = max(8, anchor.right() + 1 - size.width())
        rect = QtCore.QRect(x, anchor.bottom() + 1 + 4, size.width(), size.height())
        host.open_popup(popup, rect, opener=self.more, radius=8, delete_on_close=True)

    # geometry

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def css_changed(self) -> None:
        self.relayout()

    def relayout(self) -> None:
        width, height = self.box_width(), self.height()
        place(self.top, CARD_LEFT, 21, width - 2 * CARD_LEFT, 28)
        place(self.name, CARD_LEFT, 55, width - 2 * CARD_LEFT, 24)
        self.ring.move(CARD_LEFT, height - RING_TOP)
        place(self.bar, TRACK_LEFT, height - BAR_TOP, width - CARD_LEFT - TRACK_LEFT, 8)

    def stats(self) -> list[Stat]:
        """The numbers (D9: no movement and stopped only when there are any)."""
        counts, count = self.facts.counts, self.facts.count
        known = count is not None
        stats = [
            Stat("Completed", counts.get("completed") if known else None, "data.completed"),
            Stat("Time limits", counts.get("time_limit") if known else None, "data.timeLimit"),
            Stat("Failures", counts.get("failure") if known else None, "data.failure"),
        ]
        for key, label in (("no_movement", "No movement"), ("stopped", "Stopped")):
            if counts.get(key):
                stats.append(Stat(label, counts[key], "state.neutral"))
        remaining = None if count is None or self.facts.resolved is None else count
        if remaining is not None and self.facts.resolved is not None:
            remaining -= self.facts.resolved
        stats.append(Stat("Remaining", remaining, None))
        return stats

    def track_layout(self) -> list[tuple[Stat, float]]:
        """Each number with its column's x (``.stats``: equal columns 12 px apart)."""
        stats = self.stats()
        room = self.box_width() - CARD_LEFT - TRACK_LEFT
        column = (room - STAT_GAP * (len(stats) - 1)) / len(stats)
        return [
            (stat, TRACK_LEFT + index * (column + STAT_GAP)) for index, stat in enumerate(stats)
        ]

    def failures_icon(self) -> QtCore.QRectF | None:
        """The ``info`` icon after "Failures" (13 px, 6 px after the label)."""
        for stat, x in self.track_layout():
            if stat.label == "Failures":
                left = x + 14 + STAT_KEY.advance(stat.label) + 6
                return QtCore.QRectF(left, self.height() - STAT_KEY_TOP + 2.5, 13, 13)
        return None

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            icon = self.failures_icon()
            inside = icon is not None and icon.adjusted(-3, -3, 3, 3).contains(
                QtCore.QPointF(event.pos())
            )
            show_tooltip(self, event.globalPos(), FAILURES_TIP if inside else "")
            return True
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        self.paint_formula(painter)
        self.paint_track(painter)

    def paint_formula(self, painter: QtGui.QPainter) -> None:
        """Formula chips (24 px, 1 px line, radius 4) separated by × and closed by "= N experiments"."""
        theme = self.theme
        x, top = float(CARD_LEFT), 89.0
        total = plural(self.facts.count, "experiment") if self.facts.count is not None else ""
        tail = separator_width("=") + TOTAL.advance(total) if total else 0.0
        room = self.box_width() - 2 * CARD_LEFT - tail
        chips = fit_chips(list(self.facts.chips), room)
        for index, chip in enumerate(chips):
            if index:
                x += 12 + draw_text(
                    painter, x + 6, top + 3, "×", BODY, theme.color("text.tertiary")
                )
            width = chip.width()
            box = QtCore.QRectF(snap(x), top, snap(x + width) - snap(x), 24)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
            painter.setBrush(theme.color("surface.card"))
            painter.drawPath(rounded_path(box.adjusted(0.5, 0.5, -0.5, -0.5), 3.5))
            text_x = x + 9
            if chip.number is not None:
                number = f"{chip.number:,}"
                text_x += 4 + draw_text(
                    painter, text_x, top + 3, number, CHIP_NUMBER, theme.color("text.ink")
                )
            draw_text(painter, text_x, top + 3, chip.word, BODY_MEDIUM, theme.color("text.body"))
            x += width
        if chips and total:
            x += 12 + draw_text(painter, x + 6, top + 3, "=", BODY, theme.color("text.tertiary"))
            draw_text(painter, x, top + 3, total, TOTAL, theme.color("text.ink"))

    def paint_track(self, painter: QtGui.QPainter) -> None:
        """The rule, then the numbers with their legend dots (the ring and the bar are children)."""
        theme = self.theme
        height = self.height()
        rule_right = snap(self.box_width() - CARD_LEFT)
        painter.fillRect(
            QtCore.QRect(CARD_LEFT, height - TRACK_RULE, rule_right - CARD_LEFT, 1),
            theme.color("border.line"),
        )
        for stat, x in self.track_layout():
            text = "—" if stat.value is None else f"{stat.value:,}"
            draw_text(
                painter, x, height - STAT_VALUE_TOP, text, STAT_VALUE, theme.color("text.ink")
            )
            paint_legend_dot(painter, snap(x), height - STAT_KEY_TOP + 5, theme, stat.token)
            key_color = theme.color("text.secondary")
            draw_text(painter, x + 14, height - STAT_KEY_TOP, stat.label, STAT_KEY, key_color)
        icon = self.failures_icon()
        if icon is not None:
            origin = QtCore.QPointF(snap(icon.x()), snap(icon.y()))
            gui_icons.draw(painter, "info", theme.color("text.tertiary"), origin, 13)


class AccountingPopup(PopupCard):
    """Attempt accounting (the card's ⋯): the eight counters in a key-value list with their explanations
    as tooltips; menu-style card, 320 px."""

    def __init__(self, themes: ThemeManager, values: dict[str, int]) -> None:
        super().__init__(themes)
        self.list = KeyValueList(themes, last_open=True, parent=self)
        self.list.set_rows(
            [
                KeyValue(label, f"{values[key]:,}" if key in values else "—", tooltip=tip)
                for key, label, tip in ACCOUNTING
            ]
        )
        self.setAccessibleName("Attempt accounting")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(320, 2 * 9 + self.list.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.list.setGeometry(17, 9, self.width() - 34, self.list.height())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(
            rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7.5)
        )


# ---------- Live on the map ----------

TILE_LABEL = TextStyle(11.5, 400, 14)
TILE_VALUE = TextStyle(16, 600, 18, tabular=True)
TILE_UNIT = TextStyle(13, 400, 18)
WHO = TextStyle(13, 600, 18)
# ``.thumb .map``: at most 132×124 (three or four workers: 96×90), in the world's aspect.
WIDE_MAP, NARROW_MAP = (132.0, 124.0), (96.0, 90.0)


@dataclass(slots=True)
class LiveRun:
    """One worker's map state, fed by ``monitor.get`` at 1 Hz."""

    cursor: PathCursor = field(default_factory=PathCursor)
    layers: LiveLayers = field(default_factory=LiveLayers)
    revisions: dict[str, int] = field(default_factory=dict)
    pending: bool = False
    world_id: str = ""


def map_box(world: WorldMap | None, maximum: tuple[float, float]) -> tuple[int, int]:
    """A mini-map's size: the world's aspect fitted in ``maximum``, in whole pixels."""
    width, height = maximum
    if world is None:
        return max(1, round(width)), max(1, round(height))
    columns, rows = world.size
    scale = min(width / columns, height / rows)
    return max(1, round(columns * scale)), max(1, round(rows * scale))


class WorkerTile(Themed, CssBox):
    """``.wk``: the mini-map (up to 132×124 in the world's aspect) and three labelled values, then "W1
    TurtleBot4 Standard · rep. 3" with the attempt badge and the profile and spawn. ``narrow`` (three or four workers): a map of up
    to 96×90 with the values that fit below it (D10). The widget extends 1 px around the column for the map's
    ring; ``css_width`` includes that pixel on both sides."""

    clicked = QtCore.Signal(str)  # the attempt: open it on the Monitor

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.map = MiniMap(themes, parent=self)
        # The whole tile opens the run on the Monitor (map included).
        self.map.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusRadius", 8)
        self.setToolTip("Open this run on the Monitor")
        self.narrow = False
        self.run: dict[str, Any] = {}
        self.robot = ""
        self.values = worker_values({})
        self.look = badge_look("attempt", "STARTING")

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self._open()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
            QtCore.Qt.Key.Key_Space,
        ):
            self._open()
            return
        super().keyPressEvent(event)

    def _open(self) -> None:
        identity = (self.run.get("attempt") or {}).get("id")
        if identity:
            self.clicked.emit(str(identity))

    def set_run(self, run: dict[str, Any], robot: str, narrow: bool) -> None:
        self.run, self.robot, self.narrow = run, robot, narrow
        stopping = bool(run.get("stop_requested"))
        phase = run["attempt"]["phase"]
        self.look = (
            badge_look("campaign", "STOPPING", word="Stopping")
            if stopping
            else badge_look("attempt", phase)
        )
        self.values = worker_values(run.get("metrics") or {})
        experiment = run["experiment"]
        self.setAccessibleName(
            f"W{run['attempt']['slot'] + 1} {robot} repetition {experiment['repetition'] + 1}: "
            f"{experiment['profile_name']} · {experiment['spawn_name']}"
        )
        self.relayout()
        self.update()

    def column(self) -> tuple[float, float]:
        """The column's CSS left edge and width in this widget."""
        width = float(self.width()) if self.css_width is None else self.css_width
        return 1 + self.shift, width - 2

    def map_size(self) -> tuple[int, int]:
        _, width = self.column()
        if self.narrow:
            return map_box(self.map.world, (min(NARROW_MAP[0], width), NARROW_MAP[1]))
        return map_box(self.map.world, WIDE_MAP)

    def css_changed(self) -> None:
        self.relayout()

    def relayout(self) -> None:
        width, height = self.map_size()
        self.map.setGeometry(0, 0, width + 2, height + 2)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self.run:
            return
        painter = QtGui.QPainter(self)
        map_width, map_height = self.map_size()
        if self.narrow:
            self.paint_values_below(painter, 1 + map_height + 8)
            line1 = 1 + map_height + 8 + 18 + 8
        else:
            self.paint_values_beside(painter, map_width)
            line1 = 1 + map_height + 10
        self.paint_identity(painter, line1)

    def paint_values_beside(self, painter: QtGui.QPainter, map_width: int) -> None:
        """``.tstats``: label 11.5 px above the 16 px value (unit 13 px), 44 px apart, 14 px after the map."""
        theme = self.theme
        left, _ = self.column()
        x = left + map_width + 14
        for index, (label, value, unit) in enumerate(self.values):
            top = 1 + 0.5 + index * 44
            draw_text(painter, x, top, label, TILE_LABEL, theme.color("text.secondary"))
            # The value shares its line with the 13 px strut: baseline 30 px below the label's top.
            value_top = top + 30 - TILE_VALUE.baseline(0)
            advance = draw_text(painter, x, value_top, value, TILE_VALUE, theme.color("text.ink"))
            if unit:
                unit_top = top + 30 - TILE_UNIT.baseline(0)
                unit_color = theme.color("text.tertiary")
                draw_text(painter, x + advance + 2, unit_top, unit, TILE_UNIT, unit_color)

    def paint_values_below(self, painter: QtGui.QPainter, top: float) -> None:
        """Narrow tiles: the values in one line, 10 px apart, as many as fit the column."""
        theme = self.theme
        left, width = self.column()
        x = left
        for _, value, unit in self.values:
            size = BODY_MEDIUM.advance(value) + (2 + BODY.advance(unit) if unit else 0)
            if x + size > left + width:
                break
            x += draw_text(painter, x, top, value, BODY_MEDIUM, theme.color("text.ink"))
            if unit:
                x += 2 + draw_text(painter, x + 2, top, unit, BODY, theme.color("text.tertiary"))
            x += 10

    def paint_identity(self, painter: QtGui.QPainter, line1: float) -> None:
        """The worker tag ("W1": mono 11 px 600, 1 px line, radius 3, padding 3 4), the robot (elided before
        the badge), "· rep. N", the attempt badge at the right; the profile and spawn below."""
        theme = self.theme
        left, width = self.column()
        right = left + width
        attempt, experiment = self.run["attempt"], self.run["experiment"]
        tag_width = paint_worker_tag(painter, left, line1 + 1.5, f"W{attempt['slot'] + 1}", theme)
        badge_left = right - self.look.width if self.look is not None else right
        x = left + tag_width + 6
        repetition = f"· rep. {experiment['repetition'] + 1}"
        room = badge_left - 6 - x
        rep_width = 6 + BODY.advance(repetition)
        robot = WHO.elide(self.robot, max(0.0, room - rep_width))
        x += draw_text(painter, x, line1 + 2, robot, WHO, theme.color("text.ink")) + 6
        if x + BODY.advance(repetition) <= badge_left - 6:
            draw_text(painter, x, line1 + 2, repetition, BODY, theme.color("text.tertiary"))
        if self.look is not None:
            paint_badge(painter, badge_left, line1, self.look, theme)
        context = f"{experiment['profile_name']} · {experiment['spawn_name']}"
        context_color = theme.color("text.secondary")
        draw_text(painter, left, line1 + 25, BODY.elide(context, width), BODY, context_color)


class LiveCard(Card):
    """``.live``: "Live on the map" with "Open Monitor →" and one column per active worker in slot order (two
    at 1440; three or four narrower, D10; "No active workers" when idle). Each worker's map follows
    ``monitor.get`` at 1 Hz while the card is visible, one request in flight per worker."""

    open_monitor = QtCore.Signal()
    open_worker = QtCore.Signal(str)  # a tile was clicked: the Monitor with that run selected

    def __init__(
        self, themes: ThemeManager, transport: Transport, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(themes, parent)
        self.transport = transport
        self.title = TextLine(themes, "Live on the map", CARD_TITLE, "text.ink", self)
        self.link = LinkButton(themes, "Open Monitor", parent=self)
        self.link.clicked.connect(self.open_monitor)
        self.tiles: dict[str, WorkerTile] = {}
        self.runs: list[dict[str, Any]] = []
        self.live: dict[str, LiveRun] = {}
        self.worlds: dict[str, WorldMap] = {}
        self.world_requests: set[str] = set()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.poll)

    # data

    def set_runs(self, active: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
        # Columns follow the worker slot, so a worker keeps its column across attempts.
        self.runs = sorted(active, key=lambda run: int(run["attempt"]["slot"]))
        identities = {run["attempt"]["id"] for run in active}
        for identity in set(self.tiles) - identities:
            self.tiles.pop(identity).deleteLater()
            self.live.pop(identity, None)
        labels = {r["id"]: r.get("label", r["id"]) for r in catalog.get("robots", [])}
        narrow = len(active) > 2
        for run in self.runs:
            identity = run["attempt"]["id"]
            experiment = run["experiment"]
            if identity not in self.tiles:
                self.tiles[identity] = WorkerTile(self.themes, self)
                self.tiles[identity].clicked.connect(self.open_worker)
                self.tiles[identity].show()
                self.live[identity] = LiveRun(world_id=experiment["world_id"])
            tile = self.tiles[identity]
            live = self.live[identity]
            world = self.worlds.get(experiment["world_id"])
            tile.map.set_world(world)
            if world is None:
                self.request_world(experiment["world_id"])
            specification = experiment.get("specification")
            if isinstance(specification, str):
                specification = json.loads(specification)
            spawn = (specification or {}).get("spawn") if isinstance(specification, dict) else None
            if isinstance(spawn, dict):
                live.layers.spawn = (float(spawn["x"]), float(spawn["y"]), float(spawn["yaw"]))
            tile.map.set_layers(live.layers)
            tile.set_run(run, labels.get(experiment["robot_id"], experiment["robot_id"]), narrow)
        self.relayout()
        self.update()

    def request_world(self, world_id: str) -> None:
        if world_id not in self.world_requests:
            self.world_requests.add(world_id)
            self.transport.submit("tile-map:" + world_id, "spawns.map", {"world_id": world_id})

    def poll(self) -> None:
        if not self.isVisible():
            return
        for run in self.runs:
            self.request(run["attempt"]["id"])

    def request(self, identity: str) -> None:
        live = self.live.get(identity)
        if live is None or live.pending:
            return
        live.pending = True
        self.transport.submit(
            "tile-monitor:" + identity,
            "monitor.get",
            {
                "attempt_id": identity,
                "after_sequence": live.cursor.applied,
                "layers": list(DRAWN_LAYERS),
                "revisions": {
                    name: live.revisions[name] for name in DRAWN_LAYERS if name in live.revisions
                },
            },
        )

    def received(self, tag: str, result: Any) -> bool:
        """Handle a reply for this card; False when the tag is not one of its own."""
        if tag.startswith("tile-map:"):
            world_id = tag.removeprefix("tile-map:")
            try:
                self.worlds[world_id] = WorldMap.from_reply(result)
            except (KeyError, TypeError, ValueError):
                self.world_requests.discard(world_id)  # asked again at the next status update
                return True
            for identity, tile in self.tiles.items():
                if self.live[identity].world_id == world_id:
                    tile.map.set_world(self.worlds[world_id])
                    tile.relayout()
            return True
        if not tag.startswith("tile-monitor:"):
            return False
        identity = tag.removeprefix("tile-monitor:")
        live = self.live.get(identity)
        if live is None:
            return True
        live.pending = False
        if "history" not in result:
            return True
        applied = live.cursor.applied
        try:
            self.apply(identity, live, result)
        except (KeyError, TypeError, ValueError):
            return True
        # Catch up page by page, but only while pages advance: the engine answers from the worker's last
        # stored reply, which may hold nothing after this cursor until the worker answers again.
        if applied < live.cursor.applied < result.get("last_sequence", 0):
            self.request(identity)
        return True

    def failed(self, tag: str, error: str) -> bool:
        if tag.startswith("tile-map:"):
            self.world_requests.discard(tag.removeprefix("tile-map:"))
            return True
        if tag.startswith("tile-monitor:"):
            live = self.live.get(tag.removeprefix("tile-monitor:"))
            if live is not None:
                live.pending = False
            return True  # the tile keeps its last map; the run's own state shows in its badge
        return False

    def apply(self, identity: str, live: LiveRun, result: dict[str, Any]) -> None:
        world = self.worlds.get(live.world_id)
        apply_monitor_reply(live.layers, live.cursor, live.revisions, result, world)
        tile = self.tiles.get(identity)
        if tile is not None:
            tile.map.set_layers(live.layers)

    # visibility drives the polling (D10)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.timer.start()
        self.poll()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self.timer.stop()

    # geometry (CSS ``.live``: padding 20 24; head 24 px; workers 12 px below)

    def content(self) -> tuple[float, float]:
        """The CSS content box's left edge and width (padding 20 24 inside the 1 px border)."""
        return self.shift + 25, self.box_width() - 50

    def css_changed(self) -> None:
        self.relayout()

    def relayout(self) -> None:
        left, room = self.content()
        link = self.link.natural_width()
        place(self.title, left, 23, max(0.0, room - link - 16), 20)
        place(self.link, left + room - link, 24, link, 18)
        count = len(self.runs)
        if not count:
            return
        column = (room - 16 * (count - 1)) / count
        height = self.height() - 57 - 20
        for index, run in enumerate(self.runs):
            # ``.wk + .wk``: a 1 px rule and 16 px padding before the column's content.
            x = left + index * (column + 16) + (17 if index else 0)
            inner = column - (17 if index else 0)
            place(self.tiles[run["attempt"]["id"]], x - 1, 56, inner + 2, height + 2)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        count = len(self.runs)
        left, room = self.content()
        if not count:
            text = "No active workers"
            x = left + (room - BODY.advance(text)) / 2
            top = 57 + (self.height() - 57 - 21 - 18) / 2
            draw_text(painter, x, top, text, BODY, self.color("text.secondary"))
            return
        column = (room - 16 * (count - 1)) / count
        for index in range(1, count):
            x = snap(left + index * (column + 16))
            painter.fillRect(
                QtCore.QRect(x, 57, 1, self.height() - 57 - 21), self.color("border.line")
            )


class OverviewRow(QtWidgets.QWidget):
    """``.overview``: the campaign card (``flex: 1.5``) and Live on the map (``flex: 1``), 16 px apart, 254 px
    high. Both flex from a zero basis, so each keeps its 50 px of padding and border and the rest of the row
    is shared 1.5 : 1 (815.6 and 560.4 px at 1440)."""

    def __init__(self, card: CampaignCard, live: LiveCard) -> None:
        super().__init__()
        self.card, self.live = card, live
        card.setParent(self)
        live.setParent(self)
        self.setFixedHeight(254)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        share = max(0.0, self.width() - 16 - 2 * 50) / 2.5
        card = 50 + 1.5 * share
        place(self.card, 0, 0, card, self.height())
        place(self.live, card + 16, 0, self.width() - card - 16, self.height())


class TableCard(Card):
    """``.tablecard``: a card the campaign table fills inside its 1 px border."""

    def __init__(self, themes: ThemeManager, table: QtWidgets.QWidget) -> None:
        super().__init__(themes)
        self.table = table
        table.setParent(self)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.table.setGeometry(1, 1, self.width() - 2, self.height() - 2)


# ---------- dialogs ----------


def mebibytes(count: float) -> str:
    return f"{count / 2**20:.1f} MiB"


def confirm_deletion(
    modal: ModalHost, themes: ThemeManager, campaigns: list[dict[str, Any]]
) -> list[str]:
    """02: ask before deleting; returns the IDs to delete (only deletable ones), or [] when cancelled."""
    deletable = [c for c in campaigns if c["deletable"]]
    blocked = [c for c in campaigns if not c["deletable"]]
    rows = [
        ListRow(
            c["name"] or c["campaign_id"][:8],
            c["campaign_id"][:8],
            f"{plural(c['attempts'], 'run')} · at least {mebibytes(c['recorded_bytes'])}",
        )
        for c in deletable
    ] + [
        ListRow(c["name"] or c["campaign_id"][:8], c["campaign_id"][:8], "", reason=c["reason"])
        for c in blocked
    ]
    if not deletable:
        card = Consequences(themes, "None of the selected campaigns can be deleted now")
        card.add_items(rows)
        card.add_button("Close", False, default=True)
        modal.exec_card(card, escape_result=False)
        return []
    count = len(deletable)
    noun = "campaign" if count == 1 else "campaigns"
    card = Consequences(themes, f"Delete {count} {noun} and all their results?")
    summary = (
        f"{plural(sum(c['experiments'] for c in deletable), 'experiment')} and "
        f"{plural(sum(c['attempts'] for c in deletable), 'run')}; at least "
    )
    card.add_sentence(
        [
            Run(summary),
            Run(mebibytes(sum(c["recorded_bytes"] for c in deletable)), BOLD, "text.ink"),
            Run(" of recorded files."),
        ]
    )
    card.add_items(rows)
    card.add_footnote(
        "This cannot be undone. Profiles and spawn poses stay in the library.", ruled=False
    )
    card.add_button("Cancel", False, default=True)
    card.add_button(f"Delete {count} {noun}", True, variant="danger-solid", icon="trash")
    accepted = modal.exec_card(card, escape_result=False)
    return [c["campaign_id"] for c in deletable] if accepted else []


def confirm_stop(modal: ModalHost, themes: ThemeManager, name: str) -> bool:
    """Stop a campaign (in the style of 12)."""
    card = Consequences(themes, "Stop this campaign?")
    card.add_context(name)
    card.add_bullets(
        [
            bullet("Every active worker stops; its partial measurements are discarded."),
            bullet("Remaining experiments are left Not done."),
            bullet("Completed results remain available."),
        ]
    )
    card.add_button("Cancel", False, default=True)
    card.add_button("Stop campaign", True, variant="danger-solid", icon="stop")
    return bool(modal.exec_card(card, escape_result=False))


def confirm_restart(modal: ModalHost, themes: ThemeManager, name: str) -> bool:
    """Restart a campaign as a copy (in the style of 12)."""
    card = Consequences(themes, "Restart this campaign as a copy?")
    card.add_context(name)
    card.add_bullets(
        [
            bullet("A new copy runs with the same accepted settings."),
            bullet("This campaign and its results stay unchanged."),
            bullet(
                "The copy starts now when no other campaign is running; otherwise it runs next "
                "in the queue."
            ),
        ]
    )
    card.add_button("Cancel", False, default=True)
    card.add_button("Restart campaign", True, variant="primary", icon="retry")
    return bool(modal.exec_card(card, escape_result=False))
