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

"""Results, Runs tab (22): every final run of the selected study across its campaigns (RESULTS_STUDY.md decisions
2 and 3; RESULTS_RESEARCH RD-37).

One row per final run (``results.query`` table ``runs``, ``is_final``); a run whose experiment had earlier
attempts shows "N attempts", and expanding it inserts those attempts as indented, muted rows with their reason
(read on demand by ``experiment_id``). Filters by campaign, robot, world, profile and outcome; the default columns
of decision 3 and every other scalar record field from Columns (remembered per viewer); sorting by any shown
column. The footer counts the filter's final runs by report class and its earlier attempts
(``include: ["counts"]``). Censored values read "> L" and "not reached", never as observed times; unavailable
values "—".
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_cells import (
    CELL_ROLE,
    Align,
    Cell,
    CellKind,
    SUFFIX_ROLE,
    CellDelegate,
    ColumnStyle,
    TableFooter,
    TableHeader,
    TableLook,
    apply_columns,
    style_table,
)
from .gui_controls import BadgeLook, KeySelect, badge_look
from .gui_menus import INSET, ITEM_HEIGHT, ITEM_TEXT, SEPARATOR_HEIGHT
from .gui_style import duration, fixed, humanize
from .gui_support import PagedTable, PagedTableModel, Transport
from .gui_theme import ThemeManager, paint_check, snap
from .gui_widgets import (
    Button,
    Card,
    FlexRow,
    PopupHost,
    TextStyle,
    Themed,
    draw_text,
    rounded_path,
)

REPORTED = "reported"
OUTCOMES = (
    "completed",
    "time_limit",
    "no_movement",
    "startup_failure",
    "runtime_failure",
    "manual_stop",
    "storage_failure",
    "timing_rejection",
)
# The outcome word of a completed run that the coverage stop ended (completion_source "coverage_stop";
# coverage-stop-v2, 99.9 %).
COVERAGE_STOP_WORD = "Completed · 99.9 % coverage stop"
FOOT = TextStyle(12.5, 400, 18)
FOOT_COUNT = TextStyle(12.5, 600, 18)
NESTED_HEIGHT = 36
ROW_HEIGHT = 40


@dataclass(frozen=True, slots=True)
class Column:
    key: str
    title: str
    style: ColumnStyle
    sort: str | None  # the record field the engine sorts by (None: not sortable)
    suffix: str = ""  # faint text after the title ("C(L) v2")


def _style(kind: CellKind, width: int | None, *, align: Align | None = None) -> ColumnStyle:
    return ColumnStyle(kind, align=align, padding_left=10, padding_right=10, width=width)


DEFAULT_COLUMNS = (
    Column(
        "campaign", "Campaign", _style("text", 212), None
    ),  # the ID would order the names at random
    Column("robot", "Robot", _style("name", 166), "robot_id"),
    Column("world_spawn", "World · spawn", _style("text", 150), "world_id"),
    Column("profile", "Profile", _style("text", 116), "profile_id"),
    Column("repetition", "Rep.", _style("number", 50), "repetition"),
    Column("outcome", "Outcome", _style("badge", 250), "outcome"),
    Column("limit", "L", _style("number", 66), "time_limit_s"),
    Column("completion", "Completion", _style("number", 96, align="right"), "completion_s"),
    Column("t90", "T90", _style("number", 96, align="right"), "t90_s"),
    Column("coverage", "C(L)", _style("number", 82), "coverage_at_limit_pct", "v2"),
    Column("rtf", "Min RTF", _style("number", 96, align="right"), "rtf_min_5s"),
)
# Fields the default columns already show, and fields that are lists or identities nobody reads in a table.
SHOWN_FIELDS = frozenset(
    {
        "campaign_id",
        "robot_id",
        "world_id",
        "spawn_id",
        "profile_id",
        "repetition",
        "outcome",
        "time_limit_s",
        "completion_s",
        "t90_s",
        "coverage_at_limit_pct",
        "rtf_min_5s",
        "experiment_attempts",
    }
)
# Read-time fields (``run_record.read_time_fields``, but ``is_final``) and the attempt count: the engine does not
# sort by them.
UNSORTED_FIELDS = frozenset(
    {"replaced_reason", "report_class", "superseded_by_campaign_id", "experiment_attempts"}
)
EXTRA_WIDTH = 120
COLUMNS_KEY = "results/runColumns"  # QSettings: the viewer's shown columns, [[key, kind], ...]
GROUP_TEXT = TextStyle(12, 600, 18)
MENU_WIDTH, GROUP_HEIGHT, MENU_HEIGHT = 272, 28, 480


def run_label(record: Mapping[str, Any], name: Callable[[str, Any], str]) -> str:
    """ "Bookstore · Low Threshold · b3": a run by world, profile and block (the chips and the Files root, 26 and
    30); ``name(kind, id)`` gives the display names."""
    repetition = record.get("repetition")
    parts = (
        name("world", record.get("world_id")),
        name("profile", record.get("profile_id")),
        f"b{repetition + 1}" if isinstance(repetition, int) else "",
    )
    return " · ".join(part for part in parts if part)


@dataclass
class Names:
    """Labels the table shows instead of identities. An identity not known yet reads "Default" when it is a
    robot's or world's default (``default-<owner>``) and is humanized otherwise."""

    campaigns: dict[str, str]
    robots: dict[str, str]
    worlds: dict[str, str]
    spawns: dict[str, str]
    profiles: dict[str, str]

    def get(self, table: Mapping[str, str], key: Any) -> str:
        if key is None:
            return "—"
        identity = str(key)
        return table.get(identity) or (
            "Default" if identity.startswith("default-") else humanize(identity)
        )


def outcome_look(record: Mapping[str, Any], *, muted: bool = False) -> BadgeLook | None:
    """The outcome badge of a run record; a completed run that the coverage stop ended
    (``completion_source``) says so in its word."""
    outcome = record.get("outcome")
    stopped = outcome == "completed" and record.get("completion_source") == "coverage_stop"
    word = COVERAGE_STOP_WORD if stopped else None
    return badge_look("outcome", outcome, word=word, muted=muted)


def excluded(record: Mapping[str, Any]) -> bool:
    """A final run outside the reported population (``reported-outcomes-v2``)."""
    return record.get("report_class") not in (REPORTED, None, "not_final")


def extra_title(field: str) -> str:
    """A record field's column title: its name in words, the unit suffix kept ("Explorer cpu s")."""
    return humanize(field)


def extra_column(field: str, numeric: bool) -> Column:
    """A record field's column: 120 px, numbers right-aligned; sortable unless a read-time field."""
    style = _style("number" if numeric else "text", EXTRA_WIDTH)
    return Column(field, extra_title(field), style, None if field in UNSORTED_FIELDS else field)


def record_fields(records: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    """The scalar fields of ``records`` the default columns do not show, by name, each with whether it holds
    numbers; a field that holds a list or an object in any record is left out."""
    numeric: dict[str, bool] = {}
    nested: set[str] = set()
    for record in records:
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                nested.add(key)
            elif key not in SHOWN_FIELDS:
                number = isinstance(value, (int, float)) and not isinstance(value, bool)
                numeric[key] = numeric.get(key, False) or number
    return {key: numeric[key] for key in sorted(numeric) if key not in nested}


def ordered(columns: Iterable[Column]) -> list[Column]:
    """The default columns in their order, then the record fields in the order they were added."""
    chosen = list(columns)
    keys = {c.key for c in chosen}
    defaults = [c for c in DEFAULT_COLUMNS if c.key in keys]
    return defaults + [c for c in chosen if c.key not in {d.key for d in DEFAULT_COLUMNS}]


def load_columns() -> list[Column]:
    """The viewer's columns (QSettings), else the default columns."""
    stored = QtCore.QSettings().value(COLUMNS_KEY)
    try:
        entries = json.loads(str(stored)) if stored is not None else None
    except ValueError:
        entries = None
    if not isinstance(entries, list):
        return list(DEFAULT_COLUMNS)
    defaults = {c.key: c for c in DEFAULT_COLUMNS}
    columns: dict[str, Column] = {}
    for entry in entries:
        if not (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str)):
            continue
        key, kind = entry
        if key in defaults:
            columns[key] = defaults[key]
        elif key not in SHOWN_FIELDS:
            columns[key] = extra_column(key, kind == "number")
    return ordered(columns.values()) or list(DEFAULT_COLUMNS)


def save_columns(columns: Iterable[Column]) -> None:
    QtCore.QSettings().setValue(COLUMNS_KEY, json.dumps([[c.key, c.style.kind] for c in columns]))


def scalar_text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}" if math.isfinite(value) else "—"
    return str(value)


class RunsModel(PagedTableModel):
    """Pages of final runs with design cells."""

    def __init__(self, names: Names) -> None:
        self.shown: list[Column] = list(DEFAULT_COLUMNS)
        super().__init__([(c.key, c.title) for c in self.shown])
        self.run_names = names

    def set_shown(self, columns: list[Column]) -> None:
        self.shown = columns
        self.set_columns([(c.key, c.title) for c in columns])

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == CELL_ROLE and index.isValid():
            record = self.record(index.row())
            if record is None:
                return Cell("…") if index.column() == 0 else Cell(kind="empty")
            return run_cell(record, self.shown[index.column()].key, self.run_names)
        return super().data(index, role)

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = 0) -> Any:
        if role == SUFFIX_ROLE:
            return self.shown[section].suffix if 0 <= section < len(self.shown) else None
        return super().headerData(section, orientation, role)


def run_cell(record: Mapping[str, Any], key: str, names: Names) -> Cell:
    """What one cell of a final run shows (decision 3)."""
    out = excluded(record)
    token = "text.secondary" if out else None
    missing = Cell("—", "number", token="text.tertiary")
    if key == "campaign":
        return Cell(names.get(names.campaigns, record.get("campaign_id")), token="text.secondary")
    if key == "robot":
        return Cell(
            names.get(names.robots, record.get("robot_id")), "text" if out else "name", token=token
        )
    if key == "world_spawn":
        world = names.get(names.worlds, record.get("world_id"))
        spawn = names.get(names.spawns, record.get("spawn_id"))
        return Cell(f"{world} · {spawn}", token=token, faint="·")
    if key == "profile":
        return Cell(names.get(names.profiles, record.get("profile_id")), token=token)
    if key == "repetition":
        value = record.get("repetition")
        return Cell(str(value + 1), "number", token=token) if isinstance(value, int) else missing
    if key == "outcome":
        look = outcome_look(record, muted=out)
        attempts = record.get("experiment_attempts")
        if look is None:
            return Cell(humanize(str(record.get("outcome") or "—")), token=token)
        if isinstance(attempts, int) and attempts > 1:
            return Cell(kind="badge", badge=look, note=f"{attempts} attempts", note_kind="attempts")
        note = (
            "missing" if str(record.get("report_class", "")).startswith("missing") else "excluded"
        )
        return Cell(kind="badge", badge=look, note=note if out else "")
    if key == "limit":
        limit = record.get("time_limit_s")
        return Cell(duration(limit), "number", token=token) if limit else missing
    if key in ("completion", "t90"):
        prefix = "completion" if key == "completion" else "t90"
        value, event = record.get(f"{prefix}_s"), record.get(f"{prefix}_event")
        if event and isinstance(value, (int, float)):
            return Cell(duration(value), "number", token=token)
        # RD-8: a run that never moved never reaches (R = L)
        never = record.get("outcome") == "no_movement"
        if (event is False or never) and record.get("report_class") == REPORTED:
            if key == "completion":
                return Cell("> L", "name", token="state.warn")  # ``.cens``: right-censored at L
            return Cell("not reached", token="text.secondary")
        return missing
    if key == "coverage":
        text = fixed(record.get("coverage_at_limit_pct"), 1)
        return Cell(text, "number", unit="%", token=token) if text else missing
    if key == "rtf":
        text = fixed(record.get("rtf_min_5s"), 2)
        if not text:
            return missing
        if record.get("low_rtf"):
            return Cell(text, "name", token="state.warn", note="low", note_kind="low")
        return Cell(text, "number", token=token)
    value = record.get(key)
    return Cell(
        scalar_text(value), "number" if isinstance(value, (int, float)) else "text", token=token
    )


def nested_cell(record: Mapping[str, Any], key: str) -> Cell:
    """An earlier attempt (decision 2): muted badge with its reason; values the run record does not hold for it
    as "—"."""
    missing = Cell("—", "number", token="text.tertiary")
    if key == "outcome":
        look = badge_look("outcome", record.get("outcome"), muted=True)
        reason = "discarded" if record.get("outcome") == "manual_discard" else "replaced"
        if look is None:
            return Cell(humanize(str(record.get("outcome") or "—")), token="text.secondary")
        return Cell(kind="badge", badge=look, note=reason)
    if key == "rtf":
        text = fixed(record.get("rtf_min_5s"), 2)
        if not text:
            return missing
        if record.get("low_rtf"):
            return Cell(text, "name", token="state.warn", note="low", note_kind="low")
        return Cell(text, "number", token="text.secondary")
    if key in ("limit", "completion", "t90", "coverage"):
        return missing
    return Cell(kind="empty")


class NestedRuns(QtCore.QAbstractProxyModel):
    """The final runs with the earlier attempts of expanded experiments inserted below them."""

    def __init__(self, source: RunsModel) -> None:
        super().__init__()
        self.setSourceModel(source)
        self.runs = source
        self.earlier: dict[
            str, list[dict[str, Any]]
        ] = {}  # experiment -> earlier attempts, oldest first
        self.open: set[str] = set()
        self.layout: list[tuple[int, int]] = []  # proxy row -> (source row, -1 or child index)
        for signal in (
            source.modelReset,
            source.rowsInserted,
            source.rowsRemoved,
            source.layoutChanged,
        ):
            signal.connect(self.rebuild)
        source.dataChanged.connect(self._source_changed)
        self.rebuild()

    def rebuild(self, *_: Any) -> None:
        self.beginResetModel()
        layout: list[tuple[int, int]] = []
        for row in range(self.runs.rowCount()):
            layout.append((row, -1))
            record = self.runs.record(row)
            experiment = str(record.get("experiment_id")) if record else ""
            if record and experiment in self.open:
                layout.extend(
                    (row, child) for child in range(len(self.earlier.get(experiment, [])))
                )
        self.layout = layout
        self.endResetModel()

    def _source_changed(self, *_: Any) -> None:
        expected = sum(
            1
            + (
                len(self.earlier.get(str(r.get("experiment_id")), []))
                if r and str(r.get("experiment_id")) in self.open
                else 0
            )
            for r in (self.runs.record(row) for row in range(self.runs.rowCount()))
        )
        if expected != len(self.layout):
            self.rebuild()
        elif self.layout:
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(self.layout) - 1, self.columnCount() - 1)
            )

    def toggle(self, experiment: str, attempts: list[dict[str, Any]] | None = None) -> None:
        if attempts is not None:
            self.earlier[experiment] = attempts
        if experiment in self.open:
            self.open.discard(experiment)
        else:
            self.open.add(experiment)
        self.rebuild()

    # mapping

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self.layout)

    def columnCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex()
    ) -> int:
        return self.runs.columnCount()

    def index(
        self,
        row: int,
        column: int,
        parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> QtCore.QModelIndex:
        if (
            parent.isValid()
            or not 0 <= row < len(self.layout)
            or not 0 <= column < self.columnCount()
        ):
            return QtCore.QModelIndex()
        return self.createIndex(row, column)

    def parent(self, child: Any = None) -> Any:  # a flat table
        return QtCore.QModelIndex()

    def mapToSource(
        self, proxy: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.QModelIndex:
        if not proxy.isValid() or proxy.row() >= len(self.layout):
            return QtCore.QModelIndex()
        row, child = self.layout[proxy.row()]
        return self.runs.index(row, proxy.column()) if child < 0 else QtCore.QModelIndex()

    def mapFromSource(
        self, source: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.QModelIndex:
        if not source.isValid():
            return QtCore.QModelIndex()
        for row, entry in enumerate(self.layout):
            if entry == (source.row(), -1):
                return self.index(row, source.column())
        return QtCore.QModelIndex()

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = 0) -> Any:
        return self.runs.headerData(section, orientation, role)

    def nested(self, row: int) -> dict[str, Any] | None:
        """The earlier attempt shown on ``row``, or None for a final run."""
        if not 0 <= row < len(self.layout):
            return None
        source, child = self.layout[row]
        if child < 0:
            return None
        record = self.runs.record(source)
        experiment = str(record.get("experiment_id")) if record else ""
        attempts = self.earlier.get(experiment, [])
        return attempts[child] if child < len(attempts) else None

    def record(self, row: int) -> dict[str, Any] | None:
        """The record shown on ``row`` (a final run or an earlier attempt)."""
        if not 0 <= row < len(self.layout):
            return None
        source, child = self.layout[row]
        return self.runs.record(source) if child < 0 else self.nested(row)

    def data(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, role: int = 0) -> Any:
        if not index.isValid():
            return None
        attempt = self.nested(index.row())
        if attempt is None:
            return self.runs.data(self.mapToSource(index), role)
        if role == CELL_ROLE:
            return nested_cell(attempt, self.runs.shown[index.column()].key)
        return None

    def flags(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable


class RunsDelegate(CellDelegate):
    """Final rows as ``.tbl``; earlier attempts (``tr.nest``) on the sunken fill, 36 px, with the elbow and
    "Attempt N" across the first five columns."""

    def __init__(
        self, view: QtWidgets.QTableView, themes: ThemeManager, look: TableLook, proxy: NestedRuns
    ) -> None:
        super().__init__(view, themes, look)
        self.proxy = proxy

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        attempt = self.proxy.nested(index.row())
        if attempt is None:
            super().paint(painter, option, index)
            return
        theme = self.themes.current
        styled: Any = option
        rect: QtCore.QRect = styled.rect
        painter.save()
        painter.fillRect(rect, theme.color("surface.sunken"))
        painter.fillRect(
            QtCore.QRect(rect.x(), rect.bottom(), rect.width(), 1), theme.color("border.line")
        )
        if index.column() == 0:
            self._paint_attempt(painter, rect, attempt, theme)
        else:
            self._paint_content(painter, rect, index, theme)
        painter.restore()

    def _paint_attempt(
        self, painter: QtGui.QPainter, rect: QtCore.QRect, attempt: Mapping[str, Any], theme: Any
    ) -> None:
        """``.elbow`` (10 × 10, left and bottom 1.5 px ``border.strong``, 3 px corner; margin 0 10 3 8) and the
        attempt's number in ``text.secondary``."""
        column = self.look.column(0)
        style = TextStyle(13, 400, 18)
        content = rect.height() - 1
        line_top = rect.y() + style.middle_baseline(content) - style.baseline(0)
        x = rect.x() + column.padding_left + 8
        bottom = style.baseline(line_top) - 3
        pen = QtGui.QPen(theme.color("border.strong"), 1.5)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(pen)
        path = QtGui.QPainterPath()
        left, top = snap(x) + 0.75, bottom - 10
        path.moveTo(left, top)
        path.lineTo(left, bottom - 0.75 - 3)
        path.quadTo(left, bottom - 0.75, left + 3, bottom - 0.75)
        path.lineTo(left + 10 - 0.75, bottom - 0.75)
        painter.drawPath(path)
        text = f"Attempt {attempt.get('attempt_number', '')}"
        draw_text(painter, x + 10 + 10, line_top, text, style, theme.color("text.secondary"))


class RunsTable(PagedTable):
    """The paged final runs (``results.query`` runs) behind ``NestedRuns``."""

    opened = QtCore.Signal(object)  # a final run's record

    def __init__(self, transport: Transport, themes: ThemeManager, names: Names) -> None:
        self.run_model = RunsModel(names)
        self.run_model.set_shown(load_columns())
        super().__init__(
            transport, "results-runs", "runs", self.run_model.columns, model=self.run_model
        )
        self.themes = themes
        self.summary = False
        self.label.hide()
        layout = self.layout()
        if layout is not None:
            layout.setSpacing(0)
        self.proxy = NestedRuns(self.run_model)
        self.view.setModel(self.proxy)
        self.view.selectionModel().currentRowChanged.connect(self.select_row)
        self.look = self._look()
        style_table(self.view, themes, self.look)
        # earlier attempts are lower than final runs
        self.view.verticalHeader().setMinimumSectionSize(NESTED_HEIGHT)
        self.delegate = self._delegate()
        self._header()
        self.proxy.modelReset.connect(self._rows_changed)
        self.expanding: dict[str, str] = {}  # request tag -> experiment
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)  # Enter opens the current run

    def select_row(self, current: QtCore.QModelIndex, previous: QtCore.QModelIndex) -> None:
        return  # a click opens a run (``opened``); the current row only moves the keyboard focus

    # columns and sorting

    def _look(self) -> TableLook:
        """The shown columns' styles: the card's 20 px inset as the first column's left and the last column's
        right padding; the last column takes the room left. Dragged widths are remembered per column set."""
        shown = self.run_model.shown
        styles = [c.style for c in shown]
        styles[0] = replace(styles[0], padding_left=20)
        styles[-1] = replace(styles[-1], padding_right=20, width=None)
        keys = [c.key for c in shown]
        key = "results.study-runs"
        if keys != [c.key for c in DEFAULT_COLUMNS]:
            key += "." + hashlib.sha256(",".join(keys).encode()).hexdigest()[:8]
        return TableLook(columns=styles, key=key, last_rule=True)

    def _delegate(self) -> "RunsDelegate":
        previous = self.view.itemDelegate()
        delegate = RunsDelegate(self.view, self.themes, self.look, self.proxy)
        self.view.setItemDelegate(delegate)
        if previous is not None and previous.parent() is self.view:
            previous.deleteLater()  # its event filters go with it
        return delegate

    def _header(self) -> None:
        header = self.view.horizontalHeader()
        assert isinstance(header, TableHeader)
        header.sectionClicked.connect(self.sort_column)
        self._indicate()

    def _indicate(self) -> None:
        """The sort indicator on the column the runs are sorted by (none in the engine's run order)."""
        header = self.view.horizontalHeader()
        column = next(
            (i for i, c in enumerate(self.run_model.shown) if c.sort and c.sort == self.sort_by),
            None,
        )
        if column is None:
            header.setSortIndicatorShown(False)
            return
        order = (
            QtCore.Qt.SortOrder.DescendingOrder
            if self.sort_descending
            else QtCore.Qt.SortOrder.AscendingOrder
        )
        header.setSortIndicator(column, order)
        header.setSortIndicatorShown(True)

    def set_shown(self, columns: list[Column]) -> None:
        """Show ``columns`` (the Columns menu); remembered per viewer."""
        columns = ordered(columns)
        if not columns or [c.key for c in columns] == [c.key for c in self.run_model.shown]:
            return
        save_columns(columns)
        if self.sort_by is not None and self.sort_by not in {c.sort for c in columns}:
            self.sort_by, self.sort_descending = None, False  # its column went: the engine's order
        self.run_model.set_shown(columns)
        self.look = self._look()
        apply_columns(self.view, self.themes, self.look)
        self.delegate = self._delegate()
        self._header()
        self.offset = 0
        self.refresh()

    def sort_column(self, column: int) -> None:
        """A click on a header: sort by its field, then the other way; a column without one keeps the order."""
        shown = self.run_model.shown
        key = shown[column].sort if 0 <= column < len(shown) else None
        if key is not None:
            self.sort_descending = not self.sort_descending if self.sort_by == key else False
            self.sort_by = key
            self.refresh()
        self._indicate()  # the header moves its indicator on any click: put it where the order is

    def visible_offsets(self) -> list[int]:
        first = self.proxy.layout[max(0, self.view.rowAt(0))][0] if self.proxy.layout else 0
        bottom = self.view.rowAt(self.view.viewport().height() - 1)
        last = (
            self.proxy.layout[bottom][0]
            if 0 <= bottom < len(self.proxy.layout)
            else min(self.run_model.rowCount() - 1, first + 100)
        )
        return list(range(first // 100 * 100, max(first, last) // 100 * 100 + 1, 100))

    def _rows_changed(self) -> None:
        """Earlier attempts: 36 px rows, the elbow and number across the first five columns."""
        self.view.clearSpans()
        for row, (_, child) in enumerate(self.proxy.layout):
            self.view.setRowHeight(row, ROW_HEIGHT if child < 0 else NESTED_HEIGHT)
            if child >= 0:
                self.view.setSpan(row, 0, 1, 5)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            watched is self.view.viewport()
            and event.type() == QtCore.QEvent.Type.MouseButtonRelease
        ):
            mouse = event if isinstance(event, QtGui.QMouseEvent) else None
            if mouse is not None and mouse.button() == QtCore.Qt.MouseButton.LeftButton:
                self.click(self.view.indexAt(mouse.position().toPoint()), mouse.position().x())
        if watched is self.view and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event if isinstance(event, QtGui.QKeyEvent) else None
            if key is not None and key.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                self.click(self.view.currentIndex(), None)
                return True
        return super().eventFilter(watched, event)

    def click(self, index: QtCore.QModelIndex, x: float | None) -> None:
        """A click on "N attempts" expands or collapses them; anywhere else on a run (an earlier attempt too)
        opens it."""
        earlier = self.proxy.nested(index.row()) if index.isValid() else None
        if earlier is not None:
            self.opened.emit(earlier)
            return
        record = self.proxy.record(index.row()) if index.isValid() else None
        if record is None:
            return
        attempts = record.get("experiment_attempts")
        outcome = next((i for i, c in enumerate(self.run_model.shown) if c.key == "outcome"), -1)
        if (
            x is not None
            and index.column() == outcome
            and isinstance(attempts, int)
            and attempts > 1
        ):
            rect = self.view.visualRect(index)
            look = outcome_look(record)
            badge_end = (
                rect.x() + self.look.column(outcome).padding_left + (look.width if look else 0)
            )
            if x > badge_end:
                self.expand(str(record.get("experiment_id")))
                return
        self.opened.emit(record)

    def expand(self, experiment: str) -> None:
        if experiment in self.proxy.open or experiment in self.proxy.earlier:
            self.proxy.toggle(experiment)
            return
        tag = "results-attempts:" + uuid.uuid4().hex
        self.expanding[tag] = experiment
        self.transport.submit(
            tag,
            "results.query",
            {
                "table": "runs",
                "filters": {"experiment_id": experiment, "is_final": False},
                "offset": 0,
                "limit": 250,
            },
        )

    def attempts_received(self, tag: str, result: Mapping[str, Any]) -> bool:
        experiment = self.expanding.pop(tag, None)
        if experiment is None:
            return tag.startswith("results-attempts:")
        attempts = sorted(result.get("rows") or [], key=lambda r: r.get("attempt_number") or 0)
        self.proxy.toggle(experiment, list(attempts))
        return True


class Footer(TableFooter):
    """``.tbl-foot``: "<b>11</b> reported · <b>2</b> excluded from reports · <b>1</b> replaced attempt" over the
    current filter, with missing observations and discarded attempts when there are any."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent, padding=20)
        self.parts: list[tuple[str, str]] = []

    def set_counts(self, counts: Mapping[str, int] | None) -> None:
        parts: list[tuple[str, str]] = []
        if counts is not None:
            reported = counts.get(REPORTED, 0)
            out = sum(v for k, v in counts.items() if k.startswith("excluded"))
            missing = sum(v for k, v in counts.items() if k.startswith("missing"))
            parts = [(f"{reported:,}", " reported"), (f"{out:,}", " excluded from reports")]
            if missing:
                parts.append(
                    (
                        f"{missing:,}",
                        " missing" + (" observation" if missing == 1 else " observations"),
                    )
                )
            for key, word in (("replaced", "replaced"), ("discarded", "discarded")):
                count = counts.get(key, 0)
                if count:
                    parts.append((f"{count:,}", f" {word} attempt" + ("" if count == 1 else "s")))
        if parts != self.parts:
            self.parts = parts
            self.setAccessibleName(" · ".join(a + b for a, b in parts))
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), self.color("border.line"))
        top = 1 + (39 - 18) / 2
        x = 20.0
        for index, (count, words) in enumerate(self.parts or [("—", " reported")]):
            if index:  # ``.sep-dot``: "·" with 6 px margins inside the 8 px gaps
                x += 8 + 6
                x += draw_text(painter, x, top, "·", FOOT, self.color("text.tertiary")) + 6 + 8
            x += draw_text(painter, x, top, count, FOOT_COUNT, self.color("text.ink"))
            x += draw_text(painter, x, top, words, FOOT, self.color("text.secondary"))


class ColumnMenu(Themed):
    """The Columns menu: the default columns, then every other scalar field of the loaded records, each with a
    check box. The menu component's look and sizes with group heads, like the Monitor's Layers menu: it keeps
    open while columns are switched, scrolls when taller than its room, and keeps one column shown. The arrow
    keys, Home and End move; Enter or Space switches; Esc or a click outside closes it."""

    changed = QtCore.Signal(object)  # list[Column]

    def __init__(
        self,
        themes: ThemeManager,
        shown: list[Column],
        fields: Mapping[str, bool],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.shown = list(shown)
        extras = {c.key: c for c in shown if c.key not in {d.key for d in DEFAULT_COLUMNS}}
        # a shown record field stays listed when the loaded records no longer hold it
        self.fields = [
            extras.get(key) or extra_column(key, numeric)
            for key, numeric in sorted(
                {**{k: c.style.kind == "number" for k, c in extras.items()}, **fields}.items()
            )
        ]
        self.rows = self._rows()
        self.hovered, self.scrolled = -1, 0.0
        self.host: PopupHost | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setProperty("focusShape", "none")
        self.setAccessibleName("Table columns")

    def _rows(self) -> list[tuple[QtCore.QRectF, str, Column | None]]:
        """Rows in content coordinates: (rect, group title, None), (rect, "rule", None), (rect, "", column)."""
        rows: list[tuple[QtCore.QRectF, str, Column | None]] = []
        y, inner = float(INSET), MENU_WIDTH - 2.0 * INSET
        for title, columns in (
            ("Default columns", DEFAULT_COLUMNS),
            ("Record fields", self.fields),
        ):
            if not columns:
                continue
            if rows:
                rows.append((QtCore.QRectF(INSET, y, inner, SEPARATOR_HEIGHT), "rule", None))
                y += SEPARATOR_HEIGHT
            rows.append((QtCore.QRectF(INSET, y, inner, GROUP_HEIGHT), title, None))
            y += GROUP_HEIGHT
            for column in columns:
                rows.append((QtCore.QRectF(INSET, y, inner, ITEM_HEIGHT), "", column))
                y += ITEM_HEIGHT
        return rows

    def content_height(self) -> float:
        return (self.rows[-1][0].bottom() if self.rows else 0.0) + INSET

    def popup(
        self,
        host: PopupHost,
        anchor: QtWidgets.QWidget,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Open under ``anchor`` (the Columns button), right-aligned with it, as tall as its rows allow."""
        window = host.window_widget
        origin = anchor.mapTo(window, QtCore.QPoint(0, 0))
        top = origin.y() + anchor.height() + 4
        height = min(
            math.ceil(self.content_height()), MENU_HEIGHT, max(120, window.height() - 8 - top)
        )
        x = max(8, origin.x() + anchor.width() - MENU_WIDTH)
        self.host = host
        host.open_popup(
            self,
            QtCore.QRect(x, top, MENU_WIDTH, height),
            opener=anchor,
            on_close=on_close,
            radius=8,
            delete_on_close=True,
        )
        self.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def _items(self) -> list[int]:
        return [index for index, (_, _, column) in enumerate(self.rows) if column is not None]

    def _index_at(self, position: QtCore.QPointF) -> int:
        point = QtCore.QPointF(position.x(), position.y() + self.scrolled)
        for index, (rect, _, column) in enumerate(self.rows):
            if column is not None and rect.contains(point):
                return index
        return -1

    def _scroll_to(self, value: float) -> None:
        value = max(0.0, min(value, self.content_height() - self.height()))
        if value != self.scrolled:
            self.scrolled = value
            self.update()

    def _reveal(self, index: int) -> None:
        rect = self.rows[index][0]
        if rect.top() - INSET < self.scrolled:
            self._scroll_to(rect.top() - INSET)
        elif rect.bottom() + INSET > self.scrolled + self.height():
            self._scroll_to(rect.bottom() + INSET - self.height())

    def _toggle(self, index: int) -> None:
        column = self.rows[index][2]
        if column is None:
            return
        keys = [c.key for c in self.shown]
        if column.key in keys:
            if len(keys) == 1:
                return  # one column stays shown
            self.shown = [c for c in self.shown if c.key != column.key]
        else:
            self.shown = ordered([*self.shown, column])
        self.changed.emit(list(self.shown))
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self._index_at(event.position())
        if index != self.hovered:
            self.hovered = index
            self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hovered = -1
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            index = self._index_at(event.position())
            if index >= 0:
                self._toggle(index)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self._scroll_to(self.scrolled - event.angleDelta().y() / 120 * ITEM_HEIGHT)
        self.hovered = self._index_at(event.position())
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        items = self._items()
        if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up) and items:
            step = 1 if key == QtCore.Qt.Key.Key_Down else -1
            position = items.index(self.hovered) if self.hovered in items else -1
            self.hovered = items[(position + step) % len(items)]
            self._reveal(self.hovered)
            self.update()
        elif key in (QtCore.Qt.Key.Key_Home, QtCore.Qt.Key.Key_End) and items:
            self.hovered = items[0] if key == QtCore.Qt.Key.Key_Home else items[-1]
            self._reveal(self.hovered)
            self.update()
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space):
            if self.hovered >= 0:
                self._toggle(self.hovered)
        elif key == QtCore.Qt.Key.Key_Escape and self.host is not None:
            self.host.close_popup()
        else:
            super().keyPressEvent(event)

    def focusNextPrevChild(self, next: bool) -> bool:
        return True

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        frame = rounded_path(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7.5)
        painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
        painter.setBrush(theme.color("surface.card"))
        painter.drawPath(frame)
        painter.setClipPath(frame)
        painter.translate(0, -self.scrolled)
        shown = {c.key for c in self.shown}
        for index, (rect, title, column) in enumerate(self.rows):
            if rect.bottom() < self.scrolled or rect.top() > self.scrolled + self.height():
                continue
            if title == "rule":
                painter.fillRect(
                    QtCore.QRectF(rect.x() + 6, rect.y() + 4, rect.width() - 12, 1),
                    theme.color("border.line"),
                )
            elif column is None:
                top = rect.y() + (GROUP_HEIGHT - 18) / 2 + 2
                draw_text(
                    painter, rect.x() + 10, top, title, GROUP_TEXT, theme.color("text.secondary")
                )
            else:
                if index == self.hovered:
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(theme.color("surface.hover"))
                    painter.drawPath(rounded_path(rect, 6))
                box = QtCore.QRectF(rect.x() + 10, rect.y() + (ITEM_HEIGHT - 16) / 2, 16, 16)
                paint_check(painter, box, theme, "on" if column.key in shown else "off")
                top = rect.y() + (ITEM_HEIGHT - 18) / 2
                text = ITEM_TEXT.elide(column.title, rect.width() - 10 - 16 - 10 - 10)
                draw_text(
                    painter, rect.x() + 10 + 16 + 10, top, text, ITEM_TEXT, theme.color("text.ink")
                )


class RunsCard(Card):
    """``.card.runs``: the filter bar (campaign 236 px, robot, world, profile and outcome 150 px, Columns at the
    right), the table and the footer."""

    filters_changed = QtCore.Signal()

    def __init__(
        self,
        themes: ThemeManager,
        transport: Transport,
        names: Names,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(themes, parent)
        self.bar = FlexRow(self, gap=8, left=20, right=16)
        self.campaign, self.robot, self.world = (
            KeySelect(themes),
            KeySelect(themes),
            KeySelect(themes),
        )
        self.profile, self.outcome = KeySelect(themes), KeySelect(themes)
        self.bar.add(self.campaign, width=236)
        for select in (self.robot, self.world, self.profile, self.outcome):
            self.bar.add(select, width=150)
        self.bar.add_stretch()
        self.columns_button = Button(themes, "Columns", icon="sliders", variant="ghost", small=True)
        self.columns_button.clicked.connect(self.open_columns)
        self.bar.add(self.columns_button)
        self.selects: tuple[tuple[str, KeySelect, str], ...] = (
            ("campaign_id", self.campaign, "All campaigns"),
            ("robot_id", self.robot, "All robots"),
            ("world_id", self.world, "All worlds"),
            ("profile_id", self.profile, "All profiles"),
            ("outcome", self.outcome, "All outcomes"),
        )
        for _, select, everything in self.selects:
            select.addItem(everything, None)
            select.currentIndexChanged.connect(lambda _: self.filters_changed.emit())
        self.table = RunsTable(transport, themes, names)
        self.table.setParent(self)
        self.table.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.footer = Footer(themes, self)

    def open_columns(self) -> None:
        """The Columns menu under its button (it stays open while columns are switched)."""
        host = self.window().findChild(PopupHost)
        if host is None:
            return
        table = self.table
        fields = record_fields(record for _, record in table.run_model.records())
        menu = ColumnMenu(self.themes, table.run_model.shown, fields)
        menu.changed.connect(table.set_shown)
        menu.popup(host, self.columns_button)

    def set_options(self, key: str, options: list[tuple[str, Any]]) -> None:
        """A filter's choices (label, value) after its "All …" entry; the current choice is kept when offered."""
        for name, select, everything in self.selects:
            if name != key:
                continue
            chosen = select.currentData()
            select.blockSignals(True)
            select.clear()
            select.addItem(everything, None)
            for text, value in options:
                select.addItem(text, value)
            select.setCurrentIndex(max(0, select.findData(chosen)))
            select.blockSignals(False)

    def filters(self) -> dict[str, Any]:
        return {
            key: select.currentData()
            for key, select, _ in self.selects
            if select.currentData() is not None
        }

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        width, height = self.width(), self.height()
        self.bar.setGeometry(1, 1, width - 2, 51)
        self.table.setGeometry(1, 53, width - 2, max(0, height - 53 - 41))
        self.footer.setGeometry(1, height - 41, width - 2, 40)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        QtGui.QPainter(self).fillRect(
            QtCore.QRect(1, 52, self.width() - 2, 1), self.color("border.line")
        )
