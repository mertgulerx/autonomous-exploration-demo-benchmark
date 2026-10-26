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

"""Results, Study tab (25–28; RESULTS_STUDY.md decision 5; RESULTS_OVERHAUL RO-19, RO-21 to RO-23).

Left, the study card: the study, its evidence level as declared, plan, campaigns, newest export and the analysis
state (current for that export, out of date, refused), the study's warnings, then the outputs of the chosen
analysis one row per kind with each one's n. Right, the selected output as the analysis package produced it: its
PDF sheets on a white page (tables are typeset sheets too, RO-19; the GUI draws no table and computes no
statistic), the sidecar's n line under the head (RO-21) and "Runs behind it", each run opening its run view.
``stats.json`` is listed entry by entry as written; the coverage traces (``study.traces``) are the one view the
GUI draws, raw lines only. A refused analysis (RO-22) shows its reason and the runs that do not fit together.

The analysis outputs live beside the exports (``results/studies/<id>/analysis/<version>-<hash8>``, the hash of
the export's ``SHA256SUMS``, with ``-coding-<sha8>`` when made with a coding file); the GUI reads them through the
transport's local readers, off its thread. An analysis of the newest export, coded or not, is current and the
newest of them is shown; otherwise the newest analysis is shown as out of date.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_analysis import Jobs
from .gui_cells import MONO_TEXT, ColumnStyle, OverlayScrollBar, TableLook, style_table
from .gui_controls import BadgeLook, KeySelect, StatusBadge
from .gui_dialogs import wrap_text
from .gui_marks import KeyValue, KeyValueList
from .gui_pdf import Well
from .gui_runs import run_label
from .gui_style import coverage_text, duration, humanize, number
from .gui_support import POLL_MS, TableModel, Transport
from .gui_theme import ThemeManager
from .contracts import OUR_EXPLORER, explorer_failure, measured_as
from .gui_traces import Style, TracesChart, run_end, trace_styles
from .gui_widgets import (
    BODY,
    Button,
    Card,
    CardHead,
    SegmentedControl,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    plural,
    rounded_path,
)

KV_LABEL = TextStyle(12.5, 400, 18)
KV_VALUE = TextStyle(12.5, 500, 18)
SMALL = TextStyle(12, 400, 18)
SMALL_BOLD = TextStyle(12, 600, 18)
WARN_TEXT = TextStyle(12, 400, 16)
WARN_HEAD = TextStyle(12, 600, 16)
CAPTION = TextStyle(12.5, 400, 18)
CAPTION_BOLD = TextStyle(12.5, 600, 18)
NAME = TextStyle(13, 400, 18)
REFUSAL_TITLE = TextStyle(15, 600, 20)
TABLE_TEXT = TextStyle(12.5, 400, 18)
# <version>-<hash8 of the export>, with -coding-<sha8 of the coding file> for an analysis with a coding file
ANALYSIS_NAME = re.compile(r"^\d+\.\d+\.\d+-([0-9a-f]{8})(?:-coding-[0-9a-f]{8})?$")
REFUSAL_NAME = re.compile(r"^refused-(\d+\.\d+\.\d+-[0-9a-f]{8})\.json$")
LOG = logging.getLogger(__name__)
CHIPS = 6  # "Runs behind it" names this many runs at most and counts the rest


def export_hash(name: str) -> str | None:
    """The export's hash8 in an analysis directory's name (``ANALYSIS_NAME``), coded or not; None for another
    name."""
    found = ANALYSIS_NAME.match(name)
    return found.group(1) if found else None


# ---------- output kinds ----------


@dataclass(frozen=True, slots=True)
class OutputKind:
    """One row of the output list (RO-23): fixed files, or a family of files chosen in the viewer head
    (``variants``, a file-name prefix) with ``then`` shown under the chosen one. ``labels`` names the variants
    (a view's ``variant_labels``; ``VARIANT_NAMES`` otherwise) and ``order`` orders them (a grouped view's
    ``variant_order``: fixed label, world group, page; by name without it, as older analyses list them);
    ``runs`` is False for a view with no runs behind it (a schematic), shown without an n line and without "Runs
    behind it"."""

    code: str
    name: str
    icon: str
    files: tuple[str, ...] = ()
    variants: str = ""
    then: tuple[str, ...] = ()
    labels: tuple[tuple[str, str], ...] = ()
    runs: bool = True
    order: tuple[str, ...] = ()
    # a grouped view's ``variant_groups`` (analysis.json), when the analysis writes them
    groups: tuple[tuple[str, "VariantGroup"], ...] = ()

    @property
    def is_stats(self) -> bool:
        """The view lists ``stats.json`` entry by entry instead of showing PDF sheets."""
        return self.files == ("stats.json",)

    def available(self, outputs: Mapping[str, str]) -> bool:
        if self.code == "C(t)":
            return True
        return any(name in outputs for name in self.files) or bool(self.variant_files(outputs))

    def variant_files(self, outputs: Mapping[str, str]) -> list[str]:
        if not self.variants:
            return []
        files = sorted(
            name for name in outputs if name.startswith(self.variants) and name.endswith(".pdf")
        )
        if not self.order:
            return files
        rank = {variant: index for index, variant in enumerate(self.order)}
        # variants the order does not name follow it, by name
        return sorted(files, key=lambda f: rank.get(variant_of(self, f), len(rank)))

    def variant_label(self, key: str) -> str:
        return dict(self.labels).get(key) or VARIANT_NAMES.get(key, key)


# the GUI's own view, after every list
TRACES_KIND = OutputKind("C(t)", "Coverage traces", "activity")
# The list for an analysis without ``views`` in its analysis.json (the 1.2.x outputs; FIGURE_EXTENSION FX-4); an
# analysis that writes ``views`` (1.3.0 on) declares its own list (``kinds_from_views``).
KINDS = (
    OutputKind("T1", "Accounting", "table", ("tables/T1.pdf",)),
    OutputKind("T2", "Setup and provenance", "table", ("tables/T2.pdf",)),
    OutputKind("F1", "Coverage over time", "chart", ("figures/F1.pdf",)),
    OutputKind("F2/T3", "Attainment", "chart", variants="figures/F2_", then=("tables/T3.pdf",)),
    OutputKind("F3", "Paired effects", "chart", variants="figures/F3_"),
    OutputKind("T4", "Default per stratum", "table", ("tables/T4.pdf",)),
    OutputKind("T5", "Paired differences", "table", ("tables/T5.pdf", "tables/T5H.pdf")),
    OutputKind("S", "Sensitivity", "table", ("tables/S.pdf",)),
    OutputKind("T6", "Compute cost", "table", ("tables/T6.pdf",)),
    OutputKind("F5", "Compute", "chart", ("figures/F5.pdf",)),
    OutputKind("T7", "Reliability", "table", ("tables/T7.pdf",)),
    OutputKind("F6", "Outcomes", "chart", ("figures/F6.pdf",)),
    OutputKind("T8", "Timing validity", "table", ("tables/T8.pdf",)),
    OutputKind("F8", "Representative runs", "chart", ("figures/F8.pdf",)),
    OutputKind("stats", "Statistics", "braces", ("stats.json",)),
    TRACES_KIND,
)
REFUSAL_KIND = OutputKind("!", "Analysis refused", "ban")
VARIANT_NAMES = {
    "R_25": "25 %",
    "R_50": "50 %",
    "R_75": "75 %",
    "R_90": "90 %",
    "R_95": "95 %",
    "R_99": "99 %",  # coverage-stop-v1 analyses (run-record-v3)
    "R_99_9": "99.9 %",
    "R_completion": "Completion",
    "dA_L": "ΔA_L",
    "dC_L": "ΔC(L)",
    "dD_L": "ΔD(L)",
    "dR_90": "ΔR_90",
    "dR_completion": "ΔR_completion",
}


def variant_of(kind: OutputKind, file: str) -> str:
    """ "R_90" of ``figures/F2_R_90.pdf``."""
    return file.removeprefix(kind.variants).removesuffix(".pdf")


@dataclass(frozen=True, slots=True)
class VariantGroup:
    """Where a grouped view's variant belongs (plans/worlds/I8 §3.1): its base variant (``R_25``; empty for a
    view of fixed files such as F1), and the world group (``key``, its section ``index``, ``title``) and page."""

    base: str
    key: str
    index: int
    page: int
    pages: int
    title: str
    base_label: str = ""

    @property
    def label(self) -> str:
        """The group dropdown's entry: "Single challenge", "Single challenge · page 1 of 2"."""
        return self.title if self.pages <= 1 else f"{self.title} · page {self.page} of {self.pages}"


def variant_group(value: Any, stem: str) -> VariantGroup | None:
    """A ``variant_groups`` entry of analysis.json, or a figure sidecar's ``group`` object, as a VariantGroup;
    None when malformed. A sidecar names its base by the ungrouped variant (``variant``: "F2_R_25"), from which
    ``stem`` ("F2_", the view's file prefix) is removed, as the analysis package does (``views_document``)."""
    if not isinstance(value, Mapping):
        return None
    key, title = value.get("key"), value.get("title")
    numbers = [value.get(name) for name in ("index", "page", "pages")]
    if not isinstance(key, str) or not isinstance(title, str):
        return None
    integers = [n for n in numbers if isinstance(n, int) and not isinstance(n, bool)]
    if len(integers) != 3:
        return None
    if isinstance(value.get("base"), str):
        base = str(value["base"])
    elif isinstance(value.get("variant"), str):
        ungrouped = str(value["variant"])
        base = ungrouped.removeprefix(stem) if stem and ungrouped.startswith(stem) else ""
    else:
        return None
    label = value.get("base_label")
    index, page, pages = integers
    return VariantGroup(
        base, key, index, page, pages, title, label if isinstance(label, str) else ""
    )


# a view's kind and its default icon
VIEW_KINDS = {"table": "table", "chart": "chart", "data": "braces"}
ICON_NAMES = frozenset(path.stem for path in gui_icons.ICONS.glob("*.svg"))
RESERVED_CODES = (TRACES_KIND.code, REFUSAL_KIND.code)


class ViewError(ValueError):
    """An entry of ``analysis.json["views"]`` the tab cannot list (the reason)."""


def _strings(value: Any, name: str) -> tuple[str, ...]:
    """A list of non-empty strings."""
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ViewError(f"{name} is not a list of file names")
    return tuple(value)


def view_kind(entry: Any, index: int) -> OutputKind:
    """One entry of ``analysis.json["views"]`` (analysis/benchmark_analysis/outputs.py ``View``) as a list row;
    ViewError when it cannot be listed. An unknown kind shows as a chart and an unknown icon as its kind's, each
    logged."""
    if not isinstance(entry, Mapping):
        raise ViewError("not an object")
    code, title = entry.get("code"), entry.get("title")
    if not isinstance(code, str) or not code or code in RESERVED_CODES:
        raise ViewError(f"code {code!r} is missing or reserved")
    if not isinstance(title, str) or not title:
        raise ViewError(f"{code}: no title")
    files = _strings(entry.get("files", []), f"{code}: files")
    then = _strings(entry.get("then", []), f"{code}: then")
    variants, labels, runs = (
        entry.get("variants", ""),
        entry.get("variant_labels", {}),
        entry.get("runs", True),
    )
    if not isinstance(variants, str):
        raise ViewError(f"{code}: variants is not a file-name prefix")
    if not files and not variants:
        raise ViewError(f"{code}: neither files nor variants")
    if "stats.json" in files and files != ("stats.json",):
        raise ViewError(f"{code}: stats.json listed with other files")
    if not isinstance(labels, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in labels.items()
    ):
        raise ViewError(f"{code}: variant_labels malformed")
    if not isinstance(runs, bool):
        raise ViewError(f"{code}: runs is not true or false")
    order = _strings(entry.get("variant_order", []), f"{code}: variant_order")
    raw_groups = entry.get("variant_groups", {})
    if not isinstance(raw_groups, Mapping):
        raise ViewError(f"{code}: variant_groups malformed")
    stem = variants.rsplit("/", 1)[-1]
    groups = []
    for variant, value in raw_groups.items():
        group = variant_group(value, stem)
        if not isinstance(variant, str) or group is None:
            raise ViewError(f"{code}: variant_groups malformed")
        groups.append((variant, group))
    kind = entry.get("kind")
    if not isinstance(kind, str) or kind not in VIEW_KINDS:
        LOG.warning("analysis.json views[%d]: %s: kind %r shown as chart", index, code, kind)
        kind = "chart"
    icon = entry.get("icon")
    if not isinstance(icon, str) or icon not in ICON_NAMES:
        icon, unknown = VIEW_KINDS[kind], icon
        LOG.warning(
            "analysis.json views[%d]: %s: icon %r replaced by %r", index, code, unknown, icon
        )
    return OutputKind(
        code,
        title,
        icon,
        files,
        variants,
        then,
        tuple(sorted(labels.items())),
        runs,
        order,
        tuple(sorted(groups)),
    )


def kinds_from_views(views: Any) -> tuple[OutputKind, ...] | None:
    """The output list an analysis declares (1.3.0 on), in its order and followed by the coverage traces; None
    for an analysis without ``views`` (before 1.3.0), which keeps ``KINDS``. Entries that cannot be listed and
    a later entry repeating a code are skipped and logged; ``views`` that is not a list, or leaves no entry,
    falls back to ``KINDS`` (logged)."""
    if views is None:
        return None
    if not isinstance(views, list):
        LOG.warning("analysis.json views is not a list; using the built-in output list")
        return None
    rows: list[OutputKind] = []
    for index, entry in enumerate(views):
        try:
            row = view_kind(entry, index)
            if any(r.code == row.code for r in rows):  # a code keys the choice and its variant
                raise ViewError(f"code {row.code!r} repeated")
        except ViewError as exc:
            LOG.warning("analysis.json views[%d] skipped: %s", index, exc)
            continue
        rows.append(row)
    if not rows:
        LOG.warning("analysis.json views lists no readable view; using the built-in output list")
        return None
    return (*rows, TRACES_KIND)


def sidecar_of(file: str) -> str:
    """The sidecar beside an output file (``tables/T1.pdf`` -> ``tables/T1.sidecar.json``)."""
    stem = file.rsplit(".", 1)[0]
    return f"{stem}.sidecar.json"


# ---------- texts ----------


def newest_export(detail: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The study's newest finished export (``study.get``): the one the card shows and Run analysis reads."""
    exports: list[dict[str, Any]] = list((detail or {}).get("finished_exports") or [])
    newest = max(exports, key=lambda e: int(e.get("utc_ns") or 0), default=None)
    return dict(newest) if newest is not None else None


def short_date(ns: Any) -> str:
    if not isinstance(ns, int):
        return "—"
    moment = datetime.fromtimestamp(ns / 1e9, timezone.utc).astimezone()
    return f"{moment.day} {moment.strftime('%b %H:%M')}"


def left_out_text(kind: str, count: int) -> str:
    """One report class a view leaves out ("1 excluded: runtime failure")."""
    if kind.startswith("excluded_"):
        return f"{count} excluded: {humanize(kind.removeprefix('excluded_')).lower()}"
    if kind.startswith("missing_"):
        return f"{count} missing: {humanize(kind.removeprefix('missing_')).lower()}"
    if kind == "reported":
        return f"{count} reported " + ("run" if count == 1 else "runs") + " not used"
    return f"{count} {humanize(kind).lower()}"


def n_line(counts: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """The caption under a view's head (RO-21), as (bold, text) parts: n, what it leaves out, the replaced and
    discarded attempts and the study's shape."""
    if not isinstance(counts, Mapping):
        return [("", "The sidecar holds no counts (an analysis before version 1.2.0)")]
    parts: list[tuple[str, str]] = []
    pairs = counts.get("pairs")
    runs = counts.get("runs", 0)
    if isinstance(pairs, int):
        parts.append((str(pairs), " pairs" if pairs != 1 else " pair"))
        parts.append(("", plural(int(runs), "run")))
    else:
        parts.append((str(runs), " runs" if runs != 1 else " run"))
    left = counts.get("left_out") or {}
    for kind, count in sorted(left.items()):
        parts.append(("", left_out_text(str(kind), int(count))))
    if not left:
        parts.append(("", "no final run left out"))
    for key, word in (("replaced", "replaced attempt"), ("discarded", "discarded attempt")):
        count = counts.get(key)
        if isinstance(count, int) and count:
            parts.append(("", plural(count, word)))
    strata, profiles = counts.get("strata"), counts.get("profiles")
    if isinstance(strata, int) and isinstance(profiles, int):
        parts.append(
            (
                "",
                f"{plural(strata, 'stratum').replace('stratums', 'strata')} × {plural(profiles, 'profile')}",
            )
        )
    return parts


def row_count(kind: OutputKind, sidecar: Mapping[str, Any] | None) -> str:
    """A row's n at the right ("60 runs", "29 pairs", "48 values")."""
    if kind.code == "C(t)":
        return "interactive"
    if kind is REFUSAL_KIND:
        return "refused"
    if not kind.runs:
        return "no runs"
    if not sidecar:
        return ""
    if kind.is_stats and isinstance(sidecar.get("entries"), int):
        return plural(sidecar["entries"], "value")
    counts = sidecar.get("counts")
    if not isinstance(counts, Mapping):  # a sidecar without the counts block (or a malformed one)
        counts = {}
    if isinstance(counts.get("pairs"), int):
        return plural(counts["pairs"], "pair")
    if isinstance(counts.get("runs"), int):
        return plural(counts["runs"], "run")
    return plural(len(sidecar.get("attempt_ids") or []), "run")


def paint_parts(
    painter: QtGui.QPainter,
    x: float,
    top: float,
    parts: Sequence[tuple[str, str]],
    theme: Any,
    *,
    regular: TextStyle = CAPTION,
    bold: TextStyle = CAPTION_BOLD,
) -> float:
    """``.cap``: parts separated by a faint "·" with 6 px margins; a bold number in ink, 4 px before its text."""
    secondary, ink = theme.color("text.secondary"), theme.color("text.ink")
    for index, (strong, text) in enumerate(parts):
        if index:
            x += 6
            x += draw_text(painter, x, top, "·", regular, theme.color("text.tertiary")) + 6
        if strong:
            x += draw_text(painter, x, top, strong, bold, ink) + 4
        x += draw_text(painter, x, top, text.lstrip(), regular, secondary)
    return x


# ---------- the study card ----------


class Warnings(Themed):
    """``.warns``: "N warnings" with the alert icon over the warnings, on ``state.warnSoft``, radius 6."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.warnings: list[str] = []
        self.lines: list[list[str]] = []

    def set_warnings(self, warnings: Sequence[str], width: float) -> None:
        self.warnings = list(warnings)
        self.lines = [wrap_text(text, WARN_TEXT, width - 20) for text in self.warnings]
        height = 8 + 16 + 4 + sum(16 * len(lines) for lines in self.lines)
        height += 3 * max(0, len(self.lines) - 1) + 8
        self.setFixedHeight(height if self.warnings else 0)
        self.setVisible(bool(self.warnings))
        self.setAccessibleDescription("; ".join(self.warnings))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.color("state.warnSoft"))
        painter.drawPath(rounded_path(QtCore.QRectF(self.rect()), 6))
        warn = self.color("state.warn")
        gui_icons.draw(painter, "alert", warn, QtCore.QPointF(10, 8 + 1), 14)
        draw_text(painter, 10 + 14 + 6, 8, plural(len(self.warnings), "warning"), WARN_HEAD, warn)
        y = 8 + 16 + 4
        body = self.color("text.body")
        for index, lines in enumerate(self.lines):
            if index:
                y += 3
            for line in lines:
                draw_text(painter, 10, y, line, WARN_TEXT, body)
                y += 16


class OutputList(Themed):
    """``.olist``: 36 px rows (icon, code, name, n), the selected one on ``surface.selected`` with a 2 px ink bar;
    arrow keys move, Enter opens. ``chosen`` carries the row index."""

    chosen = QtCore.Signal(int)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.rows: list[tuple[OutputKind, str]] = []
        self.current = -1
        self.hovered = -1
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Outputs")

    def set_rows(self, rows: Sequence[tuple[OutputKind, str]]) -> None:
        self.rows = list(rows)
        self.setFixedHeight(36 * len(self.rows))
        self.update()

    def set_current(self, index: int) -> None:
        self.current = index
        self.update()

    def _row_at(self, y: float) -> int:
        row = int(y // 36)
        return row if 0 <= row < len(self.rows) else -1

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        row = self._row_at(event.position().y())
        if row != self.hovered:
            self.hovered = row
            self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hovered = -1
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        row = self._row_at(event.position().y())
        if row >= 0 and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.chosen.emit(row)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up) and self.rows:
            step = 1 if key == QtCore.Qt.Key.Key_Down else -1
            self.chosen.emit((self.current + step) % len(self.rows))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        line = theme.color("border.line")
        width = self.width()
        for index, (kind, count) in enumerate(self.rows):
            top = index * 36
            selected = index == self.current
            if selected:
                painter.fillRect(QtCore.QRect(0, top, width, 35), theme.color("surface.selected"))
                painter.fillRect(QtCore.QRect(0, top, 2, 35), theme.color("text.ink"))
            elif index == self.hovered:
                painter.fillRect(QtCore.QRect(0, top, width, 35), theme.color("surface.hover"))
            painter.fillRect(QtCore.QRect(0, top + 35, width, 1), line)
            refused = kind is REFUSAL_KIND
            icon_token = "state.warn" if refused else "text.ink" if selected else "text.tertiary"
            gui_icons.draw(
                painter, kind.icon, theme.color(icon_token), QtCore.QPointF(20, top + 10), 15
            )
            text_top = top + (35 - 18) / 2
            draw_text(
                painter, 20 + 18 + 8, text_top, kind.code, SMALL_BOLD, theme.color("text.ink")
            )
            right = width - 20
            if refused:
                style, token = TextStyle(11.5, 600, 18), "state.warn"
            else:
                style, token = TextStyle(12, 400, 18, tabular=True), "text.tertiary"
            draw_text(
                painter, right - style.advance(count), text_top, count, style, theme.color(token)
            )
            name_left = 20 + 18 + 8 + 46 + 8
            room = right - style.advance(count) - 8 - name_left
            draw_text(
                painter,
                name_left,
                text_top,
                NAME.elide(kind.name, room),
                NAME,
                theme.color("text.body"),
            )


class StudyCard(Card):
    """``.card.sc`` (360 px): the study and its declared evidence level, its facts (30 px rows), the warnings,
    "Outputs" and the output list, which scrolls when the card is short."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "", self)
        self.head.row.gap = 10
        self.badge = StatusBadge(themes)
        self.head.add(self.badge)
        self.facts = KeyValueList(themes, row_height=30, label_style=KV_LABEL, value_style=KV_VALUE)
        self.facts.setParent(self)
        self.warnings = Warnings(themes, self)
        self.outputs_title = TextLine(themes, "Outputs", SMALL_BOLD, "text.secondary", self)
        self.list_area = QtWidgets.QScrollArea(self)
        self.list_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.list_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_area.setWidgetResizable(False)
        self.list_area.viewport().setAutoFillBackground(False)
        self.list_area.setStyleSheet("QScrollArea { background: transparent; }")
        self.list = OutputList(themes)
        self.list_area.setWidget(self.list)
        OverlayScrollBar(themes, self.list_area)

    def set_study(self, title: str, look: BadgeLook | None, facts: Sequence[KeyValue]) -> None:
        self.head.title.setText(title)
        self.badge.set_look(look)
        self.head.row.relayout()
        self.facts.set_rows(facts)
        self.relayout()

    def set_warnings(self, warnings: Sequence[str]) -> None:
        self.warnings.set_warnings(warnings, self.width() - 40)
        self.relayout()

    def relayout(self) -> None:
        width = self.width()
        self.head.setGeometry(1, 1, width - 2, 52)
        top = 1 + 52 + 4
        self.facts.setGeometry(20, top, width - 40, self.facts.height())
        top += self.facts.height()
        if self.warnings.warnings:
            self.warnings.set_warnings(self.warnings.warnings, width - 40)
            top += 10
            self.warnings.setGeometry(20, top, width - 40, self.warnings.height())
            top += self.warnings.height()
        top += 14
        self.outputs_title.setGeometry(20, top + 6, width - 40, 18)
        self._title_rule = top + 29
        top += 30
        self.list_area.setGeometry(1, top, width - 2, max(0, self.height() - 1 - top))
        self.list.resize(width - 2, self.list.height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.relayout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        top = getattr(self, "_title_rule", None)
        if top is not None:
            QtGui.QPainter(self).fillRect(
                QtCore.QRect(20, top, self.width() - 40, 1), self.color("border.line")
            )


# ---------- the viewer ----------


STATS_COLUMNS = (
    ("estimand", "Estimand", ColumnStyle("text", padding_left=20, width=250)),
    ("stratum", "Stratum", ColumnStyle("text", width=170)),
    ("profile", "Profile", ColumnStyle("text", width=140)),
    # the primary population, or the sensitivity set or execution mode an entry is computed on
    ("population", "Population", ColumnStyle("text", width=150)),
    ("value", "Value (4 significant digits)", ColumnStyle("text", width=None)),
    ("n", "n", ColumnStyle("number", width=56)),
    ("role", "Role", ColumnStyle("text", padding_right=20, width=110)),
)


def stat_number(value: Any) -> str:
    found = number(value)
    return f"{found:.4g}" if found is not None else "—"


def bound_text(interval: Mapping[str, Any], side: str) -> str:
    """One bound: "> L" where the data put it beyond the limit (as the tables print it), "—" when absent."""
    if interval.get(f"{side}_beyond_limit"):
        return "> L"
    return stat_number(interval.get(side))


def interval_text(interval: Any) -> str:
    """ "[lo, hi] 95 % Wilson" as the entry states it, its note after it (a reduced level, a bound > L)."""
    if not isinstance(interval, Mapping):
        return ""
    if (
        interval.get("lower") is None
        and interval.get("upper") is None
        and not (interval.get("lower_beyond_limit") or interval.get("upper_beyond_limit"))
    ):
        return ""
    level = number(interval.get("level"))
    method = str(interval.get("method") or "")
    tail = f" {level * 100:.4g} % {method}".rstrip() if level else ""
    note = f" ({interval['note']})" if interval.get("note") else ""
    return f" [{bound_text(interval, 'lower')}, {bound_text(interval, 'upper')}]{tail}{note}"


def value_text(entry: Mapping[str, Any]) -> str:
    """An entry's main values from its own fields, formatted only (no value is computed here); the kinds with
    nested results (reliability, timing) are read in the file."""
    if "km_median" in entry or "rmst" in entry:
        # a computed median that is null was not reached by L (the curve stays above one half)
        median = (
            "> L"
            if "km_median" in entry and entry["km_median"] is None
            else stat_number(entry.get("km_median"))
        )
        return (
            f"KM median {median}{interval_text(entry.get('km_median_interval'))}"
            f" · RMST {stat_number(entry.get('rmst'))}{interval_text(entry.get('rmst_interval'))}"
        )
    if "difference" in entry:
        return f"{stat_number(entry.get('difference'))}{interval_text(entry.get('difference_interval'))}"
    if "estimate" in entry:
        estimator = f" · {entry['estimator']}" if entry.get("estimator") else ""
        return (
            f"{stat_number(entry.get('estimate'))}{interval_text(entry.get('interval'))}{estimator}"
        )
    if "median" in entry:
        return (
            f"median {stat_number(entry.get('median'))} "
            f"[Q1 {stat_number(entry.get('q1'))}, Q3 {stat_number(entry.get('q3'))}]"
        )
    return "nested values: open the file"


def stats_rows(
    document: Mapping[str, Any], name: Callable[[str, Any], str]
) -> list[dict[str, Any]]:
    """``stats.json`` entries as written, one row each."""
    rows = []
    entries = document.get("entries") or []
    robots = len({entry.get("robot_id") for entry in entries if entry.get("robot_id")}) > 1
    for entry in entries:
        estimand = str(entry.get("estimand", ""))
        for part in ("endpoint", "comparison"):
            if entry.get(part):
                estimand += f" · {entry[part]}"
        if isinstance(entry.get("horizon_s"), (int, float)):
            estimand += f" at {duration(entry['horizon_s'])}"
        stratum = " · ".join(
            name(kind, entry.get(f"{kind}_id"))
            for kind in (("robot", "world", "spawn") if robots else ("world", "spawn"))
            if entry.get(f"{kind}_id")
        )
        n = entry.get("n_pairs", entry.get("n"))
        rows.append(
            dict(
                id=entry.get("id"),
                estimand=estimand,
                stratum=stratum or "—",
                profile=name("profile", entry.get("profile_id"))
                if entry.get("profile_id")
                else "—",
                population=str(entry.get("population") or "—"),  # its ID in stats.json
                value=value_text(entry),
                n=str(n) if isinstance(n, int) else "—",
                role=str(entry.get("role") or "—"),
            )
        )
    return rows


def stats_view(themes: ThemeManager) -> tuple[QtWidgets.QTableView, TableModel]:
    """``stats.json`` as a plain read-only list in the design's table look (no statistic computed here)."""
    view = QtWidgets.QTableView()
    model = TableModel([(key, title) for key, title, _ in STATS_COLUMNS])
    view.setModel(model)
    style_table(
        view,
        themes,
        TableLook(columns=[style for _, _, style in STATS_COLUMNS], key="results.stats"),
    )
    view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    return view, model


class Chips(Themed):
    """``.behind``: a label, a chip per run (26 px, its name and a chevron) while they fit, then "+ N more".
    ``opened`` carries the clicked run's record."""

    opened = QtCore.Signal(object)
    CHIP = TextStyle(12, 400, 18)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.label = "Runs behind it"
        self.records: list[dict[str, Any]] = []
        self.total = 0
        self.boxes: list[tuple[QtCore.QRectF, dict[str, Any]]] = []
        self.setFixedHeight(52)
        self.setMouseTracking(True)

    def set_runs(
        self,
        label: str,
        records: Sequence[dict[str, Any]],
        total: int,
        name: Callable[[dict[str, Any]], str],
    ) -> None:
        self.label, self.records, self.total, self.name = label, list(records), total, name
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        inside = any(box.contains(event.position()) for box, _ in self.boxes)
        self.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
            if inside
            else QtCore.Qt.CursorShape.ArrowCursor
        )

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        for box, record in self.boxes:
            if box.contains(event.position()):
                self.opened.emit(record)
                return

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.theme
        painter.fillRect(QtCore.QRect(0, 0, self.width(), 1), theme.color("border.line"))
        middle = 1 + 51 / 2
        x = 20.0
        x += (
            draw_text(painter, x, middle - 9, self.label, SMALL_BOLD, theme.color("text.secondary"))
            + 4
        )
        self.boxes = []
        right = self.width() - 20
        shown = 0
        for record in self.records:
            text = self.name(record)
            width = 8 + self.CHIP.advance(text) + 4 + 12 + 8
            more = self.total - shown - 1
            reserve = 4 + SMALL.advance(f"+ {more} more") + 8 if more > 0 else 0
            if x + 8 + width + reserve > right:
                break
            x += 8
            box = QtCore.QRectF(x, middle - 13, width, 26)
            painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
            painter.setBrush(theme.color("surface.card"))
            painter.drawPath(rounded_path(box.adjusted(0.5, 0.5, -0.5, -0.5), 3.5))
            draw_text(painter, x + 8, middle - 9, text, self.CHIP, theme.color("text.body"))
            gui_icons.draw(
                painter,
                "cright",
                theme.color("text.tertiary"),
                QtCore.QPointF(x + 8 + self.CHIP.advance(text) + 4, middle - 6),
                12,
            )
            self.boxes.append((box, record))
            x += width
            shown += 1
        rest = self.total - shown
        if rest > 0:
            x += 8 + 4
            draw_text(
                painter, x, middle - 9, f"+ {rest} more", SMALL, theme.color("text.secondary")
            )


# ---------- coverage traces (27) ----------


class TracesView(Themed):
    """Screen 27's body: the legend row with the axis note, the n line, the chart and "Still exploring" at
    four times (each run's end from the reply; counted, not estimated)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.chart = TracesChart(themes, self)
        # (colour, label, dash pattern in px); a None colour is the carry-forward entry
        self.legend: list[tuple[QtGui.QColor | None, str, tuple[float, ...]]] = []
        self.note = ""
        self.caption: list[tuple[str, str]] = []
        self.support: list[tuple[int, float]] = []
        self.message = "Choose a world"

    def set_reply(
        self,
        reply: Mapping[str, Any],
        world: str,
        profile_name: Callable[[str], str],
        styles: Mapping[str, Style] | None = None,
        explorer_name: Callable[[str], str] | None = None,
    ) -> None:
        """``styles``: the study's profiles' colours and dashes (``TracesChart.set_runs``); ``explorer_name``
        names a compared method in its profiles' labels ("method · profile", contract C2)."""
        runs = list(reply.get("runs") or [])
        self.message = "" if runs else "No reported run of this study in " + world
        self.chart.set_runs(runs, styles)
        profiles: dict[str, dict[str, Any]] = {}
        for run in runs:
            profiles.setdefault(str(run.get("profile_id")), run)
        # ours first, then the compared methods in registry order, each method's Default before its variants
        ordered = sorted(
            profiles.values(),
            key=lambda r: (*self.chart.trace_style(r), str(r.get("profile_id"))),
        )
        self.legend = [
            (
                self.chart.colour(run),
                trace_label(run, profile_name, explorer_name),
                self.chart.dashes(run),
            )
            for run in ordered
        ]
        words = {
            "completed": "completed",
            "stuck": "stuck",
            "explorer_failure": "explorer failure",
        }
        found = {
            "explorer_failure"
            if explorer_failure(r.get("outcome"), r.get("failure_source"))
            else str(r.get("outcome"))
            for r in runs
            if measured_as(r.get("outcome"), r.get("failure_source"), r.get("motion_started"))
            in ("completed", "stuck")
        }
        carried = [words[o] for o in words if o in found]
        if carried:
            self.legend.append(
                (None, " or ".join(carried).capitalize() + ", carried forward to L", ())
            )
        axis = self.chart.axis()
        self.note = "Simulated time since first movement" + (
            f" · compressed after {duration(axis.split)}" if axis.compressed else ""
        )
        per_profile = {
            name: sum(1 for r in runs if r.get("profile_id") == profile)
            for profile, name in ((p, profile_name(p)) for p in profiles)
        }
        excluded = reply.get("excluded") or {}
        self.caption = [(str(len(runs)), f" reported runs in {world}")]
        if len(set(per_profile.values())) == 1 and per_profile:
            self.caption.append(("", f"{next(iter(per_profile.values()))} per profile"))
        else:
            self.caption.append(("", ", ".join(f"{n} {name}" for name, n in per_profile.items())))
        if excluded:
            for kind, count in excluded.items():
                self.caption.append(("", left_out_text(str(kind), int(count))))
        else:
            self.caption.append(("", "no exclusions in this world"))
        self.caption.append(("", "click a line to open its run"))
        self._layout()
        self.update()

    def _support(self) -> None:
        """Runs still exploring at four times of the chart's current axis (it changes with the width)."""
        runs = self.chart.runs
        axis = self.chart.axis()
        times = [0.0, axis.split / 3, 2 * axis.split / 3, axis.split]
        self.support = [(sum(run_end(r) > t for r in runs), t) for t in times] if runs else []

    def set_message(self, text: str) -> None:
        self.message = text
        self.chart.set_runs([])
        self.legend, self.caption, self.support, self.note = [], [], [], ""
        self.update()

    def _layout(self) -> None:
        width, height = self.width(), self.height()
        top = 52 + 36 + 14
        # the plot's padding above, and 15 px between the axis labels and the support row
        self.chart.setGeometry(20, top, width - 40, max(0, height - top - 40 - 15))
        self._support()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme, width = self.theme, self.width()
        line = theme.color("border.line")
        painter.fillRect(QtCore.QRect(0, 51, width, 1), line)
        painter.fillRect(QtCore.QRect(0, 52 + 35, width, 1), line)
        x = 20.0
        # the entries that fit before the axis note; the rest as "+ N more" (six methods and variants)
        right = width - 20 - (SMALL.advance(self.note) + 16 if self.note else 0)
        for index, (colour, label, dashes) in enumerate(self.legend):
            more = len(self.legend) - index - 1
            reserve = SMALL.advance(f"+ {more} more") + 16 if more > 0 else 0
            if x + 27 + CAPTION.advance(label) + reserve > right:
                left = len(self.legend) - index
                draw_text(painter, x, 18, f"+ {left} more", SMALL, theme.color("text.secondary"))
                break
            pen = QtGui.QPen(colour or theme.color("text.secondary"), 2 if colour else 1.6)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
            if colour is None:
                pen.setDashPattern([4 / 1.6, 3 / 1.6])
            elif dashes:
                pen.setDashPattern([value / 2 for value in dashes])  # a variant (C2)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(x + 1, 26), QtCore.QPointF(x + 19, 26))
            x += 20 + 7
            x += draw_text(painter, x, 17, label, CAPTION, theme.color("text.body")) + 16
        if self.note:
            draw_text(
                painter,
                width - 20 - SMALL.advance(self.note),
                17,
                self.note,
                SMALL,
                theme.color("text.secondary"),
            )
        if self.message:
            draw_text(painter, 20, 52 + 9, self.message, CAPTION, theme.color("text.secondary"))
            return
        paint_parts(painter, 20, 52 + 9, self.caption, theme)
        top = self.height() - 40
        painter.fillRect(QtCore.QRect(20, top, width - 40, 1), line)
        draw_text(painter, 20, top + 11, "Still exploring", SMALL, theme.color("text.secondary"))
        column = (width - 40 - 110) / 4
        for index, (count, at) in enumerate(self.support):
            x = 20 + 110 + index * column
            x += (
                draw_text(painter, x, top + 11, str(count), SMALL_BOLD, theme.color("text.ink")) + 4
            )
            draw_text(
                painter, x, top + 11, f"at {duration(at)}", SMALL, theme.color("text.secondary")
            )


# ---------- a refused analysis (28, RO-22) ----------


class RefusalView(Themed):
    """Why the analysis wrote nothing for the newest export: the reason, what it means, and each value of the
    column that differs with its campaigns, runs and profiles."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.document: Mapping[str, Any] = {}
        self.rows: list[list[str]] = []

    def set_refusal(self, document: Mapping[str, Any], name: Callable[[str, Any], str]) -> None:
        self.document = document
        self.rows = []
        for group in document.get("groups") or []:
            runs = group.get("runs") or []
            value = str(group.get("value", ""))
            self.rows.append(
                [
                    value[:12] if re.fullmatch(r"[0-9a-f]{16,}", value) else value,
                    ", ".join(sorted({name("campaign", r.get("campaign_id")) for r in runs})),
                    str(len(runs)),
                    ", ".join(sorted({name("profile", r.get("profile_id")) for r in runs})),
                ]
            )
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = self.theme
        document = self.document
        x, y = 32.0, 28.0
        gui_icons.draw(painter, "ban", theme.color("state.warn"), QtCore.QPointF(x, y), 20)
        reason = str(document.get("reason", ""))
        room = min(780.0, self.width() - 64 - 30)
        lines = wrap_text("Not produced: " + reason, REFUSAL_TITLE, room)
        for line in lines:
            draw_text(painter, x + 30, y, line, REFUSAL_TITLE, theme.color("text.ink"))
            y += 20
        y += 14
        stratum = document.get("stratum")
        if document.get(
            "column"
        ):  # runs that do not fit together; any other refusal is its reason alone
            column = humanize(str(document["column"])).lower()
            text = (
                f"The runs of {stratum} differ in {column}, "
                if stratum
                else f"The final runs differ in {column}, "
            ) + (
                "so the analysis pooled nothing and wrote no output for this export. Strata and versions "
                "are never aggregated across: analyse the runs that belong together, or rerun the others."
            )
        else:
            text = "The analysis wrote no output for this export."
        for line in wrap_text(text, TextStyle(13, 400, 19), min(780.0, self.width() - 64)):
            draw_text(painter, x, y, line, TextStyle(13, 400, 19), theme.color("text.body"))
            y += 19
        if not self.rows:
            return
        y += 14
        headers = [
            humanize(str(document.get("column") or "Value")),
            "Campaigns",
            "Runs",
            "Profiles",
        ]
        widths = [190.0, 300.0, 70.0, 300.0]
        table = QtCore.QRectF(x, y, sum(widths), 36 + 40 * len(self.rows))
        painter.setPen(QtGui.QPen(theme.color("border.line"), 1))
        painter.setBrush(theme.color("surface.card"))
        painter.drawPath(rounded_path(table.adjusted(0.5, 0.5, -0.5, -0.5), 5.5))
        painter.save()
        painter.setClipPath(rounded_path(table, 6))
        painter.fillRect(QtCore.QRectF(x, y, table.width(), 36), theme.color("surface.sunken"))
        painter.restore()
        left = x
        for header, width in zip(headers, widths, strict=True):
            draw_text(
                painter,
                left + 12,
                y + 9,
                header,
                TextStyle(12, 500, 18),
                theme.color("text.secondary"),
            )
            left += width
        y += 36
        for row in self.rows:
            painter.fillRect(QtCore.QRectF(x, y, table.width(), 1), theme.color("border.line"))
            left = x
            for index, (value, width) in enumerate(zip(row, widths, strict=True)):
                style = MONO_TEXT if index == 0 else TABLE_TEXT
                draw_text(
                    painter,
                    left + 12,
                    y + 11,
                    style.elide(value, width - 24),
                    style,
                    theme.color("text.body"),
                )
                left += width
            y += 40


# ---------- the viewer card ----------

PDF, STATS, TRACES, REFUSAL = 0, 1, 2, 3


class Caption(Themed):
    """``.cap``: the view's n line (36 px) over a rule."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.parts: list[tuple[str, str]] = []
        self.setFixedHeight(36)

    def set_parts(self, parts: Sequence[tuple[str, str]]) -> None:
        self.parts = list(parts)
        self.setAccessibleName(" · ".join((a + b) for a, b in self.parts))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(QtCore.QRect(0, 35, self.width(), 1), self.color("border.line"))
        paint_parts(painter, 20, (35 - 18) / 2, self.parts, self.theme)


class ViewerCard(Card):
    """``.card.vw``: the head (title, a muted note, the endpoint, effect or world choice, a grouped study's world
    group, the file and Open file), the n line, the view and "Runs behind it"."""

    variant_changed = QtCore.Signal(int)
    group_changed = QtCore.Signal(int)
    open_file = QtCore.Signal()

    GROUP_WIDTH = 260  # at most; "F3 Warehouse aisles · page 1 of 2" fits

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.head = CardHead(themes, "", self)
        self.note = TextLine(themes, "", BODY, "text.secondary")
        self.head.add(self.note)
        self.head.row.add_stretch()
        self.choice_holder = QtWidgets.QWidget()
        self.choice: SegmentedControl | None = None
        self.head.add(self.choice_holder)
        # A grouped study's world group and page (plans/worlds/I8 §3.1), beside the variant choice.
        self.group = KeySelect(themes)
        self.group.setAccessibleName("World group")
        self.group.setFixedWidth(self.GROUP_WIDTH)
        self.group.currentIndexChanged.connect(self._group_picked)
        self.head.row.add(self.group, width=self.GROUP_WIDTH)
        self.group.hide()
        self._filling = False
        self.file = TextLine(themes, "", MONO_TEXT, "text.secondary")
        self.head.add(self.file, gap=16)
        self.open = Button(themes, "Open file", icon="aupright", variant="ghost", small=True)
        self.open.clicked.connect(self.open_file.emit)
        self.head.add(self.open)
        self.caption = Caption(themes, self)
        self.stack = QtWidgets.QStackedWidget(self)
        self.well = Well(themes)
        self.stats, self.stats_model = stats_view(themes)
        self.traces = TracesView(themes)
        self.refusal = RefusalView(themes)
        for widget in (self.well, self.stats, self.traces, self.refusal):
            self.stack.addWidget(widget)
        self.chips = Chips(themes, self)
        self.mode = PDF

    def _group_picked(self, index: int) -> None:
        if not self._filling and index >= 0:
            self.group_changed.emit(index)

    def set_groups(self, groups: Sequence[str], index: int) -> None:
        """The world group dropdown's entries (none: hidden)."""
        self._filling = True
        try:
            if [self.group.itemText(i) for i in range(self.group.count())] != list(groups):
                self.group.clear()
                self.group.addItems(list(groups))
            self.group.setCurrentIndex(index if 0 <= index < len(groups) else 0)
        finally:
            self._filling = False
        self.group.setVisible(bool(groups))
        if groups:  # as wide as its longest entry, at most GROUP_WIDTH
            width = min(self.group.sizeHint().width(), self.GROUP_WIDTH)
            self.group.setFixedWidth(width)
            self.head.row.set_width(self.group, width)
            # The file name gives way to the dropdown so the title keeps its room; Open file names it.
            self.file.setVisible(False)
            self.open.setToolTip(self.file.text())
        else:
            self.open.setToolTip("")
        self.head.row.relayout()

    def set_head(
        self,
        title: str,
        note: str,
        file: str,
        choices: Sequence[str],
        index: int,
        groups: Sequence[str] = (),
        group_index: int = 0,
    ) -> None:
        self.head.title.setText(title)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.file.setText(file)
        self.file.setVisible(bool(file))
        self.open.setVisible(bool(file))
        if self.choice is not None:
            self.choice.deleteLater()
            self.choice = None
        if choices:
            self.choice = SegmentedControl(
                self.themes, [(text, None) for text in choices], parent=self.choice_holder
            )
            self.choice.set_index(index)
            self.choice.changed.connect(self.variant_changed.emit)
            self.choice.show()
            width = math.ceil(self.choice.natural_width())
            self.choice.setGeometry(0, 0, width, 32)
            self.choice_holder.setFixedSize(width, 32)
            self.head.row.set_width(self.choice_holder, width)
        self.choice_holder.setVisible(bool(choices))
        self.set_groups(groups, group_index)

    def set_mode(self, mode: int, *, runs: bool = True) -> None:
        """Show ``mode``'s view; a view without runs behind it (``runs`` False) has no n line and no chips."""
        self.mode = mode
        self.stack.setCurrentIndex(mode)
        self.caption.setVisible(mode in (PDF, STATS) and runs)
        self.chips.setVisible(mode in (PDF, STATS, REFUSAL) and runs)
        self._layout()

    def show_error(self, text: str) -> None:
        """A read error on the caption line, shown also on a view without an n line."""
        self.caption.set_parts([("", text)])
        self.caption.show()
        self._layout()

    def _layout(self) -> None:
        width, height = self.width(), self.height()
        self.head.setGeometry(1, 1, width - 2, 52)
        top = 53
        if self.caption.isVisible():
            self.caption.setGeometry(1, top, width - 2, 36)
            top += 36
        bottom = height - 1 - (52 if self.chips.isVisible() else 0)
        self.stack.setGeometry(1, top, width - 2, max(0, bottom - top))
        self.chips.setGeometry(1, bottom, width - 2, 52)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout()


# ---------- the tab ----------


@dataclass
class AnalysisState:
    """The analysis the tab shows: its directory and listing, the sidecars read so far, and a refusal of the
    newest export (RO-22)."""

    directory: Path | None = None
    modified_ns: int | None = None
    current: bool = False
    outputs: dict[str, str] = field(default_factory=dict)
    sidecars: dict[str, dict[str, Any]] = field(default_factory=dict)
    sidecar_errors: dict[str, str] = field(default_factory=dict)  # sidecars that could not be read
    refusal: dict[str, Any] | None = None
    refusal_ns: int | None = None
    unreadable: str | None = (
        None  # the directory has no readable analysis.json (an interrupted run)
    )
    # the list analysis.json declares (1.3.0 on); None: KINDS
    views: tuple[OutputKind, ...] | None = None


class StudyTab(QtWidgets.QWidget):
    """The Study tab: the study card (360 px) and the viewer, 16 px apart. ``set_detail`` takes ``study.get``;
    the tab then lists the study's analyses, reads the chosen one's listing and sidecars, and reads a view's
    files when it opens. Replies of an earlier study or view are dropped by their token."""

    opened = QtCore.Signal(object)  # a run's record (a chip, a trace)

    def __init__(
        self,
        themes: ThemeManager,
        transport: Transport,
        names: Callable[[str, Any], str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.themes, self.transport, self.names = themes, transport, names
        self.card = StudyCard(themes, self)
        self.viewer = ViewerCard(themes, self)
        self.detail: dict[str, Any] = {}
        self.study_dir: Path | None = None
        self.export: dict[str, Any] | None = None
        self.analysis = AnalysisState()
        self.kinds: list[OutputKind] = []
        self.current = -1
        self.variants: dict[
            str, str
        ] = {}  # kind code -> chosen variant ("R_90"), kept across analyses
        self.head_grouped = False  # the shown head has the world group dropdown
        self.worlds: list[str] = []
        self.world = 0  # the traces' stratum, an index into strata()
        self.stratum: dict[
            str, str
        ] = {}  # its filters, kept while strata() grows as the runs arrive
        self.token = ""  # the study's reads
        self.runs_token = ""  # the study's final runs, paged (a new export or analysis keeps them)
        self.view_token = ""  # the open view's reads
        self.view_files: list[str] = []
        self.picked: str | None = (
            None  # the kind the researcher chose (kept while the list changes)
        )
        self.chip_ids: list[str] = []
        self.chip_label = "Runs behind it"
        self.chip_records: dict[str, dict[str, Any]] = {}
        # the study's final runs (chip names and order)
        self.records: dict[str, dict[str, Any]] = {}
        # the page's export and analysis (the Analysis row's state and log)
        self.jobs: Jobs | None = None
        self.poll = QtCore.QTimer(self)
        self.poll.setSingleShot(True)
        self.poll.setInterval(POLL_MS)
        self.poll.timeout.connect(self._request_traces)
        self.card.list.chosen.connect(self.choose)
        self.card.facts.acted.connect(self._show_log)
        self.viewer.variant_changed.connect(self._variant)
        self.viewer.group_changed.connect(self._group)
        self.viewer.open_file.connect(self._open_file)
        self.viewer.chips.opened.connect(self.opened.emit)
        self.viewer.traces.chart.opened.connect(self._open_attempt)
        self.viewer.traces.chart.describe = self._describe_trace

    # the study

    def set_detail(self, detail: Mapping[str, Any]) -> None:
        """A ``study.get`` reply (ready): the card's facts, then the study's analyses."""
        changed = detail.get("study_id") != self.detail.get("study_id")
        if changed:
            self.poll.stop()
        self.detail = dict(detail)
        export = newest_export(detail)
        if changed or (export or {}).get("sums_sha256") != (self.export or {}).get("sums_sha256"):
            self.export = export
            self.study_dir = (
                Path(export["path"]).parent.parent if export and export.get("path") else None
            )
            self.analysis = AnalysisState()
            self.token = uuid.uuid4().hex
            if changed:
                self.current, self.variants, self.world, self.picked = -1, {}, 0, None
                self.stratum = {}
            self.worlds = [
                str(v["value"]) for v in (detail.get("selectors") or {}).get("world_id", [])
            ]
            if changed:
                self.records = {}
                self.runs_token = uuid.uuid4().hex
                self._request_runs(0)
            if self.study_dir is not None:
                self.transport.submit(
                    "study-tab-list:" + self.token,
                    "local.list",
                    {"path": str(self.study_dir / "analysis")},
                )
        self.refresh_card()
        self._rows()

    def reload_analysis(self) -> None:
        """The study's analyses again (an analysis finished or was refused)."""
        if self.study_dir is not None:
            self.token = uuid.uuid4().hex
            self.transport.submit(
                "study-tab-list:" + self.token,
                "local.list",
                {"path": str(self.study_dir / "analysis")},
            )

    def _show_log(self, row: int) -> None:
        if self.jobs is not None:
            self.jobs.show(self.jobs.job(str(self.detail.get("study_id"))))

    def _request_runs(self, offset: int) -> None:
        self.transport.submit(
            "study-tab-runs:" + self.runs_token,
            "results.query",
            {
                "table": "runs",
                "filters": {"study_id": self.detail.get("study_id"), "is_final": True},
                "offset": offset,
                "limit": 250,
            },
        )

    def refresh_card(self) -> None:
        detail = self.detail
        study = str(detail.get("study_id") or "")
        levels = detail.get("evidence_levels") or ["exploratory"]
        confirmatory = "confirmatory" in levels
        look = BadgeLook(
            "Confirmatory · declared" if confirmatory else "Exploratory", "neutral", "flag"
        )
        campaigns = list(detail.get("campaigns") or [])
        planned = next((c for c in reversed(campaigns) if c.get("plan")), None)
        facts = [
            KeyValue(
                "Plan",
                Path(str(planned["plan"])).name if planned else "None",
                value_style=MONO_TEXT if planned else None,
            ),
            KeyValue(
                "Plan hash",
                str(planned.get("plan_sha256") or "")[:8] if planned else "—",
                value_style=MONO_TEXT if planned else None,
            ),
            KeyValue(
                "Campaigns",
                ", ".join(str(c.get("name") or str(c.get("id"))[:8]) for c in campaigns) or "—",
            ),
        ]
        export = self.export
        facts.append(
            KeyValue(
                "Export",
                f"{str(export.get('sums_sha256') or '')[:8]} · {short_date(export.get('utc_ns'))}"
                if export
                else "None yet",
            )
        )
        analysis = self.analysis
        if analysis.refusal is not None and (
            analysis.directory is None or (analysis.refusal_ns or 0) > (analysis.modified_ns or 0)
        ):
            facts.append(
                KeyValue(
                    "Analysis", f"{short_date(analysis.refusal_ns)} · refused", mark="state.warn"
                )
            )
        elif analysis.directory is not None and analysis.unreadable is not None:
            facts.append(
                KeyValue(
                    "Analysis",
                    f"{short_date(analysis.modified_ns)} · unreadable",
                    mark="state.warn",
                )
            )
        elif analysis.directory is not None:
            state = "current" if analysis.current else "out of date"
            facts.append(
                KeyValue(
                    "Analysis",
                    f"{short_date(analysis.modified_ns)} · {state}",
                    mark="data.completed" if analysis.current else "data.timeLimit",
                )
            )
        else:
            facts.append(KeyValue("Analysis", "Not run yet" if export else "—"))
        job = self.jobs.job(study) if self.jobs is not None else None
        if job is not None and job.kind == "analysis" and job.state == "running":
            facts[-1] = KeyValue(
                "Analysis",
                f"Running {duration(job.elapsed())}",
                mark="accent.primary",
                action="Show log",
            )
        elif job is not None:  # this session's export or analysis keeps its log at hand
            facts[-1] = replace(facts[-1], action="Show log")
        self.card.set_study("Unassigned" if study == "unassigned" else study, look, facts)
        warnings = [str(w) for w in detail.get("warnings") or []]
        if analysis.directory is not None and analysis.unreadable is not None:
            warnings.insert(
                0,
                f"The analysis directory {analysis.directory.name} has no readable analysis.json "
                f"({analysis.unreadable}), as an interrupted run leaves it. Remove it, then run the "
                "analysis again.",
            )
        if (
            analysis.directory is not None
            and analysis.unreadable is None
            and not analysis.current
            and export
        ):
            used = export_hash(analysis.directory.name)
            warnings.insert(
                0,
                f"The analysis used export {used}; export {str(export.get('sums_sha256'))[:8]} is newer. "
                "Run the analysis again.",
            )
        if analysis.refusal is not None:
            warnings.insert(
                0, "The analysis refused the newest export: " + str(analysis.refusal.get("reason"))
            )
        self.card.set_warnings(warnings)

    def _rows(self) -> None:
        outputs = self.analysis.outputs
        listed = KINDS if self.analysis.views is None else self.analysis.views
        kinds = [kind for kind in listed if kind.available(outputs)]
        if self.analysis.refusal is not None:
            kinds.insert(0, REFUSAL_KIND)
        previous = self.kinds[self.current].code if 0 <= self.current < len(self.kinds) else None
        self.kinds = kinds
        self.card.list.set_rows([(kind, row_count(kind, self._sidecar(kind))) for kind in kinds])
        self.card.relayout()
        codes = [kind.code for kind in kinds]
        wanted = self.picked if self.picked in codes else codes[0] if codes else None
        if wanted is None:
            return
        self.current = codes.index(wanted)
        self.card.list.set_current(self.current)
        if wanted != previous:
            self._open_view()

    def _sidecar(self, kind: OutputKind) -> dict[str, Any] | None:
        files = self._files(kind)
        return self.analysis.sidecars.get(sidecar_of(files[0])) if files else None

    def _grouping(self, kind: OutputKind) -> dict[str, VariantGroup] | None:
        """A grouped view's variants (file -> group) in its order, from analysis.json's ``variant_groups``, else
        from each figure's sidecar ``group``; None for an ungrouped view, or while a sidecar is still unread
        (the head then lists the variants as before and is rebuilt when the last one arrives)."""
        variants = kind.variant_files(self.analysis.outputs)
        if not kind.order or not variants:
            return None
        declared = dict(kind.groups)
        stem = kind.variants.rsplit("/", 1)[-1]
        found: dict[str, VariantGroup] = {}
        for file in variants:
            group = declared.get(variant_of(kind, file))
            if group is None:
                sidecar = self.analysis.sidecars.get(sidecar_of(file))
                group = variant_group((sidecar or {}).get("group"), stem)
            if group is None:
                return None
            found[file] = group
        return found

    def _files(self, kind: OutputKind) -> list[str]:
        """The files the kind's view shows: the chosen variant and what goes under it, or its fixed files."""
        outputs = self.analysis.outputs
        variants = kind.variant_files(outputs)
        if variants:
            keys = [variant_of(kind, f) for f in variants]
            wanted = self.variants.get(kind.code, "R_90")
            if wanted in keys:
                chosen = variants[keys.index(wanted)]
            else:  # the default: R_90 (in a grouped view its first group), else the first in order
                grouping = self._grouping(kind) or {}
                matching = [f for f in variants if f in grouping and grouping[f].base == wanted]
                chosen = matching[0] if matching else variants[0]
            return [chosen, *[f for f in kind.then if f in outputs]]
        return [f for f in kind.files if f in outputs]

    # reading the analysis

    def _stale(self, kind: str, token: str) -> bool:
        """A reply for an earlier study (its listing and runs) or an earlier view (the output it showed)."""
        if kind == "runs":
            return token != self.runs_token
        if kind in ("list", "index", "sidecar", "refusal"):
            return token != self.token
        return kind in ("pdf", "stats", "traces", "chip", "open") and token != self.view_token

    def received(self, tag: str, result: Any) -> bool:
        if not tag.startswith("study-tab-"):
            return False
        kind, _, rest = tag.removeprefix("study-tab-").partition(":")
        token, _, path = rest.partition(":")
        if self._stale(kind, token):
            return True
        if kind == "runs":
            for row in result.get("rows") or []:
                self.records[str(row.get("attempt_id"))] = dict(row)
            if isinstance(result.get("next_offset"), int):
                self._request_runs(result["next_offset"])
            elif 0 <= self.current < len(self.kinds) and self.kinds[self.current].code == "C(t)":
                self._open_view()  # the traces' strata now come from every final run
            self._show_chips()
        elif kind == "list":
            self._listed(result.get("entries") or [])
        elif kind == "index":
            self._indexed(result)
        elif kind == "sidecar":
            self.analysis.sidecars[path] = _json(result.get("text"))
            self._rows()
            files = (
                self._files(self.kinds[self.current]) if 0 <= self.current < len(self.kinds) else []
            )
            if files and sidecar_of(files[0]) == path:
                self._show_counts()
            kind_now = self.kinds[self.current] if 0 <= self.current < len(self.kinds) else None
            if (
                kind_now is not None
                and kind_now.order
                and not self.head_grouped
                and self._grouping(kind_now) is not None
            ):  # the view's last sidecar arrived: its head gets the group dropdown
                self._open_view()
        elif kind == "refusal":
            self.analysis.refusal = _json(result.get("text")) or None
            self.refresh_card()
            self._rows()
        elif kind == "pdf":
            self.viewer.well.pages.add(bytes(result.get("bytes") or b""))
        elif kind == "stats":
            document = _json(bytes(result.get("bytes") or b"").decode("utf-8", "replace"))
            self.viewer.stats_model.replace(stats_rows(document, self.names))
        elif kind == "traces":
            self._traces(result)
        elif kind in ("chip", "open"):
            record = result.get("record") if isinstance(result, Mapping) else None
            if record and kind == "open":
                self.opened.emit(dict(record))
            elif record:
                self.chip_records[str(record.get("attempt_id"))] = dict(record)
                self._show_chips()
        return True

    def failed(self, tag: str, error: str) -> bool:
        if not tag.startswith("study-tab-"):
            return False
        kind, _, rest = tag.removeprefix("study-tab-").partition(":")
        if self._stale(kind, rest.partition(":")[0]):
            return True
        path = rest.partition(":")[2]
        if kind == "list":  # no analysis directory yet
            self.refresh_card()
            self._rows()
        elif kind == "sidecar":
            self.analysis.sidecar_errors[path] = error  # shown whenever its view opens
            self._show_counts()
        elif kind == "stats":
            self.viewer.stats_model.replace([])
            self.viewer.show_error("stats.json could not be read: " + error)
        elif kind == "refusal":
            self.analysis.refusal = dict(reason="the refusal file could not be read: " + error)
            self.refresh_card()
            self._rows()
        elif kind == "index":
            self._unreadable(error)
        elif kind == "pdf":
            self.viewer.well.pages.clear("The output could not be read: " + error)
        elif kind == "traces":
            self.viewer.traces.set_message("The traces could not be read: " + error)
        return True

    def _listed(self, entries: Sequence[Mapping[str, Any]]) -> None:
        export = self.export or {}
        hash8 = str(export.get("sums_sha256") or "")[:8]
        # the directory a running analysis has claimed is empty until it finishes: not an analysis yet
        job = self.jobs.job(str(self.detail.get("study_id"))) if self.jobs is not None else None
        writing = (
            "-" + job.export if job and job.kind == "analysis" and job.state == "running" else None
        )
        analyses = [
            e
            for e in entries
            if e.get("directory")
            and ANALYSIS_NAME.match(str(e.get("name")))
            and not (writing and str(e.get("name")).endswith(writing))
        ]
        # current: made from the newest export, with or without a coding file; the newest of them is shown
        current = [e for e in analyses if export_hash(str(e["name"])) == hash8]
        chosen = max(current or analyses, key=lambda e: e.get("modified_ns") or 0, default=None)
        refusals = [
            e
            for e in entries
            if (found := REFUSAL_NAME.match(str(e.get("name"))))
            and found.group(1).endswith("-" + hash8)
        ]
        refusal = max(refusals, key=lambda e: e.get("modified_ns") or 0, default=None)
        assert self.study_dir is not None
        state = AnalysisState()
        if chosen is not None:
            state.directory = self.study_dir / "analysis" / str(chosen["name"])
            state.modified_ns, state.current = chosen.get("modified_ns"), bool(current)
            self.transport.submit(
                "study-tab-index:" + self.token,
                "local.read",
                {"path": str(state.directory / "analysis.json")},
            )
        if refusal is not None and (
            chosen is None
            or not current
            or (refusal.get("modified_ns") or 0) > (chosen.get("modified_ns") or 0)
        ):
            state.refusal_ns = refusal.get("modified_ns")
            self.transport.submit(
                "study-tab-refusal:" + self.token,
                "local.read",
                {"path": str(self.study_dir / "analysis" / str(refusal["name"]))},
            )
        self.analysis = state
        self.refresh_card()
        self._rows()

    def _indexed(self, result: Mapping[str, Any]) -> None:
        document = _json(result.get("text"))
        found = document.get("outputs")
        if not isinstance(found, dict):
            self._unreadable("analysis.json lists no outputs")
            return
        outputs: dict[str, str] = found
        self.analysis.outputs = dict(outputs)
        self.analysis.views = kinds_from_views(document.get("views"))
        directory = self.analysis.directory
        for name in sorted(outputs):
            if name.endswith(".sidecar.json") and directory is not None:
                self.transport.submit(
                    f"study-tab-sidecar:{self.token}:{name}",
                    "local.read",
                    {"path": str(directory / name)},
                )
        self._rows()

    def _unreadable(self, error: str) -> None:
        """The chosen analysis directory holds no readable index: shown as such, never as an analysis."""
        self.analysis.unreadable = error
        self.analysis.outputs = {}
        self.refresh_card()
        self._rows()

    # views

    def choose(self, index: int) -> None:
        if not 0 <= index < len(self.kinds):
            return
        self.picked = self.kinds[index].code
        self.current = index
        self.card.list.set_current(index)
        self._open_view()

    def _variant(self, index: int) -> None:
        kind = self.kinds[self.current]
        if kind.code == "C(t)":
            strata = self.strata()
            self.world = index
            self.stratum = strata[index][1] if 0 <= index < len(strata) else {}
        else:
            files = kind.variant_files(self.analysis.outputs)
            grouping = self._grouping(kind)
            if grouping is not None:
                # A base variant: the current group and page when it has them, else its first in order.
                bases = list(dict.fromkeys(group.base for group in grouping.values()))
                if not 0 <= index < len(bases):
                    return
                current = grouping[self._files(kind)[0]]
                same = [f for f in files if grouping[f].base == bases[index]]
                place = (current.key, current.page)
                chosen = next(
                    (f for f in same if (grouping[f].key, grouping[f].page) == place), same[0]
                )
                self.variants[kind.code] = variant_of(kind, chosen)
            elif 0 <= index < len(files):
                self.variants[kind.code] = variant_of(kind, files[index])
        self._open_view()

    def _group(self, index: int) -> None:
        """A world group and page: the current base variant in that group."""
        if not 0 <= self.current < len(self.kinds):
            return
        kind = self.kinds[self.current]
        grouping = self._grouping(kind)
        if grouping is None:
            return
        base = grouping[self._files(kind)[0]].base
        same = [f for f in kind.variant_files(self.analysis.outputs) if grouping[f].base == base]
        if 0 <= index < len(same):
            self.variants[kind.code] = variant_of(kind, same[index])
            self._open_view()

    def _head_choices(
        self, kind: OutputKind, files: Sequence[str]
    ) -> tuple[list[str], int, list[str], int]:
        """The variant choice and the world group dropdown of a view: for a grouped view the base variants
        (none for a view of fixed files) and the current base's groups and pages, in the analysis's order;
        otherwise every variant and no dropdown."""
        variants = kind.variant_files(self.analysis.outputs)
        grouping = self._grouping(kind)
        if grouping is None or not files or files[0] not in grouping:
            choices = [kind.variant_label(variant_of(kind, f)) for f in variants]
            index = variants.index(files[0]) if files and files[0] in variants else 0
            return choices, index, [], 0
        current = grouping[files[0]]
        bases = list(dict.fromkeys(group.base for group in grouping.values()))
        labels = {
            g.base: g.base_label or VARIANT_NAMES.get(g.base, g.base) for g in grouping.values()
        }
        choices = [labels[b] for b in bases] if any(bases) else []
        same = [f for f in variants if grouping[f].base == current.base]
        return (
            choices,
            bases.index(current.base),
            [grouping[f].label for f in same],
            same.index(files[0]),
        )

    def _open_view(self) -> None:
        kind = self.kinds[self.current]
        self.poll.stop()  # a pending traces poll belongs to the view being left
        self.view_token = uuid.uuid4().hex
        self.chip_records = {}
        viewer = self.viewer
        if kind is REFUSAL_KIND:
            refusal = self.analysis.refusal or {}
            viewer.set_head("Analysis refused", "", "", [], 0)
            viewer.refusal.set_refusal(refusal, self.names)
            viewer.set_mode(REFUSAL)
            ids = [
                str(r.get("attempt_id"))
                for g in refusal.get("groups") or []
                for r in g.get("runs") or []
            ]
            self._chips("Runs it refused", ids)
            return
        if kind.code == "C(t)":
            strata = self.strata()
            # the chosen stratum again: the first choice holding its filters (a world alone matches its first)
            self.world = next(
                (
                    i
                    for i, (_, filters) in enumerate(strata)
                    if all(filters.get(k) == v for k, v in self.stratum.items())
                ),
                min(self.world, max(0, len(strata) - 1)),
            )
            names = [label for label, _ in strata]
            viewer.set_head(
                "Coverage traces", "Raw runs, no band or statistic", "", names, self.world
            )
            viewer.set_mode(TRACES)
            self._request_traces()
            return
        files = self._files(kind)
        choices, index, groups, group_index = self._head_choices(kind, files)
        sidecar = self.analysis.sidecars.get(sidecar_of(files[0])) if files else None
        title = self._title(kind, files, sidecar)
        file = Path(files[0]).name if files else ""
        viewer.set_head(title, "", file, choices, index, groups, group_index)
        self.head_grouped = bool(groups)
        self.view_files = files
        if kind.is_stats:
            viewer.set_mode(STATS, runs=kind.runs)
            viewer.stats_model.replace([])
            if self.analysis.directory is not None:
                self.transport.submit(
                    "study-tab-stats:" + self.view_token,
                    "local.read_bytes",
                    {"path": str(self.analysis.directory / "stats.json")},
                )
        else:
            viewer.set_mode(PDF, runs=kind.runs)
            viewer.well.pages.clear("" if files else "This output was not produced")
            viewer.well.verticalScrollBar().setValue(0)
            for file in files:
                if file.endswith(".pdf") and self.analysis.directory is not None:
                    self.transport.submit(
                        "study-tab-pdf:" + self.view_token,
                        "local.read_bytes",
                        {"path": str(self.analysis.directory / file)},
                    )
        self._show_counts()

    def _title(
        self, kind: OutputKind, files: Sequence[str], sidecar: Mapping[str, Any] | None
    ) -> str:
        grouping = self._grouping(kind) if kind.variants and files else None
        if (
            grouping is not None and files[0] in grouping
        ):  # the group shows in the dropdown, not the title
            group = grouping[files[0]]
            if not group.base:
                return f"{kind.code} · {kind.name}"
            label = group.base_label or VARIANT_NAMES.get(group.base, group.base)
            if kind.code == "F2/T3":
                return "F2 / T3 · Attainment of " + (
                    f"{label} coverage" if label.endswith("%") else label.lower()
                )
            return f"{kind.code} · {kind.name} · {label}"
        if kind.variants and files:
            label = kind.variant_label(variant_of(kind, files[0]))
            if kind.code == "F2/T3":
                return "F2 / T3 · Attainment of " + (
                    f"{label} coverage" if label.endswith("%") else label.lower()
                )
            return f"{kind.code} · {kind.name} · {label}"
        if sidecar and sidecar.get("title"):
            return f"{kind.code} · {sidecar['title']}"
        return f"{kind.code} · {kind.name}"

    def _show_counts(self) -> None:
        if not 0 <= self.current < len(self.kinds):
            return
        kind = self.kinds[self.current]
        if kind is REFUSAL_KIND or kind.code == "C(t)":
            return
        files = self._files(kind)
        sidecar = self.analysis.sidecars.get(sidecar_of(files[0])) if files else None
        if sidecar:
            self.viewer.head.title.setText(self._title(kind, files, sidecar))
            self.viewer.head.row.relayout()
        if not kind.runs:  # no n line and no runs behind it
            self._chips("Runs behind it", [])
            return
        error = self.analysis.sidecar_errors.get(sidecar_of(files[0])) if files else None
        self.viewer.caption.set_parts(
            n_line((sidecar or {}).get("counts"))
            if sidecar
            else [("", "The sidecar could not be read: " + error)]
            if error
            else [("", "Reading the sidecar…")]
        )
        if sidecar:
            self._chips("Runs behind it", [str(a) for a in sidecar.get("attempt_ids") or []])

    def _chips(self, label: str, attempt_ids: Sequence[str]) -> None:
        self.chip_ids, self.chip_label = list(attempt_ids), label
        self._show_chips()

    def _record(self, identity: str) -> dict[str, Any] | None:
        return self.records.get(identity) or self.chip_records.get(identity) or None

    def _show_chips(self) -> None:
        """The runs in the study's order (world, block, Default first); a run outside the study's final runs
        (an earlier attempt) is read on its own."""

        def order(identity: str) -> tuple[Any, ...]:
            record = self._record(identity)
            if record is None:
                return (1, "", 0, True, identity)
            repetition = record.get("repetition")
            return (
                0,
                self.names("world", record.get("world_id")),
                repetition if isinstance(repetition, int) else 0,
                not record.get("is_baseline"),
                self.names("profile", record.get("profile_id")),
            )

        ordered = sorted(self.chip_ids, key=order)
        for identity in ordered[:CHIPS]:
            if self._record(identity) is None and identity not in self.chip_records:
                self.chip_records[identity] = {}  # requested once
                self.transport.submit(
                    "study-tab-chip:" + self.view_token,
                    "results.get",
                    {"table": "attempts", "id": identity, "include": ["record"]},
                )
        records = [record for i in ordered[:CHIPS] if (record := self._record(i))]
        self.viewer.chips.set_runs(
            self.chip_label,
            records,
            len(self.chip_ids),
            lambda record: run_label(record, self.names),
        )

    def _open_file(self) -> None:
        if self.analysis.directory is not None and self.view_files:
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(self.analysis.directory / self.view_files[0]))
            )

    # traces

    def strata(self) -> list[tuple[str, dict[str, str]]]:
        """The traces' choices as (label, ``study.traces`` filters): one per world, split by robot and spawn when
        the study has several, since strata are never drawn together. The robot and spawn come from the study's
        final runs; until they are read, the worlds alone."""
        found = {
            (str(r.get("world_id")), str(r.get("robot_id")), str(r.get("spawn_id")))
            for r in self.records.values()
            if r.get("is_final")
        }
        if not found:
            return [(self.names("world", w), {"world_id": w}) for w in self.worlds]
        order = {world: index for index, world in enumerate(self.worlds)}
        robots = len({robot for _, robot, _ in found}) > 1
        spawns = any(
            len({spawn for w, _, spawn in found if w == world}) > 1 for world, _, _ in found
        )
        choices = []
        for world, robot, spawn in sorted(found, key=lambda s: (order.get(s[0], len(order)), s)):
            parts = (
                self.names("robot", robot) if robots else "",
                self.names("world", world),
                self.names("spawn", spawn) + " spawn" if spawns else "",
            )
            label = " · ".join(part for part in parts if part)
            choices.append((label, dict(world_id=world, robot_id=robot, spawn_id=spawn)))
        return choices

    def _request_traces(self) -> None:
        strata = self.strata()
        if not strata:
            self.viewer.traces.set_message("The study has no runs to draw yet")
            return
        self.transport.submit(
            "study-tab-traces:" + self.view_token,
            "study.traces",
            {
                "study_id": self.detail.get("study_id"),
                **strata[min(self.world, len(strata) - 1)][1],
            },
        )

    def _traces(self, result: Mapping[str, Any]) -> None:
        state = result.get("state")
        if state == "loading":
            self.poll.start()
            return
        strata = self.strata()
        world = strata[min(self.world, len(strata) - 1)][0] if strata else ""
        if state == "failed":
            self.viewer.traces.set_message(
                str(result.get("reason") or "The traces could not be read")
            )
            return
        runs = result.get("runs") or []
        keys = {run.get("comparison_key") for run in runs}
        limits = {run.get("time_limit_s") for run in runs}
        if (
            len(keys) > 1 or len(limits) > 1
        ):  # runs that are not comparable, which the analysis refuses too
            what = f"{len(keys)} comparison keys" if len(keys) > 1 else f"{len(limits)} time limits"
            self.viewer.traces.set_message(
                f"The runs of {world} hold {what}, so they are not drawn together"
            )
            return
        # colours by explorer and dashes by variant over the study's final runs, the same in every world (C2)
        chart = self.viewer.traces.chart
        styles = trace_styles(
            (r for r in self.records.values() if r.get("is_final")), chart.explorers
        )
        self.viewer.traces.set_reply(
            result,
            world,
            lambda profile: self.names("profile", profile),
            styles or None,
            lambda explorer: self.names("explorer", explorer),
        )

    def _describe_trace(self, run: Mapping[str, Any]) -> str:
        repetition = run.get("repetition")
        completion = number(run.get("completion_s"))
        coverage = coverage_text(run.get("coverage_at_limit_pct"))
        outcome = (
            f"Completed {duration(completion)}"
            if run.get("outcome") == "completed"
            else "Explorer failure"
            if explorer_failure(run.get("outcome"), run.get("failure_source"))
            else humanize(str(run.get("outcome")))
        )
        return "\n".join(
            (
                trace_label(
                    run,
                    lambda profile: self.names("profile", profile),
                    lambda explorer: self.names("explorer", explorer),
                )
                + (f" · block {repetition + 1}" if isinstance(repetition, int) else ""),
                outcome + (f" · C(L) {coverage} %" if coverage else ""),
                f"{self.names('robot', run.get('robot_id'))} · {self.names('spawn', run.get('spawn_id'))} spawn",
                "Click to open the run",
            )
        )

    def _open_attempt(self, attempt_id: str) -> None:
        record = self.records.get(attempt_id)
        if record is not None:
            self.opened.emit(dict(record))
            return
        self.transport.submit(
            "study-tab-open:" + self.view_token,
            "results.get",
            {"table": "attempts", "id": attempt_id, "include": ["record"]},
        )

    # layout

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.card.setGeometry(0, 0, 360, self.height())
        self.viewer.setGeometry(360 + 16, 0, max(0, self.width() - 376), self.height())

    def close_timers(self) -> None:
        self.poll.stop()

    def set_explorers(self, explorers: Sequence[str]) -> None:
        """The catalog's run-record explorer IDs in registry order: the traces' colour order (C2)."""
        self.viewer.traces.chart.explorers = list(explorers)


def trace_label(
    run: Mapping[str, Any],
    profile_name: Callable[[str], str],
    explorer_name: Callable[[str], str] | None,
) -> str:
    """A trace's profile as "method · profile" for a compared method (contract C2), unless the profile's name
    already starts with the method's ("TDLE Default"); our explorer's profiles by their names."""
    name = profile_name(str(run.get("profile_id")))
    explorer = run.get("explorer_id")
    if explorer_name is None or explorer in (None, OUR_EXPLORER):
        return name
    method = explorer_name(str(explorer))
    return name if name.startswith(method) else f"{method} · {name}"


def _json(text: Any) -> dict[str, Any]:
    try:
        value = json.loads(text) if isinstance(text, str) else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}
