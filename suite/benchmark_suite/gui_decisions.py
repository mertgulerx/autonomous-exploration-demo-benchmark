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

"""The Monitor's Decisions view column (docs/suite-gui-design/MONITOR_INSIGHT.md §3, screens 19–21).

"Explorer decisions" with the analysis status; the figures (goal switches over the last 60 simulation s and the
run, cache hits of the last tick and the run); "Why this goal" (the analysis target, its values and one sentence
on what put it first); the candidate table, linked with the map (hovering a row marks its candidate, a click pins
it). Without analysis the goal switches stay and one note replaces the panels. The explorer's orders are its
data, never "best".
"""

from collections.abc import Mapping
import math
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_cells import PaintedColumn, PaintedTable
from .gui_dialogs import wrap_text
from .gui_insight import Candidate, Decisions, candidate_facts, gain_text, number_text
from .gui_marks import KeyValue, KeyValueList
from .gui_style import fixed, number
from .gui_theme import ThemeManager, snap
from .gui_widgets import CARD_TITLE, TextStyle, Themed, draw_text

HEAD_TEXT = TextStyle(12.5, 400, 18)
SECTION_TEXT = TextStyle(12, 600, 18)
SECTION_NOTE = TextStyle(12, 400, 18, tabular=True)
FIGURE_LABEL = TextStyle(11.5, 400, 14)
FIGURE_VALUE = TextStyle(14, 600, 18, tabular=True)
FIGURE_UNIT = TextStyle(12.5, 400, 18)
FIGURE_NONE = TextStyle(14, 400, 18)
WHY_TITLE = TextStyle(13.5, 600, 20)
WHY_SUB = TextStyle(12.5, 400, 20)
NOTE_TEXT = TextStyle(12.5, 400, 17)
NOTE_BODY = TextStyle(12.5, 400, 18)
NOTE_TITLE = TextStyle(13, 600, 18)
KV_LABEL = TextStyle(12.5, 400, 18)
CELL_TEXT = TextStyle(13, 400, 18, tabular=True)
UNIT_TEXT = TextStyle(12, 400, 18)
MATCH_M = 0.10
# "Why this goal": the headline, the two key-value lists (three 32 px rows) and the sentence under them.
HEADLINE_TOP, LISTS_TOP, LISTS_HEIGHT = 144, 172, 96
SENTENCE_TOP = LISTS_TOP + LISTS_HEIGHT + 6
NO_ANALYSIS_TITLE = "No explorer analysis for this run"
NO_ANALYSIS = (
    "Runs recorded before the analysis existed, or whose analysis did not start, have no frontier layers, "
    "route, chunk cache or candidate table. Goal switches come from Nav2 and stay available."
)


def figure(value: Any, digits: int, scale: float = 1.0) -> str | None:
    """A figure scaled (cache hits: a share as %) and formatted as ``gui_style.fixed``."""
    found = number(value)
    return fixed(found * scale, digits) if found is not None else None


def why_sentence(decisions: Decisions, target: Candidate) -> str:
    """What put the target first, from the explorer's own data (no judgement)."""
    lowest = next((c for c in decisions.candidates if c.score_order == 1), None)
    greedy = decisions.greedy_first
    mode = decisions.mode or ""
    count = len(decisions.route)
    if mode == "mrtsp/dp":
        cost = number_text(decisions.route_cost, 3)
        route = f"the cheapest {count}-frontier route ({cost})"
        if lowest is not None and lowest.id == target.id:
            sentence = f"Lowest start cost, and first on {route}."
        elif lowest is not None:
            sentence = f"First on {route}; #{lowest.id} has the lowest own start cost."
        else:
            sentence = f"First on {route}."
        if greedy == target.id:
            sentence += " Greedy would choose it too."
        elif greedy is not None:
            sentence += f" Greedy would choose #{greedy}."
        return sentence
    if lowest is not None and lowest.id == target.id:
        return "First in the greedy order, with the lowest start cost."
    if lowest is not None:
        return f"First in the greedy order; #{lowest.id} has the lowest own start cost."
    return "First in the greedy order."


def route_sort(candidate: Candidate) -> tuple[int, int, int]:
    """Sorted by route, then score order (absent last), then id."""
    large = 1 << 30
    return (
        candidate.route_position or large,
        candidate.score_order or large,
        candidate.id,
    )


class CandidateTable(PaintedTable):
    """``.tbl.dense`` (32 px rows): Frontier (a filled dot for pool members, hollow outside the pool), Route,
    Order, Pool, Start cost, Gain, Distance; "—" in ``text.tertiary`` off the route or outside the pool. The
    pinned row is selected (fill and ink bar); hovering a row marks its candidate on the map."""

    hovered = QtCore.Signal(object)  # candidate id or None
    pinned = QtCore.Signal(object)

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(
            themes,
            [
                PaintedColumn("Frontier", None, padding=4, padding_right=8),
                PaintedColumn("Route", 58, padding=8, padding_right=8, right=True),
                PaintedColumn("Order", 58, padding=8, padding_right=8, right=True),
                PaintedColumn("Pool", 52, padding=8, padding_right=8, right=True),
                PaintedColumn("Start cost", 86, padding=8, padding_right=8, right=True),
                PaintedColumn("Gain", 64, padding=8, padding_right=8, right=True),
                PaintedColumn("Distance", 84, padding=8, padding_right=8, right=True),
            ],
            parent,
            header_height=32,
        )
        self.candidates: list[Candidate] = []
        self.pool_known = False
        self.pinned_id: int | None = None
        self.hovered_id: int | None = None
        self.setAccessibleName("Frontier candidates")

    def set_candidates(self, decisions: Decisions | None) -> None:
        self.candidates = sorted(decisions.candidates, key=route_sort) if decisions else []
        self.pool_known = decisions is not None and decisions.pool_size > 0
        self.rows_changed(len(self.candidates))

    def set_marks(self, pinned: int | None, hovered: int | None) -> None:
        if (pinned, hovered) != (self.pinned_id, self.hovered_id):
            self.pinned_id, self.hovered_id = pinned, hovered
            self.viewport().update()

    def row_height(self, index: int) -> int:
        return 32

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self.row_at(event.position().y())
        identity = self.candidates[index].id if index >= 0 else None
        if identity != self.hovered_id:
            self.hovered_id = identity
            self.hovered.emit(identity)
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self.hovered_id is not None:
            self.hovered_id = None
            self.hovered.emit(None)
            self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        index = self.row_at(event.position().y())
        if index >= 0 and event.button() == QtCore.Qt.MouseButton.LeftButton:
            identity = self.candidates[index].id
            self.pinned.emit(None if identity == self.pinned_id else identity)

    def paint_row(self, painter: QtGui.QPainter, index: int, rect: QtCore.QRect) -> None:
        theme = self.themes.current
        candidate = self.candidates[index]
        content = rect.height() - 1
        if candidate.id == self.pinned_id:
            painter.fillRect(rect, theme.color("surface.selected"))
            painter.fillRect(QtCore.QRect(0, rect.top(), 2, content), theme.color("text.ink"))
        elif candidate.id == self.hovered_id:
            painter.fillRect(rect, theme.color("surface.hover"))
        painter.fillRect(
            QtCore.QRect(0, rect.bottom(), rect.width(), 1), theme.color("border.line")
        )
        spans = self.column_spans()
        top = rect.top() + (content - 18) / 2
        ink, off = theme.color("text.body"), theme.color("text.tertiary")
        left, _ = spans[0]
        # ``.adot``: 8 px, filled for pool members (and when the solver keeps no pool), hollow outside it.
        dot = QtCore.QRectF(left + 4, rect.top() + (content - 8) / 2, 8, 8)
        analysis = theme.color("data.analysis")
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.pool_known and candidate.pool_rank is None:
            painter.setPen(QtGui.QPen(analysis, 1.4))
            painter.setBrush(theme.color("surface.card"))
            painter.drawEllipse(dot.adjusted(0.7, 0.7, -0.7, -0.7))
        else:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(analysis)
            painter.drawEllipse(dot)
        painter.restore()
        draw_text(painter, left + 4 + 8 + 6, top, f"#{candidate.id}", CELL_TEXT, ink)
        gain, _ = gain_text(candidate)
        values: list[tuple[str, str]] = [
            (str(candidate.route_position) if candidate.route_position else "—", ""),
            (str(candidate.score_order) if candidate.score_order else "—", ""),
            (str(candidate.pool_rank) if candidate.pool_rank else "—", ""),
            (number_text(candidate.start_cost, 3), ""),
            (gain, ""),
            (
                "—" if candidate.distance_m is None else f"{candidate.distance_m:.1f}",
                "" if candidate.distance_m is None else "m",
            ),
        ]
        for column, (text, unit) in enumerate(values, start=1):
            left, width = spans[column]
            right = left + width - self.columns[column].padding_right
            unit_width = (2 + UNIT_TEXT.advance(unit)) if unit else 0.0
            x = right - unit_width - CELL_TEXT.advance(text)
            draw_text(painter, x, top, text, CELL_TEXT, off if text == "—" else ink)
            if unit:
                draw_text(painter, right - unit_width + 2, top, unit, UNIT_TEXT, off)


class DecisionsColumn(Themed):
    """``.dec`` (padding 0 20): see the module docstring."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.lists = [
            KeyValueList(themes, row_height=32, label_style=KV_LABEL, last_open=True, parent=self)
            for _ in range(2)
        ]
        self.table = CandidateTable(themes, self)
        self.decisions: Decisions | None = None
        self.figures: Mapping[str, Any] = {}
        self.status, self.status_live = "", False
        self.available: bool | None = None
        self.sentence: list[str] = []
        self.reason = ""  # why the run has no analysis, when known

    # data

    def set_state(
        self,
        decisions: Decisions | None,
        figures: Mapping[str, Any],
        status: str,
        live: bool,
        available: bool | None,
        reason: str = "",
    ) -> None:
        changed = decisions is not self.decisions
        self.decisions, self.figures = decisions, figures
        self.status, self.status_live, self.available = status, live, available
        self.reason = reason
        if changed:
            self.table.set_candidates(decisions)
        self._update_why()
        self._layout()
        self.update()

    def set_marks(self, pinned: int | None, hovered: int | None) -> None:
        self.table.set_marks(pinned, hovered)

    def shown(self) -> Candidate | None:
        """The candidate "Why this goal" explains: the analysis target."""
        decisions = self.decisions
        return decisions.candidate(decisions.target) if decisions is not None else None

    def _update_why(self) -> None:
        target = self.shown()
        width = self.width() - 40
        if target is None or self.decisions is None:
            for widget in self.lists:
                widget.set_rows([])
            self.sentence = []
            return
        gain, unit = gain_text(target)
        distance = self.figures.get("target_goal_distance_m")
        received, token = "—", "text.ink"
        if isinstance(distance, (int, float)) and math.isfinite(distance):
            if distance <= MATCH_M:
                received = "Matches"
            else:
                received, token = f"Differs by {distance:.1f} m", "state.warn"
        self.lists[0].set_rows(
            [
                KeyValue("Start cost", number_text(target.start_cost, 3)),
                KeyValue("Distance term", number_text(target.path_cost, 3)),
                KeyValue("Time term", number_text(target.time_cost, 3)),
            ]
        )
        self.lists[1].set_rows(
            [
                KeyValue("Gain (size)", gain, unit),
                KeyValue(
                    "Distance",
                    "—" if target.distance_m is None else f"{target.distance_m:.1f}",
                    "" if target.distance_m is None else "m",
                ),
                KeyValue(
                    "Received goal",
                    received,
                    tooltip="Distance between the analysis target and the goal Nav2 received last",
                    value_token=token,
                ),
            ]
        )
        self.sentence = wrap_text(why_sentence(self.decisions, target), NOTE_TEXT, max(80.0, width))

    # geometry (``.dec``: head 52, figures 56, sections with 12 px top padding)

    def analysis_shown(self) -> bool:
        return self.available is not False

    def _layout(self) -> None:
        width = self.width() - 40.0
        half = (width - 32) / 2
        shown = self.analysis_shown() and self.shown() is not None
        for column, widget in enumerate(self.lists):
            widget.setVisible(shown)
            widget.setGeometry(
                round(20 + column * (half + 32)), LISTS_TOP, max(0, round(half)), LISTS_HEIGHT
            )
        table_top = self.table_top()
        self.table.setVisible(self.analysis_shown())
        self.table.setGeometry(
            20, table_top, max(0, round(width)), max(0, self.height() - table_top)
        )

    def table_top(self) -> int:
        """Below the candidates' title (and the "Why this goal" section above it)."""
        why_bottom = (
            HEADLINE_TOP
            + 20
            + (8 + LISTS_HEIGHT + 6 + 17 * len(self.sentence) if self.shown() else 0)
        )
        return round(why_bottom + 14 + 1 + 12 + 18 + 8)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_why()
        self._layout()

    # painting

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        theme = self.theme
        right = self.width() - 20.0
        rule = theme.color("border.line")
        secondary, ink = theme.color("text.secondary"), theme.color("text.ink")
        draw_text(painter, 20, 15.5, "Explorer decisions", CARD_TITLE, ink)
        if self.status:
            text = HEAD_TEXT.elide(self.status, right - 200)
            x = right - HEAD_TEXT.advance(text)
            draw_text(painter, x, 16.5, text, HEAD_TEXT, secondary)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("data.completed" if self.status_live else "text.tertiary"))
            painter.drawEllipse(QtCore.QRectF(x - 8 - 7, 16.5 + 5.5, 7, 7))
            painter.restore()
        painter.fillRect(QtCore.QRectF(20, 51, right - 20, 1), rule)
        self._paint_figures(painter, 52, right)
        painter.fillRect(QtCore.QRectF(20, 107, right - 20, 1), rule)
        if not self.analysis_shown():
            self._paint_note(painter, 108 + 18, right)
            return
        draw_text(painter, 20, 120, "Why this goal", SECTION_TEXT, secondary)
        target, decisions = self.shown(), self.decisions
        if target is None or decisions is None:
            text = (
                "Waiting for the analysis"
                if decisions is None
                else "No frontier route at this tick"
            )
            draw_text(painter, 20, HEADLINE_TOP, text, WHY_SUB, theme.color("text.tertiary"))
        else:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.color("data.analysis"))
            painter.drawEllipse(QtCore.QRectF(20, HEADLINE_TOP + 6, 8, 8))
            painter.restore()
            title, facts = candidate_facts(target, decisions)
            x = 20 + 8 + 14 + draw_text(painter, 20 + 8 + 14, HEADLINE_TOP, title, WHY_TITLE, ink)
            draw_text(
                painter,
                x + 8,
                HEADLINE_TOP,
                WHY_SUB.elide(facts, right - x - 8),
                WHY_SUB,
                secondary,
            )
            for index, line in enumerate(self.sentence):
                draw_text(painter, 20, SENTENCE_TOP + 17 * index, line, NOTE_TEXT, secondary)
        top = self.table_top() - 8 - 18 - 12 - 1
        painter.fillRect(QtCore.QRectF(20, top, right - 20, 1), rule)
        draw_text(painter, 20, top + 13, "Candidates", SECTION_TEXT, secondary)
        if decisions is not None:
            count = (
                f"{len(decisions.candidates)} · pool {decisions.pool_size} · "
                f"route {len(decisions.route)}"
            )
            x = right - SECTION_NOTE.advance(count)
            draw_text(painter, x, top + 13, count, SECTION_NOTE, secondary)

    def _paint_figures(self, painter: QtGui.QPainter, top: float, right: float) -> None:
        """``.dm``: four cells (rules between, 12 px inset): the label above the value and its unit."""
        theme = self.theme
        figures = self.figures
        cells = (
            ("Goal switches, 60 s", figure(figures.get("goal_switches_per_minute_60s"), 1), "/min"),
            ("Goal switches, run", figure(figures.get("goal_switches_per_minute_run"), 1), "/min"),
            ("Cache hits, last tick", figure(figures.get("cache_hit_last"), 0, 100), "%"),
            ("Cache hits, run", figure(figures.get("cache_hit_run"), 0, 100), "%"),
        )
        width = (right - 20) / 4
        cell_top = top + (55 - 35) / 2
        for index, (label, value, unit) in enumerate(cells):
            x = 20 + index * width
            if index:
                painter.fillRect(
                    QtCore.QRectF(snap(x), snap(cell_top), 1, 35), theme.color("border.line")
                )
            inset = 12 if index else 0
            draw_text(
                painter, x + inset, cell_top, label, FIGURE_LABEL, theme.color("text.secondary")
            )
            if value is None:
                draw_text(
                    painter,
                    x + inset,
                    cell_top + 17,
                    "—",
                    FIGURE_NONE,
                    theme.color("text.tertiary"),
                )
                continue
            advance = draw_text(
                painter, x + inset, cell_top + 17, value, FIGURE_VALUE, theme.color("text.ink")
            )
            unit_text = unit if unit == "/min" else " " + unit
            draw_text(
                painter,
                x + inset + advance,
                cell_top + 17,
                unit_text,
                FIGURE_UNIT,
                theme.color("text.secondary"),
            )

    def _paint_note(self, painter: QtGui.QPainter, top: float, right: float) -> None:
        """``.empty-note``: an info icon, the title and the explanation, then the run's own reason."""
        theme = self.theme
        gui_icons.draw(painter, "info", theme.color("text.tertiary"), QtCore.QPointF(20, top + 1))
        draw_text(
            painter, 20 + 16 + 10, top, NO_ANALYSIS_TITLE, NOTE_TITLE, theme.color("text.ink")
        )
        text = NO_ANALYSIS + (f" Here: {self.reason.rstrip('.')}." if self.reason else "")
        lines = wrap_text(text, NOTE_BODY, max(80.0, right - 46))
        for index, line in enumerate(lines):
            draw_text(
                painter,
                46,
                top + 18 + 18 * index,
                line,
                NOTE_BODY,
                theme.color("text.secondary"),
            )
