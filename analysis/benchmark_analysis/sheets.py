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

"""Tables typeset as PDF sheets (RESULTS_OVERHAUL RO-19; FIGURE_STYLE §9).

The sheet is what the Study tab shows and a paper-format preview of the table: the title above ("TABLE <name>"),
booktabs rules (top, middle, bottom; space between groups, no vertical rules), a repeated first-column value
written once per group, numeric columns right-aligned, headers wrapped to their column, the notes below. It is set
at 8 pt in the preset's font at the journal's double-column width; a table too wide for it is set smaller, never
below 7 pt (FIGURE_STYLE §2), and only then gets a wider page, which ``check_pdf`` reports.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from .style import PRESETS, MM_PER_INCH
from .tables import DASH, Table

POINTS_PER_INCH = 72.0
SIZES_PT = (8.0, 7.5, 7.0)  # tried in order until the table fits the double column
NOTE_PT = 7.0
MARGIN_PT = 4.0
COLUMN_GAP_PT = 6.0
ROW_FACTOR = 1.45  # row height per font size
HEADER_LINES = 3  # a header wraps to at most this many lines


@dataclass(frozen=True, slots=True)
class Layout:
    size: float
    widths: list[float]
    headers: list[list[str]]
    right: list[bool]

    @property
    def width(self) -> float:
        return sum(self.widths) + COLUMN_GAP_PT * (len(self.widths) - 1)


class Measure:
    """Text widths in points, from Matplotlib's own renderer and the current style's font."""

    def __init__(self) -> None:
        # Matplotlib's Agg renderer has no type annotations; it measures exactly as the PDF will be set
        self._renderer = RendererAgg(1, 1, POINTS_PER_INCH)  # type: ignore[no-untyped-call]

    def width(self, text: str, size: float, bold: bool = False) -> float:
        if not text:
            return 0.0
        prop = FontProperties(size=size, weight="bold" if bold else "normal")
        width, _, _ = self._renderer.get_text_width_height_descent(text, prop, ismath=False)  # type: ignore[no-untyped-call]
        return float(width)


def _numeric(cells: Sequence[str]) -> bool:
    """A column is right-aligned when most of its cells start like a number."""
    starts = [c[:1] for c in cells if c and c != DASH]
    return bool(starts) and sum(s.isdigit() or s in "−-<>" for s in starts) * 2 > len(starts)


def _wrap(text: str, limit: float, measure: Measure, size: float) -> list[str]:
    """Greedy word wrap to ``limit`` points, at most ``HEADER_LINES`` lines (the last line takes the rest)."""
    lines: list[str] = []
    for word in text.split():
        if (
            lines
            and len(lines) < HEADER_LINES
            and measure.width(f"{lines[-1]} {word}", size, True) <= limit
        ):
            lines[-1] = f"{lines[-1]} {word}"
        elif lines and len(lines) >= HEADER_LINES:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    return lines or [""]


def _layout(table: Table, size: float, measure: Measure) -> Layout:
    columns = list(zip(*table.rows, strict=True)) if table.rows else [() for _ in table.columns]
    widths, headers = [], []
    for header, cells in zip(table.columns, columns, strict=True):
        body = max((measure.width(c, size) for c in cells), default=0.0)
        longest_word = max((measure.width(w, size, True) for w in header.split()), default=0.0)
        limit = max(body, longest_word)
        lines = _wrap(header, limit, measure, size)
        widths.append(max(limit, *(measure.width(line, size, True) for line in lines)))
        headers.append(lines)
    right = [index >= 1 and _numeric(cells) for index, cells in enumerate(columns)]
    return Layout(size, widths, headers, right)


def _note_lines(notes: Sequence[str], width: float, measure: Measure) -> list[str]:
    lines: list[str] = []
    for note in notes:
        current = ""
        for word in note.split():
            trial = f"{current} {word}".strip()
            if current and measure.width(trial, NOTE_PT) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    return lines


def _without_empty_counts(table: Table) -> Table:
    """A count table's sheet leaves out the count columns that are 0 in every row and names them in a note;
    its CSV and LaTeX keep every column."""
    if not table.sparse_counts or not table.rows:
        return table
    keep = [
        i for i in range(len(table.columns)) if i < 2 or any(row[i] != "0" for row in table.rows)
    ]
    dropped = [table.columns[i] for i in range(len(table.columns)) if i not in keep]
    if not dropped:
        return table
    return Table(
        table.name,
        table.title,
        [table.columns[i] for i in keep],
        [[row[i] for i in keep] for row in table.rows],
        [*table.notes, "No run in any row (columns left out): " + ", ".join(dropped) + "."],
        table.attempt_ids,
    )


def sheet(table: Table, preset: str) -> tuple[Figure, float]:
    """The table on a page of the preset's double-column width (or wider when even 7 pt does not fit); returns
    the figure and the width it was meant to have, for ``check_pdf``."""
    table = _without_empty_counts(table)
    measure = Measure()
    target = PRESETS[preset].widths_mm["double"] / MM_PER_INCH * POINTS_PER_INCH
    layout = next(
        (
            lay
            for size in SIZES_PT
            if (lay := _layout(table, size, measure)).width <= target - 2 * MARGIN_PT
        ),
        _layout(table, SIZES_PT[-1], measure),
    )
    page_width = max(target, layout.width + 2 * MARGIN_PT)
    row = layout.size * ROW_FACTOR
    notes = _note_lines(table.notes, page_width - 2 * MARGIN_PT, measure)
    header_rows = max(len(h) for h in layout.headers)
    groups = sum(1 for i, r in enumerate(table.rows) if i and r[0] != table.rows[i - 1][0])
    height = (
        MARGIN_PT
        + row * 1.6  # title
        + row * header_rows
        + row * len(table.rows)
        + groups * row * 0.35
        + 8.0  # rules and their padding
        + (NOTE_PT * ROW_FACTOR * len(notes) + 4.0 if notes else 0.0)
        + MARGIN_PT
    )
    return _draw(table, layout, notes, page_width, height, header_rows, measure), target


def _draw(
    table: Table,
    layout: Layout,
    notes: list[str],
    page_width: float,
    height: float,
    header_rows: int,
    measure: Measure,
) -> Figure:
    """Place every text and rule; coordinates are points from the page's top-left corner."""
    size, row = layout.size, layout.size * ROW_FACTOR
    figure = Figure(
        figsize=(page_width / POINTS_PER_INCH, height / POINTS_PER_INCH), dpi=POINTS_PER_INCH
    )
    figure.patch.set_facecolor("white")

    def text(
        x: float, y: float, value: str, size: float, *, right: bool = False, bold: bool = False
    ) -> None:
        figure.text(
            x / page_width,
            1 - y / height,
            value,
            va="top",
            ha="right" if right else "left",
            fontsize=size,
            fontweight="bold" if bold else "normal",
        )

    def rule(y: float, width: float) -> None:
        level = 1 - y / height
        x0, x1 = MARGIN_PT / page_width, 1 - MARGIN_PT / page_width
        figure.add_artist(
            Line2D([x0, x1], [level, level], transform=figure.transFigure, color="black", lw=width)
        )

    def cell(column: int, y: float, value: str, *, bold: bool = False) -> None:
        right = layout.right[column]
        x = xs[column] + (layout.widths[column] if right else 0.0)
        text(x, y, value, size, right=right, bold=bold)

    xs = [MARGIN_PT + sum(layout.widths[:i]) + COLUMN_GAP_PT * i for i in range(len(layout.widths))]
    y = MARGIN_PT
    label = f"TABLE {table.name}"
    text(MARGIN_PT, y, label, size, bold=True)
    text(MARGIN_PT + measure.width(label, size, True) + size * 0.6, y, table.title, size)
    y += row * 1.6
    rule(y - row * 0.3, 0.8)
    for column, lines in enumerate(layout.headers):
        offset = (header_rows - len(lines)) * row  # headers sit on the bottom line, as in booktabs
        for number, line in enumerate(lines):
            cell(column, y + offset + number * row, line, bold=True)
    y += header_rows * row
    rule(y + row * 0.05, 0.5)
    y += row * 0.25
    previous = None
    for index, cells in enumerate(table.rows):
        if index and cells[0] != previous:
            y += row * 0.35  # \addlinespace between groups
        for column, value in enumerate(cells):
            cell(column, y, "" if column == 0 and cells[0] == previous else value)
        previous = cells[0]
        y += row
    rule(y + row * 0.1, 0.8)
    y += row * 0.5
    for line in notes:
        text(MARGIN_PT, y, line, NOTE_PT)
        y += NOTE_PT * ROW_FACTOR
    return figure


def write(table: Table, path: Path, preset: str) -> float:
    """Write the sheet; returns the width (pt) it was meant to have."""
    figure, target = sheet(table, preset)
    figure.savefig(path, metadata={"CreationDate": None})
    return target
