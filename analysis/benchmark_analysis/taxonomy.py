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

"""F23, the behaviour taxonomy, with its coding sheet and the coding file (FIGURE_EXTENSION FX-11, FX-19, FX6;
FIGURE_STYLE §3, §6, §9; MEASUREMENTS "Study analysis block").

Only a study whose plan declares a ``taxonomy`` (``analysis-block-v2``) gets these outputs: without a codebook
declared before coding there is nothing to code against, so nothing is drawn (no pointer figure either, as the
block's default says "no taxonomy figure"). The stall window is optional: without ``stall.window_s`` the crops
show the whole path.

- ``F23_sheet_<page>``: the coding sheet for the researcher, one crop per reported run (completed, time limit,
  no movement), 20 crops a page, blinded to profile and outcome: study order (stratum, profile, block execution
  order) shuffled with the plan's seed on its own stream (``SHEET_ORDER_STREAM``), each crop labelled with its
  number only (no world either: the map shows the place). A crop is the run's ground-truth path over the plan's
  ``stall.window_s`` before its end min(T_end, L) (the whole path to it without a window) on the coverage
  reference around it, coloured by the time before the end, with the window's first pose, the end and the goals
  active in the window (the shared goal marks, ``goals.map_style``); a run that never moved shows its last pose
  before movement could start. The key from crop number to attempt, stratum, profile
  and outcome is only in the page's sidecar.
- ``F23`` (only with a coding file, ``--coding``): (a) stacked absolute counts of the declared categories and
  "not coded" per stratum and profile over all reported runs, n per bar (FX-11); (b…) one example crop per
  category: its coded run of the earliest block in execution order (first attempt start, ``data.block_starts``,
  as F10 orders blocks). Descriptive counts: never in ``stats.json``, never cited.

The coding file is the researcher's YAML ``{coder, procedure, blinded, codes: [{attempt_id, category, note?}]}``,
read with a safe loader that refuses duplicate keys (``read_coding``); ``blinded`` (required, true or false) is the
coder's statement that the coding was done from the blinded, numbered sheet. Its SHA-256, coder, procedure,
``blinded`` and counts go into ``analysis.json`` and the sidecars of F23 and T9, whose captions follow ``blinded``,
and F23's sidecar lists each code and note. It is refused without a declared taxonomy, when malformed, for a
category the codebook lacks and (``check_runs``) for an attempt that is not a reported final run of the study.
Coding never changes an outcome.
"""

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from matplotlib import patheffects
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import yaml

from .data import TIME_EPSILON, Floats, Row, Stratum, Study, block_starts, run_end
from .errors import Refused
from .export import UniqueKeyLoader
from .figures import (
    Extent,
    FigureContext,
    FigureOut,
    MapCode,
    draw_reference,
    finish,
    load_reference,
    panel_title,
    reference_files,
    reference_name,
)
from .goals import (
    STATUS_LOOKS,
    Goal,
    PlacedGoal,
    close_enough_text,
    map_handles,
    map_style,
    place,
    read_goals,
)
from .settings import Settings
from .stats import random_stream
from .streams import Reads, Triple, world_from_map
from .style import EDGE, MM_PER_INCH, OUTCOMES, PRESETS, figure, page_note
from .tables import CATEGORY_NOTE, CODER_NOTE, NOT_CODED, blinding_text

CODING_KEYS = ("coder", "procedure", "blinded", "codes")
CODE_KEYS = ("attempt_id", "category", "note")
# the coding sheet's shuffled order: a stream of the plan's seed of its own (stats' streams are 1 to 5)
SHEET_ORDER_STREAM = 6
SHORT_ID = 8  # characters of an attempt ID in a crop label, more when needed to stay unique
SHEET_COLUMNS, SHEET_ROWS = 5, 4  # crops per page of the coding sheet (a page below 8.8 in)
EXAMPLE_COLUMNS = 5  # example crops per row in F23
LEGEND_COLUMNS = 5  # entries per legend row
# layout heights in mm at the double-column width
CROP_ROW_MM = 40.0  # one row of sheet crops with its number
SHEET_BOTTOM_MM = 26.0  # the sheet's colour bar, mark legend and evidence label
EXAMPLE_ROW_MM = 44.0  # one row of F23's example crops with title and label
EXAMPLE_BOTTOM_MM = 24.0  # F23's colour bar and mark legend under the example crops
BAR_ROW_MM = 5.5  # one bar of F23 (a)
BAR_AXIS_MM = 18.0  # F23 (a)'s title and x axis
LEGEND_ROW_MM = 4.5  # one row of a legend
MIN_SIDE_M = 8.0  # the smallest crop side, so a run that barely moved still shows its surroundings
PAD_M = 1.5  # free margin around the window's path and goals
GREYS = ("#4D4D4D", "#8C8C8C", "#CCCCCC", "#FFFFFF")
HATCHES = (None, "////", "\\\\\\\\", "xxxx", "....", "++++")
# a fill per declared category in declared order (up to 20, the block's maximum): greys and hatches, never a
# profile hue (FIGURE_STYLE §3); plain white is kept for "not coded"
CATEGORY_FILLS = tuple(
    (grey, hatch)
    for hatch in HATCHES
    for grey in GREYS
    if not (grey == "#FFFFFF" and hatch is None)
)
WINDOW_LABEL = "Simulated time before the run's end (s)"
# the crop's own marks; no goal status uses these shapes (goals.STATUS_LOOKS)
WINDOW_START: dict[str, Any] = dict(marker="*", ms=6.0, mew=0.5, mfc="#FFFFFF", mec="#000000")
END: dict[str, Any] = dict(marker="s", ms=4.0, mew=0.5, mfc="#000000", mec="#FFFFFF")
SHAPES = {
    "o": "circle",
    "^": "triangle",
    "X": "cross",
    "s": "square",
    "D": "diamond",
}


# The coding file ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Code:
    category: str
    note: str | None


@dataclass(frozen=True, slots=True)
class Coding:
    """The researcher's coding of runs against the plan's taxonomy (the file's SHA-256, never its path)."""

    sha256: str
    coder: str
    procedure: str
    blinded: bool  # the coder's statement: coded from the blinded, numbered sheet
    codes: Mapping[str, Code]  # attempt ID -> code

    def categories(self) -> dict[str, str]:
        """Attempt ID -> category, for the counts."""
        return {attempt: code.category for attempt, code in self.codes.items()}

    def notes(self) -> dict[str, str]:
        """Attempt ID -> the researcher's note, for the runs that have one."""
        return {a: code.note for a, code in sorted(self.codes.items()) if code.note is not None}

    def document(self) -> dict[str, Any]:
        """What ``analysis.json`` and the sidecars of the outputs that read the coding file record; ``blinded`` is
        the coding file's statement, copied."""
        return dict(
            sha256=self.sha256,
            coder=self.coder,
            procedure=self.procedure,
            codes=len(self.codes),
            notes=len(self.notes()),
            blinded=self.blinded,
        )


def _refuse(message: str) -> Refused:
    # a refusal of the call, not of the export: no refusal file is kept (errors.Refused)
    return Refused(f"coding file: {message}", kept=False)


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document[key]
    if type(value) is not str or not value.strip():
        raise _refuse(f"{key} must be a non-empty string")
    return value.strip()


def _identifier(item: Mapping[str, Any], key: str, number: int) -> str:
    value = item[key]
    if type(value) is not str or not value.strip():
        raise _refuse(f"code {number}: {key} must be a non-empty string")
    if value != value.strip():
        raise _refuse(f"code {number}: {key} {value!r} has surrounding whitespace")
    return value


def read_coding(path: Path, settings: Settings) -> Coding:
    """Read and check a coding file against the declared taxonomy; refused without one, when the file is not a
    mapping of exactly ``coder``, ``procedure``, ``blinded`` and ``codes`` (a non-empty list of {``attempt_id``,
    ``category``, optional ``note``}), names an attempt twice, or names a category the codebook lacks. The
    codebook's IDs and size were validated by the engine (``suite/benchmark_suite/analysis_block._taxonomy``)."""
    if settings.taxonomy is None:
        raise _refuse(
            "the study's plan declares no taxonomy (analysis-block-v2 key taxonomy, FIGURE_EXTENSION FX-19), "
            "so there is no codebook to code against"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _refuse(f"cannot be read ({exc.strerror})") from exc
    try:
        document = yaml.load(data.decode("utf-8"), UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise _refuse(f"not valid YAML: {exc}") from exc
    if not isinstance(document, dict) or set(document) != set(CODING_KEYS):
        found = sorted(map(str, document)) if isinstance(document, dict) else []
        missing = [key for key in CODING_KEYS if isinstance(document, dict) and key not in document]
        raise _refuse(
            "a mapping with exactly the keys coder, procedure, blinded and codes expected"
            + (f" (found {', '.join(found)}" if found else "")
            + (f"; missing {', '.join(missing)}" if found and missing else "")
            + (")" if found else "")
        )
    coder, procedure = _text(document, "coder"), _text(document, "procedure")
    blinded = document["blinded"]
    if type(blinded) is not bool:
        raise _refuse(
            "blinded must be true or false (true: coded from the blinded, numbered coding sheet)"
        )
    items = document["codes"]
    if not isinstance(items, list) or not items:
        raise _refuse("codes must be a non-empty list")
    codes: dict[str, Code] = {}
    declared = [identity for identity, _ in settings.taxonomy]
    unknown: dict[str, list[int]] = defaultdict(list)  # category -> code numbers
    for number, item in enumerate(items, 1):
        if (
            not isinstance(item, dict)
            or not {"attempt_id", "category"} <= set(item)
            or set(item) - set(CODE_KEYS)
        ):
            raise _refuse(
                f"code {number} must hold attempt_id and category, an optional note and nothing else"
            )
        attempt = _identifier(item, "attempt_id", number)
        category = _identifier(item, "category", number)
        note = item.get("note")
        if note is not None and type(note) is not str:
            raise _refuse(f"code {number}: note must be a string")
        if attempt in codes:
            raise _refuse(f"code {number}: attempt {attempt} is coded twice")
        if category not in declared:
            unknown[category].append(number)
        codes[attempt] = Code(category, note)
    if unknown:
        listed = "; ".join(
            f"{category} (code{'s' if len(numbers) > 1 else ''} {', '.join(map(str, numbers))})"
            for category, numbers in sorted(unknown.items())
        )
        raise _refuse(f"categories not in the study's taxonomy ({', '.join(declared)}): {listed}")
    return Coding(
        hashlib.sha256(data).hexdigest(), coder, procedure, blinded, MappingProxyType(codes)
    )


def check_runs(coding: Coding, study: Study) -> None:
    """Refuse a coding that names an attempt which is not a reported final run of the study."""
    reported = {run["attempt_id"] for stratum in study.strata for run in stratum.reported}
    unknown = sorted(set(coding.codes) - reported)
    if unknown:
        shown = ", ".join(unknown[:5]) + (" …" if len(unknown) > 5 else "")
        raise _refuse(
            f"{len(unknown)} attempt(s) are not reported final runs of study {study.export.study_id}: {shown}"
        )


# Runs and their streams -----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Crop:
    """One reported run's crop: its number on the blinded coding sheet, its place in study order, and the window
    [start, end] on the movement clock (start None: the whole path; end None: a run that never moved, which has no
    movement clock)."""

    number: int  # 1-based, in the sheet's shuffled order (``SHEET_ORDER_STREAM``)
    order: int  # 0-based, in study order (stratum, profile, blocks in execution order)
    run: Row
    stratum: Stratum
    short_id: str
    start: float | None
    end: float | None


@dataclass(frozen=True, slots=True)
class PathTrace:
    """Movement-relative ground-truth samples of one run: simulated s and world-frame metres, in time order."""

    t: Floats
    x: Floats
    y: Floats

    def between(self, start: float | None, end: float) -> "PathTrace | None":
        """The samples in [start, end] (from the first without a start), or None when there are none."""
        keep = self.t <= end + TIME_EPSILON
        if start is not None:
            keep &= self.t >= start - TIME_EPSILON
        return PathTrace(self.t[keep], self.x[keep], self.y[keep]) if keep.any() else None


@dataclass(frozen=True, slots=True)
class Streams:
    """The ground-truth traces, rest poses, goals and reference maps of the crops, read once."""

    traces: Mapping[str, PathTrace]
    rests: Mapping[
        str, tuple[float, float]
    ]  # the last pose of a run without movement-relative rows
    goals: Mapping[str, tuple[Goal, ...]]  # attempt -> its goals (goals.read_goals)
    alignments: Mapping[str, Triple | None]  # attempt -> its world_from_map, for goals.place
    maps: Mapping[str, tuple[np.ndarray[Any, Any], Extent] | None]  # attempt -> its reference
    inputs: Mapping[str, frozenset[str]]  # attempt -> the export files its crop read

    def read_by(self, crops: Sequence[Crop]) -> set[str]:
        """The export files the given crops read, for a figure's sidecar."""
        return {name for crop in crops for name in self.inputs[crop.run["attempt_id"]]}


def _streams(ctx: FigureContext, runs: Sequence[Row]) -> Streams:
    attempts = {run["attempt_id"] for run in runs}
    reads = Reads(ctx.study.export)
    samples: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    rests: dict[str, tuple[float, float, float]] = {}  # t_activation_s, x, y
    table = reads.table(
        "trajectory_gt", ["attempt_id", "t_s", "t_activation_s", "gt_x_m", "gt_y_m"]
    )
    if table is not None:
        for attempt, t, t_activation, x, y in zip(
            *(table.column(i).to_pylist() for i in range(5)), strict=True
        ):
            if attempt not in attempts or x is None or y is None:
                continue
            if t is not None:
                samples[attempt].append((t, x, y))
            elif t_activation is not None and (
                attempt not in rests or t_activation >= rests[attempt][0]
            ):
                rests[attempt] = (t_activation, x, y)
    traces = {
        attempt: PathTrace(*(np.array(column, dtype=np.float64) for column in zip(*sorted(rows))))
        for attempt, rows in samples.items()
    }
    # goal records from the shared reader; its rejection reads (raw navigation) are not used here, so a separate
    # record keeps them out of the crops' inputs
    goal_reads = Reads(ctx.study.export)
    records = read_goals(goal_reads, runs)
    goals = {attempt: tuple(record.goals) for attempt, record in records.items()}
    alignments = {
        attempt: world_from_map(reads, attempt) if record.goals else None
        for attempt, record in records.items()
    }
    shared = {name for name in reads.files if not name.startswith(("raw/", "maps/"))}
    shared |= {"goals.parquet"} & goal_reads.files
    loaded: dict[str, tuple[np.ndarray[Any, Any], Extent] | None] = {}
    maps: dict[str, tuple[np.ndarray[Any, Any], Extent] | None] = {}
    inputs: dict[str, frozenset[str]] = {}
    for attempt in sorted(attempts):
        name = reference_name(ctx, attempt)
        own = {f for f in reads.files if f.startswith((f"raw/{attempt}/", f"maps/{attempt}."))}
        if name is not None and name not in loaded:
            loaded[name] = load_reference(ctx, attempt)
        maps[attempt] = None if name is None else loaded[name]
        # the reference's files and provenance.json, which maps the run to it (also without a reference)
        inputs[attempt] = frozenset(shared | own | reference_files(ctx, name))
    return Streams(
        traces,
        # the pre-motion pose only stands for a run without movement-relative rows (never as a moving run's end)
        {a: (x, y) for a, (_, x, y) in rests.items() if a not in traces},
        goals,
        alignments,
        maps,
        inputs,
    )


def _execution_key(
    run: Row, starts: Mapping[tuple[str, str, str, str], int]
) -> tuple[bool, int, str]:
    start = starts.get((run["robot_id"], run["world_id"], run["spawn_id"], run["block_id"]))
    return start is None, start or 0, run["block_id"]  # blocks never started last


def _short_ids(attempts: Sequence[str]) -> dict[str, str]:
    """The shortest prefix of at least ``SHORT_ID`` characters that tells the study's reported runs apart."""
    length = SHORT_ID
    while len({a[:length] for a in attempts}) < len(set(attempts)):
        length += 1
    return {a: a[:length] for a in attempts}


def crops(ctx: FigureContext) -> list[Crop]:
    """Every reported run as a crop, in sheet order. Study order (strata as listed, profiles in figure order,
    blocks in execution order) is shuffled with the plan's seed on its own stream, so that the sheet's numbers and
    order say nothing about profile, outcome or stratum (the blinded coding sheet); the same export and plan
    always give the same numbers."""
    starts = block_starts(ctx.study)
    window = ctx.settings.stall_window_s
    ordered = [
        (stratum, run)
        for stratum in ctx.study.strata
        for profile in ctx.profiles(stratum)
        for run in sorted(
            (r for r in stratum.reported if r["profile_id"] == profile),
            key=lambda r: _execution_key(r, starts),
        )
    ]
    short = _short_ids([run["attempt_id"] for _, run in ordered])
    shuffled = random_stream(ctx.settings.seed, SHEET_ORDER_STREAM).permutation(len(ordered))
    found = []
    for number, position in enumerate(shuffled.tolist(), 1):
        stratum, run = ordered[position]
        if run["outcome"] == "no_movement":
            start, end = None, None
        else:
            end, _ = run_end(run, stratum.limit_s)
            start = None if window is None else max(0.0, end - window)
        found.append(Crop(number, position, run, stratum, short[run["attempt_id"]], start, end))
    return found


# Drawing a crop -------------------------------------------------------------------------------------------------


def _window_goals(streams: Streams, crop: Crop) -> tuple[tuple[PlacedGoal, ...], Counter[str]]:
    """Goals active at some time in the window (not ended before its start; ``goals.place`` keeps those
    dispatched by its end) placed in the world frame, and the active ones that could not be placed, by map key (no
    map-frame pose or alignment). A run without movement has no window and no movement-relative goal times."""
    if crop.end is None:
        return (), Counter()
    attempt = crop.run["attempt_id"]
    active = [
        goal
        for goal in streams.goals.get(attempt, ())
        if goal.dispatch_s is not None
        and not (
            crop.start is not None
            and goal.end_s is not None
            and goal.end_s < crop.start - TIME_EPSILON
        )
    ]
    return place(active, streams.alignments.get(attempt), crop.end)


def _square(xs: Sequence[float], ys: Sequence[float]) -> Extent:
    """A square around the points with ``PAD_M`` free on each side, at least ``MIN_SIDE_M`` wide."""
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    side = max(MIN_SIDE_M, x1 - x0 + 2 * PAD_M, y1 - y0 + 2 * PAD_M)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)


def _draw_map(ax: Any, reference: tuple[np.ndarray[Any, Any], Extent], box: Extent) -> None:
    """The part of the reference under ``box``, cells outside the reference as outside (hatched), so the crop
    holds only its own cells and every crop is drawn the same way (FIGURE_STYLE §6)."""
    codes, extent = reference
    resolution = (extent[1] - extent[0]) / codes.shape[1]
    c0 = math.floor((box[0] - extent[0]) / resolution)
    c1 = math.ceil((box[1] - extent[0]) / resolution)
    r0 = math.floor((box[2] - extent[2]) / resolution)
    r1 = math.ceil((box[3] - extent[2]) / resolution)
    canvas = np.full((r1 - r0, c1 - c0), MapCode.OUTSIDE, dtype=codes.dtype)
    rows = slice(max(r0, 0), min(r1, codes.shape[0]))
    columns = slice(max(c0, 0), min(c1, codes.shape[1]))
    if rows.start < rows.stop and columns.start < columns.stop:
        canvas[rows.start - r0 : rows.stop - r0, columns.start - c0 : columns.stop - c0] = codes[
            rows, columns
        ]
    x0, y0 = extent[0] + c0 * resolution, extent[2] + r0 * resolution
    draw_reference(
        ax, canvas, (x0, x0 + canvas.shape[1] * resolution, y0, y0 + canvas.shape[0] * resolution)
    )


def _blank(ax: Any, text: str) -> None:
    """A crop without a map: the same square frame as the others, so every crop of a page lays out alike."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=7)


@dataclass
class Drawn:
    """What a page's crops drew: the path colours (for the colour bar), and goals drawn and not placed, by map key
    (``goals.Goal.map_key``)."""

    mappable: LineCollection | None = None
    keys: Counter[str] = field(default_factory=Counter)
    unplaced: Counter[str] = field(default_factory=Counter)

    def add(self, other: "Drawn") -> None:
        self.mappable = other.mappable or self.mappable
        self.keys += other.keys
        self.unplaced += other.unplaced


def _path(ax: Any, trace: PathTrace, end: float, norm: Normalize) -> LineCollection:
    """The window's path coloured by the time before the end, cased (the styling of F8's path)."""
    segments = np.stack(
        [
            np.column_stack([trace.x[:-1], trace.y[:-1]]),
            np.column_stack([trace.x[1:], trace.y[1:]]),
        ],
        1,
    )
    lines = LineCollection(
        list(segments),
        cmap=plt.get_cmap("cividis"),
        norm=norm,
        linewidths=1.0,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    lines.set_array((trace.t[:-1] + trace.t[1:]) / 2 - end)  # seconds before the end, negative
    lines.set_path_effects(
        [patheffects.Stroke(linewidth=2.0, foreground="#333333"), patheffects.Normal()]
    )
    ax.add_collection(lines)
    return lines


def _draw_crop(ax: Any, streams: Streams, crop: Crop, norm: Normalize) -> Drawn:
    """One crop: the window's path and goals on the reference around them, or a blank square saying why not."""
    ax.set_axis_off()
    attempt = crop.run["attempt_id"]
    trace = streams.traces.get(attempt)
    rest = streams.rests.get(attempt)
    if crop.end is not None:  # a run that moved: its window, never its pre-motion pose
        window = None if trace is None else trace.between(crop.start, crop.end)
        if window is None:
            _blank(ax, "no pose in window")
            return Drawn()
        xs, ys = list(window.x), list(window.y)
    elif rest is not None:  # a run that never moved: its last pose
        window = None
        xs, ys = [rest[0]], [rest[1]]
    else:
        _blank(ax, "no ground-truth pose")
        return Drawn()
    goals, unplaced = _window_goals(streams, crop)
    box = _square(
        xs + [g.x for g in goals if g.x is not None], ys + [g.y for g in goals if g.y is not None]
    )
    reference = streams.maps.get(attempt)
    if reference is not None:
        _draw_map(ax, reference, box)
    else:
        ax.text(0.5, 0.92, "no reference", transform=ax.transAxes, ha="center", fontsize=7)
    drawn = Drawn(keys=Counter(g.key for g in goals), unplaced=unplaced)
    if window is not None and crop.end is not None and len(window.t) >= 2:
        drawn.mappable = _path(ax, window, crop.end, norm)
        ax.plot([xs[0]], [ys[0]], linestyle="none", zorder=4, **WINDOW_START)
    for goal in goals:
        ax.plot([goal.x], [goal.y], zorder=4, **map_style(goal.key))
    ax.plot([xs[-1]], [ys[-1]], linestyle="none", zorder=4, **END)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal", adjustable="box")
    return drawn


def _norm(ctx: FigureContext) -> Normalize:
    window = ctx.settings.stall_window_s
    return Normalize(-(window or max(s.limit_s for s in ctx.study.strata)), 0)


def _outcome(run: Row) -> str:
    return str(OUTCOMES[run["outcome"]]["label"]).lower()


def _window_text(ctx: FigureContext) -> str:
    window = ctx.settings.stall_window_s
    if window is None:
        return "the whole ground-truth path to min(T_end, L) (the plan declares no stall window)"
    return f"the ground-truth path over the last {window:g} s before min(T_end, L) (the plan's stall.window_s)"


def _close_enough(ctx: FigureContext) -> str:
    """The close-enough distance of the study's profiles, as the goal legend words it."""
    return close_enough_text(
        ctx.study.export, {p for stratum in ctx.study.strata for p in ctx.profiles(stratum)}
    )


def _marks_text(ctx: FigureContext) -> str:
    """How the crops are drawn, the goal marks named from the shared table (goals.STATUS_LOOKS)."""
    looks = ", ".join(
        f"{look.label} {'filled' if look.filled else 'open'} {SHAPES.get(look.marker, look.marker)}"
        for look in STATUS_LOOKS.values()
    )
    looks += (
        f"; ended by the explorer within {_close_enough(ctx)} (its close-enough guard, cancel reason "
        "completion_distance) half-filled circle, apart from the other cancels"
    )
    return (
        "Map: free floor white, other inside cells light grey, obstacles dark grey, outside the reference "
        f"hatched; crops are squares at least {MIN_SIDE_M:g} m wide with {PAD_M:g} m around the path and goals, "
        "5 m bar. Path coloured by the time before the end (export's 1 s ground-truth rows, cased as in F8); open "
        "star: the window's first pose; black square: the end (a run that never moved: its last pose before "
        f"movement could start); goals active in the window by end status: {looks} (map frame placed with the "
        "run's world_from_map). A moving run without a pose in its window is left blank."
    )


def _mark_legend(ctx: FigureContext, fig: Figure | SubFigure, drawn: Drawn) -> None:
    handles: list[Artist] = [
        Line2D([], [], linestyle="none", label="window start", **WINDOW_START),
        Line2D([], [], linestyle="none", label="end", **END),
        *map_handles(drawn.keys, _close_enough(ctx)),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=min(LEGEND_COLUMNS, len(handles)),
        fontsize=7,
    )


def _colour_bar(
    fig: Figure | SubFigure, mappable: LineCollection | None, axes: Sequence[Any]
) -> None:
    if mappable is None:
        return
    bar = fig.colorbar(
        mappable, ax=list(axes), orientation="horizontal", fraction=0.04, pad=0.02, aspect=50
    )
    bar.set_label(WINDOW_LABEL)


def _caption_notes(ctx: FigureContext, first: str, drawn: Drawn) -> list[str]:
    notes = [first, _marks_text(ctx)]
    if drawn.unplaced:
        notes.append(
            f"{drawn.unplaced.total()} goal(s) without a map-frame pose or alignment are not drawn."
        )
    return notes


# The coding sheet -----------------------------------------------------------------------------------------------


def _sheet_page(
    ctx: FigureContext, page: int, pages: int, members: Sequence[Crop], streams: Streams
) -> FigureOut:
    rows = math.ceil(len(members) / SHEET_COLUMNS)
    width_mm = PRESETS[ctx.preset].widths_mm["double"]
    fig, axes = figure(
        ctx.preset,
        "double",
        aspect=(rows * CROP_ROW_MM + SHEET_BOTTOM_MM) / width_mm,
        nrows=rows,
        ncols=SHEET_COLUMNS,
    )
    flat = [axes[r][c] for r in range(rows) for c in range(SHEET_COLUMNS)]
    norm, drawn = _norm(ctx), Drawn()
    for ax in flat:
        ax.set_axis_off()
    for ax, crop in zip(flat, members, strict=False):
        drawn.add(_draw_crop(ax, streams, crop, norm))
        ax.set_title(str(crop.number), loc="left", fontsize=7)  # blinded: the crop number only
    _colour_bar(fig, drawn.mappable, flat)
    _mark_legend(ctx, fig, drawn)
    finish(fig, ctx)
    first, last = members[0].number, members[-1].number
    return FigureOut(
        f"F23_sheet_{page:02d}",
        fig,
        "double",
        {crop.run["attempt_id"] for crop in members},
        f"Coding sheet page {page}: map crops of the end of each reported run, numbered, for coding "
        "behaviour categories.",
        _caption_notes(
            ctx,
            f"Coding sheet, page {page} of {pages} (crops {first}–{last} of the study's reported runs; "
            f"FIGURE_EXTENSION FX-19): {_window_text(ctx)}, for coding against the plan's taxonomy. Blinded to "
            "profile and outcome by convention (the order is recomputable from the plan seed and the export): "
            f"crops in an order shuffled with the plan's seed {ctx.settings.seed} (stream {SHEET_ORDER_STREAM}), "
            "labelled with the crop number only; the key from crop number to attempt, stratum, profile and outcome "
            "is in this page's sidecar, to be opened after coding. The crop itself can show a run that never moved "
            "(no path). Coding never changes a recorded outcome.",
            drawn,
        ),
        dict(
            blinded=True,
            order=dict(seed=ctx.settings.seed, stream=SHEET_ORDER_STREAM),
            # the key from crop number to run: only here, never on the sheet
            crops=[
                dict(
                    number=crop.number,
                    attempt_id=crop.run["attempt_id"],
                    short_id=crop.short_id,
                    stratum=ctx.label(crop.stratum),
                    profile_id=crop.run["profile_id"],
                    outcome=crop.run["outcome"],
                    window_start_s=crop.start,
                    end_s=crop.end,  # null for a run that never moved (no movement clock)
                )
                for crop in members
            ],
            window_s=ctx.settings.stall_window_s,
            # by map key (goals.Goal.map_key: the status, or "arrived" for a close-enough cancel)
            goals_drawn=dict(sorted(drawn.keys.items())),
            goals_not_drawn=dict(sorted(drawn.unplaced.items())),
        ),
        inputs=streams.read_by(members),
    )


def coding_sheet(
    ctx: FigureContext, found: Sequence[Crop], streams: Streams
) -> Iterator[FigureOut]:
    """The sheet's pages, one at a time (each is saved and closed before the next is drawn)."""
    per_page = SHEET_COLUMNS * SHEET_ROWS
    chunks = [found[i : i + per_page] for i in range(0, len(found), per_page)]
    for page, members in enumerate(chunks, 1):
        yield _sheet_page(ctx, page, len(chunks), members, streams)


# F23 ----------------------------------------------------------------------------------------------------------


def _fill(index: int) -> tuple[str, str | None]:
    return CATEGORY_FILLS[index % len(CATEGORY_FILLS)]


def _category_handles(categories: Sequence[str]) -> list[Artist]:
    handles: list[Artist] = [
        Patch(facecolor=_fill(i)[0], hatch=_fill(i)[1], edgecolor=EDGE, label=key)
        for i, key in enumerate(categories)
    ]
    handles.append(Patch(facecolor="#FFFFFF", edgecolor=EDGE, linestyle="--", label=NOT_CODED))
    return handles


def _bars(
    ax: Any,
    labels: Sequence[str],
    counts: Sequence[Mapping[str, int]],
    categories: Sequence[str],
    totals: Sequence[int],
) -> None:
    """Stacked horizontal bars of absolute counts, the declared categories in order, then "not coded"."""
    keys = [*categories, NOT_CODED]
    for y, (by_category, total) in enumerate(zip(counts, totals, strict=True)):
        left = 0
        for index, key in enumerate(keys):
            value = by_category.get(key, 0)
            if not value:
                continue
            colour, hatch = ("#FFFFFF", None) if key == NOT_CODED else _fill(index)
            ax.barh(
                y,
                value,
                left=left,
                height=0.7,
                color=colour,
                hatch=hatch,
                edgecolor=EDGE,
                lw=0.4,
                linestyle="--" if key == NOT_CODED else "-",
            )
            box = dict(facecolor="#FFFFFF", edgecolor="none", pad=0.6) if hatch else None
            ax.text(
                left + value / 2,
                y,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color="#FFFFFF" if colour == GREYS[0] and not hatch else "#000000",
                bbox=box,
            )
            left += value
        ax.text(total, y, f"  n = {total}", ha="left", va="center", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals, default=1) * 1.18)
    ax.set_xlabel("Reported runs (count)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    panel_title(ax, 0, "reported runs by coded category")


def f23(ctx: FigureContext, coding: Coding, found: Sequence[Crop], streams: Streams) -> FigureOut:
    taxonomy = ctx.settings.taxonomy or ()
    categories = [identity for identity, _ in taxonomy]
    codes = coding.categories()
    labels: list[str] = []
    counts: list[dict[str, int]] = []
    totals: list[int] = []
    table: list[dict[str, Any]] = []
    for stratum in ctx.study.strata:
        for profile in ctx.profiles(stratum):
            runs = [r for r in stratum.reported if r["profile_id"] == profile]
            if not runs:
                continue
            by_category = Counter(codes.get(r["attempt_id"], NOT_CODED) for r in runs)
            labels.append(f"{ctx.label(stratum)} · {ctx.styles[profile].name}")
            counts.append(dict(by_category))
            totals.append(len(runs))
            table.append(
                dict(
                    stratum=ctx.label(stratum),
                    profile_id=profile,
                    n=len(runs),
                    categories={c: by_category.get(c, 0) for c in categories},
                    not_coded=by_category.get(NOT_CODED, 0),
                )
            )
    starts = block_starts(ctx.study)
    examples: dict[str, Crop | None] = {}
    for category in categories:
        members = [c for c in found if codes.get(c.run["attempt_id"]) == category]
        examples[category] = min(
            members,
            key=lambda c: (*_execution_key(c.run, starts), c.order),
            default=None,
        )
    # F23 stacks two subfigures (the bars, the example crops) so that each keeps its own column layout under
    # constrained layout; style.figure makes one grid of subplots, so the figure is made here at the preset's
    # double-column width
    handles = _category_handles(categories)
    width_in = PRESETS[ctx.preset].widths_mm["double"] / MM_PER_INCH
    bar_mm = (
        BAR_ROW_MM * len(labels)
        + BAR_AXIS_MM
        + LEGEND_ROW_MM * math.ceil(len(handles) / LEGEND_COLUMNS)
    )
    crop_rows = math.ceil(len(categories) / EXAMPLE_COLUMNS)
    crop_mm = EXAMPLE_ROW_MM * crop_rows + EXAMPLE_BOTTOM_MM
    fig = plt.figure(figsize=(width_in, (bar_mm + crop_mm) / MM_PER_INCH))
    top, bottom = fig.subfigures(2, 1, height_ratios=[bar_mm, crop_mm])
    _bars(top.subplots(), labels, counts, categories, totals)
    top.legend(
        handles=handles,
        loc="outside lower center",
        ncols=min(LEGEND_COLUMNS, len(handles)),
        fontsize=7,
    )
    columns = min(EXAMPLE_COLUMNS, len(categories))
    grid = bottom.subplots(crop_rows, columns, squeeze=False)
    flat = [grid[r][c] for r in range(crop_rows) for c in range(columns)]
    norm, drawn = _norm(ctx), Drawn()
    for ax in flat:
        ax.set_axis_off()
    for index, (ax, category) in enumerate(zip(flat, categories, strict=False)):
        panel_title(ax, index + 1, category)
        crop = examples[category]
        if crop is None:
            _blank(ax, "no run coded")
            continue
        drawn.add(_draw_crop(ax, streams, crop, norm))
        ax.text(
            0.0,
            -0.03,
            f"{crop.number} · {crop.short_id} · {ctx.label(crop.stratum)}\n"
            f"{ctx.styles[crop.run['profile_id']].name} · {_outcome(crop.run)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
    _colour_bar(bottom, drawn.mappable, flat)
    _mark_legend(ctx, bottom, drawn)
    finish(fig, ctx, descriptive=True)
    first = (
        "Behaviour taxonomy (FIGURE_EXTENSION FX-19): reported runs (completed, time limit, no movement) coded by "
        "the researcher against the categories declared in the plan's taxonomy (definitions in T9). Counts are "
        "absolute over every reported run of each stratum and profile, n per bar; runs the coding file does not "
        "name are counted as not coded. Coding never changes a recorded outcome. Descriptive counts, not in "
        f"stats.json; not cited as evidence. {blinding_text(coding.document())} {CODER_NOTE} "
        f"{CATEGORY_NOTE} Coder, procedure, the coding file's SHA-256, each code and note in the sidecar. "
        "Example crops (labelled with profile and outcome after coding): per category, its coded run of the "
        "earliest block in execution order (first attempt start, as in F10), numbered as on the coding sheet; "
        f"{_window_text(ctx)}."
    )
    examples_used = [crop for crop in examples.values() if crop is not None]
    notes = _caption_notes(ctx, first, drawn)
    tall = page_note(
        bar_mm + crop_mm, split="category"
    )  # many categories: more bars and rows of example crops
    if tall is not None:
        notes.append(tall)
    return FigureOut(
        "F23",
        fig,
        "double",
        {r["attempt_id"] for s in ctx.study.strata for r in s.reported},
        "Stacked counts of researcher-coded behaviour categories per stratum and profile, with one map crop "
        "of an example run per category.",
        notes,
        dict(
            coding=coding.document(),
            codes=dict(sorted(codes.items())),
            code_notes=coding.notes(),
            # not "counts": that key is the sidecar's own run counts (outputs.scope_counts, the Study tab's n line)
            category_counts=table,
            examples={
                category: None if crop is None else crop.run["attempt_id"]
                for category, crop in examples.items()
            },
            window_s=ctx.settings.stall_window_s,
            # by map key (goals.Goal.map_key: the status, or "arrived" for a close-enough cancel)
            goals_drawn=dict(sorted(drawn.keys.items())),
            goals_not_drawn=dict(sorted(drawn.unplaced.items())),
        ),
        inputs=streams.read_by(examples_used),
    )


def figures(ctx: FigureContext, coding: Coding | None = None) -> Iterator[FigureOut]:
    """The coding sheet when the plan declares a taxonomy, and F23 first when a coding file is given as well;
    yielded one at a time."""
    if ctx.settings.taxonomy is None:
        return
    found = crops(ctx)
    if not found:
        return
    streams = _streams(ctx, [crop.run for crop in found])
    if coding is not None:
        yield f23(ctx, coding, found, streams)
    yield from coding_sheet(ctx, found, streams)
