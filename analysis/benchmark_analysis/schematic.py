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

"""Figure F20: the explorer's decision steps on a schematic grid (docs/plans/FIGURE_EXTENSION.md, FX-9, FX4;
FIGURE_STYLE §6, §7 DR-8).

A scripted 30 × 24-cell scene, drawn at 0.25 m per cell, with no data behind it and no randomness. The steps are
computed on the scene with the explorer's rules as implemented (``explorer-algorithm`` skill and its
``COST_MODEL.md``; the explorer README's cost model is outdated, D-01) and the benchmark's Default parameters:

- (a) the scripted decision map (the filter not simulated) in the map greys (FP-5) with the robot pose;
- (b) wavefront frontier detection: the two-level BFS from the robot cell; a frontier cell is unknown, has a free
  8-neighbour and no blocked 8-neighbour; 8-connected clusters under ``min_frontier_size_cells`` are dropped;
- (c) the cost terms of one candidate j with the symbols of COST_MODEL.md (r, ψ, c_j, m_j, s_j, D_j, Δψ_j);
- (d) the cost matrix M over the candidates (node 0 = the robot);
- (e) the DP route: the pool of the K lowest by M(0, j), the exact minimum over simple paths of h = min(H, K)
  frontiers, numbered in the route colour (FP-7); only the first frontier's goal point is sent;
- (f) the visibility gate as FIGURE_STYLE §7 describes it: at a later map tick, rays from the target pose (stopped
  at the cluster box ±1 cell) count the visible frontier cells; below min(P_j · resolution, 0.5 m) the goal is
  cancelled at once, without a replacement check (D-18).

The copy mirrors the explorer at gitlink ``11864e4`` with its uncommitted log-format-2 working tree (R8) and
``config/frontier_exploration_ros2/default.yaml`` as of 24 October 2026 (``MIRRORS``, also in the sidecar); recheck
it against ``COST_MODEL.md`` and the explorer source when either changes.

Simplifications, stated in the caption notes: the decision map is the scripted grid (the bilateral filter and
dilation are not drawn); the global costmap's blocked cells are the occupied cells (no inflation); route segments
are straight lines that give the order, not a path. The figure has no runs (an empty attempt-ID set).
"""

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import functools
import itertools
import math
from typing import Any

from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch, Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

from .figures import GREY, FigureContext, FigureOut, panel_title
from .style import EDGE, MAP_FREE, MAP_OCCUPIED, MAP_UNKNOWN, PRESETS, ROUTE_COLOUR, figure

Cell = tuple[int, int]  # (x, y) map cell
Point = tuple[float, float]  # metres in the scene frame (origin at the lower-left corner)

# The scene ------------------------------------------------------------------------------------------------------

WIDTH, HEIGHT = 30, 24  # cells
RESOLUTION_M = 0.25  # drawn coarser than the 0.05 m of the recorded maps, so cells stay visible
FREE, UNKNOWN, OCCUPIED = 0, 1, 2  # decision-map classes (the explorer's 255, 205 and 0)
ROBOT: Cell = (4, 7)
ROBOT_YAW = 0.0  # psi, rad: facing +x
LATER_ROBOT: Cell = (6, 5)  # the robot at the later map tick of panel (f)
# the pillar's unseen shadow (x0, x1, y0, y1): unknown at the decision tick, partly seen at the later tick
SHADOW = (9, 11, 7, 10)
NOTATION_LETTER = "C"  # panel (c) annotates the hall north of the room
MIRRORS = (
    "explorer gitlink 11864e4 with the uncommitted log-format-2 working tree; "
    "config/frontier_exploration_ros2/default.yaml as of 2026-10-24"
)
# panel (c)'s labels in axis units (cells): text, where, and the point a leader line goes to (scripted with the scene)
NOTATION_LABELS: tuple[tuple[str, Point, Point | None], ...] = (
    ("$r$", (3.3, 6.1), None),
    (r"$\psi$", (8.4, 6.6), None),
    (r"$\Delta\psi_j$", (7.3, 10.9), None),
    ("$D_j$", (2.3, 11.6), None),
    (r"$|r-c_j|$", (9.6, 12.6), None),
    ("$m_j$", (6.5, 22.7), (6.5, 20.5)),
    ("$c_j$", (8.6, 17.9), (6.5, 19.32)),
    ("$s_j$", (0.9, 17.9), (2.5, 16.5)),
)
ARC_RADIUS = 2.2  # cells

# Default profile (config/frontier_exploration_ros2/default.yaml), the explorer's names in the comments
W_D = 1.0  # weight_distance_wd
W_S = 1.0  # weight_gain_ws
R_S = 1.5  # sensor_effective_range_m
V_MAX = 0.5  # max_linear_speed_vmax, m/s
W_MAX = 1.0  # max_angular_speed_wmax, rad/s
CANDIDATE_LIMIT = 12  # dp_solver_candidate_limit (K before the clamp to [1, 60] and to n)
HORIZON = 8  # dp_planning_horizon (H)
MIN_CLUSTER = 5  # min_frontier_size_cells
MIN_GOAL_M = 0.8  # frontier_candidate_min_goal_distance_m
MIN_REVEAL_M = 0.5  # goal_preemption_lidar_min_reveal_length_m
# goal_preemption_complete_if_within_m (checked before the gate)
COMPLETE_WITHIN_M = 0.5
# goal_preemption_min_interval_s: the gate waits this long after dispatch, and a low-gain cancel follows at
# most once per interval
MIN_INTERVAL_S = 2.0
LIDAR_RANGE_M = 10.0  # goal_preemption_lidar_range_m
RAY_STEP_DEG = 1.0  # goal_preemption_lidar_ray_step_deg, over a 360° field of view

LETTERS = "ABCDE"
# where each candidate's letter sits, as an offset from its centroid in cells (scripted scene, scripted labels)
LETTER_OFFSETS: dict[str, Point] = {
    "A": (5.6, 0.6),
    "B": (1.9, 2.1),
    "C": (5.4, 2.6),
    "D": (0.0, 1.7),
    "E": (1.6, 0.0),
}


def _scene() -> np.ndarray[Any, Any]:
    """The decision map at the decision tick, indexed [y, x]: a room with the robot, a hall seen through its north
    door, a strip seen through its south door, a one-cell alcove, a pillar with its unseen shadow, and a corridor
    east with a room seen through a gap in its north wall."""
    grid = np.full((HEIGHT, WIDTH), UNKNOWN, dtype=np.int8)

    def put(value: int, x0: int, x1: int, y0: int, y1: int) -> None:
        grid[y0 : y1 + 1, x0 : x1 + 1] = value

    put(FREE, 1, 12, 4, 13)  # the room
    for x0, x1, y0, y1 in ((0, 0, 3, 14), (0, 13, 3, 3), (0, 13, 14, 14), (13, 13, 3, 14)):
        put(OCCUPIED, x0, x1, y0, y1)
    put(FREE, 4, 7, 14, 14)  # north door and the hall behind it
    put(FREE, 3, 9, 15, 19)
    put(FREE, 2, 4, 3, 3)  # alcove: one frontier cell, dropped
    put(FREE, 8, 11, 3, 3)  # south door and the strip behind it
    put(FREE, 7, 12, 1, 2)
    put(OCCUPIED, 7, 8, 8, 9)  # pillar and its shadow
    put(UNKNOWN, *SHADOW)
    put(FREE, 13, 13, 6, 9)  # corridor east, its open end and the room north of its gap
    put(FREE, 14, 21, 6, 9)
    put(OCCUPIED, 13, 20, 5, 5)
    put(OCCUPIED, 13, 20, 10, 10)
    put(FREE, 21, 23, 4, 11)
    put(OCCUPIED, 20, 23, 3, 3)
    put(OCCUPIED, 20, 23, 12, 12)
    put(FREE, 15, 19, 10, 10)
    put(FREE, 14, 20, 11, 13)
    put(OCCUPIED, 21, 21, 10, 13)
    return grid


def _later(grid: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The map at a later tick: the robot at ``LATER_ROBOT`` has seen the shadow cells in its line of sight (the
    shadow is free floor); everything else is unchanged."""
    later = grid.copy()
    origin = _centre(LATER_ROBOT)
    x0, x1, y0, y1 = SHADOW
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if _line_of_sight(grid, origin, _centre((x, y))):
                later[y, x] = FREE
    return later


# Geometry and the explorer's rules ------------------------------------------------------------------------------


def _centre(cell: Cell) -> Point:
    return ((cell[0] + 0.5) * RESOLUTION_M, (cell[1] + 0.5) * RESOLUTION_M)


def _cell(point: Point) -> Cell:
    return (math.floor(point[0] / RESOLUTION_M), math.floor(point[1] / RESOLUTION_M))


def _inside(cell: Cell) -> bool:
    return 0 <= cell[0] < WIDTH and 0 <= cell[1] < HEIGHT


def _neighbours(cell: Cell) -> Iterator[Cell]:
    """The 3 × 3 block around a cell, the cell included (``kNeighborOffsets`` in frontier_search.cpp)."""
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            other = (cell[0] + dx, cell[1] + dy)
            if _inside(other):
                yield other


def _line_of_sight(grid: np.ndarray[Any, Any], start: Point, end: Point) -> bool:
    steps = max(1, math.ceil(math.dist(start, end) / (RESOLUTION_M / 4)))
    for k in range(1, steps):
        f = k / steps
        x, y = _cell((start[0] + f * (end[0] - start[0]), start[1] + f * (end[1] - start[1])))
        if grid[y, x] == OCCUPIED:
            return False
    return True


def _is_frontier(grid: np.ndarray[Any, Any], cell: Cell) -> bool:
    """Unknown, with a free 8-neighbour and no blocked one (``is_frontier_point``); blocked = occupied here."""
    if grid[cell[1], cell[0]] != UNKNOWN:
        return False
    near = [int(grid[y, x]) for x, y in _neighbours(cell)]
    return OCCUPIED not in near and FREE in near


@dataclass(frozen=True, slots=True)
class Search:
    # (start cell s_j, cells) of the kept clusters, in discovery order
    clusters: list[tuple[Cell, list[Cell]]]
    dropped: list[list[Cell]]
    # BFS hop distance from the robot cell (one cell per hop, diagonals included)
    distance_m: dict[Cell, float]
    parent: dict[Cell, Cell]


def _frontier_search(grid: np.ndarray[Any, Any], robot: Cell) -> Search:
    """The two-level BFS of ``get_frontier`` (frontier_search.cpp): the map queue grows through cells next to free
    space; at a frontier cell a frontier queue collects its 8-connected cluster. Distances and parents follow the
    explorer's assignment order, so ``distance_m`` of a cluster cell is the explorer's ``robot_distance_m``."""
    map_open, map_closed, frontier_open, frontier_closed = 1, 2, 4, 8
    flags: dict[Cell, int] = {robot: map_open}
    distance = {robot: 0.0}
    parent: dict[Cell, Cell] = {}
    clusters: list[tuple[Cell, list[Cell]]] = []
    dropped: list[list[Cell]] = []

    def reach(cell: Cell, source: Cell) -> None:
        if source in distance:
            distance[cell] = distance[source] + RESOLUTION_M
            parent[cell] = source

    queue = deque([robot])
    while queue:
        point = queue.popleft()
        if flags.get(point, 0) & map_closed:
            continue
        if _is_frontier(grid, point):
            flags[point] = flags.get(point, 0) | frontier_open
            cells: list[Cell] = []
            frontier_queue = deque([point])
            while frontier_queue:
                cell = frontier_queue.popleft()
                if flags.get(cell, 0) & (map_closed | frontier_closed):
                    continue
                if _is_frontier(grid, cell):
                    cells.append(cell)
                    for other in _neighbours(cell):
                        if not flags.get(other, 0) & (frontier_open | frontier_closed | map_closed):
                            reach(other, cell)
                            flags[other] = flags.get(other, 0) | frontier_open
                            frontier_queue.append(other)
                flags[cell] = flags.get(cell, 0) | frontier_closed
            for cell in cells:
                flags[cell] = flags.get(cell, 0) | map_closed
            if len(cells) >= MIN_CLUSTER:
                clusters.append((point, cells))
            else:
                dropped.append(cells)
        for other in _neighbours(point):
            if not flags.get(other, 0) & (map_open | map_closed) and any(
                grid[y, x] == FREE for x, y in _neighbours(other)
            ):
                reach(other, point)
                flags[other] = flags.get(other, 0) | map_open
                queue.append(other)
        flags[point] = flags.get(point, 0) | map_closed
    return Search(clusters, dropped, distance, parent)


@dataclass(frozen=True, slots=True)
class Candidate:
    letter: str
    cells: list[Cell]
    size: int  # P_j
    centroid: Point  # c_j
    centre_cell: Cell  # m_j's cell
    start_cell: Cell  # s_j's cell
    goal_cell: Cell  # the dispatched goal point's cell
    bfs_m: float | None  # rho_j: the BFS distance of the centre cell

    @property
    def centre(self) -> Point:
        return _centre(self.centre_cell)

    @property
    def start(self) -> Point:
        return _centre(self.start_cell)


def _candidate(
    grid: np.ndarray[Any, Any], letter: str, start: Cell, cells: list[Cell], search: Search
) -> Candidate:
    """``build_frontier_candidate``: centroid, centre cell (nearest the centroid, ties to the lower (y, x)), start
    cell, and the goal point: the free unblocked neighbour nearest the centroid, preferring cells at least
    ``MIN_GOAL_M`` from the robot."""
    points = [_centre(c) for c in cells]
    centroid = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    centre = min(cells, key=lambda c: (round(math.dist(_centre(c), centroid) ** 2, 12), c[1], c[0]))
    robot = _centre(ROBOT)
    best_any: tuple[float, Cell] | None = None
    best_far: tuple[float, Cell] | None = None
    seen: set[Cell] = set()
    for cell in cells:
        for other in _neighbours(cell):
            if other in seen:
                continue
            seen.add(other)
            if grid[other[1], other[0]] != FREE:
                continue
            gap = math.dist(_centre(other), centroid) ** 2
            if best_any is None or gap < best_any[0]:
                best_any = (gap, other)
            far = math.dist(_centre(other), robot) >= MIN_GOAL_M
            if far and (best_far is None or gap < best_far[0]):
                best_far = (gap, other)
    chosen = best_far or best_any
    if chosen is None:
        raise RuntimeError(f"scripted scene: cluster {letter} has no free neighbour")
    return Candidate(
        letter,
        cells,
        len(cells),
        centroid,
        centre,
        start,
        chosen[1],
        search.distance_m.get(centre),
    )


@dataclass(frozen=True, slots=True)
class StartTerms:
    gain: float  # G_j
    distance_m: float  # D_j
    path_m: float  # d_0j
    time_s: float  # t_j
    turn_rad: float  # dpsi_j


def _frontier_path_cost(first_leg_m: float, origin: Point, target: Candidate) -> float:
    """``frontier_path_cost_with_start_world`` and its start-row twin ``initial_frontier_path_cost``
    (mrtsp_ordering.cpp): max(first leg + |m_j - s_j|, |origin - c_j| + |c_j - s_j|) - r_s. The first leg is
    |m_i - m_j| between frontiers and D_j from the robot; the origin is m_i or r."""
    return (
        max(
            first_leg_m + math.dist(target.centre, target.start),
            math.dist(origin, target.centroid) + math.dist(target.centroid, target.start),
        )
        - R_S
    )


def _start_terms(candidate: Candidate, robot: Point, yaw: float) -> StartTerms:
    """The start row's terms (``compute_mrtsp_start_cost``, mrtsp_ordering.cpp)."""
    gain = float(candidate.size) ** W_S
    straight = math.dist(robot, candidate.centre)
    distance = candidate.bfs_m if candidate.bfs_m is not None else straight
    path = _frontier_path_cost(distance, robot, candidate)
    heading = math.atan2(candidate.centre[1] - robot[1], candidate.centre[0] - robot[0])
    turn = abs(math.atan2(math.sin(heading - yaw), math.cos(heading - yaw)))
    return StartTerms(gain, distance, path, distance / V_MAX + turn / W_MAX, turn)


def _matrix(candidates: Sequence[Candidate], terms: Sequence[StartTerms]) -> np.ndarray[Any, Any]:
    """M with node 0 the robot (``build_cost_matrix``): M(0, j) = w_d d_0j / G_j + t_j / sqrt(G_j),
    M(i, j) = w_d d_ij / G_j, M(i, 0) = 0, the diagonal infinite."""
    n = len(candidates)
    matrix = np.full((n + 1, n + 1), np.inf)
    matrix[1:, 0] = 0.0
    for j, (target, term) in enumerate(zip(candidates, terms, strict=True)):
        matrix[0, j + 1] = W_D * term.path_m / term.gain + term.time_s / math.sqrt(term.gain)
        for i, source in enumerate(candidates):
            if i != j:
                path = _frontier_path_cost(
                    math.dist(source.centre, target.centre), source.centre, target
                )
                matrix[i + 1, j + 1] = W_D * path / term.gain
    return matrix


SCORE_TIE = 1e-12  # kScoreTieEpsilon (mrtsp_solver.cpp)
MAX_CANDIDATE_LIMIT = 60  # kMaxDpSolverCandidateLimit


def _pool_order(scores: Sequence[float], sizes: Sequence[int], a: int, b: int) -> int:
    """The pruning comparator of ``prune_mrtsp_candidates`` (mrtsp_solver.cpp): finite scores first, a lower score
    (beyond the tie epsilon), then the larger cluster, then the extraction order."""
    finite_a, finite_b = math.isfinite(scores[a]), math.isfinite(scores[b])
    if finite_a != finite_b:
        return -1 if finite_a else 1
    if finite_a and abs(scores[a] - scores[b]) > SCORE_TIE:
        return -1 if scores[a] < scores[b] else 1
    if not finite_a and scores[a] != scores[b]:
        return -1 if scores[a] < scores[b] else 1
    if sizes[a] != sizes[b]:
        return -1 if sizes[a] > sizes[b] else 1
    return -1 if a < b else (1 if a > b else 0)


def _route(matrix: np.ndarray[Any, Any], sizes: Sequence[int]) -> tuple[list[int], int, float]:
    """The pool of the K candidates lowest by M(0, j) and the exact minimum over simple paths of h = min(H, K)
    frontiers from the robot (``solve_bounded_horizon_mrtsp_order``). The minimum is found by enumerating the
    permutations, which gives the DP's result and is tractable only for the scripted scene (n = 5). Returns the
    route as candidate indices, K and the route cost."""
    n = matrix.shape[0] - 1
    scores = [float(matrix[0, j + 1]) for j in range(n)]
    order = sorted(
        range(n), key=functools.cmp_to_key(functools.partial(_pool_order, scores, sizes))
    )
    pool = [j + 1 for j in order[: min(max(CANDIDATE_LIMIT, 1), MAX_CANDIDATE_LIMIT, n)]]
    best: tuple[float, tuple[int, ...]] | None = None
    for path in itertools.permutations(pool, min(HORIZON, len(pool))):
        cost = float(matrix[0, path[0]]) + sum(
            float(matrix[a, b]) for a, b in itertools.pairwise(path)
        )
        if best is None or cost < best[0]:
            best = (cost, path)
    if best is None:
        raise RuntimeError("scripted scene: no route")
    return [j - 1 for j in best[1]], len(pool), best[0]


@dataclass(frozen=True, slots=True)
class Gate:
    visible: list[Cell]  # frontier cells counted (each once)
    reach: list[Point]  # where each ray stopped, in ray order
    bounds: tuple[int, int, int, int]  # the cluster box ±1 cell: x0, y0, x1, y1
    reveal_m: float  # V
    threshold_m: float  # min(P_j * resolution, MIN_REVEAL_M)


def _gate(later: np.ndarray[Any, Any], candidate: Candidate) -> Gate:
    """``compute_visible_reveal_gain`` (frontier_search.cpp) from the target pose on the later map (the pose of
    ``active_goal_visible_reveal_length``): rays every ``RAY_STEP_DEG`` marching half a cell, stopped outside the
    cluster box ±1 cell or at an occupied cell; the first unknown cell of a ray counts once if it is a frontier
    cell. V = counted cells × resolution."""
    xs = [c[0] for c in candidate.cells]
    ys = [c[1] for c in candidate.cells]
    x0, y0 = max(0, min(xs) - 1), max(0, min(ys) - 1)
    x1, y1 = min(WIDTH - 1, max(xs) + 1), min(HEIGHT - 1, max(ys) + 1)
    origin = _centre(candidate.goal_cell)
    # the sensor faces from the goal point to the frontier's reference point (its centroid); with a 360° field of
    # view the rays are spread evenly from that heading
    heading = math.atan2(candidate.centroid[1] - origin[1], candidate.centroid[0] - origin[0])
    count = math.ceil(360.0 / RAY_STEP_DEG)
    step = RESOLUTION_M * 0.5
    visible: list[Cell] = []
    reach: list[Point] = []
    for k in range(count):
        angle = heading + 2 * math.pi * k / count
        last = candidate.goal_cell
        end = origin
        distance = step
        while distance <= LIDAR_RANGE_M:
            end = (origin[0] + math.cos(angle) * distance, origin[1] + math.sin(angle) * distance)
            distance += step
            cell = _cell(end)
            if not _inside(cell) or not (x0 <= cell[0] <= x1 and y0 <= cell[1] <= y1):
                break
            if cell == last:
                continue
            last = cell
            value = later[cell[1], cell[0]]
            if value == OCCUPIED:
                break
            if value != UNKNOWN:
                continue
            if _is_frontier(later, cell) and cell not in visible:
                visible.append(cell)
            break
        reach.append(end)
    reveal = len(visible) * RESOLUTION_M
    return Gate(
        visible, reach, (x0, y0, x1, y1), reveal, min(candidate.size * RESOLUTION_M, MIN_REVEAL_M)
    )


# Drawing ---------------------------------------------------------------------------------------------------------

# Fixed layout in mm (constrained layout does not settle on equal-aspect maps with text strips in one pass)
MARGIN_MM, GAP_MM = 1.5, 4.0  # page edges; between columns
TITLE_MM = 7.5  # above each panel (6 pt title pad plus the bold mathtext label)
STRIP_GAP_MM = 1.5  # between a panel and its text strip
STRIP_MM = 22.5  # the text strip: 4 lines at 10 pt
LEGEND_MM = 16.0  # three legend rows at 9 pt
VIEW = (-0.2, WIDTH + 0.2, -0.2, HEIGHT + 0.2)  # every map panel's viewport, in cells
CELL_MARK = 3.0  # frontier-cell marker size (pt); one cell is about 5.5 pt at double width
# plain text is 8 pt (FIGURE_STYLE DR-10); anything with a subscript is 10 pt, so that mathtext's subscripts
# (70 %) are 7 pt
NOTE_PT = 8.0
MATH_PT = 10.0
GRID_COLOUR = "#BFBFBF"  # cell boundaries
LINE_SPACING = 1.45  # line pitch of a text strip, in multiples of the font size
CASING = [patheffects.withStroke(linewidth=2.8, foreground=EDGE)]


def _axis(point: Point) -> Point:
    """Metres to axis units (cells)."""
    return (point[0] / RESOLUTION_M, point[1] / RESOLUTION_M)


def _draw_map(ax: Any, grid: np.ndarray[Any, Any]) -> None:
    ax.imshow(
        grid,
        cmap=ListedColormap([MAP_FREE, MAP_UNKNOWN, MAP_OCCUPIED]),
        vmin=-0.5,
        vmax=2.5,
        origin="lower",
        extent=(0, WIDTH, 0, HEIGHT),
        interpolation="nearest",
        interpolation_stage="rgba",
        zorder=0,
    )
    ax.vlines(range(WIDTH + 1), 0, HEIGHT, colors=GRID_COLOUR, lw=0.4, zorder=0.5)
    ax.hlines(range(HEIGHT + 1), 0, WIDTH, colors=GRID_COLOUR, lw=0.4, zorder=0.5)
    ax.add_patch(Rectangle((0, 0), WIDTH, HEIGHT, fill=False, edgecolor=EDGE, lw=0.6, zorder=5))
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _robot(ax: Any, cell: Cell, yaw: float) -> None:
    x, y = cell[0] + 0.5, cell[1] + 0.5
    ax.plot(
        [x],
        [y],
        marker=(3, 0, math.degrees(yaw) - 90),
        ms=7,
        color="#000000",
        mec="#FFFFFF",
        mew=0.6,
        ls="none",
        zorder=4,
    )


def _cells(ax: Any, cells: Sequence[Cell], *, face: str, size: float = CELL_MARK) -> None:
    ax.plot(
        [c[0] + 0.5 for c in cells],
        [c[1] + 0.5 for c in cells],
        marker="s",
        ms=size,
        mfc=face,
        mec="#000000",
        mew=0.6,
        ls="none",
        zorder=3,
    )


def _letters(ax: Any, candidates: Sequence[Candidate]) -> None:
    for candidate in candidates:
        cx, cy = _axis(candidate.centroid)
        dx, dy = LETTER_OFFSETS[candidate.letter]
        ax.text(cx + dx, cy + dy, candidate.letter, ha="center", va="center", fontsize=9, zorder=5)


def _strip(ax: Any, lines: Sequence[tuple[str, float]]) -> None:
    """A panel's text strip: one (text, size) per line, top left, lines spaced in points."""
    ax.set_axis_off()
    offset = 0.0
    for text, size in lines:
        offset += size * LINE_SPACING
        ax.annotate(
            text,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0.0, -offset),
            textcoords="offset points",
            ha="left",
            va="baseline",
            fontsize=size,
        )


def _title(ax: Any, index: int, text: str) -> None:
    panel_title(ax, index, text)
    ax.set_title("schematic", loc="right", fontsize=NOTE_PT, color=GREY, style="italic")


def _label(ax: Any, text: str, at: Point, to: Point | None = None) -> None:
    """A symbol label (``MATH_PT``), with a thin leader line to ``to`` when given."""
    if to is None:
        ax.text(*at, text, fontsize=MATH_PT, ha="center", va="center", zorder=6)
        return
    ax.annotate(
        text,
        xy=to,
        xytext=at,
        fontsize=MATH_PT,
        ha="center",
        va="center",
        zorder=6,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=EDGE, shrinkA=1, shrinkB=1),
    )


def _notation(ax: Any, candidate: Candidate, search: Search) -> None:
    """Panel (c): the start-row geometry of one candidate."""
    robot = (ROBOT[0] + 0.5, ROBOT[1] + 0.5)
    path = [candidate.centre_cell]
    while path[-1] in search.parent:
        path.append(search.parent[path[-1]])
    px = [c[0] + 0.5 for c in path]
    py = [c[1] + 0.5 for c in path]
    ax.plot(px, py, color="#000000", lw=1.0, ls=(0, (3, 1.5)), zorder=3)
    centroid = _axis(candidate.centroid)
    centre = _axis(candidate.centre)
    start = _axis(candidate.start)
    ax.plot(*zip(robot, centroid, strict=True), color="#000000", lw=0.6, ls=":", zorder=3)
    for a in (centre, centroid):
        ax.plot(*zip(a, start, strict=True), color="#000000", lw=0.6, zorder=3)
    ax.plot([robot[0], robot[0] + 3.2], [robot[1], robot[1]], color="#000000", lw=0.6, zorder=3)
    # the arc runs from psi through the signed wrapped turn towards m_j (its size is dpsi_j)
    toward = math.atan2(centre[1] - robot[1], centre[0] - robot[0])
    turn = math.atan2(math.sin(toward - ROBOT_YAW), math.cos(toward - ROBOT_YAW))
    ends = sorted((math.degrees(ROBOT_YAW), math.degrees(ROBOT_YAW + turn)))
    ax.add_patch(
        Arc(robot, 2 * ARC_RADIUS, 2 * ARC_RADIUS, theta1=ends[0], theta2=ends[1], lw=0.8, zorder=3)
    )
    _cells(ax, candidate.cells, face="#000000")
    ax.plot(*centroid, marker="P", ms=5.5, color="#FFFFFF", mec="#000000", mew=0.7, zorder=4)
    ax.plot(*centre, marker="o", ms=5, color="#FFFFFF", mec="#000000", mew=0.8, zorder=4)
    ax.plot(*start, marker="^", ms=5, color="#FFFFFF", mec="#000000", mew=0.8, zorder=4)
    _robot(ax, ROBOT, ROBOT_YAW)
    for text, at, to in NOTATION_LABELS:
        _label(ax, text, at, to)
    _label(ax, f"$j$ = {candidate.letter}", (centre[0] + 7.5, centre[1] + 0.2))


def _matrix_table(ax: Any, matrix: np.ndarray[Any, Any], names: Sequence[str]) -> None:
    """Panel (d): M as a table, rows from, columns to."""
    n = len(names)
    ax.set_xlim(-1.3, n)
    ax.set_ylim(n, -1.3)
    ax.set_axis_off()
    for i in range(n):
        for x, y in ((i + 0.5, -0.55), (-0.65, i + 0.5)):
            ax.text(x, y, names[i], ha="center", va="center", fontsize=MATH_PT)
        for j in range(n):
            value = float(matrix[i, j])
            infinite = not math.isfinite(value)
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=MAP_UNKNOWN if infinite else "#FFFFFF",
                    edgecolor=EDGE,
                    lw=0.4,
                )
            )
            text = r"$\infty$" if infinite else f"{value:.2f}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=NOTE_PT)
    ax.text(-0.65, -0.55, "from\\to", ha="center", va="center", fontsize=NOTE_PT, color=GREY)


def _route_panel(
    ax: Any, candidates: Sequence[Candidate], order: Sequence[int], first: Candidate
) -> None:
    points = [(ROBOT[0] + 0.5, ROBOT[1] + 0.5)] + [_axis(candidates[j].centre) for j in order]
    line = ax.plot(*zip(*points, strict=True), color=ROUTE_COLOUR, lw=1.6, zorder=3)[0]
    line.set_path_effects(CASING)
    for number, point in enumerate(points[1:], start=1):
        ax.plot(
            *point, marker="o", ms=10, color="#FFFFFF", mec=EDGE, mew=0.7, zorder=4, clip_on=False
        )
        ax.text(*point, str(number), ha="center", va="center", fontsize=NOTE_PT, zorder=5)
    goal = (first.goal_cell[0] + 0.5, first.goal_cell[1] + 0.5)
    ax.plot(*goal, marker="*", ms=8, color="#000000", mec="#FFFFFF", mew=0.5, zorder=5.5)


def _gate_panel(ax: Any, search: Search, gate: Gate, target: Candidate) -> None:
    reach = [_axis(p) for p in gate.reach]
    x0, y0, x1, y1 = gate.bounds
    box = Rectangle(
        (x0, y0),
        x1 - x0 + 1,
        y1 - y0 + 1,
        fill=False,
        edgecolor="#000000",
        lw=0.8,
        ls=(0, (3, 1.5)),
        zorder=3,
    )
    ax.add_patch(box)
    # a ray stops at its first sample outside the box: the fill is clipped to the box it cannot leave
    for patch in ax.fill(
        *zip(*reach, strict=True), facecolor=ROUTE_COLOUR, alpha=0.55, lw=0, zorder=1
    ):
        patch.set_clip_path(box)
    _cells(ax, [c for _, cells in search.clusters for c in cells], face="#000000")
    _cells(ax, [c for cells in search.dropped for c in cells], face="#FFFFFF")
    if gate.visible:
        _cells(ax, gate.visible, face=ROUTE_COLOUR, size=CELL_MARK + 1.5)
    goal = (target.goal_cell[0] + 0.5, target.goal_cell[1] + 0.5)
    ax.plot(*goal, marker="*", ms=8, color="#000000", mec="#FFFFFF", mew=0.5, zorder=4.5)
    origin = _centre(LATER_ROBOT)
    yaw = math.atan2(
        _centre(target.goal_cell)[1] - origin[1], _centre(target.goal_cell)[0] - origin[0]
    )
    _robot(ax, LATER_ROBOT, yaw)


def _layout(preset: str) -> tuple[Figure, list[Any], list[Any]]:
    """Two rows of three panels at the preset's double width, each with its text strip below, the legend under
    them; every map panel the same size (one viewport, DR-8)."""
    width = PRESETS[preset].widths_mm["double"]
    column = (width - 2 * MARGIN_MM - 2 * GAP_MM) / 3
    panel = column * (VIEW[3] - VIEW[2]) / (VIEW[1] - VIEW[0])
    row = TITLE_MM + panel + STRIP_GAP_MM + STRIP_MM
    total = 2 * row + LEGEND_MM
    fig, axes = figure(preset, "double", aspect=total / width)
    fig.set_layout_engine("none")
    fig.delaxes(axes[0][0])
    maps: list[Any] = []
    strips: list[Any] = []
    for index in range(6):
        left = MARGIN_MM + (index % 3) * (column + GAP_MM)
        top = (index // 3) * row + TITLE_MM
        bottom = total - top - panel
        maps.append(fig.add_axes((left / width, bottom / total, column / width, panel / total)))
        strip = bottom - STRIP_GAP_MM - STRIP_MM
        strips.append(fig.add_axes((left / width, strip / total, column / width, STRIP_MM / total)))
    return fig, maps, strips


def _legend_handles() -> list[Any]:
    def mark(marker: str, label: str, **kw: Any) -> Line2D:
        style: dict[str, Any] = dict(color="#000000", mec="#000000", ls="none", ms=5) | kw
        return Line2D([], [], marker=marker, label=label, **style)

    return [
        Patch(facecolor=MAP_FREE, edgecolor=EDGE, lw=0.4, label="Free"),
        Patch(facecolor=MAP_UNKNOWN, edgecolor=EDGE, lw=0.4, label="Unknown"),
        Patch(facecolor=MAP_OCCUPIED, edgecolor=EDGE, lw=0.4, label="Occupied"),
        mark((3, 0, -90), "Robot (heading)", ms=7, mec="#FFFFFF", mew=0.6),  # type: ignore[arg-type]
        mark("s", "Frontier cell", ms=CELL_MARK, mew=0.6),
        mark("s", "Dropped cluster cell", ms=CELL_MARK, mfc="#FFFFFF", mew=0.6),
        mark("P", "Centroid", mfc="#FFFFFF", mew=0.7, ms=5.5),
        mark("o", "Centre cell", mfc="#FFFFFF", mew=0.8),
        mark("^", "Start cell", mfc="#FFFFFF", mew=0.8),
        mark("*", "Goal point sent", ms=8, mec="#FFFFFF", mew=0.5),
        Line2D([], [], color=ROUTE_COLOUR, lw=1.6, path_effects=CASING, label="DP route (order)"),
        Patch(facecolor=ROUTE_COLOUR, alpha=0.55, lw=0, label="Ray reach"),
        mark("s", "Visible frontier cell", ms=CELL_MARK + 1.5, mfc=ROUTE_COLOUR, mew=0.6),
        Line2D([], [], color="#000000", lw=0.8, ls=(0, (3, 1.5)), label="Cluster box ±1 cell"),
    ]


def f20(ctx: FigureContext) -> FigureOut:
    grid = _scene()
    search = _frontier_search(grid, ROBOT)
    candidates = [
        _candidate(grid, LETTERS[k], start, cells, search)
        for k, (start, cells) in enumerate(search.clusters)
    ]
    robot = _centre(ROBOT)
    terms = [_start_terms(c, robot, ROBOT_YAW) for c in candidates]
    matrix = _matrix(candidates, terms)
    order, pool, route_cost = _route(matrix, [c.size for c in candidates])
    first = candidates[order[0]]
    later = _later(grid)
    later_search = _frontier_search(later, LATER_ROBOT)
    gate = _gate(later, first)
    # the scene is scripted for these cases; the texts below are written for them
    if len(candidates) != len(LETTERS):
        raise RuntimeError(f"scripted scene: {len(candidates)} candidates, expected {len(LETTERS)}")
    if math.dist(_centre(LATER_ROBOT), _centre(first.goal_cell)) <= COMPLETE_WITHIN_M:
        raise RuntimeError("scripted scene: panel (f) robot within the completion distance")
    if gate.reveal_m >= gate.threshold_m:
        raise RuntimeError("scripted scene: panel (f) gate keeps the goal")
    index = LETTERS.index(NOTATION_LETTER)
    notation, note = candidates[index], terms[index]

    fig, maps, strips = _layout(ctx.preset)
    names = ["$r$", *(c.letter for c in candidates)]

    # (a) decision map
    ax = maps[0]
    _draw_map(ax, grid)
    _robot(ax, ROBOT, ROBOT_YAW)
    _title(ax, 0, "Scripted decision map")
    ax.add_artist(
        AnchoredSizeBar(
            ax.transData,
            1 / RESOLUTION_M,
            "1 m",
            loc="lower right",
            frameon=False,
            size_vertical=0.25,
            label_top=True,
            sep=1.5,
            borderpad=0.6,
            fontproperties={"size": 8},
        )
    )
    # (b) wavefront frontier detection
    ax = maps[1]
    _draw_map(ax, grid)
    _cells(ax, [c for _, cells in search.clusters for c in cells], face="#000000")
    _cells(ax, [c for cells in search.dropped for c in cells], face="#FFFFFF")
    _robot(ax, ROBOT, ROBOT_YAW)
    _letters(ax, candidates)
    _title(ax, 1, "Wavefront frontiers")
    # (c) cost terms of one candidate
    ax = maps[2]
    _draw_map(ax, grid)
    _notation(ax, notation, search)
    _title(ax, 2, "Cost terms of candidate $j$")
    # (d) cost matrix
    ax = maps[3]
    _matrix_table(ax, matrix, names)
    _title(ax, 3, "Cost matrix $M$")
    # (e) DP route
    ax = maps[4]
    _draw_map(ax, grid)
    _cells(ax, [c for _, cells in search.clusters for c in cells], face="#000000")
    _route_panel(ax, candidates, order, first)
    _robot(ax, ROBOT, ROBOT_YAW)
    _letters(ax, candidates)
    _title(ax, 4, "DP route")
    # (f) visibility gate
    ax = maps[5]
    _draw_map(ax, later)
    _gate_panel(ax, later_search, gate, first)
    _title(ax, 5, "Visibility gate, later tick")

    texts: list[list[tuple[str, float]]] = [
        [
            ("Scripted decision map", NOTE_PT),
            ("(filter not simulated);", NOTE_PT),
            (r"robot at $r$ with heading $\psi$", NOTE_PT),
        ],
        [
            ("Frontier cell: unknown, a free and no", NOTE_PT),
            ("blocked 8-neighbour. BFS from the robot;", NOTE_PT),
            (f"8-connected clusters under {MIN_CLUSTER} cells dropped", NOTE_PT),
        ],
        [
            (r"$d_{0j} = \max(D_j + |m_j - s_j|,$", MATH_PT),
            (rf"$\qquad |r - c_j| + |c_j - s_j|) - r_s = {note.path_m:.2f}$ m", MATH_PT),
            (rf"$t_j = L_j / v + \Delta\psi_j / w = {note.time_s:.2f}$ s", MATH_PT),
            (
                rf"$L_j = D_j = {note.distance_m:.2f}$ m (BFS), $G_j = P_j = {notation.size}$",
                MATH_PT,
            ),
        ],
        [
            (r"$M(0,j) = w_d\, d_{0j} / G_j + t_j / \sqrt{G_j}$", MATH_PT),
            (r"$M(i,j) = w_d\, d_{ij} / G_j$,  $M(i,0) = 0$", MATH_PT),
            (r"$d_{ij} = \max(|m_i - m_j| + |m_j - s_j|,$", MATH_PT),
            (r"$\qquad |m_i - c_j| + |c_j - s_j|) - r_s$", MATH_PT),
        ],
        [
            (rf"Pool: the $K = {pool}$ lowest by $M(0,j)$; exact minimum", NOTE_PT),
            (rf"over paths of $h = \min(H, K) = {min(HORIZON, pool)}$ frontiers;", NOTE_PT),
            (f"route cost {route_cost:.2f}; only frontier 1 is sent", NOTE_PT),
        ],
        [
            (
                rf"$V = {gate.reveal_m:.2f}$ m $<$ $\min(P_j \cdot \mathrm{{res}},\, {MIN_REVEAL_M:g}$ m$)$",
                MATH_PT,
            ),
            (rf"$= {gate.threshold_m:.2f}$ m: cancel at once,", MATH_PT),
            ("without a replacement check;", NOTE_PT),
            ("reselection at a later map tick", NOTE_PT),
        ],
    ]
    for strip, lines in zip(strips, texts, strict=True):
        _strip(strip, lines)
    # no finish(): its evidence-level label marks data from a study, and this figure has none; every panel is
    # tagged "schematic" instead
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=5,
        borderaxespad=0.2,
        columnspacing=1.2,
    )

    notes = [
        "Schematic: a scripted 30 × 24-cell scene drawn at 0.25 m per cell; illustrative, not data, no run behind "
        "it. The steps are computed on the scene with the explorer's rules as implemented and the Default "
        f"parameters (w_d = {W_D:g}, w_s = {W_S:g}, r_s = {R_S:g} m, v = {V_MAX:g} m/s, w = {W_MAX:g} rad/s, "
        f"K = {CANDIDATE_LIMIT}, H = {HORIZON}, clusters of at least {MIN_CLUSTER} cells, goal points at least "
        f"{MIN_GOAL_M:g} m from the robot); the numbers depend on the coarse cells (P_j counts cells).",
        "(b) Frontier cells as the explorer's two-level BFS finds them; the global costmap's blocked cells are "
        "taken as the occupied cells (no inflation), and the decision-map filter is not drawn. (c) Symbols of the "
        "cost model: robot r and heading psi, cluster size P_j, centroid c_j, centre cell m_j (the cluster cell "
        "nearest c_j), start cell s_j (the first cell the BFS reached), BFS distance D_j = rho_j (dashed: "
        "8-connected hops of one cell each through any cell next to free space, here along wall cells, then along the cluster from "
        "s_j; a grid proxy, not a path), straight distance |r - c_j| (dotted), heading change dpsi_j; "
        "G_j = P_j^w_s. This is an MRTSP-inspired ratio cost, not the classical minimum-ratio TSP objective.",
        "(d) M(0, j) = w_d d_0j / G_j + t_j / sqrt(G_j), M(i, j) = w_d d_ij / G_j, M(i, 0) = 0, diagonal infinite "
        "(rows: from; columns: to). (e) The route's straight segments give the order, not a path; the sequence is "
        "recomputed after map or frontier updates.",
        f"(f) At a later map tick (at least {MIN_INTERVAL_S:g} s after dispatch, "
        f"goal_preemption_min_interval_s) the robot is more than {COMPLETE_WITHIN_M:g} m from the target. Rays "
        f"every {RAY_STEP_DEG:g} degree from the target pose, starting at the heading towards the cluster's "
        "centroid, run on the SLAM map and stop at the cluster box ±1 cell or at an occupied cell; the first "
        "unknown cell of a ray counts once if it is a frontier cell; V = counted cells × resolution. The goal is "
        f"kept while V >= min(P_j × resolution, {MIN_REVEAL_M:g} m); otherwise it is cancelled at once without a "
        "replacement check, and reselection at a later map tick may pick the same frontier (D-18). Such a "
        f"low-gain cancel happens at most once per goal_preemption_min_interval_s ({MIN_INTERVAL_S:g} s). Only "
        "blocked goals and reveals that cannot be evaluated take the two-hit replacement path (not drawn).",
    ]

    def rounded(point: Point) -> list[float]:
        return [round(point[0], 4), round(point[1], 4)]

    extra = {
        "schematic": dict(
            cells=[WIDTH, HEIGHT],
            resolution_m=RESOLUTION_M,
            parameters=dict(
                weight_distance_wd=W_D,
                weight_gain_ws=W_S,
                sensor_effective_range_m=R_S,
                max_linear_speed_vmax=V_MAX,
                max_angular_speed_wmax=W_MAX,
                dp_solver_candidate_limit=CANDIDATE_LIMIT,
                dp_planning_horizon=HORIZON,
                min_frontier_size_cells=MIN_CLUSTER,
                frontier_candidate_min_goal_distance_m=MIN_GOAL_M,
                goal_preemption_lidar_min_reveal_length_m=MIN_REVEAL_M,
                goal_preemption_lidar_range_m=LIDAR_RANGE_M,
                goal_preemption_lidar_ray_step_deg=RAY_STEP_DEG,
                goal_preemption_complete_if_within_m=COMPLETE_WITHIN_M,
                goal_preemption_min_interval_s=MIN_INTERVAL_S,
            ),
            mirrors=MIRRORS,
            robot=dict(cell=list(ROBOT), yaw_rad=ROBOT_YAW),
            candidates=[
                dict(
                    letter=c.letter,
                    size=c.size,
                    centroid_m=rounded(c.centroid),
                    centre_cell=list(c.centre_cell),
                    start_cell=list(c.start_cell),
                    goal_cell=list(c.goal_cell),
                    bfs_m=c.bfs_m,
                    gain=t.gain,
                    path_m=round(t.path_m, 4),
                    time_s=round(t.time_s, 4),
                    turn_rad=round(t.turn_rad, 4),
                )
                for c, t in zip(candidates, terms, strict=True)
            ],
            dropped_cells=[[list(c) for c in cells] for cells in search.dropped],
            matrix=[
                [round(float(v), 4) if math.isfinite(v) else None for v in row] for row in matrix
            ],
            route=[candidates[j].letter for j in order],
            route_cost=round(route_cost, 4),
            pool=pool,
            gate=dict(
                target=first.letter,
                robot_cell=list(LATER_ROBOT),
                box_cells=list(gate.bounds),
                visible_cells=[list(c) for c in gate.visible],
                reveal_m=gate.reveal_m,
                threshold_m=gate.threshold_m,
                cancelled=True,
            ),
        )
    }
    return FigureOut(
        "F20",
        fig,
        "double",
        set(),
        "Schematic of the explorer's decision steps on a synthetic grid: the decision map, wavefront frontier "
        "detection, the cost terms of one candidate, the cost matrix, the DP route and the visibility gate.",
        notes,
        extra,
    )


def figures(ctx: FigureContext) -> Iterator[FigureOut]:
    """F20, built when the caller asks for it."""
    yield f20(ctx)
