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

"""Figure style: journal presets, encodings and the figure factory (docs/research/FIGURE_STYLE.md §2, §3, §6;
FP-4 to FP-7, FP-9).

- Presets (FP-6): ``ieee-journal`` (9 pt legends), ``ieee-conference`` and ``elsevier`` (8 pt); widths in mm.
  ``sans`` switches to Liberation Sans with sans-serif math for sans-serif journals (FP-9).
- Profiles: Default black; the other profiles take Okabe–Ito colours in the order of their parameter hashes,
  so one profile keeps one colour in every figure of a study; sky blue is reserved for routes (FP-7). Colour is
  always paired with a marker and a line style.
- Outcomes in greys and hatches, never a profile hue (FP-4); map greys with the outside hatched (FP-5).
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .errors import Refused  # noqa: E402

STYLE = Path(__file__).with_name("publication.mplstyle")
MM_PER_INCH = 25.4


@dataclass(frozen=True, slots=True)
class Preset:
    widths_mm: Mapping[str, float]
    legend_pt: float


PRESETS = {
    "ieee-journal": Preset({"single": 88.9, "double": 182.0}, 9.0),
    "ieee-conference": Preset({"single": 88.9, "double": 182.0}, 8.0),
    "elsevier": Preset({"single": 90.0, "onehalf": 140.0, "double": 190.0}, 8.0),
}
SANS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
}
BASELINE_COLOUR = "#000000"
PROFILE_COLOURS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
)
ROUTE_COLOUR = "#56B4E9"  # reserved for routes and rays in method figures, never a profile (FP-7)
MARKERS = ("o", "s", "^", "D", "v", "P")
LINESTYLES: tuple[Any, ...] = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
OUTCOMES: dict[str, dict[str, Any]] = {
    "completed": dict(facecolor="#4D4D4D", hatch=None, label="Completed"),
    "time_limit": dict(facecolor="#8C8C8C", hatch=None, label="Time limit"),
    "no_movement": dict(facecolor="#CCCCCC", hatch=None, label="No movement"),
    "missing_startup": dict(facecolor="#FFFFFF", hatch="////", label="Missing (startup)"),
    "missing_platform": dict(facecolor="#FFFFFF", hatch="\\\\\\\\", label="Missing (platform)"),
    "excluded_runtime_failure": dict(facecolor="#FFFFFF", hatch="xxxx", label="Runtime failure"),
    "excluded_stopped": dict(facecolor="#FFFFFF", hatch="....", label="Stopped"),
    "not_done": dict(facecolor="none", hatch=None, label="Not done"),
}
MAP_FREE, MAP_UNKNOWN, MAP_OCCUPIED, MAP_OUTSIDE = "#FFFFFF", "#D9D9D9", "#6E6E6E", "#EDEDED"


@dataclass(frozen=True, slots=True)
class ProfileStyle:
    name: str
    color: str
    marker: str
    linestyle: Any
    is_baseline: bool


def profile_styles(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, ProfileStyle]:
    """One encoding per profile of a study, keyed by profile ID."""
    styles = {}
    others = sorted(
        (row["parameter_hash"], profile_id)
        for profile_id, row in profiles.items()
        if not row["is_baseline"]
    )
    if len(others) > len(PROFILE_COLOURS):
        raise Refused(
            f"{len(others)} profiles besides Default: at most {len(PROFILE_COLOURS)} per study"
        )
    for profile_id, row in profiles.items():
        if row["is_baseline"]:
            styles[profile_id] = ProfileStyle(
                row["profile_name"], BASELINE_COLOUR, MARKERS[0], LINESTYLES[0], True
            )
    for index, (_, profile_id) in enumerate(others):
        styles[profile_id] = ProfileStyle(
            profiles[profile_id]["profile_name"],
            PROFILE_COLOURS[index],
            MARKERS[index + 1],
            LINESTYLES[index + 1],
            False,
        )
    return styles


@contextmanager
def context(preset: str, sans: bool) -> Iterator[None]:
    """The style with the preset's legend size (and the sans-serif variant) for the figures drawn inside."""
    if preset not in PRESETS:
        raise Refused(f"unknown preset {preset}; use one of {', '.join(PRESETS)}")
    overrides: dict[str, Any] = {"legend.fontsize": PRESETS[preset].legend_pt}
    if sans:
        overrides |= SANS
    with plt.style.context(str(STYLE)), matplotlib.rc_context(overrides):
        yield


def figure(
    preset: str,
    width: str = "single",
    *,
    aspect: float = 0.62,
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kw: Any,
) -> tuple[Figure, Any]:
    """A figure at its final print size (the journal's column width)."""
    width_in = PRESETS[preset].widths_mm[width] / MM_PER_INCH
    return plt.subplots(
        nrows, ncols, figsize=(width_in, width_in * aspect), squeeze=False, **subplot_kw
    )


def width_pt(preset: str, width: str) -> float:
    """The page width a saved figure must have, in PostScript points (FIGURE_STYLE §11 item 2)."""
    return PRESETS[preset].widths_mm[width] / MM_PER_INCH * 72
