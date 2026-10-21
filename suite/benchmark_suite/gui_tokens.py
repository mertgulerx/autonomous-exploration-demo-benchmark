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

"""Colour tokens of the approved Main design (light and dark) and the few fixed colours.

Generated from the design package's ``tokens.json`` (docs/suite-gui-design/, 17 October 2026), which is itself
generated from the mockups' ``styles.css``. The package never reads ``docs/`` at runtime. Colours are CSS
strings, ``#RRGGBB`` or ``#RRGGBBAA`` with the alpha last; ``gui_theme.css_color`` converts them for Qt.
"""

from types import MappingProxyType
from typing import Final, Mapping

LIGHT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "surface.canvas": "#F5F6F8",
        "surface.card": "#FFFFFF",
        "surface.sunken": "#F2F3F5",
        "surface.hover": "#F7F8FA",
        "surface.selected": "#EEF0F3",
        "border.line": "#E3E5E9",
        "border.strong": "#CBCFD6",
        "text.ink": "#15171C",
        "text.body": "#2A2D34",
        "text.secondary": "#5C6068",
        "text.tertiary": "#686C74",
        "text.disabled": "#A9ADB4",
        "text.onInk": "#FFFFFF",
        "accent.primary": "#0C7CA4",
        "accent.hover": "#086D90",
        "accent.pressed": "#075D7C",
        "accent.disabled": "#8CC7E5",
        "accent.disabledText": "#FFFFFF",
        "accent.soft": "#D3EFFD",
        "accent.softText": "#056A8D",
        "state.ok": "#15803D",
        "state.okSoft": "#EAF6EE",
        "state.okHalo": "#E3F4EA",
        "state.okLine": "#D3EBDB",
        "state.okStrong": "#1E5A35",
        "state.warn": "#A15C07",
        "state.warnSoft": "#FDF3E1",
        "state.bad": "#C42B1C",
        "state.badSoft": "#FDEDEB",
        "state.badLine": "#EBB4AE",
        "state.badSolid": "#C42B1C",
        "state.neutral": "#4B5058",
        "state.neutralSoft": "#F0F1F3",
        "data.completed": "#1F9D55",
        "data.timeLimit": "#D9A21B",
        "data.failure": "#D9480F",
        "data.analysis": "#6D4FC2",
        "data.remaining": "#E7E9ED",
        "data.remainingHatchA": "#E7E9ED",
        "data.remainingHatchB": "#F1F2F5",
        "data.legendRemainingA": "#D8DBE0",
        "data.legendRemainingB": "#EEF0F2",
        "data.path": "#E8590C",
        "data.coverage": "#0C7CA4",
        "control.checkBorder": "#8B9098",
        "control.checkBorderDisabled": "#D5D8DD",
        "control.switchOff": "#C9CDD3",
        "map.outside": "#F1F2F4",
        "map.hatch": "#2B2E3512",
        "map.halo": "#FFFFFF",
        "map.free": "#FFFFFF",
        "map.grid": "#15171C17",
        "map.gridAxis": "#15171C2E",
        "map.float": "#FFFFFFF0",
        "chart.grid": "#ECEEF1",
        "chart.axis": "#D5D8DD",
        "overlay.scrim": "#11131857",
        "overlay.shadowSmall": "#1113180F",
        "overlay.tooltip": "#15171C",
        "overlay.tooltipText": "#FFFFFF",
        "canvas.dot": "#15171C24",
    }
)
DARK: Final[Mapping[str, str]] = MappingProxyType(
    {
        "surface.canvas": "#181B20",
        "surface.card": "#1F2329",
        "surface.sunken": "#24282F",
        "surface.hover": "#262B32",
        "surface.selected": "#282D34",
        "border.line": "#30353D",
        "border.strong": "#3D434C",
        "text.ink": "#D9DCE1",
        "text.body": "#C2C6CD",
        "text.secondary": "#979DA6",
        "text.tertiary": "#8E949D",
        "text.disabled": "#5D636C",
        "text.onInk": "#1F2329",
        "accent.primary": "#0C7CA4",
        "accent.hover": "#086D90",
        "accent.pressed": "#075D7C",
        "accent.disabled": "#1E3A45",
        "accent.disabledText": "#6B8792",
        "accent.soft": "#18343F",
        "accent.softText": "#74B4D0",
        "state.ok": "#64B98A",
        "state.okSoft": "#1C3226",
        "state.okHalo": "#1C3226",
        "state.okLine": "#24402F",
        "state.okStrong": "#9FD3B4",
        "state.warn": "#D3A35C",
        "state.warnSoft": "#352C1D",
        "state.bad": "#DE8178",
        "state.badSoft": "#3B2422",
        "state.badLine": "#5E2C28",
        "state.badSolid": "#B23A2E",
        "state.neutral": "#A0A6AF",
        "state.neutralSoft": "#2C3037",
        "data.completed": "#3E9F6A",
        "data.timeLimit": "#C99A38",
        "data.failure": "#D06A3E",
        "data.analysis": "#A390DC",
        "data.remaining": "#30353D",
        "data.remainingHatchA": "#30353D",
        "data.remainingHatchB": "#272B32",
        "data.legendRemainingA": "#4A505A",
        "data.legendRemainingB": "#30353D",
        "data.path": "#E58448",
        "data.coverage": "#4D9BBE",
        "control.checkBorder": "#6A707A",
        "control.checkBorderDisabled": "#3D434C",
        "control.switchOff": "#3D434C",
        "map.outside": "#1C1F24",
        "map.hatch": "#FFFFFF0A",
        "map.halo": "#1F2329",
        "map.free": "#262B33",
        "map.grid": "#D9DCE112",
        "map.gridAxis": "#D9DCE124",
        "map.float": "#1F2329EB",
        "chart.grid": "#2C3138",
        "chart.axis": "#3D434C",
        "overlay.scrim": "#00000080",
        "overlay.shadowSmall": "#0000004C",
        "overlay.tooltip": "#D9DCE1",
        "overlay.tooltipText": "#1F2329",
        "canvas.dot": "#D9DCE11A",
    }
)
assert LIGHT.keys() == DARK.keys()

# overlay.shadowPopup as layers (dx, dy, blur, spread, colour), outermost first.
POPUP_SHADOW: Final[Mapping[str, tuple[tuple[int, int, int, int, str], ...]]] = MappingProxyType(
    {
        "light": ((0, 12, 32, 0, "#11131829"), (0, 2, 6, 0, "#11131814")),
        "dark": ((0, 14, 36, 0, "#00000073"), (0, 0, 0, 1, "#FFFFFF0A")),
    }
)
# Tooltips keep one shadow in both themes: 0 4px 12px at 18 %.
TOOLTIP_SHADOW: Final = (0, 4, 12, 0, "#1113182E")
# The switch knob's shadow in both themes: 0 1px 2px black at 20 %.
KNOB_SHADOW: Final = (0, 1, 2, 0, "#00000033")

# World map layers per theme (the engine's grey image: 245 free, 30 wall, 170 elsewhere); explored cells are
# data.coverage at EXPLORED_OPACITY over free cells.
MAP: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "light": MappingProxyType(
            {
                "free": "#FFFFFF",
                "wall": "#2B2E35",
                "excludedFree": "#FFFFFF",
                "excludedWall": "#BCC0C7",
            }
        ),
        "dark": MappingProxyType(
            {
                "free": "#262B33",
                "wall": "#8D949E",
                "excludedFree": "#22262C",
                "excludedWall": "#4A5059",
            }
        ),
    }
)
EXPLORED_OPACITY: Final[Mapping[str, float]] = MappingProxyType({"light": 0.22, "dark": 0.24})

# Colours that do not change with the theme.
ON_ACCENT: Final = "#FFFFFF"
ROBOT_ARROW_OUTLINE: Final = "#FFFFFF"
ROBOT_HALO: Final = "#E8590C2E"
ROBOT_HALO_STROKE: Final = "#E8590C73"
ROBOT_GLOW_DARK: Final = "#FFFFFF1A"
REPORT_PAPER_DARK: Final = "#DDE0E4"
