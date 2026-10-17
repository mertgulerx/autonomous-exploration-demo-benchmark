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

"""Value formatting shared by the GUI pages: durations, words, numbers and table cells ("—" when a value is
unavailable)."""

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


def duration(seconds: Any) -> str:
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def humanize(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def finite(value: object) -> bool:
    """A finite int or float (not a boolean)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value: object) -> float | None:
    """A finite number as a float; None for anything else (None, NaN, booleans, text)."""
    return float(value) if isinstance(value, (int, float)) and finite(value) else None


def fixed(value: object, digits: int) -> str | None:
    """``value`` with ``digits`` decimals and a true minus sign (no sign on a rounded zero); None when the value is
    unavailable (HANDOFF §6.6)."""
    found = number(value)
    if found is None:
        return None
    text = f"{found:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "\u2212")


def worker_values(metrics: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """A worker's Coverage, Sim time and Distance (Campaigns tiles and Monitor cards): label, value ("—" when
    unavailable) and unit (none when unavailable)."""
    coverage = fixed(metrics.get("coverage_percent"), 1)
    distance = fixed(metrics.get("distance_m"), 1)
    return [
        ("Coverage", coverage or "—", "%" if coverage else ""),
        ("Sim time", duration(metrics.get("simulation_elapsed")), ""),
        ("Distance", distance or "—", "m" if distance else ""),
    ]


def cell_text(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key in {
        "simulation_elapsed",
        "pre_motion_simulation_elapsed",
        "pre_motion_wall_elapsed",
    }:
        return duration(value)
    if key in {"cpu_percent", "cpu_average_percent", "coverage_percent"} and isinstance(
        value, (float, int)
    ):
        return f"{value:.1f}%"
    if key in {"rss_mib", "rss_average_mib"} and isinstance(value, (float, int)):
        return f"{value:.1f} MiB"
    if key == "distance_m" and isinstance(value, (float, int)):
        return f"{value:.2f} m"
    if key == "average_speed_mps" and isinstance(value, (float, int)):
        return f"{value:.2f} m/s"
    if key == "bytes" and isinstance(value, int):
        return f"{value / (1024**2):.1f} MiB" if value >= 1024**2 else f"{value / 1024:.1f} KiB"
    if key == "path" and value:
        return Path(str(value)).name
    if key.endswith("_ns") and isinstance(value, int):
        return (
            datetime.fromtimestamp(value / 1e9, timezone.utc)
            .astimezone()
            .strftime("%d %b %Y · %H:%M")
        )
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key == "repetition" and isinstance(value, int):
        return str(value + 1)
    if key in {"state", "phase", "outcome", "report_state", "kind"}:
        return humanize(str(value))
    if key in {"id", "experiment_id", "attempt_id", "campaign_id"}:
        return str(value)[:8]
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".") if math.isfinite(value) else "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
