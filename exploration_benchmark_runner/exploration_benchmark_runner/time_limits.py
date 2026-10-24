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

"""Time-limit rule ``kinematic-time-limit-v1`` (docs/MEASUREMENTS.md, "Time limits
and outcomes").

L = min(14400, max(600, ceil(30 * A / (2 * R * v)))) simulation seconds, with A the
target area of the world's coverage reference (target cells times resolution
squared), R the robot's catalog lidar range and v its catalog top linear speed, both
from the Default physical settings. The quotient is computed exactly from the
decimal values (a float is read as its shortest decimal form), so floating-point
noise in a stored area cannot add a second. Standard library only: the suite's
tools load this module without ROS.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

TIME_LIMIT_RULE = "kinematic-time-limit-v1"
# The rule takes A from a world's metric-version-2 coverage reference (I4 §3.1); a
# world without one has no rule values. The suite's catalog, campaign check and tool
# read it.
RULE_METRIC_VERSION = 2
FACTOR = 30
FLOOR_S = 600
CAP_S = 14400


@dataclass(frozen=True, slots=True)
class TimeLimit:
    """A rule value: ``seconds`` is the limit, ``uncapped_seconds`` the value
    without the cap (floor applied), ``capped`` whether the cap binds."""

    seconds: int
    uncapped_seconds: int
    capped: bool


def exact(value, field):
    """The exact value of a positive finite int or float; a float is read as its
    shortest round-trip decimal (0.05 -> 1/20). Errors name ``field``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number, not {type(value).__name__}")
    try:
        if isinstance(value, int):
            result = Fraction(int(value))
            finite = True
        else:
            number = float(value)
            finite = math.isfinite(number)
            result = Fraction(float.__repr__(number)) if finite else None
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} is not a usable number: {exc}") from exc
    if not finite or result <= 0:
        raise ValueError(f"{field} must be positive and finite, got {value!r}")
    return result


def target_area_m2(cells, resolution):
    """The rule's A in m², exactly (a Fraction): target cells x resolution squared, the
    resolution read as its decimal value (``exact``). The rule, the suite's catalog and
    campaign check, and its tool all compute A here."""
    if isinstance(cells, bool) or not isinstance(cells, int):
        raise ValueError(f"target_cells must be an integer, not {type(cells).__name__}")
    return exact(cells, "target_cells") * exact(resolution, "resolution_m") ** 2


def kinematic_time_limit(target_cells, resolution_m, range_m, speed_mps):
    """Return the ``kinematic-time-limit-v1`` limit for one world and robot."""
    area = target_area_m2(target_cells, resolution_m)
    lidar_range = exact(range_m, "range_m")
    speed = exact(speed_mps, "speed_mps")
    quotient = FACTOR * area / (2 * lidar_range * speed)
    uncapped = max(FLOOR_S, math.ceil(quotient))
    return TimeLimit(min(CAP_S, uncapped), uncapped, uncapped > CAP_S)
