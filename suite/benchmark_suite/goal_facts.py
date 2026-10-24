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

"""Per-goal facts of one run from Nav2 telemetry and feedback (dataset schema 10, stream ``goal_facts``).

The worker feeds every telemetry event and every feedback message at full rate, whether or not feedback
rows are recorded (RESULTS_RESEARCH RD-42), so recoveries are counted either way. Telemetry events carry
the navigator's goal sequence and feedback only the native UUID; a feedback message belongs to the goal
whose ``initialized`` event is the latest received before it and which has not ended (docs/MEASUREMENTS.md,
"Dataset schema 10"). Since dataset schema 11 the worker also writes one ``navigation`` row with
``event = 'recovery'`` for each count a goal's feedback reaches first (``Goal.new_recoveries``), so a goal's
recovery rows add up to its ``recoveries``.
"""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

# Telemetry terminal statuses (exploration_benchmark_telemetry) and the end of a preempted goal.
TERMINAL_STATUSES = frozenset({"succeeded", "canceled", "failed"})
PREEMPTED = "preempted"
OPEN = "open"
OTHER = "other"  # a terminal status outside the telemetry's list (none is known)
# Debug streams keep one feedback row per simulated second per goal, plus its first and last (RO-9).
FEEDBACK_INTERVAL_NS = 1_000_000_000


def end_status(status: str) -> str:
    """The ``goal_facts`` status of a goal ended with telemetry ``status``."""
    return status if status in TERMINAL_STATUSES or status == PREEMPTED else OTHER


@dataclass(slots=True)
class Goal:
    """One goal while it is active, with the debug-stream decimation state of its feedback rows."""

    sequence: int
    dispatch_clock_ns: int | None
    native_uuid: str | None = None
    recoveries: int | None = None
    min_distance_remaining_m: float | None = None
    feedback_messages: int = 0
    status: str = OPEN
    last_kept_clock_ns: int | None = None
    held_row: dict[str, Any] | None = None
    # The recovery counts the latest feedback message reached first (one recovery row each, schema 11);
    # valid only until the goal's next feedback message.
    new_recoveries: range = range(0)

    def keep_feedback(self, clock_ns: int | None, row: dict[str, Any]) -> bool:
        """Whether to record this feedback row now; otherwise it is held as the goal's possible last."""
        if self.last_kept_clock_ns is None or (
            clock_ns is not None and clock_ns - self.last_kept_clock_ns >= FEEDBACK_INTERVAL_NS
        ):
            self.last_kept_clock_ns, self.held_row = clock_ns, None
            return True
        self.held_row = row
        return False

    def fields(self, end_clock_ns: int | None) -> dict[str, Any]:
        """The goal's ``goal_facts`` columns."""
        return dict(
            goal_sequence=self.sequence,
            native_uuid=self.native_uuid,
            dispatch_clock_ns=self.dispatch_clock_ns,
            end_clock_ns=end_clock_ns,
            status=self.status,
            recoveries=self.recoveries,
            min_distance_remaining_m=self.min_distance_remaining_m,
            feedback_messages=self.feedback_messages,
        )


class GoalAccumulator:
    """The goals of one run and their facts; associates feedback by the module's rule."""

    def __init__(self) -> None:
        self.active: dict[int, Goal] = {}
        self.latest: int | None = None
        self.totals: Counter[str] = Counter()
        self.unassociated_feedback = 0

    def started(self, sequence: int, clock_ns: int | None) -> None:
        """An ``initialized`` telemetry event."""
        self.active[sequence] = Goal(sequence, clock_ns)
        self.latest = sequence
        self.totals["sent"] += 1

    def ended(self, sequence: int, status: str) -> Goal | None:
        """A terminal or ``preempted`` event; returns the goal it ends, or None for an unknown goal."""
        goal = self.active.pop(sequence, None)
        if goal is not None:
            goal.status = end_status(status)
            self.totals[goal.status] += 1
        return goal

    def feedback(self, native_uuid: str, distance_remaining: float, recoveries: int) -> Goal | None:
        """One feedback message; returns the goal it belongs to, or None (counted as unassociated).

        Feedback of another native goal than the one the latest goal already holds (a new goal's
        feedback arriving before its telemetry) is never merged into it.
        """
        goal = self.active.get(self.latest) if self.latest is not None else None
        if goal is None or goal.native_uuid not in (None, native_uuid):
            self.unassociated_feedback += 1
            return None
        goal.native_uuid = native_uuid
        goal.feedback_messages += 1
        previous = goal.recoveries or 0
        goal.recoveries = (
            recoveries if goal.recoveries is None else max(goal.recoveries, recoveries)
        )
        goal.new_recoveries = range(previous + 1, goal.recoveries + 1)
        if math.isfinite(distance_remaining):
            goal.min_distance_remaining_m = (
                distance_remaining
                if goal.min_distance_remaining_m is None
                else min(goal.min_distance_remaining_m, distance_remaining)
            )
        return goal

    def close(self) -> list[Goal]:
        """The goals still active at the end of the run, in dispatch order."""
        goals = sorted(self.active.values(), key=lambda goal: goal.sequence)
        self.active.clear()
        self.totals[OPEN] += len(goals)
        return goals

    def facts(self) -> dict[str, Any]:
        """Run totals for the manifest's ``run_facts``."""
        return dict(
            goals=dict(sorted(self.totals.items())),
            unassociated_feedback=self.unassociated_feedback,
        )
