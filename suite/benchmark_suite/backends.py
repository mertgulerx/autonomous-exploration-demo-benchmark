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

from collections.abc import Callable
import time
from typing import Any, Protocol


class WorkerHandle(Protocol):
    pid: int | None
    pid_created: float | None
    closed: bool

    def poll(self) -> list[dict[str, Any]]: ...
    def stop(self, reason: str, detail: str) -> None: ...
    def inspection(self, after: int, map_revision: int) -> dict[str, Any]: ...


class Backend(Protocol):
    synthetic: bool

    def preflight(self) -> None: ...
    def start(
        self, attempt: dict[str, Any], experiment: dict[str, Any], campaign: dict[str, Any]
    ) -> WorkerHandle: ...
    def recover(self, attempts: list[dict[str, Any]]) -> None: ...


class SyntheticWorker:
    def __init__(self, outcome: str = "completed", steps: int = 4):
        self.pid: int | None = None
        self.pid_created: float | None = None
        self.closed = False
        self.step = 0
        self.steps, self.outcome = steps, outcome
        self.stop_reason: str | None = None

    def poll(self) -> list[dict[str, Any]]:
        if self.closed:
            return []
        self.step += 1
        if self.stop_reason:
            self.closed = True
            return [{"kind": "terminal", "outcome": self.stop_reason, "detail": "Synthetic stop"}]
        if self.step == 1:
            return [{"kind": "state", "phase": "READY"}]
        if self.step == 2:
            return [{"kind": "state", "phase": "RUNNING"}]
        if self.step >= self.steps + 2:
            self.closed = True
            return [{"kind": "terminal", "outcome": self.outcome, "detail": "Synthetic execution"}]
        return [
            {
                "kind": "metrics",
                "source": "synthetic",
                "wall_monotonic": time.monotonic(),
                "clock": float(self.step),
                "simulation_elapsed": float(self.step - 2),
                "wall_elapsed": None,
                "motion_started": True,
            }
        ]

    def stop(self, reason: str, detail: str) -> None:
        self.stop_reason = reason

    def inspection(self, after: int, map_revision: int) -> dict[str, Any]:
        return {"source": "synthetic", "sequence": self.step, "path": [], "map": None}


class SyntheticBackend:
    synthetic = True

    def __init__(
        self, outcome: Callable[[dict[str, Any], bool], str] | None = None, steps: int = 4
    ):
        self.outcome = outcome or (lambda experiment, retry: "completed")
        self.steps = steps

    def preflight(self) -> None:
        return

    def start(
        self, attempt: dict[str, Any], experiment: dict[str, Any], campaign: dict[str, Any]
    ) -> WorkerHandle:
        return SyntheticWorker(self.outcome(experiment, bool(attempt["retry"])), self.steps)

    def recover(self, attempts: list[dict[str, Any]]) -> None:
        return
