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

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass(slots=True)
class TimingPolicy:
    samples: deque[tuple[float, float]] = field(default_factory=deque)
    low_since: float | None = None
    rtf: float | None = None
    rejected: bool = False
    brief_drops: int = 0
    reset_observed: bool = False

    def update(self, wall: float, simulation: float, *, active: bool = True) -> bool:
        if not active:
            self.samples.clear()
            self.low_since = None
            self.rtf = None
            return self.rejected
        if not math.isfinite(wall) or not math.isfinite(simulation):
            return self.rejected
        if self.samples and wall - self.samples[-1][0] > 5:
            self.samples.clear()
            self.low_since = None
            self.rtf = None
        if self.samples and (wall <= self.samples[-1][0] or simulation < self.samples[-1][1]):
            if simulation < self.samples[-1][1]:
                self.reset_observed = True
            self.samples.clear()
            self.low_since = None
        self.samples.append((wall, simulation))
        while len(self.samples) > 1 and wall - self.samples[1][0] >= 5:
            self.samples.popleft()
        elapsed = wall - self.samples[0][0]
        if elapsed >= 5:
            self.rtf = (simulation - self.samples[0][1]) / elapsed
            if self.rtf < 0.90:
                if self.low_since is None:
                    self.low_since = wall
                    self.brief_drops += 1
                if wall - self.low_since >= 10:
                    self.rejected = True
            else:
                self.low_since = None
        return self.rejected
