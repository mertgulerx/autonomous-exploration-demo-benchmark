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

"""Pure measurement policy; simulation time never substitutes for wall time."""

from collections import deque
import math


class QualityMonitor:
    def __init__(self, window=5.0, threshold=0.95, persistence=10.0):
        self.window, self.threshold, self.persistence = window, threshold, persistence
        self.samples = deque()
        self.low_since = None
        self.flags = set()
        self.rtf = float("nan")

    def update(self, wall, simulation):
        if self.samples and simulation < self.samples[-1][1]:
            self.flags.add("clock_reset")
            self.samples.clear()
            self.low_since = None
        self.samples.append((wall, simulation))
        while len(self.samples) > 1 and wall - self.samples[1][0] >= self.window:
            self.samples.popleft()
        elapsed = wall - self.samples[0][0]
        if elapsed >= self.window:
            self.rtf = (simulation - self.samples[0][1]) / elapsed
            if self.rtf < self.threshold:
                if self.low_since is None:
                    self.low_since = wall
                if wall - self.low_since >= self.persistence:
                    self.flags.add("low_rtf")
            else:
                self.low_since = None
        return self.rtf


class Measurements:
    MOTION_TRANSLATION_METRES = 0.001
    MOTION_ROTATION_RADIANS = 0.001

    def __init__(self, wall_start, simulation_start, *, wait_for_motion=False):
        self.wall_start, self.simulation_start = wall_start, simulation_start
        self.movement_wall_start = None if wait_for_motion else wall_start
        self.movement_simulation_start = None if wait_for_motion else simulation_start
        self.motion_origin = self.motion_origin_yaw = None
        self.cpu_sum = self.rss_sum = self.samples = 0
        self.previous_cpu = self.previous_cpu_wall = None
        self.cpu_seconds = self.cpu = self.rss = self.distance = 0.0
        self.last_position = None
        self.frozen = False
        self.quality = QualityMonitor()
        self.latest = {}

    @property
    def motion_started(self):
        return self.movement_wall_start is not None

    def odometry(self, x, y, *, yaw=None, wall=None, simulation=None):
        """Distance and first motion use raw odometry, independent of SLAM TF.

        A one-millimetre translation or one-milliradian rotation from the
        stationary origin rejects numerical jitter while also detecting slow
        movement. Discontinuities above the distance jump gate cannot start time.
        The worker seeds the origin from the last stationary warm-up sample.
        """
        if self.frozen or not all(math.isfinite(v) for v in (x, y)):
            return False
        valid_step = True
        if self.last_position is not None:
            step = math.hypot(x - self.last_position[0], y - self.last_position[1])
            valid_step = step <= 2.0
            if valid_step:
                self.distance += step
        self.last_position = (x, y)
        if yaw is not None and not math.isfinite(yaw):
            yaw = None
        if self.motion_started:
            return False
        if self.motion_origin is None or not valid_step:
            self.motion_origin, self.motion_origin_yaw = (x, y), yaw
            return False
        translation = math.hypot(x - self.motion_origin[0], y - self.motion_origin[1])
        rotation = 0.0
        if yaw is not None:
            if self.motion_origin_yaw is None:
                self.motion_origin_yaw = yaw
            rotation = abs(
                math.atan2(
                    math.sin(yaw - self.motion_origin_yaw),
                    math.cos(yaw - self.motion_origin_yaw),
                )
            )
        if (
            translation >= self.MOTION_TRANSLATION_METRES
            or rotation >= self.MOTION_ROTATION_RADIANS
        ):
            if (
                wall is not None
                and simulation is not None
                and all(math.isfinite(v) for v in (wall, simulation))
                and wall >= self.wall_start
                and simulation >= self.simulation_start
            ):
                self.movement_wall_start = wall
                self.movement_simulation_start = simulation
                return True
        return False

    def sample(self, wall, simulation, cpu_seconds=None, rss_mib=None, stream_gaps=()):
        if self.frozen:
            return dict(self.latest)
        if cpu_seconds is not None and rss_mib is not None:
            self.cpu_seconds = max(0.0, cpu_seconds)
            if self.previous_cpu is not None and wall > self.previous_cpu_wall:
                self.cpu = max(
                    0.0,
                    (cpu_seconds - self.previous_cpu)
                    / (wall - self.previous_cpu_wall)
                    * 100,
                )
                self.rss = rss_mib
                self.cpu_sum += self.cpu
                self.rss_sum += self.rss
                self.samples += 1
            self.previous_cpu, self.previous_cpu_wall = cpu_seconds, wall
        self.quality.update(wall, simulation)
        for gap in stream_gaps:
            self.quality.flags.add("stream_gap:" + gap)
        self.latest = dict(
            cpu_percent=self.cpu,
            rss_mib=self.rss,
            cpu_average_percent=self.cpu_sum / self.samples if self.samples else 0.0,
            rss_average_mib=self.rss_sum / self.samples if self.samples else 0.0,
            cpu_seconds=self.cpu_seconds,
            distance_m=self.distance,
            simulation_elapsed=(
                max(0.0, simulation - self.movement_simulation_start)
                if self.motion_started
                else 0.0
            ),
            motion_started=self.motion_started,
            pre_motion_wall_elapsed=max(
                0.0,
                (self.movement_wall_start if self.motion_started else wall)
                - self.wall_start,
            ),
            pre_motion_simulation_elapsed=max(
                0.0,
                (self.movement_simulation_start if self.motion_started else simulation)
                - self.simulation_start,
            ),
            rtf=self.quality.rtf,
            stream_gaps=list(stream_gaps),
            quality_flags=sorted(self.quality.flags),
            frozen=False,
        )
        return dict(self.latest)

    def freeze(self):
        if not self.motion_started:
            self.quality.flags.add("motion_not_observed")
        if not self.samples:
            self.quality.flags.add("resource_samples_missing")
        if not math.isfinite(self.quality.rtf):
            self.quality.flags.add("rtf_unmeasured")
        self.latest["quality_flags"] = sorted(self.quality.flags)
        self.frozen = True
        self.latest["frozen"] = True
        return dict(self.latest)
