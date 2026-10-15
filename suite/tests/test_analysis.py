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

from dataclasses import replace
import math
import unittest

from benchmark_suite.analysis import (
    DetectorSettings,
    MotionDetector,
    coverage_plateaus,
    tail_durations,
    with_navigation,
    navigation_no_progress,
)
from benchmark_suite.contracts import ContractError


def observation(t, **fields):
    return dict(
        source="raw_odometry",
        valid=True,
        simulation_elapsed_ns=round(t * 1e9),
        clock_ns=round((t + 5) * 1e9),
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed=0.0,
        wz=0.0,
        command_vx=0.0,
        command_wz=0.0,
        **fields,
    )


class AnalysisTests(unittest.TestCase):
    def test_lateral_commands_and_invalid_motion_have_explicit_support(self):
        detector = MotionDetector(0.8, [], DetectorSettings())
        incidents = []
        for index in range(32):
            row = observation(index * 0.2)
            row["command_vy"] = 0.3
            incidents.extend(detector.feed(row))
        incidents.extend(detector.close_segments())
        self.assertIn("command_motion_discrepancy", [item["type"] for item in incidents])
        invalid = observation(6.4)
        invalid["valid"] = False
        detector.feed(invalid)
        detector.feed(observation(6.6))
        self.assertEqual(detector.gaps, 1)
        self.assertAlmostEqual(sum(detector.durations.values()), 6.2)

    def test_navigation_no_progress_keeps_uuid_and_breaks_feedback_gaps(self):
        rows = []
        for index in range(61):
            row = observation(index * 0.2)
            row.update(event="feedback", native_uuid="native-1", distance_remaining=4.0)
            rows.append(row)
        incidents = list(navigation_no_progress(rows, DetectorSettings()))
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]["duration_ns"], 12_000_000_000)
        self.assertIn('"goal_sequence_association": "unavailable"', incidents[0]["evidence_json"])
        rows[30]["valid"] = False
        self.assertFalse(list(navigation_no_progress(rows, DetectorSettings())))

    def test_stationary_and_command_discrepancy_have_observed_intervals(self):
        detector = MotionDetector(0.8, [], DetectorSettings())
        incidents = []
        for index in range(31):
            row = observation(index * 0.2)
            row["command_vx"] = 0.2
            incidents.extend(detector.feed(row))
        row = observation(6.2)
        row.update(speed=0.4, x=0.1)
        incidents.extend(detector.feed(row))
        self.assertEqual(
            {item["type"] for item in incidents}, {"stationary", "command_motion_discrepancy"}
        )
        self.assertEqual(incidents[0]["duration_ns"], 6_200_000_000)
        self.assertAlmostEqual(detector.durations["stationary"], 6.2)

    def test_missing_and_nonmonotonic_motion_break_intervals(self):
        detector = MotionDetector(0.8, [], DetectorSettings())
        incidents = []
        for time in (0, 0.2, 0.4, 10, 10.2, 10.1):
            incidents.extend(detector.feed(observation(time)))
        incidents.extend(detector.close_segments())
        self.assertFalse(incidents)
        self.assertEqual(detector.gaps, 2)
        self.assertAlmostEqual(sum(detector.durations.values()), 0.6)

    def test_slow_motion_requires_observed_active_navigation(self):
        detector = MotionDetector(0.8, [], DetectorSettings())
        motion = []
        for index in range(61):
            row = observation(index * 0.2)
            row.update(x=index * 0.02, speed=0.1)
            motion.append(row)
        events = [
            dict(event="initialized", goal_sequence=4, clock_ns=7_000_000_000),
            dict(event="terminal", clock_ns=15_000_000_000),
        ]
        incidents = [
            incident for row in with_navigation(motion, events) for incident in detector.feed(row)
        ]
        self.assertEqual([item["type"] for item in incidents], ["slow_motion"])
        self.assertEqual(incidents[0]["start_ns"], 2_000_000_000)
        self.assertEqual(incidents[0]["end_ns"], 10_000_000_000)

    def test_closed_loop_requires_time_distance_and_spatial_return(self):
        detector = MotionDetector(0.8, [[0.2, 0.2]], DetectorSettings())
        incidents = []
        for index in range(101):
            t = index * 0.2
            row = observation(t)
            row.update(
                x=math.cos(t * math.pi / 10),
                y=math.sin(t * math.pi / 10),
                speed=math.pi / 10,
                wz=math.pi / 10,
            )
            incidents.extend(detector.feed(row))
        loops = [item for item in incidents if item["type"] == "closed_loop"]
        self.assertEqual(len(loops), 1)
        self.assertGreater(loops[0]["distance_m"], 5)
        self.assertGreaterEqual(loops[0]["duration_ns"], 15_000_000_000)

    def test_fast_coverage_then_long_tail_and_stale_gap(self):
        samples = []
        for second in range(71):
            row = observation(second)
            row.update(status="valid", coverage_percent=90 + second / 1000)
            samples.append(row)
        intervals = list(coverage_plateaus(samples, DetectorSettings()))
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["duration_ns"], 70_000_000_000)
        samples[15]["status"] = "stale"
        samples[45]["status"] = "stale"
        self.assertFalse(list(coverage_plateaus(samples, DetectorSettings())))
        milestones = [
            dict(percent=90, reached=True, simulation_elapsed=5),
            dict(percent=95, reached=True, simulation_elapsed=65),
        ]
        self.assertEqual(
            tail_durations(milestones, 75), {"90_to_95_seconds": 60, "90_to_completion_seconds": 70}
        )
        self.assertIsNone(tail_durations(milestones, None)["90_to_completion_seconds"])

    def test_unreached_and_unavailable_values_are_not_zero(self):
        self.assertEqual(
            tail_durations([], None), {"90_to_95_seconds": None, "90_to_completion_seconds": None}
        )
        detector = MotionDetector(0.8, [], DetectorSettings())
        row = observation(0)
        row["simulation_elapsed_ns"] = None
        self.assertFalse(detector.feed(row))
        self.assertEqual(detector.valid_samples, 0)
        for value in (
            False,
            {"interval_seconds": float("inf")},
            {"slow_limit_fraction": 2},
            {"unknown": 5},
        ):
            with self.assertRaises(ContractError):
                DetectorSettings.accept(value)
        self.assertEqual(
            DetectorSettings.accept({"interval_seconds": 6}),
            replace(DetectorSettings(), interval_seconds=6),
        )
