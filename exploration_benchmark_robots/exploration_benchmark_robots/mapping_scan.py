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

"""Mapping-only scan for SLAM Toolbox (mapping-known-pose-v1, K4 and K5).

Every reading at or above the message's ``range_max`` (no-return beams arrive as
+inf) becomes ``V = range_max - FILL_OFFSET_M``; NaN stays NaN and every other
reading, -inf included, is unchanged. SLAM Toolbox's ``max_laser_range`` is
``range_max - THRESHOLD_MARGIN_M``, below V, so Karto traces such a beam as free
space and marks no obstacle (V >= threshold, V < range_max); the patched SLAM
Toolbox limits that free ray to known_pose.NO_RETURN_FREE_LENGTH_M (5 m,
mapping-known-pose-v4). A real return in [threshold, range_max) is traced the
same way (free, no obstacle) and is counted. Nav2, the explorer and the recorder keep the
real /scan.

A message whose ``range_max`` does not sit exactly ``THRESHOLD_MARGIN_M`` above
the configured threshold (``consistent``) is withheld and counted; one withheld
scan makes the node not ok for the rest of the run, and the worker ends a
running run on it (runtime_failure).

laser_filters' LaserScanRangeFilter is not used: it also replaces every reading
at or below its lower threshold, and its thresholds and replacement values are
fixed parameters rather than the message's own ``range_max``.
"""

from array import array
import json

import numpy as np

MAPPING_SCAN_TOPIC = "/benchmark/scan_mapping"
# Condition-defining constants of known_pose.MAPPING_CONDITION: changing one requires
# a new MAPPING_CONDITION id (recorded in comparison_profile.mapping_parameters).
# V = range_max - FILL_OFFSET_M; SLAM max_laser_range = range_max - THRESHOLD_MARGIN_M.
FILL_OFFSET_M = 0.01
THRESHOLD_MARGIN_M = 0.02
# Tolerance of the margin check: message range_max is float32 (ulp 3.8e-6 m at 40 m).
MARGIN_TOLERANCE_M = 1e-4
COUNTERS = ("readings", "filled_inf", "filled_finite", "free_marked_returns", "nan")


def fill_no_returns(ranges, range_max, offset=FILL_OFFSET_M):
    """Return (filled float32 ranges, number of replaced readings)."""
    values = np.array(ranges, dtype=np.float32)
    replace = values >= np.float32(range_max)  # NaN compares False, -inf too
    values[replace] = np.float32(range_max - offset)
    return values, int(np.count_nonzero(replace))


def count_readings(ranges, range_max, threshold):
    """Per-scan counters (``COUNTERS``) of the float32 readings before filling.

    filled_inf: +inf; filled_finite: finite readings >= range_max; free_marked_returns:
    real returns in [threshold, range_max), which Karto traces as free with no obstacle;
    nan: NaN readings (ignored by Karto).
    """
    # Karto compares the float32 readings as doubles with the message's range_max.
    values = np.asarray(ranges, dtype=np.float32).astype(np.float64)
    top = float(np.float32(range_max))
    finite = np.isfinite(values)
    return dict(
        readings=int(values.size),
        filled_inf=int(np.count_nonzero(np.isposinf(values))),
        filled_finite=int(np.count_nonzero(finite & (values >= top))),
        free_marked_returns=int(
            np.count_nonzero(finite & (values >= threshold) & (values < top))
        ),
        nan=int(np.count_nonzero(np.isnan(values))),
    )


def same_limits(a, b):
    """Equal scan limits, NaN equal to NaN (so a NaN limit is reported once)."""
    if a is None or b is None:
        return a is b
    return all(x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(a, b))


def consistent(range_min, range_max, threshold, offset=FILL_OFFSET_M):
    """Whether a message's limits fit the configured threshold.

    Two-sided: range_max - threshold must equal THRESHOLD_MARGIN_M (within
    MARGIN_TOLERANCE_M), and Karto must clear a filled beam (threshold <= V <
    range_max, V > range_min).
    """
    if not (np.isfinite(range_max) and np.isfinite(range_min)):
        return False
    value = float(np.float32(range_max - offset))
    return bool(
        abs((range_max - threshold) - THRESHOLD_MARGIN_M) <= MARGIN_TOLERANCE_M
        and threshold <= value < range_max
        and value > range_min
    )


def main(args=None):
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String

    from .known_pose import MAPPING_STATUS_TOPIC, STATUS_PERIOD_NS, run_node

    latched = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    class MappingScan(Node):
        def __init__(self):
            super().__init__("mapping_scan_filter")
            self.threshold = self.declare_parameter("max_laser_range", 0.0).value
            self.offset = self.declare_parameter("fill_offset", FILL_OFFSET_M).value
            source = self.declare_parameter("input_topic", "/scan").value
            target = self.declare_parameter("output_topic", MAPPING_SCAN_TOPIC).value
            if not self.threshold > 0 or not 0 < self.offset < THRESHOLD_MARGIN_M:
                raise ValueError(
                    "mapping_scan_filter needs max_laser_range, fill_offset"
                )
            self.scans = self.rejected = 0
            self.counts = dict.fromkeys(COUNTERS, 0)
            self.limits = None
            self.accepted = False
            self.last_status = None
            self.last_status_ns = None
            self.status = self.create_publisher(String, MAPPING_STATUS_TOPIC, latched)
            # Reliable output matches both reliable and best-effort subscribers.
            self.output = self.create_publisher(LaserScan, target, 10)
            self.create_subscription(
                LaserScan, source, self.receive, qos_profile_sensor_data
            )
            self.detail = "waiting for scans"
            self.report(0, force=True)

        def receive(self, message):
            stamp = message.header.stamp
            stamp_ns = stamp.sec * 10**9 + stamp.nanosec
            limits = (message.range_min, message.range_max)
            if not same_limits(limits, self.limits):
                self.limits = limits
                self.accepted = consistent(*limits, self.threshold, self.offset)
                if not self.accepted:
                    self.get_logger().error(
                        f"Scan range [{limits[0]}, {limits[1]}] m does not fit "
                        f"max_laser_range {self.threshold} m (range_max - "
                        f"{THRESHOLD_MARGIN_M} m); mapping scans are withheld"
                    )
            if not self.accepted:
                self.rejected += 1
                self.detail = "scan range_max inconsistent with max_laser_range"
                self.report(stamp_ns)
                return
            for name, value in count_readings(
                message.ranges, message.range_max, self.threshold
            ).items():
                self.counts[name] += value
            ranges, _ = fill_no_returns(message.ranges, message.range_max, self.offset)
            message.ranges = array("f", ranges.tobytes())
            self.output.publish(message)
            self.scans += 1
            if self.rejected == 0:
                self.detail = "publishing"
            self.report(stamp_ns)

        def snapshot(self):
            return dict(
                node="mapping_scan_filter",
                ok=self.scans > 0 and self.rejected == 0,
                detail=self.detail,
                scans=self.scans,
                rejected_scans=self.rejected,
                **self.counts,
                range_limits=self.limits,
                max_laser_range=self.threshold,
            )

        def report(self, stamp_ns, force=False):
            status = self.snapshot()
            state = (status["ok"], self.detail)
            if (
                not force
                and state == self.last_status
                and self.last_status_ns is not None
                and 0 <= stamp_ns - self.last_status_ns < STATUS_PERIOD_NS
            ):
                return
            self.last_status, self.last_status_ns = state, stamp_ns
            self.status.publish(String(data=json.dumps(status)))

        def final_report(self):
            status = dict(self.snapshot(), final=True)
            self.get_logger().info("Final status: " + json.dumps(status))
            self.status.publish(String(data=json.dumps(status)))

    run_node(MappingScan, args)


if __name__ == "__main__":
    main()
