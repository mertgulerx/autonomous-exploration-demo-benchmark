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

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import pyarrow.parquet as pq

from benchmark_suite.recording import Recorder
from benchmark_suite.storage import file_hash


class RecordingTests(unittest.TestCase):
    def test_checkpoint_types_freezing_and_lossless_map(self):
        with tempfile.TemporaryDirectory() as directory:
            configuration = dict(
                suite_workspace=directory,
                suite_identity=dict(
                    campaign_id="campaign",
                    experiment_id="experiment",
                    attempt_id="attempt",
                    robot_id="mogi",
                    world_id="warehouse",
                    profile_id="profile",
                    spawn_id="spawn",
                    repetition=1,
                ),
                robot={"id": "mogi"},
                requested_parameters={},
                effective_parameters={},
            )
            reference = NS(
                metadata={"metric_version": 1},
                metadata_bytes=b"{}",
                grid_bytes=b"grid",
                descriptor=lambda: {"id": "reference"},
            )
            recorder = Recorder(directory, configuration, reference)
            recorder.clock = 2.0
            before = recorder.row("raw_odometry", 2_000_000_000)
            self.assertIsNone(before["simulation_elapsed_ns"])
            recorder.origins = 2.0, 3.0
            recorder.clock = 4.25
            row = recorder.row("raw_odometry", 4_250_000_000, receipt_monotonic_ns=5_000_000_000)
            self.assertEqual(row["simulation_elapsed_ns"], 2_250_000_000)
            self.assertEqual(row["wall_elapsed_ns"], 2_000_000_000)
            stale = recorder.row(
                "delayed_sensor",
                1_500_000_000,
                clock_ns=1_500_000_000,
                receipt_monotonic_ns=2_000_000_000,
            )
            self.assertIsNone(stale["simulation_elapsed_ns"])
            self.assertIsNone(stale["wall_elapsed_ns"])
            recorder.observation("trajectory", dict(row, x=1.0, y=2.0, speed=0.5))
            grid = NS(
                header=NS(stamp=NS(sec=4, nanosec=250_000_000), frame_id="map"),
                info=NS(
                    width=2,
                    height=2,
                    resolution=0.05,
                    origin=NS(
                        position=NS(x=-1.0, y=-2.0, z=0.0),
                        orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                ),
                data=[-1, 0, 50, 100],
            )
            recorder.manifest["terminal_trigger_utc_ns"] = 123
            recorder.finish(
                "time_limit",
                "Budget reached",
                {"motion_started": True, "coverage_status": "valid"},
                grid,
                None,
                -1,
                [],
            )
            recorder.observation("trajectory", dict(row, x=99.0))
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["phase"], "FINISHED")
            self.assertEqual(manifest["terminal_utc_ns"], 123)
            self.assertEqual(manifest["schema_version"], 5)
            segments = json.loads((Path(directory) / "segments.json").read_text())["segments"]
            trajectory = next(value for value in segments if value["kind"] == "trajectory")
            path = Path(directory) / trajectory["path"]
            self.assertEqual(file_hash(path), trajectory["sha256"])
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 1)
            self.assertEqual(table["simulation_elapsed_ns"][0].as_py(), 2_250_000_000)
            with np.load(Path(directory) / "final_map.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["occupancy"], np.array([[-1, 0], [50, 100]]))
                self.assertEqual(json.loads(str(saved["metadata"]))["frame"], "map")
            self.assertFalse(recorder.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
