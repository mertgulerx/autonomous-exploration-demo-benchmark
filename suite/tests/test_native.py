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

from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from benchmark_suite.native import NativeWorker
from benchmark_suite.storage import atomic_json


class NativeTransportTests(unittest.TestCase):
    def test_thread_start_failure_closes_owned_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            scope, lease = MagicMock(), MagicMock()
            scope.start.return_value.pid = 123
            with (
                patch("benchmark_suite.native.ProcessScope", return_value=scope),
                patch("benchmark_suite.native.DomainLease", return_value=lease),
                patch("benchmark_suite.native.psutil.Process"),
                patch(
                    "benchmark_suite.native.threading.Thread.start",
                    side_effect=RuntimeError("No thread"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "No thread"):
                    NativeWorker(
                        {
                            "id": "attempt",
                            "directory": directory,
                            "token": "owner",
                            "domain": 61,
                            "partition": "worker",
                        },
                        {"suite_socket": directory + "/control.sock", "project_root": directory, "robot": {"family": "mogi"}},
                    )
            scope.stop.assert_called_once()
            lease.close.assert_called_once()

    def worker(self, directory, manifest):
        worker = NativeWorker.__new__(NativeWorker)
        worker.directory = Path(directory)
        worker.configuration = {"startup_timeout": 180}
        worker.events = queue.Queue(128)
        worker.lock = threading.Lock()
        worker.inspection_request = worker.stop_requested = worker.stop_wall = None
        worker.started_wall = time.monotonic()
        worker.latest_inspection = {"sequence": 0, "path": [], "map": None}
        worker.sequence = -1
        worker.last_state = ""
        worker.last_metrics_wall = worker.trigger = None
        worker.terminal = worker.closed = False
        polls = iter([None, 0])
        worker.process = SimpleNamespace(
            poll=lambda: next(polls), wait=lambda **kwargs: 0, returncode=0
        )
        worker.scope = SimpleNamespace(stop=lambda: None)
        worker.lease = SimpleNamespace(close=lambda: None)
        worker.client = SimpleNamespace(
            close=lambda: None,
            call=lambda *args: {
                "sequence": 1,
                "state": {
                    "phase": "STOPPING",
                    "outcome": "runtime_failure",
                    "detail": "Lost stream",
                },
                "metrics": {"wall_monotonic": 1.0, "frozen": False},
            },
        )
        atomic_json(worker.directory / "manifest.json", manifest)
        return worker

    def test_committed_metrics_replace_trigger_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(
                directory,
                {
                    "phase": "FINISHED",
                    "outcome": "runtime_failure",
                    "final_metrics": {"frozen": True, "quality_flags": ["source_gap"]},
                },
            )
            worker._transport()
            terminals = [event for event in worker.poll() if event["kind"] == "terminal"]
            self.assertEqual(
                terminals[-1]["metrics"], {"frozen": True, "quality_flags": ["source_gap"]}
            )
            self.assertTrue(worker.closed)

    def test_uncommitted_failure_keeps_worker_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(directory, {"phase": "FINALIZING", "outcome": "runtime_failure"})
            worker._transport()
            self.assertEqual(worker.poll()[-1]["outcome"], "runtime_failure")

    def test_startup_deadline_and_inspection_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = self.worker(directory, {"phase": "FINISHED", "outcome": "startup_failure"})
            worker.started_wall -= 300
            worker._transport()
            self.assertEqual(worker.stop_requested[0], "startup_failure")
            worker.latest_inspection = {
                "sequence": 5,
                "path": [[1, 2], [3, 4], [5, 6]],
                "map_revision": 7,
                "map": {"image_png": "data"},
            }
            result = worker.inspection(4, 7)
            self.assertEqual(result["path"], [[5, 6]])
            self.assertIsNone(result["map"])


if __name__ == "__main__":
    unittest.main()
