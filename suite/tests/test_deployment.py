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

import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from benchmark_suite.contracts import ContractError
from benchmark_suite.deployment import docker, gpu_arguments


class DeploymentTests(unittest.TestCase):
    def test_nvidia_cdi_devices_are_requested_without_legacy_hook(self):
        info = {"Runtimes": {"nvidia": {}}, "CDISpecDirs": ["/var/run/cdi"]}
        discovery = subprocess.CompletedProcess([], 0, "nvidia.com/gpu=all\n", "")
        with (
            patch("benchmark_suite.deployment.shutil.which", return_value="/usr/bin/tool"),
            patch("benchmark_suite.deployment.subprocess.run", return_value=discovery),
        ):
            mode, args = gpu_arguments(info)
        self.assertEqual(mode, "cdi")
        self.assertIn("nvidia.com/gpu=all", args)
        self.assertNotIn("--gpus", args)

    def test_legacy_nvidia_access_explicitly_requests_device_nodes(self):
        with patch("benchmark_suite.deployment.shutil.which", return_value="/usr/bin/tool"):
            mode, args = gpu_arguments({"Runtimes": {"nvidia": {}}})
        self.assertEqual(mode, "explicit")
        self.assertIn("--runtime", args)
        self.assertIn("--gpus", args)
        self.assertEqual(gpu_arguments({"Runtimes": {}}), ("none", []))

    def stalled_command(self, **options):
        launch = subprocess.Popen
        processes = []

        def stalled(command, **kwargs):
            process = launch([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
            processes.append(process)
            return process

        started = time.monotonic()
        with patch("benchmark_suite.deployment.subprocess.Popen", side_effect=stalled):
            try:
                with self.assertRaises(ContractError) as error:
                    docker("image", "inspect", "unused", **options)
                self.assertLess(time.monotonic() - started, 2)
                self.assertIsNotNone(processes[0].poll())
                return error.exception
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()

    def test_workspace_cancellation_closes_owned_stalled_command(self):
        deadline = time.monotonic() + 0.15
        error = self.stalled_command(cancelled=lambda: time.monotonic() > deadline)
        self.assertEqual(error.code, "operation_cancelled")

    def test_docker_command_has_bounded_deadline(self):
        error = self.stalled_command(timeout=0.15)
        self.assertIn("timed out", str(error))
