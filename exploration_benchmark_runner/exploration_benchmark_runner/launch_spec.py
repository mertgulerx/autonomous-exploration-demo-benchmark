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

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True, slots=True)
class WorkerLaunchSpec:
    configuration_json: str

    @classmethod
    def from_configuration(cls, configuration):
        required = {"slot", "run_id", "domain_id", "observer_domain", "gazebo_partition", "process_token",
                    "result_directory", "project_root", "session_root", "scenario", "single",
                    "spec", "profile", "seed", "simulation_time_limit", "startup_timeout", "robot"}
        missing = required - configuration.keys()
        if missing:
            raise ValueError("Worker launch specification missing: " + ", ".join(sorted(missing)))
        domain = configuration["domain_id"]
        if type(domain) is not int or not 0 <= domain <= 101:
            raise ValueError("Worker ROS domain must be an integer from 0 to 101")
        observer = configuration["observer_domain"]
        if type(observer) is not int or not 0 <= observer <= 101 or observer == domain:
            raise ValueError("Observer ROS domain must be valid and distinct from the worker")
        if type(configuration["seed"]) is not int or configuration["seed"] != 1234:
            raise ValueError("Gazebo seed is fixed at 1234")
        for key in ("simulation_time_limit", "startup_timeout"):
            value = configuration[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(key + " must be finite and nonnegative")
        for key in ("run_id", "gazebo_partition", "process_token"):
            if not isinstance(configuration[key], str) or not configuration[key]:
                raise ValueError(key + " must be nonempty")
        if type(configuration["single"]) is not bool:
            raise ValueError("Worker single mode must be boolean")
        return cls(json.dumps(configuration, sort_keys=True, allow_nan=False))

    def dictionary(self):
        return json.loads(self.configuration_json)
