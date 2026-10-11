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

"""Launch the sole explorer from a validated, immutable parameter snapshot."""

from copy import deepcopy
from pathlib import Path
import yaml


def resolve_parameters(spec, directory):
    requested = {"frontier_explorer": {"ros__parameters": deepcopy(spec.overrides)}}
    Path(directory, "requested_parameters.yaml").write_text(
        yaml.safe_dump(requested, sort_keys=False)
    )
    integration = {
        "use_sim_time": True,
        "completion_event_enabled": True,
        "completion_event_topic": "exploration_complete",
        "autostart": True,
    }
    parameters = deepcopy(requested)
    parameters["frontier_explorer"]["ros__parameters"].update(integration)
    path = Path(directory) / "resolved_parameters.yaml"
    path.write_text(yaml.safe_dump(parameters, sort_keys=False))
    return str(path), integration


def command(profile, parameters):
    return [
        "ros2",
        "launch",
        profile["package"],
        profile["launch"],
        "use_sim_time:=true",
        f"params_file:={parameters}",
    ]
