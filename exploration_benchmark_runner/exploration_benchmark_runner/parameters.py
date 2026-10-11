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

"""Shared, offline explorer parameter schema, resolution, and ROS conversion.

No explorer node is instantiated for discovery. Values come from the pinned
benchmark baseline; installed metadata describes the supported research inputs.
"""

from copy import deepcopy
import math
from pathlib import Path

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Avoid silently accepting a different value than a researcher intended."""


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "Parameter YAML keys must be unique strings",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def ros_parameters(values):
    from rclpy.parameter import Parameter

    return [
        Parameter(name, value=value).to_parameter_msg()
        for name, value in values.items()
    ]


def parameter_values(messages):
    from rclpy.parameter import Parameter

    values = {}
    for message in messages:
        if message.name in values:
            raise ValueError(f"Duplicate parameter: {message.name}")
        parameter = Parameter.from_parameter_msg(message)
        if parameter.type_ == Parameter.Type.NOT_SET:
            raise ValueError(f"Parameter has no value: {message.name}")
        values[message.name] = parameter.value
    return values


class ParameterSchema:
    def __init__(self, baseline, metadata):
        self.defaults = yaml.safe_load(Path(baseline).read_text())["frontier_explorer"][
            "ros__parameters"
        ]
        self.defaults["use_sim_time"] = True
        self.metadata = yaml.safe_load(Path(metadata).read_text())
        if set(self.metadata) != set(self.defaults):
            raise ValueError(
                "Parameter metadata does not match the Default configuration"
            )
        self.validate(self.defaults)

    def validate(self, values):
        if not isinstance(values, dict):
            raise ValueError("Parameter overrides must be a mapping")
        for name, value in values.items():
            if name not in self.defaults:
                raise ValueError(f"Unknown explorer parameter: {name}")
            default, definition = self.defaults[name], self.metadata[name]
            if type(value) is not type(default):
                raise ValueError(
                    f"{name}: expected {type(default).__name__}, got {type(value).__name__}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name}: value must be finite")
            if type(value) is int and not -(2**63) <= value < 2**63:
                raise ValueError(f"{name}: value must fit a ROS signed 64-bit integer")
            if definition["managed"] and value != default:
                raise ValueError(
                    f"{name}: managed by the benchmark; expected {default!r}"
                )
            if definition.get("choices") and value not in definition["choices"]:
                raise ValueError(
                    f"{name}: choose one of {', '.join(definition['choices'])}"
                )
            if "minimum" in definition and value < definition["minimum"]:
                raise ValueError(f"{name}: minimum is {definition['minimum']}")
            if (
                "exclusive_minimum" in definition
                and value <= definition["exclusive_minimum"]
            ):
                raise ValueError(
                    f"{name}: must be greater than {definition['exclusive_minimum']}"
                )
            if "maximum" in definition and value > definition["maximum"]:
                raise ValueError(f"{name}: maximum is {definition['maximum']}")

    def import_yaml(self, text):
        try:
            data = yaml.load(text, Loader=UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid parameter YAML: {exc}") from exc
        if not isinstance(data, dict) or not data:
            raise ValueError("Expected a nonempty ROS parameter mapping")
        values = {}
        for scope, node in data.items():
            if scope not in ("frontier_explorer", "/frontier_explorer", "/**"):
                raise ValueError(f"Unsupported parameter scope: {scope}")
            if not isinstance(node, dict) or set(node) != {"ros__parameters"}:
                raise ValueError(f"{scope}: expected only ros__parameters")
            parameters = node["ros__parameters"]
            if not isinstance(parameters, dict):
                raise ValueError(f"{scope}: ros__parameters must be a mapping")
            if scope == "/**" and set(parameters) - {"use_sim_time"}:
                raise ValueError("Wildcard scope supports only managed use_sim_time")
            if set(values) & set(parameters):
                raise ValueError("Duplicate parameters across YAML scopes")
            values.update(parameters)
        self.validate(values)
        return values

    def resolve(self, text="", overrides=None):
        values = deepcopy(self.defaults)
        if text:
            values.update(self.import_yaml(text))
        if overrides is not None:
            self.validate(overrides)
            values.update(deepcopy(overrides))
        return values

    @staticmethod
    def yaml(values):
        return yaml.safe_dump(
            {"frontier_explorer": {"ros__parameters": values}}, sort_keys=False
        )

    def message(self):
        from exploration_benchmark_interfaces.msg import (
            ParameterCatalog,
            ParameterDefinition,
        )

        definitions = []
        for name, default in self.defaults.items():
            metadata = self.metadata[name]
            minimum = metadata.get("minimum", metadata.get("exclusive_minimum"))
            definitions.append(
                ParameterDefinition(
                    name=name,
                    group=metadata["group"],
                    description=metadata["description"],
                    default_value=ros_parameters({name: default})[0].value,
                    managed=metadata["managed"],
                    choices=metadata.get("choices", []),
                    has_minimum=minimum is not None,
                    minimum=float(minimum or 0),
                    exclusive_minimum="exclusive_minimum" in metadata,
                    has_maximum="maximum" in metadata,
                    maximum=float(metadata.get("maximum", 0)),
                )
            )
        return ParameterCatalog(parameters=definitions)
