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

from typing import Any
import yaml

from .contracts import ContractError, mapping


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader: UniqueLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)  # type: ignore[no-untyped-call]
        if not isinstance(key, str) or key in result:
            raise ContractError("yaml", "YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=True)  # type: ignore[no-untyped-call]
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_mapping(source: str) -> dict[str, Any]:
    if len(source.encode()) > 4 * 1024 * 1024:
        raise ContractError("yaml", "YAML exceeds 4 MiB")
    try:
        return mapping(yaml.load(source, Loader=UniqueLoader), "yaml")
    except yaml.YAMLError as exc:
        raise ContractError("yaml", str(exc)) from exc
