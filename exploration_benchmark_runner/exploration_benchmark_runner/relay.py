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

"""Presentation-only copies. Source timestamps, URDF, and worker inputs stay intact."""

from copy import deepcopy


def prefix_frames(message, prefix):
    result = deepcopy(message)

    def visit(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, "get_fields_and_field_types"):
            for field in value.get_fields_and_field_types():
                child = getattr(value, field)
                if field in ("frame_id", "child_frame_id") and isinstance(child, str):
                    if child:
                        setattr(value, field, prefix + "/" + child.lstrip("/"))
                else:
                    visit(child)

    visit(result)
    return result
