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

"""Statistics, tables and figures of one frozen study export (RESULTS_OVERHAUL R5, §7; RESULTS_RESEARCH
RD-22 to RD-25).

The package reads only the export: its tables, ``study.yaml`` (with the plan's ``analysis`` block) and
``SHA256SUMS``. Every output is written once, with a JSON sidecar naming the code, environment, inputs, seeds,
settings and the attempts behind it. Methods: docs/research/METHODOLOGY.md; figures: FIGURE_STYLE.md.
"""

__version__ = "1.4.0"
