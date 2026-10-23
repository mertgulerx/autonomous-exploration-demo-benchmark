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

"""The one error the analysis reports as a refusal (exit code 2) rather than a failure."""

from collections.abc import Mapping, Sequence
from typing import Any


class Refused(Exception):
    """The export cannot be analysed as asked; the message says why.

    A refusal over runs that do not fit together also names the ``column`` whose values differ, the
    ``stratum`` it happened in (None for the whole study) and ``groups``: each value with its runs (attempt,
    campaign, block and profile), which ``cli`` writes to the refusal file (RO-22). ``kept`` is False for a
    refusal about the call rather than the study (an existing output directory, an output inside the export).
    """

    def __init__(
        self,
        message: str,
        *,
        column: str | None = None,
        stratum: str | None = None,
        groups: Sequence[Mapping[str, Any]] = (),
        kept: bool = True,
    ) -> None:
        super().__init__(message)
        self.column, self.stratum, self.groups = column, stratum, list(groups)
        self.kept = kept

    def document(self) -> dict[str, Any]:
        return dict(
            reason=str(self),
            column=self.column,
            stratum=self.stratum,
            groups=[dict(group) for group in self.groups],
        )
