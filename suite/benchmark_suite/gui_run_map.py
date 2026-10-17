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

"""Results run map: a finished run's coverage and path map, drawn with the Monitor's map widget."""

from typing import Any
import uuid

from PySide6 import QtCore, QtWidgets

from .gui_monitor import LiveMap
from .gui_style import label
from .gui_support import Transport, stale_engine_message
from .monitoring import PathCursor, accept_page

CAPTION = (
    "Line: Gazebo ground-truth path · Shaded cells: explored in the final map (coverage metric)"
)
IDLE_TEXT = "Select a run to see its coverage and path map"


class RunMap(QtWidgets.QWidget):
    """Map of the selected run from ``results.map``; one request in flight, stale replies re-requested."""

    def __init__(self, transport: Transport) -> None:
        super().__init__()
        self.transport = transport
        self.attempt_id: str | None = None
        self.in_flight = ""
        self.requested: str | None = None  # attempt of the request in flight
        self.path_cursor = PathCursor()
        self.first: dict[str, Any] = {}
        self.unsupported = False
        self.has_base = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        caption = label(CAPTION, "muted")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        self.map = LiveMap()
        self.map.setMinimumHeight(220)
        layout.addWidget(self.map, 1)
        self.status = label(IDLE_TEXT, "muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.retry = QtCore.QTimer(self)
        self.retry.setSingleShot(True)
        self.retry.timeout.connect(self.request)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    def show_attempt(self, identity: str | None, empty_text: str = IDLE_TEXT) -> None:
        if identity == self.attempt_id and identity is not None:
            return
        self.clear()
        self.attempt_id = identity
        if identity is None:
            self.status.setText(empty_text)
        elif not self.unsupported:
            self.status.setText("Loading the run map…")
            self.request()

    def clear(self) -> None:
        self.retry.stop()
        self.attempt_id = None
        self.path_cursor, self.first, self.has_base = PathCursor(), {}, False
        self.map.clear_view()
        if not self.unsupported:
            self.status.setText(IDLE_TEXT)

    def request(self) -> None:
        if self.unsupported or not self.attempt_id or self.in_flight:
            return
        self.in_flight = "run-map:" + uuid.uuid4().hex
        self.requested = self.attempt_id
        self.transport.submit(
            self.in_flight,
            "results.map",
            {"attempt_id": self.attempt_id, "after_sequence": self.path_cursor.applied},
        )

    def received(self, tag: str, result: Any) -> None:
        if tag != self.in_flight:
            return
        self.in_flight = ""
        if result.get("attempt_id") != self.attempt_id:
            self.request()  # the selection changed while this request was out
            return
        state = result.get("state")
        if state == "building":
            self.status.setText("Building the run map…")
            self.retry.start(250)
        elif state == "running":
            # A finished run shows its map without reselecting it.
            self.status.setText(result.get("reason", "Run in progress"))
            self.retry.start(2000)
        elif state in ("failed", "unavailable"):
            self.status.setText("Run map unavailable: " + str(result.get("reason", "")))
        elif state == "ready":
            try:
                self.apply(result)
            except ValueError as exc:
                self.status.setText(f"Run map unreadable: {exc}")
                return
            if self.path_cursor.applied < result["last_sequence"]:
                self.request()

    def apply(self, result: dict[str, Any]) -> None:
        if result["after"] == 0:
            self.first = result
            if result.get("base"):
                self.map.set_reference(result["base"])
                self.has_base = True
            self.map.set_explored(result.get("explored"))
        reset, vertices = accept_page(self.path_cursor, result)
        if reset:
            self.map.clear_path()
        if vertices:
            self.map.append_vertices(vertices)
        if self.path_cursor.applied >= result["last_sequence"]:
            self.map.set_tail(result.get("tail"))
            if not self.has_base and self.map.real_vertices:
                # Changed reference: no base map, so fit the view to the path alone.
                self.map.getViewBox().autoRange(padding=0.04)
            self.status.setText(self.summary(result))

    def summary(self, result: dict[str, Any]) -> str:
        parts = [
            f"{self.map.real_vertices:,} path vertices"
            if result.get("path_available")
            else "No ground-truth path"
        ]
        explored = self.first.get("explored")
        if explored:
            parts.append(f"explored {explored['share_percent']:.2f} % of reference cells")
        breaks = [f"{reason} {count}" for reason, count in result["breaks"].items() if count]
        if breaks:
            parts.append("breaks: " + ", ".join(breaks))
        if result["collision_samples"]:
            parts.append(f"{result['collision_samples']} samples in obstacle cells")
        if result["outside_samples"]:
            parts.append(f"{result['outside_samples']} samples outside the reference")
        dropped = result["dropped"] + self.path_cursor.dropped
        if dropped:
            parts.append(f"{dropped:,} vertices dropped")
        notes = list(result.get("notes") or [])
        return " · ".join(parts) + ("\n" + "\n".join(notes) if notes else "")

    def failed(self, tag: str, error: str) -> None:
        if tag != self.in_flight:
            return
        self.in_flight = ""
        if self.requested != self.attempt_id:
            self.request()  # the failure belongs to a run that is no longer selected
            return
        stale = stale_engine_message(
            error, "Unsupported operation: results.map", "the Results run map"
        )
        if stale:
            self.unsupported = True
            self.status.setText(stale)
        else:
            self.status.setText("Run map unavailable: " + error)

    def close_timers(self) -> None:
        self.retry.stop()
