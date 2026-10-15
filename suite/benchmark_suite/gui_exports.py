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
from typing import Any
import uuid

from PySide6 import QtCore, QtWidgets

from .gui_support import PagedTable, Transport, button
from .gui_style import label


class ExportCopy(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, source: Path, destination: Path):
        super().__init__()
        self.source, self.destination = source, destination

    def run(self) -> None:
        from .exports import copy_export

        try:
            result = copy_export(
                self.source,
                self.destination,
                self.isInterruptionRequested,
                lambda done, total: self.progress.emit(done // 1024, total // 1024),
            )
            self.completed.emit(result)
        except (OSError, ValueError) as error:
            self.failed.emit(str(error))


class ExportBrowser(QtWidgets.QWidget):
    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self.campaign_id: str | None = None
        self.selected: dict[str, Any] | None = None
        self.preparation_tag = ""
        self.worker: ExportCopy | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(label("Portable datasets", "section"))
        layout.addWidget(
            label(
                "Export recorded measurements, maps, logs and immutable configuration snapshots.",
                "muted",
            )
        )
        self.table = PagedTable(
            transport,
            "export-table",
            "exports",
            [
                ("id", "Export"),
                ("state", "State"),
                ("detail", "Details"),
                ("path", "Bundle directory"),
            ],
        )
        self.table.selected.connect(lambda row: setattr(self, "selected", row))
        layout.addWidget(self.table, 1)
        controls = QtWidgets.QHBoxLayout()
        button("Prepare export", self.prepare, controls, role="primary")
        button("Copy bundle", self.copy, controls)
        self.status = QtWidgets.QLabel(
            "Portable Parquet, maps, logs, configuration and hash receipts"
        )
        self.status.setWordWrap(True)
        controls.addWidget(self.status, 1)
        layout.addLayout(controls)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)
        timer = QtCore.QTimer(self)
        timer.setInterval(1000)
        timer.timeout.connect(lambda: self.table.refresh() if self.isVisible() else None)
        timer.start()

    def set_campaign(self, identity: str) -> None:
        if identity == self.campaign_id:
            return
        self.campaign_id, self.selected = identity, None
        self.preparation_tag = ""
        self.table.filters, self.table.offset = {"campaign_id": identity}, 0
        self.table.model.replace([])
        self.table.view.clearSelection()
        self.table.reset_page()
        if not self.worker or not self.worker.isRunning():
            self.status.setText("Choose or prepare an export for this campaign")

    def prepare(self) -> None:
        if self.campaign_id:
            self.preparation_tag = "dataset-export:" + self.campaign_id + ":" + uuid.uuid4().hex
            self.transport.submit(
                self.preparation_tag, "dataset.export", {"campaign_id": self.campaign_id}
            )

    def copy(self) -> None:
        if (
            not self.selected
            or self.selected["state"] != "FINISHED"
            or self.selected.get("campaign_id") != self.campaign_id
        ):
            self.status.setText("Select a finished export from the current campaign")
            return
        if self.worker and self.worker.isRunning():
            self.status.setText("A bundle copy is already running")
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose export destination")
        if parent:
            destination = Path(parent) / (
                self.selected["campaign_id"] + "-" + self.selected["id"][:8]
            )
            self.worker = ExportCopy(Path(self.selected["path"]), destination)
            self.worker.progress.connect(
                lambda done, total: self.status.setText(
                    f"Copying and checking {done:,} / {total:,} KiB"
                )
            )
            self.worker.failed.connect(self.status.setText)
            self.worker.completed.connect(
                lambda result: self.status.setText("Export copied: " + result["path"])
            )
            self.worker.start()

    def received(self, tag: str, result: Any) -> None:
        if tag == self.preparation_tag:
            self.status.setText("Preparing export: " + result["export_id"])
            self.table.refresh()

    def failed(self, tag: str, message: str) -> None:
        if tag in (self.preparation_tag, "export-table"):
            self.status.setText(message)

    def close_copy(self) -> bool:
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            return self.worker.wait(1500)
        return True
