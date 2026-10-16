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

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_support import PagedTable, Transport, button
from .gui_style import DetailsPanel, duration, label


class ReportBrowser(QtWidgets.QWidget):
    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self.campaign_id: str | None = None
        self.report_id: str | None = None
        self.artifact: dict[str, Any] | None = None
        self.preview_tag = ""
        self.generation_tag = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(label("Scientific reports", "section"))
        layout.addWidget(
            label(
                "Choose a report, preview its figures, or open the full publication PDF.", "muted"
            )
        )
        self.reports = PagedTable(
            transport,
            "report-table",
            "reports",
            [
                ("created_ns", "Created"),
                ("state", "State"),
                ("detail", "Details"),
            ],
        )
        self.reports.selected.connect(self.select_report)
        self.reports.setMinimumHeight(140)
        self.reports.view.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.reports.view.setColumnWidth(1, 110)
        layout.addWidget(self.reports, 1)
        split = QtWidgets.QSplitter()
        split.setObjectName("reports-split")
        self.artifact_table = PagedTable(
            transport,
            "report-artifacts",
            "artifacts",
            [("kind", "Artifact"), ("bytes", "Size"), ("path", "File")],
            method="reports.get",
            result_key="artifacts",
        )
        self.artifact_table.setEnabled(False)
        self.artifact_table.sync_timer.stop()
        self.artifacts, self.model = self.artifact_table.view, self.artifact_table.model
        for column, width in enumerate((150, 85)):
            self.artifacts.setColumnWidth(column, width)
        self.artifact_table.selected.connect(self.select_artifact)
        self.artifact_table.loaded.connect(self.show_report)
        self.artifacts.doubleClicked.connect(lambda _: self.open_artifact())
        split.addWidget(self.artifact_table)
        panel = QtWidgets.QWidget()
        pane = QtWidgets.QVBoxLayout(panel)
        self.preview = QtWidgets.QLabel("Select a PNG preview or open a PDF")
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 100)
        self.preview.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored
        )
        pane.addWidget(self.preview, 1)
        self.detail = DetailsPanel("Report details")
        pane.addWidget(self.detail)
        split.addWidget(panel)
        layout.addWidget(split, 2)
        controls = QtWidgets.QHBoxLayout()
        button("Open selected file", self.open_artifact, controls)
        button("Generate reports", self.generate, controls, role="primary")
        controls.addStretch()
        layout.addLayout(controls)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    def set_campaign(self, identity: str) -> None:
        if identity == self.campaign_id:
            return
        self.campaign_id = identity
        self.report_id = self.artifact = None
        self.preview_tag = self.generation_tag = ""
        self.artifact_table.setEnabled(False)
        self.artifact_table.sync_timer.stop()
        self.artifact_table.filters = {}
        self.model.replace([])
        self.artifacts.clearSelection()
        self.preview.clear()
        self.detail.clear()
        self.reports.filters = {"campaign_id": identity}
        self.reports.model.replace([])
        self.reports.view.clearSelection()
        self.reports.reset_page()

    def select_report(self, row: dict[str, Any]) -> None:
        expected = self.reports.filters.get("campaign_id")
        if expected and row["campaign_id"] != expected:
            return
        if self.report_id == row["id"]:
            return
        self.report_id = row["id"]
        self.campaign_id = row["campaign_id"]
        self.artifact = None
        self.preview_tag = ""
        self.preview.clear()
        self.artifact_table.setEnabled(True)
        self.artifact_table.filters = {"report_id": self.report_id}
        self.artifact_table.reset_page()
        self.artifact_table.sync_timer.start()

    def select_artifact(self, row: dict[str, Any]) -> None:
        self.artifact = row
        self.preview_tag = ""
        self.preview.clear()
        if Path(self.artifact["path"]).suffix == ".png":
            self.preview_tag = "report-preview-" + uuid.uuid4().hex
            self.transport.submit(
                self.preview_tag, "local.read_image", {"path": self.artifact["path"]}
            )

    def open_artifact(self) -> None:
        if self.artifact:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.artifact["path"]))

    def generate(self) -> None:
        if self.campaign_id:
            self.generation_tag = "report-generate:" + self.campaign_id + ":" + uuid.uuid4().hex
            self.transport.submit(
                self.generation_tag, "reports.generate", {"campaign_id": self.campaign_id}
            )

    def show_report(self, result: dict[str, Any]) -> None:
        report, manifest = result["report"], result.get("manifest") or {}
        self.detail.set_record(
            {
                "state": report.get("state"),
                "created_ns": report.get("created_ns"),
                "finished_ns": report.get("finished_ns"),
                "recorded_runs": manifest.get("selection", {}).get("retained_attempts"),
                "files": result["total"],
                "generation_time": duration(manifest.get("elapsed_wall_seconds")),
                **({"message": report["detail"]} if report.get("detail") else {}),
            }
        )

    def received(self, tag: str, result: Any) -> None:
        if tag == self.preview_tag:
            image = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(result["image"]))
            if image.isNull():
                self.preview.setText("Preview could not be decoded")
            else:
                self.preview.setPixmap(
                    image.scaled(
                        self.preview.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                )
        elif tag == self.generation_tag:
            self.detail.setPlainText("Report queued")
            self.reports.refresh()

    def failed(self, tag: str, message: str) -> None:
        if tag in (self.preview_tag, self.generation_tag):
            self.detail.setPlainText(message)
