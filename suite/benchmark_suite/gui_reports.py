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

import json
from pathlib import Path
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_support import PagedTable, TableModel, Transport, button
from .gui_style import DetailsPanel, label


class ReportBrowser(QtWidgets.QWidget):
    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self.campaign_id: str | None = None
        self.report_id: str | None = None
        self.artifact: dict[str, Any] | None = None
        self.offset, self.total = 0, 0
        self.request_tag = ""
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
        self.artifacts = QtWidgets.QTableView()
        self.model = TableModel(
            [("kind", "Artifact"), ("attempt_id", "Attempt"), ("bytes", "Bytes"), ("path", "File")]
        )
        self.artifacts.setModel(self.model)
        self.artifacts.verticalHeader().hide()
        self.artifacts.setShowGrid(False)
        self.artifacts.verticalHeader().setDefaultSectionSize(40)
        self.artifacts.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.artifacts.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.artifacts.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((150, 85, 85)):
            self.artifacts.setColumnWidth(column, width)
        self.artifacts.selectionModel().currentRowChanged.connect(
            lambda current, previous: self.select_artifact(current)
        )
        self.artifacts.doubleClicked.connect(lambda _: self.open_artifact())
        split.addWidget(self.artifacts)
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
        button("Previous artifacts", lambda: self.page(-100), controls)
        button("Next artifacts", lambda: self.page(100), controls)
        button("Open selected file", self.open_artifact, controls)
        button("Generate reports", self.generate, controls, role="primary")
        self.count = QtWidgets.QLabel()
        controls.addWidget(self.count, 1)
        layout.addLayout(controls)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)

    def set_campaign(self, identity: str) -> None:
        if identity == self.campaign_id:
            return
        self.campaign_id = identity
        self.report_id = self.artifact = None
        self.request_tag = self.preview_tag = self.generation_tag = ""
        self.offset = self.total = 0
        self.model.replace([])
        self.artifacts.clearSelection()
        self.preview.clear()
        self.detail.clear()
        self.count.clear()
        self.reports.filters = {"campaign_id": identity}
        self.reports.model.replace([])
        self.reports.view.clearSelection()
        self.reports.reset_page()

    def select_report(self, row: dict[str, Any]) -> None:
        expected = self.reports.filters.get("campaign_id")
        if expected and row["campaign_id"] != expected:
            return
        self.report_id, self.offset, self.total = row["id"], 0, 0
        self.campaign_id = row["campaign_id"]
        self.artifact = None
        self.preview_tag = ""
        self.preview.clear()
        self.model.replace([])
        self.load()

    def page(self, delta: int) -> None:
        candidate = max(0, self.offset + delta)
        if candidate < self.total or candidate == 0:
            self.offset = candidate
            self.load()

    def load(self) -> None:
        if self.report_id:
            self.request_tag = "report-detail-" + uuid.uuid4().hex
            self.transport.submit(
                self.request_tag,
                "reports.get",
                {"report_id": self.report_id, "offset": self.offset, "limit": 100},
            )

    def select_artifact(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid() or index.row() >= len(self.model.rows):
            return
        self.artifact = self.model.rows[index.row()]
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

    def received(self, tag: str, result: Any) -> None:
        if tag == self.request_tag:
            self.total = result["total"]
            self.artifact = None
            self.preview_tag = ""
            self.model.replace(result["artifacts"])
            self.count.setText(
                f"{self.offset + 1 if self.total else 0}–{min(self.offset + 100, self.total)} of {self.total} artifacts"
            )
            self.detail.setPlainText(
                json.dumps(result["manifest"] or result["report"], ensure_ascii=False, indent=2)
            )
        elif tag == self.preview_tag:
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
            self.detail.setPlainText("Report queued: " + result["report_id"])
            self.reports.refresh()

    def failed(self, tag: str, message: str) -> None:
        if tag in (self.request_tag, self.preview_tag, self.generation_tag):
            self.detail.setPlainText(message)
