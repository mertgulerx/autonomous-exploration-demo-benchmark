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

from collections import deque
from pathlib import Path
import re
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from .gui_support import Transport, button


class LogViewer(QtWidgets.QWidget):
    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self.attempt_id: str | None = None
        self.directory: str | None = None
        self.offset = 0
        self.pending = ""
        self.partial = ""
        self.lines: deque[str] = deque(maxlen=2000)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        self.component = QtWidgets.QComboBox()
        self.component.addItems(["worker", "simulation", "robot", "navigation", "slam", "explorer"])
        self.component.currentIndexChanged.connect(self.reset)
        self.severity = QtWidgets.QComboBox()
        self.severity.addItems(["All severities", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"])
        self.severity.currentIndexChanged.connect(self.render)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search buffered log lines")
        self.search.textChanged.connect(self.render)
        self.follow = QtWidgets.QCheckBox("Follow")
        self.follow.setChecked(True)
        for widget in (self.component, self.severity, self.search, self.follow):
            controls.addWidget(widget)
        button("Load more", self.load, controls)
        button("Open logs", self.open_directory, controls)
        layout.addLayout(controls)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.document().setMaximumBlockCount(2000)
        layout.addWidget(self.text)
        self.status = QtWidgets.QLabel("Select an active or retained attempt")
        layout.addWidget(self.status)
        transport.received.connect(self.received)
        transport.failed.connect(self.failed)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def select_attempt(self, identity: str, directory: str | None = None) -> None:
        self.attempt_id, self.directory = identity, directory
        self.reset()

    def reset(self, *args: Any) -> None:
        self.offset, self.partial, self.pending = 0, "", ""
        self.lines.clear()
        self.text.clear()
        self.load()

    def poll(self) -> None:
        if self.isVisible() and self.follow.isChecked():
            self.load()

    def load(self) -> None:
        if self.attempt_id and not self.pending:
            self.pending = "component-log-" + uuid.uuid4().hex
            self.transport.submit(
                self.pending,
                "logs.read",
                {
                    "attempt_id": self.attempt_id,
                    "component": self.component.currentText(),
                    "offset": self.offset,
                    "limit": 65536,
                },
            )

    def received(self, tag: str, result: Any) -> None:
        if tag != self.pending:
            return
        self.pending = ""
        self.offset = result["next_offset"]
        parts = (self.partial + result["text"]).split("\n")
        self.partial = parts.pop()[-65536:]
        self.lines.extend(parts)
        self.status.setText(
            f"{self.offset:,} bytes read · {len(self.lines):,} buffered lines"
            if result["available"]
            else "No stored log for this component"
        )
        self.render()

    def render(self, *args: Any) -> None:
        needle, severity = self.search.text().casefold(), self.severity.currentText()
        lines = []
        for line in (*self.lines, self.partial):
            if needle not in line.casefold():
                continue
            level = re.search(r"\[(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL)\]", line)
            if self.severity.currentIndex() and (
                not level or level[1].replace("WARNING", "WARN") != severity
            ):
                continue
            lines.append(line)
        position = self.text.verticalScrollBar().value()
        self.text.setPlainText("\n".join(lines))
        self.text.verticalScrollBar().setValue(
            self.text.verticalScrollBar().maximum() if self.follow.isChecked() else position
        )

    def failed(self, tag: str, message: str) -> None:
        if tag == self.pending:
            self.pending = ""
            self.status.setText(message)

    def open_directory(self) -> None:
        if self.directory:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(self.directory))))
