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

"""Results, Export study and Run analysis (31; RESULTS_STUDY.md decision 6; RESULTS_OVERHAUL A13, A15, RD-23).

Export study asks the engine for a study export (``study.export``) and follows it (``study.export_status``,
polled each second). Run analysis starts the analysis package on the newest finished export in its own process
and environment (``.venv-analysis``, never the GUI's): ``python -m benchmark_analysis --export DIR`` in the
checkout's ``analysis/``, its output streamed line by line. Its protocol (``benchmark_analysis.cli``): progress and
``warning:`` lines, ``done: <dir>`` last, ``refused: …`` on stderr; exit 0 done, 2 refused, 1 failed, 143 stopped.
Cancel sends SIGTERM, which the package answers by removing what it had started. Either job shows its state,
elapsed time and log in a non-modal dialog that can be closed and opened again from the study card; closing it
never stops the job. One analysis and one export run at a time.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Literal

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_tokens
from .gui_controls import BadgeLook, StatusBadge
from .gui_dialogs import wrap_text
from .gui_style import duration
from .gui_support import ROOT, Transport
from .gui_theme import ThemeManager
from .gui_widgets import (
    DIALOG_TITLE,
    LOG_TEXT,
    Button,
    FlexRow,
    TextLine,
    TextStyle,
    Themed,
    draw_text,
    paint_shadow,
    rounded_path,
    shadow_margin,
)

ANALYSIS = ROOT / "analysis"
PYTHON = ROOT / ".venv-analysis/bin/python"
STATUS_TEXT = TextStyle(12.5, 400, 18)
STATUS_MONO = TextStyle(12, 400, 18, mono=True)
NOTE_TEXT = TextStyle(12, 400, 18)
LINES = 400  # the log keeps this many lines
STATUS_RETRIES = (
    5  # export status polls that may fail in a row before the export is reported failed
)
State = Literal["running", "finished", "refused", "failed", "stopped"]
LOOKS: dict[State, BadgeLook] = {
    "running": BadgeLook("Running", "running", "loader"),
    "finished": BadgeLook("Finished", "ok", "ccheck"),
    "refused": BadgeLook("Refused", "warn", "ban"),
    "failed": BadgeLook("Failed", "bad", "cx"),
    "stopped": BadgeLook("Stopped", "neutral", "cstop"),
}


def analysis_available() -> str:
    """Why Run analysis cannot start here ("" when it can)."""
    if not (ANALYSIS / "benchmark_analysis").is_dir():
        return "The analysis package is not in this checkout"
    if not PYTHON.is_file():
        return "The analysis environment (.venv-analysis) is missing; ./suite_launch.sh builds it"
    return ""


@dataclass
class Job:
    """One export or analysis of a study, as the dialog and the study card show it."""

    kind: Literal["analysis", "export"]
    study_id: str
    export: str  # the export's short hash (analysis) or "" until the engine names it
    state: State = "running"
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    lines: list[str] = field(default_factory=list)
    detail: str = ""  # the outputs' directory, the export's path, or the reason it did not finish
    export_id: str | None = None
    status_failures: int = 0  # export status polls failed in a row
    cancelled: bool = False  # Cancel was pressed: a stop, not a failure, ends it
    partial: dict[bool, str] = field(default_factory=lambda: {False: "", True: ""})  # by stderr

    def elapsed(self) -> float:
        return (self.ended or time.monotonic()) - self.started

    def add(self, line: str) -> None:
        self.lines.append(line)
        del self.lines[:-LINES]


class LogBox(Themed):
    """``.lw``: the newest lines on the sunken fill (1 px rule, radius 6), 12/20 px mono, the last one in ink."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, parent)
        self.lines: list[str] = []
        self.setFixedHeight(200)

    def set_lines(self, lines: list[str]) -> None:
        self.lines = lines
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        box = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.sunken"))
        painter.drawPath(rounded_path(box, 5.5))
        shown = max(1, (self.height() - 20) // 20)
        lines = self.lines[-shown:]
        for index, line in enumerate(lines):
            token = "text.ink" if index == len(lines) - 1 else "text.body"
            text = LOG_TEXT.elide(
                line, self.width() - 26
            )  # ``.lw`` padding 10 12 inside its 1 px rule
            draw_text(painter, 13, 11 + 20 * index, text, LOG_TEXT, self.color(token))


class LogDialog(Themed):
    """``.dialog.alog`` (600 px), non-modal over the page: the title, the state badge with the elapsed time and
    what runs, the log, a note, and Cancel analysis (while an analysis runs) and Close. Its popup shadow is
    painted in its own margin."""

    cancel = QtCore.Signal()

    WIDTH = 600

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget) -> None:
        super().__init__(themes, parent)
        self.margin = shadow_margin(gui_tokens.POPUP_SHADOW["light"])
        self.title = TextLine(themes, "", DIALOG_TITLE, "text.ink", self)
        self.badge = StatusBadge(themes, None, self)
        self.status: list[tuple[str, bool]] = []  # (text, mono)
        self.log = LogBox(themes, self)
        self.note = ""
        self.foot = FlexRow(self, gap=8, left=24, right=24)
        self.stop = Button(themes, "Cancel analysis", icon="stop", variant="danger")
        self.foot.add(self.stop)
        self.foot.add_stretch()
        self.close_button = Button(themes, "Close")
        self.foot.add(self.close_button)
        self.stop.clicked.connect(self.cancel.emit)
        self.close_button.clicked.connect(self.hide)
        self.hide()

    def show_job(self, job: Job) -> None:
        self.title.setText(
            ("Run analysis" if job.kind == "analysis" else "Export study") + " · " + job.study_id
        )
        self.badge.set_look(LOOKS[job.state])
        self.status = [(duration(job.elapsed()), False)]
        if job.export:
            self.status.append(("export " + job.export, True))
        if job.kind == "analysis":
            self.status.append(("its own environment, not the GUI", False))
        self.log.set_lines(job.lines)
        self.stop.setVisible(job.kind == "analysis" and job.state == "running")
        self.stop.setEnabled(not job.cancelled)  # one stop request; the package is cleaning up
        self.note = {
            "running": "The outputs replace the study's current ones only when the run finishes; closing this "
            "window keeps it running."
            if job.kind == "analysis"
            else "The export is a new, immutable directory; closing this window keeps it running.",
            "finished": "Written to " + job.detail,
            "refused": "Refused: " + job.detail,
            "failed": "Failed: " + job.detail,
            "stopped": "Stopped; nothing was kept.",
        }[job.state]
        self._layout()
        self.foot.relayout()
        self.update()

    def _layout(self) -> None:
        m = self.margin
        width = self.WIDTH
        lines = wrap_text(self.note, NOTE_TEXT, width - 48)
        height = 1 + 20 + 22 + 10 + 22 + 12 + 200 + 10 + 18 * len(lines) + 16 + 1 + 61
        parent = self.parentWidget()
        if parent is not None:
            # where screen 31 puts it: over the Study tab's viewer, below the page head
            x = max(8, (parent.width() - width) // 2 + 32)
            self.setGeometry(x - m, 56 + 196 - m, width + 2 * m, height + 2 * m)
        self.title.setGeometry(m + 1 + 24, m + 1 + 20, width - 50, 22)
        self.badge.setGeometry(
            m + 1 + 24, m + 1 + 20 + 22 + 10, int(self.badge.natural_width()) + 1, 22
        )
        self.log.setGeometry(m + 1 + 24, m + 1 + 20 + 22 + 10 + 22 + 12, width - 50, 200)
        self.foot.setGeometry(m + 1, m + height - 1 - 61 + 15, width - 2, 32)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        m = self.margin
        card = QtCore.QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)
        paint_shadow(painter, card.toRect(), 10, gui_tokens.POPUP_SHADOW[self.theme.name])
        painter.setPen(QtGui.QPen(self.color("border.line"), 1))
        painter.setBrush(self.color("surface.card"))
        painter.drawPath(rounded_path(card.adjusted(0.5, 0.5, -0.5, -0.5), 9.5))
        # the status row: the badge, then its parts 10 px apart with faint dots between them
        x = m + 1 + 24 + self.badge.natural_width() + 10
        top = m + 1 + 20 + 22 + 10 + 2
        for index, (text, mono) in enumerate(self.status):
            if index:
                x += draw_text(painter, x, top, "·", STATUS_TEXT, self.color("text.tertiary")) + 10
            style = STATUS_MONO if mono else STATUS_TEXT
            x += draw_text(painter, x, top, text, style, self.color("text.secondary")) + 10
        y = m + 1 + 20 + 22 + 10 + 22 + 12 + 200 + 10
        for line in wrap_text(self.note, NOTE_TEXT, card.width() - 48):
            draw_text(painter, m + 1 + 24, y, line, NOTE_TEXT, self.color("text.secondary"))
            y += 18
        rule = card.bottom() - 61
        painter.fillRect(
            QtCore.QRectF(card.left() + 1, rule, card.width() - 2, 1), self.color("border.line")
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()  # clicks on the dialog stay on it


class Jobs(QtCore.QObject):
    """The study's export and analysis: started from the page head, followed here, shown in the dialog.

    ``changed`` carries the study whose job changed (the study card's Analysis row), ``exported`` and
    ``analysed`` a study whose export or outputs are new (the page reads ``study.get`` again)."""

    changed = QtCore.Signal(str)
    exported = QtCore.Signal(str)
    analysed = QtCore.Signal(str)

    def __init__(
        self,
        themes: ThemeManager,
        transport: Transport,
        dialog_parent: Callable[[], QtWidgets.QWidget | None],
    ) -> None:
        super().__init__()
        self.themes, self.transport, self.dialog_parent = themes, transport, dialog_parent
        self.analysis: Job | None = None
        self.export: Job | None = None
        self.process: QtCore.QProcess | None = None
        self.dialog: LogDialog | None = None
        self.shown: Job | None = None
        self.tick = QtCore.QTimer(self)
        self.tick.setInterval(1000)
        self.tick.timeout.connect(self._tick)
        self.export_poll = QtCore.QTimer(self)
        self.export_poll.setSingleShot(True)
        self.export_poll.setInterval(1000)
        self.export_poll.timeout.connect(self._poll_export)

    def job(self, study_id: str | None) -> Job | None:
        """The study's latest job of this session (the analysis first: the card's Analysis row)."""
        for job in (self.analysis, self.export):
            if job is not None and job.study_id == study_id:
                return job
        return None

    def running(self) -> bool:
        return self.process is not None

    # the dialog

    def show(self, job: Job | None) -> None:
        parent = self.dialog_parent()
        if job is None or parent is None:
            return
        if self.dialog is None or self.dialog.parentWidget() is not parent:
            self.dialog = LogDialog(self.themes, parent)
            self.dialog.cancel.connect(self.cancel)
        self.shown = job
        self.dialog.show_job(job)
        self.dialog.show()
        self.dialog.raise_()

    def _refresh(self, job: Job) -> None:
        if self.dialog is not None and self.shown is job and self.dialog.isVisible():
            self.dialog.show_job(job)
        self.changed.emit(job.study_id)

    def _tick(self) -> None:
        for job in (self.analysis, self.export):
            if job is not None and job.state == "running":
                self._refresh(job)
        if not any(j is not None and j.state == "running" for j in (self.analysis, self.export)):
            self.tick.stop()

    # the analysis

    def start_analysis(self, study_id: str, export: Mapping[str, Any]) -> None:
        """Run the analysis package on ``export`` (a finished export of ``study.get``)."""
        if self.process is not None:
            self.show(self.analysis)
            return
        hash8 = str(export.get("sums_sha256") or "")[:8]
        job = Job("analysis", study_id, hash8)
        self.analysis = job
        reason = analysis_available()
        path = str(export.get("path") or "")
        if reason or not path:
            job.state, job.ended, job.detail = (
                "failed",
                time.monotonic(),
                reason or "the export has no path",
            )
            self.show(job)
            self.changed.emit(study_id)
            return
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(ANALYSIS))
        process.setProgram(str(PYTHON))
        process.setArguments(["-m", "benchmark_analysis", "--export", path])
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        for name in (
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "PYTHONHOME",
        ):  # its own environment only, never the GUI's
            environment.remove(name)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(lambda: self._read(process, False))
        process.readyReadStandardError.connect(lambda: self._read(process, True))
        process.finished.connect(lambda code, status: self._finished(process, code, status))
        process.errorOccurred.connect(lambda error: self._error(process, error))
        self.process = process
        job.add(f"$ python -m benchmark_analysis --export {Path(path).name}")
        process.start()
        self.tick.start()
        self.show(job)
        self.changed.emit(study_id)

    def _read(self, process: QtCore.QProcess, error: bool) -> None:
        job = self.analysis
        if job is None or process is not self.process:
            return
        data = process.readAllStandardError() if error else process.readAllStandardOutput()
        # a read may end inside a line: its start waits for the rest (all of it once the process ended)
        lines = (job.partial[error] + bytes(data.data()).decode("utf-8", "replace")).split("\n")
        job.partial[error] = (
            "" if process.state() == QtCore.QProcess.ProcessState.NotRunning else lines.pop()
        )
        for line in (line.rstrip("\r") for line in lines):
            if not line:
                continue
            job.add(line)
            if line.startswith("done: "):
                job.detail = line.removeprefix("done: ")
            elif line.startswith("refused: "):
                job.detail = line.removeprefix("refused: ")
        self._refresh(job)

    def _finished(
        self, process: QtCore.QProcess, code: int, status: QtCore.QProcess.ExitStatus
    ) -> None:
        job = self.analysis
        if job is None or process is not self.process:
            return
        self._read(process, False)
        self._read(process, True)
        job.ended = time.monotonic()
        # 143: the package stopped on SIGTERM; a crash counts as a stop only when Cancel asked for it
        crashed = status == QtCore.QProcess.ExitStatus.CrashExit
        stopped = code == 143 or (crashed and job.cancelled)
        job.state = (
            "finished"
            if code == 0
            else "refused"
            if code == 2
            else "stopped"
            if stopped
            else "failed"
        )
        if job.state == "failed" and crashed:
            job.detail = (
                "the process ended unexpectedly; it may have left an empty output directory in the study's "
                "analysis folder, which must be removed before this export is analysed again"
            )
        elif job.state == "failed" and not job.detail:
            job.detail = f"exit code {code}; the log above says why"
        job.add(f"exit {code}")
        self.process = None
        process.deleteLater()
        self._refresh(job)
        # any end: new outputs, a kept refusal (RO-22), or a directory a failed run left (listed as unreadable)
        self.analysed.emit(job.study_id)

    def _error(self, process: QtCore.QProcess, error: QtCore.QProcess.ProcessError) -> None:
        if (
            error == QtCore.QProcess.ProcessError.FailedToStart
            and process is self.process
            and self.analysis
        ):
            job = self.analysis
            job.state, job.ended, job.detail = (
                "failed",
                time.monotonic(),
                "the analysis environment did not start",
            )
            self.process = None
            process.deleteLater()
            self._refresh(job)

    def cancel(self) -> None:
        """SIGTERM: the package removes its claimed and staging directories and exits 143."""
        if self.process is not None:
            if self.analysis is not None:
                if self.analysis.cancelled:
                    return
                self.analysis.cancelled = True
                self.analysis.add("Stopping…")
                self._refresh(self.analysis)
            self.process.terminate()

    # the export

    def start_export(self, study_id: str) -> None:
        if self.export is not None and self.export.state == "running":
            self.show(self.export)
            return
        job = Job("export", study_id, "")
        self.export = job
        job.add("Asking the engine for a study export…")
        self.transport.submit("results-export:" + study_id, "study.export", {"study_id": study_id})
        self.tick.start()
        self.show(job)
        self.changed.emit(study_id)

    def _poll_export(self) -> None:
        job = self.export
        if job is not None and job.state == "running":
            params: dict[str, Any] = {"study_id": job.study_id}
            if job.export_id:
                params["export_id"] = job.export_id
            self.transport.submit(
                "results-export-status:" + job.study_id, "study.export_status", params
            )

    def received(self, tag: str, result: Any) -> bool:
        job = self.export
        if tag.startswith("results-export:"):
            if job is not None and job.state == "running":
                job.export_id = result.get("export_id")
                job.add("Export accepted; the engine writes it off its loop")
                self.export_poll.start()
                self._refresh(job)
            return True
        if tag.startswith("results-export-status:"):
            if job is None or job.state != "running":
                return True
            state = result.get("state")
            job.status_failures = 0
            if state == "running":
                self.export_poll.start()
                return True
            job.ended = time.monotonic()
            if state == "finished":
                job.state = "finished"
                job.export = str(result.get("sums_sha256") or "")[:8]
                job.detail = str(result.get("path") or "")
                job.add(
                    f"{result.get('files', '?')} files, {result.get('final_runs', '?')} final runs"
                )
                for warning in result.get("warnings") or []:
                    job.add("warning: " + str(warning))
                job.add("SHA256SUMS " + job.export)
                self.exported.emit(job.study_id)
            else:
                job.state, job.detail = (
                    "failed",
                    str(result.get("detail") or result.get("reason") or state),
                )
            self._refresh(job)
            return True
        return False

    def failed(self, tag: str, error: str) -> bool:
        if not tag.startswith(("results-export:", "results-export-status:")):
            return False
        job = self.export
        if job is not None and job.state == "running" and tag.startswith("results-export-status:"):
            # one unanswered status poll says nothing about the export, which runs on in the engine
            job.status_failures += 1
            if job.status_failures < STATUS_RETRIES:
                job.add(f"Status not read ({error}); asking again")
                self.export_poll.start()
                self._refresh(job)
                return True
        if job is not None and job.state == "running":
            job.state, job.ended, job.detail = (
                "refused" if tag.startswith("results-export:") else "failed",
                time.monotonic(),
                error
                if tag.startswith("results-export:")
                else f"its status could not be read ({error}); the export may still finish and "
                "then appears on the study card",
            )
            job.add(error)
            self._refresh(job)
        return True

    def close(self) -> None:
        """Window closing: a running analysis is stopped (it removes what it had started)."""
        self.tick.stop()
        self.export_poll.stop()
        if self.process is not None:
            self.process.terminate()
            if not self.process.waitForFinished(5000):
                self.process.kill()
