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

from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets


COLORS = {
    "surface.base": "#F4EEE6",
    "surface.raised": "#FAF6EF",
    "surface.sunken": "#EDE4D3",
    "surface.selected": "#ECD9CC",
    "border.subtle": "#CEC3B1",
    "border.control": "#66655A",
    "text.strong": "#2A1D12",
    "text.heading": "#000000",
    "text.body": "#5D5D51",
    "text.muted": "#66655A",
    "text.accent": "#9A4A2F",
    "text.on-accent": "#FAF6EF",
    "icon.decorative": "#8F8D81",
    "accent.default": "#B15739",
    "action.primary": "#9A4A2F",
    "action.secondary": "#000000",
    "focus.ring": "#000000",
    "status.success": "#000000",
    "status.success.soft": "#EDE4D3",
    "status.warning": "#7D5A10",
    "status.warning.soft": "#E7DDCE",
    "status.error": "#A63A2E",
    "status.error.soft": "#E9D2C7",
    "status.info": "#3D6470",
    "status.info.soft": "#DAD8D0",
}


def overlay(foreground: str, background: str, opacity: float) -> str:
    front, back = QtGui.QColor(foreground), QtGui.QColor(background)
    channels = [
        round(f * opacity + b * (1 - opacity))
        for f, b in zip(
            (front.red(), front.green(), front.blue()),
            (back.red(), back.green(), back.blue()),
        )
    ]
    return QtGui.QColor(*channels).name()


for prefix, background, foreground in (
    ("interaction", "surface.raised", "text.strong"),
    ("interaction.selected", "surface.selected", "text.strong"),
    ("action.primary", "action.primary", "text.strong"),
    ("action.secondary", "action.secondary", "surface.raised"),
):
    for state, opacity in (("hover", 0.10), ("pressed", 0.18)):
        COLORS[f"{prefix}.{state}"] = overlay(COLORS[foreground], COLORS[background], opacity)

COLORS.update(
    {
        "text.disabled": overlay(COLORS["text.body"], COLORS["surface.base"], 0.4),
        "surface.disabled": overlay(COLORS["surface.sunken"], COLORS["surface.base"], 0.4),
        "border.disabled": overlay(COLORS["border.control"], COLORS["surface.base"], 0.4),
    }
)


@lru_cache(maxsize=80)
def icon(name: str, color: str = COLORS["text.muted"]) -> QtGui.QIcon:
    source = Path(__file__).with_name("icons") / (name + ".svg")
    data = source.read_text().replace("currentColor", color).encode()
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(data))
    pixmap = QtGui.QPixmap(48, 48)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QtGui.QIcon(pixmap)


def label(text: str, kind: str = "body") -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setProperty("kind", kind)
    return widget


def surface() -> QtWidgets.QFrame:
    widget = QtWidgets.QFrame()
    widget.setProperty("surface", True)
    return widget


def duration(seconds: Any) -> str:
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


# Display names that differ from the stored identifier's words.
LABELS = {"manual_stop": "Stopped"}


def humanize(value: str) -> str:
    return LABELS.get(value) or value.replace("_", " ").strip().capitalize()


def cell_text(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key in {
        "simulation_elapsed",
        "wall_elapsed",
        "pre_motion_simulation_elapsed",
        "pre_motion_wall_elapsed",
    }:
        return duration(value)
    if key in {"cpu_percent", "cpu_average_percent", "coverage_percent"} and isinstance(
        value, (float, int)
    ):
        return f"{value:.1f}%"
    if key in {"rss_mib", "rss_average_mib"} and isinstance(value, (float, int)):
        return f"{value:.1f} MiB"
    if key == "distance_m" and isinstance(value, (float, int)):
        return f"{value:.2f} m"
    if key == "bytes" and isinstance(value, int):
        return f"{value / (1024**2):.1f} MiB" if value >= 1024**2 else f"{value / 1024:.1f} KiB"
    if key == "path" and value:
        return Path(str(value)).name
    if key.endswith("_ns") and isinstance(value, int):
        return (
            datetime.fromtimestamp(value / 1e9, timezone.utc)
            .astimezone()
            .strftime("%d %b %Y · %H:%M")
        )
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key == "repetition" and isinstance(value, int):
        return str(value + 1)
    if key in {"state", "phase", "outcome", "report_state", "kind"}:
        return humanize(str(value))
    if key in {"id", "experiment_id", "attempt_id", "campaign_id"}:
        return str(value)[:8]
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".") if math.isfinite(value) else "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class Badge(QtWidgets.QLabel):
    def __init__(self, text: str = "Idle", state: str = "neutral"):
        super().__init__(text)
        self.setProperty("badge", True)
        self.set_state(text, state)

    def set_state(self, text: str, state: str = "neutral") -> None:
        self.setText(text)
        if self.property("tone") != state:
            self.setProperty("tone", state)
            self.style().unpolish(self)
            self.style().polish(self)


class DetailsPanel(QtWidgets.QWidget):
    def __init__(self, title: str | None = "Details"):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if title:
            layout.addWidget(label(title, "section"))
        self.body = QtWidgets.QLabel("Select an item to see its details")
        self.body.setWordWrap(True)
        self.body.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.body.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)
        layout.addWidget(scroll, 1)

    def setPlainText(self, text: str) -> None:
        self.body.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.body.setText(text)

    def set_record(self, record: Any) -> None:
        from html import escape

        self.body.setTextFormat(QtCore.Qt.TextFormat.RichText)
        if not isinstance(record, dict):
            self.body.setText(escape(str(record)))
            return
        fields = []
        for key, value in record.items():
            if (
                isinstance(value, (dict, list, tuple))
                or key.endswith(("_id", "_hash", "_sha256"))
                or key
                in {
                    "id",
                    "hash",
                    "directory",
                    "path",
                    "owner",
                    "schema_version",
                    "analysis_version",
                    "configuration",
                    "specification",
                    "record",
                }
            ):
                continue
            fields.append(
                f"<tr><td style='padding:4px 16px 4px 0'>{escape(humanize(key))}</td>"
                f"<td style='padding:4px 0'>{escape(cell_text(key, value))}</td></tr>"
            )
        self.body.setText("<table>" + "".join(fields[:24]) + "</table>")

    def toPlainText(self) -> str:
        if self.body.textFormat() == QtCore.Qt.TextFormat.PlainText:
            return self.body.text()
        document = QtGui.QTextDocument()
        document.setHtml(self.body.text())
        return document.toPlainText()

    def clear(self) -> None:
        self.setPlainText("Select an item to see its details")


def stylesheet() -> str:
    c = COLORS
    assets = Path(__file__).with_name("icons")
    return f"""
    QWidget {{ color:{c["text.body"]}; font-size:13px; outline:0; }}
    QMainWindow, QDialog {{ background:{c["surface.base"]}; }}
    QWidget#topbar {{ background:{c["surface.raised"]}; border-bottom:1px solid {c["border.subtle"]}; }}
    QFrame[surface="true"], QGroupBox {{ background:{c["surface.raised"]}; border:1px solid {c["border.subtle"]}; border-radius:10px; }}
    QLabel {{ background:transparent; border:none; }}
    QLabel[kind="brand"] {{ color:{c["text.heading"]}; font-size:17px; font-weight:600; }}
    QLabel[kind="title"] {{ color:{c["text.heading"]}; font-size:25px; font-weight:600; }}
    QLabel[kind="section"] {{ color:{c["text.heading"]}; font-size:15px; font-weight:600; }}
    QLabel[kind="metric"] {{ color:{c["text.strong"]}; font-size:25px; font-weight:600; }}
    QWidget#campaignCounters QLabel[kind="counter"] {{ color:{c["text.heading"]}; font-size:22px; font-weight:600; padding-right:2px; }}
    QWidget#campaignCounters QLabel[kind="counter"][total="true"] {{ font-size:25px; }}
    QWidget#campaignCounters QLabel[kind="muted"] {{ font-size:12px; }}
    QFrame#campaignCounterDivider {{ background:{c["border.subtle"]}; border:0; }}
    QLabel[kind="muted"] {{ color:{c["text.muted"]}; font-size:12px; }}
    QLabel[kind="error"] {{ color:{c["status.error"]}; }}
    QLabel[kind="success"] {{ color:{c["status.success"]}; }}
    QLabel[badge="true"] {{ background:{c["surface.sunken"]}; color:{c["text.body"]}; border-radius:5px; padding:4px 8px; font-size:11px; font-weight:600; }}
    QLabel[tone="active"] {{ background:{c["surface.selected"]}; color:{c["text.heading"]}; }}
    QLabel[tone="success"] {{ background:{c["status.success.soft"]}; color:{c["text.heading"]}; border:1px solid {c["status.success"]}; }}
    QLabel[tone="warning"] {{ background:{c["status.warning.soft"]}; color:{c["text.heading"]}; border:1px solid {c["status.warning"]}; }}
    QLabel[tone="danger"] {{ background:{c["status.error.soft"]}; color:{c["text.heading"]}; border:1px solid {c["status.error"]}; }}
    QListWidget#navigation {{ background:transparent; border:0; }}
    QListWidget#navigation::item {{ padding:9px 13px; margin:2px 4px; border:1px solid {c["border.subtle"]}; border-radius:7px; background:{c["surface.raised"]}; color:{c["text.body"]}; }}
    QListWidget#navigation::item:hover:enabled {{ background:{c["interaction.hover"]}; }}
    QListWidget#navigation::item:selected {{ background:{c["surface.selected"]}; border-color:{c["text.heading"]}; color:{c["text.heading"]}; font-weight:600; }}
    QListWidget#navigation::item:selected:hover:enabled {{ background:{c["interaction.selected.hover"]}; }}
    QListWidget#navigation::item:focus {{ border:2px solid {c["focus.ring"]}; padding:8px 12px; }}
    QPushButton, QToolButton {{ color:{c["text.heading"]}; background:{c["surface.raised"]}; border:1px solid {c["border.control"]}; border-radius:7px; padding:7px 12px; min-height:18px; }}
    QPushButton:hover:enabled, QToolButton:hover:enabled {{ background:{c["interaction.hover"]}; }}
    QPushButton:pressed:enabled, QToolButton:pressed:enabled {{ background:{c["interaction.pressed"]}; }}
    QPushButton[role="secondary"], QToolButton[role="secondary"] {{ background:{c["action.secondary"]}; border-color:{c["action.secondary"]}; color:{c["surface.base"]}; }}
    QPushButton[role="secondary"]:hover:enabled, QToolButton[role="secondary"]:hover:enabled {{ background:{c["action.secondary.hover"]}; }}
    QPushButton[role="secondary"]:pressed:enabled, QToolButton[role="secondary"]:pressed:enabled {{ background:{c["action.secondary.pressed"]}; }}
    QPushButton[role="primary"], QToolButton[role="primary"] {{ background:{c["action.primary"]}; border-color:{c["action.primary"]}; color:{c["text.on-accent"]}; font-weight:600; }}
    QPushButton[role="primary"]:hover:enabled, QToolButton[role="primary"]:hover:enabled {{ background:{c["action.primary.hover"]}; }}
    QPushButton[role="primary"]:pressed:enabled, QToolButton[role="primary"]:pressed:enabled {{ background:{c["action.primary.pressed"]}; }}
    QPushButton[role="selected"], QToolButton[role="selected"], QPushButton:checked, QToolButton:checked {{ background:{c["surface.selected"]}; color:{c["text.heading"]}; border-color:{c["text.heading"]}; font-weight:600; }}
    QPushButton[role="selected"]:hover:enabled, QToolButton[role="selected"]:hover:enabled, QPushButton:checked:hover:enabled, QToolButton:checked:hover:enabled {{ background:{c["interaction.selected.hover"]}; }}
    QPushButton[role="selected"]:pressed:enabled, QToolButton[role="selected"]:pressed:enabled, QPushButton:checked:pressed:enabled, QToolButton:checked:pressed:enabled {{ background:{c["interaction.selected.pressed"]}; }}
    QPushButton[role="danger"], QToolButton[role="danger"] {{ color:{c["status.error"]}; border-color:{c["status.error"]}; }}
    QPushButton:focus, QToolButton:focus {{ border:2px solid {c["focus.ring"]}; padding:6px 11px; }}
    QPushButton[role="secondary"]:focus, QToolButton[role="secondary"]:focus {{ border-color:{c["accent.default"]}; }}
    QPushButton:disabled, QPushButton[role]:disabled, QPushButton:checked:disabled, QToolButton:disabled, QToolButton[role]:disabled, QToolButton:checked:disabled {{ color:{c["text.disabled"]}; background:{c["surface.disabled"]}; border-color:{c["border.disabled"]}; }}
    QToolButton[tableAction="true"], QToolButton[tableAction="true"]:focus {{ padding:0; min-height:0; }}
    QToolButton[tableAction="true"]::menu-indicator {{ image:none; }}
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QAbstractSpinBox {{ background:{c["surface.raised"]}; color:{c["text.body"]}; border:1px solid {c["border.control"]}; border-radius:7px; padding:7px 9px; selection-background-color:{c["surface.selected"]}; selection-color:{c["text.heading"]}; }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border:2px solid {c["focus.ring"]}; padding:6px 8px; }}
    QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{ color:{c["text.disabled"]}; background:{c["surface.disabled"]}; border-color:{c["border.disabled"]}; }}
    QComboBox::drop-down {{ border:0; width:24px; }}
    QComboBox::down-arrow {{ image:url({assets / "chevron-down.svg"}); width:12px; height:12px; }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ border:0; width:22px; background:transparent; }}
    QAbstractSpinBox::up-button:hover:enabled, QAbstractSpinBox::down-button:hover:enabled {{ background:{c["interaction.hover"]}; }}
    QAbstractSpinBox::up-button:pressed:enabled, QAbstractSpinBox::down-button:pressed:enabled {{ background:{c["interaction.pressed"]}; }}
    QAbstractSpinBox::up-arrow {{ image:url({assets / "chevron-up.svg"}); width:10px; height:10px; }}
    QAbstractSpinBox::down-arrow {{ image:url({assets / "chevron-down.svg"}); width:10px; height:10px; }}
    QComboBox QAbstractItemView {{ background:{c["surface.raised"]}; selection-background-color:{c["surface.selected"]}; selection-color:{c["text.heading"]}; }}
    QCheckBox, QRadioButton {{ spacing:8px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width:16px; height:16px; border:1px solid {c["border.control"]}; border-radius:4px; background:{c["surface.raised"]}; }}
    QRadioButton::indicator {{ border-radius:8px; }}
    QCheckBox::indicator:hover:enabled, QRadioButton::indicator:hover:enabled {{ background:{c["interaction.hover"]}; }}
    QCheckBox::indicator:pressed:enabled, QRadioButton::indicator:pressed:enabled {{ background:{c["interaction.pressed"]}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background:{c["action.primary"]}; border-color:{c["action.primary"]}; image:url({assets / "check.svg"}); }}
    QCheckBox::indicator:checked:hover:enabled, QRadioButton::indicator:checked:hover:enabled {{ background:{c["action.primary.hover"]}; }}
    QCheckBox::indicator:checked:pressed:enabled, QRadioButton::indicator:checked:pressed:enabled {{ background:{c["action.primary.pressed"]}; }}
    QCheckBox::indicator:focus, QRadioButton::indicator:focus {{ border:2px solid {c["focus.ring"]}; width:14px; height:14px; }}
    QCheckBox:disabled, QRadioButton:disabled {{ color:{c["text.disabled"]}; }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ background:{c["surface.disabled"]}; border-color:{c["border.disabled"]}; }}
    QAbstractItemView {{ background:{c["surface.raised"]}; alternate-background-color:{c["surface.base"]}; border:1px solid {c["border.subtle"]}; border-radius:8px; gridline-color:{c["border.subtle"]}; selection-background-color:{c["surface.selected"]}; selection-color:{c["text.heading"]}; }}
    QAbstractItemView::item {{ padding:7px 8px; border:0; }}
    QAbstractItemView::item:selected {{ background:{c["surface.selected"]}; color:{c["text.heading"]}; }}
    QAbstractItemView::item:focus {{ border:2px solid {c["focus.ring"]}; padding:5px 6px; }}
    QTableView::item:focus {{ border:0; padding:7px 8px; }}
    QHeaderView::section {{ background:{c["surface.sunken"]}; color:{c["text.heading"]}; padding:9px 8px; border:0; border-bottom:1px solid {c["border.subtle"]}; font-size:12px; font-weight:600; }}
    QTableView QTableCornerButton::section {{ background:{c["surface.sunken"]}; border:0; }}
    QScrollArea {{ border:0; background:transparent; }}
    QScrollArea > QWidget > QWidget {{ background:transparent; }}
    QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
    QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
    QScrollBar::handle {{ background:{c["border.control"]}; border-radius:4px; min-height:28px; min-width:28px; }}
    QScrollBar::handle:hover {{ background:{c["text.heading"]}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}
    QTabWidget::pane {{ border:0; background:transparent; }}
    QTabBar::tab {{ color:{c["text.body"]}; background:{c["surface.raised"]}; padding:9px 15px; margin:0 6px 4px 0; border:1px solid {c["border.control"]}; border-radius:7px; }}
    QTabBar::tab:selected {{ color:{c["text.heading"]}; background:{c["surface.selected"]}; border-color:{c["text.heading"]}; font-weight:600; }}
    QTabBar::tab:hover:enabled {{ background:{c["interaction.hover"]}; }}
    QTabBar::tab:selected:hover:enabled {{ background:{c["interaction.selected.hover"]}; }}
    QTabBar::tab:focus {{ border:2px solid {c["focus.ring"]}; padding:8px 14px; }}
    QTabBar::tab:disabled {{ color:{c["text.disabled"]}; background:{c["surface.disabled"]}; border-color:{c["border.disabled"]}; }}
    QProgressBar {{ border:0; border-radius:4px; background:{c["surface.sunken"]}; min-height:6px; max-height:6px; }}
    QProgressBar::chunk {{ background:{c["accent.default"]}; border-radius:4px; }}
    QSplitter::handle {{ background:transparent; width:8px; height:8px; }}
    QStatusBar {{ background:{c["surface.base"]}; color:{c["text.muted"]}; border-top:1px solid {c["border.subtle"]}; }}
    QMenu, QToolTip {{ background:{c["surface.raised"]}; color:{c["text.body"]}; border:1px solid {c["border.control"]}; padding:6px; }}
    QMenu::item {{ padding:7px 20px; border-radius:4px; }}
    QMenu::item:selected {{ background:{c["surface.selected"]}; color:{c["text.heading"]}; }}
    QGroupBox {{ margin-top:12px; padding:16px 12px 12px; }}
    QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; color:{c["text.heading"]}; }}
    QFrame[worker="true"] {{ background:{c["surface.raised"]}; border:1px solid {c["border.subtle"]}; border-radius:10px; }}
    QFrame[worker="true"] QLabel[kind="section"] {{ font-size:13px; }}
    QFrame[worker="true"][selected="true"] {{ border:1px solid {c["accent.default"]}; background:{c["surface.selected"]}; }}
    QFrame[worker="true"]:focus {{ border:2px solid {c["focus.ring"]}; }}
    QFrame[robotCard="true"], QFrame[worldCard="true"] {{ background:{c["surface.raised"]}; border:1px solid {c["border.subtle"]}; border-radius:12px; }}
    QFrame[robotCard="true"]:hover, QFrame[worldCard="true"]:hover {{ border-color:{c["border.control"]}; }}
    QFrame[robotCard="true"][selected="true"], QFrame[worldCard="true"][selected="true"] {{ background:{c["surface.selected"]}; border-color:{c["text.heading"]}; }}
    QFrame[robotCard="true"][focused="true"], QFrame[worldCard="true"][focused="true"] {{ border:2px solid {c["focus.ring"]}; }}
    QFrame[robotCard="true"] QLabel[kind="section"] {{ font-size:14px; }}
    QFrame[robotCard="true"] QCheckBox::indicator:focus {{ border:1px solid {c["border.control"]}; width:16px; height:16px; }}
    QFrame#robotProfilesPopup, QFrame#worldSpawnsPopup {{ background:{c["surface.raised"]}; border:1px solid {c["border.control"]}; border-radius:10px; }}
    QFrame[robotProfileRow="true"] {{ border:0; border-radius:6px; background:transparent; }}
    QFrame[robotProfileRow="true"][selected="true"] {{ background:{c["surface.selected"]}; }}
    """
