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

"""The Monitor's events (HANDOFF §8.12): one campaign's lifecycle events and the Events view.

``EventStream`` reads the campaign's history through ``events.read`` page by page when the campaign is first
shown, then the new events every second while the Monitor is visible, into a 2,000-event buffer. Engines
without the campaign scope (before milestone E4) answer with every event; those of other campaigns are left
out here. ``AttemptSlots`` names each event's attempt "W1 · 2" (worker slot and attempt number) from
``results.query`` and the active runs. The view shows the campaign or the selected run, filtered by event kind
and text, with an outcome badge on terminal events and an expand control on details that do not fit.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any
import uuid

from PySide6 import QtCore, QtGui, QtWidgets

from . import gui_icons
from .gui_controls import BadgeLook, KeySelect, SearchField, badge_look, paint_badge
from .gui_dialogs import wrap_text
from .gui_cells import PaintedColumn, PaintedTable
from .gui_logs import FeedView, MenuItems, copy_text
from .gui_style import cell_text, humanize
from .gui_support import Transport
from .gui_theme import ThemeManager, snap
from .gui_widgets import (
    BODY,
    BODY_MEDIUM,
    BODY_TABULAR,
    SegmentedControl,
    TextStyle,
    draw_text,
    plural,
    rounded_path,
)

BUFFER_EVENTS = 2000
PAGE_EVENTS = 500  # the engine's largest page


def event_detail(event: Mapping[str, Any]) -> str:
    """An event's detail: phase or state and the detail text ("Starting · Waiting for: …"); a sequential
    replacement says so. The outcome is the badge's (terminal events) and is not repeated."""
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return ""
    parts = []
    if event.get("kind") == "attempt.reserved" and payload.get("retry"):
        parts.append("sequential replacement")
    for key in ("phase", "state"):
        if payload.get(key):
            parts.append(cell_text(key, payload[key]))
    if payload.get("detail"):
        parts.append(str(payload["detail"]))
    if payload.get("direction"):
        parts.append(str(payload["direction"]))
    return " · ".join(parts)


def event_title(kind: str) -> str:
    return humanize(kind.replace(".", " "))


def event_outcome(event: Mapping[str, Any]) -> BadgeLook | None:
    """The outcome badge of a terminal event."""
    payload = event.get("payload")
    if event.get("kind") != "attempt.terminal" or not isinstance(payload, dict):
        return None
    outcome = payload.get("outcome")
    return badge_look("outcome", outcome) if isinstance(outcome, str) else None


class EventStream(QtCore.QObject):
    """One campaign's events (see the module docstring). ``changed`` follows every buffer change."""

    changed = QtCore.Signal()

    def __init__(self, transport: Transport, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.transport = transport
        self.campaign_id: str | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=BUFFER_EVENTS)
        self.after = 0
        self.pending = ""
        self.loaded = False  # the history has been read
        self.error = ""

    def set_campaign(self, campaign_id: str | None) -> None:
        """Follow another campaign; its events are read at the next ``poll``."""
        if campaign_id == self.campaign_id:
            return
        self.campaign_id = campaign_id
        self.events.clear()
        self.after, self.pending, self.loaded, self.error = 0, "", False, ""
        self.changed.emit()

    def poll(self) -> None:
        if self.campaign_id is None or self.pending:
            return
        self.pending = "monitor-events:" + uuid.uuid4().hex
        self.transport.submit(
            self.pending,
            "events.read",
            {
                "after_sequence": self.after,
                "limit": PAGE_EVENTS,
                "campaign_id": self.campaign_id,
            },
        )

    def received(self, tag: str, result: Any) -> bool:
        if not tag.startswith("monitor-events:"):
            return False
        if tag != self.pending:
            return True  # a reply for an earlier campaign
        self.pending, self.error = "", ""
        if result.get("snapshot_required"):
            # The log restarted below the cursor: read the campaign again.
            self.events.clear()
            self.after, self.loaded = 0, False
            self.changed.emit()
            self.poll()
            return True
        events = result.get("events") or []
        added = False
        for event in events:
            sequence = event.get("sequence")
            if not isinstance(sequence, int) or sequence <= self.after:
                continue
            if event.get("campaign_id") == self.campaign_id:
                self.events.append(event)
                added = True
        last = result.get("last_sequence")
        if isinstance(last, int) and last > self.after:
            self.after = last
        loaded = self.loaded
        if len(events) >= PAGE_EVENTS:
            self.poll()  # more history to read
        else:
            self.loaded = True
        if added or self.loaded != loaded:
            self.changed.emit()
        return True

    def failed(self, tag: str, error: str) -> bool:
        if not tag.startswith("monitor-events:"):
            return False
        if tag == self.pending:
            self.pending, self.error = "", error
            self.changed.emit()
        return True

    def latest_state(self, attempt_id: str) -> dict[str, Any] | None:
        """The payload of the attempt's latest ``attempt.state`` event in the buffer."""
        for event in reversed(self.events):
            if event.get("attempt_id") == attempt_id and event.get("kind") == "attempt.state":
                payload = event.get("payload")
                return payload if isinstance(payload, dict) else None
        return None


class AttemptSlots(QtCore.QObject):
    """Worker slot and attempt number of the campaign's attempts: from the active runs and from
    ``results.query`` (table ``attempts``), read again when an event names an attempt not yet known."""

    changed = QtCore.Signal()
    INTERVAL = 2.0  # s between reads

    def __init__(self, transport: Transport, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.transport = transport
        self.campaign_id: str | None = None
        self.slots: dict[str, tuple[int, int]] = {}
        self.pending = ""
        self.offset = 0
        self.last_read = -1e9

    def set_campaign(self, campaign_id: str | None) -> None:
        if campaign_id != self.campaign_id:
            self.campaign_id = campaign_id
            self.slots.clear()
            self.pending, self.offset, self.last_read = "", 0, -1e9

    def add(self, attempt: Mapping[str, Any]) -> None:
        slot, number = attempt.get("slot"), attempt.get("number")
        if isinstance(slot, int) and isinstance(number, int) and isinstance(attempt.get("id"), str):
            self.slots[attempt["id"]] = (slot, number)

    def label(self, attempt_id: str | None) -> str | None:
        """The worker and attempt ("W1 · 2"); None when the attempt is unknown (shown as "—")."""
        if attempt_id is None or attempt_id not in self.slots:
            return None
        slot, number = self.slots[attempt_id]
        return f"W{slot + 1} · {number}"

    def need(self, attempt_ids: set[str]) -> None:
        """Read the attempts again when some are unknown (at most every two seconds)."""
        if self.campaign_id is None or self.pending or not attempt_ids - set(self.slots):
            return
        if time.monotonic() - self.last_read < self.INTERVAL:
            return
        self.offset = 0
        self._request()

    def _request(self) -> None:
        self.pending = "monitor-attempts:" + uuid.uuid4().hex
        self.last_read = time.monotonic()
        self.transport.submit(
            self.pending,
            "results.query",
            {
                "table": "attempts",
                "filters": {"campaign_id": self.campaign_id},
                "offset": self.offset,
                "limit": 250,
                "summary": True,
            },
        )

    def received(self, tag: str, result: Any) -> bool:
        if not tag.startswith("monitor-attempts:"):
            return False
        if tag != self.pending:
            return True
        self.pending = ""
        for row in result.get("rows") or []:
            if isinstance(row, dict):
                self.add(row)
        following = result.get("next_offset")
        if isinstance(following, int) and following > self.offset:
            self.offset = following
            self._request()
        self.changed.emit()
        return True

    def failed(self, tag: str, error: str) -> bool:
        if not tag.startswith("monitor-attempts:"):
            return False
        if tag == self.pending:
            self.pending = ""
        return True


# ---------- the view ----------


@dataclass(frozen=True, slots=True)
class EventRow:
    sequence: int
    time: str
    worker: str | None  # "W1 · 2"; None for events without a known attempt
    title: str
    outcome: BadgeLook | None
    detail: str


EVENT_COLUMNS = (  # 11's ``colgroup``: 164, 112, 160, 160, the rest and 40 px
    PaintedColumn("Time", 164, 20),
    PaintedColumn("Worker · attempt", 112),
    PaintedColumn("Event", 160),
    PaintedColumn("Outcome", 160),
    PaintedColumn("Detail", None),
    PaintedColumn("", 40),
)
FAINT = TextStyle(13, 400, 15, tabular=True)  # ``span.faint`` (a middle-aligned inline box)
DETAIL_COLUMN = 4
TOGGLE = 28  # ``.btn.sm.ghost.icon``


class EventTable(PaintedTable):
    """``.evt .tbl``: 32 px rows with a bottom rule (none under the last); Time (``text.body``, tabular),
    Worker · attempt, Event (500 ink), Outcome (badge), Detail (``text.body``, elided) and a 28 px ghost
    ``cdown`` toggle on details that do not fit, which wraps the detail onto as many 18 px lines as it needs
    (``cup`` to close)."""

    def __init__(self, themes: ThemeManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(themes, EVENT_COLUMNS, parent, key="monitor.events")
        self.rows: list[EventRow] = []
        self.expanded: set[int] = set()
        self.hover_toggle = -1
        self.viewport().installEventFilter(self)

    def set_rows(self, rows: Sequence[EventRow]) -> None:
        self.rows = list(rows)
        self.rows_changed(len(self.rows))

    def detail_room(self) -> float:
        _, width = self.column_spans()[DETAIL_COLUMN]
        return max(0.0, width - 24)

    def long_detail(self, row: EventRow) -> bool:
        return BODY.advance(row.detail) > self.detail_room()

    def detail_lines(self, row: EventRow) -> list[str]:
        return wrap_text(row.detail, BODY, self.detail_room())

    def row_height(self, index: int) -> int:
        row = self.rows[index]
        if row.sequence in self.expanded and self.long_detail(row):
            return 32 + 18 * (len(self.detail_lines(row)) - 1)
        return 32

    def toggle_rect(self, index: int) -> QtCore.QRectF | None:
        """The expand control of a row whose detail does not fit (viewport coordinates)."""
        if not 0 <= index < len(self.rows) or not self.long_detail(self.rows[index]):
            return None
        left, width = self.column_spans()[-1]
        content = 32 - (0 if index == len(self.rows) - 1 else 1)  # the first line's cell
        top = self.row_rect(index).top() + (content - TOGGLE) / 2
        # A button's box (and its icon) is painted on whole pixels.
        return QtCore.QRectF(snap(left + width - 8 - TOGGLE), snap(top), TOGGLE, TOGGLE)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.viewport():
            kind = event.type()
            if kind == QtCore.QEvent.Type.MouseMove and isinstance(event, QtGui.QMouseEvent):
                index = self.row_at(event.position().y())
                rect = self.toggle_rect(index)
                hovered = index if rect is not None and rect.contains(event.position()) else -1
                if hovered != self.hover_toggle:
                    self.hover_toggle = hovered
                    self.viewport().update()
            elif kind == QtCore.QEvent.Type.Leave and self.hover_toggle != -1:
                self.hover_toggle = -1
                self.viewport().update()
            elif (
                kind == QtCore.QEvent.Type.MouseButtonPress
                and isinstance(event, QtGui.QMouseEvent)
                and event.button() == QtCore.Qt.MouseButton.LeftButton
            ):
                index = self.row_at(event.position().y())
                rect = self.toggle_rect(index)
                if rect is not None and rect.contains(event.position()):
                    self.toggle(index)
                    return True
        return super().eventFilter(watched, event)

    def toggle(self, index: int) -> None:
        sequence = self.rows[index].sequence
        if sequence in self.expanded:
            self.expanded.discard(sequence)
        else:
            self.expanded.add(sequence)
        value = self.verticalScrollBar().value()
        follow, self.follow = self.follow, False
        self.rows_changed(len(self.rows))
        self.follow = follow
        self.verticalScrollBar().setValue(value)

    def paint_row(self, painter: QtGui.QPainter, index: int, rect: QtCore.QRect) -> None:
        theme = self.themes.current
        row = self.rows[index]
        last = index == len(self.rows) - 1
        if not last:
            painter.fillRect(
                QtCore.QRect(rect.left(), rect.bottom(), rect.width(), 1),
                theme.color("border.line"),
            )
        content = 32 - (0 if last else 1)  # the first line's cell height
        top = rect.top()
        line_top = top + BODY.middle_baseline(content) - BODY.baseline(0)
        spans = self.column_spans()
        body = theme.color("text.body")
        x, width = spans[0]
        draw_text(
            painter, x + 20, line_top, BODY_TABULAR.elide(row.time, width - 32), BODY_TABULAR, body
        )
        x, width = spans[1]
        if row.worker is not None:
            draw_text(painter, x + 12, line_top, row.worker, BODY_TABULAR, body)
        else:
            dash_top = top + BODY.middle_top(content, 15)
            draw_text(painter, x + 12, dash_top, "—", FAINT, theme.color("text.tertiary"))
        x, width = spans[2]
        title = BODY_MEDIUM.elide(row.title, width - 24)
        draw_text(painter, x + 12, line_top, title, BODY_MEDIUM, theme.color("text.ink"))
        x, width = spans[3]
        if row.outcome is not None:
            paint_badge(painter, x + 12, top + BODY.middle_top(content, 22), row.outcome, theme)
        x, width = spans[DETAIL_COLUMN]
        if row.detail:
            if row.sequence in self.expanded and self.long_detail(row):
                for number, text in enumerate(self.detail_lines(row)):
                    draw_text(painter, x + 12, line_top + 18 * number, text, BODY, body)
            else:
                draw_text(painter, x + 12, line_top, BODY.elide(row.detail, width - 24), BODY, body)
        toggle = self.toggle_rect(index)
        if toggle is not None:
            if self.hover_toggle == index:
                painter.save()
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(theme.color("surface.hover"))
                painter.drawPath(rounded_path(toggle, 6))
                painter.restore()
            icon = "cup" if row.sequence in self.expanded else "cdown"
            gui_icons.draw(painter, icon, body, QtCore.QPointF(toggle.x() + 6, toggle.y() + 6))

    def tooltip_at(self, index: int, position: QtCore.QPoint) -> str:
        row = self.rows[index]
        toggle = self.toggle_rect(index)
        if toggle is not None and toggle.contains(QtCore.QPointF(position)):
            return (
                "Hide the full detail" if row.sequence in self.expanded else "Show the full detail"
            )
        # A cut detail in full over its cell.
        left, width = self.column_spans()[DETAIL_COLUMN]
        cut = self.long_detail(row) and row.sequence not in self.expanded
        return row.detail if cut and left <= position.x() < left + width else ""


class EventView(FeedView):
    """The Events view of the run card (``11``): the scope (Campaign or This run), the event kind and the text
    filter over the event table."""

    def __init__(
        self,
        themes: ThemeManager,
        stream: EventStream,
        slots: AttemptSlots,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        table = EventTable(themes)
        super().__init__(themes, table, parent)
        self.event_table = table
        self.stream, self.slots = stream, slots
        self.attempt_id: str | None = None
        self.shown: list[EventRow] = []
        self.scope = SegmentedControl(
            themes, [("Campaign", None), ("This run", None)], segment_width=86
        )
        self.scope.changed.connect(self.refresh)
        self.kind = KeySelect(themes, "Event")
        self.kind.addItem("All", None)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.search = SearchField(themes, "Filter events")
        self.search.textChanged.connect(self.refresh)
        self.add_filters((self.scope, 180), (self.kind, 180), (self.search, 260))
        stream.changed.connect(self.refresh)
        slots.changed.connect(self.refresh)
        self.refresh()

    def set_attempt(self, attempt_id: str | None) -> None:
        if attempt_id != self.attempt_id:
            self.attempt_id = attempt_id
            self.refresh()

    def scope_events(self) -> list[dict[str, Any]]:
        if self.scope.index() == 1:
            return [e for e in self.stream.events if e.get("attempt_id") == self.attempt_id]
        return list(self.stream.events)

    def refresh(self, *_: object) -> None:
        events = self.scope_events()
        self._fill_kinds(events)
        self.slots.need({e["attempt_id"] for e in events if isinstance(e.get("attempt_id"), str)})
        wanted = self.kind.currentData()
        needle = self.search.text().casefold()
        rows = []
        for event in events:
            if wanted is not None and event.get("kind") != wanted:
                continue
            row = EventRow(
                sequence=int(event["sequence"]),
                time=cell_text("utc_ns", event.get("utc_ns")),
                worker=self.slots.label(event.get("attempt_id")),
                title=event_title(str(event.get("kind", ""))),
                outcome=event_outcome(event),
                detail=event_detail(event),
            )
            if needle and needle not in self.row_text(row).casefold():
                continue
            rows.append(row)
        self.shown = rows
        self.event_table.set_rows(rows)
        place = "this run" if self.scope.index() == 1 else "this campaign"
        total = len(events)
        if self.stream.error:
            text = "Events unavailable: " + self.stream.error
        elif not self.stream.loaded and self.stream.campaign_id is not None:
            text = "Reading events…"
        elif len(rows) != total:
            text = f"{len(rows):,} of {plural(total, 'event')} in {place}"
        else:
            text = f"{plural(total, 'event')} in {place}"
        self.footer.set_text(text)

    def _fill_kinds(self, events: Sequence[Mapping[str, Any]]) -> None:
        """The kinds present in the scope, by name, after All; the chosen kind stays while it is present."""
        kinds = sorted({str(e.get("kind")) for e in events if e.get("kind")}, key=event_title)
        present = [self.kind.itemData(i) for i in range(1, self.kind.count())]
        if kinds == present:
            return
        current = self.kind.currentData()
        self.kind.blockSignals(True)
        while self.kind.count() > 1:
            self.kind.removeItem(1)
        for kind in kinds:
            self.kind.addItem(event_title(kind), kind)
        index = self.kind.findData(current) if current is not None else 0
        self.kind.setCurrentIndex(max(0, index))
        self.kind.blockSignals(False)

    @staticmethod
    def row_text(row: EventRow) -> str:
        """A row as copied: tab-separated columns."""
        outcome = row.outcome.word if row.outcome else ""
        return "\t".join([row.time, row.worker or "—", row.title, outcome, row.detail])

    def copy_shown(self) -> None:
        copy_text("\n".join(self.row_text(row) for row in self.shown))

    def menu_items(self, row: int) -> MenuItems:
        items: MenuItems = []
        if row >= 0:
            text = self.row_text(self.event_table.rows[row])
            items.append(("Copy event", "copy", lambda: copy_text(text), True))
        items.append(("Copy shown events", "copy", self.copy_shown, bool(self.shown)))
        return items
