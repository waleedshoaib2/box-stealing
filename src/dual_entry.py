"""Temporal rule engine: dual entry — take a parcel, leave, come back for another.

Causal sequence (required)
--------------------------
1. ENTERED   — a person is in the room.
2. CARRYING  — they hold / pick a parcel this visit.
3. EXITED    — they leave, and the held parcel is no longer in the room
              (a completed takeaway).
4. REENTERED — someone comes back within the reentry window.
5. EVENT     — they pick another parcel. Alert: dual_entry.

Track IDs usually reset after the person walks out, so the cycle is room-level,
not per-ID. Two pickups in one visit (without leaving) do not fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.association import hold_score, interact_score
from src.tracker import Track


IDLE = "idle"
ENTERED = "entered"
INTERACTING = "interacting"
CARRYING = "carrying"
EXITED = "exited"
REENTERED = "reentered"
EVENT = "dual_entry"


@dataclass
class DualEntryEvent:
    frame_idx: int
    timestamp: float
    person_id: int
    box_id: int | None
    visit: int
    takeaways: int
    hold_score: float
    reason: str
    event: str = EVENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": EVENT,
            "frame": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "person_id": self.person_id,
            "box_id": self.box_id,
            "visit": self.visit,
            "takeaways": self.takeaways,
            "hold_score": round(self.hold_score, 3),
            "reason": self.reason,
        }


@dataclass
class DualPersonState:
    person_id: int
    state: str = IDLE
    hold_score: float = 0.0
    interact_score: float = 0.0
    visit: int = 0
    takeaways: int = 0
    interactions: int = 0
    carries: int = 0
    box_id: int | None = None


@dataclass
class DualEntryEngine:
    hold_score_threshold: float = 0.50
    min_hold_frames: int = 10
    interact_score_threshold: float = 0.45
    min_interact_frames: int = 8
    min_present_frames: int = 8
    exit_confirm_frames: int = 45
    exit_confirm_seconds: float = 1.5
    reentry_window_seconds: float = 180.0
    event_cooldown_frames: int = 90
    _people: dict[int, DualPersonState] = field(default_factory=dict)
    present: bool = False
    absent_frames: int = 0
    absent_since: float | None = None
    present_frames: int = 0
    visits: int = 0
    takeaways: int = 0
    interactions: int = 0
    carries: int = 0
    picked_this_visit: bool = False
    interacted_this_visit: bool = False
    carried_this_visit: bool = False
    fired_this_visit: bool = False
    hold_frames: int = 0
    interact_frames: int = 0
    hold_score_value: float = 0.0
    interact_score_value: float = 0.0
    held_box_id: int | None = None
    last_person_id: int | None = None
    last_takeaway_time: float | None = None
    cooldown: int = 0
    state: str = IDLE

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DualEntryEngine":
        keys = (
            "hold_score_threshold",
            "min_hold_frames",
            "interact_score_threshold",
            "min_interact_frames",
            "min_present_frames",
            "exit_confirm_frames",
            "exit_confirm_seconds",
            "reentry_window_seconds",
            "event_cooldown_frames",
        )
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    def states(self) -> dict[int, DualPersonState]:
        return self._people

    def status_line(self) -> str:
        return (
            f"visits {self.visits}  interacts {self.interactions}  "
            f"carries {self.carries}  takeaways {self.takeaways}  {self.state}"
        )

    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp: float,
    ) -> list[DualEntryEvent]:
        persons = [t for t in tracks if t.cls == "person" and t.missed == 0]
        boxes = [t for t in tracks if t.cls in ("box", "open_box")]
        visible_box_ids = {b.track_id for b in boxes if b.missed == 0}
        events: list[DualEntryEvent] = []

        if self.cooldown > 0:
            self.cooldown -= 1

        if persons:
            events.extend(self._on_present(persons, boxes, frame_idx, timestamp))
        else:
            event = self._on_absent(visible_box_ids, timestamp)
            if event is not None:
                events.append(event)

        live = {p.track_id for p in persons}
        for pid in list(self._people):
            if pid not in live:
                del self._people[pid]
        return events

    def _on_present(
        self,
        persons: list[Track],
        boxes: list[Track],
        frame_idx: int,
        timestamp: float,
    ) -> list[DualEntryEvent]:
        events: list[DualEntryEvent] = []
        person = max(persons, key=lambda t: t.conf)
        self.last_person_id = person.track_id
        self.absent_frames = 0
        self.absent_since = None

        if not self.present:
            self._begin_visit(timestamp)
        self.present = True
        self.present_frames += 1

        best_box, hold = _best_hold(person, boxes)
        interact_box, interact = _best_interact(person, boxes)
        self.hold_score_value = hold
        self.interact_score_value = interact
        if interact_box is not None and interact >= self.interact_score_threshold:
            self.interact_frames += 1
            if self.held_box_id is None:
                self.held_box_id = interact_box.track_id
        else:
            self.interact_frames = 0
        if best_box is not None and hold >= self.hold_score_threshold:
            self.hold_frames += 1
            self.held_box_id = best_box.track_id
        else:
            self.hold_frames = 0

        if self.interact_frames >= self.min_interact_frames:
            if not self.interacted_this_visit:
                self.interacted_this_visit = True
                self.interactions += 1
            if self.state in (ENTERED, REENTERED, IDLE):
                self.state = INTERACTING

        if self.hold_frames >= self.min_hold_frames:
            if not self.carried_this_visit:
                self.carried_this_visit = True
                self.carries += 1
            self.picked_this_visit = True
            if self.state in (ENTERED, REENTERED, INTERACTING, IDLE):
                self.state = CARRYING
            if (
                self.takeaways >= 1
                and self.visits >= 2
                and not self.fired_this_visit
                and self.cooldown == 0
            ):
                events.append(self._emit(person, frame_idx, timestamp, "second_pickup"))

        ctx = self._people.setdefault(person.track_id, DualPersonState(person_id=person.track_id))
        ctx.state = self.state
        ctx.hold_score = self.hold_score_value
        ctx.interact_score = self.interact_score_value
        ctx.visit = self.visits
        ctx.takeaways = self.takeaways
        ctx.interactions = self.interactions
        ctx.carries = self.carries
        ctx.box_id = self.held_box_id
        return events

    def _begin_visit(self, timestamp: float) -> None:
        self.present_frames = 0
        self.hold_frames = 0
        self.interact_frames = 0
        self.picked_this_visit = False
        self.interacted_this_visit = False
        self.carried_this_visit = False
        self.fired_this_visit = False
        self.held_box_id = None
        if self.takeaways >= 1 and self._within_reentry_window(timestamp):
            self.visits += 1
            self.state = REENTERED
            return
        if self.takeaways == 0:
            self.visits = 1
            self.state = ENTERED
            return
        self.visits = 1
        self.takeaways = 0
        self.interactions = 0
        self.carries = 0
        self.last_takeaway_time = None
        self.state = ENTERED

    def _on_absent(self, visible_box_ids: set[int], timestamp: float) -> DualEntryEvent | None:
        if not self.present:
            if self.state not in (EXITED, EVENT):
                self.state = IDLE
            return None
        self.absent_frames += 1
        if self.absent_since is None:
            self.absent_since = timestamp
        if self.absent_frames < self.exit_confirm_frames:
            return None
        if self.exit_confirm_seconds > 0 and (timestamp - self.absent_since) < self.exit_confirm_seconds:
            return None
        self.present = False
        self.absent_since = None
        too_brief = self.present_frames < self.min_present_frames
        held_gone = self.held_box_id is None or self.held_box_id not in visible_box_ids
        took_parcel = (not too_brief) and self.picked_this_visit and held_gone
        if took_parcel:
            self.takeaways += 1
            self.last_takeaway_time = timestamp
            self.state = EXITED
        else:
            if self.takeaways >= 1:
                self.state = EXITED
            else:
                self.state = IDLE
                self.visits = 0
        self.hold_frames = 0
        self.interact_frames = 0
        self.picked_this_visit = False
        self.present_frames = 0
        return None

    def _within_reentry_window(self, timestamp: float) -> bool:
        if self.last_takeaway_time is None:
            return False
        return (timestamp - self.last_takeaway_time) <= self.reentry_window_seconds

    def _emit(self, person: Track, frame_idx: int, timestamp: float, reason: str) -> DualEntryEvent:
        self.state = EVENT
        self.fired_this_visit = True
        self.cooldown = self.event_cooldown_frames
        event = DualEntryEvent(
            frame_idx=frame_idx,
            timestamp=timestamp,
            person_id=person.track_id,
            box_id=self.held_box_id,
            visit=self.visits,
            takeaways=self.takeaways,
            hold_score=self.hold_score_value,
            reason=reason,
        )
        ctx = self._people.setdefault(person.track_id, DualPersonState(person_id=person.track_id))
        ctx.state = EVENT
        ctx.hold_score = self.hold_score_value
        ctx.interact_score = self.interact_score_value
        ctx.visit = self.visits
        ctx.takeaways = self.takeaways
        ctx.interactions = self.interactions
        ctx.carries = self.carries
        ctx.box_id = self.held_box_id
        return event


def _best_hold(person: Track, boxes: list[Track]) -> tuple[Track | None, float]:
    best: Track | None = None
    best_score = -1.0
    for box in boxes:
        if box.missed != 0:
            continue
        score = hold_score(person, box)
        if score > best_score:
            best, best_score = box, score
    if best is None:
        return None, 0.0
    return best, best_score


def _best_interact(person: Track, boxes: list[Track]) -> tuple[Track | None, float]:
    best: Track | None = None
    best_score = -1.0
    for box in boxes:
        if box.missed != 0:
            continue
        score = interact_score(person, box)
        if score > best_score:
            best, best_score = box, score
    if best is None:
        return None, 0.0
    return best, best_score
