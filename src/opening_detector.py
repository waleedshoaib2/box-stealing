"""Temporal rule engine: a person opens a box.

Causal sequence (required)
--------------------------
1. INTERACTING — person is working on a box (hands at the top, or carrying it)
                 for several frames while the box still looks *closed*.
2. OPENING     — that same box gets taller (flaps up), YOLO flips to open_box,
                 or a lid lifts off the top.
3. EVENT       — the opening cue holds for min_open_frames.

A box that is already open when the person walks up does not fire: the engine
records a closed baseline height / closed label first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.association import height_growth, interact_score, is_open_box, lid_lift_score
from src.geometry import height
from src.tracker import Track


IDLE = "idle"
INTERACTING = "interacting"
OPENING = "opening"
EVENT = "open_box"


@dataclass
class OpenBoxEvent:
    frame_idx: int
    timestamp: float
    person_id: int
    box_id: int | None
    interact_score: float
    growth: float
    lid_score: float
    reason: str
    event: str = EVENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": EVENT,
            "frame": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "person_id": self.person_id,
            "box_id": self.box_id,
            "interact_score": round(self.interact_score, 3),
            "growth": round(self.growth, 3),
            "lid_score": round(self.lid_score, 3),
            "reason": self.reason,
        }


@dataclass
class OpeningState:
    person_id: int
    state: str = IDLE
    box_id: int | None = None
    interact_frames: int = 0
    open_frames: int = 0
    interact_score: float = 0.0
    growth: float = 0.0
    lid_score: float = 0.0
    min_height: float = 1e9
    had_closed: bool = False
    last_box_bbox: tuple[float, float, float, float] | None = None
    cooldown: int = 0
    last_event_frame: int | None = None


@dataclass
class OpeningEngine:
    interact_score_threshold: float = 0.45
    min_interact_frames: int = 8
    height_growth_threshold: float = 0.20
    lid_score_threshold: float = 0.70
    min_open_frames: int = 5
    event_cooldown_frames: int = 60
    _people: dict[int, OpeningState] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "OpeningEngine":
        keys = (
            "interact_score_threshold",
            "min_interact_frames",
            "height_growth_threshold",
            "lid_score_threshold",
            "min_open_frames",
            "event_cooldown_frames",
        )
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    def states(self) -> dict[int, OpeningState]:
        return self._people

    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp: float,
    ) -> list[OpenBoxEvent]:
        persons = [t for t in tracks if t.cls == "person" and t.missed == 0]
        boxes = [t for t in tracks if t.cls in ("box", "open_box")]
        lids = [t for t in tracks if t.cls == "lid" and t.missed == 0]
        events: list[OpenBoxEvent] = []
        live = {p.track_id for p in persons}

        for person in persons:
            ctx = self._people.setdefault(person.track_id, OpeningState(person_id=person.track_id))
            event = self._step(ctx, person, boxes, lids, frame_idx, timestamp)
            if event is not None:
                events.append(event)

        for pid, ctx in list(self._people.items()):
            if pid in live:
                continue
            ctx.interact_frames = 0
            ctx.open_frames = 0
            if ctx.cooldown > 0:
                ctx.cooldown -= 1
                ctx.state = EVENT
                if ctx.cooldown == 0:
                    ctx.state = IDLE
                    ctx.box_id = None
            elif ctx.state != EVENT:
                ctx.state = IDLE
                ctx.box_id = None
        return events

    def _step(
        self,
        ctx: OpeningState,
        person: Track,
        boxes: list[Track],
        lids: list[Track],
        frame_idx: int,
        timestamp: float,
    ) -> OpenBoxEvent | None:
        if ctx.cooldown > 0:
            ctx.cooldown -= 1
            ctx.state = EVENT
            if ctx.cooldown == 0:
                ctx.state = IDLE
            return None

        target, score, visible = self._resolve_box(ctx, person, boxes)
        ctx.interact_score = score
        if target is None:
            ctx.interact_frames = 0
            ctx.open_frames = 0
            ctx.box_id = None
            ctx.min_height = 1e9
            ctx.had_closed = False
            ctx.growth = 0.0
            ctx.lid_score = 0.0
            ctx.state = IDLE
            return None

        ctx.box_id = target.track_id
        ctx.last_box_bbox = target.bbox
        if visible:
            ctx.interact_frames += 1
            if not is_open_box(target):
                ctx.had_closed = True
                ctx.min_height = min(ctx.min_height, height(target.bbox))

        baseline = ctx.min_height if ctx.min_height < 1e8 else height(target.bbox)
        ctx.growth = height_growth(target.bbox, baseline)
        ctx.lid_score = 0.0
        if lids:
            ctx.lid_score = max(lid_lift_score(target, lid) for lid in lids)

        long_enough = ctx.interact_frames >= self.min_interact_frames
        grew = ctx.growth >= self.height_growth_threshold
        became_open = is_open_box(target) and ctx.had_closed
        lid_up = ctx.lid_score >= self.lid_score_threshold and ctx.had_closed
        opening_now = visible and long_enough and ctx.had_closed and (grew or became_open or lid_up)

        if opening_now:
            ctx.open_frames += 1
            ctx.state = OPENING
        else:
            ctx.open_frames = 0
            ctx.state = INTERACTING if long_enough else IDLE

        if ctx.open_frames >= self.min_open_frames:
            reason = "box_grew_open" if grew else ("label_open_box" if became_open else "lid_lifted")
            return self._emit(ctx, frame_idx, timestamp, reason)
        return None

    def _resolve_box(
        self,
        ctx: OpeningState,
        person: Track,
        boxes: list[Track],
    ) -> tuple[Track | None, float, bool]:
        by_id = {b.track_id: b for b in boxes}
        if ctx.box_id is not None and ctx.box_id in by_id:
            prev = by_id[ctx.box_id]
            score = interact_score(person, prev)
            visible = prev.missed == 0
            if not visible:
                return prev, score, False
            if score >= self.interact_score_threshold * 0.5 or ctx.state == OPENING:
                return prev, score, True

        best: Track | None = None
        best_score = -1.0
        for box in boxes:
            if box.missed != 0:
                continue
            score = interact_score(person, box)
            if score > best_score:
                best, best_score = box, score
        if best is not None and best_score >= self.interact_score_threshold:
            return best, best_score, True
        return None, 0.0, False

    def _emit(self, ctx: OpeningState, frame_idx: int, timestamp: float, reason: str) -> OpenBoxEvent:
        ctx.state = EVENT
        ctx.cooldown = self.event_cooldown_frames
        ctx.last_event_frame = frame_idx
        ctx.open_frames = 0
        ctx.min_height = 1e9
        ctx.had_closed = False
        event = OpenBoxEvent(
            frame_idx=frame_idx,
            timestamp=timestamp,
            person_id=ctx.person_id,
            box_id=ctx.box_id,
            interact_score=ctx.interact_score,
            growth=ctx.growth,
            lid_score=ctx.lid_score,
            reason=reason,
        )
        ctx.box_id = None
        return event
