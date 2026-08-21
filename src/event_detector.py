"""Temporal rule engine: person puts a box into a bag.

Causal sequence (required)
--------------------------
1. HOLDING   — person is associated with a box for several frames, and that
               box is still *outside* the bag (low containment).
2. NEAR_BAG  — that same person is also within reach of a bag.
3. INSERTING — the held box's overlap with the bag rises from low → high.
4. EVENT     — insertion persists for min_insert_frames, *or* the box track
               disappears while its last pose was inside the bag.

A box that is already sitting in a bag never fires: the engine remembers the
minimum box-in-bag containment observed while the person was holding it.

The held-box ID is locked during insertion so the association is not lost when
the box leaves the hands and enters the bag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.association import hold_score, insert_score, near_bag_score
from src.geometry import containment, expand
from src.tracker import Track


IDLE = "idle"
HOLDING = "holding_box"
NEAR_BAG = "near_bag"
INSERTING = "inserting"
EVENT = "put_box_in_bag"


@dataclass
class PutBoxInBagEvent:
    frame_idx: int
    timestamp: float
    person_id: int
    box_id: int | None
    bag_id: int | None
    hold_score: float
    near_bag_score: float
    insert_score: float
    containment: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": EVENT,
            "frame": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
            "person_id": self.person_id,
            "box_id": self.box_id,
            "bag_id": self.bag_id,
            "hold_score": round(self.hold_score, 3),
            "near_bag_score": round(self.near_bag_score, 3),
            "insert_score": round(self.insert_score, 3),
            "containment": round(self.containment, 3),
            "reason": self.reason,
        }


@dataclass
class PersonState:
    person_id: int
    state: str = IDLE
    held_box_id: int | None = None
    bag_id: int | None = None
    hold_frames: int = 0
    near_bag_frames: int = 0
    insert_frames: int = 0
    missing_in_bag_frames: int = 0
    hold_score: float = 0.0
    near_score: float = 0.0
    insert_score: float = 0.0
    min_containment_while_held: float = 1.0
    last_box_bbox: tuple[float, float, float, float] | None = None
    cooldown: int = 0
    last_event_frame: int | None = None


@dataclass
class RuleEngine:
    hold_score_threshold: float = 0.50
    min_hold_frames: int = 8
    near_bag_score_threshold: float = 0.40
    min_near_bag_frames: int = 5
    insert_score_threshold: float = 0.45
    outside_containment: float = 0.20
    min_insert_frames: int = 6
    event_cooldown_frames: int = 60
    _people: dict[int, PersonState] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RuleEngine":
        keys = (
            "hold_score_threshold",
            "min_hold_frames",
            "near_bag_score_threshold",
            "min_near_bag_frames",
            "insert_score_threshold",
            "outside_containment",
            "min_insert_frames",
            "event_cooldown_frames",
        )
        return cls(**{k: cfg[k] for k in keys if k in cfg})

    def states(self) -> dict[int, PersonState]:
        return self._people

    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        timestamp: float,
    ) -> list[PutBoxInBagEvent]:
        persons = [t for t in tracks if t.cls == "person" and t.missed == 0]
        boxes = [t for t in tracks if t.cls == "box"]
        bags = [t for t in tracks if t.cls == "bag" and t.missed == 0]
        events: list[PutBoxInBagEvent] = []

        live_ids = {p.track_id for p in persons}
        for person in persons:
            ctx = self._people.setdefault(person.track_id, PersonState(person_id=person.track_id))
            event = self._step_person(ctx, person, boxes, bags, frame_idx, timestamp)
            if event is not None:
                events.append(event)

        for pid, ctx in list(self._people.items()):
            if pid in live_ids:
                continue
            ctx.hold_frames = 0
            ctx.near_bag_frames = 0
            ctx.insert_frames = 0
            ctx.missing_in_bag_frames = 0
            if ctx.cooldown > 0:
                ctx.cooldown -= 1
                ctx.state = EVENT
                if ctx.cooldown == 0:
                    ctx.state = IDLE
                    ctx.held_box_id = None
                    ctx.bag_id = None
            elif ctx.state != EVENT:
                ctx.state = IDLE
                ctx.held_box_id = None
                ctx.bag_id = None

        return events

    def _step_person(
        self,
        ctx: PersonState,
        person: Track,
        boxes: list[Track],
        bags: list[Track],
        frame_idx: int,
        timestamp: float,
    ) -> PutBoxInBagEvent | None:
        if ctx.cooldown > 0:
            ctx.cooldown -= 1
            ctx.state = EVENT
            if ctx.cooldown == 0:
                ctx.state = IDLE
            return None

        held_box, h_score, box_visible = self._resolve_held_box(ctx, person, boxes, bags)
        nearby_bag, n_score = self._resolve_bag(person, bags)

        ctx.hold_score = h_score
        ctx.near_score = n_score

        if held_box is None:
            ctx.hold_frames = 0
            ctx.held_box_id = None
            ctx.min_containment_while_held = 1.0
            ctx.missing_in_bag_frames = 0
        else:
            ctx.held_box_id = held_box.track_id
            ctx.last_box_bbox = held_box.bbox
            if box_visible:
                ctx.hold_frames += 1

        if nearby_bag is None:
            ctx.near_bag_frames = 0
            ctx.bag_id = None
        else:
            ctx.bag_id = nearby_bag.track_id
            ctx.near_bag_frames += 1

        box_in_bag = 0.0
        i_score = 0.0
        if held_box is not None and nearby_bag is not None:
            i_score = insert_score(held_box, nearby_bag)
            box_in_bag = containment(held_box.bbox, expand(nearby_bag.bbox, 0.12))
            ctx.insert_score = i_score
            if box_visible:
                ctx.min_containment_while_held = min(ctx.min_containment_while_held, box_in_bag)
        elif held_box is not None and box_visible:
            ctx.min_containment_while_held = min(ctx.min_containment_while_held, 0.0)
            ctx.insert_score = 0.0
        else:
            ctx.insert_score = i_score

        was_outside = ctx.min_containment_while_held <= self.outside_containment
        held_long_enough = ctx.hold_frames >= self.min_hold_frames
        near_long_enough = ctx.near_bag_frames >= self.min_near_bag_frames

        inserting_now = (
            held_box is not None
            and nearby_bag is not None
            and box_visible
            and i_score >= self.insert_score_threshold
            and was_outside
            and held_long_enough
            and near_long_enough
        )

        disappeared_in_bag = (
            held_box is not None
            and nearby_bag is not None
            and not box_visible
            and was_outside
            and held_long_enough
            and near_long_enough
            and containment(held_box.bbox, expand(nearby_bag.bbox, 0.20)) >= 0.35
        )

        if inserting_now:
            ctx.insert_frames += 1
            ctx.missing_in_bag_frames = 0
            ctx.state = INSERTING
        elif disappeared_in_bag:
            ctx.missing_in_bag_frames += 1
            ctx.state = INSERTING
        else:
            ctx.insert_frames = 0
            ctx.missing_in_bag_frames = 0
            if held_long_enough and nearby_bag is not None and near_long_enough:
                ctx.state = NEAR_BAG
            elif held_long_enough:
                ctx.state = HOLDING
            else:
                ctx.state = IDLE

        if ctx.insert_frames >= self.min_insert_frames:
            return self._emit(ctx, frame_idx, timestamp, box_in_bag, "box_entered_bag")
        if ctx.missing_in_bag_frames >= max(2, self.min_insert_frames // 2):
            return self._emit(
                ctx,
                frame_idx,
                timestamp,
                containment(held_box.bbox, expand(nearby_bag.bbox, 0.20)) if held_box and nearby_bag else box_in_bag,
                "box_disappeared_in_bag",
            )
        return None

    def _resolve_held_box(
        self,
        ctx: PersonState,
        person: Track,
        boxes: list[Track],
        bags: list[Track],
    ) -> tuple[Track | None, float, bool]:
        """Return (box, hold_score, visible). Lock onto the previous box while inserting."""
        by_id = {b.track_id: b for b in boxes}

        if ctx.held_box_id is not None and ctx.held_box_id in by_id:
            prev = by_id[ctx.held_box_id]
            score = hold_score(person, prev)
            visible = prev.missed == 0
            if not visible:
                return prev, score, False
            inserting = False
            if bags:
                inserting = max(insert_score(prev, bag) for bag in bags) >= self.insert_score_threshold * 0.5
            if score >= self.hold_score_threshold * 0.45 or inserting or ctx.state == INSERTING:
                return prev, score, True

        best: Track | None = None
        best_score = -1.0
        for box in boxes:
            if box.missed != 0:
                continue
            score = hold_score(person, box)
            if score > best_score:
                best, best_score = box, score
        if best is not None and best_score >= self.hold_score_threshold:
            return best, best_score, True
        return None, 0.0, False

    def _resolve_bag(self, person: Track, bags: list[Track]) -> tuple[Track | None, float]:
        best: Track | None = None
        best_score = -1.0
        for bag in bags:
            score = near_bag_score(person, bag)
            if score > best_score:
                best, best_score = bag, score
        if best is not None and best_score >= self.near_bag_score_threshold:
            return best, best_score
        return None, 0.0

    def _emit(
        self,
        ctx: PersonState,
        frame_idx: int,
        timestamp: float,
        box_in_bag: float,
        reason: str,
    ) -> PutBoxInBagEvent:
        ctx.state = EVENT
        ctx.cooldown = self.event_cooldown_frames
        ctx.last_event_frame = frame_idx
        ctx.insert_frames = 0
        ctx.missing_in_bag_frames = 0
        ctx.min_containment_while_held = 1.0
        event = PutBoxInBagEvent(
            frame_idx=frame_idx,
            timestamp=timestamp,
            person_id=ctx.person_id,
            box_id=ctx.held_box_id,
            bag_id=ctx.bag_id,
            hold_score=ctx.hold_score,
            near_bag_score=ctx.near_score,
            insert_score=ctx.insert_score,
            containment=box_in_bag,
            reason=reason,
        )
        ctx.held_box_id = None
        return event
