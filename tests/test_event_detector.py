"""Synthetic-box tests for the put-box-in-bag rule engine (no YOLO required)."""

from __future__ import annotations

import unittest

from src.association import hold_score, insert_score, near_bag_score
from src.event_detector import EVENT, HOLDING, INSERTING, NEAR_BAG, RuleEngine
from src.geometry import containment
from src.tracker import Track


def box_track(track_id: int, xyxy: tuple[float, float, float, float], missed: int = 0) -> Track:
    return Track(track_id=track_id, cls="box", bbox=xyxy, conf=0.9, missed=missed, history=[xyxy])


def bag_track(track_id: int, xyxy: tuple[float, float, float, float]) -> Track:
    return Track(track_id=track_id, cls="bag", bbox=xyxy, conf=0.9, history=[xyxy])


def person_track(track_id: int, xyxy: tuple[float, float, float, float]) -> Track:
    return Track(track_id=track_id, cls="person", bbox=xyxy, conf=0.9, history=[xyxy])


PERSON = (100.0, 50.0, 220.0, 400.0)
HELD_BOX = (130.0, 220.0, 190.0, 290.0)
FAR_BAG = (420.0, 300.0, 540.0, 430.0)
NEAR_BAG_BOX = (230.0, 280.0, 360.0, 420.0)
BOX_IN_BAG = (250.0, 300.0, 330.0, 390.0)


def lerp(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, ...]:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class AssociationTests(unittest.TestCase):
    def test_held_box_scores_high(self) -> None:
        person = person_track(1, PERSON)
        box = box_track(2, HELD_BOX)
        self.assertGreater(hold_score(person, box), 0.5)

    def test_far_bag_scores_low(self) -> None:
        person = person_track(1, PERSON)
        bag = bag_track(3, FAR_BAG)
        self.assertLess(near_bag_score(person, bag), 0.4)

    def test_box_inside_bag_has_high_insert_score(self) -> None:
        box = box_track(2, BOX_IN_BAG)
        bag = bag_track(3, NEAR_BAG_BOX)
        self.assertGreater(insert_score(box, bag), 0.45)
        self.assertGreater(containment(box.bbox, bag.bbox), 0.5)


class RuleEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RuleEngine(
            hold_score_threshold=0.50,
            min_hold_frames=5,
            near_bag_score_threshold=0.40,
            min_near_bag_frames=3,
            insert_score_threshold=0.45,
            outside_containment=0.20,
            min_insert_frames=4,
            event_cooldown_frames=30,
        )

    def _run(self, frames: list[list[Track]]) -> list:
        events = []
        for i, tracks in enumerate(frames):
            events.extend(self.engine.update(tracks, i, i / 30.0))
        return events

    def test_put_box_in_bag_emits_event(self) -> None:
        frames: list[list[Track]] = []
        # Carry the box, bag is far away.
        for _ in range(8):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, HELD_BOX),
                    bag_track(3, FAR_BAG),
                ]
            )
        # Walk the bag into reach while still holding the box.
        for _ in range(5):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, HELD_BOX),
                    bag_track(3, NEAR_BAG_BOX),
                ]
            )
        # Slide the box into the bag.
        for i in range(8):
            t = (i + 1) / 8.0
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, lerp(HELD_BOX, BOX_IN_BAG, t)),
                    bag_track(3, NEAR_BAG_BOX),
                ]
            )

        events = self._run(frames)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.person_id, 1)
        self.assertEqual(event.box_id, 2)
        self.assertEqual(event.bag_id, 3)
        self.assertEqual(event.reason, "box_entered_bag")
        self.assertEqual(self.engine.states()[1].state, EVENT)

    def test_box_already_in_bag_does_not_fire(self) -> None:
        frames = []
        for _ in range(20):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, BOX_IN_BAG),
                    bag_track(3, NEAR_BAG_BOX),
                ]
            )
        events = self._run(frames)
        self.assertEqual(events, [])

    def test_holding_without_bag_does_not_fire(self) -> None:
        frames = []
        for _ in range(20):
            frames.append([person_track(1, PERSON), box_track(2, HELD_BOX)])
        events = self._run(frames)
        self.assertEqual(events, [])
        self.assertEqual(self.engine.states()[1].state, HOLDING)

    def test_box_disappearing_inside_bag_fires(self) -> None:
        frames: list[list[Track]] = []
        for _ in range(8):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, HELD_BOX),
                    bag_track(3, FAR_BAG),
                ]
            )
        for _ in range(4):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, HELD_BOX),
                    bag_track(3, NEAR_BAG_BOX),
                ]
            )
        for _ in range(3):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, BOX_IN_BAG),
                    bag_track(3, NEAR_BAG_BOX),
                ]
            )
        # Box track is lost while last pose is inside the bag.
        for _ in range(6):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, BOX_IN_BAG, missed=3),
                    bag_track(3, NEAR_BAG_BOX),
                ]
            )

        events = self._run(frames)
        self.assertTrue(any(e.reason == "box_disappeared_in_bag" for e in events), events)

    def test_reaches_near_bag_state(self) -> None:
        frames = []
        for _ in range(8):
            frames.append(
                [
                    person_track(1, PERSON),
                    box_track(2, HELD_BOX),
                    bag_track(3, FAR_BAG),
                ]
            )
        events = self._run(frames)
        self.assertEqual(events, [])
        self.assertEqual(self.engine.states()[1].state, HOLDING)

        more = []
        for _ in range(4):
            more.append(
                [
                    person_track(1, PERSON),
                    box_track(2, HELD_BOX),
                    bag_track(3, NEAR_BAG_BOX),
                ]
            )
        events = self._run(more)
        self.assertEqual(events, [])
        self.assertIn(self.engine.states()[1].state, {NEAR_BAG, INSERTING})


if __name__ == "__main__":
    unittest.main()
