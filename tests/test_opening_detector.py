"""Synthetic tests for person-opens-box rules (no YOLO required)."""

from __future__ import annotations

import unittest

from src.association import interact_score
from src.opening_detector import EVENT, OpeningEngine
from src.tracker import Track


def box_track(track_id: int, xyxy: tuple[float, float, float, float], cls: str = "box") -> Track:
    return Track(track_id=track_id, cls=cls, bbox=xyxy, conf=0.9, history=[xyxy])


def person_track(track_id: int, xyxy: tuple[float, float, float, float]) -> Track:
    return Track(track_id=track_id, cls="person", bbox=xyxy, conf=0.9, history=[xyxy])


PERSON = (100.0, 50.0, 220.0, 400.0)
CLOSED_BOX = (130.0, 280.0, 200.0, 340.0)
OPEN_BOX = (125.0, 210.0, 205.0, 345.0)


def lerp(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, ...]:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class OpeningEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = OpeningEngine(
            interact_score_threshold=0.45,
            min_interact_frames=5,
            height_growth_threshold=0.20,
            min_open_frames=4,
            event_cooldown_frames=20,
        )

    def _run(self, frames: list[list[Track]]) -> list:
        events = []
        for i, tracks in enumerate(frames):
            events.extend(self.engine.update(tracks, i, i / 30.0))
        return events

    def test_interact_score_is_high_when_person_is_on_the_box(self) -> None:
        self.assertGreater(
            interact_score(person_track(1, PERSON), box_track(2, CLOSED_BOX)),
            0.45,
        )

    def test_opening_a_closed_box_emits_event(self) -> None:
        frames: list[list[Track]] = []
        for _ in range(8):
            frames.append([person_track(1, PERSON), box_track(2, CLOSED_BOX)])
        for i in range(8):
            t = (i + 1) / 8.0
            frames.append([person_track(1, PERSON), box_track(2, lerp(CLOSED_BOX, OPEN_BOX, t))])
        events = self._run(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, EVENT)
        self.assertEqual(events[0].person_id, 1)
        self.assertEqual(events[0].box_id, 2)
        self.assertGreater(events[0].growth, 0.20)

    def test_already_open_box_does_not_fire(self) -> None:
        frames = []
        for _ in range(20):
            frames.append([person_track(1, PERSON), box_track(2, OPEN_BOX, cls="open_box")])
        self.assertEqual(self._run(frames), [])

    def test_person_away_from_box_does_not_fire(self) -> None:
        far_person = (500.0, 50.0, 620.0, 400.0)
        frames = []
        for _ in range(8):
            frames.append([person_track(1, far_person), box_track(2, CLOSED_BOX)])
        for i in range(8):
            t = (i + 1) / 8.0
            frames.append([person_track(1, far_person), box_track(2, lerp(CLOSED_BOX, OPEN_BOX, t))])
        self.assertEqual(self._run(frames), [])

    def test_closed_label_then_open_label_fires(self) -> None:
        frames = []
        for _ in range(8):
            frames.append([person_track(1, PERSON), box_track(2, CLOSED_BOX)])
        for _ in range(6):
            frames.append([person_track(1, PERSON), box_track(2, OPEN_BOX, cls="open_box")])
        events = self._run(frames)
        self.assertTrue(any(e.reason == "open_box_detected" for e in events), events)
        self.assertEqual(self.engine.states()[1].state, EVENT)


if __name__ == "__main__":
    unittest.main()
