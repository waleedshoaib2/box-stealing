"""Synthetic tests for dual-entry (take parcel, leave, come back for another)."""

from __future__ import annotations

import unittest

from src.dual_entry import EVENT, DualEntryEngine
from src.tracker import Track


def box_track(track_id: int, xyxy: tuple[float, float, float, float], missed: int = 0) -> Track:
    return Track(track_id=track_id, cls="box", bbox=xyxy, conf=0.9, missed=missed, history=[xyxy])


def person_track(track_id: int, xyxy: tuple[float, float, float, float]) -> Track:
    return Track(track_id=track_id, cls="person", bbox=xyxy, conf=0.9, history=[xyxy])


PERSON = (100.0, 50.0, 220.0, 400.0)
HELD_BOX = (130.0, 220.0, 190.0, 290.0)
FLOOR_BOX = (400.0, 300.0, 470.0, 380.0)
FAR_PERSON = (500.0, 50.0, 620.0, 400.0)


class DualEntryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DualEntryEngine(
            hold_score_threshold=0.50,
            min_hold_frames=5,
            min_present_frames=3,
            exit_confirm_frames=4,
            exit_confirm_seconds=0.0,
            reentry_window_seconds=30.0,
            event_cooldown_frames=20,
        )

    def _run(self, frames: list[list[Track]], start: int = 0) -> list:
        events = []
        for i, tracks in enumerate(frames):
            events.extend(self.engine.update(tracks, start + i, (start + i) / 30.0))
        return events

    def test_enter_take_leave_return_pick_fires_dual_entry(self) -> None:
        frames: list[list[Track]] = []
        # Visit 1: pick box 2, other box stays on the floor.
        for _ in range(8):
            frames.append([person_track(1, PERSON), box_track(2, HELD_BOX), box_track(3, FLOOR_BOX)])
        # Leave with box 2 (person gone, held box gone, floor box remains).
        for _ in range(6):
            frames.append([box_track(3, FLOOR_BOX)])
        # Visit 2: new track ID, pick the remaining parcel.
        for _ in range(8):
            frames.append([person_track(9, PERSON), box_track(3, HELD_BOX)])
        events = self._run(frames)
        self.assertTrue(any(e.event == EVENT for e in events), events)
        event = next(e for e in events if e.event == EVENT)
        self.assertEqual(event.reason, "second_pickup")
        self.assertEqual(event.person_id, 9)
        self.assertGreaterEqual(event.visit, 2)
        self.assertGreaterEqual(event.takeaways, 1)

    def test_leave_without_picking_then_pick_once_does_not_fire(self) -> None:
        frames: list[list[Track]] = []
        for _ in range(8):
            frames.append([person_track(1, FAR_PERSON), box_track(2, HELD_BOX)])
        for _ in range(6):
            frames.append([box_track(2, HELD_BOX)])
        for _ in range(8):
            frames.append([person_track(4, PERSON), box_track(2, HELD_BOX)])
        self.assertEqual(self._run(frames), [])

    def test_two_pickups_same_visit_without_leaving_does_not_fire(self) -> None:
        frames: list[list[Track]] = []
        for _ in range(8):
            frames.append([person_track(1, PERSON), box_track(2, HELD_BOX)])
        for _ in range(8):
            frames.append([person_track(1, PERSON), box_track(4, HELD_BOX)])
        self.assertEqual(self._run(frames), [])
        self.assertEqual(self.engine.takeaways, 0)

    def test_takeaway_without_return_does_not_fire(self) -> None:
        frames: list[list[Track]] = []
        for _ in range(8):
            frames.append([person_track(1, PERSON), box_track(2, HELD_BOX)])
        for _ in range(6):
            frames.append([])
        self.assertEqual(self._run(frames), [])
        self.assertEqual(self.engine.takeaways, 1)
        self.assertEqual(self.engine.state, "exited")

    def test_reentry_outside_window_resets(self) -> None:
        frames: list[list[Track]] = []
        for _ in range(8):
            frames.append([person_track(1, PERSON), box_track(2, HELD_BOX)])
        for _ in range(6):
            frames.append([])
        events = self._run(frames)
        self.assertEqual(events, [])
        # Jump far past the reentry window, then pick again.
        later = []
        for i in range(8):
            later.append([person_track(8, PERSON), box_track(5, HELD_BOX)])
        start = 8 + 6 + int(40 * 30)
        self.assertEqual(self._run(later, start=start), [])
        self.assertEqual(self.engine.takeaways, 0)

    def test_brief_dropout_is_not_an_exit(self) -> None:
        engine = DualEntryEngine(
            hold_score_threshold=0.50,
            min_hold_frames=5,
            min_present_frames=3,
            exit_confirm_frames=2,
            exit_confirm_seconds=1.0,
            reentry_window_seconds=30.0,
            event_cooldown_frames=20,
        )
        for i in range(8):
            engine.update([person_track(1, PERSON), box_track(2, HELD_BOX)], i, i / 30.0)
        for i in range(8, 11):
            engine.update([], i, i / 30.0)
        for i in range(11, 19):
            engine.update([person_track(1, PERSON), box_track(4, HELD_BOX)], i, i / 30.0)
        self.assertEqual(engine.takeaways, 0)
        self.assertEqual(engine.visits, 1)


if __name__ == "__main__":
    unittest.main()
