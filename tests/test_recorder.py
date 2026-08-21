"""Recorder preroll / postroll tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.event_detector import PutBoxInBagEvent
from src.recorder import Recorder


def _event() -> PutBoxInBagEvent:
    return PutBoxInBagEvent(
        frame_idx=10,
        timestamp=1.0,
        person_id=1,
        box_id=2,
        bag_id=3,
        hold_score=0.9,
        near_bag_score=0.8,
        insert_score=0.7,
        containment=0.6,
        reason="box_entered_bag",
    )


class RecorderTests(unittest.TestCase):
    def test_event_clip_includes_preroll_and_postroll(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(
                {
                    "enabled": True,
                    "event_clips": True,
                    "continuous": False,
                    "preroll_seconds": 0.3,
                    "postroll_seconds": 0.2,
                    "annotated": True,
                },
                tmp,
                fps=10,
                frame_size=(64, 48),
            )
            for i in range(8):
                frame = np.full((48, 64, 3), i, dtype=np.uint8)
                recorder.push(frame)
            path = recorder.trigger(_event())
            self.assertIsNotNone(path)
            for i in range(5):
                frame = np.full((48, 64, 3), 200, dtype=np.uint8)
                recorder.push(frame)
            recorder.close()

            clip = Path(path)
            self.assertTrue(clip.exists())
            self.assertGreater(clip.stat().st_size, 0)
            cap = cv2.VideoCapture(str(clip))
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            cap.release()
            # 3 preroll (buffer maxlen) + 2 postroll, with a little encoder slack
            self.assertGreaterEqual(count, 4)
            self.assertLessEqual(count, 8)

    def test_disabled_recorder_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(
                {"enabled": False, "event_clips": True, "continuous": True},
                tmp,
                fps=10,
                frame_size=(64, 48),
            )
            recorder.push(np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertIsNone(recorder.trigger(_event()))
            recorder.close()
            self.assertEqual(list(Path(tmp).rglob("*.mp4")), [])

    def test_manual_toggle_writes_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(
                {
                    "enabled": True,
                    "event_clips": False,
                    "continuous": False,
                    "preroll_seconds": 0.2,
                },
                tmp,
                fps=10,
                frame_size=(64, 48),
            )
            for _ in range(4):
                recorder.push(np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertFalse(recorder.is_manual_recording)
            self.assertIsNone(recorder.toggle_manual())
            self.assertTrue(recorder.is_manual_recording)
            for _ in range(6):
                recorder.push(np.full((48, 64, 3), 80, dtype=np.uint8))
            path = recorder.toggle_manual()
            self.assertFalse(recorder.is_manual_recording)
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
