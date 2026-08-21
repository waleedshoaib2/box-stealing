"""On-screen REC button hit tests."""

from __future__ import annotations

import unittest

import numpy as np

from src.visualizer import RecordButton, draw_record_button


class RecordButtonTests(unittest.TestCase):
    def test_button_rect_is_top_right_and_clickable(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rect = draw_record_button(frame, recording=False)
        x1, y1, x2, y2 = rect
        self.assertGreater(x1, 1280 // 2)
        self.assertLess(y2, 120)
        button = RecordButton()
        button.rect = rect
        self.assertTrue(button.hit((x1 + x2) // 2, (y1 + y2) // 2))
        self.assertFalse(button.hit(10, 10))

    def test_recording_state_draws_stop_label(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_record_button(frame, recording=True, elapsed_s=65)
        # Red stop fill lives in the button region (top-right).
        self.assertGreater(int(frame[40, 600, 2]), 80)


if __name__ == "__main__":
    unittest.main()
