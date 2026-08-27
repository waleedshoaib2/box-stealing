"""Banner text must follow the last event, not fall back to the bag task."""

from __future__ import annotations

import unittest

from src.opening_detector import OpenBoxEvent
from src.visualizer import banner_label


class BannerTests(unittest.TestCase):
    def test_open_box_banner_stays_open_box_after_event_frame(self) -> None:
        event = OpenBoxEvent(
            frame_idx=167,
            timestamp=2.8,
            person_id=7,
            box_id=6,
            interact_score=0.6,
            growth=0.0,
            reason="open_box_detected",
        )
        kind, text = banner_label(event)
        self.assertEqual(kind, "open_box")
        self.assertTrue(text.startswith("PERSON OPENING BOX"))
        self.assertIn("person#7", text)

    def test_empty_banner_is_not_put_box_in_bag(self) -> None:
        kind, text = banner_label(None)
        self.assertEqual(kind, "")
        self.assertEqual(text, "ALERT")
        self.assertNotIn("BAG", text)


if __name__ == "__main__":
    unittest.main()
