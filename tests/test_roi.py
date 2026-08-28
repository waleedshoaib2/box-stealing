"""ROI filter: person feet / box overlap inside the zone."""

from __future__ import annotations

import unittest

from src.roi import detection_in_roi, filter_detections, parse_roi, to_pixels
from src.tracker import Detection


def _det(cls: str, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(bbox=bbox, cls=cls, conf=0.9)


class RoiTests(unittest.TestCase):
    def test_parse_rejects_empty(self) -> None:
        self.assertIsNone(parse_roi([]))
        self.assertIsNone(parse_roi(None))

    def test_person_outside_roi_is_dropped(self) -> None:
        roi = (0.0, 0.0, 0.4, 1.0)
        person = _det("person", (500.0, 50.0, 600.0, 400.0))
        kept = filter_detections([person], roi, (1000, 500), classes=("person",))
        self.assertEqual(kept, [])

    def test_person_feet_inside_roi_is_kept(self) -> None:
        roi = (0.0, 0.0, 0.5, 1.0)
        person = _det("person", (100.0, 50.0, 200.0, 400.0))
        kept = filter_detections([person], roi, (1000, 500), classes=("person",))
        self.assertEqual(len(kept), 1)

    def test_box_overlapping_roi_is_kept(self) -> None:
        roi_px = to_pixels((0.0, 0.0, 0.5, 1.0), 1000, 500)
        box = _det("box", (400.0, 200.0, 520.0, 300.0))
        self.assertTrue(detection_in_roi(box, roi_px))

    def test_ungated_class_passes_through(self) -> None:
        roi = (0.0, 0.0, 0.2, 0.2)
        bag = _det("bag", (800.0, 400.0, 900.0, 480.0))
        kept = filter_detections([bag], roi, (1000, 500), classes=("person",))
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
