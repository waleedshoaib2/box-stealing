"""YOLO result parsing — no model download required."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.detector import YOLODetector, _class_name


class _CpuArray:
    def __init__(self, values: list) -> None:
        self._values = np.asarray(values)

    def cpu(self) -> _CpuArray:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


def _result(names: object, cls_id: int = 1) -> SimpleNamespace:
    boxes = SimpleNamespace(
        xyxy=_CpuArray([[10.0, 20.0, 30.0, 40.0]]),
        conf=_CpuArray([0.64]),
        cls=_CpuArray([cls_id]),
    )
    return SimpleNamespace(names=names, boxes=boxes)


class ClassNameTests(unittest.TestCase):
    def test_dict_names(self) -> None:
        self.assertEqual(_class_name({0: "person", 1: "box"}, 1), "box")

    def test_list_names_after_set_classes(self) -> None:
        self.assertEqual(_class_name(["person", "cardboard box", "parcel"], 1), "cardboard box")

    def test_missing_index_falls_back(self) -> None:
        self.assertEqual(_class_name(["person"], 9), "9")


class ParseResultsTests(unittest.TestCase):
    def test_list_names_map_to_canonical_box(self) -> None:
        detector = YOLODetector.__new__(YOLODetector)
        detector._world_label_map = {"person": "person", "cardboard box": "box", "parcel": "box"}
        detections = detector._parse_results([_result(["person", "cardboard box"], 1)], {})
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].cls, "box")
        self.assertEqual(detections[0].raw_label, "cardboard box")


if __name__ == "__main__":
    unittest.main()
