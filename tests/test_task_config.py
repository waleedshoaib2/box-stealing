"""Task-flag helpers used by the CLI and the PyQt app."""

from __future__ import annotations

import unittest

from src.pipeline import TASK_LABELS, apply_task_to_cfg


class TaskConfigTests(unittest.TestCase):
    def test_labels_match_requested_behaviors(self) -> None:
        self.assertEqual(TASK_LABELS["bag"], "Put box in the bag")
        self.assertEqual(TASK_LABELS["dual_entry"], "Leaving and entering with packages")
        self.assertEqual(TASK_LABELS["open"], "Opening packages")

    def test_bag_enables_only_bag_rules(self) -> None:
        cfg = apply_task_to_cfg({}, "bag")
        self.assertTrue(cfg["rules"]["enabled"])
        self.assertFalse(cfg["opening"]["enabled"])
        self.assertFalse(cfg["dual_entry"]["enabled"])

    def test_open_enables_only_opening(self) -> None:
        cfg = apply_task_to_cfg({}, "open")
        self.assertFalse(cfg["rules"]["enabled"])
        self.assertTrue(cfg["opening"]["enabled"])
        self.assertFalse(cfg["dual_entry"]["enabled"])

    def test_dual_entry_enables_only_dual(self) -> None:
        cfg = apply_task_to_cfg({}, "dual_entry")
        self.assertFalse(cfg["rules"]["enabled"])
        self.assertFalse(cfg["opening"]["enabled"])
        self.assertTrue(cfg["dual_entry"]["enabled"])

    def test_dual_entry_prompts_are_person_and_parcel_only(self) -> None:
        cfg = apply_task_to_cfg({}, "dual_entry")
        prompts = cfg["detector"]["yolo_world"]["prompts"]
        self.assertEqual(set(prompts), {"person", "box"})
        self.assertIn("parcel", prompts["box"])
        self.assertNotIn("bag", prompts)
        self.assertNotIn("open_box", prompts)

    def test_open_prompts_include_open_box_not_bag(self) -> None:
        cfg = apply_task_to_cfg({}, "open")
        self.assertIn("open_box", cfg["detector"]["yolo_world"]["prompts"])
        self.assertNotIn("bag", cfg["detector"]["yolo_world"]["prompts"])

    def test_bag_prompts_include_bag_not_open_box(self) -> None:
        cfg = apply_task_to_cfg({}, "bag")
        self.assertIn("bag", cfg["detector"]["yolo_world"]["prompts"])
        self.assertNotIn("open_box", cfg["detector"]["yolo_world"]["prompts"])


if __name__ == "__main__":
    unittest.main()
