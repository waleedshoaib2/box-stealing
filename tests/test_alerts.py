"""Alert formatting tests (no live notification or network)."""

from __future__ import annotations

import unittest

from src.alerts import _webhook_body, format_alert
from src.event_detector import PutBoxInBagEvent
from src.opening_detector import OpenBoxEvent


def _event() -> PutBoxInBagEvent:
    return PutBoxInBagEvent(
        frame_idx=10,
        timestamp=1.5,
        person_id=1,
        box_id=2,
        bag_id=3,
        hold_score=0.9,
        near_bag_score=0.8,
        insert_score=0.7,
        containment=0.6,
        reason="box_entered_bag",
    )


class AlertFormatTests(unittest.TestCase):
    def test_open_box_message(self) -> None:
        event = OpenBoxEvent(
            frame_idx=1,
            timestamp=0.0,
            person_id=4,
            box_id=9,
            interact_score=1.0,
            growth=0.3,
            reason="open_box_detected",
        )
        self.assertEqual(format_alert(event), "Open box detected — person #4 (open_box_detected)")

    def test_message_includes_ids(self) -> None:
        text = format_alert(_event())
        self.assertIn("Person #1", text)
        self.assertIn("box#2", text)
        self.assertIn("bag#3", text)

    def test_slack_webhook_shape(self) -> None:
        body = _webhook_body("https://hooks.slack.com/services/T/B/X", {"a": 1}, "hello")
        self.assertIn("text", body)

    def test_discord_webhook_shape(self) -> None:
        body = _webhook_body("https://discord.com/api/webhooks/1/2", {"a": 1}, "hello")
        self.assertEqual(body["content"].startswith("🚨"), True)

    def test_generic_webhook_sends_payload(self) -> None:
        payload = {"event": "put_box_in_bag"}
        body = _webhook_body("https://example.com/hook", payload, "hello")
        self.assertEqual(body, payload)


if __name__ == "__main__":
    unittest.main()
