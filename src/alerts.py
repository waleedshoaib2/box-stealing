"""Alerts fired when a person puts a box into a bag.

Channels (all optional, configured in config.yaml → alerts):
  - console           always, if alerts.enabled
  - snapshot          annotated JPEG under outputs/alerts/
  - sound + desktop   macOS Notification Center (Sosumi)
  - webhook           generic HTTP POST (Slack / Discord / n8n / custom)
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.event_detector import PutBoxInBagEvent
from src.opening_detector import OpenBoxEvent


class Alerter:
    def __init__(self, cfg: dict[str, Any] | None, output_dir: str | Path) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.save_snapshot = bool(cfg.get("snapshot", True))
        self.sound = bool(cfg.get("sound", True))
        self.desktop = bool(cfg.get("desktop", True))
        self.webhook_url = str(cfg.get("webhook_url") or "").strip()
        self.cooldown_seconds = float(cfg.get("cooldown_seconds", 8))
        self.alert_dir = Path(output_dir) / "alerts"
        self._last_fire = 0.0

    def notify(
        self,
        event: PutBoxInBagEvent | OpenBoxEvent,
        frame: np.ndarray | None,
        extras: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        now = time.time()
        if now - self._last_fire < self.cooldown_seconds:
            return None
        self._last_fire = now

        message = format_alert(event)
        snapshot_path = self._write_snapshot(event, frame)
        print(f"ALERT: {message}" + (f"  snapshot={snapshot_path}" if snapshot_path else ""))

        if self.desktop or self.sound:
            threading.Thread(
                target=_macos_notify,
                args=(desktop_message(event), self.desktop, self.sound, alert_title(event)),
                daemon=True,
            ).start()
        if self.webhook_url:
            payload = event.to_dict()
            payload["message"] = message
            if snapshot_path is not None:
                payload["snapshot"] = str(snapshot_path)
            if extras:
                payload.update(extras)
            threading.Thread(
                target=_post_webhook,
                args=(self.webhook_url, payload, message),
                daemon=True,
            ).start()
        return snapshot_path

    def _write_snapshot(self, event: PutBoxInBagEvent | OpenBoxEvent, frame: np.ndarray | None) -> Path | None:
        if not self.save_snapshot or frame is None:
            return None
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        kind = getattr(event, "event", "event")
        path = self.alert_dir / f"{kind}_{stamp}_p{event.person_id}_f{event.frame_idx}.jpg"
        cv2.imwrite(str(path), frame)
        return path


def format_alert(event: PutBoxInBagEvent | OpenBoxEvent) -> str:
    kind = getattr(event, "event", "")
    box = f"box#{event.box_id}" if event.box_id is not None else "box"
    if kind == "open_box":
        return f"Open box detected — person #{event.person_id} ({event.reason})"
    bag = f"bag#{getattr(event, 'bag_id', None)}" if getattr(event, "bag_id", None) is not None else "bag"
    insert = getattr(event, "insert_score", 0.0)
    return (
        f"Person #{event.person_id} put {box} into {bag} "
        f"({event.reason}, insert={insert:.2f})"
    )


def alert_title(event: PutBoxInBagEvent | OpenBoxEvent) -> str:
    if getattr(event, "event", "") == "open_box":
        return "Open box"
    return "Box in bag"


def desktop_message(event: PutBoxInBagEvent | OpenBoxEvent) -> str:
    if getattr(event, "event", "") == "open_box":
        return "Open box detected"
    return format_alert(event)


def _macos_notify(message: str, desktop: bool, sound: bool, title: str = "Box alert") -> None:
    if sys.platform != "darwin":
        if sound:
            sys.stdout.write("\a")
            sys.stdout.flush()
        return
    if desktop:
        sound_clause = " sound name \"Sosumi\"" if sound else ""
        script = (
            f'display notification "{_osa_escape(message)}" '
            f'with title "{_osa_escape(title)}"{sound_clause}'
        )
        _run(["osascript", "-e", script])
        return
    if sound:
        _run(["afplay", "/System/Library/Sounds/Sosumi.aiff"])


def _post_webhook(url: str, payload: dict[str, Any], message: str) -> None:
    body = _webhook_body(url, payload, message)
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=8) as response:
            response.read()
    except (URLError, TimeoutError, OSError) as exc:
        print(f"ALERT webhook failed: {exc}")


def _webhook_body(url: str, payload: dict[str, Any], message: str) -> dict[str, Any]:
    lowered = url.lower()
    if "hooks.slack.com" in lowered:
        return {"text": f":rotating_light: {message}"}
    if "discord.com/api/webhooks" in lowered or "discordapp.com/api/webhooks" in lowered:
        return {"content": f"🚨 {message}"}
    return payload


def _osa_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass
