"""Overlay tracks, person state, events, and the on-screen REC button."""

from __future__ import annotations

import cv2
import numpy as np

from src.event_detector import PersonState
from src.opening_detector import OpeningState
from src.tracker import Track


WINDOW_NAME = "put-box-in-bag"

COLORS = {
    "person": (40, 200, 80),
    "box": (0, 165, 255),
    "open_box": (0, 220, 255),
    "bag": (255, 160, 40),
}

STATE_COLORS = {
    "idle": (180, 180, 180),
    "holding_box": (0, 165, 255),
    "near_bag": (255, 180, 0),
    "inserting": (0, 80, 255),
    "put_box_in_bag": (0, 0, 255),
    "interacting": (0, 200, 220),
    "opening": (0, 140, 255),
    "open_box": (0, 90, 255),
}

BANNER_TEXT = {
    "put_box_in_bag": "PUT BOX IN BAG",
    "open_box": "PERSON OPENING BOX",
}

BUTTON_SIZE = (176, 56)
BUTTON_MARGIN = 16


class RecordButton:
    """Hit-test target for the on-screen REC / STOP control."""

    def __init__(self) -> None:
        self.rect = (0, 0, 0, 0)
        self._clicked = False

    def attach(self, window_name: str = WINDOW_NAME) -> None:
        cv2.setMouseCallback(window_name, self._on_mouse)

    def consume_click(self) -> bool:
        clicked = self._clicked
        self._clicked = False
        return clicked

    def hit(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONUP and self.hit(x, y):
            self._clicked = True


def draw(
    frame: np.ndarray,
    tracks: list[Track],
    states: dict[int, PersonState],
    events: list,
    banner_frames: int = 0,
    draw_scores: bool = True,
    opening_states: dict[int, OpeningState] | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    for track in tracks:
        if track.missed > 0:
            continue
        color = COLORS.get(track.cls, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{track.cls} #{track.track_id} {track.conf:.2f}"
        _label(canvas, label, x1, max(0, y1 - 8), color)

        if track.cls == "person":
            ctx = None
            extra = ""
            open_ctx = opening_states.get(track.track_id) if opening_states else None
            bag_ctx = states.get(track.track_id)
            if open_ctx is not None and open_ctx.state not in ("idle",):
                ctx = open_ctx
                extra = open_ctx.state
                if draw_scores:
                    extra += f" i={open_ctx.interact_score:.2f} g={open_ctx.growth:.2f}"
            elif bag_ctx is not None:
                ctx = bag_ctx
                extra = bag_ctx.state
                if draw_scores and bag_ctx.state != "idle":
                    extra += f" h={bag_ctx.hold_score:.2f} n={bag_ctx.near_score:.2f} i={bag_ctx.insert_score:.2f}"
            if ctx is not None:
                state_color = STATE_COLORS.get(ctx.state, (200, 200, 200))
                _label(canvas, extra, x1, y2 + 16, state_color)

        if len(track.history) >= 2:
            pts = np.array([[int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)] for b in track.history], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, color, 1)

    if events or banner_frames > 0:
        latest = events[-1] if events else None
        kind = getattr(latest, "event", "put_box_in_bag") if latest is not None else "put_box_in_bag"
        text = BANNER_TEXT.get(kind, kind.replace("_", " ").upper())
        if latest is not None:
            text = f"{text}  person#{latest.person_id}  ({latest.reason})"
        overlay = canvas.copy()
        h, w = canvas.shape[:2]
        color = (0, 90, 220) if kind == "open_box" else (0, 0, 200)
        cv2.rectangle(overlay, (0, 0), (w, 54), color, -1)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
        cv2.putText(canvas, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def draw_record_button(
    canvas: np.ndarray,
    recording: bool,
    elapsed_s: float = 0.0,
) -> tuple[int, int, int, int]:
    """Draw REC / STOP in the top-right. Returns the clickable rectangle."""
    h, w = canvas.shape[:2]
    bw, bh = BUTTON_SIZE
    x2 = w - BUTTON_MARGIN
    x1 = x2 - bw
    y1 = BUTTON_MARGIN
    y2 = y1 + bh

    fill = (0, 0, 190) if recording else (36, 36, 36)
    border = (220, 220, 255) if recording else (210, 210, 210)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2)

    cx, cy = x1 + 28, (y1 + y2) // 2
    if recording:
        cv2.circle(canvas, (cx, cy), 11, (40, 40, 255), -1)
        cv2.circle(canvas, (cx, cy), 11, (255, 255, 255), 2)
        label = f"STOP  {_fmt_elapsed(elapsed_s)}"
    else:
        cv2.circle(canvas, (cx, cy), 11, (0, 0, 255), -1)
        cv2.circle(canvas, (cx, cy), 11, (255, 255, 255), 2)
        label = "REC"

    cv2.putText(
        canvas,
        label,
        (x1 + 48, y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    hint = "click or press R"
    cv2.putText(
        canvas,
        hint,
        (x1 + 8, y2 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return (x1, y1, x2, y2)


def _fmt_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 4, y)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 8, y + 4), color, -1)
    cv2.putText(image, text, (x + 4, y - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
