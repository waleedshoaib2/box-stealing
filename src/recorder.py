"""Video recorder: event clips with preroll/postroll, plus optional continuous segments."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.event_detector import PutBoxInBagEvent


class Recorder:
    def __init__(
        self,
        cfg: dict[str, Any] | None,
        output_dir: str | Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.event_clips = bool(cfg.get("event_clips", True))
        self.continuous = bool(cfg.get("continuous", True))
        self.annotated = bool(cfg.get("annotated", True))
        self.preroll_seconds = float(cfg.get("preroll_seconds", 6))
        self.postroll_seconds = float(cfg.get("postroll_seconds", 8))
        self.segment_seconds = float(cfg.get("segment_seconds", 300))
        self.max_segments = int(cfg.get("max_segments", 24))
        self.fps = max(float(fps), 1.0)
        self.frame_size = frame_size
        self.root = Path(output_dir) / "recordings"
        self.events_dir = self.root / "events"
        self.continuous_dir = self.root / "continuous"

        preroll_frames = max(1, int(round(self.fps * self.preroll_seconds)))
        self._postroll_frames = max(1, int(round(self.fps * self.postroll_seconds)))
        self._buffer: deque[np.ndarray] = deque(maxlen=preroll_frames)

        self._clip_writer: cv2.VideoWriter | None = None
        self._clip_path: Path | None = None
        self._clip_frames_left = 0

        self._seg_writer: cv2.VideoWriter | None = None
        self._seg_path: Path | None = None
        self._seg_started = 0.0
        self.clips_written: list[Path] = []

        self.manual_dir = self.root / "manual"
        self._manual_writer: cv2.VideoWriter | None = None
        self._manual_path: Path | None = None
        self._manual_started = 0.0
        self.manual_clips: list[Path] = []

    def push(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        copied = frame.copy()
        self._buffer.append(copied)
        if self._clip_writer is not None:
            self._clip_writer.write(copied)
            self._clip_frames_left -= 1
            if self._clip_frames_left <= 0:
                self._close_clip()
        if self._manual_writer is not None:
            self._manual_writer.write(copied)
        if self.continuous:
            self._write_segment(copied)

    def trigger(self, event: PutBoxInBagEvent) -> Path | None:
        if not self.enabled or not self.event_clips:
            return None
        if self._clip_writer is not None:
            self._clip_frames_left = self._postroll_frames
            return self._clip_path

        self.events_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.events_dir / (
            f"put_box_in_bag_{stamp}_p{event.person_id}_f{event.frame_idx}.mp4"
        )
        writer = _open_writer(path, self.fps, self.frame_size)
        if writer is None:
            print(f"RECORD failed to open clip writer: {path}")
            return None
        for buffered in self._buffer:
            writer.write(buffered)
        self._clip_writer = writer
        self._clip_path = path
        self._clip_frames_left = self._postroll_frames
        print(f"RECORDING event clip → {path}  (preroll {len(self._buffer)} frames, postroll {self.postroll_seconds:.0f}s)")
        return path

    @property
    def is_manual_recording(self) -> bool:
        return self._manual_writer is not None

    @property
    def manual_elapsed(self) -> float:
        if self._manual_writer is None:
            return 0.0
        return max(0.0, time.time() - self._manual_started)

    def toggle_manual(self) -> Path | None:
        """Start or stop a user-triggered recording. Returns the clip path after a stop."""
        if not self.enabled:
            print("Recorder is disabled in config.")
            return None
        if self._manual_writer is not None:
            return self.stop_manual()
        self.start_manual()
        return None

    def start_manual(self) -> Path | None:
        if not self.enabled or self._manual_writer is not None:
            return self._manual_path
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.manual_dir / f"manual_{stamp}.mp4"
        writer = _open_writer(path, self.fps, self.frame_size)
        if writer is None:
            print(f"RECORD failed to open manual writer: {path}")
            return None
        for buffered in self._buffer:
            writer.write(buffered)
        self._manual_writer = writer
        self._manual_path = path
        self._manual_started = time.time()
        print(f"RECORDING (button) → {path}  (includes {len(self._buffer)} preroll frames)")
        return path

    def stop_manual(self) -> Path | None:
        path = self._manual_path
        self._close_manual()
        return path

    def close(self) -> None:
        self._close_clip()
        self._close_manual()
        self._close_segment()

    def _write_segment(self, frame: np.ndarray) -> None:
        now = time.time()
        if self._seg_writer is None or (now - self._seg_started) >= self.segment_seconds:
            self._close_segment()
            self.continuous_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = self.continuous_dir / f"{stamp}.mp4"
            writer = _open_writer(path, self.fps, self.frame_size)
            if writer is None:
                print(f"RECORD failed to open segment writer: {path}")
                return
            self._seg_writer = writer
            self._seg_path = path
            self._seg_started = now
            print(f"RECORDING segment → {path}")
            self._prune_segments()
        assert self._seg_writer is not None
        self._seg_writer.write(frame)

    def _close_clip(self) -> None:
        if self._clip_writer is None:
            return
        self._clip_writer.release()
        if self._clip_path is not None:
            self.clips_written.append(self._clip_path)
            print(f"Saved event clip {self._clip_path}")
        self._clip_writer = None
        self._clip_path = None
        self._clip_frames_left = 0

    def _close_manual(self) -> None:
        if self._manual_writer is None:
            return
        self._manual_writer.release()
        if self._manual_path is not None:
            self.manual_clips.append(self._manual_path)
            print(f"Saved recording {self._manual_path}")
        self._manual_writer = None
        self._manual_path = None
        self._manual_started = 0.0

    def _close_segment(self) -> None:
        if self._seg_writer is None:
            return
        self._seg_writer.release()
        if self._seg_path is not None:
            print(f"Saved segment {self._seg_path}")
        self._seg_writer = None
        self._seg_path = None

    def _prune_segments(self) -> None:
        if self.max_segments <= 0 or not self.continuous_dir.exists():
            return
        files = sorted(self.continuous_dir.glob("*.mp4"))
        extra = len(files) - self.max_segments
        for path in files[: max(0, extra)]:
            try:
                path.unlink()
            except OSError:
                pass


def _open_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter | None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        writer.release()
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, frame_size)
    if not writer.isOpened():
        writer.release()
        return None
    return writer
