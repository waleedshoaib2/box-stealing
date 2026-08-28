"""End-to-end video pipeline: detect → track → rule engine → overlay."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import yaml

from src.alerts import Alerter
from src.detector import YOLODetector
from src.dual_entry import DualEntryEngine
from src.event_detector import RuleEngine
from src.opening_detector import OpeningEngine
from src.recorder import Recorder
from src.tracker import IoUTracker
from src.visualizer import WINDOW_NAME, RecordButton, draw, draw_record_button


TASK_LABELS = {
    "bag": "Put box in the bag",
    "dual_entry": "Leaving and entering with packages",
    "open": "Opening packages",
}


def apply_task_to_cfg(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    """Enable exactly one behavior (or both bag+open). Mutates cfg and returns it."""
    vis = cfg.setdefault("visualizer", {})
    if task == "bag":
        cfg.setdefault("rules", {})["enabled"] = True
        cfg.setdefault("opening", {})["enabled"] = False
        cfg.setdefault("dual_entry", {})["enabled"] = False
        vis["window_name"] = "put-box-in-bag"
    elif task == "open":
        cfg.setdefault("rules", {})["enabled"] = False
        cfg.setdefault("opening", {})["enabled"] = True
        cfg.setdefault("dual_entry", {})["enabled"] = False
        vis["window_name"] = "open-box"
    elif task == "dual_entry":
        cfg.setdefault("rules", {})["enabled"] = False
        cfg.setdefault("opening", {})["enabled"] = False
        cfg.setdefault("dual_entry", {})["enabled"] = True
        vis["window_name"] = "dual-entry"
    elif task == "both":
        cfg.setdefault("rules", {})["enabled"] = True
        cfg.setdefault("opening", {})["enabled"] = True
        cfg.setdefault("dual_entry", {})["enabled"] = False
    else:
        raise ValueError(f"Unknown task {task!r}")
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    local_path = path.with_name("config.local.yaml")
    if local_path.exists():
        with open(local_path, encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle) or {}
        cfg = _deep_merge(cfg, overlay)
    return cfg


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Pipeline:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.detector = YOLODetector(cfg["detector"])
        tracker_cfg = cfg.get("tracker", {})
        self.tracker = IoUTracker(
            iou_threshold=float(tracker_cfg.get("iou_threshold", 0.3)),
            max_missed=int(tracker_cfg.get("max_missed", 20)),
        )
        rules_cfg = cfg.get("rules", {})
        self.engine: RuleEngine | None = None
        if bool(rules_cfg.get("enabled", True)):
            self.engine = RuleEngine.from_config(rules_cfg)
        opening_cfg = cfg.get("opening", {})
        self.opening: OpeningEngine | None = None
        if bool(opening_cfg.get("enabled", True)):
            self.opening = OpeningEngine.from_config(opening_cfg)
        dual_cfg = cfg.get("dual_entry", {})
        self.dual: DualEntryEngine | None = None
        if bool(dual_cfg.get("enabled", False)):
            self.dual = DualEntryEngine.from_config(dual_cfg)
        vis = cfg.get("visualizer", {})
        self.show = bool(vis.get("show", True))
        self.write = bool(vis.get("write", True))
        self.output_dir = Path(vis.get("output_dir", "outputs"))
        self.draw_scores = bool(vis.get("draw_scores", True))
        self.window_name = str(vis.get("window_name") or WINDOW_NAME)
        self.alerter = Alerter(cfg.get("alerts"), self.output_dir)
        self._recorder_cfg = cfg.get("recorder", {})
        self.recorder: Recorder | None = None
        self._banner = 0
        self._banner_event = None
        self.task = self._active_task_name()

    def _active_task_name(self) -> str:
        if self.dual is not None:
            return "dual_entry"
        if self.opening is not None and self.engine is None:
            return "open"
        if self.engine is not None and self.opening is None:
            return "bag"
        if self.engine is not None and self.opening is not None:
            return "both"
        return "bag"

    def apply_task(self, task: str) -> None:
        apply_task_to_cfg(self.cfg, task)
        self.window_name = str(self.cfg.get("visualizer", {}).get("window_name") or WINDOW_NAME)
        self.task = task
        self.reset_tracking()

    def reset_tracking(self) -> None:
        tracker_cfg = self.cfg.get("tracker", {})
        self.tracker = IoUTracker(
            iou_threshold=float(tracker_cfg.get("iou_threshold", 0.3)),
            max_missed=int(tracker_cfg.get("max_missed", 20)),
        )
        rules_cfg = self.cfg.get("rules", {})
        self.engine = RuleEngine.from_config(rules_cfg) if bool(rules_cfg.get("enabled")) else None
        opening_cfg = self.cfg.get("opening", {})
        self.opening = OpeningEngine.from_config(opening_cfg) if bool(opening_cfg.get("enabled")) else None
        dual_cfg = self.cfg.get("dual_entry", {})
        self.dual = DualEntryEngine.from_config(dual_cfg) if bool(dual_cfg.get("enabled")) else None
        self._banner = 0
        self._banner_event = None
        if self.alerter is not None:
            self.alerter._last_fire = 0.0

    def process_frame(self, frame: Any, frame_idx: int, fps: float) -> tuple[Any, list]:
        detections = self.detector.infer(frame)
        tracks = self.tracker.update(detections)
        timestamp = frame_idx / fps if fps > 0 else 0.0
        events = []
        if self.engine is not None:
            events.extend(self.engine.update(tracks, frame_idx, timestamp))
        if self.opening is not None:
            events.extend(self.opening.update(tracks, frame_idx, timestamp))
        if self.dual is not None:
            events.extend(self.dual.update(tracks, frame_idx, timestamp))
        if events:
            self._banner = int(max(fps, 15.0) * 2)
            self._banner_event = events[-1]

        vis = draw(
            frame,
            tracks,
            self.engine.states() if self.engine else {},
            events,
            banner_frames=self._banner,
            draw_scores=self.draw_scores,
            opening_states=self.opening.states() if self.opening else None,
            dual_states=self.dual.states() if self.dual else None,
            status_line=self.dual.status_line() if self.dual else None,
            last_event=self._banner_event,
        )
        record_frame = vis if (self.recorder and self.recorder.annotated) else frame
        if self.recorder is not None:
            self.recorder.push(record_frame)
        clip_path = None
        if events and self.recorder is not None:
            clip_path = self.recorder.trigger(events[0])
        extras = {"clip": str(clip_path)} if clip_path else None
        for event in events:
            self.alerter.notify(event, vis, extras=extras)
        if self._banner > 0:
            self._banner -= 1
            if self._banner <= 0:
                self._banner_event = None
        return vis, events

    def run(self, source: str | int | None = None, output_name: str = "annotated") -> list[dict[str, Any]]:
        src_cfg = self.cfg.get("source", {})
        if source is None:
            source = src_cfg.get("path", 0)
        stride = int(src_cfg.get("stride", 1))
        source = _coerce_source(source)
        is_rtsp = _is_rtsp(source)
        reconnect = bool(src_cfg.get("reconnect", is_rtsp))
        reconnect_delay = float(src_cfg.get("reconnect_delay", 1.0))

        cap = _open_capture(source, src_cfg)
        if cap is None or not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {_redact_source(source)}")

        fps, width, height, cap = _probe_stream(cap, source, src_cfg)
        print(f"Source: {_redact_source(source)}  {width}x{height} @ {fps:.1f} fps")
        self.recorder = Recorder(self._recorder_cfg, self.output_dir, fps, (width, height))
        if self.recorder.enabled:
            kind = []
            if self.recorder.event_clips:
                kind.append(f"event clips {self.recorder.preroll_seconds:.0f}s+{self.recorder.postroll_seconds:.0f}s")
            if self.recorder.continuous:
                kind.append(f"continuous {self.recorder.segment_seconds:.0f}s segments")
            print("Recorder: " + (", ".join(kind) if kind else "idle"))

        writer = None
        events_path = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.output_dir / f"{output_name}_events.jsonl"
        if self.write:
            writer = cv2.VideoWriter(
                str(self.output_dir / f"{output_name}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        events_log: list[dict[str, Any]] = []
        frame_idx = 0
        record_button = RecordButton()
        if self.show:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, width, height)
            record_button.attach(self.window_name)
        mode = []
        if self.engine is not None:
            mode.append("put-box-in-bag")
        if self.opening is not None:
            mode.append("open-box")
        if self.dual is not None:
            mode.append("dual-entry")
        print("Mode: " + (" + ".join(mode) if mode else "detect only"))
        print("Running. Click REC or press R to record. Press q to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if reconnect:
                        print("Stream dropped, reconnecting...")
                        cap.release()
                        time.sleep(reconnect_delay)
                        cap = _open_capture(source, src_cfg)
                        if cap is None or not cap.isOpened():
                            print("Reconnect failed, retrying...")
                            continue
                        continue
                    break
                if stride > 1 and frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                vis, events = self.process_frame(frame, frame_idx, fps)
                for event in events:
                    payload = event.to_dict()
                    if self.recorder is not None and self.recorder._clip_path is not None:
                        payload["clip"] = str(self.recorder._clip_path)
                    events_log.append(payload)
                    print(json.dumps(payload))
                    if events_path is not None:
                        with events_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(payload) + "\n")
                if writer is not None:
                    writer.write(vis)
                if self.show:
                    display = vis.copy()
                    recording = bool(self.recorder and self.recorder.is_manual_recording)
                    elapsed = self.recorder.manual_elapsed if self.recorder else 0.0
                    record_button.rect = draw_record_button(display, recording, elapsed)
                    cv2.imshow(self.window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r") or record_button.consume_click():
                        if self.recorder is not None:
                            self.recorder.toggle_manual()
                frame_idx += 1
        finally:
            if self.recorder is not None:
                self.recorder.close()
            cap.release()
            if writer is not None:
                writer.release()
            if self.show:
                cv2.destroyAllWindows()

        summary_path = self.output_dir / f"{output_name}_events.json"
        summary_path.write_text(json.dumps(events_log, indent=2), encoding="utf-8")
        print(f"Wrote {len(events_log)} event(s) to {summary_path}")
        return events_log


def output_name_for(source: str | int, fallback: str = "annotated") -> str:
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return "webcam"
    text = str(source)
    if _is_rtsp(text):
        parsed = urlsplit(text)
        channel = Path(parsed.path.rstrip("/") or "stream").name
        host = (parsed.hostname or "camera").replace(".", "-")
        return f"rtsp_{host}_{channel}"
    stem = Path(text).stem
    return stem or fallback


def _is_rtsp(source: str | int) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


def _redact_source(source: str | int) -> str:
    if not isinstance(source, str):
        return str(source)
    return re.sub(r":([^:@/]+)@", r":***@", source)


def _open_capture(source: str | int, src_cfg: dict[str, Any]) -> cv2.VideoCapture | None:
    if _is_rtsp(source):
        transport = str(src_cfg.get("rtsp_transport", "tcp")).lower()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


def _probe_stream(
    cap: cv2.VideoCapture,
    source: str | int,
    src_cfg: dict[str, Any],
) -> tuple[float, int, int, cv2.VideoCapture]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width > 0 and height > 0 and fps > 1:
        return float(fps), width, height, cap

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise FileNotFoundError(f"Opened source but received no frames: {_redact_source(source)}")
    height, width = frame.shape[:2]
    if fps <= 1:
        fps = 25.0
    cap.release()
    cap = _open_capture(source, src_cfg)
    if cap is None or not cap.isOpened():
        raise FileNotFoundError(f"Could not reopen video source: {_redact_source(source)}")
    return float(fps), int(width), int(height), cap


def _coerce_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    return text
