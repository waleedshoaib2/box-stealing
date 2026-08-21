"""Class-aware IoU tracker.

Persons, boxes, and bags are tracked independently so a box cannot steal a
person ID (and vice versa). Matching is greedy by IoU within each class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.geometry import BBox, as_xyxy, iou


CANONICAL_CLASSES = ("person", "box", "bag")


@dataclass
class Detection:
    bbox: BBox
    cls: str
    conf: float
    raw_label: str = ""


@dataclass
class Track:
    track_id: int
    cls: str
    bbox: BBox
    conf: float
    hits: int = 1
    missed: int = 0
    age: int = 1
    raw_label: str = ""
    history: list[BBox] = field(default_factory=list)

    def record(self, bbox: BBox, conf: float, raw_label: str, trail_length: int = 30) -> None:
        self.bbox = bbox
        self.conf = conf
        self.raw_label = raw_label or self.raw_label
        self.hits += 1
        self.missed = 0
        self.age += 1
        self.history.append(bbox)
        if len(self.history) > trail_length:
            self.history = self.history[-trail_length:]


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def active(self, cls: str | None = None) -> list[Track]:
        tracks = [t for t in self._tracks.values() if t.missed == 0]
        if cls is not None:
            tracks = [t for t in tracks if t.cls == cls]
        return tracks

    def get(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def update(self, detections: Iterable[Detection]) -> list[Track]:
        by_class: dict[str, list[Detection]] = {c: [] for c in CANONICAL_CLASSES}
        for det in detections:
            if det.cls in by_class:
                by_class[det.cls].append(det)

        matched_ids: set[int] = set()
        for cls, dets in by_class.items():
            class_tracks = [t for t in self._tracks.values() if t.cls == cls]
            pairs = _greedy_match(class_tracks, dets, self.iou_threshold)
            used_dets: set[int] = set()
            for track, det_idx in pairs:
                det = dets[det_idx]
                track.record(as_xyxy(det.bbox), det.conf, det.raw_label)
                matched_ids.add(track.track_id)
                used_dets.add(det_idx)
            for i, det in enumerate(dets):
                if i in used_dets:
                    continue
                track = Track(
                    track_id=self._next_id,
                    cls=cls,
                    bbox=as_xyxy(det.bbox),
                    conf=det.conf,
                    raw_label=det.raw_label,
                    history=[as_xyxy(det.bbox)],
                )
                self._tracks[self._next_id] = track
                matched_ids.add(self._next_id)
                self._next_id += 1

        dropped: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_ids:
                continue
            track.missed += 1
            track.age += 1
            if track.missed > self.max_missed:
                dropped.append(track_id)
        for track_id in dropped:
            del self._tracks[track_id]
        return self.tracks


def _greedy_match(
    tracks: list[Track],
    detections: list[Detection],
    iou_threshold: float,
) -> list[tuple[Track, int]]:
    candidates: list[tuple[float, int, int]] = []
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            score = iou(track.bbox, det.bbox)
            if score >= iou_threshold:
                candidates.append((score, t_idx, d_idx))
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matches: list[tuple[Track, int]] = []
    for _, t_idx, d_idx in candidates:
        if t_idx in used_tracks or d_idx in used_dets:
            continue
        used_tracks.add(t_idx)
        used_dets.add(d_idx)
        matches.append((tracks[t_idx], d_idx))
    return matches
