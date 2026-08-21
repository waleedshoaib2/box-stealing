"""Bounding-box geometry used by tracking, association, and rules."""

from __future__ import annotations

from typing import Sequence

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def point_in_rect(point: tuple[float, float], box: BBox, margin: float = 0.0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def expand(box: BBox, scale: float) -> BBox:
    """Expand a box by `scale` of its width/height on every side."""
    x1, y1, x2, y2 = box
    dx = width(box) * scale
    dy = height(box) * scale
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def intersection(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    if inter <= 0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` that lies inside `outer` (intersection / area(inner))."""
    inner_area = area(inner)
    if inner_area <= 0:
        return 0.0
    return intersection(inner, outer) / inner_area


def centroid_distance(a: BBox, b: BBox) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def normalized_distance(a: BBox, b: BBox, scale_box: BBox | None = None) -> float:
    """Centroid distance divided by the reference height (person height by default)."""
    ref = height(scale_box if scale_box is not None else a)
    if ref <= 1e-6:
        ref = max(width(a), 1.0)
    return centroid_distance(a, b) / ref


def clamp_box(box: BBox) -> BBox:
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (float(x1), float(y1), float(x2), float(y2))


def as_xyxy(values: Sequence[float]) -> BBox:
    return clamp_box((float(values[0]), float(values[1]), float(values[2]), float(values[3])))
