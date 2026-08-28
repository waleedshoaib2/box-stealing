"""Region of interest: keep YOLO person/box detections inside a zone."""

from __future__ import annotations

from src.geometry import BBox, area, center, clamp_box, containment, intersection, point_in_rect
from src.tracker import Detection

NormRect = tuple[float, float, float, float]
DEFAULT_ROI_CLASSES = ("person", "box", "bag", "open_box")


def parse_roi(values: object) -> NormRect | None:
    if not values:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in values)
    box = clamp_box((x1, y1, x2, y2))
    if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
        return None
    return box


def to_pixels(roi: NormRect, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = roi
    return (
        x1 * width,
        y1 * height,
        x2 * width,
        y2 * height,
    )


def to_norm(roi: BBox, width: int, height: int) -> NormRect:
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    x1, y1, x2, y2 = clamp_box(roi)
    return (x1 / w, y1 / h, x2 / w, y2 / h)


def detection_in_roi(det: Detection, roi_px: BBox) -> bool:
    """Person must have feet in the zone; boxes/bags need overlap."""
    if det.cls == "person":
        feet = ((det.bbox[0] + det.bbox[2]) / 2.0, det.bbox[3])
        return point_in_rect(feet, roi_px)
    if intersection(det.bbox, roi_px) <= 0:
        return False
    if area(det.bbox) <= 0:
        return point_in_rect(center(det.bbox), roi_px)
    return containment(det.bbox, roi_px) >= 0.15 or point_in_rect(center(det.bbox), roi_px)


def filter_detections(
    detections: list[Detection],
    roi_norm: NormRect | None,
    frame_size: tuple[int, int],
    classes: tuple[str, ...] | set[str] = DEFAULT_ROI_CLASSES,
) -> list[Detection]:
    if roi_norm is None:
        return detections
    width, height = frame_size
    roi_px = to_pixels(roi_norm, width, height)
    kept: list[Detection] = []
    gated = set(classes)
    for det in detections:
        if det.cls in gated and not detection_in_roi(det, roi_px):
            continue
        kept.append(det)
    return kept
