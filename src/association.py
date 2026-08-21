"""Spatial scores that feed the put-box-in-bag state machine.

All distances are normalized by person height so the same thresholds work
at different resolutions and subject distances from the camera.
"""

from __future__ import annotations

from src.geometry import (
    BBox,
    area,
    center,
    containment,
    expand,
    height,
    intersection,
    iou,
    normalized_distance,
    point_in_rect,
    width,
)
from src.tracker import Track


def carry_zone(person: BBox) -> BBox:
    """Torso + hands: skip the head and keep the lower 75% of the person."""
    x1, y1, x2, y2 = person
    head_cut = y1 + 0.25 * height(person)
    return (x1, head_cut, x2, y2)


def hold_score(person: Track, box: Track) -> float:
    """How likely the person is carrying this box (0..~1.5)."""
    person_box, obj = person.bbox, box.bbox
    contained = containment(obj, person_box)
    in_person = point_in_rect(center(obj), person_box)
    in_hands = point_in_rect(center(obj), expand(carry_zone(person_box), 0.08))
    dist = normalized_distance(person_box, obj, scale_box=person_box)

    score = 0.0
    score += min(contained, 1.0) * 0.7
    if in_person:
        score += 0.35
    if in_hands:
        score += 0.25
    score += max(0.0, 0.55 - dist)
    score += iou(person_box, obj) * 0.2
    return float(score)


def near_bag_score(person: Track, bag: Track) -> float:
    """How close the person is to a bag they could insert into."""
    dist = normalized_distance(person.bbox, bag.bbox, scale_box=person.bbox)
    overlap = iou(person.bbox, bag.bbox)
    bag_in_reach = containment(bag.bbox, expand(person.bbox, 0.35))
    score = overlap * 1.4 + max(0.0, 1.15 - dist) + bag_in_reach * 0.4
    return float(score)


def insert_score(box: Track, bag: Track) -> float:
    """How far the box has gone into the bag.

    Uses containment (intersection / box area) rather than IoU so a small box
    fully inside a large bag still scores high.
    """
    contained = containment(box.bbox, expand(bag.bbox, 0.12))
    center_in = point_in_rect(center(box.bbox), expand(bag.bbox, 0.18))
    score = contained
    if center_in:
        score += 0.30
    score += iou(box.bbox, bag.bbox) * 0.25
    return float(score)


def interact_score(person: Track, box: Track) -> float:
    """How likely the person is working on this box (held or on a table)."""
    dist = normalized_distance(person.bbox, box.bbox, scale_box=person.bbox)
    overlap = iou(person.bbox, expand(box.bbox, 0.30))
    hands = expand(carry_zone(person.bbox), 0.10)
    x1, y1, x2, y2 = box.bbox
    bw, bh = max(width(box.bbox), 1.0), max(height(box.bbox), 1.0)
    top_zone = (x1 - 0.2 * bw, y1 - 0.45 * bh, x2 + 0.2 * bw, y1 + 0.55 * bh)
    hands_area = area(hands)
    top_overlap = (intersection(hands, top_zone) / hands_area) if hands_area > 0 else 0.0
    score = max(0.0, 1.05 - dist) + overlap * 1.1 + top_overlap * 1.3
    if point_in_rect(center(box.bbox), expand(person.bbox, 0.25)):
        score += 0.25
    score += hold_score(person, box) * 0.25
    return float(score)


def is_open_box(track: Track) -> bool:
    if track.cls == "open_box":
        return True
    label = (track.raw_label or "").lower()
    return "open" in label and "box" in label


def height_growth(current: BBox, baseline_height: float) -> float:
    if baseline_height <= 1e-3:
        return 0.0
    return (height(current) - baseline_height) / baseline_height


def lid_lift_score(box: Track, lid: Track) -> float:
    """Lid sitting on / lifting off the top of the box."""
    bx1, by1, bx2, by2 = box.bbox
    lx, ly = center(lid.bbox)
    x_overlap = intersection((bx1, by1, bx2, by2), (lid.bbox[0], by1, lid.bbox[2], by2))
    box_w = max(width(box.bbox), 1.0)
    horiz = min(1.0, x_overlap / (box_w * height(box.bbox) + 1e-6) * height(box.bbox) / box_w)
    # Simpler horizontal overlap of x-ranges:
    overlap_x = max(0.0, min(bx2, lid.bbox[2]) - max(bx1, lid.bbox[0]))
    horiz = overlap_x / box_w
    above = ly <= (by1 + by2) / 2.0
    near_top = abs(lid.bbox[3] - by1) / max(height(box.bbox), 1.0)
    score = horiz * 0.8
    if above:
        score += 0.4
    score += max(0.0, 0.7 - near_top)
    return float(score)


def best_pair(
    left: list[Track],
    right: list[Track],
    score_fn,
    threshold: float,
) -> tuple[Track, Track, float] | None:
    best: tuple[Track, Track, float] | None = None
    for a in left:
        for b in right:
            score = score_fn(a, b)
            if score < threshold:
                continue
            if best is None or score > best[2]:
                best = (a, b, score)
    return best
