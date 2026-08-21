"""Spatial scores that feed the put-box-in-bag state machine.

All distances are normalized by person height so the same thresholds work
at different resolutions and subject distances from the camera.
"""

from __future__ import annotations

from src.geometry import BBox, center, containment, expand, height, iou, normalized_distance, point_in_rect
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
