"""YOLO detectors for person, box, and bag.

Modes
-----
yolo_world : Ultralytics YOLO-World with text prompts (default, no custom weights).
custom     : A single trained YOLO whose class ids map to person/box/bag.
dual       : Person model (e.g. COCO yolov8n) + a separate box/bag model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.tracker import Detection


CANONICAL = {"person", "box", "bag", "open_box", "lid"}


class YOLODetector:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "yolo_world")).lower()
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.45))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.device = cfg.get("device") or None
        self._models: list[tuple[Any, dict[int, str]]] = []
        self._load()

    def _load(self) -> None:
        if self.mode == "yolo_world":
            self._load_yolo_world()
        elif self.mode == "custom":
            self._load_custom()
        elif self.mode == "dual":
            self._load_dual()
        else:
            raise ValueError(f"Unknown detector mode: {self.mode!r}. Use yolo_world, custom, or dual.")

    def _load_yolo_world(self) -> None:
        section = self.cfg.get("yolo_world", {})
        weights = section.get("weights", "yolov8s-worldv2.pt")
        prompts = section.get("prompts") or {
            "person": ["person"],
            "box": ["cardboard box", "box"],
            "bag": ["shopping bag", "bag", "backpack"],
        }
        names: list[str] = []
        label_to_canonical: dict[str, str] = {}
        for canonical, aliases in prompts.items():
            if canonical not in CANONICAL:
                raise ValueError(f"Prompt group {canonical!r} is not person/box/bag/open_box/lid")
            for alias in aliases:
                key = str(alias).strip().lower()
                if key not in label_to_canonical:
                    names.append(str(alias).strip())
                    label_to_canonical[key] = canonical

        model = _load_world_model(weights)
        if not hasattr(model, "set_classes"):
            raise RuntimeError(
                "This Ultralytics build cannot set YOLO-World classes. "
                "Upgrade ultralytics or switch detector.mode to custom/dual."
            )
        model.set_classes(names)
        self._world_label_map = label_to_canonical
        self._models.append((model, {}))

    def _load_custom(self) -> None:
        from ultralytics import YOLO

        section = self.cfg.get("custom", {})
        weights = section["weights"]
        class_map = {int(k): str(v).lower() for k, v in section.get("class_map", {}).items()}
        _validate_class_map(class_map)
        self._models.append((YOLO(weights), class_map))
        self._world_label_map = {}

    def _load_dual(self) -> None:
        from ultralytics import YOLO

        section = self.cfg.get("dual", {})
        person_cfg = section.get("person", {})
        object_cfg = section.get("objects", {})

        person_ids = {int(i) for i in person_cfg.get("class_ids", [0])}
        person_map = {i: "person" for i in person_ids}
        object_map = {int(k): str(v).lower() for k, v in object_cfg.get("class_map", {}).items()}
        _validate_class_map(object_map)

        self._models.append((YOLO(person_cfg["weights"]), person_map))
        self._models.append((YOLO(object_cfg["weights"]), object_map))
        self._world_label_map = {}

    def infer(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for model, class_map in self._models:
            results = model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            detections.extend(self._parse_results(results, class_map))
        return detections

    def _parse_results(self, results: list[Any], class_map: dict[int, str]) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections

        names = result.names if getattr(result, "names", None) else {}
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

        for bbox, conf, cls_id in zip(xyxy, confs, clss):
            raw_label = str(names.get(int(cls_id), cls_id)).strip()
            canonical = self._to_canonical(int(cls_id), raw_label, class_map)
            if canonical is None:
                continue
            detections.append(
                Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    cls=canonical,
                    conf=float(conf),
                    raw_label=raw_label,
                )
            )
        return detections

    def _to_canonical(self, cls_id: int, raw_label: str, class_map: dict[int, str]) -> str | None:
        if class_map:
            return class_map.get(cls_id)
        mapped = self._world_label_map.get(raw_label.lower())
        if mapped:
            return mapped
        lowered = raw_label.lower()
        for canonical in ("open_box", "person", "bag", "lid", "box"):
            token = canonical.replace("_", " ")
            if canonical in lowered or token in lowered:
                return canonical
        return None


def _load_world_model(weights: str) -> Any:
    try:
        from ultralytics import YOLOWorld

        return YOLOWorld(weights)
    except Exception:
        from ultralytics import YOLO

        return YOLO(weights)


def _validate_class_map(class_map: dict[int, str]) -> None:
    unknown = {v for v in class_map.values() if v not in CANONICAL}
    if unknown:
        raise ValueError(f"class_map values must be person/box/bag/open_box/lid, got {unknown}")
