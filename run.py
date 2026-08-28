#!/usr/bin/env python3
"""Detect a person putting a box into a bag.

Examples
--------
Webcam (YOLO-World person/box/bag):
    python run.py --source 0

Video file:
    python run.py --source path/to/video.mp4

RTSP camera (Hikvision main stream is also the default in config.yaml):
    python run.py --source rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101/

Custom single model (classes must map to person/box/bag in config.yaml):
    python run.py --mode custom --source video.mp4

Person model + box/bag model:
    python run.py --mode dual --source video.mp4
"""

from __future__ import annotations

import argparse

from src.alerts import Alerter
from src.event_detector import PutBoxInBagEvent
from src.pipeline import Pipeline, apply_task_to_cfg, load_config, output_name_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based: person puts a box in a bag")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--task",
        choices=["bag", "open", "both", "dual_entry"],
        default=None,
        help="bag = put-box-in-bag, open = person opening a box, dual_entry = take parcel / leave / come back",
    )
    parser.add_argument("--source", default=None, help="Webcam index, video path, image path, or RTSP URL")
    parser.add_argument("--mode", choices=["yolo_world", "custom", "dual"], default=None)
    parser.add_argument("--person-model", default=None, help="Person YOLO weights (dual mode)")
    parser.add_argument("--object-model", default=None, help="Box+bag YOLO weights (dual/custom)")
    parser.add_argument("--output-name", default="annotated")
    parser.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    parser.add_argument("--no-write", action="store_true", help="Do not write annotated video / events")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps")
    parser.add_argument("--test-alert", action="store_true", help="Fire a sample alert and exit")
    parser.add_argument("--no-record", action="store_true", help="Disable video recorder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "open" and args.config == "config.yaml":
        args.config = "config.open.yaml"
    if args.task == "dual_entry" and args.config == "config.yaml":
        args.config = "config.dual_entry.yaml"
    cfg = load_config(args.config)
    if args.task:
        apply_task_to_cfg(cfg, args.task)

    if args.mode:
        cfg["detector"]["mode"] = args.mode
    if args.conf is not None:
        cfg["detector"]["conf"] = args.conf
    if args.device:
        cfg["detector"]["device"] = args.device
    if args.person_model:
        cfg["detector"].setdefault("dual", {}).setdefault("person", {})["weights"] = args.person_model
    if args.object_model:
        mode = cfg["detector"]["mode"]
        if mode == "custom":
            cfg["detector"].setdefault("custom", {})["weights"] = args.object_model
        else:
            cfg["detector"].setdefault("dual", {}).setdefault("objects", {})["weights"] = args.object_model
            if args.person_model or mode == "dual":
                cfg["detector"]["mode"] = "dual"
    if args.source is not None:
        cfg.setdefault("source", {})["path"] = args.source
    if args.no_show:
        cfg.setdefault("visualizer", {})["show"] = False
    if args.no_write:
        cfg.setdefault("visualizer", {})["write"] = False
    if args.no_record:
        cfg.setdefault("recorder", {})["enabled"] = False

    if args.test_alert:
        _fire_test_alert(cfg)
        return

    pipeline = Pipeline(cfg)
    source = cfg.get("source", {}).get("path", 0)
    name = args.output_name
    if name == "annotated":
        name = output_name_for(source)
    pipeline.run(source=source, output_name=name)


def _fire_test_alert(cfg: dict) -> None:
    import numpy as np

    vis = cfg.get("visualizer", {})
    alerter = Alerter(cfg.get("alerts"), vis.get("output_dir", "outputs"))
    alerter.cooldown_seconds = 0
    event = PutBoxInBagEvent(
        frame_idx=0,
        timestamp=0.0,
        person_id=1,
        box_id=2,
        bag_id=3,
        hold_score=0.9,
        near_bag_score=0.8,
        insert_score=0.7,
        containment=0.6,
        reason="test_alert",
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    path = alerter.notify(event, frame)
    import time

    time.sleep(1.5)
    print(f"Test alert sent. Snapshot: {path or '(disabled)'}")
    print("If desktop alerts are on, check Notification Center (allow Terminal/Python if macOS asks).")


if __name__ == "__main__":
    main()
