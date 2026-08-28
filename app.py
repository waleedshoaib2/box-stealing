#!/usr/bin/env python3
"""Live PyQt monitor for bag / dual-entry / opening detection from RTSP."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 app.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.gui import launch


if __name__ == "__main__":
    raise SystemExit(launch())
