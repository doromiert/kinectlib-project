"""
nkit/replay.py — read back sessions recorded by nkit/record.py

Recordings hold raw RGB/IR/depth, no landmarks (record.py deliberately runs
no detection), so replay feeds them through whatever pipeline you point at
them. That's the point: a fix can be checked against the same take as many
times as needed, with no Kinect and nobody standing in front of it.

    from nkit.replay import sessions, frames
    for s in sessions("recordings-livingroom/recordings"):
        for rgb, depth, ir, meta in frames(s):
            ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def sessions(root: str | Path) -> list[Path]:
    """Every session directory under root, chronological (names are timestamps)."""
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.jsonl").exists())


def manifest(session: str | Path) -> list[dict]:
    session = Path(session)
    with open(session / "manifest.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def frames(session: str | Path) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    """
    Yields (rgb, depth, ir, meta) in recorded order.

    rgb comes back 3-channel BGR rather than the sensor's 4-channel BGRX —
    everything downstream slices [:, :, :3] anyway, so the alpha byte was
    never read. depth/ir were stored as uint16 png and are restored to the
    float32 mm / intensity the trackers expect.
    """
    session = Path(session)
    for meta in manifest(session):
        base = session / "frames" / f"{meta['i']:06d}"
        rgb = cv2.imread(str(base) + ".jpg", cv2.IMREAD_COLOR)
        depth = cv2.imread(str(base) + "_depth.png", cv2.IMREAD_UNCHANGED)
        ir = cv2.imread(str(base) + "_ir.png", cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or ir is None:
            continue
        yield rgb, depth.astype(np.float32), ir.astype(np.float32), meta
