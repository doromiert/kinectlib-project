"""
nzk/hands.py — 3D hand tracking with fist detection

mediapipe hand landmarker + kinect depth for real Z coordinates.
IR fallback in low light. fist detection via per-finger curl ratio.

usage:
    from nzk.hands import HandTracker
    from nzk.kinect import Kinect

    with Kinect() as k, HandTracker() as tracker:
        rgb, depth, ir = k.get_frames()
        hands = tracker.process(rgb, depth, ir)
        for hand in hands:
            print(hand.wrist)       # Vec3(x, y, z_mm)
            print(hand.is_fist)     # bool
            print(hand.fist_score)  # 0.0–1.0
"""

from __future__ import annotations
import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .types import HandResult, Vec3, Source
from ._ir import (
    RGB_W, RGB_H, DEPTH_W, DEPTH_H,
    depth_at, detection_image,
)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

LANDMARK_NAMES: dict[int, str] = {
    0:  "wrist",
    1:  "thumb_cmc",  2:  "thumb_mcp",  3:  "thumb_ip",   4:  "thumb_tip",
    5:  "index_mcp",  6:  "index_pip",  7:  "index_dip",  8:  "index_tip",
    9:  "middle_mcp", 10: "middle_pip", 11: "middle_dip", 12: "middle_tip",
    13: "ring_mcp",   14: "ring_pip",   15: "ring_dip",   16: "ring_tip",
    17: "pinky_mcp",  18: "pinky_pip",  19: "pinky_dip",  20: "pinky_tip",
}

SCALE_X = DEPTH_W / RGB_W
SCALE_Y = DEPTH_H / RGB_H

# ── fist detection ────────────────────────────────────────────────────────────
# thumb deliberately excluded: thumbs-up would read as a fist otherwise.
CURL_FINGERS      = ["index", "middle", "ring", "pinky"]
CURL_THRESHOLD    = 0.55   # per-finger curl to count as "curled"
FIST_MIN_CURLED   = 3      # how many fingers must exceed threshold


def _vec3_dist(a: Vec3, b: Vec3) -> float:
    return ((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2) ** 0.5


def _finger_curl(landmarks: dict[str, Vec3], finger: str) -> float:
    """
    Curl ratio for one finger: 0.0 = fully open, 1.0 = fully curled.
    Computed as 1 - (tip_to_wrist / mcp_to_wrist), clamped [0, 1].
    Uses 3D distance so depth improves accuracy when available.
    """
    wrist = landmarks["wrist"]
    mcp   = landmarks[f"{finger}_mcp"]
    tip   = landmarks[f"{finger}_tip"]
    mcp_d = _vec3_dist(wrist, mcp)
    if mcp_d < 1e-6:
        return 0.0
    tip_d = _vec3_dist(wrist, tip)
    return float(np.clip(1.0 - tip_d / mcp_d, 0.0, 1.0))


def _detect_fist(landmarks: dict[str, Vec3]) -> tuple[bool, float]:
    curls    = [_finger_curl(landmarks, f) for f in CURL_FINGERS]
    score    = float(np.mean(curls))
    n_curled = sum(1 for c in curls if c >= CURL_THRESHOLD)
    return n_curled >= FIST_MIN_CURLED, score


# ── landmark extraction ───────────────────────────────────────────────────────

def _landmarks_from_result(
    result,
    depth_frame: np.ndarray,
) -> list[HandResult]:
    if not result.hand_landmarks:
        return []

    hands = []
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        landmarks: dict[str, Vec3] = {}

        for idx, lm in enumerate(hand_landmarks):
            x_rgb = int(np.clip(lm.x * RGB_W, 0, RGB_W - 1))
            y_rgb = int(np.clip(lm.y * RGB_H, 0, RGB_H - 1))
            x_d   = int(np.clip(x_rgb * SCALE_X, 0, DEPTH_W - 1))
            y_d   = int(np.clip(y_rgb * SCALE_Y, 0, DEPTH_H - 1))
            z_mm  = depth_at(depth_frame, x_d, y_d)
            landmarks[LANDMARK_NAMES[idx]] = Vec3(x_rgb, y_rgb, z_mm)

        is_fist, fist_score = _detect_fist(landmarks)
        label = handedness[0].category_name

        hands.append(HandResult(
            hand        = label,
            source      = "rgb",    # overwritten below if IR was used
            wrist       = landmarks["wrist"],
            thumb_tip   = landmarks["thumb_tip"],
            index_tip   = landmarks["index_tip"],
            middle_tip  = landmarks["middle_tip"],
            ring_tip    = landmarks["ring_tip"],
            pinky_tip   = landmarks["pinky_tip"],
            landmarks   = landmarks,
            is_fist     = is_fist,
            fist_score  = fist_score,
        ))

    return hands


# ── tracker ───────────────────────────────────────────────────────────────────

class HandTracker:
    def __init__(self, max_hands: int = 2, confidence: float = 0.5):
        if not os.path.exists(MODEL_PATH):
            print("downloading hand landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=max_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def process(
        self,
        rgb_frame:   np.ndarray,
        depth_frame: np.ndarray,
        ir_frame:    np.ndarray | None = None,
    ) -> list[HandResult]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32  — optional IR fallback

        returns list of HandResult.
        """
        img, source = detection_image(rgb_frame, ir_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result   = self._detector.detect(mp_image)

        hands = _landmarks_from_result(result, depth_frame)
        for h in hands:
            h.source = source

        return hands

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
