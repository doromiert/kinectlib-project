"""
nkit/hands.py — 3D hand tracking with fist detection

mediapipe hand landmarker + kinect depth for real Z coordinates.
Adaptive RGB/IR source selection by default; optional dual-source fusion.
Fist detection via per-finger curl ratio.

usage:
    from nkit.hands import HandTracker
    from nkit.kinect import Kinect
    from nkit.types import GestureConfig

    config = GestureConfig()
    with Kinect() as k, HandTracker() as tracker:
        rgb, depth, ir = k.get_frames()
        hands = tracker.process(rgb, depth, ir, config)
        for hand in hands:
            print(hand.wrist)       # Vec3(x, y, z_mm)
            print(hand.is_fist)     # bool
            print(hand.fist_score)  # 0.0-1.0

Hand <-> skeleton association (skeleton_id / side) is NOT done here — see
body.py:associate_hands_to_bodies(), which needs both hands and bodies.
"""

from __future__ import annotations
import os
import urllib.request
from typing import Literal

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .types import HandResult, Vec3, GestureConfig
from . import _vision, _mp
from ._vision import RGB_W, RGB_H, DEPTH_W, DEPTH_H, depth_at, Roi

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

# thumb deliberately excluded from curl scoring: a thumbs-up would otherwise read as a fist.
CURL_FINGERS = ["index", "middle", "ring", "pinky"]

# matching hands detected independently in the RGB pass vs the IR pass (fusion mode)
FUSION_MATCH_MAX_DIST_PX = 150

# half-width (px) of the crop searched around a skeleton's own wrist
# estimate when bodies are supplied to process() — generous enough for a
# full hand plus some slack for the wrist estimate not being pixel-perfect,
# small enough that a foot or other body part elsewhere in frame is simply
# never in the search region at all
WRIST_CROP_HALF_PX = 260


def _vec3_dist(a: Vec3, b: Vec3) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


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


def _detect_fist(landmarks: dict[str, Vec3], config: GestureConfig) -> tuple[bool, float]:
    curls    = [_finger_curl(landmarks, f) for f in CURL_FINGERS]
    score    = float(np.mean(curls))
    n_curled = sum(1 for c in curls if c >= config.fist_curl_threshold)
    return n_curled >= config.fist_min_curled, score


def _landmarks_xy_from_result(result, roi: Roi, config: GestureConfig) -> list[tuple[dict[str, Vec3], str, float]]:
    """
    extract (landmarks_dict_with_xy_only, handedness_label, confidence) per
    detected hand, z left at 0.0. pt.x/pt.y are normalized against the
    detection image, which may be a crop — map through roi to full RGB
    pixel space.

    Hands below config.hand_confidence_threshold are dropped here — this is
    the actual plausibility check for "is this really a hand" (mediapipe
    will confidently misread a foot/limb as a hand often enough that this
    matters), applied before a candidate ever reaches identity tracking.

    The detection image is always mirrored (see _vision.detection_image's
    mirror param) so mediapipe's own Left/Right handedness label comes out
    anatomically correct for Kinect's non-mirrored feed — no separate label
    swap needed, mirroring the input is what fixes it. x is un-mirrored
    here (w - pt.x*w instead of pt.x*w) to land back in true camera space.
    """
    if not result.hand_landmarks:
        return []
    x1, y1, x2, y2 = roi
    w, h = x2 - x1, y2 - y1
    out = []
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        score = handedness[0].score
        if score < config.hand_confidence_threshold:
            continue
        lm: dict[str, Vec3] = {}
        for idx, pt in enumerate(hand_landmarks):
            x = int(np.clip(x1 + (1.0 - pt.x) * w, 0, RGB_W - 1))
            y = int(np.clip(y1 + pt.y * h, 0, RGB_H - 1))
            lm[LANDMARK_NAMES[idx]] = Vec3(x, y, 0.0)
        out.append((lm, handedness[0].category_name, score))
    return out


def _resample_z(landmarks: dict[str, Vec3], depth_frame: np.ndarray) -> dict[str, Vec3]:
    out = {}
    for name, v in landmarks.items():
        x_d = int(np.clip(v.x * SCALE_X, 0, DEPTH_W - 1))
        y_d = int(np.clip(v.y * SCALE_Y, 0, DEPTH_H - 1))
        out[name] = Vec3(v.x, v.y, depth_at(depth_frame, x_d, y_d))
    return out


def _fuse_hand_lists(
    rgb_hands: list[tuple[dict[str, Vec3], str, float]],
    ir_hands: list[tuple[dict[str, Vec3], str, float]],
    rgb_weight: float,
    ir_weight: float,
) -> list[tuple[dict[str, Vec3], str, float]]:
    """
    Greedy nearest-neighbour 1:1 match of hands detected independently in
    the RGB pass and the IR pass (matched by wrist pixel distance), then
    merges matched pairs (confidence: the higher of the two). Unmatched
    hands pass through from whichever source found them.
    """
    candidates = []
    for i, (rgb_lm, _, _s) in enumerate(rgb_hands):
        for j, (ir_lm, _, _s2) in enumerate(ir_hands):
            d = _vec3_dist(rgb_lm["wrist"], ir_lm["wrist"])
            if d <= FUSION_MATCH_MAX_DIST_PX:
                candidates.append((d, i, j))
    candidates.sort(key=lambda c: c[0])

    used_rgb: set[int] = set()
    used_ir:  set[int] = set()
    fused: list[tuple[dict[str, Vec3], str, float]] = []

    for d, i, j in candidates:
        if i in used_rgb or j in used_ir:
            continue
        used_rgb.add(i)
        used_ir.add(j)
        rgb_lm, label, rgb_score = rgb_hands[i]
        ir_lm, _, ir_score       = ir_hands[j]
        merged = _vision.merge_landmarks(rgb_lm, ir_lm, rgb_weight, ir_weight)
        fused.append((merged, label, max(rgb_score, ir_score)))

    for i, (rgb_lm, label, score) in enumerate(rgb_hands):
        if i not in used_rgb:
            fused.append((rgb_lm, label, score))
    for j, (ir_lm, label, score) in enumerate(ir_hands):
        if j not in used_ir:
            fused.append((ir_lm, label, score))

    return fused


def _build_hand(landmarks: dict[str, Vec3], label: str, score: float, source, config: GestureConfig) -> HandResult:
    is_fist, fist_score = _detect_fist(landmarks, config)
    return HandResult(
        hand        = label,
        source      = source,
        wrist       = landmarks["wrist"],
        thumb_tip   = landmarks["thumb_tip"],
        index_tip   = landmarks["index_tip"],
        middle_tip  = landmarks["middle_tip"],
        ring_tip    = landmarks["ring_tip"],
        pinky_tip   = landmarks["pinky_tip"],
        landmarks   = landmarks,
        is_fist     = is_fist,
        fist_score  = fist_score,
        hand_confidence = score,
    )


class HandTracker:
    def __init__(self, max_hands: int = 4, confidence: float = 0.5, use_gpu: bool = True):
        # max_hands is per detection pass; in fusion mode two passes run,
        # so effectively up to max_hands*2 candidates get matched/merged.
        if not os.path.exists(MODEL_PATH):
            print("downloading hand landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        # see BodyTracker.__init__ — GPU delegate is GL compute, CPU fallback
        self._detector = _mp.create_detector(
            vision.HandLandmarker,
            lambda delegate: vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH, delegate=delegate),
                num_hands=max_hands,
                min_hand_detection_confidence=confidence,
                min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            ),
            use_gpu,
            "hand landmarker",
        )
        self._roi_tracker = _vision.PersonRoiTracker()

    def process(
        self,
        rgb_frame:   np.ndarray,
        depth_frame: np.ndarray,
        ir_frame:    np.ndarray | None = None,
        config:      GestureConfig | None = None,
        enable_fusion: bool = False,
        roi: Roi | None | Literal["auto"] = "auto",
        bodies=None,
    ) -> list[HandResult]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32  — optional, required for IR fallback/fusion
        config:      GestureConfig — thresholds; defaults if omitted
        enable_fusion: run both RGB and IR every frame and merge, instead of
                       picking one adaptively (~2x inference cost)
        roi: "auto" (default) computes a depth-based crop each call; pass an
             explicit (x1,y1,x2,y2) to reuse one computed once and shared
             with BodyTracker.process() in the same frame (nkit/bridge.py
             does this); pass None to disable cropping. Ignored when
             `bodies` is given (see below).
        bodies: list[BodyResult] from THIS frame's BodyTracker.process(), if
             pose is enabled. When given, hand detection runs in a small
             crop around each body's OWN wrist estimate instead of
             searching the whole frame/roi — a hand can only be found
             where a skeleton says a hand should be, which is what
             actually rejects a foot/limb false-positive rather than
             hoping confidence filtering catches it after the fact. Falls
             back to the whole-frame/roi search above when bodies is None
             or empty (e.g. skeleton tracking is off — see
             GestureConfig / bridge.py's live body_enabled toggle).

        returns list of HandResult. skeleton_id/side are unset — see
        body.py:associate_hands_to_bodies() (or nkit/identity.py, which
        nkit/bridge.py actually uses).
        """
        config = config or GestureConfig()

        if bodies:
            hands = []
            for body in bodies:
                for wrist in (body.left_wrist, body.right_wrist):
                    crop = _vision.point_crop_roi(wrist.x, wrist.y, WRIST_CROP_HALF_PX)
                    hands.extend(self._detect_in_roi(rgb_frame, depth_frame, ir_frame, config, enable_fusion, crop))
            return hands

        resolved_roi = self._roi_tracker.find(depth_frame) if roi == "auto" else roi
        return self._detect_in_roi(rgb_frame, depth_frame, ir_frame, config, enable_fusion, resolved_roi)

    def _detect_in_roi(
        self,
        rgb_frame:   np.ndarray,
        depth_frame: np.ndarray,
        ir_frame:    np.ndarray | None,
        config:      GestureConfig,
        enable_fusion: bool,
        roi: Roi | None,
    ) -> list[HandResult]:
        if enable_fusion and ir_frame is not None:
            rgb_img, ir_img, rgb_w, ir_w, roi_used = _vision.detection_images_fused(rgb_frame, ir_frame, roi, mirror=True)

            rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            ir_mp  = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_img)
            rgb_result = self._detector.detect(rgb_mp)
            ir_result  = self._detector.detect(ir_mp)

            rgb_hands = _landmarks_xy_from_result(rgb_result, roi_used, config)
            ir_hands  = _landmarks_xy_from_result(ir_result, roi_used, config)
            fused = _fuse_hand_lists(rgb_hands, ir_hands, rgb_w, ir_w)

            hands = []
            for landmarks, label, score in fused:
                landmarks = _resample_z(landmarks, depth_frame)
                hands.append(_build_hand(landmarks, label, score, "fused", config))
            return hands

        force_source = "ir" if config.hand_prefer_ir else None
        img, source, roi_used = _vision.detection_image(rgb_frame, ir_frame, config, roi, mirror=True, force_source=force_source)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result   = self._detector.detect(mp_image)

        hands = []
        for landmarks, label, score in _landmarks_xy_from_result(result, roi_used, config):
            landmarks = _resample_z(landmarks, depth_frame)
            hands.append(_build_hand(landmarks, label, score, source, config))
        return hands

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
