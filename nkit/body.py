"""
nkit/body.py — full-body skeleton tracking

mediapipe pose landmarker + kinect depth for real Z coordinates.
Adaptive RGB/IR source selection by default; optional dual-source fusion
(shared logic in nkit._vision). Assigns a stable skeleton_id per person
across frames via a small centroid tracker, since mediapipe gives no
persistent ID on its own — everything downstream (face association, hand
association, gesture state) keys off that ID.

usage:
    from nkit.body import BodyTracker
    from nkit.kinect import Kinect

    with Kinect() as k, BodyTracker() as tracker:
        rgb, depth, ir = k.get_frames()
        bodies = tracker.process(rgb, depth, ir)
        for body in bodies:
            print(body.skeleton_id, body.nose)
"""

from __future__ import annotations
import os
import urllib.request
from typing import Literal

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .types import BodyResult, HandResult, Vec3, GestureConfig
from . import _vision, _mp
from ._vision import RGB_W, RGB_H, DEPTH_W, DEPTH_H, depth_at, Roi

# lite tier, not full — the extra accuracy of _full isn't worth 2-3x the
# inference cost for gesture control (see the perf pass in the overview doc)
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

LANDMARK_NAMES: dict[int, str] = {
    0:  "nose",
    1:  "left_eye_inner",   2:  "left_eye",        3:  "left_eye_outer",
    4:  "right_eye_inner",  5:  "right_eye",        6:  "right_eye_outer",
    7:  "left_ear",         8:  "right_ear",
    9:  "mouth_left",       10: "mouth_right",
    11: "left_shoulder",    12: "right_shoulder",
    13: "left_elbow",       14: "right_elbow",
    15: "left_wrist",       16: "right_wrist",
    17: "left_pinky",       18: "right_pinky",
    19: "left_index",       20: "right_index",
    21: "left_thumb",       22: "right_thumb",
    23: "left_hip",         24: "right_hip",
    25: "left_knee",        26: "right_knee",
    27: "left_ankle",       28: "right_ankle",
    29: "left_heel",        30: "right_heel",
    31: "left_foot_index",  32: "right_foot_index",
}

SCALE_X = DEPTH_W / RGB_W
SCALE_Y = DEPTH_H / RGB_H

# max px distance (RGB space) to match a face bbox to a skeleton nose
FACE_NOSE_MAX_DIST_PX = 200

# max px distance (RGB space) to match a hand wrist to a skeleton wrist
HAND_WRIST_MAX_DIST_PX = 150

FUSION_MATCH_MAX_DIST_PX = 200


def _dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _vec3_dist(a: Vec3, b: Vec3) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


class _SkeletonIdTracker:
    """
    Assigns stable integer skeleton_id's to per-frame pose detections by
    greedy nearest-neighbour matching on nose position against last-seen
    positions. Survives brief occlusion/misses via a WALL-CLOCK grace
    period (not a frame count — a frame-count grace period silently shrinks
    in real time whenever fps drops, which is exactly backwards: you want
    MORE tolerance when detection is struggling, not less) so a gesture in
    progress doesn't get orphaned by a dropped detection or two.
    """

    def __init__(self, max_dist_px: float = 250.0, grace_seconds: float = 1.0):
        self._next_id = 0
        self._tracks: dict[int, dict] = {}   # id -> {"pos": (x,y), "last_seen": float}
        self._max_dist = max_dist_px
        self._grace_seconds = grace_seconds

    def assign(self, centroids: list[tuple[float, float]], timestamp: float) -> list[int]:
        candidates = []
        track_ids = list(self._tracks.keys())
        for ci, (cx, cy) in enumerate(centroids):
            for tid in track_ids:
                tx, ty = self._tracks[tid]["pos"]
                d = _dist_xy(cx, cy, tx, ty)
                if d <= self._max_dist:
                    candidates.append((d, ci, tid))
        candidates.sort(key=lambda c: c[0])

        assigned: dict[int, int] = {}   # centroid index -> track id
        used_tracks: set[int] = set()
        for d, ci, tid in candidates:
            if ci in assigned or tid in used_tracks:
                continue
            assigned[ci] = tid
            used_tracks.add(tid)

        result = []
        seen_this_frame: set[int] = set()
        for ci, (cx, cy) in enumerate(centroids):
            tid = assigned.get(ci)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = {"pos": (cx, cy), "last_seen": timestamp}
            seen_this_frame.add(tid)
            result.append(tid)

        # age out tracks not seen within the grace window — wall-clock, not frame count
        for tid in list(self._tracks.keys()):
            if tid in seen_this_frame:
                continue
            if timestamp - self._tracks[tid]["last_seen"] > self._grace_seconds:
                del self._tracks[tid]

        return result


def _landmarks_xy_from_result(pose_landmarks, roi: Roi) -> dict[str, Vec3]:
    """
    pt.x/pt.y are normalized against the detection image, which may be a
    crop — map through roi to full RGB pixel space. The detection image is
    always mirrored (see _vision.detection_image's mirror param) so
    mediapipe's left_*/right_* landmark naming comes out anatomically
    correct for Kinect's non-mirrored feed — so x is un-mirrored here
    (w - pt.x*w instead of pt.x*w) to land back in true camera space.
    """
    x1, y1, x2, y2 = roi
    w, h = x2 - x1, y2 - y1
    lm: dict[str, Vec3] = {}
    for idx, pt in enumerate(pose_landmarks):
        x = int(np.clip(x1 + (1.0 - pt.x) * w, 0, RGB_W - 1))
        y = int(np.clip(y1 + pt.y * h, 0, RGB_H - 1))
        lm[LANDMARK_NAMES[idx]] = Vec3(x, y, 0.0)
    return lm


LANDMARK_DEPTH_RADIUS = 7   # wider than the ungated default; the gate does the rejecting


def _torso_reference(landmarks: dict[str, Vec3], depth_frame: np.ndarray) -> float | None:
    """
    Depth of the chest, or None if it can't be read.

    The point between the shoulders is the most reliable sample on a person:
    it's broad and flat, so a patch there lands entirely on them, where a
    patch at a wrist or elbow is half background. That makes it a good
    reference for rejecting background bleed-through everywhere else — see
    _vision.depth_at's `reference`.
    """
    ls, rs = landmarks.get("left_shoulder"), landmarks.get("right_shoulder")
    if ls is None or rs is None:
        return None
    x_d = int(np.clip((ls.x + rs.x) / 2 * SCALE_X, 0, DEPTH_W - 1))
    y_d = int(np.clip((ls.y + rs.y) / 2 * SCALE_Y, 0, DEPTH_H - 1))
    z = depth_at(depth_frame, x_d, y_d, radius=9)
    return z if z > 0 else None


def _resample_z(landmarks: dict[str, Vec3], depth_frame: np.ndarray) -> dict[str, Vec3]:
    reference = _torso_reference(landmarks, depth_frame)
    out = {}
    for name, v in landmarks.items():
        x_d = int(np.clip(v.x * SCALE_X, 0, DEPTH_W - 1))
        y_d = int(np.clip(v.y * SCALE_Y, 0, DEPTH_H - 1))
        out[name] = Vec3(v.x, v.y, depth_at(depth_frame, x_d, y_d,
                                            radius=LANDMARK_DEPTH_RADIUS,
                                            reference=reference))
    return out


def _fuse_body_lists(
    rgb_bodies: list[dict[str, Vec3]],
    ir_bodies: list[dict[str, Vec3]],
    rgb_weight: float,
    ir_weight: float,
) -> list[dict[str, Vec3]]:
    """Same greedy nearest-neighbour merge pattern as hands.py's fusion, keyed on nose position."""
    candidates = []
    for i, a in enumerate(rgb_bodies):
        for j, b in enumerate(ir_bodies):
            d = _dist_xy(a["nose"].x, a["nose"].y, b["nose"].x, b["nose"].y)
            if d <= FUSION_MATCH_MAX_DIST_PX:
                candidates.append((d, i, j))
    candidates.sort(key=lambda c: c[0])

    used_rgb: set[int] = set()
    used_ir:  set[int] = set()
    fused: list[dict[str, Vec3]] = []

    for d, i, j in candidates:
        if i in used_rgb or j in used_ir:
            continue
        used_rgb.add(i)
        used_ir.add(j)
        fused.append(_vision.merge_landmarks(rgb_bodies[i], ir_bodies[j], rgb_weight, ir_weight))

    for i, a in enumerate(rgb_bodies):
        if i not in used_rgb:
            fused.append(a)
    for j, b in enumerate(ir_bodies):
        if j not in used_ir:
            fused.append(b)

    return fused


def _build_body(landmarks: dict[str, Vec3], source, skeleton_id: int) -> BodyResult:
    nose = landmarks["nose"]
    return BodyResult(
        source          = source,
        skeleton_id     = skeleton_id,
        nose            = nose,
        left_shoulder   = landmarks["left_shoulder"],
        right_shoulder  = landmarks["right_shoulder"],
        left_elbow      = landmarks["left_elbow"],
        right_elbow     = landmarks["right_elbow"],
        left_wrist      = landmarks["left_wrist"],
        right_wrist     = landmarks["right_wrist"],
        left_hip        = landmarks["left_hip"],
        right_hip       = landmarks["right_hip"],
        left_knee       = landmarks["left_knee"],
        right_knee      = landmarks["right_knee"],
        landmarks       = landmarks,
        face_center_px  = (nose.x, nose.y),
    )


# ── cross-modality association ──────────────────────────────────────────────

def associate_faces_to_bodies(faces: list[dict], bodies: list[BodyResult]) -> list[dict]:
    """
    Greedy nearest-neighbour match of insightface detections to skeletons.

    faces:   list of dicts with at least "bbox": (x1, y1, x2, y2)
    bodies:  list of BodyResult

    returns: list of {"face": dict, "body": BodyResult | None}
    """
    if not bodies:
        return [{"face": f, "body": None} for f in faces]

    candidates = []
    for fi, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]
        fcx, fcy = (x1 + x2) / 2, (y1 + y2) / 2
        for bi, body in enumerate(bodies):
            nx, ny = body.face_center_px
            d = _dist_xy(fcx, fcy, nx, ny)
            if d <= FACE_NOSE_MAX_DIST_PX:
                candidates.append((d, fi, bi))
    candidates.sort(key=lambda c: c[0])

    matched_face: dict[int, int] = {}
    used_bodies: set[int] = set()
    for d, fi, bi in candidates:
        if fi in matched_face or bi in used_bodies:
            continue
        matched_face[fi] = bi
        used_bodies.add(bi)

    results = []
    for fi, face in enumerate(faces):
        bi = matched_face.get(fi)
        results.append({"face": face, "body": bodies[bi] if bi is not None else None})
    return results


def associate_hands_to_bodies(hands: list[HandResult], bodies: list[BodyResult]) -> list[HandResult]:
    """
    Greedy nearest-neighbour match of each hand's wrist to the nearest
    unclaimed (skeleton_id, side) wrist slot. Mutates and returns `hands`
    with skeleton_id/side set. A hand with no match within threshold keeps
    skeleton_id=None/side=None (untracked — e.g. someone half out of frame).
    """
    if not bodies:
        return hands

    candidates = []
    for hi, hand in enumerate(hands):
        for body in bodies:
            for side, wrist in (("left", body.left_wrist), ("right", body.right_wrist)):
                d = _vec3_dist(hand.wrist, wrist)
                if d <= HAND_WRIST_MAX_DIST_PX:
                    candidates.append((d, hi, body.skeleton_id, side))
    candidates.sort(key=lambda c: c[0])

    used_slots: set[tuple[int, str]] = set()
    matched_hands: set[int] = set()
    for d, hi, skeleton_id, side in candidates:
        slot = (skeleton_id, side)
        if hi in matched_hands or slot in used_slots:
            continue
        used_slots.add(slot)
        matched_hands.add(hi)
        hands[hi].skeleton_id = skeleton_id
        hands[hi].side = side

    return hands


class BodyTracker:
    def __init__(self, max_poses: int = 4, confidence: float = 0.5, use_gpu: bool = True):
        if not os.path.exists(MODEL_PATH):
            print("downloading pose landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        # mediapipe's GPU delegate compiles to OpenGL ES compute shaders, not
        # CUDA/ROCm, so it runs on any Mesa-capable GPU — AMD discrete and APU
        # alike. Falls back to CPU rather than dying on a machine that can't
        # give it a GL context (headless CI, no DRM node).
        self._detector = _mp.create_detector(
            vision.PoseLandmarker,
            lambda delegate: vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH, delegate=delegate),
                num_poses=max_poses,
                min_pose_detection_confidence=confidence,
                min_pose_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            ),
            use_gpu,
            "pose landmarker",
        )
        self._ids = _SkeletonIdTracker()
        self._roi_tracker = _vision.LandmarkRoiTracker()

    def process(
        self,
        rgb_frame:   np.ndarray,
        depth_frame: np.ndarray,
        ir_frame:    np.ndarray | None = None,
        config:      GestureConfig | None = None,
        enable_fusion: bool = False,
        roi: Roi | None | Literal["auto"] = "auto",
        timestamp: float = 0.0,
        scale_inv: int = 2,
    ) -> list[BodyResult]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32  — optional, required for IR fallback/fusion
        config:      GestureConfig — thresholds; defaults if omitted
        enable_fusion: run both RGB and IR every frame and merge (~2x inference cost)
        roi: "auto" (default) computes a depth-based crop around whoever's
             in frame each call (see _vision.PersonRoiTracker) so mediapipe
             runs on a smaller image instead of the full 1920x1080 frame.
             Pass an explicit (x1,y1,x2,y2) to reuse a roi computed once and
             shared with HandTracker.process() in the same frame (that's
             what nkit/bridge.py does). Pass None to disable cropping.
        timestamp: seconds (e.g. time.monotonic()) — drives the skeleton
             tracker's wall-clock grace period, not frame count.

        returns list of BodyResult, one per detected person, each with a
        skeleton_id stable across frames (best-effort).
        """
        config = config or GestureConfig()
        # "auto" is now detect-then-track off our own previous detections, so
        # it needs a real timestamp to age out — a caller leaving timestamp at
        # its 0.0 default would hold the last crop forever.
        resolved_roi = self._roi_tracker.roi(timestamp) if roi == "auto" else roi

        if enable_fusion and ir_frame is not None:
            rgb_img, ir_img, rgb_w, ir_w, roi_used = _vision.detection_images_fused(rgb_frame, ir_frame, resolved_roi, mirror=True, scale_inv=scale_inv)
            rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            ir_mp  = mp.Image(image_format=mp.ImageFormat.SRGB, data=ir_img)
            rgb_result = self._detector.detect(rgb_mp)
            ir_result  = self._detector.detect(ir_mp)

            rgb_bodies = [_landmarks_xy_from_result(l, roi_used) for l in (rgb_result.pose_landmarks or [])]
            ir_bodies  = [_landmarks_xy_from_result(l, roi_used) for l in (ir_result.pose_landmarks or [])]
            fused_lm_list = _fuse_body_lists(rgb_bodies, ir_bodies, rgb_w, ir_w)
            source = "fused"
        else:
            img, source, roi_used = _vision.detection_image(rgb_frame, ir_frame, config, resolved_roi, mirror=True, scale_inv=scale_inv)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            result = self._detector.detect(mp_image)
            fused_lm_list = [_landmarks_xy_from_result(l, roi_used) for l in (result.pose_landmarks or [])]

        if not fused_lm_list:
            self._ids.assign([], timestamp)
            return []

        centroids = [(lm["nose"].x, lm["nose"].y) for lm in fused_lm_list]
        skeleton_ids = self._ids.assign(centroids, timestamp)

        bodies = []
        for lm, sid in zip(fused_lm_list, skeleton_ids):
            lm = _resample_z(lm, depth_frame)
            bodies.append(_build_body(lm, source, sid))

        if roi == "auto":
            # only aim our own tracker when we own it; a caller passing an
            # explicit roi drives its own (see bridge.py)
            self._roi_tracker.update(bodies, timestamp)
        return bodies

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
