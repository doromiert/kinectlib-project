"""
nzk/body.py — full-body skeleton tracking

mediapipe pose landmarker + kinect depth for real Z coordinates.
IR fallback in low light (shared logic from nzk._ir).

usage:
    from nzk.body import BodyTracker
    from nzk.kinect import Kinect

    with Kinect() as k, BodyTracker() as tracker:
        rgb, depth, ir = k.get_frames()
        bodies = tracker.process(rgb, depth, ir)
        for body in bodies:
            print(body.nose)           # Vec3
            print(body.left_wrist)     # Vec3
"""

from __future__ import annotations
import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .types import BodyResult, Vec3, Source
from ._ir import (
    RGB_W, RGB_H, DEPTH_W, DEPTH_H,
    depth_at, detection_image,
)

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker.task"

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

KEY_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
]

# max px distance (RGB space) to match a face bbox to a skeleton nose
FACE_NOSE_MAX_DIST_PX = 200


def _build_body(pose_landmarks, depth_frame: np.ndarray, source: Source) -> BodyResult:
    landmarks: dict[str, Vec3] = {}
    for idx, lm in enumerate(pose_landmarks):
        name = LANDMARK_NAMES[idx]
        x_rgb = int(np.clip(lm.x * RGB_W, 0, RGB_W - 1))
        y_rgb = int(np.clip(lm.y * RGB_H, 0, RGB_H - 1))
        x_d   = int(np.clip(x_rgb * SCALE_X, 0, DEPTH_W - 1))
        y_d   = int(np.clip(y_rgb * SCALE_Y, 0, DEPTH_H - 1))
        z_mm  = depth_at(depth_frame, x_d, y_d)
        landmarks[name] = Vec3(x_rgb, y_rgb, z_mm)

    nose = landmarks["nose"]
    return BodyResult(
        source          = source,
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


def associate_faces_to_bodies(
    faces: list[dict],
    bodies: list[BodyResult],
) -> list[dict]:
    """
    Greedy nearest-neighbour match of insightface detections to skeletons.

    faces:   list of dicts with at least "bbox": (x1, y1, x2, y2)
    bodies:  list of BodyResult

    returns: list of {"face": dict, "body": BodyResult | None}
    """
    if not bodies:
        return [{"face": f, "body": None} for f in faces]

    results    = []
    used_bodies: set[int] = set()

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        fcx = (x1 + x2) / 2
        fcy = (y1 + y2) / 2

        best_body = None
        best_dist = float("inf")

        for i, body in enumerate(bodies):
            if i in used_bodies:
                continue
            nx, ny = body.face_center_px
            dist = ((fcx - nx) ** 2 + (fcy - ny) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_body = (i, body)

        if best_body and best_dist <= FACE_NOSE_MAX_DIST_PX:
            used_bodies.add(best_body[0])
            results.append({"face": face, "body": best_body[1]})
        else:
            results.append({"face": face, "body": None})

    return results


class BodyTracker:
    def __init__(self, max_poses: int = 4, confidence: float = 0.5):
        if not os.path.exists(MODEL_PATH):
            print("downloading pose landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            num_poses=max_poses,
            min_pose_detection_confidence=confidence,
            min_pose_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def process(
        self,
        rgb_frame:   np.ndarray,
        depth_frame: np.ndarray,
        ir_frame:    np.ndarray | None = None,
    ) -> list[BodyResult]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32  — optional IR fallback

        returns list of BodyResult, one per detected person.
        """
        img, source = detection_image(rgb_frame, ir_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = self._detector.detect(mp_image)

        if not result.pose_landmarks:
            return []

        return [
            _build_body(lms, depth_frame, source)
            for lms in result.pose_landmarks
        ]

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
