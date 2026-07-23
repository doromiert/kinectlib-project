"""
pose.py — full-body skeleton tracking via mediapipe pose landmarker + kinect depth

gives you 33 landmarks per person with real Z from depth sensor.
key landmarks exposed as shortcuts: nose, left/right shoulder, left/right wrist,
left/right hip — enough to associate a face bbox to a skeleton and know
body position for the UI.

usage:
    from kinect import Kinect
    from pose import PoseTracker

    with Kinect() as k, PoseTracker() as tracker:
        rgb, depth, ir = k.get_frames()
        bodies = tracker.process(rgb, depth, ir)
        for body in bodies:
            print(body["nose"])          # (x_px, y_px, z_mm)
            print(body["left_wrist"])    # same
            print(body["landmarks"])     # all 33 by name

associating a face to a skeleton:
    from pose import associate_faces_to_bodies

    associations = associate_faces_to_bodies(faces, bodies)
    # returns list of {"face": face_dict, "body": body_dict}
    # face_dict must have "bbox": (x1, y1, x2, y2)
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# same constants as hands.py — keep in sync
RGB_W,   RGB_H   = 1920, 1080
DEPTH_W, DEPTH_H = 512,  424

SCALE_X = DEPTH_W / RGB_W
SCALE_Y = DEPTH_H / RGB_H

IR_FALLBACK_THRESHOLD = 60  # same as hands.py

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = "pose_landmarker.task"

# mediapipe pose landmark indices → names
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_NAMES = {
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

# shortcuts we expose directly on each body dict
KEY_LANDMARKS = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
]

# max pixel distance (in RGB space) to consider a face "belonging" to a skeleton
FACE_NOSE_MAX_DIST_PX = 200


def _depth_at(depth: np.ndarray, x: int, y: int, radius: int = 3) -> float:
    x1, x2 = max(0, x - radius), min(DEPTH_W - 1, x + radius)
    y1, y2 = max(0, y - radius), min(DEPTH_H - 1, y + radius)
    patch = depth[y1:y2, x1:x2]
    valid = patch[patch > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def _is_dark(rgb_frame: np.ndarray, threshold: int = IR_FALLBACK_THRESHOLD) -> bool:
    gray = cv2.cvtColor(rgb_frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def _ir_to_detection_image(ir: np.ndarray) -> np.ndarray:
    """IR float32 (512x424) → uint8 RGB (1920x1080) — same as hands.py"""
    ir_norm = np.clip(ir, 0, np.percentile(ir[ir > 0], 99) if np.any(ir > 0) else 1)
    ir_norm = (ir_norm / ir_norm.max() * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    ir_enhanced = clahe.apply(ir_norm)
    ir_large = cv2.resize(ir_enhanced, (RGB_W, RGB_H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(ir_large, cv2.COLOR_GRAY2RGB)


def _build_body(pose_landmarks, depth_frame, source: str) -> dict:
    landmarks = {}
    for idx, lm in enumerate(pose_landmarks):
        name = LANDMARK_NAMES[idx]

        # mediapipe gives normalized [0,1] coords
        x_rgb = int(np.clip(lm.x * RGB_W, 0, RGB_W - 1))
        y_rgb = int(np.clip(lm.y * RGB_H, 0, RGB_H - 1))

        x_d = int(np.clip(x_rgb * SCALE_X, 0, DEPTH_W - 1))
        y_d = int(np.clip(y_rgb * SCALE_Y, 0, DEPTH_H - 1))
        z_mm = _depth_at(depth_frame, x_d, y_d)

        landmarks[name] = (x_rgb, y_rgb, z_mm)

    body = {
        "landmarks": landmarks,
        "source":    source,
    }
    for key in KEY_LANDMARKS:
        body[key] = landmarks[key]

    # convenience: face center in RGB px (avg of nose + ears, ignoring missing Z)
    nose = landmarks["nose"]
    body["face_center_px"] = (nose[0], nose[1])

    return body


def associate_faces_to_bodies(faces: list[dict], bodies: list[dict]) -> list[dict]:
    """
    match insightface detections to pose skeletons by proximity.

    faces:  list of dicts with at least "bbox": (x1, y1, x2, y2)
    bodies: list of dicts from PoseTracker.process()

    returns list of {"face": ..., "body": ...}
    unmatched faces get body=None, unmatched bodies are ignored.

    matching is greedy nearest-neighbor (good enough for ≤4 people).
    """
    if not bodies:
        return [{"face": f, "body": None} for f in faces]

    results = []
    used_bodies = set()

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2

        best_body = None
        best_dist = float("inf")

        for i, body in enumerate(bodies):
            if i in used_bodies:
                continue
            nx, ny, _ = body["nose"]
            dist = ((face_cx - nx) ** 2 + (face_cy - ny) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_body = (i, body)

        if best_body and best_dist <= FACE_NOSE_MAX_DIST_PX:
            used_bodies.add(best_body[0])
            results.append({"face": face, "body": best_body[1]})
        else:
            results.append({"face": face, "body": None})

    return results


class PoseTracker:
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

    def process(self, rgb_frame: np.ndarray, depth_frame: np.ndarray,
                ir_frame: np.ndarray = None) -> list[dict]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32 — optional IR fallback

        returns list of body dicts, one per detected person
        """
        use_ir = ir_frame is not None and _is_dark(rgb_frame)

        if use_ir:
            detect_img = _ir_to_detection_image(ir_frame)
            source = "ir"
        else:
            bgr = rgb_frame[:, :, :3]
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            detect_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source = "rgb"

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_img)
        result = self._detector.detect(mp_image)

        if not result.pose_landmarks:
            return []

        return [
            _build_body(pose_lms, depth_frame, source)
            for pose_lms in result.pose_landmarks
        ]

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from kinect import Kinect

    print("opening kinect...")
    with Kinect() as k, PoseTracker() as tracker:
        for _ in range(30):
            k.get_frames()

        print("stand in front of the kinect, grabbing frame...")
        rgb, depth, ir = k.get_frames()

        if rgb is None:
            print("timeout")
        else:
            bodies = tracker.process(rgb, depth, ir)
            if not bodies:
                print("no bodies detected")
            else:
                for i, body in enumerate(bodies):
                    print(f"\nbody {i} (source: {body['source']}):")
                    print(f"  nose:           {body['nose']}")
                    print(f"  left_shoulder:  {body['left_shoulder']}")
                    print(f"  right_shoulder: {body['right_shoulder']}")
                    print(f"  left_wrist:     {body['left_wrist']}")
                    print(f"  right_wrist:    {body['right_wrist']}")

            out = rgb[:, :, :3].copy()
            colors = [(0, 255, 0), (0, 100, 255), (255, 0, 100), (255, 255, 0)]
            for i, body in enumerate(bodies):
                color = colors[i % len(colors)]
                for name, (x, y, z) in body["landmarks"].items():
                    cv2.circle(out, (x, y), 5, color, -1)
                # draw shoulder-to-shoulder line as sanity check
                lx, ly, _ = body["left_shoulder"]
                rx, ry, _ = body["right_shoulder"]
                cv2.line(out, (lx, ly), (rx, ry), color, 2)

            cv2.imwrite("pose.png", out)
            print("\nsaved pose.png")
