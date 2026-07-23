"""
hands.py — 3D hand tracking using mediapipe + kinect depth + IR fallback

in good lighting: uses RGB for detection
in low light:     falls back to IR (upscaled, normalized to uint8 grayscale→RGB)
always:           uses depth for Z coords

usage:
    from kinect import Kinect
    from hands import HandTracker

    with Kinect() as k, HandTracker() as tracker:
        rgb, depth, ir = k.get_frames()
        hands = tracker.process(rgb, depth, ir)
        for hand in hands:
            print(hand["wrist"])  # (x_px, y_px, z_mm)
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

LANDMARK_NAMES = {
    0:  "wrist",       1:  "thumb_cmc",  2:  "thumb_mcp",
    3:  "thumb_ip",    4:  "thumb_tip",  5:  "index_mcp",
    6:  "index_pip",   7:  "index_dip",  8:  "index_tip",
    9:  "middle_mcp",  10: "middle_pip", 11: "middle_dip",
    12: "middle_tip",  13: "ring_mcp",   14: "ring_pip",
    15: "ring_dip",    16: "ring_tip",   17: "pinky_mcp",
    18: "pinky_pip",   19: "pinky_dip",  20: "pinky_tip",
}

RGB_W,   RGB_H   = 1920, 1080
DEPTH_W, DEPTH_H = 512,  424
IR_W,    IR_H    = 512,  424   # IR has same res as depth

SCALE_X = DEPTH_W / RGB_W
SCALE_Y = DEPTH_H / RGB_H

# brightness threshold below which we switch to IR
# 0-255, tune to taste
IR_FALLBACK_THRESHOLD = 60

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"


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
    """
    convert IR float32 (512x424) → uint8 RGB (1080x1920)
    so mediapipe can run on it at full RGB resolution
    normalizes and CLAHEs to maximize contrast
    """
    # normalize to 0-255
    ir_norm = np.clip(ir, 0, np.percentile(ir[ir > 0], 99) if np.any(ir > 0) else 1)
    ir_norm = (ir_norm / ir_norm.max() * 255).astype(np.uint8)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    ir_enhanced = clahe.apply(ir_norm)

    # upscale to RGB resolution
    ir_large = cv2.resize(ir_enhanced, (RGB_W, RGB_H), interpolation=cv2.INTER_LINEAR)

    # mediapipe wants 3-channel RGB
    return cv2.cvtColor(ir_large, cv2.COLOR_GRAY2RGB)


def _landmarks_from_result(result, depth_frame, coord_scale_x=1.0, coord_scale_y=1.0):
    """
    extract landmark dicts from a mediapipe result
    coord_scale: if detection ran on a differently-sized image, scale back to RGB coords
    """
    if not result.hand_landmarks:
        return []

    hands = []
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        landmarks = {}
        for idx, lm in enumerate(hand_landmarks):
            x_rgb = int(lm.x * RGB_W * coord_scale_x)
            y_rgb = int(lm.y * RGB_H * coord_scale_y)
            x_rgb = max(0, min(RGB_W - 1, x_rgb))
            y_rgb = max(0, min(RGB_H - 1, y_rgb))
            x_d   = max(0, min(DEPTH_W - 1, int(x_rgb * SCALE_X)))
            y_d   = max(0, min(DEPTH_H - 1, int(y_rgb * SCALE_Y)))
            z_mm  = _depth_at(depth_frame, x_d, y_d)
            landmarks[LANDMARK_NAMES[idx]] = (x_rgb, y_rgb, z_mm)

        label = handedness[0].category_name
        hands.append({
            "hand":       label,
            "landmarks":  landmarks,
            "wrist":      landmarks["wrist"],
            "index_tip":  landmarks["index_tip"],
            "middle_tip": landmarks["middle_tip"],
            "thumb_tip":  landmarks["thumb_tip"],
            "source":     "rgb",   # overridden below if IR
        })
    return hands


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

    def process(self, rgb_frame: np.ndarray, depth_frame: np.ndarray,
                ir_frame: np.ndarray = None) -> list[dict]:
        """
        rgb_frame:   (1080, 1920, 4) uint8  BGRX
        depth_frame: (424, 512)      float32 mm
        ir_frame:    (424, 512)      float32 — optional, used as fallback in low light

        returns list of hand dicts with 3D landmark coords
        """
        use_ir = (
            ir_frame is not None
            and _is_dark(rgb_frame)
        )

        if use_ir:
            detect_img = _ir_to_detection_image(ir_frame)
        else:
            bgr = rgb_frame[:, :, :3]
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            detect_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_img)
        result = self._detector.detect(mp_image)

        hands = _landmarks_from_result(result, depth_frame)

        if use_ir:
            for h in hands:
                h["source"] = "ir"

        return hands

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
    with Kinect() as k, HandTracker() as tracker:
        for _ in range(30):
            k.get_frames()

        print("wave your hand, grabbing frame...")
        rgb, depth, ir = k.get_frames()

        if rgb is None:
            print("timeout")
        else:
            hands = tracker.process(rgb, depth, ir)
            if not hands:
                print("no hands detected")
            else:
                for hand in hands:
                    print(f"\n{hand['hand']} hand (source: {hand['source']}):")
                    print(f"  wrist:      {hand['wrist']}")
                    print(f"  index tip:  {hand['index_tip']}")

            out = rgb[:, :, :3].copy()
            for hand in hands:
                for name, (x, y, z) in hand["landmarks"].items():
                    cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
            cv2.imwrite("hands.png", out)
            print("saved hands.png")
