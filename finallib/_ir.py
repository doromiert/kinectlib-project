"""
nzk/_ir.py — shared IR preprocessing utilities

single source of truth for IR fallback logic used by hands, body, and face modules.
"""

import cv2
import numpy as np

RGB_W,   RGB_H   = 1920, 1080
DEPTH_W, DEPTH_H = 512,  424

# mean luminance below this switches detection to IR
IR_FALLBACK_THRESHOLD = 60


def is_dark(rgb_frame: np.ndarray, threshold: int = IR_FALLBACK_THRESHOLD) -> bool:
    """True when the RGB frame is too dark for reliable detection."""
    gray = cv2.cvtColor(rgb_frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def ir_to_detection_image(ir: np.ndarray) -> np.ndarray:
    """
    IR float32 (512×424) → uint8 RGB (1920×1080) suitable for mediapipe / insightface.
    normalises to uint8, applies CLAHE for contrast, upscales to RGB resolution.
    """
    valid_max = np.percentile(ir[ir > 0], 99) if np.any(ir > 0) else 1.0
    ir_norm = np.clip(ir, 0, valid_max)
    ir_norm = (ir_norm / ir_norm.max() * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    ir_enhanced = clahe.apply(ir_norm)

    ir_large = cv2.resize(ir_enhanced, (RGB_W, RGB_H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(ir_large, cv2.COLOR_GRAY2RGB)


def enhance_rgb(rgb_frame: np.ndarray) -> np.ndarray:
    """CLAHE-enhanced RGB → uint8 RGB, ready for model input."""
    bgr = rgb_frame[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detection_image(rgb_frame: np.ndarray, ir_frame: np.ndarray | None) -> tuple[np.ndarray, str]:
    """
    Choose the best detection image given current lighting.
    Returns (image_uint8_rgb, source) where source is "rgb" or "ir".
    """
    if ir_frame is not None and is_dark(rgb_frame):
        return ir_to_detection_image(ir_frame), "ir"
    return enhance_rgb(rgb_frame), "rgb"


def depth_at(depth: np.ndarray, x: int, y: int, radius: int = 3) -> float:
    """Median depth in a small patch around (x, y). Returns 0.0 if no valid data."""
    x1 = max(0, x - radius)
    x2 = min(DEPTH_W - 1, x + radius)
    y1 = max(0, y - radius)
    y2 = min(DEPTH_H - 1, y + radius)
    patch = depth[y1:y2, x1:x2]
    valid = patch[patch > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0
