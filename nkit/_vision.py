"""
nkit/_vision.py — shared RGB/IR/depth preprocessing utilities

Single source of truth for turning raw Kinect frames into detector input,
used by hands.py, body.py, and face/recognize.py.

Two modes, both driven by GestureConfig.ir_fallback_brightness:

  adaptive (default): pick ONE 2D source per frame — RGB or IR — based on
  brightness. Cheap: one detector pass per frame.

  fusion (enable_fusion=True): run the detector on BOTH RGB and IR every
  frame; the caller (body.py / hands.py) merges the two landmark sets via
  merge_landmarks() below, weighted by each source's contrast score from
  this module. Roughly 2x inference cost.
"""

from __future__ import annotations
import math
from typing import Literal
import cv2
import numpy as np

from .types import Vec3, GestureConfig

RGB_W,   RGB_H   = 1920, 1080
DEPTH_W, DEPTH_H = 512,  424

# depth space -> RGB space (inverse of the RGB->depth SCALE_X/Y used elsewhere)
RGB_PER_DEPTH_X = RGB_W / DEPTH_W
RGB_PER_DEPTH_Y = RGB_H / DEPTH_H

Source = Literal["rgb", "ir"]
Roi = tuple[int, int, int, int]   # (x1, y1, x2, y2) in RGB pixel space

# depth-based person ROI: only consider depth in this range "foreground" —
# steals the original 2010 Kinect's actual trick (depth pre-filters the
# expensive part) so mediapipe runs on a crop instead of the full frame
ROI_DEPTH_MIN_MM  = 400.0
ROI_DEPTH_MAX_MM  = 4000.0
ROI_MARGIN_FRAC   = 0.2     # padding around the detected foreground, as a fraction of its own size
ROI_MIN_SIZE_PX   = 500     # never crop tighter than this — avoids over-cropping on noisy/tiny detections
ROI_MIN_FOREGROUND_PX = 300  # below this many foreground depth pixels, don't trust it — use the full frame


ROI_MORPH_CLOSE_KERNEL = 7   # bridges small depth-noise gaps (e.g. ToF edge artifacts during fast motion)
ROI_SMOOTHING = 0.5          # 0=frozen, 1=no smoothing; EMA weight given to the new detection each frame


class PersonRoiTracker:
    """
    Stateful depth-based foreground crop. A plain per-frame "biggest
    connected blob" pick isn't robust to fast motion: pushing a hand
    quickly toward the camera can transiently separate it from the arm/
    torso in the depth mask (time-of-flight noise at fast-moving edges), and
    a stateless picker can suddenly jump onto a different, wrong blob —
    which crops out real body parts and drops pose tracking mid-gesture.

    Three things make this version robust to that instead of just fast:
      - a morphological close on the mask bridges small gaps before
        labeling, so a hand doesn't disconnect from the torso from a few
        noisy/missing depth pixels alone
      - the chosen component is whichever overlaps the PREVIOUS frame's roi
        most, not just whichever is biggest by area — so a momentary
        secondary blob can't hijack the crop
      - the returned box is exponentially smoothed against the previous
        one, so the crop doesn't visibly snap frame to frame

    One instance per tracked scene (BodyTracker/HandTracker/Recognizer each
    keep their own for standalone "auto" use; nkit/bridge.py keeps ONE
    shared instance and computes the roi once per frame for all three).
    """

    def __init__(self, smoothing: float = ROI_SMOOTHING):
        self._smoothing = smoothing
        self._prev_roi: Roi | None = None

    def find(self, depth_frame: np.ndarray) -> Roi | None:
        mask = ((depth_frame > ROI_DEPTH_MIN_MM) & (depth_frame < ROI_DEPTH_MAX_MM)).astype(np.uint8)
        if int(mask.sum()) < ROI_MIN_FOREGROUND_PX:
            self._prev_roi = None
            return None

        kernel = np.ones((ROI_MORPH_CLOSE_KERNEL, ROI_MORPH_CLOSE_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:   # only the background label — nothing found
            self._prev_roi = None
            return None

        label = self._pick_component(stats, n_labels)
        if label is None:
            self._prev_roi = None
            return None

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        raw_roi = _pad_and_clamp_roi(
            int(x * RGB_PER_DEPTH_X), int(y * RGB_PER_DEPTH_Y),
            int((x + w) * RGB_PER_DEPTH_X), int((y + h) * RGB_PER_DEPTH_Y),
        )
        smoothed = self._smooth(raw_roi)
        self._prev_roi = smoothed
        return smoothed

    def _pick_component(self, stats: np.ndarray, n_labels: int) -> int | None:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.max() < ROI_MIN_FOREGROUND_PX:
            return None
        biggest = 1 + int(np.argmax(areas))

        if self._prev_roi is None:
            return biggest

        # depth-space bbox of the previous (RGB-space) roi, to test overlap against
        px1, py1, px2, py2 = self._prev_roi
        dpx1, dpy1 = px1 / RGB_PER_DEPTH_X, py1 / RGB_PER_DEPTH_Y
        dpx2, dpy2 = px2 / RGB_PER_DEPTH_X, py2 / RGB_PER_DEPTH_Y

        best_label, best_overlap = None, 0.0
        for label in range(1, n_labels):
            if stats[label, cv2.CC_STAT_AREA] < ROI_MIN_FOREGROUND_PX // 4:
                continue
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            ox1, oy1 = max(x, dpx1), max(y, dpy1)
            ox2, oy2 = min(x + w, dpx2), min(y + h, dpy2)
            overlap = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        # nothing overlaps where we last saw them (they moved fast, or a
        # different person entirely) — fall back to the biggest blob
        return best_label if best_label is not None else biggest

    def _smooth(self, raw: Roi) -> Roi:
        if self._prev_roi is None:
            return raw
        a = self._smoothing
        x1, y1, x2, y2 = (int(a * r + (1 - a) * p) for r, p in zip(raw, self._prev_roi))
        return (x1, y1, x2, y2)


# landmark-driven roi (detect-then-track). Wider margin than the depth crop
# used, because this box is grown from where limbs WERE — it has to still
# contain them a frame later, after motion, or tracking oscillates: crop
# tight, lose the arm, fall back to full frame, re-acquire, crop tight again.
LANDMARK_ROI_MARGIN_FRAC = 0.35
LANDMARK_ROI_MIN_SIZE_PX = 320
LANDMARK_ROI_GRACE_S     = 0.4    # keep cropping this long after detections stop
LANDMARK_ROI_SMOOTHING   = 0.5


class LandmarkRoiTracker:
    """
    Crops the next frame to where the last frame's skeletons actually were.

    Replaces the depth-blob approach (see git history for PersonRoiTracker),
    which assumed the person is the nearest large connected thing. Measured at
    desk range that fails completely: everything in a small room sits inside
    any sane depth band, floor and walls stay connected to everything, and the
    "person" blob spans the frame — full-frame roi on 60/60 frames. Tightening
    the band didn't help, and any band that works across a room breaks at a
    desk, which nkit has to do both of.

    Landmarks have no such assumption. They're also self-correcting: the crop
    is only ever as good as the last detection, and when detection fails the
    grace period expires and it re-acquires on the full frame.

    Why this matters beyond speed: mediapipe's pose model resizes its input to
    256x256 regardless. Handing it a full 1920x1080 frame with someone at 4m
    leaves them ~95px tall in model space; cropping to the person fills that
    256px at any distance. The crop is the range fix, not just a perf win.
    """

    def __init__(self, smoothing: float = LANDMARK_ROI_SMOOTHING, grace_s: float = LANDMARK_ROI_GRACE_S):
        self._smoothing = smoothing
        self._grace_s = grace_s
        self._roi: Roi | None = None
        self._last_seen_ts: float | None = None

    def roi(self, timestamp: float) -> Roi | None:
        """The crop for THIS frame, or None (full frame) to re-acquire."""
        if self._roi is None or self._last_seen_ts is None:
            return None
        if timestamp - self._last_seen_ts > self._grace_s:
            self._roi = None
            return None
        return self._roi

    def update(self, bodies, timestamp: float) -> None:
        """Feed back this frame's detections to aim the next frame's crop."""
        xs: list[float] = []
        ys: list[float] = []
        for b in bodies:
            for lm in b.landmarks.values():
                # mediapipe emits off-frame landmarks for occluded joints it
                # is guessing at; letting those into the box drags it away
                # from the person and it inflates toward full-frame again
                if 0 <= lm.x < RGB_W and 0 <= lm.y < RGB_H:
                    xs.append(lm.x)
                    ys.append(lm.y)
        if not xs:
            return

        raw = _pad_and_clamp_roi(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)),
                                 margin_frac=LANDMARK_ROI_MARGIN_FRAC,
                                 min_size_px=LANDMARK_ROI_MIN_SIZE_PX)
        if self._roi is None:
            self._roi = raw
        else:
            a = self._smoothing
            self._roi = tuple(int(a * r + (1 - a) * p) for r, p in zip(raw, self._roi))  # type: ignore[assignment]
        self._last_seen_ts = timestamp


def _pad_and_clamp_roi(x1: int, y1: int, x2: int, y2: int,
                       margin_frac: float = ROI_MARGIN_FRAC,
                       min_size_px: int = ROI_MIN_SIZE_PX) -> Roi:
    mx = int((x2 - x1) * margin_frac)
    my = int((y2 - y1) * margin_frac)
    x1, x2 = max(0, x1 - mx), min(RGB_W, x2 + mx)
    y1, y2 = max(0, y1 - my), min(RGB_H, y2 + my)

    if x2 - x1 < min_size_px:
        cx = (x1 + x2) // 2
        half = min_size_px // 2
        x1, x2 = max(0, cx - half), min(RGB_W, cx + half)
    if y2 - y1 < min_size_px:
        cy = (y1 + y2) // 2
        half = min_size_px // 2
        y1, y2 = max(0, cy - half), min(RGB_H, cy + half)

    return (x1, y1, x2, y2)


def point_crop_roi(x: int, y: int, half_size_px: int) -> Roi:
    """
    Small square roi (side 2*half_size_px) centered on (x, y), clamped to
    frame bounds. Used to constrain hand/face detection to a tight region
    around a skeleton's own wrist/nose estimate instead of searching the
    whole frame — a false detection elsewhere (a foot, background clutter)
    can't be found at all if the search never looks there.
    """
    x1, x2 = max(0, x - half_size_px), min(RGB_W, x + half_size_px)
    y1, y2 = max(0, y - half_size_px), min(RGB_H, y + half_size_px)
    return (x1, y1, x2, y2)


# is_dark averages the whole frame down to one number, so it doesn't need
# every pixel — every 8th in each axis lands within a fraction of a grey
# level of the true mean, far inside any sane threshold's precision.
# Reading all 1920x1080 cost ~13ms (9ms of that just the non-contiguous
# [:, :, :3] slice copy) and detection_image() calls this once per
# invocation — three times a frame (body + one per wrist), so ~39ms/frame
# went to deciding "is it dark?". Subsampled it's ~0.15ms.
IS_DARK_SUBSAMPLE = 8


def is_dark(rgb_frame: np.ndarray, threshold: float) -> bool:
    """True when the RGB frame is too dark for reliable detection."""
    small = np.ascontiguousarray(rgb_frame[::IS_DARK_SUBSAMPLE, ::IS_DARK_SUBSAMPLE, :3])
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def ir_to_detection_image(ir: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    IR float32 (native depth resolution, or a crop of it) -> uint8 RGB
    suitable for mediapipe / insightface. normalises to uint8, applies
    CLAHE for contrast, resizes to target_size (default: full RGB resolution).
    """
    target = target_size or (RGB_W, RGB_H)
    valid_max = np.percentile(ir[ir > 0], 99) if np.any(ir > 0) else 1.0
    ir_norm = np.clip(ir, 0, valid_max)
    denom = ir_norm.max()
    ir_norm = (ir_norm / denom * 255).astype(np.uint8) if denom > 0 else ir_norm.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    ir_enhanced = clahe.apply(ir_norm)

    ir_large = cv2.resize(ir_enhanced, target, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(ir_large, cv2.COLOR_GRAY2RGB)


def enhance_rgb(rgb_frame: np.ndarray) -> np.ndarray:
    """CLAHE-enhanced RGB -> uint8 RGB, ready for model input."""
    bgr = rgb_frame[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def contrast_score(img_uint8_rgb: np.ndarray) -> float:
    """Cheap proxy for 'how much reliable detail is in this detection image'."""
    gray = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.std(gray))


def _stride_down(img: np.ndarray, scale_inv: int) -> np.ndarray:
    """Cheap 1/N downscale by taking every Nth pixel. See detection_image()."""
    if scale_inv <= 1:
        return img
    return np.ascontiguousarray(img[::scale_inv, ::scale_inv])


def _crop_ir_to_roi(ir_frame: np.ndarray, roi: Roi) -> np.ndarray:
    """roi is in RGB pixel space — map it down to the IR/depth frame's native resolution."""
    x1, y1, x2, y2 = roi
    dx1, dy1 = int(x1 / RGB_PER_DEPTH_X), int(y1 / RGB_PER_DEPTH_Y)
    dx2, dy2 = int(x2 / RGB_PER_DEPTH_X), int(y2 / RGB_PER_DEPTH_Y)
    dx2, dy2 = max(dx2, dx1 + 1), max(dy2, dy1 + 1)
    return ir_frame[dy1:dy2, dx1:dx2]


def detection_image(
    rgb_frame: np.ndarray,
    ir_frame: np.ndarray | None,
    config: GestureConfig,
    roi: Roi | None = None,
    mirror: bool = False,
    force_source: Source | None = None,
    scale_inv: int = 1,
) -> tuple[np.ndarray, Source, Roi]:
    """
    Adaptive mode: choose the single best detection image for current lighting.
    roi (RGB pixel space) crops before any preprocessing if given — pass
    None to use the full frame. Returns (image_uint8_rgb, source, roi_used);
    roi_used is always concrete, even when roi was None, so callers can map
    normalized landmark coords back to full-frame pixel space uniformly.

    mirror: horizontally flip the returned image. mediapipe's Pose/Hands
    landmark naming (left_*/right_*, and Hands' own handedness label) both
    assume mirrored/selfie-style input — Kinect's feed isn't mirrored, so
    without this their L/R semantics come out backwards. body.py/hands.py
    pass mirror=True; face/recognize.py leaves it False so face-recognition
    embeddings stay consistent with how the enrolled gallery was captured.
    Callers that mirror must un-mirror x when mapping landmarks back to
    full-frame pixel space — see body.py/hands.py's landmark extraction.

    force_source: skip the brightness-based adaptive pick and always use
    this source (falls back to "rgb" if "ir" is forced but ir_frame is
    None). hands.py uses this for GestureConfig.hand_prefer_ir.

    scale_inv: feed the detector a 1/N-scale image. Landmarks come back
    normalized, so roi_used still maps them to full-frame pixels unchanged
    and nothing downstream needs to know. The models run at their own small
    input size regardless, so this costs accuracy only once the roi gets
    near that size. Applied BEFORE preprocessing, not after — CLAHE is the
    expensive part, and shrinking afterwards would pay it at full res for
    nothing.
    """
    roi_used = roi if roi is not None else (0, 0, RGB_W, RGB_H)
    x1, y1, x2, y2 = roi_used
    scale_inv = max(1, scale_inv)
    target_size = (max(1, (x2 - x1) // scale_inv), max(1, (y2 - y1) // scale_inv))

    if force_source == "ir" and ir_frame is not None:
        use_ir = True
    elif force_source == "rgb":
        use_ir = False
    else:
        use_ir = ir_frame is not None and is_dark(rgb_frame, config.ir_fallback_brightness)

    if use_ir:
        # IR is natively 512x424, so target_size drives the one resize it
        # already does — a smaller target is strictly less work, and skips
        # upscaling past the source resolution for detail that isn't there.
        ir_crop = _crop_ir_to_roi(ir_frame, roi_used) if roi is not None else ir_frame
        img = ir_to_detection_image(ir_crop, target_size)
    else:
        rgb_crop = rgb_frame[y1:y2, x1:x2] if roi is not None else rgb_frame
        # stride-slice rather than cv2.resize: resizing the frame costs
        # ~12ms whatever the interpolation (the [:, :, :3] slice off BGRA is
        # non-contiguous, forcing a full copy), striding costs ~1ms.
        img = enhance_rgb(_stride_down(rgb_crop, scale_inv))

    source: Source = "ir" if use_ir else "rgb"
    return (cv2.flip(img, 1) if mirror else img), source, roi_used


def detection_images_fused(
    rgb_frame: np.ndarray,
    ir_frame: np.ndarray,
    roi: Roi | None = None,
    mirror: bool = False,
    scale_inv: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, float, Roi]:
    """
    Fusion mode: preprocess BOTH sources (optionally cropped to roi), return
    their images plus a contrast weight each. Caller runs the detector on
    both and calls merge_landmarks(). Returns (..., roi_used) — see detection_image().
    mirror: see detection_image().
    scale_inv: see detection_image(). Both sources must come out the same
    size for merge_landmarks() to compare them, so ir renders straight to
    the already-scaled target that rgb strides down to.
    """
    roi_used = roi if roi is not None else (0, 0, RGB_W, RGB_H)
    x1, y1, x2, y2 = roi_used
    scale_inv = max(1, scale_inv)

    rgb_crop = rgb_frame[y1:y2, x1:x2] if roi is not None else rgb_frame
    ir_crop  = _crop_ir_to_roi(ir_frame, roi_used) if roi is not None else ir_frame

    rgb_img = enhance_rgb(_stride_down(rgb_crop, scale_inv))
    # match rgb exactly rather than recomputing from the roi — integer
    # division on each axis can differ by a pixel otherwise
    h, w = rgb_img.shape[:2]
    ir_img = ir_to_detection_image(ir_crop, (w, h))

    if mirror:
        rgb_img = cv2.flip(rgb_img, 1)
        ir_img  = cv2.flip(ir_img, 1)
    return rgb_img, ir_img, contrast_score(rgb_img), contrast_score(ir_img), roi_used


def merge_landmarks(
    rgb_landmarks: dict[str, Vec3] | None,
    ir_landmarks: dict[str, Vec3] | None,
    rgb_weight: float,
    ir_weight: float,
) -> dict[str, Vec3] | None:
    """
    Weighted-average merge of two same-shaped landmark dicts (x, y only —
    z is stale after merging x/y and must be re-sampled from depth by the
    caller at the new merged position). Falls back to whichever source
    actually detected something if only one did.
    """
    if rgb_landmarks is None:
        return ir_landmarks
    if ir_landmarks is None:
        return rgb_landmarks

    total = rgb_weight + ir_weight
    if total <= 0:
        rgb_weight = ir_weight = 1.0
        total = 2.0

    merged: dict[str, Vec3] = {}
    for name, a in rgb_landmarks.items():
        b = ir_landmarks.get(name, a)
        mx = (a.x * rgb_weight + b.x * ir_weight) / total
        my = (a.y * rgb_weight + b.y * ir_weight) / total
        merged[name] = Vec3(int(round(mx)), int(round(my)), a.z)
    return merged


# Kinect v2 color camera FOV — commonly-cited spec values (84.1 x 53.8 deg),
# not derived from this specific unit's own calibration. Good enough for
# converting a pixel displacement at a known depth into an approximate
# real-world mm displacement (pinhole camera model); if push-gesture feel
# is off, this is the first thing to re-derive from an actual calibration.
RGB_FOV_X_DEG = 84.1
RGB_FOV_Y_DEG = 53.8
RGB_FOCAL_PX_X = (RGB_W / 2) / math.tan(math.radians(RGB_FOV_X_DEG) / 2)
RGB_FOCAL_PX_Y = (RGB_H / 2) / math.tan(math.radians(RGB_FOV_Y_DEG) / 2)


def pixel_delta_to_mm(dx_px: float, dy_px: float, z_mm: float) -> tuple[float, float]:
    """Approximate real-world mm displacement for a pixel displacement at a given depth."""
    if z_mm <= 0:
        return 0.0, 0.0
    return dx_px * z_mm / RGB_FOCAL_PX_X, dy_px * z_mm / RGB_FOCAL_PX_Y


def depth_at(depth: np.ndarray, x: int, y: int, radius: int = 3) -> float:
    """Median depth in a small patch around (x, y). Returns 0.0 if no valid data."""
    x1 = max(0, x - radius)
    x2 = min(DEPTH_W - 1, x + radius)
    y1 = max(0, y - radius)
    y2 = min(DEPTH_H - 1, y + radius)
    patch = depth[y1:y2, x1:x2]
    valid = patch[patch > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0
