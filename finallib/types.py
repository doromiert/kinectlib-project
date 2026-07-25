"""
nzk/types.py — shared types for the Negative Zero Kinect SDK
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


Source = Literal["rgb", "ir"]


@dataclass
class Vec3:
    x: int      # px in RGB space (1920×1080)
    y: int      # px in RGB space
    z: float    # mm from depth sensor; 0.0 = no depth data

    def __iter__(self):
        """unpack as (x, y, z) for backwards compat with tuple-based code"""
        return iter((self.x, self.y, self.z))

    def xy(self) -> tuple[int, int]:
        return (self.x, self.y)

    def xyz(self) -> tuple[int, int, float]:
        return (self.x, self.y, self.z)


@dataclass
class HandResult:
    hand:        Literal["Left", "Right"]
    source:      Source
    wrist:       Vec3
    thumb_tip:   Vec3
    index_tip:   Vec3
    middle_tip:  Vec3
    ring_tip:    Vec3
    pinky_tip:   Vec3
    landmarks:   dict[str, Vec3]  # all 21 mediapipe landmarks by name
    is_fist:     bool             # True when enough fingers are curled
    fist_score:  float            # 0.0–1.0 mean curl across index/middle/ring/pinky


@dataclass
class BodyResult:
    source:          Source
    nose:            Vec3
    left_shoulder:   Vec3
    right_shoulder:  Vec3
    left_elbow:      Vec3
    right_elbow:     Vec3
    left_wrist:      Vec3
    right_wrist:     Vec3
    left_hip:        Vec3
    right_hip:       Vec3
    left_knee:       Vec3
    right_knee:      Vec3
    landmarks:       dict[str, Vec3]  # all 33 mediapipe landmarks by name
    face_center_px:  tuple[int, int]  # (nose_x, nose_y) convenience


@dataclass
class FaceResult:
    name:        str                     # best match name, or "unknown"
    confidence:  float                   # cosine similarity 0.0–1.0
    known:       bool                    # False if below CONFIDENCE_THRESHOLD
    bbox:        tuple[int,int,int,int]  # x1,y1,x2,y2 in RGB px
    embedding:   np.ndarray              # (512,) raw insightface embedding
    source:      Source
    body:        BodyResult | None       # skeleton this face was matched to


@dataclass
class WakeWordResult:
    mode:        Literal["assistant", "stt"]  # which wake word triggered
    wake_word:   str                          # "hey_zane" | "zane_write"
    text:        str                          # whisper transcript
    language:    str                          # "en" | "pl" | etc.
    words:       list[dict]                   # [{word, start, end, probability}]
    confidence:  float                        # openwakeword detection score


@dataclass
class RawFrames:
    rgb:   np.ndarray   # (1080, 1920, 4) uint8  BGRX
    depth: np.ndarray   # (424,  512)     float32 mm
    ir:    np.ndarray   # (424,  512)     float32 raw intensity


@dataclass
class ZaneFrame:
    """One fully-processed Kinect frame — all enabled modalities."""
    frames:  RawFrames
    bodies:  list[BodyResult]
    hands:   list[HandResult]
    faces:   list[FaceResult]
    # voice arrives async via callback — not in ZaneFrame


@dataclass
class CaptureResult:
    """Returned by Enroller.try_capture() every call."""
    captured:  bool                    # a capture actually happened this call
    complete:  bool                    # all cells done
    cell:      tuple[int, int] | None  # (col, row) of detected position
    n_done:    int                     # total captures so far
    n_needed:  int                     # total needed to finish
    face:      FaceResult | None       # face that was captured (if captured=True)
