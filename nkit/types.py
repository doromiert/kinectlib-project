"""
nkit/types.py — shared types for the Negative Zero Kinect Interface Toolkit
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


Source = Literal["rgb", "ir", "fused"]


@dataclass
class Vec3:
    x: int      # px in RGB space (1920x1080)
    y: int      # px in RGB space
    z: float    # mm from depth sensor; 0.0 = no depth data

    def __iter__(self):
        """unpack as (x, y, z) for tuple-style code"""
        return iter((self.x, self.y, self.z))

    def xy(self) -> tuple[int, int]:
        return (self.x, self.y)

    def xyz(self) -> tuple[int, int, float]:
        return (self.x, self.y, self.z)


@dataclass
class HandResult:
    hand:        Literal["Left", "Right"]     # mediapipe's own per-crop handedness guess
    source:      Source
    wrist:       Vec3
    thumb_tip:   Vec3
    index_tip:   Vec3
    middle_tip:  Vec3
    ring_tip:    Vec3
    pinky_tip:   Vec3
    landmarks:   dict[str, Vec3]        # all 21 mediapipe landmarks by name
    is_fist:     bool                   # True when enough fingers are curled
    fist_score:  float                  # 0.0-1.0 mean curl across index/middle/ring/pinky
    hand_confidence: float = 1.0        # mediapipe's handedness classification score
    skeleton_id: int | None = None      # set by associate_hands_to_bodies()
    side:        Literal["left", "right"] | None = None  # body-relative, not mediapipe's guess


@dataclass
class BodyResult:
    source:          Source
    skeleton_id:     int | None   # None if IdentityTracker dropped this detection (new-track throttle) — see identity.py
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
    landmarks:       dict[str, Vec3]    # all 33 mediapipe landmarks by name
    face_center_px:  tuple[int, int]    # (nose_x, nose_y) convenience


@dataclass
class FaceResult:
    name:        str                      # best match name, or "unknown"
    confidence:  float                    # cosine similarity 0.0-1.0
    known:       bool                     # False if below CONFIDENCE_THRESHOLD
    bbox:        tuple[int, int, int, int]  # x1,y1,x2,y2 in RGB px
    embedding:   np.ndarray               # (512,) raw insightface embedding
    source:      Source
    skeleton_id: int | None               # skeleton this face was matched to


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
class nkitFrame:
    """One fully-processed Kinect frame — all enabled modalities."""
    frames:  RawFrames
    bodies:  list[BodyResult]
    hands:   list[HandResult]
    faces:   list[FaceResult]
    # voice arrives async via callback — not part of nkitFrame


@dataclass
class CaptureResult:
    """Returned by Enroller.try_capture() every call."""
    captured:  bool                    # a capture actually happened this call
    complete:  bool                    # all cells done
    cell:      tuple[int, int] | None  # (col, row) of detected position
    n_done:    int                     # total captures so far
    n_needed:  int                     # total needed to finish
    face:      FaceResult | None       # face that was captured (if captured=True)


# ── gestures ─────────────────────────────────────────────────────────────────

GestureKind = Literal[
    "cursor_move",   # continuous; x, y in screen space
    "grab_start", "grab_end",   # fist held past a debounce window
    "push",                     # palm-mode z-push — maps to a click
    "push_progress",             # continuous 0.0-1.0 while a push is arming; draw it
    "push_cancel",                # armed push abandoned — clear the indicator
    "swipe_edge",                  # hand entered from a screen edge, moved inward
    "grab_swipe",                   # fist held + moved vertically past threshold
]

Edge = Literal["left", "right", "top", "bottom"]
SwipeDirection = Literal["up", "down"]


@dataclass
class GestureEvent:
    skeleton_id: int
    side:        Literal["left", "right"]
    kind:        GestureKind
    x:           int | None
    y:           int | None
    edge:        Edge | None
    direction:   SwipeDirection | None
    timestamp:   float
    progress:    float | None = None   # 0.0-1.0, push_progress only


@dataclass
class GestureConfig:
    """
    Every tunable threshold the gesture/detection pipeline uses, in one
    mutable place. GestureTracker (and friends) read from a live instance
    every frame, so updates from a UI (e.g. debug sliders) apply
    immediately with no restart. save()/load() persist to JSON.
    """
    # fist detection (hands.py)
    fist_curl_threshold:  float = 0.55   # per-finger curl ratio to count as "curled"
    fist_min_curled:      int   = 3      # how many of index/middle/ring/pinky must curl

    # RGB/IR source selection (_vision.py)
    ir_fallback_brightness: float = 60.0   # mean luminance below this switches to IR

    # cursor mapping (gestures.py) — trims this fraction off each frame edge
    # before mapping linearly to screen space, so you don't have to physically
    # reach the camera's frame edge to hit the screen edge
    cursor_margin_frac: float = 0.15

    # gesture state machine (gestures.py)
    grab_debounce_ms:            float = 150.0   # sustained fist before grab_start fires
    push_delta_mm:                float = 30.0    # fallback only — absolute mode, used when no arm is visible
    # extend-and-retract must complete within this. 350ms (the old default)
    # is shorter than an actual deliberate push — measured against a
    # synthesised ~650ms poke it never fired at all, at 30fps or 12fps, since
    # the window closed mid-gesture and rebased the baseline. This is
    # wall-clock, so frame rate doesn't change it; it just needs to be longer
    # than a real arm movement takes.
    push_window_ms:                float = 600.0   # time window the delta must happen within
    push_debounce_ms:               float = 400.0   # minimum gap between two pushes

    # push, travel mode (see gestures.py:_update_push_travel) — the hand's
    # forward displacement from the torso, in mm, past a slow baseline.
    #
    # Measured rather than guessed: peri-tap averaging of a real session put
    # a *deliberate* forward reach at a ~174mm peak with a clean rise-and-fall,
    # while a deliberately SLIGHT push produced no coherent excursion at all —
    # it sat inside the +-50-100mm noise floor of torso-relative depth. So the
    # gesture has to be a real movement to be seeable, and the threshold sits
    # above what incidental reaching produces.
    #
    # push_progress events stream 0.0-1.0 as the gesture arms, so the UI can
    # draw a filling indicator. That's not decoration: it makes the gesture
    # self-correcting, since you can see it charging and pull back before it
    # fires, which is what makes a large deliberate push usable instead of
    # startling.
    push_travel_mm:        float = 260.0   # forward travel that completes a push
    push_arm_mm:           float = 60.0    # travel before progress starts reporting
    push_release_frac:     float = 0.5     # fall back through this fraction to fire

    # push, arm-relative mode (legacy — see gestures.py:_update_push).
    # "reach" is |shoulder->wrist| / |shoulder->elbow|: roughly 1.0 with the
    # arm folded, ~2.0 fully extended. Being a RATIO of two body measurements
    # it's unitless and scale-free, so one threshold holds at desk range and
    # across a living room, where push_delta_mm (absolute mm) cannot — 30mm
    # is a deliberate poke up close and inside the depth noise floor at 4m.
    # It also cancels whole-body motion: leaning in moves shoulder and wrist
    # together, leaving reach unchanged, where an absolute delta reads a lean
    # as a push.
    push_reach_delta:      float = 0.22   # increase in reach ratio that arms a push
    push_reach_release:    float = 0.4    # fraction of the delta to fall back through to fire
    push_min_arm_mm:       float = 80.0   # below this |shoulder->elbow| the ratio is too noisy — use mm fallback

    # landmark smoothing (_filter.py, One Euro). min_cutoff trades stillness
    # jitter against lag; beta is how quickly smoothing yields once moving.
    # Raise min_cutoff if the cursor feels laggy, raise beta if fast motion
    # still drags behind.
    smoothing_min_cutoff: float = 1.0
    smoothing_beta:       float = 0.007
    swipe_edge_band_px:              int   = 120     # how close to a screen edge counts as "entering from it"
    swipe_min_distance_px:            int   = 150     # inward travel needed to fire swipe_edge
    swipe_max_duration_ms:             float = 600.0   # swipe must complete within this window
    grab_swipe_min_distance_px:         int   = 200     # vertical travel while grabbing to fire grab_swipe

    # hand detection plausibility (hands.py) — mediapipe's own handedness
    # classification score. Higher = stricter about accepting a NEW hand
    # (this is what rejects a foot/limb read as a hand); doesn't affect a
    # hand that's already tracked (see hand_hold_ms below for continuity).
    hand_confidence_threshold: float = 0.75

    # identity tracking (identity.py) — the "buffer around the limbs" each
    # tracked skeleton claims: a pose/hand detection within this radius (px)
    # of a track's last-known position counts as still that person.
    limb_buffer_px: float = 250.0
    face_buffer_px: float = 200.0
    # a detection that doesn't fall in ANY existing track's buffer only
    # gets to spawn a brand-new skeleton at most this often — a false
    # detection (mediapipe misreading a foot as a hand, a jitter blip) that
    # only shows up for a frame or two won't survive to the next check,
    # instead of spawning (and, worse, triggering face recognition for) a
    # perf-killing ghost track every single time it flickers.
    new_track_interval_s: float = 1.0
    # the ACTUAL fix for "a misfire spawns a ghost track": a brand-new
    # identity isn't trusted off a single frame at all — the same
    # approximate spot has to reappear this many separate times within
    # new_track_confirm_window_s before it's promoted to a real track (then
    # still subject to new_track_interval_s above). A one-off false
    # detection essentially never reappears at a precise, consistent
    # position multiple frames running the way a real hand/face does.
    new_track_confirm_frames: int = 3
    new_track_confirm_window_s: float = 1.0
    # opposite side of the same coin as hand_confidence_threshold: once a
    # hand IS tracked, hold its last real position/state for up to this
    # long if a frame's detection momentarily misses it (mediapipe flicker
    # — a genuinely-present hand still drops out for a frame here and
    # there), instead of the gesture tracker seeing a hard "gone" every time.
    hand_hold_ms: float = 200.0

    # hand detection source (hands.py) — prefer IR over the adaptive RGB/IR
    # switch. IR is lower-res but actively illuminated (not dependent on
    # ambient light/exposure the way RGB is), which plausibly means less
    # motion blur during fast gestures (swipes) — worth it especially at
    # this project's close-range (0.5-3m) seated use case where IR contrast
    # is strong. Live-toggleable to A/B test against plain adaptive.
    hand_prefer_ir: bool = True

    # face recognition (bridge.py) — only attempt it once a track has stuck
    # around a bit (not a one-frame noise blip), both hands have been seen
    # recently (not just once — "all 3 limbs visible"), and its head is
    # actually visible (proxy: pose's nose landmark has valid depth; tracks
    # that never got pose data at all, e.g. hands-only mode, aren't blocked
    # by this — see identity.py's _Track.head_visible).
    recognition_min_track_age_s: float = 1.0

    def save(self, path: str) -> None:
        import json
        from dataclasses import asdict
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GestureConfig":
        import json
        import os
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ── audio ────────────────────────────────────────────────────────────────────

@dataclass
class AecCalibration:
    """Result of VoiceListener/aec.py's one-time-per-setup calibration routine."""
    delay_ms:      float   # measured round-trip delay between reference and mic capture
    gain:          float   # reference signal gain correction
    quality_score: float   # 0.0-1.0, how confidently the delay/gain were estimated
