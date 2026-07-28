"""
nkit — Negative Zero Kinect Interface Toolkit

Only `types` is imported eagerly (dataclasses + numpy, nothing heavy).
Everything else resolves on first attribute access via PEP 562, so the
public API is unchanged:

    import nkit
    tracker = nkit.BodyTracker()      # pulls in mediapipe here, not at import

This isn't micro-optimisation — it's what lets a consumer that doesn't do
detection avoid the detection dependencies entirely. nkit/record.py only
needs kinect + cv2 + websockets, all of which are in nixpkgs, so with lazy
imports `nix run .#record-server` builds from nixpkgs alone instead of
needing a pip venv for a mediapipe it never calls.

Heavier optional subsystems stay explicit imports:

    from nkit.face.recognize import Recognizer
    from nkit.face.enroll import Enroller
    from nkit.audio.voice import VoiceListener
    from nkit.audio import aec
    from nkit.bridge import Bridge
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # never executed — gives editors and type-checkers the real symbols, which
    # they can't infer through the __getattr__ below
    from .kinect import Kinect, ThreadedKinect
    from .body import BodyTracker, associate_faces_to_bodies, associate_hands_to_bodies
    from .hands import HandTracker
    from .gestures import GestureTracker

from .types import (
    Vec3, HandResult, BodyResult, FaceResult, WakeWordResult, RawFrames, nkitFrame,
    CaptureResult, GestureEvent, GestureConfig, AecCalibration,
)

# name -> submodule it lives in. kinect is in here too: importing it needs
# KINECT_SHIM_SO set, which shouldn't be a precondition for `import nkit`.
_LAZY = {
    "Kinect":                     ".kinect",
    "ThreadedKinect":             ".kinect",
    "BodyTracker":                ".body",
    "associate_faces_to_bodies":  ".body",
    "associate_hands_to_bodies":  ".body",
    "HandTracker":                ".hands",
    "GestureTracker":             ".gestures",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value      # subsequent lookups skip __getattr__ entirely
    return value


def __dir__():
    return sorted(__all__)


__all__ = [
    "Vec3", "HandResult", "BodyResult", "FaceResult", "WakeWordResult", "RawFrames", "nkitFrame",
    "CaptureResult", "GestureEvent", "GestureConfig", "AecCalibration",
    "Kinect", "ThreadedKinect", "BodyTracker", "HandTracker", "GestureTracker",
    "associate_faces_to_bodies", "associate_hands_to_bodies",
]
