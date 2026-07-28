"""
nkit — Negative Zero Kinect Interface Toolkit

Only the always-cheap core is imported eagerly here (types, kinect, body,
hands, gestures). Face recognition (insightface) and voice (openwakeword/
faster-whisper/pyaudio) are heavier optional subsystems — import them
explicitly when you need them:

    from nkit.face.recognize import Recognizer
    from nkit.face.enroll import Enroller
    from nkit.audio.voice import VoiceListener
    from nkit.audio import aec
    from nkit.bridge import Bridge
"""

from .types import (
    Vec3, HandResult, BodyResult, FaceResult, WakeWordResult, RawFrames, nkitFrame,
    CaptureResult, GestureEvent, GestureConfig, AecCalibration,
)
from .kinect import Kinect
from .body import BodyTracker, associate_faces_to_bodies, associate_hands_to_bodies
from .hands import HandTracker
from .gestures import GestureTracker

__all__ = [
    "Vec3", "HandResult", "BodyResult", "FaceResult", "WakeWordResult", "RawFrames", "nkitFrame",
    "CaptureResult", "GestureEvent", "GestureConfig", "AecCalibration",
    "Kinect", "BodyTracker", "HandTracker", "GestureTracker",
    "associate_faces_to_bodies", "associate_hands_to_bodies",
]
