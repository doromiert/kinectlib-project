"""
nkit/_mp.py — mediapipe detector construction shared by body.py / hands.py

Exists so the GPU-delegate fallback lives in one place instead of being
duplicated in both trackers. Deliberately separate from _vision.py, which is
pure cv2/numpy preprocessing and has no mediapipe dependency.
"""

from __future__ import annotations
from typing import Callable, TypeVar

from mediapipe.tasks import python as mp_tasks

Delegate = mp_tasks.BaseOptions.Delegate
T = TypeVar("T")


def create_detector(
    detector_cls: type[T],
    build_options: Callable,
    use_gpu: bool,
    label: str,
) -> T:
    """
    Build a mediapipe Tasks detector, preferring the GPU delegate.

    build_options takes a Delegate and returns that detector's *Options.
    Tries GPU then CPU; a machine with no usable GL context degrades to CPU
    with a warning instead of failing to start.
    """
    delegates = (Delegate.GPU, Delegate.CPU) if use_gpu else (Delegate.CPU,)
    errors = []
    for delegate in delegates:
        try:
            detector = detector_cls.create_from_options(build_options(delegate))  # type: ignore[attr-defined]
            if delegate is Delegate.CPU and use_gpu:
                print(f"{label}: GPU delegate unavailable, running on CPU")
            return detector
        except Exception as e:  # noqa: BLE001 — any init failure is worth falling back on
            errors.append(f"{delegate.name}: {e}")
    raise RuntimeError(f"could not create {label} on any delegate — {'; '.join(errors)}")
