"""
nzk/kinect.py — ctypes wrapper around kinect_shim.so

usage:
    from nzk.kinect import Kinect
    with Kinect() as k:
        rgb, depth, ir = k.get_frames()
        # rgb:   (1080, 1920, 4) uint8   BGRX
        # depth: (424, 512)      float32 mm
        # ir:    (424, 512)      float32 raw intensity
"""

import ctypes
import os
import numpy as np


def _load_shim():
    so = os.environ.get("KINECT_SHIM_SO")
    if so and os.path.exists(so):
        return ctypes.CDLL(so)
    raise RuntimeError("KINECT_SHIM_SO not set — are you inside nix develop?")


lib = _load_shim()

lib.kinect_open.restype  = ctypes.c_void_p
lib.kinect_open.argtypes = []

lib.kinect_close.restype  = None
lib.kinect_close.argtypes = [ctypes.c_void_p]

lib.kinect_grab.restype  = ctypes.c_int
lib.kinect_grab.argtypes = [ctypes.c_void_p, ctypes.c_int]

lib.kinect_frame_data.restype  = ctypes.POINTER(ctypes.c_uint8)
lib.kinect_frame_data.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]

FRAME_COLOR = 0
FRAME_DEPTH = 1
FRAME_IR    = 2


class Kinect:
    def __init__(self):
        self._ctx = None

    def open(self):
        self._ctx = lib.kinect_open()
        if not self._ctx:
            raise RuntimeError(
                "kinect_open failed — is the kinect plugged in and udev rules applied?"
            )
        print("kinect opened")

    def close(self):
        if self._ctx:
            lib.kinect_close(self._ctx)
            self._ctx = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def get_frames(self, timeout_ms: int = 5000):
        """
        returns (rgb, depth, ir) or (None, None, None) on timeout
          rgb:   (1080, 1920, 4) uint8   BGRX
          depth: (424, 512)      float32 mm
          ir:    (424, 512)      float32 raw intensity
        """
        if not lib.kinect_grab(self._ctx, timeout_ms):
            return None, None, None

        rgb   = self._read_frame(FRAME_COLOR, np.uint8)
        depth = self._read_frame(FRAME_DEPTH, np.float32)
        ir    = self._read_frame(FRAME_IR,    np.float32)

        return rgb, depth, ir

    def _read_frame(self, frame_type, dtype):
        w = ctypes.c_int(0)
        h = ctypes.c_int(0)
        bpp = ctypes.c_int(0)

        ptr = lib.kinect_frame_data(
            self._ctx, frame_type,
            ctypes.byref(w), ctypes.byref(h), ctypes.byref(bpp)
        )
        if not ptr:
            return None

        w, h, bpp = w.value, h.value, bpp.value
        n_bytes = w * h * bpp
        buf = (ctypes.c_uint8 * n_bytes).from_address(
            ctypes.cast(ptr, ctypes.c_void_p).value
        )
        arr = np.frombuffer(buf, dtype=dtype).copy()

        channels = bpp // np.dtype(dtype).itemsize
        if channels > 1:
            return arr.reshape(h, w, channels)
        return arr.reshape(h, w)
