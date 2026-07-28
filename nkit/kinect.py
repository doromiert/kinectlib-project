"""
nkit/kinect.py — ctypes wrapper around kinect_shim.so

usage:
    from nkit.kinect import Kinect
    with Kinect() as k:
        rgb, depth, ir = k.get_frames()
        # rgb:   (1080, 1920, 4) uint8   BGRX
        # depth: (424, 512)      float32 mm
        # ir:    (424, 512)      float32 raw intensity
"""

import ctypes
import os
import threading
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


class ThreadedKinect:
    """
    Runs get_frames() on a background thread and hands the consumer whichever
    frame set is newest.

    kinect_grab() blocks until the sensor delivers. At 30fps that means the
    caller spends most of a frame interval doing nothing — measured ~22ms of
    an ~81ms pipeline, pure idle. None of that wait needs the main thread, so
    running it here overlaps it with the processing of the previous frame.

    Only the newest frame is kept; anything the consumer was too slow to
    collect is dropped rather than queued. When processing is slower than the
    sensor (which it is), a backlog would only feed the pipeline progressively
    staler frames — for interactive tracking the current frame is the only one
    worth having.

    Frames are safe to hand across the thread boundary because _read_frame()
    copies out of the shim's buffer rather than viewing it.

    usage:
        with Kinect() as k:
            grabber = ThreadedKinect(k).start()
            rgb, depth, ir = grabber.latest()
            ...
            grabber.stop()
    """

    def __init__(self, kinect: "Kinect", timeout_ms: int = 5000):
        self._kinect = kinect
        self._timeout_ms = timeout_ms
        self._cv = threading.Condition()
        self._frames: tuple = (None, None, None)
        self._seq = 0            # bumped per delivered frame
        self._taken = 0          # last seq the consumer received
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "ThreadedKinect":
        self._thread = threading.Thread(target=self._run, name="kinect-grab", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._kinect.get_frames(self._timeout_ms)
            except BaseException as e:   # surface it on the consumer's thread
                with self._cv:
                    self._error = e
                    self._cv.notify_all()
                return
            if frames[0] is None:
                continue            # timeout — just try again
            with self._cv:
                self._frames = frames
                self._seq += 1
                self._cv.notify_all()

    def latest(self, timeout: float = 2.0):
        """
        Block until a frame newer than the last one returned, then return it.
        Returns (None, None, None) if nothing arrives within timeout.
        Re-raises on the consumer's thread anything the grab thread hit.
        """
        with self._cv:
            got = self._cv.wait_for(lambda: self._seq != self._taken or self._error is not None, timeout)
            if self._error is not None:
                raise self._error
            if not got:
                return (None, None, None)
            self._taken = self._seq
            return self._frames

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
