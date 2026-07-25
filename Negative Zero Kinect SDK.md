# nzk SDK — API Design

## Overview

A Python library (`nzk/`) wrapping all sensor modalities into a
clean, composable API. No Flask, no UI, no side effects — pure data.
Tools like `enroll.py`, `trainer.py`, and `visualizer.py` are
**consumers** of this library, not part of it.

---

## Language recommendation

**Stay Python.** The entire stack — libfreenect2 ctypes shim,
mediapipe, insightface, openwakeword, faster-whisper, pyaudio — is
Python-native or has Python bindings. Wrapping this in Rust or C
would mean either reimplementing all of that or bridging it via
subprocesses/sockets, which adds latency and complexity for no real
gain at this stage.

If you later need a native consumer (a game engine, a Rust app,
a browser via WASM), the right move is to expose the library over a
**local IPC channel** (Unix socket or named pipe, msgpack frames)
from a Python daemon — not to rewrite the library itself. That keeps
the sensor/ML code in Python where it works and lets any language
consume it.

---

## Module layout

```
nzk/
  __init__.py          re-exports the main public API
  kinect.py            raw frame source (unchanged from current)
  streams.py           nzkStreams — the main entry point
  body.py              PoseTracker wrapper (skeleton)
  hands.py             HandTracker wrapper + fist detection
  face/
    __init__.py
    recognize.py       Recognizer (live detection)
    enroll.py          Enroller (programmatic enrollment, no Flask)
  voice.py             VoiceListener (wake word + STT + raw audio)
  types.py             all shared dataclasses / TypedDicts
```

---

## Core types  (`nzk/types.py`)

```python
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

Source = Literal["rgb", "ir"]

@dataclass
class Vec3:
    x: int        # px in RGB space
    y: int        # px in RGB space
    z: float      # mm from depth sensor (0.0 = no data)

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
    landmarks:   dict[str, Vec3]   # all 21 by name
    is_fist:     bool              # see fist detection below
    fist_score:  float             # 0.0–1.0 continuous confidence

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
    landmarks:       dict[str, Vec3]   # all 33

@dataclass
class FaceResult:
    name:        str           # "unknown" if not recognized
    confidence:  float         # cosine similarity 0.0–1.0
    known:       bool
    bbox:        tuple[int,int,int,int]   # x1,y1,x2,y2 in RGB px
    embedding:   np.ndarray    # (512,) raw insightface embedding
    source:      Source
    body:        BodyResult | None   # skeleton this face was matched to

@dataclass
class WakeWordResult:
    mode:        Literal["assistant", "stt"]
    wake_word:   str       # "hey_zane" | "zane_write"
    text:        str       # whisper transcript
    language:    str       # "en" | "pl" | ...
    words:       list[dict]  # [{word, start, end, probability}]
    confidence:  float     # openwakeword score

@dataclass
class RawFrames:
    rgb:   np.ndarray   # (1080, 1920, 4) uint8 BGRX
    depth: np.ndarray   # (424, 512)      float32 mm
    ir:    np.ndarray   # (424, 512)      float32

@dataclass
class nzkFrame:
    """One processed frame — all modalities, all results."""
    frames:  RawFrames
    bodies:  list[BodyResult]
    hands:   list[HandResult]
    faces:   list[FaceResult]
    # voice results arrive async via callback, not here
```

---

## Fist detection  (`nzk/hands.py`)

Mediapipe gives normalized 3D landmark positions. Fist detection
uses **finger curl**: for each finger, compare the angle at the MCP
and PIP joints. A curled finger has its tip closer to the palm than
its MCP knuckle.

```python
def _finger_curl(landmarks: dict[str, Vec3], finger: str) -> float:
    """
    0.0 = fully extended, 1.0 = fully curled.
    computed as: 1 - (tip_to_wrist / mcp_to_wrist)
    clamped to [0, 1].
    """
    wrist = landmarks["wrist"]
    mcp   = landmarks[f"{finger}_mcp"]
    tip   = landmarks[f"{finger}_tip"]

    def dist(a: Vec3, b: Vec3) -> float:
        return ((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2) ** 0.5

    mcp_dist = dist(wrist, mcp)
    tip_dist = dist(wrist, tip)
    if mcp_dist < 1e-6:
        return 0.0
    return float(np.clip(1.0 - tip_dist / mcp_dist, 0.0, 1.0))

FINGERS = ["index", "middle", "ring", "pinky"]
FIST_CURL_THRESHOLD  = 0.6   # per-finger
FIST_FINGER_COUNT    = 3     # how many fingers must be curled

def _detect_fist(landmarks: dict[str, Vec3]) -> tuple[bool, float]:
    curls  = [_finger_curl(landmarks, f) for f in FINGERS]
    score  = float(np.mean(curls))
    n_curled = sum(1 for c in curls if c >= FIST_CURL_THRESHOLD)
    return n_curled >= FIST_FINGER_COUNT, score
```

thumb is excluded — a thumbs-up would otherwise read as a fist.
tune `FIST_CURL_THRESHOLD` and `FIST_FINGER_COUNT` to taste.

---

## Main entry point  (`nzk/streams.py`)

```python
class nzkStreams:
    """
    The main library entry point. Owns all hardware and ML model
    lifecycles. Use as a context manager.

    example:

        def on_voice(result: WakeWordResult):
            print(result.text)

        with nzkStreams(voice_callback=on_voice) as z:
            for frame in z.frames():
                # frame.bodies, frame.hands, frame.faces
                do_something(frame)

            # or just raw frames, skip ML entirely:
            for raw in z.raw_frames():
                show(raw.rgb)
    """

    def __init__(
        self,
        *,
        # feature toggles — disable what you don't need for perf
        enable_body:    bool = True,
        enable_hands:   bool = True,
        enable_face:    bool = True,
        enable_voice:   bool = True,

        # face recognition
        enroll_dir:     str   = "enroll",
        face_providers: list  = None,  # onnxruntime providers

        # voice
        hey_zane_model:   str   = "wakeword_models/hey_zane.onnx",
        zane_write_model: str   = "wakeword_models/zane_write.onnx",
        voice_callback          = None,   # fn(WakeWordResult)
        raw_audio_callback      = None,   # fn(bytes) — 16kHz int16 chunks

        # tuning
        max_hands:      int   = 2,
        max_bodies:     int   = 4,
        confidence:     float = 0.5,
    ): ...

    # ── iteration API ─────────────────────────────────────────────

    def frames(self) -> Iterator[nzkFrame]:
        """
        Yields one nzkFrame per Kinect frame. Blocks until the next
        frame is available. All enabled ML pipelines run before yield.
        Voice results come via callback, not here (they're async).
        """

    def raw_frames(self) -> Iterator[RawFrames]:
        """
        Yields raw (rgb, depth, ir) with no ML processing.
        Use when you want to drive your own pipeline.
        """

    # ── one-shot API (for tools that don't loop) ──────────────────

    def get_frame(self) -> nzkFrame:
        """grab and process one frame, return it"""

    def get_raw(self) -> RawFrames:
        """grab one raw frame"""

    # ── raw audio ─────────────────────────────────────────────────
    # raw_audio_callback receives 16kHz int16 bytes chunks (512 samples)
    # fired from the same background thread as wake word detection.
    # if you want the stream without wake word detection, set
    # enable_voice=True but don't pass hey_zane_model/zane_write_model.

    # ── context manager ───────────────────────────────────────────

    def __enter__(self) -> "nzkStreams": ...
    def __exit__(self, *_): ...
```

---

## Programmatic enrollment  (`nzk/face/enroll.py`)

The current `enroll.py` is a Flask app — fine as a tool, but the
library should expose enrollment as a callable API so you can drive
it from anything (your own UI, a script, a test).

```python
class Enroller:
    """
    Programmatic face enrollment. No Flask, no UI.
    The caller is responsible for the capture loop and trigger logic.

    example — enroll one frame worth of captures:

        with nzkStreams(enable_voice=False) as z, Enroller("Alice") as e:
            for frame in z.frames():
                if frame.faces:
                    result = e.try_capture(frame)
                    if result.captured:
                        print(f"captured cell {result.cell}, "
                              f"{result.n_done}/{result.n_needed}")
                    if result.complete:
                        break
    """

    def __init__(
        self,
        name:       str,
        enroll_dir: str   = "enroll",
        n_per_cell: int   = 3,
        grid_cols:  int   = 5,
        grid_rows:  int   = 3,
    ): ...

    def try_capture(self, frame: nzkFrame) -> "CaptureResult":
        """
        Attempt to capture from the current frame.
        Returns a CaptureResult regardless — inspect .captured to know
        if a capture actually happened (face must be detected + in-cell).
        Call this every frame; it handles cell assignment internally.
        """

    @property
    def current_cell(self) -> tuple[int,int] | None: ...

    @property
    def done_cells(self) -> dict[tuple[int,int], int]: ...

    @property
    def is_complete(self) -> bool: ...

    def reload(self) -> None:
        """force re-scan of enroll_dir (after external enrollment)"""

    def __enter__(self) -> "Enroller": ...
    def __exit__(self, *_): ...

@dataclass
class CaptureResult:
    captured:  bool
    complete:  bool
    cell:      tuple[int,int] | None
    n_done:    int
    n_needed:  int
    face:      FaceResult | None
```

---

## Voice + raw audio  (`nzk/voice.py`)

The existing `VoiceListener` is already pretty clean. Two additions:

1. **`raw_audio_callback`** — fires on every mic chunk (512 samples,
   16kHz int16 bytes) before wake word detection. Lets the caller
   do their own VAD, visualize levels, pipe to another system, etc.

2. **No-model mode** — if wake word model paths are `None`, the
   listener still opens the mic and fires `raw_audio_callback` only.
   Useful if you just want the audio stream.

```python
class VoiceListener:
    def __init__(
        self,
        hey_zane_model:    str | None,
        zane_write_model:  str | None,
        callback:          Callable[[WakeWordResult], None] | None = None,
        raw_audio_callback: Callable[[bytes], None] | None = None,
        ...
    ): ...
```

---

## What enroll.py / visualizer.py / trainer.py become

They stay as standalone tools, but now import from `nzk`:

```python
# enroll.py (simplified)
from nzk import nzkStreams, Enroller

with nzkStreams(enable_voice=False) as z, Enroller(args.name) as e:
    for frame in z.frames():
        update_ui(e.current_cell, e.done_cells, frame)
        if trigger.should_capture():
            e.try_capture(frame)
        if e.is_complete:
            break
```

---

## IPC layer (future / optional)

If you decide to write the final app in Rust, C, or JS:

```
[Python nzk daemon] ──msgpack over Unix socket──> [your app]
```

Each message is a serialized `nzkFrame` (raw arrays sent as
flatbuffers or raw bytes, structured data as msgpack). The daemon
runs `nzkStreams` and blasts frames down the socket. Your app
connects and reads.

This is strictly future work — don't design for it now. Start in
Python; if you hit a hard performance wall or need native rendering,
add the IPC layer then. The library API above is designed so the
daemon would just be a thin loop around `z.frames()`.

---

## Performance notes

- **Disable what you don't need.** `enable_face=False` skips
  insightface entirely (~30ms/frame saved). Voice is always async.
- **IR fallback is automatic** — both hands and face detection check
  `_is_dark(rgb)` and switch to IR. No caller involvement needed.
- **Fist detection is free** — it's a few distance calculations on
  already-computed landmarks, negligible cost.
- **Threading model:** Kinect frame loop runs on the caller's thread
  (via `frames()` iterator). Voice runs in two background threads
  (listen + transcribe). No locks needed between them since they
  don't share state.
