"""
nkit/bridge.py — local WebSocket bridge between nkit and a web UI

This is the "IPC layer — future work" from the original design doc,
promoted to required once the UI stack became HTML/CSS/JS: a browser page
can't import a Python library, so this runs the full nkit sensor pipeline
(skeleton, hands, face, gestures, voice) on a background thread and
broadcasts newline-delimited JSON events to every connected WebSocket
client. Binds 127.0.0.1 only — this has no auth, never expose it beyond
localhost.

Outbound (bridge -> client), one JSON object per text message:
  {"type": "gesture", "skeleton_id", "side", "kind", "x", "y", "edge", "direction", "timestamp"}
  {"type": "face", "skeleton_id", "name", "known", "confidence", "source"}
  {"type": "voice", "mode", "wake_word", "text", "language", "confidence"}
  {"type": "frame", "skeletons": [...], "hands": [...]}   — only with enable_frame_stream=True;
      immediately followed by a SEPARATE binary WS message containing the JPEG bytes for that frame.

Inbound (client -> bridge), one JSON object per text message:
  {"type": "config", "values": {"push_delta_mm": 60, ...}, "persist": true}
      — partial GestureConfig field updates, applied live; persist=true also
      writes them to disk (see --config) so they survive a restart.

usage:
    python -m nkit.bridge --enroll-dir enroll --port 8765 --voice \\
        --hey-zane-model wakeword_models/hey_zane.onnx \\
        --zane-write-model wakeword_models/zane_write.onnx
"""

from __future__ import annotations
import argparse
import asyncio
import dataclasses
import json
import threading
import time

import cv2
import websockets

from .types import GestureConfig, AecCalibration
from . import _vision
from .kinect import Kinect, ThreadedKinect
from .body import BodyTracker
from .hands import HandTracker
from .identity import IdentityTracker
from .gestures import GestureTracker


class Bridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        enroll_dir: str = "enroll",
        enable_body: bool = True,
        enable_hands: bool = True,
        enable_face: bool = True,
        enable_voice: bool = False,
        enable_frame_stream: bool = False,
        enable_fusion: bool = False,
        hey_zane_model: str | None = None,
        zane_write_model: str | None = None,
        aec_calibration_path: str | None = None,
        screen_w: int = 1920,
        screen_h: int = 1080,
        config_path: str = "nkit_config.json",
        jpeg_quality: int = 70,
    ):
        self.host = host
        self.port = port
        self.enroll_dir = enroll_dir
        # live-toggleable (see _handle_inbound's "body_enabled" message) —
        # BodyTracker is always constructed below so the toggle always has
        # something to flip; this only gates the per-frame inference cost.
        # --no-body just changes the STARTING value now, not whether pose
        # tracking is available at all.
        self.body_enabled = enable_body
        self.enable_hands = enable_hands
        self.enable_face = enable_face
        self.enable_voice = enable_voice
        self.enable_frame_stream = enable_frame_stream
        self.enable_fusion = enable_fusion
        self.hey_zane_model = hey_zane_model
        self.zane_write_model = zane_write_model
        self.aec_calibration_path = aec_calibration_path
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.config_path = config_path
        self.jpeg_quality = jpeg_quality

        self.config = GestureConfig.load(config_path)

        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._sensor_thread: threading.Thread | None = None
        self._voice = None

    # ── outbound plumbing (called from the sensor thread, hops onto the asyncio loop) ──

    def _broadcast(self, message: dict, binary: bytes | None = None) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(message, binary), self._loop)

    async def _broadcast_async(self, message: dict, binary: bytes | None = None) -> None:
        if not self._clients:
            return
        text = json.dumps(message)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(text)
                if binary is not None:
                    await ws.send(binary)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # ── inbound handling ─────────────────────────────────────────────────────────

    async def _handle_client(self, websocket, _path=None):
        self._clients.add(websocket)
        print(f"[bridge] client connected ({len(self._clients)} total)")
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                self._handle_inbound(msg)
        finally:
            self._clients.discard(websocket)
            print(f"[bridge] client disconnected ({len(self._clients)} total)")

    def _handle_inbound(self, msg: dict) -> None:
        if msg.get("type") == "body_enabled":
            # live toggle: skip pose inference entirely and fall back to
            # hands+face-only identity tracking (see nkit/identity.py's
            # 4-element design — it was already built to keep working with
            # pose absent, this just makes that a deliberate runtime choice
            # instead of only a fallback for when pose fails to detect).
            value = msg.get("value")
            if isinstance(value, bool):
                self.body_enabled = value
                print(f"[bridge] body tracking {'enabled' if value else 'disabled'}")
            return

        if msg.get("type") == "screen_size":
            # the renderer reports its OWN real window size — using this
            # instead of the --screen-w/--screen-h CLI flags means gesture
            # coordinates always land in real viewport pixels regardless of
            # actual window/display resolution, no manual flag-matching
            # needed. Plain attribute writes are fine cross-thread here
            # (same pattern as self.config below) — a stale read for one
            # frame is harmless.
            w, h = msg.get("width"), msg.get("height")
            if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
                self.screen_w, self.screen_h = int(w), int(h)
            return

        if msg.get("type") != "config":
            return
        valid_fields = {f.name for f in dataclasses.fields(GestureConfig)}
        values = msg.get("values", {})
        applied = {}
        for k, v in values.items():
            if k in valid_fields:
                setattr(self.config, k, v)
                applied[k] = v
        if applied:
            print(f"[bridge] config updated: {applied}")
        if msg.get("persist") and applied:
            self.config.save(self.config_path)

    # ── sensor loop (background thread — Kinect/mediapipe/insightface are all blocking) ──

    def _sensor_loop(self) -> None:
        recognizer = None
        if self.enable_face:
            from .face.recognize import Recognizer
            recognizer = Recognizer(enroll_dir=self.enroll_dir)

        gesture_tracker = GestureTracker()
        body_tracker = None
        hand_tracker = None

        # face recognition is expensive and identity doesn't change frame to
        # frame — only run it when a currently-tracked identity doesn't have
        # a cached result yet, then keep it for as long as that identity is
        # tracked. see the overview doc's perf pass for why.
        face_cache: dict[int, object] = {}
        roi_tracker = _vision.LandmarkRoiTracker()

        # pose/left-hand/right-hand/face are tracked as 4 independent signals
        # of "this person is still here" — a track only dies once ALL FOUR
        # have gone unmatched past the grace window, not the instant pose
        # alone drops (someone leaning out of frame, an angle mediapipe
        # can't pose-lock, etc. shouldn't kill their hands/face identity).
        identity = IdentityTracker()

        with Kinect() as kinect:
            # always constructed — self.body_enabled (live-toggleable) gates
            # whether it's actually CALLED per frame, see below
            body_tracker = BodyTracker()
            if self.enable_hands:
                hand_tracker = HandTracker()

            print("[bridge] warming up kinect...")
            for _ in range(10):
                kinect.get_frames()

            # grab on its own thread from here on — kinect_grab() blocks for
            # most of a 30fps frame interval and nothing about that wait needs
            # this thread. See ThreadedKinect.
            grabber = ThreadedKinect(kinect).start()
            if self.enable_frame_stream:
                self._start_frame_streamer()
            print("[bridge] sensor loop running")

            # temporary per-stage timing — prints an average every 30 frames
            # so a perf regression can be pinned to a specific stage instead
            # of guessed at. cheap (a few dict adds), safe to leave running.
            stage_totals: dict[str, float] = {}
            frame_count = 0

            while not self._stop.is_set():
                t0 = time.monotonic()
                rgb, depth, ir = grabber.latest()
                if rgb is None:
                    continue
                now = time.monotonic()
                stage_totals["grab"] = stage_totals.get("grab", 0.0) + (now - t0)

                # detect-then-track: crop to where the last frame's skeletons
                # were, computed once and shared across body/hands/face rather
                # than each tracker re-deriving it. None until the first
                # detection lands (and again after the grace period), which
                # means "search the full frame and re-acquire".
                t1 = time.monotonic()
                roi = roi_tracker.roi(now)
                t2 = time.monotonic()
                stage_totals["roi"] = stage_totals.get("roi", 0.0) + (t2 - t1)

                # hands now runs AFTER bodies (was parallel) when pose is on
                # — it needs each body's own wrist estimate to crop around
                # (see hands.py's `bodies` param): a hand can only be found
                # where a skeleton says a hand should be, instead of
                # searching the whole frame/roi and hoping confidence
                # filtering rejects a foot/limb misread after the fact.
                # When pose is off there's nothing to anchor to, so hands
                # falls back to the old whole-roi search — no dependency,
                # no serialization cost in that mode.
                bodies = (body_tracker.process(rgb, depth, ir, self.config, self.enable_fusion, roi, now)
                          if self.body_enabled else [])
                # aim the next frame's crop from what we just found
                roi_tracker.update(bodies, now)

                hands = (hand_tracker.process(rgb, depth, ir, self.config, self.enable_fusion, roi, bodies=bodies)
                         if hand_tracker else [])
                t3 = time.monotonic()
                stage_totals["body_hands"] = stage_totals.get("body_hands", 0.0) + (t3 - t2)

                # assigns skeleton_id (bodies, hands) from persistent 4-element
                # identity — see nkit/identity.py. faces=[] here; recognition
                # only actually runs (and gets matched in) below, on the rare
                # frames a tracked identity doesn't have a cached result yet.
                identity.update(bodies, hands, [], now, self.config)
                # smooths brief mediapipe flicker (a genuinely-present hand
                # momentarily missing a frame or two) by reusing each
                # track's last real hand for up to hand_hold_ms — applied
                # after identity matching so held hands already carry valid
                # skeleton_id/side, and everything downstream (gestures,
                # the frame overlay) just sees a hand that's still there
                hands = identity.fill_held_hands(hands, now, self.config)
                active_ids = identity.active_ids

                for sid in list(face_cache):
                    if sid not in active_ids:
                        del face_cache[sid]

                t4 = time.monotonic()
                stage_totals["identity"] = stage_totals.get("identity", 0.0) + (t4 - t3)

                if recognizer:
                    # not just "missing a cached result" — also gated on
                    # track age (not a one-frame noise blip) and head
                    # visibility (see identity.recognition_eligible /
                    # GestureConfig.recognition_min_track_age_s)
                    ids_needing_face = {
                        sid for sid in (active_ids - set(face_cache.keys()))
                        if identity.recognition_eligible(sid, now, self.config)
                    }
                    if ids_needing_face:
                        # bodies passed when available — crops per-body
                        # around each nose estimate instead of one big
                        # region (same idea as hands' wrist cropping).
                        # identity.update() below re-derives skeleton_id
                        # against the SAME persistent tracks pose/hands
                        # just updated this frame either way (redundant but
                        # harmless when Recognizer's own association above
                        # already got it right from the crop).
                        faces = recognizer.process(rgb, depth, ir, self.config, bodies=bodies, roi=roi)
                        identity.update([], [], faces, now, self.config)
                        for f in faces:
                            if f.skeleton_id is not None and f.skeleton_id in ids_needing_face and f.skeleton_id not in face_cache:
                                face_cache[f.skeleton_id] = f
                                self._broadcast({
                                    "type": "face", "skeleton_id": f.skeleton_id, "name": f.name,
                                    "known": f.known, "confidence": f.confidence, "source": f.source,
                                })
                t5 = time.monotonic()
                stage_totals["face"] = stage_totals.get("face", 0.0) + (t5 - t4)

                events = gesture_tracker.process(hands, now, self.config, self.screen_w, self.screen_h,
                                                 bodies=bodies)
                for ev in events:
                    self._broadcast({
                        "type": "gesture", "skeleton_id": ev.skeleton_id, "side": ev.side,
                        "kind": ev.kind, "x": ev.x, "y": ev.y, "edge": ev.edge,
                        "direction": ev.direction, "timestamp": ev.timestamp,
                        "progress": ev.progress,
                    })
                t6 = time.monotonic()
                stage_totals["gesture"] = stage_totals.get("gesture", 0.0) + (t6 - t5)

                if self.enable_frame_stream:
                    self._submit_frame(rgb, bodies, hands)
                t7 = time.monotonic()
                stage_totals["frame_stream"] = stage_totals.get("frame_stream", 0.0) + (t7 - t6)
                stage_totals["total"] = stage_totals.get("total", 0.0) + (t7 - t0)

                frame_count += 1
                if frame_count % 30 == 0:
                    avg_ms = {k: round(v / frame_count * 1000, 1) for k, v in stage_totals.items()}
                    print(f"[bridge] perf (avg ms/frame over {frame_count} frames): {avg_ms}")

        if self.enable_frame_stream:
            self._stop_frame_streamer()
        if body_tracker:
            body_tracker.close()
        if hand_tracker:
            hand_tracker.close()

    # ── preview frame stream (own thread) ────────────────────────────────────
    # JPEG encoding measured ~13ms/frame — 27% of the sensor loop — and
    # nothing in tracking depends on its result, so it has no business
    # blocking the pipeline. Only the newest un-encoded frame is kept: a
    # preview that can't keep up should skip frames, not accumulate latency.
    # Frames are per-frame copies out of the shim (see Kinect._read_frame), so
    # handing one to this thread while the sensor loop moves on is safe.

    def _start_frame_streamer(self) -> None:
        self._fs_cv = threading.Condition()
        self._fs_pending: tuple | None = None
        self._fs_stop = threading.Event()
        self._fs_thread = threading.Thread(target=self._frame_stream_loop, name="frame-stream", daemon=True)
        self._fs_thread.start()

    def _submit_frame(self, rgb, bodies, hands) -> None:
        with self._fs_cv:
            self._fs_pending = (rgb, bodies, hands)   # newest wins, older dropped
            self._fs_cv.notify()

    def _stop_frame_streamer(self) -> None:
        self._fs_stop.set()
        with self._fs_cv:
            self._fs_cv.notify_all()
        self._fs_thread.join(timeout=2.0)

    def _frame_stream_loop(self) -> None:
        while not self._fs_stop.is_set():
            with self._fs_cv:
                if self._fs_pending is None:
                    self._fs_cv.wait(0.5)
                    continue
                rgb, bodies, hands = self._fs_pending
                self._fs_pending = None
            self._broadcast_frame(rgb, bodies, hands)

    def _broadcast_frame(self, rgb, bodies, hands) -> None:
        bgr = rgb[:, :, :3]
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        skeletons = [
            {"skeleton_id": b.skeleton_id,
             "landmarks": {k: [v.x, v.y, v.z] for k, v in b.landmarks.items()}}
            for b in bodies
        ]
        hand_summary = [
            {"skeleton_id": h.skeleton_id, "side": h.side, "is_fist": h.is_fist,
             "fist_score": h.fist_score, "wrist": [h.wrist.x, h.wrist.y, h.wrist.z]}
            for h in hands
        ]
        self._broadcast({"type": "frame", "skeletons": skeletons, "hands": hand_summary}, binary=jpeg.tobytes())

    # ── voice ────────────────────────────────────────────────────────────────────

    def _on_voice(self, result) -> None:
        self._broadcast({
            "type": "voice", "mode": result.mode, "wake_word": result.wake_word,
            "text": result.text, "language": result.language, "confidence": result.confidence,
        })

    # ── lifecycle ────────────────────────────────────────────────────────────────

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()

        self._sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self._sensor_thread.start()

        if self.enable_voice and (self.hey_zane_model or self.zane_write_model):
            from .audio.voice import VoiceListener
            calibration = None
            if self.aec_calibration_path:
                calibration = _load_aec_calibration(self.aec_calibration_path)
            self._voice = VoiceListener(
                hey_zane_model=self.hey_zane_model,
                zane_write_model=self.zane_write_model,
                callback=self._on_voice,
                aec_calibration=calibration,
            )
            self._voice.start()

        async with websockets.serve(self._handle_client, self.host, self.port):
            print(f"[bridge] listening on ws://{self.host}:{self.port}")
            await asyncio.Future()   # run forever

    def run(self) -> None:
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            if self._voice:
                self._voice.stop()


def _load_aec_calibration(path: str) -> AecCalibration:
    with open(path) as f:
        data = json.load(f)
    return AecCalibration(**data)


def main():
    parser = argparse.ArgumentParser(description="nkit WebSocket bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--enroll-dir", default="enroll")
    parser.add_argument("--no-body", action="store_true",
                         help="start with skeleton/pose tracking off (hands+face only) — live-toggleable from the UI regardless")
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--frame-stream", action="store_true", help="broadcast JPEG frames + skeleton/hand overlays")
    parser.add_argument("--fusion", action="store_true", help="fuse RGB+IR every frame instead of adaptive switching")
    parser.add_argument("--hey-zane-model", default=None)
    parser.add_argument("--zane-write-model", default=None)
    parser.add_argument("--aec-calibration", default=None, help="path to a saved AecCalibration JSON")
    parser.add_argument("--screen-w", type=int, default=1920)
    parser.add_argument("--screen-h", type=int, default=1080)
    parser.add_argument("--config", default="nkit_config.json")
    args = parser.parse_args()

    bridge = Bridge(
        host=args.host, port=args.port, enroll_dir=args.enroll_dir,
        enable_body=not args.no_body, enable_hands=not args.no_hands, enable_face=not args.no_face,
        enable_voice=args.voice, enable_frame_stream=args.frame_stream, enable_fusion=args.fusion,
        hey_zane_model=args.hey_zane_model, zane_write_model=args.zane_write_model,
        aec_calibration_path=args.aec_calibration, screen_w=args.screen_w, screen_h=args.screen_h,
        config_path=args.config,
    )
    bridge.run()


if __name__ == "__main__":
    main()
