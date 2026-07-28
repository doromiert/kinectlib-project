"""
nkit/record.py — phone-controlled capture rig for labelled gesture data

Records raw RGB + IR + depth to disk while you label what you're doing from a
phone on the same wifi. Exists because every gesture fix so far has been
validated against synthetic landmarks and none of them survived contact with
a real hand — this produces the ground truth to test against instead.

Deliberately does NO detection while capturing. Running mediapipe costs ~40ms
a frame and would throttle the recording, and landmarks derived now would
freeze in today's model/ROI/threshold settings. Raw feeds replay through
whatever pipeline we want later, as many times as we want, with no Kinect and
nobody standing in front of it.

Labels come from the phone, so recording in the living room doesn't mean
running back to the keyboard between takes — which is the whole reason
distance coverage kept not happening.

    python -m nkit.record --out recordings/

Then open the printed http://<laptop-ip>:8080 on your phone.

NOTE: binds 0.0.0.0, not localhost — that's the point (the phone is a
different machine), but it does mean the preview is reachable by anything on
your network while it runs. It's a camera pointed at your living room; don't
leave it running on a network you don't trust.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import socket
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import websockets

from .kinect import Kinect, ThreadedKinect

# The first three are NEGATIVES, and they matter as much as the gestures.
# The loudest complaint is push firing when nobody pushed, and a threshold (or
# a classifier) can only be checked against that if the data contains the
# motions that falsely trigger it. Each negative is aimed at a specific
# false positive:
#
#   rest          arms down, still — the trivial case
#   moving        natural non-gesture motion: shifting weight, walking, talking
#                 with your hands, scratching your face. Catch-all negative.
#   reach_forward reaching for a drink/keyboard. Hand travels toward the camera
#                 and the arm extends, which is exactly what push measures —
#                 the hardest negative for push, and probably the actual cause.
#   wave          lateral hand motion that must NOT read as a page swipe; the
#                 same shape swipe_left/right look for, minus the intent.
#
# Recording only gestures would tell us nothing about any of these.
ACTIONS = [
    "rest",
    "moving",
    "reach_forward",
    "wave",
    "push",
    "swipe_left",
    "swipe_right",
    "fist",
    "grab_swipe_up",
    "grab_swipe_down",
]

# recorded alongside every frame so one session can cover several conditions
# without re-launching, and so "works at the desk, not in the living room" is
# answerable from the data rather than from memory
POSITIONS = ["desk", "living_room"]
LIGHTING = ["light", "dark"]

PREVIEW_EVERY_N = 3       # phone doesn't need every frame
PREVIEW_WIDTH   = 480
PREVIEW_QUALITY = 55
RGB_QUALITY     = 85      # archived rgb; 1080p jpeg at this is ~250KB


class Recorder:
    """
    Owns the disk writes, on its own thread.

    Encoding three feeds costs more than a frame interval, so doing it inline
    would drop the capture rate to whatever the disk felt like. The queue is
    bounded and blocking: if the disk genuinely can't keep up we'd rather slow
    capture than silently lose labelled frames, since a gap in a recording is
    invisible later and a slow recording isn't.
    """

    def __init__(self, out_root: str):
        self._out_root = out_root
        self._q: queue.Queue = queue.Queue(maxsize=60)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.session_dir: str | None = None
        self.frame_count = 0
        self._manifest = None

    def start_session(self, meta: dict) -> str:
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = os.path.join(self._out_root, name)
        os.makedirs(os.path.join(self.session_dir, "frames"), exist_ok=True)
        with open(os.path.join(self.session_dir, "session.json"), "w") as f:
            json.dump({"started": name, **meta}, f, indent=2)
        self._manifest = open(os.path.join(self.session_dir, "manifest.jsonl"), "a")
        self.frame_count = 0

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        return self.session_dir

    def submit(self, rgb, depth, ir, meta: dict) -> None:
        self._q.put((rgb, depth, ir, meta))
        self.frame_count += 1

    def stop_session(self) -> None:
        self._stop.set()
        self._q.put(None)                     # wake the writer out of its get()
        if self._thread:
            self._thread.join(timeout=15.0)   # let the backlog flush
            self._thread = None
        if self._manifest:
            self._manifest.close()
            self._manifest = None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            rgb, depth, ir, meta = item
            try:
                self._write(rgb, depth, ir, meta)
            except Exception as e:      # a bad frame shouldn't end the take
                print(f"[record] write failed on frame {meta.get('i')}: {e}")

    def _write(self, rgb, depth, ir, meta: dict) -> None:
        i = meta["i"]
        base = os.path.join(self.session_dir, "frames", f"{i:06d}")

        cv2.imwrite(f"{base}.jpg", rgb[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, RGB_QUALITY])
        # depth is float32 mm and ir is float32 intensity; uint16 png holds
        # both losslessly at the precision the sensor actually delivers, and
        # costs a fraction of what float32 npy would
        cv2.imwrite(f"{base}_depth.png", np.clip(depth, 0, 65535).astype(np.uint16))
        cv2.imwrite(f"{base}_ir.png", np.clip(ir, 0, 65535).astype(np.uint16))

        self._manifest.write(json.dumps(meta) + "\n")
        self._manifest.flush()


class RecordServer:
    def __init__(self, out_root: str, http_port: int = 8080, ws_port: int = 8766):
        self.out_root = out_root
        self.http_port = http_port
        self.ws_port = ws_port

        self.recorder = Recorder(out_root)
        self.recording = False
        self.action = "rest"
        self.position = "desk"
        self.lighting = "light"

        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._status_lock = threading.Lock()

    # ── capture ──────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        with Kinect() as kinect:
            print("[record] warming up kinect...")
            for _ in range(10):
                kinect.get_frames()
            grabber = ThreadedKinect(kinect).start()
            print("[record] capture running")

            n = 0
            while not self._stop.is_set():
                rgb, depth, ir = grabber.latest()
                if rgb is None:
                    continue
                n += 1

                with self._status_lock:
                    recording = self.recording
                    meta = {
                        "i": self.recorder.frame_count,
                        "t": time.time(),
                        "action": self.action,
                        "position": self.position,
                        "lighting": self.lighting,
                    }
                if recording:
                    self.recorder.submit(rgb, depth, ir, meta)

                if n % PREVIEW_EVERY_N == 0:
                    self._send_preview(rgb)

            grabber.stop()

    def _send_preview(self, rgb) -> None:
        if not self._clients or self._loop is None:
            return
        # same stride on both axes so the aspect ratio survives
        step = max(1, rgb.shape[1] // PREVIEW_WIDTH)
        small = rgb[::step, ::step, :3]
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
        if not ok:
            return
        self._broadcast(jpeg.tobytes())
        self._broadcast(json.dumps(self._status()))

    def _status(self) -> dict:
        with self._status_lock:
            return {
                "type": "status",
                "recording": self.recording,
                "action": self.action,
                "position": self.position,
                "lighting": self.lighting,
                "frames": self.recorder.frame_count,
                "session": os.path.basename(self.recorder.session_dir or ""),
                "actions": ACTIONS, "positions": POSITIONS, "lightings": LIGHTING,
            }

    def _broadcast(self, payload) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    async def _broadcast_async(self, payload) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # ── control ──────────────────────────────────────────────────────────────

    async def _handle_client(self, ws) -> None:
        self._clients.add(ws)
        print(f"[record] phone connected ({len(self._clients)} total)")
        try:
            await ws.send(json.dumps(self._status()))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                self._on_message(msg)
                await self._broadcast_async(json.dumps(self._status()))
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            print(f"[record] phone disconnected ({len(self._clients)} left)")

    def _on_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "set":
            with self._status_lock:
                if msg.get("action") in ACTIONS:
                    self.action = msg["action"]
                if msg.get("position") in POSITIONS:
                    self.position = msg["position"]
                if msg.get("lighting") in LIGHTING:
                    self.lighting = msg["lighting"]
        elif kind == "start" and not self.recording:
            meta = {"position": self.position, "lighting": self.lighting}
            path = self.recorder.start_session(meta)
            with self._status_lock:
                self.recording = True
            print(f"[record] recording -> {path}")
        elif kind == "stop" and self.recording:
            with self._status_lock:
                self.recording = False
            self.recorder.stop_session()
            print(f"[record] stopped ({self.recorder.frame_count} frames)")

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def _serve_ws(self) -> None:
        self._loop = asyncio.get_running_loop()
        async with websockets.serve(self._handle_client, "0.0.0.0", self.ws_port,
                                    max_size=None):
            print(f"[record] control ws on :{self.ws_port}")
            await asyncio.Future()

    def run(self) -> None:
        page = _PAGE.replace("__WS_PORT__", str(self.ws_port))
        httpd = ThreadingHTTPServer(("0.0.0.0", self.http_port), _make_handler(page))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        cap = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        cap.start()

        ip = _lan_ip()
        print(f"\n[record] open on your phone:  http://{ip}:{self.http_port}\n")
        try:
            asyncio.run(self._serve_ws())
        except KeyboardInterrupt:
            print("\n[record] shutting down")
        finally:
            self._stop.set()
            if self.recording:
                self.recorder.stop_session()
            httpd.shutdown()


def _lan_ip() -> str:
    """Address the phone should dial. No packets are actually sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _make_handler(page: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):   # noqa: A002 — matches base signature
            pass   # don't spam the capture log with one line per request

    return Handler


_PAGE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>nkit recorder</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin:0; background:#111; color:#eee; font:15px/1.4 system-ui,sans-serif; padding:12px; }
  img { width:100%; border-radius:10px; background:#000; display:block; aspect-ratio:16/9; object-fit:contain; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#888; margin:16px 0 6px; }
  .row { display:flex; flex-wrap:wrap; gap:8px; }
  button { flex:1 1 auto; min-width:30%; padding:16px 10px; font-size:15px; border-radius:10px;
           border:1px solid #333; background:#1c1c1c; color:#ddd; }
  button.on { background:#2d6cdf; border-color:#2d6cdf; color:#fff; font-weight:600; }
  #rec { width:100%; padding:22px; font-size:19px; font-weight:700; margin-top:14px; }
  #rec.live { background:#c62828; border-color:#c62828; color:#fff; }
  #st { margin-top:10px; font-size:13px; color:#999; text-align:center; }
  .big { display:block; text-align:center; font-size:26px; font-weight:700; color:#fff; margin:10px 0 2px; }
</style></head><body>

<img id="v" alt="camera preview">
<span class="big" id="cur">rest</span>
<div id="st">connecting…</div>

<h2>Action</h2><div class="row" id="acts"></div>
<h2>Position</h2><div class="row" id="poss"></div>
<h2>Lighting</h2><div class="row" id="lits"></div>
<button id="rec">● START RECORDING</button>

<script>
const ws = new WebSocket(`ws://${location.hostname}:__WS_PORT__`);
ws.binaryType = "arraybuffer";
let url = null, S = {};

ws.onmessage = (e) => {
  if (typeof e.data !== "string") {                   // preview jpeg
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(new Blob([e.data], {type:"image/jpeg"}));
    document.getElementById("v").src = url;
    return;
  }
  S = JSON.parse(e.data);
  if (S.type !== "status") return;
  paint("acts", S.actions,   S.action,   (v)=>({action:v}));
  paint("poss", S.positions, S.position, (v)=>({position:v}));
  paint("lits", S.lightings, S.lighting, (v)=>({lighting:v}));
  document.getElementById("cur").textContent = S.action;
  const r = document.getElementById("rec");
  r.className = S.recording ? "live" : "";
  r.textContent = S.recording ? "■ STOP RECORDING" : "● START RECORDING";
  document.getElementById("st").textContent =
    S.recording ? `recording ${S.session} — ${S.frames} frames` : "idle";
};
ws.onclose = () => document.getElementById("st").textContent = "disconnected";

function paint(id, opts, cur, mk) {
  const el = document.getElementById(id);
  if (el.dataset.n != opts.length) {                  // build once, then just restyle
    el.innerHTML = "";
    for (const o of opts) {
      const b = document.createElement("button");
      b.textContent = o.replace(/_/g," ");
      b.onclick = () => ws.send(JSON.stringify({type:"set", ...mk(o)}));
      b.dataset.v = o;
      el.appendChild(b);
    }
    el.dataset.n = opts.length;
  }
  for (const b of el.children) b.className = (b.dataset.v === cur) ? "on" : "";
}

document.getElementById("rec").onclick =
  () => ws.send(JSON.stringify({type: S.recording ? "stop" : "start"}));
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="phone-controlled Kinect capture rig")
    p.add_argument("--out", default="recordings", help="root directory for sessions")
    p.add_argument("--http-port", type=int, default=8080)
    p.add_argument("--ws-port", type=int, default=8766)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    RecordServer(args.out, args.http_port, args.ws_port).run()


if __name__ == "__main__":
    main()
