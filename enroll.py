"""
enroll.py — face enrollment web UI (mobile-friendly)

open http://<your-ip>:5001 on your phone

grid:
  cols (X): 5 positions, left → right
  rows (Z): 3 distances, near → far

per-cell: captures N_CAPTURES_PER_CELL face crops + embeddings via insightface
saves to enroll/<person_name>/ as .npz files

usage:
    python enroll.py --name "Alice"
    python enroll.py --name "Alice" --captures 5   # per cell, default 3

press "Capture" on your phone when you're in position.
the grid highlights your current cell and which cells are done.
"""

import argparse
import base64
import json
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

# ── lazy imports (need kinect hw) ──────────────────────────────────────────────
import sys
sys.path.insert(0, ".")
from kinect import Kinect
from pose import PoseTracker, associate_faces_to_bodies
from hands import _is_dark, _ir_to_detection_image

# insightface
import insightface
from insightface.app import FaceAnalysis

# ── grid config ────────────────────────────────────────────────────────────────
GRID_COLS = 5   # X positions: left → right
GRID_ROWS = 3   # Z positions: near → far

# depth thresholds (mm) for Z buckets — tune to your space
# near: <1200mm, mid: 1200-1800mm, far: >1800mm
Z_THRESHOLDS = [0, 1200, 1800, 9999]   # len = GRID_ROWS + 1

# X thresholds as fraction of frame width (0.0 = left, 1.0 = right)
X_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # len = GRID_COLS + 1

RGB_W = 1920

N_CAPTURES_PER_CELL = 3   # overridden by --captures

DISPLAY_W = 640
DISPLAY_H = 360

# ── state ──────────────────────────────────────────────────────────────────────
_state = {
    "person_name":   "",
    "current_cell":  None,   # (col, row) or None if not detected
    "done_cells":    {},      # (col, row) -> count of captures
    "last_frame_b64": None,
    "status":        "waiting",   # waiting | detected | captured | done
    "status_msg":    "Stand in front of the Kinect",
    "total_cells":   GRID_COLS * GRID_ROWS,
    "n_per_cell":    N_CAPTURES_PER_CELL,
}
_lock = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────

def x_to_col(x_px: int) -> int:
    frac = x_px / RGB_W
    for i in range(GRID_COLS):
        if X_FRACTIONS[i] <= frac < X_FRACTIONS[i + 1]:
            return i
    return GRID_COLS - 1


def z_to_row(z_mm: float) -> int | None:
    """near=row 0, far=row 2. returns None if z is 0 (no depth data)"""
    if z_mm <= 0:
        return None
    for i in range(GRID_ROWS):
        if Z_THRESHOLDS[i] <= z_mm < Z_THRESHOLDS[i + 1]:
            return i
    return GRID_ROWS - 1


def frame_to_b64(bgr: np.ndarray) -> str:
    small = cv2.resize(bgr, (DISPLAY_W, DISPLAY_H))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def save_capture(name: str, cell: tuple, face_img: np.ndarray, embedding: np.ndarray,
                 index: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    col, row = cell
    fname = f"{name}_c{col}r{row}_{index:02d}.npz"
    np.savez(os.path.join(out_dir, fname),
             face_img=face_img,
             embedding=embedding,
             cell=np.array(cell),
             name=np.array(name))


# ── capture trigger interface ──────────────────────────────────────────────────
# swap this class out for AutoCaptureTrigger when you want dwell-based capture

class ManualTrigger:
    """fires when capture() is called externally (e.g. from HTTP endpoint)"""
    def __init__(self):
        self._pending = threading.Event()

    def request_capture(self):
        self._pending.set()

    def should_capture(self) -> bool:
        if self._pending.is_set():
            self._pending.clear()
            return True
        return False

    def reset(self):
        self._pending.clear()


# placeholder for future auto-capture
class DwellTrigger:
    """fires after person holds position for dwell_s seconds"""
    def __init__(self, dwell_s: float = 2.0):
        self.dwell_s = dwell_s
        self._cell_since = {}   # cell -> time first seen
        self._fired = set()     # cells already fired this dwell

    def should_capture(self, cell) -> bool:
        if cell is None:
            return False
        now = time.time()
        if cell not in self._cell_since:
            self._cell_since[cell] = now
        elapsed = now - self._cell_since[cell]
        if elapsed >= self.dwell_s and cell not in self._fired:
            self._fired.add(cell)
            return True
        return False

    def reset_cell(self, cell):
        self._cell_since.pop(cell, None)
        self._fired.discard(cell)

    def reset(self):
        self._cell_since.clear()
        self._fired.clear()

    def dwell_progress(self, cell) -> float:
        """0.0-1.0 progress toward dwell threshold"""
        if cell not in self._cell_since:
            return 0.0
        return min(1.0, (time.time() - self._cell_since[cell]) / self.dwell_s)


# ── flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
_trigger = ManualTrigger()


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/state")
def get_state():
    with _lock:
        return jsonify({
            "current_cell":   _state["current_cell"],
            "done_cells":     {f"{k[0]},{k[1]}": v for k, v in _state["done_cells"].items()},
            "status":         _state["status"],
            "status_msg":     _state["status_msg"],
            "n_per_cell":     _state["n_per_cell"],
            "total_captured": sum(_state["done_cells"].values()),
            "total_needed":   _state["total_cells"] * _state["n_per_cell"],
            "person_name":    _state["person_name"],
            "frame_b64":      _state["last_frame_b64"],
        })


@app.route("/capture", methods=["POST"])
def capture():
    _trigger.request_capture()
    return jsonify({"ok": True})


# ── tracking loop (runs in background thread) ──────────────────────────────────

def tracking_loop(person_name: str, n_per_cell: int, out_dir: str):
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    with Kinect() as kinect, PoseTracker() as pose_tracker:
        print("warming up kinect...")
        for _ in range(30):
            kinect.get_frames()
        print("ready — open the web UI on your phone")

        while True:
            rgb, depth, ir = kinect.get_frames()
            if rgb is None:
                continue

            bgr = rgb[:, :, :3]

            # ── face detection (IR fallback in low light) ──────────────────
            use_ir = ir is not None and _is_dark(rgb)
            if use_ir:
                detect_img = _ir_to_detection_image(ir)
            else:
                detect_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            faces_raw = face_app.get(detect_img)
            faces = [{"bbox": tuple(map(int, f.bbox)), "embedding": f.embedding,
                      "kps": f.kps} for f in faces_raw]

            # ── pose / skeleton ────────────────────────────────────────────
            bodies = pose_tracker.process(rgb, depth, ir)

            # ── associate face → body ──────────────────────────────────────
            associations = associate_faces_to_bodies(faces, bodies)

            # pick the primary subject (closest / largest face)
            subject_face = None
            subject_body = None
            if associations:
                # largest face bbox = most prominent person
                best = max(associations, key=lambda a: (
                    (a["face"]["bbox"][2] - a["face"]["bbox"][0]) *
                    (a["face"]["bbox"][3] - a["face"]["bbox"][1])
                ))
                subject_face = best["face"]
                subject_body = best["body"]

            # ── determine grid cell ────────────────────────────────────────
            current_cell = None
            if subject_face and subject_body:
                nx, ny, nz = subject_body["nose"]
                col = x_to_col(nx)
                row = z_to_row(nz)
                if row is not None:
                    current_cell = (col, row)

            # ── draw overlay (use IR image as base in low light) ───────────
            if use_ir:
                vis = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
            else:
                vis = bgr.copy()
            if subject_face:
                x1, y1, x2, y2 = subject_face["bbox"]
                color = (80, 220, 80) if current_cell else (80, 80, 220)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            if subject_body:
                nx, ny, _ = subject_body["nose"]
                cv2.circle(vis, (nx, ny), 10, (0, 255, 255), -1)

            frame_b64 = frame_to_b64(vis)

            # ── capture logic ──────────────────────────────────────────────
            with _lock:
                done = dict(_state["done_cells"])

            captured_now = False
            if _trigger.should_capture() and current_cell and subject_face:
                n_so_far = done.get(current_cell, 0)
                if n_so_far < n_per_cell:
                    x1, y1, x2, y2 = subject_face["bbox"]
                    face_crop = bgr[max(0,y1):y2, max(0,x1):x2]
                    embedding = subject_face["embedding"]
                    save_capture(person_name, current_cell, face_crop,
                                 embedding, n_so_far, out_dir)
                    done[current_cell] = n_so_far + 1
                    captured_now = True

            # ── update state ───────────────────────────────────────────────
            total_captured = sum(done.values())
            total_needed   = GRID_COLS * GRID_ROWS * n_per_cell

            if total_captured >= total_needed:
                status = "done"
                msg = f"Enrollment complete! {total_captured} captures saved."
            elif captured_now:
                col, row = current_cell
                status = "captured"
                msg = f"Captured! Cell ({col+1},{row+1}): {done[current_cell]}/{n_per_cell}"
            elif current_cell:
                col, row = current_cell
                n_here = done.get(current_cell, 0)
                if n_here >= n_per_cell:
                    status = "detected"
                    msg = f"Cell ({col+1},{row+1}) done ✓ — move to next position"
                else:
                    status = "detected"
                    msg = f"Cell ({col+1},{row+1}): {n_here}/{n_per_cell} — ready to capture"
            else:
                status = "waiting"
                msg = "Stand in front of the Kinect"

            with _lock:
                _state["current_cell"]   = current_cell
                _state["done_cells"]     = done
                _state["status"]         = status
                _state["status_msg"]     = msg
                _state["last_frame_b64"] = frame_b64
                _state["person_name"]    = person_name
                _state["n_per_cell"]     = n_per_cell


# ── HTML (single-file, no external deps) ──────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Enrollment</title>
<style>
  :root {
    --bg:       #0d0d0f;
    --surface:  #16161a;
    --border:   #2a2a30;
    --text:     #e8e8ec;
    --muted:    #666672;
    --green:    #3ddc84;
    --amber:    #f0b429;
    --blue:     #4a9eff;
    --red:      #ff5c5c;
    --cell-gap: 6px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Fira Mono', monospace;
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
  }

  /* header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .person-name { font-size: 1.1rem; font-weight: 600; letter-spacing: .02em; }
  .progress-label { font-size: .75rem; color: var(--muted); }

  /* progress bar */
  .progress-track {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--green);
    border-radius: 2px;
    transition: width .3s ease;
    width: 0%;
  }

  /* camera preview */
  .preview-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 16/9;
    background: var(--surface);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .preview-wrap img {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
  }
  .preview-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .75rem;
    color: var(--muted);
  }

  /* grid */
  .grid-section { display: flex; flex-direction: column; gap: 8px; }
  .axis-label {
    font-size: .65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .1em;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: var(--cell-gap);
  }
  .cell {
    aspect-ratio: 1;
    border-radius: 6px;
    border: 1.5px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: .6rem;
    color: var(--muted);
    position: relative;
    transition: border-color .15s, background .15s;
  }
  .cell.done {
    border-color: var(--green);
    background: #1a2e1f;
    color: var(--green);
  }
  .cell.current {
    border-color: var(--amber);
    background: #2a2010;
    color: var(--amber);
  }
  .cell.current.done {
    border-color: var(--green);
    background: #1a2e1f;
    color: var(--green);
  }
  .cell-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    margin-bottom: 3px;
  }
  .cell-count { font-size: .55rem; }

  /* you-are-here indicator */
  .you-marker {
    position: absolute;
    top: 3px; right: 3px;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--amber);
  }

  /* z-axis labels on left */
  .grid-with-labels {
    display: grid;
    grid-template-columns: 28px 1fr;
    gap: 6px;
    align-items: center;
  }
  .z-labels {
    display: grid;
    grid-template-rows: repeat(3, 1fr);
    gap: var(--cell-gap);
    height: 100%;
  }
  .z-label {
    font-size: .55rem;
    color: var(--muted);
    text-align: right;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 4px;
  }
  .grid-rows {
    display: grid;
    grid-template-rows: repeat(3, 1fr);
    gap: var(--cell-gap);
  }

  /* status */
  .status-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: .8rem;
    line-height: 1.4;
    min-height: 48px;
    display: flex;
    align-items: center;
  }
  .status-bar.detected { border-color: var(--amber); color: var(--amber); }
  .status-bar.captured { border-color: var(--green); color: var(--green); }
  .status-bar.done     { border-color: var(--green); color: var(--green); }
  .status-bar.waiting  { color: var(--muted); }

  /* capture button */
  .capture-btn {
    width: 100%;
    padding: 18px;
    font-size: 1.1rem;
    font-family: inherit;
    font-weight: 700;
    letter-spacing: .05em;
    background: var(--green);
    color: #0d1a10;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: opacity .1s, transform .1s;
    -webkit-tap-highlight-color: transparent;
  }
  .capture-btn:active { opacity: .8; transform: scale(.98); }
  .capture-btn:disabled {
    background: var(--border);
    color: var(--muted);
    cursor: default;
    transform: none; opacity: 1;
  }
</style>
</head>
<body>

<header>
  <div class="person-name" id="personName">—</div>
  <div class="progress-label" id="progressLabel">0 / 0</div>
</header>

<div class="progress-track">
  <div class="progress-fill" id="progressFill"></div>
</div>

<div class="preview-wrap">
  <img id="preview" src="" alt="">
  <div class="preview-overlay" id="previewOverlay">connecting…</div>
</div>

<div class="grid-section">
  <div class="axis-label">← left &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; right →</div>
  <div class="grid-with-labels">
    <div class="z-labels">
      <div class="z-label">near</div>
      <div class="z-label">mid</div>
      <div class="z-label">far</div>
    </div>
    <div class="grid-rows" id="gridRows"></div>
  </div>
</div>

<div class="status-bar waiting" id="statusBar">Stand in front of the Kinect</div>

<button class="capture-btn" id="captureBtn" disabled>CAPTURE</button>

<script>
const COLS = 5, ROWS = 3;
let cells = {};   // "col,row" -> DOM element
let lastFrameB64 = null;

// build grid
const gridRows = document.getElementById('gridRows');
for (let row = 0; row < ROWS; row++) {
  const rowEl = document.createElement('div');
  rowEl.className = 'grid';
  for (let col = 0; col < COLS; col++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.innerHTML = '<div class="cell-dot"></div><div class="cell-count">0</div>';
    cells[`${col},${row}`] = cell;
    rowEl.appendChild(cell);
  }
  gridRows.appendChild(rowEl);
}

function updateUI(s) {
  // name + progress
  document.getElementById('personName').textContent = s.person_name || '—';
  const pct = s.total_needed > 0
    ? Math.round(s.total_captured / s.total_needed * 100) : 0;
  document.getElementById('progressLabel').textContent =
    `${s.total_captured} / ${s.total_needed}`;
  document.getElementById('progressFill').style.width = pct + '%';

  // camera frame
  if (s.frame_b64) {
    const img = document.getElementById('preview');
    img.src = 'data:image/jpeg;base64,' + s.frame_b64;
    document.getElementById('previewOverlay').style.display = 'none';
    img.style.display = 'block';
  }

  // grid cells
  for (const key in cells) {
    const [col, row] = key.split(',').map(Number);
    const el = cells[key];
    const count = s.done_cells[key] || 0;
    const isCurrent = s.current_cell &&
      s.current_cell[0] === col && s.current_cell[1] === row;
    const isDone = count >= s.n_per_cell;

    el.className = 'cell' + (isDone ? ' done' : '') + (isCurrent ? ' current' : '');
    el.querySelector('.cell-count').textContent = `${count}/${s.n_per_cell}`;

    // you-marker
    let marker = el.querySelector('.you-marker');
    if (isCurrent && !isDone) {
      if (!marker) {
        marker = document.createElement('div');
        marker.className = 'you-marker';
        el.appendChild(marker);
      }
    } else if (marker) {
      marker.remove();
    }
  }

  // status
  const bar = document.getElementById('statusBar');
  bar.textContent = s.status_msg;
  bar.className = 'status-bar ' + s.status;

  // button
  const btn = document.getElementById('captureBtn');
  if (s.status === 'done') {
    btn.textContent = 'COMPLETE ✓';
    btn.disabled = true;
  } else {
    btn.textContent = 'CAPTURE';
    btn.disabled = (s.status === 'waiting');
  }
}

// poll state
async function poll() {
  try {
    const r = await fetch('/state');
    const s = await r.json();
    updateUI(s);
  } catch(e) {}
  setTimeout(poll, 150);
}
poll();

// capture button
document.getElementById('captureBtn').addEventListener('click', async () => {
  await fetch('/capture', { method: 'POST' });
});
</script>
</body>
</html>
"""


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",     required=True, help="person name for enrollment")
    parser.add_argument("--captures", type=int, default=N_CAPTURES_PER_CELL,
                        help="captures per grid cell (default 3)")
    parser.add_argument("--out",      default="enroll", help="output directory")
    parser.add_argument("--port",     type=int, default=5001)
    args = parser.parse_args()

    out_dir = os.path.join(args.out, args.name)
    print(f"enrolling: {args.name}")
    print(f"output:    {out_dir}")
    print(f"grid:      {GRID_COLS}×{GRID_ROWS}, {args.captures} captures/cell "
          f"({GRID_COLS * GRID_ROWS * args.captures} total)")

    with _lock:
        _state["person_name"] = args.name
        _state["n_per_cell"]  = args.captures

    t = threading.Thread(
        target=tracking_loop,
        args=(args.name, args.captures, out_dir),
        daemon=True,
    )
    t.start()

    print(f"\nopen http://<your-ip>:{args.port} on your phone\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
