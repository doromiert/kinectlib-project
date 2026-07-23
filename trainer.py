"""
trainer.py — wake word sample recorder, mobile UI

records "hey zane" and "zane write" at 3 distances (near/mid/far),
10 samples each = 30 per word = 60 total.

saves into CoreWorxLab's expected folder structure:
  wakeword_samples/
    hey_zane/
      positive/
        near_00.wav ... near_09.wav
        mid_00.wav  ... mid_09.wav
        far_00.wav  ... far_09.wav
    zane_write/
      positive/
        near_00.wav ... etc.

when all done, prints the docker command to kick off training.

usage:
    python trainer.py
    open http://<your-ip>:5002 on your phone
    hold RECORD to record, release to stop
"""

import os
import io
import json
import wave
import threading
import time
import subprocess

import pyaudio
from flask import Flask, Response, jsonify, request

# ── audio config ───────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000   # openwakeword expects 16kHz
CHANNELS     = 1
SAMPLE_WIDTH = 2       # 16-bit
CHUNK        = 512

# kinect mic — set KINECT_MIC_INDEX env var if autodetect fails
KINECT_MIC_INDEX = int(os.environ.get("KINECT_MIC_INDEX", -1))

# ── enrollment config ─────────────────────────────────────────────────────────
WAKE_WORDS = ["hey_zane", "zane_write"]
WAKE_WORD_LABELS = {
    "hey_zane":   "hey zane",
    "zane_write": "zane write",
}
DISTANCES     = ["near", "mid", "far"]
DISTANCE_HINT = {
    "near": "~1m away",
    "mid":  "~2m away",
    "far":  "~3m away",
}
N_PER_SLOT    = 10   # recordings per word × distance
OUT_DIR       = "wakeword_samples"

# ── state ─────────────────────────────────────────────────────────────────────
_state = {
    "word":         WAKE_WORDS[0],
    "distance":     DISTANCES[0],
    "done":         {w: {d: 0 for d in DISTANCES} for w in WAKE_WORDS},
    "recording":    False,
    "status":       "ready",     # ready | recording | saved | done
    "status_msg":   "Hold RECORD and say the wake word clearly",
}
_lock        = threading.Lock()
_audio_buf   = []
_rec_thread  = None
_pa          = None
_stream      = None


# ── audio helpers ──────────────────────────────────────────────────────────────

def _find_kinect_mic(pa: pyaudio.PyAudio) -> int:
    """find kinect mic by name, fallback to default"""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info["name"].lower()
        if "kinect" in name or "xbox" in name or "nui" in name:
            print(f"found kinect mic: [{i}] {info['name']}")
            return i
    print("kinect mic not found — using default input")
    return pa.get_default_input_device_info()["index"]


def _rec_worker():
    global _audio_buf, _stream
    _audio_buf = []
    while True:
        with _lock:
            if not _state["recording"]:
                break
        data = _stream.read(CHUNK, exception_on_overflow=False)
        _audio_buf.append(data)


def _save_wav(word: str, distance: str, index: int) -> str:
    out_path = os.path.join(OUT_DIR, word, "positive")
    os.makedirs(out_path, exist_ok=True)
    fname = os.path.join(out_path, f"{distance}_{index:02d}.wav")
    with wave.open(fname, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(_audio_buf))
    return fname


def _next_slot():
    """advance to next unfilled (word, distance) slot, or return None if done"""
    with _lock:
        done = _state["done"]
    for w in WAKE_WORDS:
        for d in DISTANCES:
            if done[w][d] < N_PER_SLOT:
                return w, d
    return None, None


def _total_done():
    with _lock:
        return sum(v for wd in _state["done"].values() for v in wd.values())


def _total_needed():
    return len(WAKE_WORDS) * len(DISTANCES) * N_PER_SLOT


def _print_docker_instructions():
    abs_out = os.path.abspath(OUT_DIR)
    print("\n" + "═" * 60)
    print("  RECORDING COMPLETE — run training:")
    print("═" * 60)
    print("\n  1. clone the training repo (once):")
    print("     git clone https://github.com/CoreWorxLab/openwakeword-training.git")
    print("     cd openwakeword-training")
    print("     docker compose build trainer")
    print()
    print("  2. copy your recordings in:")
    for w in WAKE_WORDS:
        src = os.path.join(abs_out, w, "positive")
        dst = f"data/{w}/positive/"
        print(f"     cp {src}/*.wav {dst}")
    print()
    print("  3. train each word:")
    for w in WAKE_WORDS:
        label = WAKE_WORD_LABELS[w]
        print(f'     docker compose run --rm trainer python train.py --wake-word "{label}"')
    print()
    print(f"  output: my_custom_model/hey_zane.onnx + zane_write.onnx")
    print("═" * 60 + "\n")


# ── flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return TRAINER_HTML


@app.route("/state")
def get_state():
    with _lock:
        s = dict(_state)
        s["done"] = {w: dict(d) for w, d in _state["done"].items()}
    s["total_done"]   = _total_done()
    s["total_needed"] = _total_needed()
    s["n_per_slot"]   = N_PER_SLOT
    s["wake_words"]   = WAKE_WORDS
    s["distances"]    = DISTANCES
    s["distance_hint"] = DISTANCE_HINT
    s["word_labels"]  = WAKE_WORD_LABELS
    return jsonify(s)


@app.route("/record/start", methods=["POST"])
def record_start():
    global _rec_thread, _stream, _pa
    with _lock:
        if _state["recording"] or _state["status"] == "done":
            return jsonify({"ok": False})
        _state["recording"] = True
        _state["status"]    = "recording"
        _state["status_msg"] = f'Recording — say "{WAKE_WORD_LABELS[_state["word"]]}"'

    if _pa is None:
        _pa = pyaudio.PyAudio()
        mic_idx = KINECT_MIC_INDEX if KINECT_MIC_INDEX >= 0 else _find_kinect_mic(_pa)
        _stream = _pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=mic_idx,
            frames_per_buffer=CHUNK,
        )

    _rec_thread = threading.Thread(target=_rec_worker, daemon=True)
    _rec_thread.start()
    return jsonify({"ok": True})


@app.route("/record/stop", methods=["POST"])
def record_stop():
    global _rec_thread
    with _lock:
        if not _state["recording"]:
            return jsonify({"ok": False})
        _state["recording"] = False
        word     = _state["word"]
        distance = _state["distance"]

    if _rec_thread:
        _rec_thread.join(timeout=1.0)

    # need at least ~0.3s of audio
    if len(_audio_buf) < int(SAMPLE_RATE * 0.3 / CHUNK):
        with _lock:
            _state["status"]     = "ready"
            _state["status_msg"] = "Too short — hold longer and try again"
        return jsonify({"ok": False, "reason": "too_short"})

    with _lock:
        idx = _state["done"][word][distance]

    fname = _save_wav(word, distance, idx)

    with _lock:
        _state["done"][word][distance] += 1

    total = _total_done()
    needed = _total_needed()

    if total >= needed:
        with _lock:
            _state["status"]     = "done"
            _state["status_msg"] = "All recordings done! Check the terminal for training instructions."
        _print_docker_instructions()
        return jsonify({"ok": True, "done": True})

    # advance slot
    next_word, next_dist = _next_slot()
    with _lock:
        _state["word"]     = next_word
        _state["distance"] = next_dist
        _state["status"]   = "saved"
        _state["status_msg"] = f"Saved! {total}/{needed}"

    return jsonify({"ok": True, "done": False})


# ── HTML ───────────────────────────────────────────────────────────────────────

TRAINER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Wake Word Trainer</title>
<style>
  :root {
    --bg:      #0d0d0f;
    --surface: #16161a;
    --border:  #2a2a30;
    --text:    #e8e8ec;
    --muted:   #666672;
    --green:   #3ddc84;
    --amber:   #f0b429;
    --red:     #ff5c5c;
    --blue:    #4a9eff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Fira Mono', monospace;
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 16px;
    user-select: none;
    -webkit-user-select: none;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .title    { font-size: 1rem; font-weight: 700; letter-spacing: .04em; }
  .progress { font-size: .75rem; color: var(--muted); }

  .progress-track {
    height: 3px; background: var(--border); border-radius: 2px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: var(--green); border-radius: 2px;
    transition: width .3s ease; width: 0%;
  }

  /* current target */
  .target-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .target-label { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
  .target-word  { font-size: 1.6rem; font-weight: 700; color: var(--amber); letter-spacing: .02em; }
  .target-dist  { font-size: .85rem; color: var(--text); }
  .target-hint  { font-size: .7rem; color: var(--muted); }

  /* word × distance grid */
  .grid-section { display: flex; flex-direction: column; gap: 8px; }
  .grid-label { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
  .word-block { display: flex; flex-direction: column; gap: 4px; }
  .word-row-label { font-size: .7rem; color: var(--muted); padding: 2px 0; }
  .dist-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }
  .slot {
    border-radius: 6px;
    border: 1.5px solid var(--border);
    background: var(--surface);
    padding: 8px 4px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    font-size: .6rem;
    color: var(--muted);
    transition: border-color .15s, background .15s;
  }
  .slot.done    { border-color: var(--green); background: #1a2e1f; color: var(--green); }
  .slot.current { border-color: var(--amber); background: #2a2010; color: var(--amber); }
  .slot.current.done { border-color: var(--green); background: #1a2e1f; color: var(--green); }
  .slot-dist  { font-size: .55rem; text-transform: uppercase; letter-spacing: .08em; }
  .slot-count { font-weight: 700; font-size: .75rem; }

  /* status */
  .status-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: .8rem;
    min-height: 44px;
    display: flex;
    align-items: center;
    transition: border-color .15s, color .15s;
  }
  .status-bar.recording { border-color: var(--red);   color: var(--red); }
  .status-bar.saved     { border-color: var(--green); color: var(--green); }
  .status-bar.done      { border-color: var(--green); color: var(--green); }
  .status-bar.ready     { color: var(--muted); }

  /* record button */
  .record-btn {
    width: 100%;
    padding: 22px;
    font-size: 1.2rem;
    font-family: inherit;
    font-weight: 700;
    letter-spacing: .06em;
    background: var(--red);
    color: #fff;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    transition: opacity .1s, transform .1s, background .15s;
    -webkit-tap-highlight-color: transparent;
    touch-action: none;
  }
  .record-btn.recording { background: #8b1a1a; }
  .record-btn:disabled  { background: var(--border); color: var(--muted); }
  .record-btn:active    { transform: scale(.97); }
</style>
</head>
<body>

<header>
  <div class="title">WAKE WORD TRAINER</div>
  <div class="progress" id="progressLabel">0 / 0</div>
</header>

<div class="progress-track">
  <div class="progress-fill" id="progressFill"></div>
</div>

<div class="target-card">
  <div class="target-label">say this</div>
  <div class="target-word" id="targetWord">—</div>
  <div class="target-dist" id="targetDist">—</div>
  <div class="target-hint" id="targetHint"></div>
</div>

<div class="grid-section">
  <div class="grid-label">progress</div>
  <div id="wordBlocks"></div>
</div>

<div class="status-bar ready" id="statusBar">Loading…</div>

<button class="record-btn" id="recordBtn" disabled>HOLD TO RECORD</button>

<script>
let S = null;
let isRecording = false;

async function poll() {
  try {
    const r = await fetch('/state');
    S = await r.json();
    render(S);
  } catch(e) {}
  setTimeout(poll, 200);
}

function render(s) {
  const done   = s.total_done;
  const needed = s.total_needed;
  const pct    = needed > 0 ? Math.round(done / needed * 100) : 0;

  document.getElementById('progressLabel').textContent = `${done} / ${needed}`;
  document.getElementById('progressFill').style.width  = pct + '%';

  const word = s.word;
  const dist = s.distance;

  if (s.status === 'done') {
    document.getElementById('targetWord').textContent = '✓ complete';
    document.getElementById('targetDist').textContent = 'check terminal for training instructions';
    document.getElementById('targetHint').textContent = '';
  } else {
    document.getElementById('targetWord').textContent = '"' + (s.word_labels[word] || word) + '"';
    document.getElementById('targetDist').textContent = dist + '  —  ' + (s.distance_hint[dist] || '');
    document.getElementById('targetHint').textContent =
      `${s.done[word][dist]} / ${s.n_per_slot} at this distance`;
  }

  // build grid if needed
  const container = document.getElementById('wordBlocks');
  if (container.children.length === 0) {
    for (const w of s.wake_words) {
      const block = document.createElement('div');
      block.className = 'word-block';
      const lbl = document.createElement('div');
      lbl.className = 'word-row-label';
      lbl.textContent = s.word_labels[w] || w;
      block.appendChild(lbl);
      const row = document.createElement('div');
      row.className = 'dist-row';
      for (const d of s.distances) {
        const slot = document.createElement('div');
        slot.className = 'slot';
        slot.id = `slot-${w}-${d}`;
        slot.innerHTML = `<div class="slot-dist">${d}</div><div class="slot-count">0</div>`;
        row.appendChild(slot);
      }
      block.appendChild(row);
      container.appendChild(block);
    }
  }

  // update slots
  for (const w of s.wake_words) {
    for (const d of s.distances) {
      const el = document.getElementById(`slot-${w}-${d}`);
      if (!el) continue;
      const count   = s.done[w][d];
      const isDone  = count >= s.n_per_slot;
      const isCur   = (w === s.word && d === s.distance && s.status !== 'done');
      el.className  = 'slot' + (isDone ? ' done' : '') + (isCur ? ' current' : '');
      el.querySelector('.slot-count').textContent = `${count}/${s.n_per_slot}`;
    }
  }

  // status
  const bar = document.getElementById('statusBar');
  bar.textContent  = s.status_msg;
  bar.className    = 'status-bar ' + s.status;

  // button
  const btn = document.getElementById('recordBtn');
  btn.disabled = (s.status === 'done');
  if (isRecording) {
    btn.textContent = '● RECORDING…';
    btn.className   = 'record-btn recording';
  } else {
    btn.textContent = 'HOLD TO RECORD';
    btn.className   = 'record-btn';
  }
}

async function startRec() {
  if (!S || S.status === 'done' || isRecording) return;
  isRecording = true;
  render(S);
  await fetch('/record/start', { method: 'POST' });
}

async function stopRec() {
  if (!isRecording) return;
  isRecording = false;
  await fetch('/record/stop', { method: 'POST' });
}

const btn = document.getElementById('recordBtn');
btn.addEventListener('pointerdown', e => { e.preventDefault(); startRec(); });
btn.addEventListener('pointerup',   e => { e.preventDefault(); stopRec();  });
btn.addEventListener('pointercancel', e => { stopRec(); });

poll();
</script>
</body>
</html>
"""


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--out",  default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"wake word trainer — {_total_needed()} recordings needed")
    print(f"output: {os.path.abspath(OUT_DIR)}")
    print(f"\nopen http://<your-ip>:{args.port} on your phone\n")

    app.run(host="0.0.0.0", port=args.port, threaded=True)
