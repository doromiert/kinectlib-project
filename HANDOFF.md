# nkit — handoff

State as of 2026-07-28. Read `nkit — Overview.md` first for what the project
is; this covers what changed, what's proven, and what's still broken.

---

## TL;DR

- **Perf is solved.** 132ms/frame → ~46ms (7.5 → ~21fps). Don't re-optimise
  without profiling; the bottleneck was never the ML.
- **A serious depth bug is fixed.** 21% of frames had garbage joint depth.
  This invalidated every gesture measurement taken before it.
- **Push is still not working.** Four hand-designed features were measured at
  or near chance. The current implementation is the fifth attempt and is
  unvalidated live.
- **Infrastructure to stop guessing now exists**: `nkit/record.py` (capture)
  + `nkit/replay.py` (replay through the live pipeline).

---

## The one rule

**Never tune a gesture against synthetic landmarks.** Four consecutive push
fixes were reasoned from geometry, "verified" against landmarks generated in
the test itself, and every one failed on real hands. Synthetic data is
noiseless; it tests the maths and says nothing about the signal.

Record, then replay, then change something. See
`~/.claude/projects/.../memory/nkit-replay-before-tuning.md`.

Corollary: sanity-check derived quantities against physical constants. The
depth bug was found by noticing `|shoulder→elbow|` — a *bone* — measuring
270mm one frame and 2.6m the next.

---

## What's proven (measured, trust these)

### Performance: 132ms → ~46ms/frame

| stage | before | after |
|---|---|---|
| `grab` | 22.3 | ~1 |
| `roi` | 3.0 | 0.0 |
| `body_hands` | 52.9 | ~33 |
| `frame_stream` | 12.9 | 0.0 |
| **total** | **132.8** | **~46** |

What did it, in order of payoff:

1. `is_dark()` read all 2M pixels to compute one average brightness, and
   `detection_image()` called it 3×/frame — **~39ms/frame**. Now subsampled.
2. Full-frame CLAHE (25ms). Now downscales *before* CLAHE, stride-slicing
   (`img[::n, ::n]`, ~1ms) rather than `cv2.resize` (~12ms regardless of
   interpolation — the `[:, :, :3]` slice off BGRA is non-contiguous and
   forces a full copy).
3. `ThreadedKinect` — `kinect_grab()` blocks most of a 30fps interval doing
   nothing. Now overlapped.
4. Frame stream (JPEG) moved to its own thread.

**MediaPipe's GPU delegate works on AMD** — it's OpenGL ES compute shaders,
not CUDA/ROCm (pose 7.9ms → 4.9ms). It's on with CPU fallback (`nkit/_mp.py`).
Note it logs `tensor.cc: Tensors are designed for single writes` with two
detectors on GPU; unverified whether that corrupts anything. `use_gpu=False`
on both tracker constructors to rule it out.

### Depth sampling was broken — this is the big one

`depth_at()` medianed a 7×7 patch at each landmark. At a limb that patch
straddles the person and the background, and once over half of it lands on
background **the median is the background**.

| | median | p90 | max | frames >600mm |
|---|---|---|---|---|
| before | 277 | 925 | 2658 | **21%** |
| after | 266 | 337 | 970 | **0%** |

(`|shoulder→elbow|`, a bone, ~270mm, cannot change length.)

Fixed by gating samples against the torso depth, **asymmetrically** — a hand
reaches far in *front* of the torso but never far behind it. A symmetric
±600mm window was tried first and threw away every sample on 55% of push
frames.

**Anything measured before this fix is untrustworthy.**

### ROI: landmark-driven, and it's the range fix

`PersonRoiTracker` (depth-blob) returned **full-frame on 60/60 frames** at
desk range — indoors everything sits inside any sane depth band, and
floor/walls stay connected to everything. Replaced with
`LandmarkRoiTracker` (detect-then-track off the previous frame's landmarks).

This is not just perf: **mediapipe resizes its input to 256×256 regardless**,
so a person at 4m in a full 1920×1080 frame is only ~95px tall in model
space. Cropping to the person fills that 256px at any range. Measured crop:
19% of frame close up, 5% at distance.

---

## What's broken

### Push detection

Current implementation (`gestures.py:_update_push_travel`) requires forward
travel (`push_travel_mm`, 200mm) **and** speed (`push_min_velocity`,
450mm/s), measured body-relative so it's distance-invariant. Streams
`push_progress` 0.0–1.0 for the UI ring, `push_cancel` on abort.

**Unvalidated live.** Last user feedback on the previous iteration was "feels
the same (bad) — too unreliable/slow to react forward, too hyperactive
backward." The dropout fix below directly targets that and has not been tried.

Features measured and **rejected** (don't retry these):

| feature | push vs reach_forward |
|---|---|
| arm extension ratio `\|sh→wr\|/\|sh→el\|` | 51% (chance 51%) |
| static forward offset | 71% (chance 70%) |
| peak forward velocity | 88% (chance 81%) |
| motion axiality (z vs xy path) | 68% balanced |
| 10-feature logistic regression, sliding window | 47/56 push, **19/24 negatives also fired** |

**Why**: the user naturally holds their wrist near their shoulder and pushes
with a folded arm. Peri-tap averaging showed a deliberate *reach* produces a
clean ~174mm forward excursion peaking 0.5s after the mark, while their
slight push produced **no coherent excursion at all** — inside the ±50–100mm
noise floor. The signal was never in the data. That's why the design moved to
a deliberate push with an arming indicator.

### The likely current culprit: depth dropout when extended

| elbow forward | wrist depth valid |
|---|---|
| 0–300mm | 96–99% |
| **400–500mm** | **71%** |

A hand held toward the camera is a small target whose depth patch is mostly
background. Rates computed across those gaps are fiction — which explains
"dead on the way out, hyperactive on the way back" exactly, since the return
is when depth comes good again. Mitigated (velocity history cleared across
gaps; armed push survives 0.35s of missing depth) but **not eliminated**.

**If push still fails, this is where to look**, and the promising alternative
is to stop sampling depth *at a point*. As the hand approaches the camera its
apparent **size** grows, in both RGB and the depth image. Hand area is a far
more robust proximity cue than a point sample on a small target, because it
degrades gracefully instead of dropping out.

### Untouched

- **Swipe has never worked, not once.** Never diagnosed. "Never fires" is a
  bug signature, not a tuning one — a bad model degrades gracefully. Suspect
  the cursor never enters the edge band, or the state machine can't advance.
  A live trace of `cursor_move` x/y would answer it in minutes.
- **Fist is shaky.** Probably wants One Euro on `fist_score` plus hysteresis.
- Sensor loop can die while the WebSocket server stays up, so the app
  connects to a zombie that looks healthy. Happened once, cause unknown. The
  loop dying should take the process down.
- `push_release_frac` is now unused (left in to avoid churn).

---

## Data

- `recordings/` — 124 sessions, sticky-label format, desk/light. Each take is
  one short clip (~13 frames, 0.4s) of pure gesture. **Not usable for
  streaming detection**: pre-segmented clips contain no onset to learn from,
  and clip duration leaks as a feature (a take-level classifier scored 96%
  largely by learning "short clip = push").
- `recordings-livingroom/` — 38 sessions, same format, living room.
- `recordings-marked/` — **the good one.** 2 continuous takes (1.7min dark,
  2.7min light), 125 timestamp marks, 57–73% unmarked context. This is the
  format to use.

All gitignored (`recordings*`). ~4GB total.

### Recording

```
nix run .#record-server        # then open http://<lan-ip>:8080 on a phone
```

Record **continuously** and tap a button at the moment of each gesture;
unmarked time is implicitly rest. Do not record one clip per gesture.

The tap lands *before* the gesture — peri-tap averaging put the motion peak
at **t=+0.5s**. `replay.label_frames()` windows accordingly (`before_s=0.1`,
`after_s=1.0`). A symmetric window caught push's *retraction* and gave it a
median backward displacement, poisoning everything trained on it.

### Analysing

```python
from nkit.replay import sessions, frames, marks, label_frames
for s in sessions("recordings-marked"):
    labels = label_frames(s)
    for rgb, depth, ir, meta in frames(s):
        ...   # run whatever pipeline you like
```

Full re-extraction over ~4000 frames takes ~2 min.

---

## Constraints

- **Must work at desk (<1.5m) AND living room (2–4m).** No distance
  assumptions anywhere — that's what killed the depth-blob ROI. Prefer
  body-relative quantities, which are distance-invariant by construction.
- The user's natural posture is **wrist near shoulder, folded arm**.
- One Kinect, one process. Stop the bridge before running the recorder.
- `pkill -f "nkit.bridge"` **matches its own shell wrapper** and kills the
  command. Kill by port instead:
  `kill $(ss -ltnp | grep 8765 | grep -oP 'pid=\K[0-9]+')`
- Python buffers stdout when piped — run the bridge with `python3 -u` or you
  see no logs.

---

## Suggested next steps

1. **Try the current push live.** The dropout fix is the best-grounded change
   yet and is untested.
2. **If it still fails**, switch the proximity cue from point-depth to hand
   *area* (see above) rather than tuning thresholds again.
3. **Diagnose swipe.** Cheapest open win — it's a bug, not a tuning problem.
4. Fist: One Euro on `fist_score` + hysteresis.
5. Make a dead sensor loop kill the process.

Everything in "what's proven" is independent of how push resolves and can be
built on.
