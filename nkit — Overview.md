# nkit — Negative Zero Kinect Interface Toolkit — Overview

Scoping doc before I start building. Supersedes `Negative Zero Kinect SDK.md`
(the `nzk` design) — same idea, renamed, plus the capabilities below.
`finallib/` was the first pass at `nzk`; it's gone from the working tree but
still in git history (`git show HEAD:finallib/...`) and is the base `nkit`
builds from, not a from-scratch rewrite.

This revision: UI stack decided (HTML/CSS/JS, hardware-accelerated, in
gamescope), which changes how nkit talks to the app — it now needs a real
IPC bridge, not just a Python API. Also adding live-tunable gesture
thresholds since we're already building a debug UI.

---

## Decisions locked so far

1. **RGB+IR fusion** — adaptive (best single source per frame) is the
   default. Full dual-source fusion behind `enable_fusion=True`.
2. **AEC** — `speexdsp`.
3. **STT input handling** — callback-only, nkit never touches the OS.
4. **Final app UI** — HTML + CSS + JS, hardware-accelerated animations,
   Metro-style, running inside **gamescope** for embedding.
5. **Gesture thresholds** — live-tunable via sliders, not just hardcoded
   constants (§F/§G).
6. **App shell runtime — Electron, not plain kiosk Chromium.** The emulator
   requirement is what tips it: launching an emulator process and telling
   gamescope to focus/embed it needs real OS access — spawning processes,
   talking to gamescope — and a page running in plain browser kiosk mode is
   sandboxed away from all of that on purpose, no way around it from JS
   alone. Electron's Node main process gives you that without leaving
   JS/Node. Tauri was the other option, but on Linux its webview is
   WebKitGTK, not Chromium — hardware-accelerated CSS animation/compositing
   there has historically been shakier than Chromium's, especially nested
   inside a Wayland compositor like gamescope, which is a real risk given
   animations are the thing you most explicitly care about. Electron keeps
   the exact same Chromium rendering engine the kiosk-only plan already
   assumed, so nothing about the animation approach in §H changes — it just
   also gets a privileged backend process. §F's IPC bridge choice
   (WebSocket) stands either way.

---

## Requirements → status

| # | Requirement | Status |
|---|---|---|
| 1 | Skeleton tracking on RGB/IR/depth | Settled (§A) |
| 2 | Face → known-identity match against a specifiable dir | Already works as-asked |
| 3 | Per-hand, per-skeleton tracking, palm/fist, gestures | Association (§B) + vocabulary (§C) designed |
| 4 | Wake word + calibratable AEC + STT | AEC new (§D), STT callback-only (§E) |

---

## A. RGB+IR+depth (settled)

`_vision.py` (renamed from `_ir.py`) keeps the adaptive brightness-based
source switch as default. `enable_fusion=True` runs both RGB and IR through
the detector and merges landmarks weighted by local contrast per source —
roughly 2x inference cost. Depth is unconditionally used for Z either way.

## B. Hand ↔ skeleton association

`associate_hands_to_bodies()` in `body.py`, same shape as
`associate_faces_to_bodies()` — nearest-neighbour match of wrist positions,
tags each hand with `skeleton_id` + body-relative `side`. This pairing is
the stable per-hand identity key §C tracks against, since mediapipe gives
no persistent hand ID.

## C. Gesture vocabulary

`gestures.py` sits on top of `hands.py`/`body.py`, consumes the per-frame
`HandResult` stream, emits discrete `GestureEvent`s per hand-identity.

**Design call, unchanged:** nkit does not hit-test UI elements. It emits a
continuous cursor position plus discrete primitives; the DOM (§F/§G) does
its own hit-testing via `document.elementFromPoint()`, same as it would for
a mouse. "Hover on an element" / "click on an element" / "click on empty
space" fall out of normal browser event dispatch once nkit's stream is
translated into synthetic pointer events — not something nkit computes.

```python
@dataclass
class GestureEvent:
    skeleton_id: int
    side:        Literal["left", "right"]
    kind:        Literal[
        "cursor_move",   # continuous; x, y in screen space
        "grab_start", "grab_end",     # fist held past a debounce window
        "push",                       # palm-mode z-push — maps to "click"
        "swipe_edge",                 # hand entered from a screen edge, moved inward
        "grab_swipe",                 # fist held + moved vertically past threshold
    ]
    x: int | None
    y: int | None
    edge:      Literal["left","right","top","bottom"] | None
    direction: Literal["up","down"] | None
    timestamp: float
```

Mapping: **hover** → `cursor_move` → DOM `pointermove`. **grab** →
`grab_start`/`grab_end` (sustained fist, debounced against a momentary fist
mid-push). **click on element/empty space** → `push` → DOM
`click`/`pointerdown`+`pointerup` at the cursor position; the browser
decides what it landed on. **swipe from an edge** → `swipe_edge`, hand
enters a configurable edge band and moves inward past a distance/velocity
threshold. **grab swipe up/down** → `grab_swipe`, mirrors the actual Xbox
Kinect shell gesture (raise hand, close to a fist to "grab" the screen,
pull down/up), fires while `grab_start` is active and vertical movement
crosses a threshold.

### Gesture sound format

**WAV, PCM16, 48kHz, mono, clips well under 1s.** Uncompressed so there's
zero decode latency between gesture-fires and sound-plays — anything
compressed (MP3/OGG/Opus) pays a small but real and inconsistent decode
cost, which reads as lag on feedback sounds. File size isn't a concern at
these lengths. Optional later: pan stereo by the gesture's screen-space X
for spatial feedback — cheap to add after v1, not needed now.

### Live-tunable thresholds

`GestureConfig` — a plain mutable dataclass holding every threshold
(`push_delta_mm`, `push_window_ms`, `grab_debounce_ms`,
`swipe_edge_band_px`, `swipe_min_distance_px`,
`grab_swipe_min_distance_px`, plus the existing fist-curl threshold/count
and the IR-fallback brightness threshold). `GestureTracker` reads from a
live instance every frame, so updates apply immediately with no restart.
Gets a `save()`/`load()` to/from a small JSON file so once you've tuned it
by feel via the sliders (§G), it persists across runs instead of resetting
to defaults every time.

---

## D. Calibratable self-sound cancellation (AEC)

`speexdsp` echo canceller. Reference signal from the system loopback/
monitor source (PipeWire `*.monitor`), calibration routine measures
delay+gain once per setup by playing a known tone and comparing capture
timing, stored as an offset. Sits ahead of wake word/VAD/whisper in the
audio pipeline.

## E. Voice delivery

Two wake words, two modes, transcript delivered via callback, unchanged.
No OS-level typing or injection inside nkit — consumer's job.

---

## F. IPC bridge — nkit ↔ the web UI

This is new this round, and it's not optional anymore: an HTML/CSS/JS UI
can't import a Python library directly, so nkit needs a real transport.
This is the same idea the original `nzk` design doc flagged as "future
work, add if you hit a wall" — the UI stack decision just moved it from
future work to required now.

**`nkit/bridge.py`** — a local WebSocket server (`websockets`, bind
`127.0.0.1` only, no external exposure) that runs the normal nkit
processing loop and publishes JSON messages to connected clients:

- `GestureEvent`s (§C) as they fire — this is the primary channel the UI
  drives interaction from
- `FaceResult` summaries (name/known/confidence) when recognition changes,
  for personalization ("welcome back Alice")
- optionally, JPEG-encoded frames + skeleton landmark JSON as a *separate*
  lower-priority channel, only for apps that want a camera-feed backdrop
  (the test app does, §G; the Metro UI probably doesn't need this)

Inbound (client → bridge): `GestureConfig` field updates, applied live and
optionally persisted — this is what the threshold sliders talk to.

Framing: newline-delimited JSON is enough for events/config; frames (if
used) go out as separate binary WebSocket messages so JSON parsing never
has to deal with embedded binary blobs.

---

## G. Test app — gesture testbed

Small standalone consumer of nkit, separate from the final app: camera feed
as fullscreen backdrop, skeleton overlay drawn from `BodyResult` landmarks,
one button on top to test hover/grab/click, plus a debug panel of the
threshold sliders from §C, all against a real gesture stream before the
Metro UI exists.

**Same stack as the final app: Electron** (§ Decisions, item 6) — worth
building the test app on Electron from the start even though it doesn't
spawn anything yet, so the packaging/process-model path is exercised early
rather than discovered for the first time in the final app. Renderer talks
to `nkit/bridge.py` over `new WebSocket("ws://127.0.0.1:PORT")`, same as a
plain browser page would — Electron doesn't change that part, it just adds
`main.js` (Node, eventually process spawn + gamescope control) and
`preload.js` (contextBridge, so the renderer stays sandboxed/no
`nodeIntegration`) around it.

Page layout: `<canvas>` for the camera feed (frames drawn via
`createImageBitmap` from the JPEG binary messages) with a second
transparent `<canvas>` on top for skeleton lines, a plain HTML `<button>`,
and a slide-out panel of `<input type="range">` sliders wired to send
`GestureConfig` updates back over the socket as you drag them. `push`
events get dispatched as synthetic `PointerEvent`s at the cursor position
via `document.elementFromPoint(x, y).dispatchEvent(...)` so the button
responds exactly like it would to a mouse click.

---

## H. The final app

Fullscreen (1080p/4k), Metro-style, HTML/CSS/JS with GPU-accelerated
animations (CSS `transform`/`opacity` transitions, which Chromium
composites on the GPU — that's the animation path to lean on for tile
motion, no canvas/WebGL needed unless you want something fancier later),
embedding other apps gamescope-style, Wayland-only.

**Embedding — `gamescope`, not a custom compositor.** Gamescope already
solves DRM/KMS output and embedding arbitrary Wayland/X11 clients; this is
the Steam Deck model (gamescope as session compositor, the Qt Steam UI and
each game as separate gamescope-managed sessions you switch focus between).
Writing a compositor (`smithay`/Rust) would give tighter control over
transitions tied to swipe gestures but is a multi-week project duplicating
what gamescope already does.

**Open risk to spike early, unchanged from last round:** gamescope's
programmatic focus/session switching is less documented than something
like `swaymsg`. Worth a short spike — launch gamescope, launch two nested
clients (the kiosk Chromium shell + one throwaway app), confirm you can
switch focus between them programmatically — before committing further.

**Shape:** gamescope hosts the Electron shell (the Metro UI) plus whichever
apps get embedded, emulators included. The shell's renderer connects to the
same `nkit/bridge.py` WebSocket as the test app, driving Metro tile
hover/press/launch animations off `GestureEvent`s, and (via `FaceResult`)
personalizing what's shown per recognized person. `grab_swipe` up/down is
the natural hook for the shell-switch gesture (bring up the shell over
whatever's currently focused, Xbox-Kinect-shell-style) — same event your
test app already exercises via §G, just wired to gamescope focus calls
instead of a button. Launching an emulator (or any embedded app) is
Electron's `main.js` spawning the process and telling gamescope to focus
it; the renderer never touches the OS directly, it just tells `main.js`
"launch this" over Electron's own IPC.

**Two different things "embed" could mean here — worth pinning down before
I build it, since they're very different amounts of work:**

- **Full-screen swap** (my default assumption so far): launching an
  emulator gamescope-focuses it fullscreen, hiding the Metro shell
  entirely, and `grab_swipe` brings the shell back over it — exactly the
  Steam Deck / Xbox-shell model. This is "just" process spawn + gamescope
  focus calls, no different in kind from launching any other embedded app.
- **Live tile inside the DOM** — the emulator's actual video rendered
  *inside* a Metro tile alongside other UI (a "continue playing" tile
  showing the live game), not fullscreen. Browser content can't host a
  foreign window's pixels natively, so this needs either (a) gamescope
  positioning the real emulator surface to overlay a specific screen
  region kept in sync with that tile's on-screen bounding box as the UI
  scrolls/animates — fragile, since DOM layout and compositor coordinates
  have to stay pixel-synced every frame — or (b) capturing the emulator's
  frames (PipeWire screen capture) and streaming them into a `<canvas>`,
  same shape as the JPEG frame channel §F/§G already has for the camera
  feed, just a second source.

**Recommendation:** build full-screen swap first — it's a strict subset of
what live-tile would need (you need working launch/focus/return either
way), and it's enough to actually use the emulators. Live-tile previews are
a nice-to-have worth revisiting once the basics work, not a day-one
requirement, unless you disagree and want it from the start.

---

## Renamed module layout

```
nkit/
  __init__.py
  kinect.py             # unchanged from finallib/kinect.py
  _vision.py             # was _ir.py — adaptive by default, enable_fusion flag
  body.py                # + associate_hands_to_bodies()
  hands.py               # palm/fist detection, unchanged
  gestures.py             # GestureEvent state machine + GestureConfig (§C)
  bridge.py                # new — WebSocket server, events out / config in (§F)
  face/
    recognize.py
    enroll.py             # programmatic version of today's Flask enroll.py
  audio/
    aec.py                 # speexdsp echo cancellation + calibration (§D)
    voice.py                 # wake word + VAD + whisper, downstream of aec.py
  types.py

nkit-testapp/             # consumer — Electron: main.js/preload.js + HTML/CSS/JS (§G)
```

New/changed types: `HandResult` gains `skeleton_id`/`side`; new
`GestureEvent`, `GestureConfig` (§C); new `AecCalibration(delay_ms, gain,
quality_score)`.

---

## Resolved

- App runtime: **Electron** (needed for process spawn + gamescope control,
  emulator embedding). WebSocket confirmed for `bridge.py`.

## Open questions

1. OK to spend a short spike on gamescope's programmatic focus control
   before building the rest of §H on top of it? Still the one real
   unknown in this plan.
2. §H embedding depth — fine starting with full-screen swap and treating
   in-DOM live-tile previews as later work, or is a live preview tile
   something you want from day one?
3. Any specific emulator(s) to scope early testing around, or is
   RetroArch (single process, clean CLI launch, covers most consoles via
   cores) a reasonable first target?

Let me know and I'll start on nkit (§A–F) plus the test app (§G) — the
final app (§H) stays notes-only until the gamescope spike lands.
