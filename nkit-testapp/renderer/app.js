// nkit-testapp/renderer/app.js
//
// Connects to nkit/bridge.py over a plain WebSocket (no Node needed for
// this — same code would run in a bare browser tab) and exercises the
// whole sensor pipeline: camera feed + skeleton overlay, per-person face
// ID, per-hand fist/palm state, the gesture vocabulary (dispatched as
// synthetic pointer events at the test button, same as a real UI would
// consume them), live-tunable GestureConfig sliders, and voice/wake-word
// status.

const WS_URL = "ws://127.0.0.1:8765";
const RECONNECT_MS = 1500;

const feedCanvas    = document.getElementById("feed");
const overlayCanvas = document.getElementById("overlay");
const feedCtx       = feedCanvas.getContext("2d");
const overlayCtx     = overlayCanvas.getContext("2d");
const connStatusEl  = document.getElementById("connStatus");
const peopleListEl  = document.getElementById("peopleList");
const voiceStatusEl = document.getElementById("voiceStatus");
const gestureLogEl  = document.getElementById("gestureLog");
const slidersEl     = document.getElementById("sliders");
const persistToggle = document.getElementById("persistToggle");
const cursorsEl     = document.getElementById("cursors");
const testBtn       = document.getElementById("testBtn");
const clickFlash    = document.getElementById("clickFlash");
const quitBtn       = document.getElementById("quitBtn");
const bodyToggle    = document.getElementById("bodyToggle");
const irToggle      = document.getElementById("irToggle");

quitBtn.addEventListener("click", () => window.nkitApp?.quit());

bodyToggle.addEventListener("change", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "body_enabled", value: bodyToggle.checked }));
});

irToggle.addEventListener("change", () => {
  sendConfig({ hand_prefer_ir: irToggle.checked });
});

// ── websocket ────────────────────────────────────────────────────────────────

let ws = null;
let pendingFrameMeta = null;
const people = new Map();   // skeleton_id -> {name, known, confidence, hands: {left, right}}

function connect() {
  ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    connStatusEl.textContent = "connected";
    connStatusEl.className = "status connected";
    sendScreenSize();
    ws.send(JSON.stringify({ type: "body_enabled", value: bodyToggle.checked }));
  };

  ws.onclose = () => {
    connStatusEl.textContent = "disconnected — retrying";
    connStatusEl.className = "status disconnected";
    setTimeout(connect, RECONNECT_MS);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === "frame") {
        pendingFrameMeta = msg;   // binary JPEG follows immediately
      } else {
        handleMessage(msg);
      }
    } else if (pendingFrameMeta) {
      renderFrame(ev.data, pendingFrameMeta);
      pendingFrameMeta = null;
    }
  };
}

function sendConfig(values) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "config", values, persist: persistToggle.checked }));
}

// tells the bridge our REAL window size so gesture x/y land in actual
// viewport pixels instead of whatever --screen-w/--screen-h happened to be
// passed on the command line (see nkit/bridge.py's screen_size handler)
function sendScreenSize() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "screen_size", width: window.innerWidth, height: window.innerHeight }));
}
window.addEventListener("resize", sendScreenSize);

function handleMessage(msg) {
  if (msg.type === "gesture") handleGesture(msg);
  else if (msg.type === "face") handleFace(msg);
  else if (msg.type === "voice") handleVoice(msg);
}

// ── camera feed + skeleton/hand overlay ─────────────────────────────────────────

const BONES = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["right_hip", "right_knee"],
  ["nose", "left_shoulder"], ["nose", "right_shoulder"],
];

async function renderFrame(buf, meta) {
  try {
    const bitmap = await createImageBitmap(new Blob([buf], { type: "image/jpeg" }));
    feedCtx.drawImage(bitmap, 0, 0, feedCanvas.width, feedCanvas.height);
    bitmap.close?.();
  } catch { /* decode hiccup — skip this frame */ }

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const noseBySkeleton = new Map();   // skeleton_id -> [x, y] — only populated when pose is actually tracked
  for (const skel of meta.skeletons || []) {
    overlayCtx.strokeStyle = "#3ddc84";
    overlayCtx.lineWidth = 3;
    for (const [a, b] of BONES) {
      const pa = skel.landmarks[a], pb = skel.landmarks[b];
      if (!pa || !pb) continue;
      overlayCtx.beginPath();
      overlayCtx.moveTo(pa[0], pa[1]);
      overlayCtx.lineTo(pb[0], pb[1]);
      overlayCtx.stroke();
    }
    const nose = skel.landmarks.nose;
    if (nose) {
      noseBySkeleton.set(skel.skeleton_id, nose);
      // head marker — same visual weight as a hand circle, so it actually reads as "the head", not just a label
      overlayCtx.beginPath();
      overlayCtx.arc(nose[0], nose[1], 22, 0, Math.PI * 2);
      overlayCtx.strokeStyle = "#f0b429";
      overlayCtx.lineWidth = 4;
      overlayCtx.stroke();
      overlayCtx.fillStyle = "#f0b429";
      overlayCtx.font = "28px monospace";
      overlayCtx.fillText(`#${skel.skeleton_id}`, nose[0] + 28, nose[1] - 16);
    }
  }

  // prune anything not actually live this frame — otherwise every skeleton
  // that ever briefly flickered into view stays in the sidebar forever
  const liveSkeletonIds = new Set((meta.skeletons || []).map((s) => s.skeleton_id));
  for (const id of Array.from(people.keys())) {
    if (!liveSkeletonIds.has(id)) people.delete(id);
  }
  const liveHandKeys = new Set(
    (meta.hands || []).filter((h) => h.skeleton_id != null && h.side).map((h) => `${h.skeleton_id}:${h.side}`)
  );
  for (const [key, el] of Array.from(cursorEls)) {
    if (!liveHandKeys.has(key)) {
      el.remove();
      cursorEls.delete(key);
    }
  }
  for (const [key, hovered] of Array.from(hoveredEls)) {
    if (!liveHandKeys.has(key)) {
      hovered?.classList.remove("nkit-hover");
      hoveredEls.delete(key);
    }
  }

  for (const hand of meta.hands || []) {
    const [x, y] = hand.wrist;

    const nose = hand.skeleton_id != null ? noseBySkeleton.get(hand.skeleton_id) : null;
    if (nose) {
      overlayCtx.beginPath();
      overlayCtx.moveTo(nose[0], nose[1]);
      overlayCtx.lineTo(x, y);
      overlayCtx.strokeStyle = "rgba(240, 180, 41, 0.5)";
      overlayCtx.lineWidth = 2;
      overlayCtx.stroke();
    }

    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 18, 0, Math.PI * 2);
    overlayCtx.strokeStyle = hand.is_fist ? "#f0b429" : "#4a9eff";
    overlayCtx.lineWidth = 4;
    overlayCtx.stroke();

    if (hand.skeleton_id != null && hand.side) {
      const p = people.get(hand.skeleton_id) || {};
      p[hand.side === "left" ? "leftFist" : "rightFist"] = hand.is_fist;
      people.set(hand.skeleton_id, p);
    }
  }
  renderPeople();   // cheap — a handful of DOM cards, fine to redo every frame
}

// ── people panel (face ID + per-hand state) ─────────────────────────────────────

function handleFace(msg) {
  if (msg.skeleton_id == null) return;   // face not matched to a tracked skeleton this frame — nothing to pin a card to
  const p = people.get(msg.skeleton_id) || {};
  p.name = msg.name;
  p.known = msg.known;
  p.confidence = msg.confidence;
  people.set(msg.skeleton_id, p);
  renderPeople();
}

function renderPeople() {
  if (people.size === 0) {
    peopleListEl.className = "empty";
    peopleListEl.textContent = "no skeletons tracked";
    return;
  }
  peopleListEl.className = "";
  peopleListEl.innerHTML = "";
  for (const [id, p] of people) {
    const card = document.createElement("div");
    card.className = "person-card";
    const nameClass = p.known ? "known" : "unknown";
    const nameText = p.name ? `${p.name} (${(p.confidence ?? 0).toFixed(2)})` : "unrecognized";
    card.innerHTML = `
      <div>#${id} — <span class="name ${nameClass}">${nameText}</span></div>
      <div class="hand-row">
        <span class="${p.leftFist ? "fist" : "palm"}">L: ${p.leftFist === undefined ? "—" : (p.leftFist ? "fist" : "palm")}</span>
        <span class="${p.rightFist ? "fist" : "palm"}">R: ${p.rightFist === undefined ? "—" : (p.rightFist ? "fist" : "palm")}</span>
      </div>`;
    peopleListEl.appendChild(card);
  }
}

// ── gesture handling: cursor dots, synthetic pointer events, sound, log ─────────

const cursorEls = new Map();   // "skeletonId:side" -> div
const soundCache = {};

function playSound(kind) {
  if (!soundCache[kind]) soundCache[kind] = new Audio(`sounds/${kind}.wav`);
  const a = soundCache[kind];
  a.currentTime = 0;
  a.play().catch(() => {});   // no-op if you haven't dropped a wav in sounds/ yet
}

function logGesture(ev) {
  const entry = document.createElement("div");
  entry.className = `entry ${ev.kind}`;
  const extra = ev.kind === "swipe_edge" ? ev.edge : (ev.kind === "grab_swipe" ? ev.direction : "");
  entry.innerHTML = `<span class="kind">${ev.kind}</span> #${ev.skeleton_id}.${ev.side} ${extra} (${ev.x},${ev.y})`;
  gestureLogEl.prepend(entry);
  while (gestureLogEl.children.length > 80) gestureLogEl.removeChild(gestureLogEl.lastChild);
}

function updateCursorDot(ev, grabbing) {
  const key = `${ev.skeleton_id}:${ev.side}`;
  let el = cursorEls.get(key);
  if (!el) {
    el = document.createElement("div");
    el.className = `cursor-dot ${ev.side}`;
    cursorsEl.appendChild(el);
    cursorEls.set(key, el);
  }
  if (ev.x != null && ev.y != null) {
    el.style.left = `${ev.x}px`;
    el.style.top = `${ev.y}px`;
  }
  if (grabbing !== undefined) el.classList.toggle("grabbing", grabbing);
  return el;
}

// push arming: drive the ring straight off push_progress. The dot may not
// exist yet if progress arrives before the first cursor_move for this hand,
// so reuse updateCursorDot to create it.
function setPushProgress(ev) {
  const el = updateCursorDot(ev);
  el.style.setProperty("--p", ev.progress ?? 0);
  el.classList.add("arming");
  el.classList.remove("fired");
}

function clearPushProgress(ev, fired) {
  const el = cursorEls.get(`${ev.skeleton_id}:${ev.side}`);
  if (!el) return;
  if (fired) {
    el.classList.add("fired");
    setTimeout(() => el.classList.remove("fired", "arming"), 160);
  } else {
    el.classList.remove("arming");
  }
  el.style.setProperty("--p", 0);
}

function dispatchPointer(type, x, y) {
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  el.dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y }));
  return el;
}

// Browsers drive :hover from real OS pointer state, not from dispatched
// events — a synthetic pointermove fires JS listeners bound to it, but
// does NOT change what the browser considers "under the mouse" for CSS
// :hover matching. No amount of coordinate correctness fixes that; hover
// has to be faked by hand here, same as any other virtual-pointer system
// (VR/AR pointer UIs hit this exact wall).
const hoveredEls = new Map();   // "skeletonId:side" -> currently-hovered element

function updateHover(ev) {
  const key = `${ev.skeleton_id}:${ev.side}`;
  const el = document.elementFromPoint(ev.x, ev.y);
  const prev = hoveredEls.get(key);
  if (prev === el) return;
  if (prev) prev.classList.remove("nkit-hover");
  if (el) el.classList.add("nkit-hover");
  hoveredEls.set(key, el || null);
}

function flashClick(x, y) {
  clickFlash.style.left = `${x}px`;
  clickFlash.style.top = `${y}px`;
  clickFlash.classList.remove("flash");
  void clickFlash.offsetWidth;   // restart the animation
  clickFlash.classList.add("flash");
}

function handleGesture(ev) {
  logGesture(ev);

  switch (ev.kind) {
    case "cursor_move":
      updateCursorDot(ev);
      dispatchPointer("pointermove", ev.x, ev.y);
      updateHover(ev);
      break;

    case "grab_start":
      updateCursorDot(ev, true);
      playSound("grab_start");
      break;

    case "grab_end":
      updateCursorDot(ev, false);
      playSound("grab_end");
      break;

    case "push_progress":
      setPushProgress(ev);
      break;

    case "push_cancel":
      clearPushProgress(ev, false);
      break;

    case "push": {
      clearPushProgress(ev, true);
      const el = dispatchPointer("pointerdown", ev.x, ev.y);
      dispatchPointer("pointerup", ev.x, ev.y);
      if (el) {
        el.click();
        if (el === testBtn) {
          testBtn.classList.add("pressed");
          setTimeout(() => testBtn.classList.remove("pressed"), 150);
        }
      }
      flashClick(ev.x, ev.y);
      playSound(el === testBtn ? "click" : "click_empty");
      break;
    }

    case "swipe_edge":
      playSound("swipe_edge");
      break;

    case "grab_swipe":
      playSound("grab_swipe");
      break;
  }
}

function handleVoice(msg) {
  voiceStatusEl.className = "";
  voiceStatusEl.innerHTML = `
    <div><span class="mode">${msg.mode}</span> — "${msg.wake_word}" (${(msg.confidence ?? 0).toFixed(2)})</div>
    <div style="margin-top:4px;color:var(--text)">${msg.text || "(empty)"}</div>
    <div style="margin-top:2px;color:var(--muted)">${msg.language || ""}</div>`;
}

// ── gesture threshold sliders (GestureConfig, mirrors nkit/types.py) ────────────

const SLIDERS = [
  { key: "fist_curl_threshold",        label: "Fist curl threshold",       min: 0,   max: 1,    step: 0.01, def: 0.55 },
  { key: "fist_min_curled",            label: "Fist min fingers curled",   min: 1,   max: 4,    step: 1,    def: 3 },
  { key: "ir_fallback_brightness",     label: "IR fallback brightness",    min: 0,   max: 255,  step: 1,    def: 60 },
  { key: "cursor_margin_frac",         label: "Cursor margin (frac)",      min: 0,   max: 0.4,  step: 0.01, def: 0.15 },
  { key: "grab_debounce_ms",           label: "Grab debounce (ms)",        min: 0,   max: 1000, step: 10,   def: 150 },
  { key: "push_travel_mm",             label: "Push travel (mm)",          min: 80,  max: 500,  step: 10,   def: 200 },
  { key: "push_min_velocity",          label: "Push min speed (mm/s)",     min: 100, max: 1500, step: 25,   def: 450 },
  { key: "push_arm_mm",                label: "Push arm at (mm)",          min: 20,  max: 200,  step: 5,    def: 60 },
  { key: "push_window_ms",             label: "Push window (ms)",          min: 50,  max: 2000, step: 25,   def: 600 },
  { key: "push_debounce_ms",           label: "Push debounce (ms)",        min: 50,  max: 1500, step: 10,   def: 400 },
  { key: "swipe_edge_band_px",         label: "Swipe edge band (px)",      min: 20,  max: 400,  step: 5,    def: 120 },
  { key: "swipe_min_distance_px",      label: "Swipe min distance (px)",   min: 20,  max: 600,  step: 5,    def: 150 },
  { key: "swipe_max_duration_ms",      label: "Swipe max duration (ms)",   min: 100, max: 2000, step: 10,   def: 600 },
  { key: "grab_swipe_min_distance_px", label: "Grab-swipe min distance",   min: 20,  max: 800,  step: 5,    def: 200 },
];

function buildSliders() {
  for (const spec of SLIDERS) {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <label>${spec.label} <span class="val">${spec.def}</span></label>
      <input type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${spec.def}">`;
    const input = row.querySelector("input");
    const val = row.querySelector(".val");
    input.addEventListener("input", () => {
      val.textContent = input.value;
      sendConfig({ [spec.key]: Number(input.value) });
    });
    slidersEl.appendChild(row);
  }
}

// ── boot ─────────────────────────────────────────────────────────────────────

buildSliders();
connect();
