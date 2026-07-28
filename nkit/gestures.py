"""
nkit/gestures.py — gesture vocabulary state machine

Consumes the per-frame HandResult stream (already tagged with skeleton_id /
side by body.py:associate_hands_to_bodies) and emits discrete GestureEvents
per hand-identity: a continuous cursor position plus five discrete
primitives — grab, push (click), swipe from an edge, and grab+swipe.

Deliberately does NOT do UI hit-testing: it has no idea what's on screen.
"Hover on an element" / "click on an element" / "click on empty space" are
what you get for free once cursor_move/push are fed into a real UI
toolkit's native pointer events (see nkit/bridge.py and the demo app) — the
toolkit's own hit-testing answers those, nkit just supplies the pointer.

Also deliberately does NOT play sounds — nkit stays pure data/no side
effects (see the SDK design doc). Map GestureEvent.kind -> a sound file and
play it in the consuming app.

usage:
    from nkit.gestures import GestureTracker
    from nkit.types import GestureConfig

    tracker = GestureTracker()
    config = GestureConfig()   # mutate live from a UI to retune feel
    events = tracker.process(hands, timestamp, config, screen_w=1920, screen_h=1080)
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .types import BodyResult, HandResult, GestureEvent, GestureConfig, Vec3
from ._vision import RGB_W, RGB_H, pixel_delta_to_mm
from ._filter import PointFilter


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _displacement_mm(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    True 3D displacement in real-world mm from wrist position a to b
    (x_px, y_px, z_mm each). Converts the pixel (x, y) part to mm via the
    camera's FOV at the current depth — a straight pixel distance would mix
    units with the z (mm) part meaninglessly.
    """
    dx_mm, dy_mm = pixel_delta_to_mm(b[0] - a[0], b[1] - a[1], b[2])
    dz_mm = b[2] - a[2]
    return (dx_mm, dy_mm, dz_mm)


# Longer than a frame interval but far shorter than a gesture: bridges the
# 1-in-3 dropouts while the hand is extended, without bridging a real absence.
PUSH_MAX_SAMPLE_GAP_S = 0.20

# how long an armed push survives with no usable hand depth at all
PUSH_HOLD_THROUGH_DROPOUT_S = 0.35


def _forward_velocity(hist: list) -> float:
    """
    Forward speed in mm/s from a short (timestamp, forward_mm) history.

    Median of the frame-to-frame rates rather than the mean, and rates over
    implausible gaps are dropped: a single bad depth sample otherwise produces
    a velocity spike large enough to arm anything. Measured on a real session,
    even after the depth-gate fix, unsmoothed rates reached 9 m/s while
    someone sat still.
    """
    rates = [
        (hist[i][1] - hist[i - 1][1]) / (hist[i][0] - hist[i - 1][0])
        for i in range(1, len(hist))
        if 0 < hist[i][0] - hist[i - 1][0] < 0.4
    ]
    if not rates:
        return 0.0
    rates.sort()
    return rates[len(rates) // 2]


def _magnitude(v: tuple[float, float, float]) -> float:
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass
class _HandState:
    last_seen_ts: float = 0.0

    # One Euro smoothing of this hand's wrist — per hand identity, so two
    # hands never share velocity history
    pos_filter: PointFilter = field(default_factory=PointFilter)
    last_valid_z: float | None = None

    # grab
    grabbing:         bool = False
    fist_since_ts:    float | None = None
    grab_origin_y:    int | None = None
    grab_swipe_fired: bool = False

    # push (palm-mode poke) — positions are (x_px, y_px, z_mm)
    push_phase:            str = "idle"   # "idle" | "extended"
    push_baseline_pos:     tuple[float, float, float] | None = None
    push_origin:           tuple[float, float, float] | None = None   # frozen anchor when extension started
    push_direction:        tuple[float, float, float] | None = None   # unit vector, the push's own axis
    push_phase_start_ts:   float | None = None
    last_push_ts:          float = -1e9

    # push, arm-relative mode — baseline is an EMA of the resting reach ratio
    push_reach_baseline: float | None = None

    # push, travel mode — baseline is an EMA of the hand's resting distance
    # in front of the torso, so slouching or stepping forward drifts out
    push_fwd_baseline: float | None = None
    push_peak_travel:  float = 0.0
    push_armed:        bool = False
    # short history of (timestamp, forward_mm) for the velocity gate — median
    # of the last few frame-to-frame rates, so one bad depth sample can't
    # manufacture a spike
    push_fwd_hist:     list = field(default_factory=list)
    push_fwd_last_ts:  float | None = None   # when we last had usable hand depth
    push_origin_z:     float | None = None   # point A, frozen at motion onset

    # swipe from edge
    swipe_edge_name:   str | None = None
    swipe_origin:      tuple[int, int] | None = None
    swipe_start_ts:    float | None = None
    swipe_cooldown:    bool = False


class GestureTracker:
    def __init__(self):
        self._states: dict[tuple[int, str], _HandState] = {}

    def _map_to_screen(self, wrist_x: float, wrist_y: float, screen_w: int, screen_h: int, config: GestureConfig) -> tuple[int, int]:
        margin_x = RGB_W * config.cursor_margin_frac
        margin_y = RGB_H * config.cursor_margin_frac
        denom_x = RGB_W - 2 * margin_x
        denom_y = RGB_H - 2 * margin_y
        fx = (wrist_x - margin_x) / denom_x if denom_x > 0 else 0.5
        fy = (wrist_y - margin_y) / denom_y if denom_y > 0 else 0.5
        fx = _clip(fx, 0.0, 1.0)
        fy = _clip(fy, 0.0, 1.0)
        return int(fx * screen_w), int(fy * screen_h)

    def _reach(self, arm: tuple[Vec3, Vec3, Vec3],
               config: GestureConfig) -> tuple[float, tuple[float, float, float]] | None:
        """
        (reach_ratio, elbow->wrist unit vector) for this arm, or None if the
        arm isn't measurable this frame.

        reach = |shoulder->wrist| / |shoulder->elbow|. Both legs are real mm
        (via _displacement_mm, so the pixel parts get FOV-converted at the
        right depth), which is what makes the ratio unitless and therefore
        the same number at any distance from the camera.

        All three points come from the POSE model, deliberately. An earlier
        version took shoulder/elbow from pose but the wrist from the hand
        model, which put two networks' independent estimates into one ratio —
        wherever they disagreed (and they do), reach was wrong by that
        disagreement every frame, so pushes fired on noise and real ones
        missed. The arm chain has to be internally consistent; the hand model
        is still what decides is_fist, it just doesn't define where the arm is.
        """
        shoulder, elbow, wrist = arm
        s = (shoulder.x, shoulder.y, shoulder.z)
        e = (elbow.x, elbow.y, elbow.z)
        pos = (wrist.x, wrist.y, wrist.z)
        if shoulder.z <= 0 or elbow.z <= 0 or wrist.z <= 0:
            return None

        upper_arm = _magnitude(_displacement_mm(s, e))
        if upper_arm < config.push_min_arm_mm:
            # arm foreshortened to almost nothing (pointing at the camera) or
            # landmarks collapsed — the ratio's denominator is unusable
            return None

        reach = _magnitude(_displacement_mm(s, pos)) / upper_arm

        forearm = _displacement_mm(e, pos)
        mag = _magnitude(forearm)
        if mag <= 0:
            return None
        axis = (forearm[0] / mag, forearm[1] / mag, forearm[2] / mag)
        return reach, axis

    # torso depth candidates, best first. Shoulders are the most reliable
    # sample on a person (broad and flat, so the depth patch lands entirely on
    # them) but they are NOT always visible — seated at a desk, or turned, or
    # half out of frame, and requiring both of them meant push silently did
    # nothing. Hips and nose are worse references individually but far better
    # than having none, so take the median of whatever is actually there.
    _TORSO_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose")

    def _torso_z(self, body: BodyResult) -> float | None:
        """Depth of the body's core, from whichever landmarks have valid depth."""
        zs = []
        for name in self._TORSO_LANDMARKS:
            v = getattr(body, name, None)
            if v is not None and v.z > 0:
                zs.append(v.z)
        if not zs:
            return None
        zs.sort()
        return zs[len(zs) // 2]      # median: one bad landmark can't drag it

    def _forward_mm(self, body: BodyResult, wrist: Vec3) -> float | None:
        """
        How far in front of the torso the hand is, in mm (+ve = toward camera).

        Body-relative, so it's distance-invariant by construction — it doesn't
        change when you stand further from the camera — and leaning cancels
        out, since torso and hand move together.
        """
        if wrist.z <= 0:
            return None
        torso_z = self._torso_z(body)
        if torso_z is None:
            return None
        return torso_z - wrist.z

    def _update_push_hand(
        self, state: _HandState, z_mm: float, timestamp: float, config: GestureConfig,
    ) -> tuple[bool, float | None, bool]:
        """
        Push from the hand's own trajectory alone. Returns
        (fired, progress_or_None, cancelled).

        Two quantities, both about the hand and nothing else:
          1. speed  — is it moving toward the camera, fast enough to be meant
          2. travel — how far it has come from where the motion STARTED

        No body reference. Depth is already millimetres, so travel is
        distance-invariant without normalising against the torso, and dropping
        the torso removes a whole second measurement that could fail — it was
        requiring shoulders that aren't visible when seated, and every torso
        dropout became a push dropout.

        Origin is captured at motion ONSET and then frozen. The previous
        version measured against a slowly-drifting EMA baseline, which a push
        could outrun: the baseline crept toward the hand mid-gesture and ate
        the very displacement being measured. A frozen point A can't do that,
        and "distance from A to B" is what a push actually is.
        """
        hist = state.push_fwd_hist
        velocity = _forward_velocity(hist)      # +ve = toward the camera

        if not state.push_armed:
            if velocity < config.push_min_velocity:
                return False, None, False
            # freeze point A at the oldest sample still in the window — the
            # motion began there, not at the frame we happened to notice it
            state.push_origin_z = hist[0][1] if hist else z_mm
            state.push_armed = True
            state.push_peak_travel = 0.0
            state.push_phase_start_ts = timestamp

        # z_mm arrives already negated (+ve = toward the camera), so travel is
        # current minus origin, not the other way round
        travel = z_mm - (state.push_origin_z if state.push_origin_z is not None else z_mm)
        state.push_peak_travel = max(state.push_peak_travel, travel)
        progress = _clip(travel / max(1.0, config.push_travel_mm), 0.0, 1.0)

        if (timestamp - (state.push_phase_start_ts or timestamp)) * 1000.0 > config.push_window_ms:
            state.push_armed = False
            return False, None, True

        if travel >= config.push_travel_mm:
            state.push_armed = False
            if (timestamp - state.last_push_ts) * 1000.0 >= config.push_debounce_ms:
                state.last_push_ts = timestamp
                return True, 1.0, False
            return False, None, True

        # gave up on the way out — hand reversed before completing
        if travel < state.push_peak_travel - config.push_arm_mm:
            state.push_armed = False
            return False, None, True

        return False, progress, False

    def _update_push_travel(
        self, state: _HandState, forward_mm: float, timestamp: float, config: GestureConfig,
    ) -> tuple[bool, float | None, bool]:
        """
        Travel-mode push. Returns (fired, progress_or_None, cancelled).

        The gesture is a deliberate forward movement of the hand, measured
        against a slow baseline of where the hand normally sits, and it
        reports progress the whole way so the UI can show it arming.

        This replaced measuring arm EXTENSION, which could not work: measured
        on a real recording the user's push kept the arm folded, so the
        extension ratio during a push was lower than at rest and separated
        from its hardest negative at exactly chance. Hand travel is what
        actually moves during a push.

        Fires the moment travel completes, so the click lands when you push
        rather than when you pull back — an earlier version fired on the
        return, which was correct at rejecting drift but felt wrong to use.
        Drift is rejected by push_window_ms instead: arming starts a clock, so
        a hand that creeps forward slowly (leaning in, reaching for something
        and holding it) times out and cancels before it ever completes.
        """
        # A hand held out toward the camera is a small target whose depth patch
        # is mostly background, so its depth sample drops out — measured, ~29%
        # of frames once the elbow is 400mm+ forward, against 1-4% near the
        # body. Rates computed ACROSS such a gap are fiction: the hand moved
        # while we weren't looking, so the jump on reappearance reads as a huge
        # velocity. That made push fire on the way back (the return is when
        # depth comes good again) while feeling dead on the way out.
        gap = timestamp - (state.push_fwd_last_ts or timestamp)
        if gap > PUSH_MAX_SAMPLE_GAP_S:
            state.push_fwd_hist.clear()
        state.push_fwd_last_ts = timestamp

        state.push_fwd_hist.append((timestamp, forward_mm))
        del state.push_fwd_hist[:-5]
        velocity = _forward_velocity(state.push_fwd_hist)

        if state.push_fwd_baseline is None:
            state.push_fwd_baseline = forward_mm
            return False, None, False

        travel = forward_mm - state.push_fwd_baseline

        if not state.push_armed:
            # baseline only drifts while idle, so a slow push can't outrun it
            state.push_fwd_baseline = state.push_fwd_baseline * 0.93 + forward_mm * 0.07
            if travel < config.push_arm_mm or velocity < config.push_min_velocity:
                return False, None, False
            state.push_armed = True
            state.push_peak_travel = travel
            state.push_phase_start_ts = timestamp

        state.push_peak_travel = max(state.push_peak_travel, travel)
        progress = _clip(state.push_peak_travel / max(1.0, config.push_travel_mm), 0.0, 1.0)

        if (timestamp - (state.push_phase_start_ts or timestamp)) * 1000.0 > config.push_window_ms:
            state.push_armed = False
            state.push_fwd_baseline = forward_mm
            return False, None, True          # took too long — abandon, clear the UI

        if travel >= config.push_travel_mm:
            state.push_armed = False
            state.push_fwd_baseline = forward_mm   # rebase where the hand now is
            if (timestamp - state.last_push_ts) * 1000.0 >= config.push_debounce_ms:
                state.last_push_ts = timestamp
                return True, 1.0, False
            return False, None, True

        if travel < config.push_arm_mm * 0.5:
            state.push_armed = False          # pulled back before completing
            return False, None, True

        return False, progress, False

    def _update_push_reach(
        self, state: _HandState, reach: float, axis: tuple[float, float, float],
        timestamp: float, config: GestureConfig,
    ) -> bool:
        """
        Arm-relative push. Same arm/extend/retract phases as the absolute-mm
        path below, but measuring the reach RATIO rather than a displacement,
        and taking the push axis straight from the forearm instead of
        inferring it from the first few mm of motion — that inference read its
        direction off the noisiest part of the gesture, which is most of why
        pushes triggered erratically.
        """
        if state.push_reach_baseline is None:
            state.push_reach_baseline = reach
            return False

        if state.push_phase == "idle":
            # slow EMA so a resting arm's natural reach becomes the reference
            state.push_reach_baseline = state.push_reach_baseline * 0.95 + reach * 0.05
            if reach - state.push_reach_baseline >= config.push_reach_delta:
                state.push_direction = axis          # forearm axis: the push's trajectory
                state.push_phase = "extended"
                state.push_phase_start_ts = timestamp
            return False

        # extended: waiting for the arm to fold back
        if (timestamp - state.push_phase_start_ts) * 1000.0 > config.push_window_ms:
            state.push_phase = "idle"
            state.push_reach_baseline = reach
            return False

        release_at = state.push_reach_baseline + config.push_reach_delta * config.push_reach_release
        if reach <= release_at:
            state.push_phase = "idle"
            if (timestamp - state.last_push_ts) * 1000.0 >= config.push_debounce_ms:
                state.last_push_ts = timestamp
                return True
        return False

    def _update_push(self, state: _HandState, pos: tuple[float, float, float],
                     timestamp: float, config: GestureConfig) -> bool:
        """
        Absolute-mm push — the fallback for when no arm is visible (person
        side-on, shoulder or elbow dropped this frame). Prefer
        _update_push_reach: this path's threshold is in mm, so it can only be
        correct at one distance from the camera.

        Returns True if a push fired this frame.

        Measures displacement PROJECTED ONTO the push's own axis, not raw
        3D distance to a fixed point — a "cone" from slightly behind the
        hand's start position to in front of it, rather than a razor-thin
        line. The axis is established from whichever direction the hand
        first moved past the threshold, then both the forward extension and
        the retraction are measured along THAT axis. This makes it tolerant
        of the lateral wobble any real hand motion has (a plain "return
        within a few mm of the exact starting point" check is close to
        impossible to hit in practice), while still ignoring a push that's
        mostly a sideways swipe rather than something moving through the
        hand's own established forward/back line — see
        _displacement_mm / _dot for the projection math. Also still uses
        true 3D distance (not sensor-Z alone) for the axis itself, so an
        angled sitting position doesn't underweight the gesture.
        """
        if state.push_baseline_pos is None:
            state.push_baseline_pos = pos
            return False

        if state.push_phase == "idle":
            # slow EMA so the resting baseline drifts with natural hand position
            bx, by, bz = state.push_baseline_pos
            state.push_baseline_pos = (bx * 0.95 + pos[0] * 0.05, by * 0.95 + pos[1] * 0.05, bz * 0.95 + pos[2] * 0.05)

            disp = _displacement_mm(state.push_baseline_pos, pos)
            mag = _magnitude(disp)
            if mag >= config.push_delta_mm:
                # establish the push's own axis from this initial motion —
                # a unit vector; projecting disp onto itself here just gives
                # back mag, so entry/exit thresholds stay on the same scale
                state.push_origin = state.push_baseline_pos
                state.push_direction = (disp[0] / mag, disp[1] / mag, disp[2] / mag)
                state.push_phase = "extended"
                state.push_phase_start_ts = timestamp
            return False

        # phase == "extended": waiting for progress along the established
        # axis to drop back down, confirming a poke (not just a stall)
        elapsed_ms = (timestamp - state.push_phase_start_ts) * 1000.0
        if elapsed_ms > config.push_window_ms:
            # stalled forward without retracting in time — abandon, rebase
            state.push_phase = "idle"
            state.push_baseline_pos = pos
            return False

        depth = _dot(_displacement_mm(state.push_origin, pos), state.push_direction)
        if depth <= config.push_delta_mm * 0.4:
            state.push_phase = "idle"
            state.push_baseline_pos = pos
            if (timestamp - state.last_push_ts) * 1000.0 >= config.push_debounce_ms:
                state.last_push_ts = timestamp
                return True
        return False

    def _update_swipe_edge(
        self, state: _HandState, x: int, y: int, timestamp: float,
        config: GestureConfig, screen_w: int, screen_h: int,
    ) -> str | None:
        band = config.swipe_edge_band_px
        dists = {
            "left":   x,
            "right":  screen_w - x,
            "top":    y,
            "bottom": screen_h - y,
        }
        nearest_edge = min(dists, key=dists.get)
        in_band = dists[nearest_edge] <= band

        if state.swipe_cooldown:
            if not in_band:
                state.swipe_cooldown = False
            return None

        if state.swipe_edge_name is None:
            if in_band:
                state.swipe_edge_name = nearest_edge
                state.swipe_origin = (x, y)
                state.swipe_start_ts = timestamp
            return None

        # currently tracking a candidate swipe from state.swipe_edge_name
        elapsed_ms = (timestamp - state.swipe_start_ts) * 1000.0
        if elapsed_ms > config.swipe_max_duration_ms:
            state.swipe_edge_name = None
            state.swipe_origin = None
            state.swipe_cooldown = True
            return None

        ox, oy = state.swipe_origin
        edge = state.swipe_edge_name
        if edge == "left":
            inward = x - ox
        elif edge == "right":
            inward = ox - x
        elif edge == "top":
            inward = y - oy
        else:  # bottom
            inward = oy - y

        if inward >= config.swipe_min_distance_px:
            fired_edge = edge
            state.swipe_edge_name = None
            state.swipe_origin = None
            state.swipe_cooldown = True
            return fired_edge

        return None

    def process(
        self,
        hands: list[HandResult],
        timestamp: float,
        config: GestureConfig,
        screen_w: int = 1920,
        screen_h: int = 1080,
        stale_after_s: float = 5.0,
        bodies: list[BodyResult] | None = None,
    ) -> list[GestureEvent]:
        """
        timestamp: seconds, monotonic — caller's clock (e.g. time.monotonic()).
        Untracked hands (skeleton_id/side is None, i.e. not yet associated to
        a skeleton by body.associate_hands_to_bodies) are skipped entirely.

        bodies: the same frame's skeletons. Supplying them switches push
        detection to the arm-relative path (see _update_push_reach), which is
        distance-invariant and immune to whole-body lean; without them push
        falls back to the absolute-mm path, which is only correct at roughly
        the distance push_delta_mm was tuned for.
        """
        events: list[GestureEvent] = []
        active_keys: set[tuple[int, str]] = set()
        by_skeleton = {b.skeleton_id: b for b in (bodies or []) if b.skeleton_id is not None}

        for hand in hands:
            if hand.skeleton_id is None or hand.side is None:
                continue
            key = (hand.skeleton_id, hand.side)
            active_keys.add(key)
            state = self._states.setdefault(key, _HandState())
            state.last_seen_ts = timestamp

            # smooth before anything reads the position, so the cursor, the
            # push measurement and the swipe distances all work off the same
            # de-jittered point. z can drop to 0 when depth has no data at the
            # wrist; feeding that through would yank the filter toward zero,
            # so hold the last good value and let push skip the frame instead.
            state.pos_filter.retune(config.smoothing_min_cutoff, config.smoothing_beta)
            z_valid = hand.wrist.z > 0
            z_in = hand.wrist.z if z_valid else (state.last_valid_z or 0.0)
            sx, sy, sz = state.pos_filter((hand.wrist.x, hand.wrist.y, z_in), timestamp)
            if z_valid:
                state.last_valid_z = hand.wrist.z
            pos = (sx, sy, sz)

            x, y = self._map_to_screen(sx, sy, screen_w, screen_h, config)

            events.append(GestureEvent(
                skeleton_id=hand.skeleton_id, side=hand.side, kind="cursor_move",
                x=x, y=y, edge=None, direction=None, timestamp=timestamp,
            ))

            # ── grab ──────────────────────────────────────────────────────
            if hand.is_fist:
                if state.fist_since_ts is None:
                    state.fist_since_ts = timestamp
                elif not state.grabbing and (timestamp - state.fist_since_ts) * 1000.0 >= config.grab_debounce_ms:
                    state.grabbing = True
                    state.grab_origin_y = y
                    state.grab_swipe_fired = False
                    events.append(GestureEvent(
                        skeleton_id=hand.skeleton_id, side=hand.side, kind="grab_start",
                        x=x, y=y, edge=None, direction=None, timestamp=timestamp,
                    ))
                # palm-mode gestures don't apply while fisted
                state.push_phase = "idle"
                state.push_baseline_pos = None
                state.push_reach_baseline = None
                state.push_armed = False
                state.push_fwd_baseline = None
            else:
                if state.grabbing:
                    state.grabbing = False
                    events.append(GestureEvent(
                        skeleton_id=hand.skeleton_id, side=hand.side, kind="grab_end",
                        x=x, y=y, edge=None, direction=None, timestamp=timestamp,
                    ))
                state.fist_since_ts = None

            # ── grab + swipe (Xbox-shell-style, fires once per grab hold) ──
            if state.grabbing and not state.grab_swipe_fired and state.grab_origin_y is not None:
                dy = y - state.grab_origin_y
                if abs(dy) >= config.grab_swipe_min_distance_px:
                    state.grab_swipe_fired = True
                    events.append(GestureEvent(
                        skeleton_id=hand.skeleton_id, side=hand.side, kind="grab_swipe",
                        x=x, y=y, edge=None, direction=("down" if dy > 0 else "up"),
                        timestamp=timestamp,
                    ))

            # ── push (palm-mode click) ───────────────────────────────────
            # arm-relative when this hand's skeleton gave us a shoulder and
            # elbow this frame, absolute mm otherwise
            arm = None
            body = by_skeleton.get(hand.skeleton_id)
            if body is not None:
                arm = ((body.left_shoulder, body.left_elbow, body.left_wrist) if hand.side == "left"
                       else (body.right_shoulder, body.right_elbow, body.right_wrist))

            pushed = False
            if not hand.is_fist and z_valid:
                fwd = None
                if body is not None:
                    # the pose model's own wrist, not the hand model's — see
                    # _reach's note on not mixing the two networks' estimates.
                    # Raw depth, no torso reference: see _update_push_hand.
                    pose_wrist = body.left_wrist if hand.side == "left" else body.right_wrist
                    if pose_wrist.z > 0:
                        fwd = pose_wrist.z
                if fwd is None and state.push_armed and \
                        timestamp - (state.push_fwd_last_ts or 0.0) <= PUSH_HOLD_THROUGH_DROPOUT_S:
                    # mid-push dropout: hold the armed state and keep the ring
                    # up rather than cancelling. Dropouts cluster exactly when
                    # the arm is extended, so cancelling on them aborts the
                    # gesture precisely at its peak.
                    events.append(GestureEvent(
                        skeleton_id=hand.skeleton_id, side=hand.side, kind="push_progress",
                        x=x, y=y, edge=None, direction=None, timestamp=timestamp,
                        progress=_clip(state.push_peak_travel / max(1.0, config.push_travel_mm), 0.0, 1.0),
                    ))
                elif fwd is not None:
                    gap = timestamp - (state.push_fwd_last_ts or timestamp)
                    if gap > PUSH_MAX_SAMPLE_GAP_S:
                        state.push_fwd_hist.clear()
                    state.push_fwd_last_ts = timestamp
                    # store as +ve-toward-camera so velocity reads naturally
                    state.push_fwd_hist.append((timestamp, -fwd))
                    del state.push_fwd_hist[:-5]
                    pushed, progress, cancelled = self._update_push_hand(state, -fwd, timestamp, config)
                    if progress is not None and not pushed:
                        events.append(GestureEvent(
                            skeleton_id=hand.skeleton_id, side=hand.side, kind="push_progress",
                            x=x, y=y, edge=None, direction=None, timestamp=timestamp,
                            progress=progress,
                        ))
                    elif cancelled:
                        events.append(GestureEvent(
                            skeleton_id=hand.skeleton_id, side=hand.side, kind="push_cancel",
                            x=x, y=y, edge=None, direction=None, timestamp=timestamp,
                        ))
                else:
                    # no usable torso/hand depth this frame — fall back
                    pushed = self._update_push(state, pos, timestamp, config)

            if pushed:
                events.append(GestureEvent(
                    skeleton_id=hand.skeleton_id, side=hand.side, kind="push",
                    x=x, y=y, edge=None, direction=None, timestamp=timestamp,
                ))

            # ── swipe from a screen edge ──────────────────────────────────
            fired_edge = self._update_swipe_edge(state, x, y, timestamp, config, screen_w, screen_h)
            if fired_edge:
                events.append(GestureEvent(
                    skeleton_id=hand.skeleton_id, side=hand.side, kind="swipe_edge",
                    x=x, y=y, edge=fired_edge, direction=None, timestamp=timestamp,
                ))

        # drop state for hands not seen in a while (person left, hand lost, etc.)
        stale = [k for k, s in self._states.items()
                 if k not in active_keys and timestamp - s.last_seen_ts > stale_after_s]
        for k in stale:
            del self._states[k]

        return events
