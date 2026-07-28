"""
nkit/identity.py — unified person identity across pose/hands/face

A full mediapipe Pose lock isn't the only signal that someone is present.
Hands (mediapipe Hands) and a face (insightface) are independent detectors
that can lock on even when the full-body pose can't — someone leaning out
of frame, an awkward angle, partial occlusion, or just standing close
enough that only their upper body is visible. The old design treated
"skeleton_id" as purely a pose concept: the instant Pose failed to find a
person, everything downstream (hand association, gesture state, face
identity) died with it, even if a hand or face was still tracking fine.

IdentityTracker treats a tracked person as FOUR independent elements —
pose, left hand, right hand, face — each with its own last-known position
and last-seen time. A track only gets pruned once ALL FOUR have gone
unmatched past the grace window, not the instant any single one drops.

New-track spawning is throttled (GestureConfig.new_track_interval_s) and
match radii ("buffers", GestureConfig.limb_buffer_px / face_buffer_px) are
live-tunable — together these stop a spurious one- or two-frame detection
(mediapipe misreading a foot as a hand, a jitter blip) from spawning a
duplicate "ghost" track — and, worse, triggering an expensive face
recognition attempt for it — every time it flickers. A detection that
misses the throttle window is just dropped for that frame (its
skeleton_id/side come back None) rather than forced into existence.

usage (this is what nkit/bridge.py does; a standalone consumer of body.py/
hands.py/face/recognize.py directly can use this the same way):

    identity = IdentityTracker()
    identity.update(bodies, hands, [], now, config)     # pose + hands this frame
    identity.update([], [], faces, now, config)         # faces, only when recognized this frame
    identity.active_ids                                  # currently-alive skeleton_ids
    identity.recognition_eligible(tid, now, config)      # gate for bridge.py's face recognition
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .types import BodyResult, HandResult, FaceResult, GestureConfig

GRACE_SECONDS = 1.0

# a detection that missed its normal same-element match (see
# GestureConfig.limb_buffer_px / face_buffer_px) but still lands this close
# to an EXISTING track's last-known position — on ANY of its 4 elements —
# is almost certainly the same person seen slightly off (a bit of drift, a
# brief detection gap, mediapipe jitter), not someone new. Attach it to
# that track instead of spawning a duplicate "ghost" id right next to a
# real one. Deliberately generous and deliberately not element-specific: a
# false merge (two people standing close together get treated as one) is a
# smaller cost for this use case than the ghost-duplicate churn it fixes.
NEW_TRACK_SUPPRESS_DIST_PX = 350.0

_ELEMENTS = ("pose", "left_hand", "right_hand", "face")


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# how close (px) two frames' detections have to land to count as "the same
# candidate, still there" while it's being confirmed — tight on purpose,
# tighter than the track match buffers, since a one-off false positive
# rarely reappears at a precise, consistent spot multiple frames running
NEW_TRACK_CONFIRM_MATCH_PX = 80.0


@dataclass
class _PendingCandidate:
    pos: tuple[float, float]
    count: int
    last_seen: float


@dataclass
class _Track:
    positions:    dict = field(default_factory=lambda: {k: None for k in _ELEMENTS})
    last_seen:    dict = field(default_factory=lambda: {k: -1e9 for k in _ELEMENTS})
    first_seen:   float = -1.0
    head_visible: bool = True   # optimistic default — only real pose data (nose depth) can set this False
    # last real HandResult per hand slot — kept so fill_held_hands() can
    # reuse it (real is_fist/landmarks/etc, not just a position) when a
    # frame's detection momentarily misses an already-tracked hand
    last_hand_result: dict = field(default_factory=lambda: {"left_hand": None, "right_hand": None})


class IdentityTracker:
    def __init__(self, grace_seconds: float = GRACE_SECONDS):
        self._next_id = 0
        self._tracks: dict[int, _Track] = {}
        self._grace = grace_seconds
        self._last_spawn_ts = -1e9
        # unconfirmed candidates per element ("hand" is shared between left
        # and right — side isn't known until a track actually exists) —
        # see _spawn_or_none()
        self._pending: dict[str, list[_PendingCandidate]] = {"pose": [], "hand": [], "face": []}

    @property
    def active_ids(self) -> set[int]:
        return set(self._tracks.keys())

    def recognition_eligible(self, tid: int, timestamp: float, config: GestureConfig) -> bool:
        """"all 3 limbs visible for a second": old enough, head visible, AND both hands seen recently (not just once, ever)."""
        track = self._tracks.get(tid)
        if track is None:
            return False
        if not track.head_visible:
            return False
        if (timestamp - track.first_seen) < config.recognition_min_track_age_s:
            return False
        hold_s = config.hand_hold_ms / 1000.0
        for slot in ("left_hand", "right_hand"):
            if timestamp - track.last_seen[slot] > hold_s:
                return False
        return True

    def fill_held_hands(self, hands: list[HandResult], timestamp: float, config: GestureConfig) -> list[HandResult]:
        """
        Bridges brief per-frame hand-detection dropouts — mediapipe
        genuinely, briefly missing an already-tracked hand happens often
        enough to feel like flicker — by reusing each track's last real
        HandResult for up to hand_hold_ms if this frame's detection didn't
        find it. Call AFTER update()'s hand matching, so `hands` already
        carries this frame's real skeleton_id/side assignments; the
        returned list is hands plus any held stand-ins (already carrying
        valid skeleton_id/side from whenever they were last really seen).
        Deliberately NOT extrapolated — real hand motion over a couple
        hundred ms is small enough that a stale-but-recent position is a
        fine stand-in, and it keeps this simple.
        """
        present = {(h.skeleton_id, h.side) for h in hands if h.skeleton_id is not None}
        held = []
        hold_s = config.hand_hold_ms / 1000.0
        for tid, track in self._tracks.items():
            for slot in ("left_hand", "right_hand"):
                side = "left" if slot == "left_hand" else "right"
                if (tid, side) in present:
                    continue
                last_result = track.last_hand_result[slot]
                if last_result is None:
                    continue
                if timestamp - track.last_seen[slot] > hold_s:
                    continue
                held.append(last_result)
        return hands + held

    def _find_nearby_track(self, pos: tuple[float, float], exclude: set[int]) -> int | None:
        """Closest track (any element, any track not in `exclude`) within NEW_TRACK_SUPPRESS_DIST_PX, or None."""
        best_tid, best_d = None, NEW_TRACK_SUPPRESS_DIST_PX
        for tid, track in self._tracks.items():
            if tid in exclude:
                continue
            for element in _ELEMENTS:
                p = track.positions[element]
                if p is None:
                    continue
                d = _dist(pos, p)
                if d < best_d:
                    best_d = d
                    best_tid = tid
        return best_tid

    def _spawn_or_none(
        self, element: str, pos: tuple[float, float], exclude: set[int], timestamp: float, config: GestureConfig,
    ) -> int | None:
        """
        Attach to a nearby track if one exists (always allowed, no throttle
        or confirmation — this is continuity, not a genuinely new identity).

        Otherwise this is a candidate for a brand-new identity, which now
        requires TWO things before it's trusted, not just one:
          - temporal confirmation: the same approximate spot has to show up
            across config.new_track_confirm_frames separate update() calls
            within config.new_track_confirm_window_s — a one-off false
            detection (mediapipe misreading a foot as a hand for a frame or
            two) essentially never reappears at a precise, consistent
            position multiple frames running the way a real hand does, so
            this is what actually stops a misfire from ever spawning a
            track, not just rate-limiting how often it can
          - the existing rate throttle (new_track_interval_s), checked only
            once a candidate is otherwise confirmed

        Returns the new/attached track id, or None if the detection should
        be dropped for this frame (still unconfirmed, or confirmed but the
        throttle hasn't opened up yet).
        """
        tid = self._find_nearby_track(pos, exclude)
        if tid is not None:
            return tid

        pending = self._pending[element]
        pending[:] = [p for p in pending if timestamp - p.last_seen <= config.new_track_confirm_window_s]

        match = None
        for p in pending:
            if _dist(pos, p.pos) <= NEW_TRACK_CONFIRM_MATCH_PX:
                match = p
                break

        if match is None:
            pending.append(_PendingCandidate(pos=pos, count=1, last_seen=timestamp))
            return None

        match.pos = pos
        match.count += 1
        match.last_seen = timestamp
        if match.count < config.new_track_confirm_frames:
            return None

        if timestamp - self._last_spawn_ts < config.new_track_interval_s:
            return None

        pending.remove(match)
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = _Track(first_seen=timestamp)
        self._last_spawn_ts = timestamp
        return tid

    def update(
        self,
        bodies: list[BodyResult],
        hands: list[HandResult],
        faces: list[FaceResult],
        timestamp: float,
        config: GestureConfig,
    ) -> None:
        """
        Mutates bodies/hands/faces in place, assigning .skeleton_id (and
        .side for hands) from persistent identity — None if a detection was
        dropped (missed every track's buffer, and the new-track throttle
        hadn't opened up yet). Any of the three lists can be empty on a
        given call (e.g. faces is only populated on frames where
        recognition actually ran) — pruning still runs every call
        regardless of what's passed.
        """
        self._match_pose(bodies, timestamp, config)
        self._match_hands(hands, timestamp, config)
        self._match_faces(faces, timestamp, config)
        self._prune(timestamp)

    # ── pose ─────────────────────────────────────────────────────────────────

    def _match_pose(self, bodies: list[BodyResult], timestamp: float, config: GestureConfig) -> None:
        points = [(b.nose.x, b.nose.y) for b in bodies]
        ids = self._match_category("pose", points, timestamp, config)
        for body, tid in zip(bodies, ids):
            body.skeleton_id = tid
            if tid is None:
                continue
            track = self._tracks[tid]
            track.head_visible = body.nose.z > 0
            # a live pose refreshes the hand anchors from its own wrist
            # estimate, so losing the standalone hand detector for a frame
            # or two still has somewhere sane to match against next time
            track.positions["left_hand"] = (body.left_wrist.x, body.left_wrist.y)
            track.positions["right_hand"] = (body.right_wrist.x, body.right_wrist.y)

    # ── hands ────────────────────────────────────────────────────────────────

    def _match_hands(self, hands: list[HandResult], timestamp: float, config: GestureConfig) -> None:
        candidates = []
        track_ids = list(self._tracks.keys())
        for hi, hand in enumerate(hands):
            pos = (hand.wrist.x, hand.wrist.y)
            for tid in track_ids:
                for slot in ("left_hand", "right_hand"):
                    last = self._tracks[tid].positions[slot]
                    if last is None:
                        continue
                    d = _dist(pos, last)
                    if d <= config.limb_buffer_px:
                        candidates.append((d, hi, tid, slot))
        candidates.sort(key=lambda c: c[0])

        used_slots: set[tuple[int, str]] = set()
        matched_hands: set[int] = set()
        for d, hi, tid, slot in candidates:
            key = (tid, slot)
            if hi in matched_hands or key in used_slots:
                continue
            used_slots.add(key)
            matched_hands.add(hi)
            side = "left" if slot == "left_hand" else "right"
            hands[hi].skeleton_id = tid
            hands[hi].side = side
            self._tracks[tid].positions[slot] = (hands[hi].wrist.x, hands[hi].wrist.y)
            self._tracks[tid].last_seen[slot] = timestamp
            self._tracks[tid].last_hand_result[slot] = hands[hi]

        # unmatched hands: no track has a nearby left/right anchor within
        # the buffer. Attach to a near-enough existing track if one exists
        # (any element), else spawn a new one IF the throttle allows —
        # otherwise drop the detection (skeleton_id/side stay None). This
        # is the fix for a false hand-on-foot detection spawning a ghost
        # track (and an expensive recognition attempt for it) every frame.
        claimed_tids = {tid for tid, _slot in used_slots}
        for hi, hand in enumerate(hands):
            if hi in matched_hands:
                continue
            pos = (hand.wrist.x, hand.wrist.y)
            tid = self._spawn_or_none("hand", pos, claimed_tids, timestamp, config)
            hand.skeleton_id = tid
            hand.side = None
            if tid is None:
                continue
            claimed_tids.add(tid)
            # mediapipe's own Left/Right guess picks the initial slot on a
            # genuine new track since there's no track-relative history yet
            side = "left" if hand.hand == "Left" else "right"
            slot = f"{side}_hand"
            hand.side = side
            self._tracks[tid].positions[slot] = (hand.wrist.x, hand.wrist.y)
            self._tracks[tid].last_seen[slot] = timestamp
            self._tracks[tid].last_hand_result[slot] = hand

    # ── face ─────────────────────────────────────────────────────────────────

    def _match_faces(self, faces: list[FaceResult], timestamp: float, config: GestureConfig) -> None:
        """
        Match against each track's own face history when it has one (frame-
        to-frame continuity), but fall back to the track's POSE position
        when it doesn't — which is every track's first-ever face match,
        since a track spawned from pose/hands has no face history yet. A
        face-only anchor would never find that track (no prior face
        position to compare against) and would spawn a brand-new track
        every time instead, which — since bridge.py only calls this when a
        track is missing a cached face result — means the cache could NEVER
        actually populate for the real track and recognition would run
        every single frame forever instead of once. Same match distance as
        the pose fallback either way: this mirrors the original (pre-
        identity-tracker) associate_faces_to_bodies, which matched a face
        bbox center directly against the body's nose position.
        """
        candidates = []
        track_ids = list(self._tracks.keys())
        for fi, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            pos = ((x1 + x2) / 2, (y1 + y2) / 2)
            for tid in track_ids:
                anchor = self._tracks[tid].positions["face"] or self._tracks[tid].positions["pose"]
                if anchor is None:
                    continue
                d = _dist(pos, anchor)
                if d <= config.face_buffer_px:
                    candidates.append((d, fi, tid))
        candidates.sort(key=lambda c: c[0])

        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        for d, fi, tid in candidates:
            if fi in assigned or tid in used_tracks:
                continue
            assigned[fi] = tid
            used_tracks.add(tid)

        for fi, face in enumerate(faces):
            tid = assigned.get(fi)
            if tid is None:
                x1, y1, x2, y2 = face.bbox
                pos = ((x1 + x2) / 2, (y1 + y2) / 2)
                tid = self._spawn_or_none("face", pos, used_tracks, timestamp, config)
                if tid is None:
                    face.skeleton_id = None
                    continue
                used_tracks.add(tid)
            x1, y1, x2, y2 = face.bbox
            self._tracks[tid].positions["face"] = ((x1 + x2) / 2, (y1 + y2) / 2)
            self._tracks[tid].last_seen["face"] = timestamp
            face.skeleton_id = tid

    # ── shared single-anchor matching (pose) ────────────────────────────────────

    def _match_category(self, element: str, points: list[tuple[float, float]], timestamp: float, config: GestureConfig) -> list[int | None]:
        """Greedy nearest-neighbour match of this frame's points (one element) against
        tracks' last-known position for that element; unmatched points spawn new tracks
        (or are dropped — None — if the new-track throttle hasn't opened up)."""
        max_dist = config.limb_buffer_px
        candidates = []
        track_ids = list(self._tracks.keys())
        for pi, pos in enumerate(points):
            for tid in track_ids:
                last = self._tracks[tid].positions[element]
                if last is None:
                    continue
                d = _dist(pos, last)
                if d <= max_dist:
                    candidates.append((d, pi, tid))
        candidates.sort(key=lambda c: c[0])

        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        for d, pi, tid in candidates:
            if pi in assigned or tid in used_tracks:
                continue
            assigned[pi] = tid
            used_tracks.add(tid)

        result: list[int | None] = []
        for pi, pos in enumerate(points):
            tid = assigned.get(pi)
            if tid is None:
                tid = self._spawn_or_none(element, pos, used_tracks, timestamp, config)
                if tid is not None:
                    used_tracks.add(tid)
            if tid is not None:
                self._tracks[tid].positions[element] = pos
                self._tracks[tid].last_seen[element] = timestamp
            result.append(tid)
        return result

    # ── pruning ──────────────────────────────────────────────────────────────

    def _prune(self, timestamp: float) -> None:
        stale = [
            tid for tid, track in self._tracks.items()
            if all(timestamp - track.last_seen[k] > self._grace for k in _ELEMENTS)
        ]
        for tid in stale:
            del self._tracks[tid]
