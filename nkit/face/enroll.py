"""
nkit/face/enroll.py — programmatic face enrollment

No Flask, no UI — the caller drives the capture loop (from nkit/bridge.py,
a script, a test, whatever) and this handles cell assignment + saving.
Captures happen automatically as the subject moves through a position grid
(X = left/right, Z = near/far, from their tracked skeleton's nose), no
manual "capture" trigger needed — same auto-capture-on-position model as
the grid enrollment in the original prototype, just without the Flask app
wrapped around it.

usage:
    from nkit.face.enroll import Enroller

    with Enroller("Alice", enroll_dir="enroll") as e:
        for rgb, depth, ir in frames:
            faces  = recognizer.process(rgb, depth, ir, config, bodies)
            bodies = body_tracker.process(rgb, depth, ir, config)
            result = e.try_capture(faces, bodies, rgb)
            if result.captured:
                print(f"captured cell {result.cell}, {result.n_done}/{result.n_needed}")
            if e.is_complete:
                break
"""

from __future__ import annotations
import os
import glob

import numpy as np

from ..types import FaceResult, BodyResult, CaptureResult
from .._vision import RGB_W

# 5x5 grid: X = left->right across the frame, Z = near->far from the sensor.
# proven defaults carried over from the original prototype's enroll.py —
# tune Z_THRESHOLDS (mm) to your physical space if capture cells feel off.
DEFAULT_GRID_COLS = 5
DEFAULT_GRID_ROWS = 5
DEFAULT_X_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_Z_THRESHOLDS = [0, 1200, 1800, 2600, 3400, 9999]

CAPTURE_COOLDOWN_S = 0.4   # min gap between two captures in the same cell — mild pose diversity


def _save_capture(out_dir: str, name: str, cell: tuple[int, int], face_img: np.ndarray,
                   embedding: np.ndarray, index: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    col, row = cell
    fname = f"{name}_c{col}r{row}_{index:02d}.npz"
    np.savez(os.path.join(out_dir, fname),
              face_img=face_img,
              embedding=embedding,
              cell=np.array(cell),
              name=np.array(name))


class Enroller:
    def __init__(
        self,
        name:         str,
        enroll_dir:   str = "enroll",
        n_per_cell:   int = 3,
        grid_cols:    int = DEFAULT_GRID_COLS,
        grid_rows:    int = DEFAULT_GRID_ROWS,
        x_fractions:  list[float] | None = None,
        z_thresholds: list[float] | None = None,
        capture_cooldown_s: float = CAPTURE_COOLDOWN_S,
    ):
        self.name = name
        self.enroll_dir = enroll_dir
        self.n_per_cell = n_per_cell
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.x_fractions = x_fractions or DEFAULT_X_FRACTIONS
        self.z_thresholds = z_thresholds or DEFAULT_Z_THRESHOLDS
        self.capture_cooldown_s = capture_cooldown_s

        if len(self.x_fractions) != grid_cols + 1:
            raise ValueError("x_fractions must have grid_cols+1 entries")
        if len(self.z_thresholds) != grid_rows + 1:
            raise ValueError("z_thresholds must have grid_rows+1 entries")

        self._out_dir = os.path.join(enroll_dir, name)
        self._done_cells: dict[tuple[int, int], int] = {}
        self._current_cell: tuple[int, int] | None = None
        self._last_capture_ts: dict[tuple[int, int], float] = {}
        self._scan_existing()

    # ── state ────────────────────────────────────────────────────────────────

    def _scan_existing(self) -> None:
        """resume progress if this name already has captures on disk"""
        os.makedirs(self._out_dir, exist_ok=True)
        for path in glob.glob(os.path.join(self._out_dir, "*.npz")):
            data = np.load(path)
            cell = tuple(int(v) for v in data["cell"].tolist())
            self._done_cells[cell] = self._done_cells.get(cell, 0) + 1

    @property
    def current_cell(self) -> tuple[int, int] | None:
        return self._current_cell

    @property
    def done_cells(self) -> dict[tuple[int, int], int]:
        return dict(self._done_cells)

    @property
    def is_complete(self) -> bool:
        return all(
            self._done_cells.get((c, r), 0) >= self.n_per_cell
            for c in range(self.grid_cols) for r in range(self.grid_rows)
        )

    def reload(self) -> None:
        """force re-scan of enroll_dir (after external enrollment changes)"""
        self._done_cells.clear()
        self._scan_existing()

    # ── grid mapping ─────────────────────────────────────────────────────────

    def _x_to_col(self, x_px: int) -> int:
        frac = x_px / RGB_W
        for i in range(self.grid_cols):
            if self.x_fractions[i] <= frac < self.x_fractions[i + 1]:
                return i
        return self.grid_cols - 1

    def _z_to_row(self, z_mm: float) -> int | None:
        if z_mm <= 0:
            return None
        for i in range(self.grid_rows):
            if self.z_thresholds[i] <= z_mm < self.z_thresholds[i + 1]:
                return i
        return self.grid_rows - 1

    # ── capture ──────────────────────────────────────────────────────────────

    def try_capture(
        self,
        faces:  list[FaceResult],
        bodies: list[BodyResult],
        rgb_frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> CaptureResult:
        """
        Call every frame. Captures automatically once a face's skeleton
        lands in a not-yet-full grid cell — no external trigger needed.
        timestamp: seconds (e.g. time.monotonic()), used for capture_cooldown_s.
        """
        bodies_by_id = {b.skeleton_id: b for b in bodies}

        subject: FaceResult | None = None
        subject_body: BodyResult | None = None
        best_area = -1
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                subject = face
                subject_body = bodies_by_id.get(face.skeleton_id)

        self._current_cell = None
        if subject is not None and subject_body is not None:
            col = self._x_to_col(subject_body.nose.x)
            row = self._z_to_row(subject_body.nose.z)
            if row is not None:
                self._current_cell = (col, row)

        n_needed = self.grid_cols * self.grid_rows * self.n_per_cell
        n_done   = sum(self._done_cells.values())

        captured = False
        if self._current_cell is not None and subject is not None:
            n_here = self._done_cells.get(self._current_cell, 0)
            last_ts = self._last_capture_ts.get(self._current_cell, -1e9)
            if n_here < self.n_per_cell and (timestamp - last_ts) >= self.capture_cooldown_s:
                x1, y1, x2, y2 = subject.bbox
                bgr = rgb_frame[:, :, :3]
                face_crop = bgr[max(0, y1):y2, max(0, x1):x2]
                _save_capture(self._out_dir, self.name, self._current_cell,
                              face_crop, subject.embedding, n_here)
                self._done_cells[self._current_cell] = n_here + 1
                self._last_capture_ts[self._current_cell] = timestamp
                captured = True
                n_done += 1

        return CaptureResult(
            captured = captured,
            complete = self.is_complete,
            cell     = self._current_cell,
            n_done   = n_done,
            n_needed = n_needed,
            face     = subject if captured else None,
        )

    def __enter__(self) -> "Enroller":
        return self

    def __exit__(self, *_):
        pass
