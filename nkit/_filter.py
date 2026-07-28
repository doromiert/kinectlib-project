"""
nkit/_filter.py — One Euro filter for landmark/cursor smoothing

Landmark positions arrive jittery (single-frame model noise plus depth
resampling), and the pipeline runs well under display refresh, so raw values
read as both shaky and choppy. A plain EMA can't fix that: the constant that
holds a resting hand still also drags visibly behind a moving one, and
gestures live at both extremes — a hand hovering to aim, then a fast push.

The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) adapts its cutoff to
speed instead: heavy smoothing while slow, cutoff opening up as the point
accelerates, so jitter dies at rest without adding lag to a deliberate move.
Two knobs with physical meaning — min_cutoff sets stillness-jitter, beta sets
how fast it lets go once moving.

Timestamps are the caller's monotonic seconds; the filter derives its own
sample rate per update, so a variable/dropping frame rate stays correct
without being told what the rate is.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Scalar One Euro filter. One instance per independent signal."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, timestamp: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev, self._t_prev = x, timestamp
            return x

        dt = timestamp - self._t_prev
        # a zero/backwards dt would divide by zero in _alpha; a huge one means
        # the hand was lost and came back, where blending across the gap would
        # drag the value in from wherever it used to be
        if dt <= 0.0:
            return self._x_prev
        if dt > 1.0:
            self.reset()
            self._x_prev, self._t_prev = x, timestamp
            return x

        dx = (x - self._x_prev) / dt
        dx_hat = self._dx_prev + _alpha(self.d_cutoff, dt) * (dx - self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._x_prev + _alpha(cutoff, dt) * (x - self._x_prev)

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, timestamp
        return x_hat


@dataclass
class PointFilter:
    """
    One Euro over a 3D point, one filter per axis.

    x/y are pixels and z is mm, so they carry different noise scales — but
    min_cutoff/beta are in units of the signal's own change per second, so a
    shared setting behaves sensibly on both rather than needing per-axis
    tuning. Split them here if that stops being true.
    """
    min_cutoff: float = 1.0
    beta: float = 0.007
    _fx: OneEuroFilter = field(init=False)
    _fy: OneEuroFilter = field(init=False)
    _fz: OneEuroFilter = field(init=False)

    def __post_init__(self):
        self._fx = OneEuroFilter(self.min_cutoff, self.beta)
        self._fy = OneEuroFilter(self.min_cutoff, self.beta)
        self._fz = OneEuroFilter(self.min_cutoff, self.beta)

    def retune(self, min_cutoff: float, beta: float) -> None:
        """Apply live GestureConfig changes without dropping filter history."""
        if min_cutoff == self.min_cutoff and beta == self.beta:
            return
        self.min_cutoff, self.beta = min_cutoff, beta
        for f in (self._fx, self._fy, self._fz):
            f.min_cutoff, f.beta = min_cutoff, beta

    def __call__(self, p: tuple[float, float, float], timestamp: float) -> tuple[float, float, float]:
        return (self._fx(p[0], timestamp), self._fy(p[1], timestamp), self._fz(p[2], timestamp))
