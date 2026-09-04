"""
Resampling guards, because writing the trap down twice did not prevent it.

Two statistics in this toolkit break under the ordinary bootstrap, and both
broke in practice before the reason was noticed:

  nearest-neighbour distance   resampling with replacement puts a point in the
                               sample twice; the duplicate pair sits at distance
                               zero, so min NN collapses to 0 in EVERY arm at
                               once and every difference reads as exactly
                               +0.0000. That looks like a clean null.

  pairwise similarity          the same duplicate pair scores a Jaccard of 1.0,
                               so mean overlap inflates and the interval floats
                               ABOVE the point estimate it is meant to bracket.
                               That looks like a wide but usable CI.

The first failure was diagnosed in the render best-of-K work and written into
the ledger; the second happened anyway, in a different script, because a note in
a log is not a guard. So the guard lives here and the check is mechanical:
`subsample` never draws with replacement, and `bootstrap_ci` refuses outright
for statistic kinds where replacement is invalid.

Use `bootstrap_ci(rows, fn, kind=...)` and state the kind. For "mean" and
"count" statistics replacement is fine and is used; for "pairwise" and
"neighbour" it is refused and a 75% subsample is used instead. A returned
interval that does not bracket the point estimate raises, since that is the
signature of exactly this bug.
"""
from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np

Kind = Literal["mean", "count", "pairwise", "neighbour"]
NO_REPLACEMENT: set[str] = {"pairwise", "neighbour"}


def subsample(rows: Sequence, frac: float = 0.75, rng=None) -> list:
    """Draw a fraction of `rows` WITHOUT replacement. Never duplicates a row."""
    rng = rng or np.random.default_rng()
    m = max(int(len(rows) * frac), 8)
    m = min(m, len(rows))
    return [rows[i] for i in rng.choice(len(rows), m, replace=False)]


def bootstrap_ci(rows: Sequence, fn: Callable[[Sequence], float], *,
                 kind: Kind, n: int = 2000, frac: float = 0.75,
                 seed: int = 0, check: bool = True) -> tuple[float, float, float]:
    """Return (point, lo, hi). Replacement is used only where it is valid.

    `check` verifies the interval brackets the point estimate and raises if it
    does not -- the observable signature of resampling a statistic that cannot
    tolerate duplicates.
    """
    if kind not in ("mean", "count", "pairwise", "neighbour"):
        raise ValueError(f"unknown statistic kind {kind!r}")
    rng = np.random.default_rng(seed)
    point = float(fn(list(rows)))
    draws: list[float] = []
    for _ in range(n):
        if kind in NO_REPLACEMENT:
            sample = subsample(rows, frac, rng)
        else:
            sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        try:
            v = fn(sample)
        except Exception:
            continue
        if np.isfinite(v):
            draws.append(float(v))
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    if check and not (lo - 1e-9 <= point <= hi + 1e-9):
        raise AssertionError(
            f"interval [{lo:.4f}, {hi:.4f}] does not bracket the point estimate "
            f"{point:.4f} for kind={kind!r}. This is the duplicate-row signature: "
            f"a pairwise or neighbour statistic resampled with replacement.")
    return point, lo, hi


def paired_delta(a: Sequence, b: Sequence, fn: Callable[[Sequence], float], *,
                 kind: Kind, n: int = 2000, frac: float = 0.75,
                 seed: int = 0) -> tuple[float, float, float]:
    """(delta, lo, hi) for fn(b) - fn(a), resampling each arm under its kind."""
    rng = np.random.default_rng(seed)
    point = float(fn(list(b))) - float(fn(list(a)))
    d: list[float] = []
    for _ in range(n):
        if kind in NO_REPLACEMENT:
            sa, sb = subsample(a, frac, rng), subsample(b, frac, rng)
        else:
            sa = [a[i] for i in rng.integers(0, len(a), len(a))]
            sb = [b[i] for i in rng.integers(0, len(b), len(b))]
        try:
            v = fn(sb) - fn(sa)
        except Exception:
            continue
        if np.isfinite(v):
            d.append(float(v))
    if not d:
        return point, float("nan"), float("nan")
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return point, lo, hi
