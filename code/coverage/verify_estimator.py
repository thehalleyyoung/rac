"""
Ground-truth validation of the Monte-Carlo coverage estimator.

Before trusting covered-fraction on real embeddings, we check the estimator on
cases whose answer is known analytically:

  A. UNIFORM CUBE, one ball. The covered fraction of [0,1]^d by a ball of
     radius eps centered at the middle is the ball's volume (for eps small
     enough that it does not touch the faces): V = pi^(d/2) eps^d / Gamma(d/2+1).
     We compare the MC estimate against that closed form.

  B. GRID OF DISJOINT BALLS. m centers spaced further than 2 eps apart in a
     cube: the union's volume is exactly m * V (no overlap), so the covered
     fraction is m*V / 1. Checks that the union logic does not double count.

  C. NESTED / IDENTICAL CENTERS. Adding a duplicate center must add exactly
     zero coverage -- the property that makes the functional submodular and
     the property a buggy implementation (e.g. summing per-center counts
     instead of unioning) would violate.

  D. KNOWN-ANSWER SELECTION. On a 1-D grid of equally spaced reference points
     with eps chosen so each chosen point covers exactly (2r+1) of them, the
     optimal k-subset covers exactly k*(2r+1) points; greedy must attain it.

Run: python3 verify_estimator.py   ->  figures/summary_estimator.json
"""
from __future__ import annotations

import json
from math import gamma, pi
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
SEED = 20260830


def ball_volume(d: int, eps: float) -> float:
    return pi ** (d / 2) * eps ** d / gamma(d / 2 + 1)


def covered_fraction(pool: np.ndarray, S: np.ndarray, eps: float) -> float:
    from scipy.spatial.distance import cdist
    return float((cdist(pool, S).min(axis=1) <= eps).mean())


def check_single_ball(rng) -> dict:
    out = {}
    for d, eps in ((2, 0.20), (3, 0.25), (4, 0.30)):
        truth = ball_volume(d, eps)
        P = 400_000
        pool = rng.random((P, d))
        S = np.full((1, d), 0.5)
        est = covered_fraction(pool, S, eps)
        se = float(np.sqrt(truth * (1 - truth) / P))
        out[f"d={d},eps={eps}"] = {
            "analytic_volume": round(truth, 6),
            "mc_estimate": round(est, 6),
            "abs_error": round(abs(est - truth), 6),
            "mc_standard_error": round(se, 6),
            "within_3_se": bool(abs(est - truth) <= 3 * se),
        }
    return out


def check_disjoint_balls(rng) -> dict:
    d, eps = 2, 0.05
    centers = np.array([[x, y] for x in (0.2, 0.5, 0.8) for y in (0.2, 0.5, 0.8)])
    m = len(centers)
    # spacing 0.3 > 2*eps = 0.1, so the balls are pairwise disjoint
    truth = m * ball_volume(d, eps)
    P = 800_000
    pool = rng.random((P, d))
    est = covered_fraction(pool, centers, eps)
    se = float(np.sqrt(truth * (1 - truth) / P))
    return {
        "m_centers": m, "eps": eps,
        "min_center_spacing": round(float(np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)[np.triu_indices(m, 1)].min())), 4),
        "analytic_union_volume": round(truth, 6),
        "mc_estimate": round(est, 6),
        "abs_error": round(abs(est - truth), 6),
        "mc_standard_error": round(se, 6),
        "within_3_se": bool(abs(est - truth) <= 3 * se),
    }


def check_duplicate_center(rng) -> dict:
    d, eps, P = 3, 0.25, 200_000
    pool = rng.random((P, d))
    S1 = np.full((1, d), 0.5)
    S2 = np.vstack([S1, S1, S1])       # three identical centers
    a, b = covered_fraction(pool, S1, eps), covered_fraction(pool, S2, eps)
    return {"one_center": round(a, 6), "three_identical_centers": round(b, 6),
            "delta": round(b - a, 9), "duplicate_adds_zero": bool(a == b)}


def check_known_optimum() -> dict:
    """1-D: reference points at integers 0..N-1; candidates at the same spots.
    With eps = r + 0.5 each chosen point covers exactly 2r+1 reference points,
    and k disjointly-placed points cover exactly k(2r+1). Greedy should hit it."""
    N, r, k = 300, 3, 10
    ref = np.arange(N, dtype=float)[:, None]
    cand = np.arange(N, dtype=float)[:, None]
    eps = r + 0.5
    cover = np.abs(cand[:, None, 0] - ref[None, :, 0]) <= eps   # (N, N) bool
    optimum = k * (2 * r + 1)
    covered = np.zeros(N, dtype=bool)
    chosen = []
    for _ in range(k):
        gains = (cover & ~covered).sum(axis=1)
        gains[chosen] = -1
        i = int(np.argmax(gains))
        chosen.append(i)
        covered |= cover[i]
    got = int(covered.sum())
    return {"N": N, "r": r, "k": k, "eps": eps,
            "analytic_optimum_points_covered": optimum,
            "greedy_points_covered": got,
            "greedy_equals_optimum": bool(got == optimum),
            "ratio": round(got / optimum, 4)}


def main():
    FIGDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {
        "A_single_ball_vs_analytic_volume": check_single_ball(rng),
        "B_disjoint_balls_union": check_disjoint_balls(rng),
        "C_duplicate_center_adds_zero": check_duplicate_center(rng),
        "D_known_optimum_1d": check_known_optimum(),
    }
    ok = (all(v["within_3_se"] for v in res["A_single_ball_vs_analytic_volume"].values())
          and res["B_disjoint_balls_union"]["within_3_se"]
          and res["C_duplicate_center_adds_zero"]["duplicate_adds_zero"]
          and res["D_known_optimum_1d"]["greedy_equals_optimum"])
    res["all_checks_passed"] = bool(ok)
    with open(FIGDIR / "summary_estimator.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("\nALL ESTIMATOR CHECKS PASSED" if ok else "\nESTIMATOR CHECK FAILED")


if __name__ == "__main__":
    main()
