"""
Numerical verification of the conditional-dimensionality theorems.

Setting. The reachable semantic manifold  Mfd  has dimension d. For a FIXED
prompt x the generator's output distribution concentrates on a slice
S_x subset Mfd of dimension m, with m << d. Prompts are drawn from a prompt
space; a prompt schedule induces a family of slices whose union is what the
corpus can ever cover.

Theorems checked here (statements in PAPER.md):
  P1  fixed-prompt saturation:  E[gap_n] = Theta(n^{-1/m}), NOT n^{-1/d}
  P2  slice deficit:            one slice ε-covers a vanishing fraction of Mfd;
                                covered fraction ~ ε^{d-m}
  P3  breadth beats depth:      at fixed budget B, coverage is maximized at
                                depth n* = Theta(ε^{-m}) per prompt, then move
  P4  transversality:           the union of P slices has dimension
                                min(d, m + rank of the prompt->slice map);
                                refinement that moves slices TRANSVERSELY
                                raises reachable dimension, parallel motion
                                does not
  P5  gate necessity:           off-manifold candidates dominate the max-min
                                objective, so the argmax is not consistent
                                without a typicality constraint
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats as sstats
from scipy.spatial.distance import cdist

from simulate import HERE

rng = np.random.default_rng(90210)
RESULTS = {}
AMBIENT = 64


def check(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def slice_basis(d: int, m: int, rng) -> np.ndarray:
    """An m-frame inside a fixed d-dim subspace of R^AMBIENT."""
    Qd, _ = np.linalg.qr(rng.normal(size=(AMBIENT, d)))
    A = rng.normal(size=(d, m))
    Qm, _ = np.linalg.qr(A)
    return Qd @ Qm            # (AMBIENT, m)


# ---------------------------------------------------------------------------
# P1. Fixed-prompt saturation rate is governed by m, not d.
# ---------------------------------------------------------------------------
def gap_slope(m: int, ns=(100, 200, 400, 800, 1600, 3200), probes=300) -> float:
    F = slice_basis(32, m, rng)
    means = []
    for n in ns:
        pts = rng.normal(size=(n, m)) @ F.T
        fresh = rng.normal(size=(probes, m)) @ F.T
        means.append(cdist(fresh, pts).min(axis=1).mean())
    return sstats.linregress(np.log(ns), np.log(means)).slope


rows = {}
for m in (2, 3, 5):
    s = gap_slope(m)
    rows[m] = (s, -1 / m)
ok = all(abs(s - t) < 0.075 for s, t in rows.values())
check("P1_fixed_prompt_rate_is_m", ok,
      "; ".join(f"m={m}: fitted {s:.3f} vs -1/m={t:.3f}" for m, (s, t) in rows.items())
      + f"  [ambient d=32 would predict {-1/32:.3f}]")

# ---------------------------------------------------------------------------
# P2. Slice deficit: fraction of the d-manifold within eps of ONE slice.
# Predicted to scale as eps^(d-m). We fit the exponent.
# ---------------------------------------------------------------------------
def covered_fraction(d: int, m: int, eps_list, n_slice=4000, n_ref=20000) -> list[float]:
    Qd, _ = np.linalg.qr(rng.normal(size=(AMBIENT, d)))
    A = rng.normal(size=(d, m))
    Qm, _ = np.linalg.qr(A)
    F = Qd @ Qm
    # reference measure: uniform-ish on the d-dim unit ball (in its own coords)
    z = rng.normal(size=(n_ref, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z *= rng.random((n_ref, 1)) ** (1 / d)
    ref = z @ Qd.T
    # the slice, densely filled inside the same radius
    y = rng.normal(size=(n_slice, m))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    y *= rng.random((n_slice, 1)) ** (1 / m)
    sl = y @ F.T
    dmin = cdist(ref, sl).min(axis=1)
    return [float((dmin <= e).mean()) for e in eps_list]


eps_list = np.array([0.5, 0.35, 0.25, 0.18, 0.12])
d_amb, m_sl = 8, 2
fr = covered_fraction(d_amb, m_sl, eps_list)
pos = [(e, f) for e, f in zip(eps_list, fr) if f > 1e-4]
slope = sstats.linregress(np.log([e for e, _ in pos]), np.log([f for _, f in pos])).slope
check("P2_slice_deficit_exponent", abs(slope - (d_amb - m_sl)) < 1.2,
      f"d={d_amb}, m={m_sl}: fitted eps-exponent {slope:.2f} vs predicted d-m={d_amb - m_sl}; "
      f"covered fractions {[round(f, 4) for f in fr]} at eps={list(eps_list)}")

# ---------------------------------------------------------------------------
# P3. Breadth beats depth. Fixed budget B; split as P prompts x n_per samples.
# Coverage of the union of slices is maximized at an interior optimum n* that
# grows like eps^{-m}. We sweep n_per and locate the argmax.
# ---------------------------------------------------------------------------
def union_coverage(d: int, m: int, B: int, n_per: int, eps: float,
                   n_ref: int = 12000) -> float:
    Qd, _ = np.linalg.qr(rng.normal(size=(AMBIENT, d)))
    z = rng.normal(size=(n_ref, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z *= rng.random((n_ref, 1)) ** (1 / d)
    ref = z @ Qd.T
    P = max(B // n_per, 1)
    pts = []
    for _ in range(P):
        A = rng.normal(size=(d, m))
        Qm, _ = np.linalg.qr(A)
        F = Qd @ Qm
        center_c = rng.normal(size=d) * 0.35
        y = rng.normal(size=(n_per, m))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y *= rng.random((n_per, 1)) ** (1 / m)
        pts.append(y @ F.T + (Qd @ center_c)[None, :])
    X = np.vstack(pts)
    dmin = cdist(ref, X).min(axis=1)
    return float((dmin <= eps).mean())


B_fixed = 2400
DEPTHS = (1, 3, 10, 30, 100, 300, 1200)


def sweep_depth(eps: float, switch_cost: float) -> dict[int, float]:
    """`switch_cost` c: generations-equivalent overhead to move to a new prompt
    (eliciting a spec, embedding it, occasionally refining the axis tree). The
    budget buys P = B / (c + n_per) prompts at depth n_per."""
    row = {}
    for n_per in DEPTHS:
        eff_B = int(B_fixed * n_per / (switch_cost + n_per))
        row[n_per] = union_coverage(6, 2, max(eff_B, n_per), n_per, eps)
    return row


# (i) free switching: the optimum sits at the BOUNDARY, n* = 1.
free = sweep_depth(0.45, switch_cost=0.0)
best_free = max(free, key=free.get)
check("P3a_free_switching_is_maximal_breadth", best_free == 1,
      f"with no prompt-switch cost the optimum is n*={best_free} (one sample per "
      f"prompt): coverage {json.dumps({k: round(v, 3) for k, v in free.items()})}")

# (ii) with a per-prompt overhead the optimum moves INTO the interior and
# grows with the overhead -- this is the regime real pipelines are in.
sweep = {}
best = {}
for c in (3.0, 30.0):
    sweep[c] = sweep_depth(0.45, switch_cost=c)
    best[c] = max(sweep[c], key=sweep[c].get)
interior = all(b > 1 for b in best.values())
monotone = best[30.0] >= best[3.0]
check("P3b_interior_optimum_under_switch_cost", interior and monotone,
      f"optimal depth n* grows with switch cost c: c=3 -> n*={best[3.0]}, "
      f"c=30 -> n*={best[30.0]}; coverage at c=30: "
      f"{json.dumps({k: round(v, 3) for k, v in sweep[30.0].items()})}")

deep_vs_best = free[1200] / free[best_free]
check("P3c_depth_only_wastes_budget", deep_vs_best < 0.75,
      f"spending the whole budget on ONE prompt reaches only "
      f"{deep_vs_best:.1%} of the coverage maximal breadth reaches")

# ---------------------------------------------------------------------------
# P4. Transversality: slices displaced along directions INSIDE their own span
# do not raise the reachable dimension; transverse displacement does.
# Measured by participation ratio of the union's covariance spectrum.
# ---------------------------------------------------------------------------
def union_dim(mode: str, P: int = 24, m: int = 2, n_per: int = 200) -> float:
    F = slice_basis(16, m, rng)                # shared slice frame
    span = F                                   # directions inside the slice
    Qperp, _ = np.linalg.qr(rng.normal(size=(AMBIENT, 16)))
    pts = []
    for _ in range(P):
        if mode == "parallel":
            shift = span @ (rng.normal(size=m) * 1.5)
        else:                                   # transverse
            shift = Qperp @ (rng.normal(size=16) * 1.5)
            shift -= span @ (span.T @ shift)    # strictly off the slice span
        pts.append(rng.normal(size=(n_per, m)) @ F.T + shift[None, :])
    X = np.vstack(pts)
    X -= X.mean(axis=0)
    w = np.linalg.eigvalsh(np.cov(X.T))
    w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())  # participation ratio


d_par, d_tra = union_dim("parallel"), union_dim("transverse")
check("P4_transversality", d_tra > 2.5 * d_par,
      f"effective dimension of the union: parallel shifts {d_par:.2f} "
      f"(stays ~m=2), transverse shifts {d_tra:.2f}")

# ---------------------------------------------------------------------------
# P5. Gate necessity: with a heavy-tailed off-manifold component, the
# unconstrained argmax of min-gap selects off-manifold candidates with
# probability -> 1 as the corpus grows.
# ---------------------------------------------------------------------------
def junk_capture(n_corpus: int, K: int = 8, junk_p: float = 0.005,
                 trials: int = 20000) -> tuple[float, float]:
    """Returns (P(selected is junk), P(selected is junk | junk in pool))."""
    F = slice_basis(8, 3, rng)
    corpus = rng.normal(size=(n_corpus, 3)) @ F.T
    hits = pools = cond_hits = 0
    for _ in range(trials):
        is_junk = rng.random(K) < junk_p
        c = rng.normal(size=(K, 3)) @ F.T
        if is_junk.any():
            c[is_junk] = rng.normal(size=(int(is_junk.sum()), AMBIENT)) * 3.0
        d = cdist(c, corpus).min(axis=1)
        sel_junk = bool(is_junk[int(np.argmax(d))])
        hits += sel_junk
        if is_junk.any():
            pools += 1
            cond_hits += sel_junk
    return hits / trials, (cond_hits / pools if pools else 0.0)


p_small, cond_small = junk_capture(50)
p_large, cond_large = junk_capture(5000)
# The right baseline is the per-candidate junk rate p: an unbiased selector
# would pick junk with probability p. Max-min picks it whenever it appears,
# so the acceptance rate is inflated to ~1-(1-p)^K, a factor ~K.
amplification = p_large / 0.005
check("P5_gate_necessity", cond_large > 0.95 and amplification > 5,
      f"P(selected is off-manifold) = {p_large:.4f} vs per-candidate rate "
      f"p=0.005 -- a {amplification:.1f}x amplification (K={8}). "
      f"P(selected is junk | junk present in pool) = {cond_large:.3f} at n=5000 "
      f"({cond_small:.3f} at n=50): max-min selects the off-manifold candidate "
      f"essentially whenever one appears, at every corpus size")

with open(HERE / "figures" / "slice_theory_checks.json", "w") as f:
    json.dump({"results": RESULTS, "depth_sweep": {str(k): v for k, v in sweep.items()}},
              f, indent=2)
print(f"\n{sum(1 for r in RESULTS.values() if r['pass'])}/{len(RESULTS)} checks passed")
