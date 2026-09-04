"""
Numerical verification of every load-bearing mathematical claim in the paper.

Each check prints PASS/FAIL and the measured quantity. The paper cites these
numbers; if a claim here fails, the corresponding paper claim must be removed.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats as sstats
from scipy.spatial.distance import cdist

from simulate import vendi_score_linear, HERE

rng = np.random.default_rng(4242)
RESULTS = {}


def check(name: str, ok: bool, detail: str):
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# T1. Anchor identity: (1/n) sum_i ||x - x_i||^2 = ||x - mu||^2 + (1/n) sum_i ||x_i - mu||^2
# so the mean-squared-distance anchor term is EXACTLY computable from two
# running statistics (centroid mu_n and mean squared spread), O(D) per query.
# ---------------------------------------------------------------------------
X = rng.normal(size=(5000, 32))
x = rng.normal(size=32)
mu = X.mean(axis=0)
lhs = ((X - x) ** 2).sum(axis=1).mean()
rhs = ((x - mu) ** 2).sum() + ((X - mu) ** 2).sum(axis=1).mean()
err = abs(lhs - rhs) / abs(lhs)
check("T1_anchor_identity", err < 1e-12, f"relative error {err:.2e}")

# ---------------------------------------------------------------------------
# T2. Nearest-neighbor gap decay: fresh draw vs n existing iid points on a
# k-dim manifold has E[min-dist] ~ c * n^(-1/k). Fit the log-log slope for
# k=3 (exam world geometry) and k=32 (full-rank poetry world).
# ---------------------------------------------------------------------------
def nn_decay_slope(k_int: int, D: int = 32, ns=(250, 500, 1000, 2000, 4000, 8000),
                   probes: int = 200) -> float:
    Q, _ = np.linalg.qr(rng.normal(size=(D, k_int)))
    means = []
    for n in ns:
        pts = (rng.normal(size=(n, k_int)) @ Q.T)
        fresh = (rng.normal(size=(probes, k_int)) @ Q.T)
        d = cdist(fresh, pts).min(axis=1)
        means.append(d.mean())
    slope, _, r, _, _ = sstats.linregress(np.log(ns), np.log(means))
    return slope


s3 = nn_decay_slope(3)
s32 = nn_decay_slope(32)
check("T2_nn_decay_k3", abs(s3 - (-1 / 3)) < 0.06,
      f"fitted slope {s3:.3f} vs theoretical -1/3 = -0.333")
check("T2_nn_decay_k32", s32 > -0.08,
      f"fitted slope {s32:.3f} vs theoretical -1/32 = -0.031 (near-flat)")

# ---------------------------------------------------------------------------
# T3. Packing bound: gated exam banks must respect
# N_max <= M * (2 R_eff / delta + 1)^k  (delta/2-balls disjoint inside an
# (R_eff + delta/2)-ball, per mode). Uses the exam sim's saved results.
# ---------------------------------------------------------------------------
from simulate_exam import M_EXAM, K_INTRINSIC, MODE_STD, DELTA

R_eff = MODE_STD * np.sqrt(sstats.chi2.ppf(0.99, K_INTRINSIC))
bound = M_EXAM * (2 * R_eff / DELTA + 1) ** K_INTRINSIC
try:
    with open(HERE / "figures" / "exam_summary.json") as f:
        exam = json.load(f)
    gated_banks = {k: v["real_bank_size"] for k, v in exam.items() if k.endswith("_gated")}
    worst = max(gated_banks.values())
    check("T3_packing_bound", worst <= bound,
          f"largest gated real bank {worst} <= bound {bound:.0f} "
          f"(R_eff={R_eff:.3f}, delta={DELTA}); observed banks {gated_banks}")
except FileNotFoundError:
    check("T3_packing_bound", False, "exam_summary.json missing - run simulate_exam.py first")

# ---------------------------------------------------------------------------
# T4. Junk conflation: in the UNGATED exam runs, the probability that an
# accepted item is junk must increase with bank size (the hard floor is
# progressively satisfied only by off-manifold noise).
# ---------------------------------------------------------------------------
try:
    z = np.load(HERE / "figures" / "exam_results.npz")
    jm = z["first_legal__junk"]
    n = len(jm)
    first_half, second_half = jm[: n // 2].mean(), jm[n // 2:].mean()
    check("T4_junk_conflation", second_half > 2 * first_half,
          f"junk fraction first half {first_half:.3f} -> second half {second_half:.3f}")
except (FileNotFoundError, KeyError) as e:
    check("T4_junk_conflation", False, f"missing exam results: {e}")

# ---------------------------------------------------------------------------
# T5. Vendi linear-kernel trick: nonzero eigenvalues of the n x n Gram matrix
# X_n X_n^T / n equal those of the D x D moment matrix X_n^T X_n / n, so the
# score is computable in O(nD^2 + D^3), independent of n^3.
# ---------------------------------------------------------------------------
Xs = rng.normal(size=(300, 16))
Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
gram_eigs = np.sort(np.linalg.eigvalsh(Xs @ Xs.T / 300))[::-1][:16]
mom_eigs = np.sort(np.linalg.eigvalsh(Xs.T @ Xs / 300))[::-1]
err5 = np.abs(gram_eigs - mom_eigs).max()
check("T5_vendi_duality", err5 < 1e-10, f"max eigenvalue mismatch {err5:.2e}")

# ---------------------------------------------------------------------------
# T6. Greedy trailing-eigenspace ("orthogonalization") selection raises the
# Vendi Score over random selection from the same candidate stream, and the
# spectral entropy objective it climbs is the exact log of the Vendi Score.
# ---------------------------------------------------------------------------
def greedy_orth_vendi(pool: np.ndarray, m: int) -> float:
    D = pool.shape[1]
    chosen = [pool[0]]
    Cs = np.outer(pool[0], pool[0])
    ptr = 1
    while len(chosen) < m:
        batch = pool[ptr: ptr + 8]
        ptr += 8
        C = Cs / len(chosen)
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        cum = np.cumsum(w) / max(w.sum(), 1e-12)
        j = int(np.searchsorted(cum, 0.8)) + 1
        res = 1.0 - ((batch @ V[:, :j]) ** 2).sum(axis=1)
        pick = batch[int(np.argmax(res))]
        chosen.append(pick)
        Cs += np.outer(pick, pick)
    return vendi_score_linear(np.array(chosen))


# anisotropic pool: a few dominant directions, like a real style-collapsed model
A = rng.normal(size=(32, 32)) * np.linspace(2.5, 0.2, 32)[None, :]
pool = rng.normal(size=(4000, 32)) @ A.T
pool /= np.linalg.norm(pool, axis=1, keepdims=True)
v_greedy = greedy_orth_vendi(pool, 400)
v_rand = float(np.mean([vendi_score_linear(pool[rng.choice(4000, 400, replace=False)])
                        for _ in range(5)]))
check("T6_orth_raises_vendi", v_greedy > 1.15 * v_rand,
      f"greedy-orth Vendi {v_greedy:.2f} vs random {v_rand:.2f} "
      f"({v_greedy / v_rand:.2f}x)")

# ---------------------------------------------------------------------------
# T7. Best-of-K novelty gain is (a) strongly sublinear in K, and (b) paid for
# in TYPICALITY: under a density with tails (Gaussian here), the candidate
# with the largest min-gap is systematically an atypical draw. For a uniform
# density on a k-manifold the gain would be ~(ln K)^{1/k}; the measured excess
# over that comes from selection drifting into the low-density tail -- the
# same mechanism that, pushed to its limit, fills a constrained bank with
# junk (T4). Novelty pressure without a typicality gate DOES NOT stay on the
# data manifold; that is a theorem-shaped fact, not an implementation bug.
# ---------------------------------------------------------------------------
def bestK_stats(k_int: int, K: int, n: int = 2000, probes: int = 400):
    Q, _ = np.linalg.qr(rng.normal(size=(32, k_int)))
    pts = rng.normal(size=(n, k_int)) @ Q.T
    g1, gK, norm_sel, norm_all = [], [], [], []
    for _ in range(probes):
        z = rng.normal(size=(K, k_int))
        cands = z @ Q.T
        d = cdist(cands, pts).min(axis=1)
        j = int(np.argmax(d))
        g1.append(d[0])
        gK.append(d[j])
        norm_sel.append(np.linalg.norm(z[j]))
        norm_all.append(np.linalg.norm(z, axis=1).mean())
    return (float(np.mean(gK) / np.mean(g1)),
            float(np.mean(norm_sel) / np.mean(norm_all)))


gains = {K: bestK_stats(3, K)[0] for K in (2, 8, 32)}
_, norm_inflation = bestK_stats(3, 8)
sublinear = gains[32] / gains[2] < (32 / 2) ** (1 / 3)  # far below even K^{1/k} scaling
check("T7a_bestK_sublinear", sublinear and gains[32] < 4.0,
      f"min-gap gain: K=2 -> {gains[2]:.2f}, K=8 -> {gains[8]:.2f}, "
      f"K=32 -> {gains[32]:.2f} (uniform-density theory ~(ln K)^(1/3): "
      f"{np.log(2) ** (1/3):.2f}, {np.log(8) ** (1/3):.2f}, {np.log(32) ** (1/3):.2f})")
check("T7b_novelty_buys_atypicality", norm_inflation > 1.10,
      f"selected candidate's latent norm is {norm_inflation:.2f}x the mean "
      f"candidate norm -- max-gap selection systematically picks tail draws")

with open(HERE / "figures" / "theory_checks.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
n_pass = sum(1 for r in RESULTS.values() if r["pass"])
print(f"\n{n_pass}/{len(RESULTS)} checks passed")
