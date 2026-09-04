"""
Budget-matched selection benchmark on real gpt-5.6-luna corpora.

THE DIAGNOSIS THIS SCRIPT IMPLEMENTS
------------------------------------
Our first real-data comparison (real_coverage.py) scored whole GENERATION
PIPELINES against a denominator built from unconditioned prompting, and every
conditioned pipeline scored ~0. Three things were wrong with it, and only the
third is a fact about the world:

  (1) IT COMPARED THE WRONG THING. "Persona-Hub vs ours" conflates two
      separable questions: what candidates get generated, and which of them
      get kept. This paper's contribution is a SELECTION rule under a covering
      objective. Selection is only comparable when every rule sees the same
      candidates, so here every method selects from ONE shared pool.

  (2) EPSILON WAS CALIBRATED ON THE WRONG SET. We calibrated eps on the dense
      naive corpus, then scored against other denominators. eps must be
      commensurate with the reference measure it is applied to: a radius tuned
      to the tightest region in the space makes every diffuse method score
      zero everywhere, which is what happened. Here eps is calibrated on the
      reference pool that is actually used, and we report its percentile in
      that pool's own distance distribution.

  (3) THE DENOMINATOR ENCODED A DISTRIBUTION. Unconditioned prompting is not
      "the reachable space"; it is one region of it. We build the reference as
      a STRATUM-BALANCED union over every generation method, so no single
      conditioning regime (including no-conditioning) sets the denominator.
      Arms contribute equally by construction.

BUDGET MATCHING. Every method here selects k items from the same pool of
generated candidates, so all of them consume exactly |POOL| generations and
differ only in which k they keep. The comparison is therefore budget-matched
by construction -- no method can buy coverage by generating more, which is the
confound that matched-n comparisons hide.

BASELINES (all real, from the literature, implemented faithfully)
  random          uniform subset -- what naive repeated prompting yields
  kcenter         Gonzalez (1985) farthest-point traversal; 2-approx for
                  k-center. The packing objective, and the closest classical
                  neighbour of a coverage objective
  semdedup        SemDeDup (Abbas et al. 2023): drop near-duplicates within a
                  radius, sample uniformly from survivors
  rouge_filter    Self-Instruct / Alpaca (Wang et al. 2023): sequential accept
                  with ROUGE-L <= 0.7 against everything kept so far
  dpp_greedy      greedy MAP inference for a determinantal point process
                  (Kulesza & Taskar 2012; Chen et al. 2018 fast greedy MAP)
  coverage_greedy OURS: submodular greedy marginal eps-ball coverage
  coverage_stream OURS, deployable: best-of-K streaming greedy, the rule an
                  online generation pipeline can actually run

Run: python3 real_benchmark.py  ->  figures/summary_benchmark.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REAL = HERE.parent / "real"
FIGDIR = HERE / "figures"
SEED = 20260830
K_BUDGET = 400          # items each method selects
STREAM_K = 8            # candidates per step for the streaming variant

from real_coverage import (DOMAINS, load_arm, dedup, blocked_min_dist,  # noqa: E402
                           self_nn, vendi, ngram)


# ------------------------------------------------------------------ helpers

def min_gap(S: np.ndarray) -> float:
    S2 = (S ** 2).sum(axis=1)
    d2 = S2[:, None] - 2.0 * (S @ S.T) + S2[None, :]
    np.fill_diagonal(d2, np.inf)
    return round(float(np.sqrt(max(d2.min(), 0.0))), 4)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(a: list[str], b: list[str]) -> float:
    l = lcs_len(a, b)
    if l == 0:
        return 0.0
    p, r = l / len(b), l / len(a)
    return 2 * p * r / (p + r)


# ------------------------------------------------------------------ methods

def sel_random(pool, texts, cover, k, rng, **kw):
    return list(rng.permutation(len(pool))[:k])


def sel_kcenter(pool, texts, cover, k, rng, **kw):
    """Gonzalez farthest-point traversal (2-approximation for k-center)."""
    start = int(rng.integers(len(pool)))
    chosen = [start]
    d = np.linalg.norm(pool - pool[start], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(pool - pool[i], axis=1))
    return chosen


def sel_semdedup(pool, texts, cover, k, rng, dedup_radius=None, **kw):
    """SemDeDup: remove near-duplicates inside a radius, then sample uniformly.
    Survivors are found greedily in random order (the standard cheap variant)."""
    order = rng.permutation(len(pool))
    keep: list[int] = []
    keptE: list[np.ndarray] = []
    for i in order:
        if keptE:
            if np.linalg.norm(np.array(keptE) - pool[i], axis=1).min() < dedup_radius:
                continue
        keep.append(int(i))
        keptE.append(pool[i])
        if len(keep) >= k:
            break
    if len(keep) < k:   # radius too aggressive: top up at random
        rest = [i for i in order if i not in set(keep)]
        keep.extend(int(x) for x in rest[:k - len(keep)])
    return keep


def sel_rouge_filter(pool, texts, cover, k, rng, rouge_max=0.7, **kw):
    """Self-Instruct / Alpaca: sequential accept if ROUGE-L vs every kept item
    is <= rouge_max. Compared against a bounded recent window for tractability,
    which is what the reference implementations do at scale."""
    order = rng.permutation(len(pool))
    toks = {}
    keep: list[int] = []
    window = 200
    for i in order:
        ti = toks.setdefault(int(i), texts[int(i)].lower().split())
        ok = True
        for j in keep[-window:]:
            tj = toks.setdefault(j, texts[j].lower().split())
            if rouge_l(ti, tj) > rouge_max:
                ok = False
                break
        if ok:
            keep.append(int(i))
        if len(keep) >= k:
            break
    if len(keep) < k:
        rest = [int(i) for i in order if i not in set(keep)]
        keep.extend(rest[:k - len(keep)])
    return keep


def sel_dpp_greedy(pool, texts, cover, k, rng, **kw):
    """Greedy MAP for a DPP with linear (cosine) kernel, via incremental
    Cholesky (Chen et al. 2018). Maximizes log-det = volume spanned."""
    n = len(pool)
    d2 = np.ones(n)                       # diag of L for unit-norm vectors
    chosen: list[int] = []
    C = np.zeros((k, n))
    for it in range(k):
        i = int(np.argmax(d2))
        chosen.append(i)
        if it == k - 1:
            break
        ci = (pool @ pool[i] - C[:it].T @ C[:it, i]) / max(np.sqrt(d2[i]), 1e-12)
        C[it] = ci
        d2 = np.maximum(d2 - ci ** 2, 1e-12)
        d2[chosen] = -np.inf
    return chosen


def sel_coverage_greedy(pool, texts, cover, k, rng, **kw):
    """OURS: submodular greedy on the MC coverage functional."""
    covered = np.zeros(cover.shape[1], dtype=bool)
    chosen: list[int] = []
    for _ in range(k):
        gains = cover[:, ~covered].sum(axis=1)
        gains[chosen] = -1
        i = int(np.argmax(gains))
        if gains[i] <= 0:      # saturated: fall back to farthest-point
            d = np.linalg.norm(pool - pool[chosen[-1]], axis=1)
            d[chosen] = -1
            i = int(np.argmax(d))
        chosen.append(i)
        covered |= cover[i]
    return chosen


def sel_coverage_stream(pool, texts, cover, k, rng, stream_k=STREAM_K, **kw):
    """OURS, deployable: the pool arrives in random order in batches of
    stream_k; keep the batch member with the largest marginal gain. This is
    what an online pipeline can run -- it never rescans the pool."""
    order = rng.permutation(len(pool))
    covered = np.zeros(cover.shape[1], dtype=bool)
    chosen: list[int] = []
    for j in range(k):
        batch = order[j * stream_k:(j + 1) * stream_k]
        if len(batch) == 0:
            break
        g = cover[batch][:, ~covered].sum(axis=1)
        i = int(batch[int(np.argmax(g))])
        chosen.append(i)
        covered |= cover[i]
    return chosen


METHODS = {
    "random": sel_random,
    "kcenter": sel_kcenter,
    "semdedup": sel_semdedup,
    "rouge_filter": sel_rouge_filter,
    "dpp_greedy": sel_dpp_greedy,
    "coverage_greedy (ours)": sel_coverage_greedy,
    "coverage_stream (ours)": sel_coverage_stream,
}


# ------------------------------------------------------------------ driver

def build_pool_and_ref(domain: str, rng) -> dict:
    """Stratum-balanced split: every arm contributes equally to BOTH the
    candidate pool and the reference, and no item is in both."""
    per_arm = {}
    for arm in DOMAINS[domain]:
        got = load_arm(domain, arm)
        if got is None:
            continue
        texts, E = got
        t, e, st = dedup(texts, E)
        per_arm[arm] = (t, e, st)
    take = min(len(e) for _, e, _ in per_arm.values())
    pool_t, pool_e, pool_arm, ref_e = [], [], [], []
    for arm, (t, e, st) in per_arm.items():
        idx = rng.permutation(len(e))[:take]
        half = len(idx) // 2
        for i in idx[:half]:
            pool_t.append(t[i]); pool_e.append(e[i]); pool_arm.append(arm)
        ref_e.append(e[idx[half:]])
    return {
        "texts": pool_t,
        "E": np.array(pool_e),
        "arm": pool_arm,
        "REF": np.vstack(ref_e),
        "per_arm_stats": {a: st for a, (_, _, st) in per_arm.items()},
        "per_arm_take": int(take),
    }


def analyze(domain: str, rng) -> dict:
    print(f"\n=== {domain} ===", flush=True)
    B = build_pool_and_ref(domain, rng)
    pool, texts, REF = B["E"], B["texts"], B["REF"]
    print(f"  pool={len(pool)} candidates (balanced over {len(B['per_arm_stats'])} arms, "
          f"{B['per_arm_take']} deduped items each), reference={len(REF)}")

    # eps calibrated ON THE REFERENCE ITSELF -- the fix for confound (2)
    nn = self_nn(REF)
    delta = float(np.median(nn))
    eps_list = [round(0.6 * delta, 4), round(delta, 4), round(1.6 * delta, 4)]
    eps_sel = eps_list[1]
    # where does eps sit in the reference's own pairwise distance distribution?
    sub = REF[rng.permutation(len(REF))[:1500]]
    pd = np.sqrt(np.clip(2 - 2 * (sub @ sub.T), 0, None))
    iu = np.triu_indices(len(sub), 1)
    pct = {str(e): round(float((pd[iu] <= e).mean()), 4) for e in eps_list}
    print(f"  eps (calibrated on REFERENCE): delta={delta:.4f} -> {eps_list}")
    print(f"  eps percentile of reference pairwise distances: {pct}")

    # NO TEST LEAKAGE. An earlier version of this script built ONE incidence
    # matrix against REF, handed it to the selectors, and then scored against
    # that same REF -- so coverage_greedy was optimizing the evaluation set
    # directly and its margin measured tautology, not generalization. The
    # reference is now split into two disjoint halves: EST is the estimation
    # measure the selector may see (it stands for a pipeline's own cheap
    # draws), EVAL is held out for scoring and no method ever sees it.
    from scipy.spatial.distance import cdist
    perm = rng.permutation(len(REF))
    half = len(perm) // 2
    REF_EST, REF_EVAL = REF[perm[:half]], REF[perm[half:]]
    cover = cdist(pool, REF_EST) <= eps_sel
    print(f"  estimation incidence {cover.shape} (mean balls/candidate="
          f"{cover.sum(axis=1).mean():.1f}); held-out eval reference "
          f"{len(REF_EVAL)}")

    dedup_radius = float(np.quantile(nn, 0.25))
    results = {}
    for name, fn in METHODS.items():
        t0 = time.time()
        idx = fn(pool, texts, cover, K_BUDGET, np.random.default_rng(SEED + 5),
                 dedup_radius=dedup_radius)
        idx = list(dict.fromkeys(int(i) for i in idx))[:K_BUDGET]
        S = pool[idx]
        d = blocked_min_dist(REF_EVAL, S)
        sel_texts = [texts[i] for i in idx]
        arms_used = {}
        for i in idx:
            arms_used[B["arm"][i]] = arms_used.get(B["arm"][i], 0) + 1
        results[name] = {
            "k": len(idx),
            "generations_consumed": len(pool),
            "covered": {str(e): round(float((d <= e).mean()), 4) for e in eps_list},
            "min_gap": min_gap(S),
            "vendi_centered": vendi(S, True),
            "vendi_uncentered": vendi(S, False),
            **ngram(sel_texts),
            "arm_mix": dict(sorted(arms_used.items(), key=lambda kv: -kv[1])),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        r = results[name]
        print(f"  {name:24s} cov@mid={r['covered'][str(eps_sel)]:.4f} "
              f"gap={r['min_gap']:.4f} vendi_c={r['vendi_centered']:6.2f} "
              f"d2={r['distinct_2']:.3f} ({r['elapsed_sec']}s)")

    best = max(results, key=lambda m: results[m]["covered"][str(eps_sel)])
    ours = "coverage_greedy (ours)"
    base = [m for m in results if "ours" not in m]
    best_base = max(base, key=lambda m: results[m]["covered"][str(eps_sel)])
    return {
        "domain": domain,
        "pool_size": len(pool), "reference_size": len(REF),
        "reference_estimation_half": int(len(REF_EST)),
        "reference_heldout_eval_half": int(len(REF_EVAL)),
        "leakage_note": "selector sees REF_EST only; scoring uses disjoint REF_EVAL",
        "per_arm_take": B["per_arm_take"],
        "arms": list(B["per_arm_stats"].keys()),
        "duplicate_rates": {a: st["exact_duplicate_rate"]
                            for a, st in B["per_arm_stats"].items()},
        "eps_list": eps_list, "eps_selection": eps_sel,
        "eps_percentile_of_reference_pairwise": pct,
        "k_budget": K_BUDGET,
        "methods": results,
        "best_overall": best,
        "best_baseline": best_base,
        "ours_vs_best_baseline_abs": round(
            results[ours]["covered"][str(eps_sel)]
            - results[best_base]["covered"][str(eps_sel)], 4),
        "ours_vs_best_baseline_rel": round(
            results[ours]["covered"][str(eps_sel)]
            / max(results[best_base]["covered"][str(eps_sel)], 1e-9), 3),
    }


def main():
    FIGDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "k_budget": K_BUDGET, "stream_k": STREAM_K, "domains": {}}
    for domain in DOMAINS:
        out["domains"][domain] = analyze(domain, rng)
    with open(FIGDIR / "summary_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote figures/summary_benchmark.json")
    for d, r in out["domains"].items():
        print(f"{d}: best={r['best_overall']}  ours-vs-best-baseline "
              f"{r['ours_vs_best_baseline_abs']:+.4f} ({r['ours_vs_best_baseline_rel']}x)")


if __name__ == "__main__":
    main()
