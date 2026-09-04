"""
Budget-matched comparison: every method consumes EXACTLY B = 10,000 generator
calls on the same world, and keeps whatever corpus its strategy yields.

This is the fair frame for comparing diverse-generation approaches: selection
methods buy their diversity by discarding candidates, so plotting against
accepted-corpus size hides their cost. Here the x-axis is generations burned.

Methods (existing approaches):
  naive            accept every draw                                (n = B)
  high_temp        temperature-style: wider modes, flatter weights,
                   4x junk rate; accept every draw                  (n = B)
  random_latent    idealized persona/random-seed prompting: the latent
                   mode is forced uniform (perfect controllability
                   assumed, which flatters the baseline); accept all (n = B)
  dedup_post       generate all B naively, then SemDeDup-style greedy
                   near-duplicate removal at radius tau              (n = survivors)
  fps_post         generate all B naively, then farthest-point
                   subset selection (k-center greedy; the DPP-like
                   reranking family) down to B/K items               (n = B/K)
  novelty_online   online max-min accept, K candidates per step      (n = B/K)

Ours + ablations (all online, K candidates per step, n = B/K):
  ihd_full         quality + orthogonalization + capped repulsion,
                   typicality gate, symbolic ledger
  -gate            full minus the typicality gate
  -orth            full minus the orthogonalization term (weight to others)
  -quality         full minus the judged-quality term
  -repulsion       full minus the local min-gap term
  -ledger          full minus the symbolic ledger (== spectral policy)

Every method reports: corpus size, Vendi Score, mean true quality, junk
fraction, median/min pairwise gap, mode coverage. One world, one seed per
method, same B.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial.distance import cdist

from simulate import (D, HERE, K, M, GATE_Z, LEDGER_AVOID_TOP_K, LEDGER_OVERSAMPLE,
                      LEDGER_ROUND, RNG_SEED, Ledger, MixtureWorld, RunningStats,
                      min_dist_to_history, vendi_score_linear)

B = 10_000


def final_metrics(X: np.ndarray, comps: np.ndarray, world: MixtureWorld,
                  true_q: np.ndarray) -> dict:
    n = len(X)
    if n < 2:
        return {"n": n}
    d = cdist(X, X)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    junk = comps == world.junk_id
    return {
        "n": int(n),
        "vendi": round(vendi_score_linear(X), 2),
        "mean_quality": round(float(true_q.mean()), 3),
        "junk_fraction": round(float(junk.mean()), 4),
        "median_gap": round(float(np.median(nn)), 3),
        "min_gap": round(float(nn.min()), 4),
        "modes_covered": int(len(set(comps[~junk].tolist()))),
    }


def gen_all(world: MixtureWorld, b: int, mode: str, rng: np.random.Generator):
    """Draw b candidates under one of the generation-side regimes."""
    if mode == "plain":
        pts, comps = world.sample(b)
    elif mode == "high_temp":
        u = rng.random(b)
        junk_mask = u < 4 * 0.005
        comps = np.zeros(b, dtype=int)
        idx = np.where(~junk_mask)[0]
        w_flat = np.sqrt(world.weights)
        w_flat /= w_flat.sum()
        comps[idx] = rng.choice(M, size=len(idx), p=w_flat)
        pts = np.empty((b, D))
        c = comps[idx]
        pts[idx] = world.centers[c] + rng.normal(size=(len(idx), D)) * (1.6 * world.stds[c, None])
        j = np.where(junk_mask)[0]
        if len(j):
            pts[j] = rng.normal(size=(len(j), D)) * 6.0
        comps[junk_mask] = M
    elif mode == "random_latent":
        u = rng.random(b)
        junk_mask = u < 0.005
        comps = np.zeros(b, dtype=int)
        idx = np.where(~junk_mask)[0]
        comps[idx] = rng.choice(M, size=len(idx))  # uniform over modes
        pts = np.empty((b, D))
        c = comps[idx]
        pts[idx] = world.centers[c] + rng.normal(size=(len(idx), D)) * world.stds[c, None]
        j = np.where(junk_mask)[0]
        if len(j):
            pts[j] = rng.normal(size=(len(j), D)) * 6.0
        comps[junk_mask] = M
    else:
        raise ValueError(mode)
    return pts, comps


def run_accept_all(world, mode, seed):
    rng = np.random.default_rng(seed)
    pts, comps = gen_all(world, B, mode, rng)
    q = world.quality_of(comps)
    return final_metrics(pts, comps, world, q)


def run_dedup_post(world, seed, tau=0.6):
    rng = np.random.default_rng(seed)
    pts, comps = gen_all(world, B, "plain", rng)
    q = world.quality_of(comps)
    keep_idx = []
    kept = np.empty((0, D))
    for i in range(B):
        if len(keep_idx) == 0 or cdist(pts[i:i + 1], kept).min() >= tau:
            keep_idx.append(i)
            kept = np.vstack([kept, pts[i:i + 1]])
    ki = np.array(keep_idx)
    m = final_metrics(pts[ki], comps[ki], world, q[ki])
    m["tau"] = tau
    return m


def run_fps_post(world, seed, m_out=B // K):
    rng = np.random.default_rng(seed)
    pts, comps = gen_all(world, B, "plain", rng)
    q = world.quality_of(comps)
    # greedy k-center (farthest point sampling) from the full pool
    chosen = [int(rng.integers(B))]
    dmin = cdist(pts, pts[chosen[-1]][None, :]).ravel()
    while len(chosen) < m_out:
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, cdist(pts, pts[j][None, :]).ravel())
    ci = np.array(chosen)
    return final_metrics(pts[ci], comps[ci], world, q[ci])


def run_online(world, seed, *, w_q, w_o, w_r, gate, use_ledger, maxmin=False):
    """One generic online selector: n_steps = B / (K [*oversample if ledger])."""
    rng = np.random.default_rng(seed)
    per_step = K * (LEDGER_OVERSAMPLE if use_ledger else 1)
    n_steps = B // per_step
    hist = np.empty((n_steps, D))
    comps_hist = np.empty(n_steps, dtype=int)
    q_hist = np.empty(n_steps)
    stats = RunningStats.init(D)
    ledger = None
    recent_tags: list[set] = []
    if use_ledger:
        ledger = Ledger(str(HERE / "ledger" / "budget_ledger.jsonl"))
    t = 0
    for step in range(n_steps):
        if use_ledger:
            raw, raw_comps = world.sample(per_step)
            tags = world.tags_of(raw_comps)
            overlap = np.array([len(s & ledger.avoid_set) for s in tags])
            order = np.argsort(overlap, kind="stable")[:K]
            cands, comps = raw[order], raw_comps[order]
        else:
            cands, comps = world.sample(K)
        true_q = world.quality_of(comps)
        judged = world.judge(true_q)
        nn = min_dist_to_history(cands, hist[:t], rng)
        if maxmin:
            i = int(np.argmax(nn))
        else:
            centroid = stats.centroid()
            z_anchor = np.linalg.norm(cands - centroid[None, :], axis=1) / stats.r_ema
            orth = stats.orth_residual(cands)
            z_rep = np.minimum(np.where(np.isfinite(nn), nn, 0) / stats.s_ema, 2.0) / 2.0
            U = w_q * judged + w_o * orth + w_r * z_rep
            if gate and stats.count > 50:
                bad = z_anchor > GATE_Z
                if not bad.all():
                    U = np.where(bad, -np.inf, U)
            i = int(np.argmax(U))
        x = cands[i]
        hist[t] = x
        comps_hist[t] = comps[i]
        q_hist[t] = true_q[i]
        stats.update_after_accept(x, nn[i] if np.isfinite(nn[i]) else 0.0)
        if use_ledger:
            tag_i = set(world.mode_tags[comps[i]].tolist()) if comps[i] < M else set()
            ledger.observe(tag_i)
            recent_tags.append(tag_i)
            if len(recent_tags) > LEDGER_ROUND:
                recent_tags.pop(0)
            if (t + 1) % LEDGER_ROUND == 0:
                ledger.mine_round((t + 1) // LEDGER_ROUND, recent_tags)
        t += 1
    if ledger:
        ledger.close()
    return final_metrics(hist[:t], comps_hist[:t], world, q_hist[:t])


def main():
    rng = np.random.default_rng(RNG_SEED)
    world = MixtureWorld.build(rng)
    (HERE / "ledger").mkdir(exist_ok=True)
    FULL = dict(w_q=0.40, w_o=0.35, w_r=0.25, gate=True, use_ledger=True)
    runs = {
        # existing approaches
        "naive": lambda s: run_accept_all(world, "plain", s),
        "high_temp": lambda s: run_accept_all(world, "high_temp", s),
        "random_latent": lambda s: run_accept_all(world, "random_latent", s),
        "dedup_post": lambda s: run_dedup_post(world, s),
        "fps_post": lambda s: run_fps_post(world, s),
        "novelty_online": lambda s: run_online(world, s, w_q=0, w_o=0, w_r=1,
                                               gate=False, use_ledger=False, maxmin=True),
        # ours + ablations
        "ihd_full": lambda s: run_online(world, s, **FULL),
        "ihd_no_gate": lambda s: run_online(world, s, **{**FULL, "gate": False}),
        "ihd_no_orth": lambda s: run_online(world, s, **{**FULL, "w_q": 0.55, "w_o": 0.0, "w_r": 0.45}),
        "ihd_no_quality": lambda s: run_online(world, s, **{**FULL, "w_q": 0.0, "w_o": 0.55, "w_r": 0.45}),
        "ihd_no_repulsion": lambda s: run_online(world, s, **{**FULL, "w_q": 0.5, "w_o": 0.5, "w_r": 0.0}),
        "ihd_no_ledger": lambda s: run_online(world, s, **{**FULL, "use_ledger": False}),
    }
    out = {}
    for i, (name, fn) in enumerate(runs.items()):
        out[name] = fn(RNG_SEED + 31 * i)
        print(f"{name:18s} {json.dumps(out[name])}")
    with open(HERE / "figures" / "budget_comparison.json", "w") as f:
        json.dump({"budget_generations": B, "K": K, "results": out}, f, indent=2)


if __name__ == "__main__":
    main()
