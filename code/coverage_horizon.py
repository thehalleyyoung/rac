"""
The coverage-horizon curve: "inability to be novel" as total_generated -> 10,000.

At log-spaced checkpoints g (generator calls consumed so far), probe the
corpus each method has built by then: draw K fresh candidates from the plain
generator and measure the best achievable min-gap, E[max_j min_i d(c_j, x_i)].
That is the novelty HEADROOM still available to the next item. Define

    inability(g) = 1 - headroom(g) / headroom(g_0),   g_0 = 5

so 0 means "as easy to be novel as at the start" and -> 1 means "the corpus
has closed over the generator's reachable space". The probe uses the PLAIN
generator for every method, so the curves measure what each corpus does to
future novelty, not how weird each method's own candidate stream is.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial.distance import cdist

from experiment_budget import gen_all
from calculus import occupied_basis
from simulate import (D, GATE_Z, HERE, K, M, RNG_SEED, MixtureWorld,
                      RunningStats, min_dist_to_history)

B = 10_000
CHECKPOINTS = np.unique(np.round(np.logspace(np.log10(5), np.log10(B), 24)).astype(int))
PROBES = 120  # fresh best-of-K probes per checkpoint (median-aggregated)


def headroom(corpus: np.ndarray, world: MixtureWorld, rng: np.random.Generator) -> float:
    """Median best-of-K min-gap achievable by a fresh COHERENT candidate.

    Junk candidates are excluded from the probes: a probe including one would
    report huge headroom, but that headroom is only reachable by leaving the
    manifold -- the T7b failure mode, not novelty. The median (vs mean)
    guards the small-n checkpoints against the same tail."""
    if len(corpus) == 0:
        return np.inf
    vals = []
    for _ in range(PROBES):
        cands, comps = world.sample(2 * K)
        legit = cands[comps < world.n_modes][:K]
        if len(legit) == 0:
            continue
        vals.append(cdist(legit, corpus).min(axis=1).max())
    return float(np.median(vals))


def track_accept_all(world, mode, seed):
    rng = np.random.default_rng(seed)
    pts, _ = gen_all(world, B, mode, rng)
    out = []
    for g in CHECKPOINTS:
        h = headroom(pts[:g], world, rng)
        out.append((int(g), int(g), h, h))
    return out


def track_online(world, seed, *, w_q, w_o, w_r, gate, maxmin=False):
    rng = np.random.default_rng(seed)
    n_steps = B // K
    hist = np.empty((n_steps, D))
    stats = RunningStats.init(D)
    out = []
    cp = set(CHECKPOINTS.tolist())
    t = 0
    for step in range(n_steps):
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
        hist[t] = cands[i]
        stats.update_after_accept(cands[i], nn[i] if np.isfinite(nn[i]) else 0.0)
        t += 1
        g = (step + 1) * K
        for c in sorted(cp):
            if g >= c:
                h = headroom(hist[:t], world, rng)
                out.append((int(c), t, h, h))  # fixed generator: self == probe
                cp.discard(c)
    return out


def track_refining(probe_world: MixtureWorld, world_seed: int, seed: int, split_every: int = 150,
                    n_split: int = 5, jitter: float = 0.55, guided: bool = False):
    """The recursive mechanism, in the simulator's terms.

    Every `split_every` accepted items, the modes carrying the most recent
    mass are SPLIT: each becomes `n_split` child modes displaced by `jitter`
    within the mode's own neighborhood, with tighter spread. That is exactly
    what asking the generator to sub-divide a saturated axis level does --
    it does not invent a new region of R^D at random, it resolves structure
    that the coarse level was averaging over. Support grows; the corpus's
    reachable space grows with it.

    Crucially the children stay near the parent (jitter is comparable to the
    parent's own radius), so this is refinement, not a random walk off the
    manifold -- the quality gate would reject the latter anyway.
    """
    rng = np.random.default_rng(seed)
    w2 = MixtureWorld.build(np.random.default_rng(world_seed))
    n_steps = B // K
    hist = np.empty((n_steps, D))
    stats = RunningStats.init(D)
    out, cp = [], set(CHECKPOINTS.tolist())
    t, n_splits = 0, 0
    for step in range(n_steps):
        cands, comps = w2.sample(K)
        judged = w2.judge(w2.quality_of(comps))
        nn = min_dist_to_history(cands, hist[:t], rng)
        centroid = stats.centroid()
        z_anchor = np.linalg.norm(cands - centroid[None, :], axis=1) / stats.r_ema
        orth = stats.orth_residual(cands)
        z_rep = np.minimum(np.where(np.isfinite(nn), nn, 0) / stats.s_ema, 2.0) / 2.0
        U = 0.40 * judged + 0.35 * orth + 0.25 * z_rep
        if stats.count > 50:
            bad = z_anchor > GATE_Z
            if not bad.all():
                U = np.where(bad, -np.inf, U)
        i = int(np.argmax(U))
        hist[t] = cands[i]
        stats.update_after_accept(cands[i], nn[i] if np.isfinite(nn[i]) else 0.0)
        t += 1

        if t % split_every == 0:
            recent = hist[max(0, t - split_every):t]
            d_to_centers = cdist(recent, w2.centers)
            hot = np.bincount(d_to_centers.argmin(axis=1), minlength=len(w2.centers))
            targets = np.argsort(hot)[::-1][:4]
            # WHERE to place the children is the whole question. `guided`
            # applies the calculus's transversality criterion: displace along
            # directions the corpus does NOT already span. Unguided refinement
            # displaces isotropically, which -- since the corpus already spans
            # its own occupied directions -- lands most children back inside
            # covered territory.
            Vocc = occupied_basis(hist[:t]) if guided else None
            new_c, new_s, new_w, new_tags, new_q = [], [], [], [], []
            for m in targets:
                for _ in range(n_split):
                    d = rng.normal(size=D)
                    if Vocc is not None and Vocc.shape[1] > 0:
                        d = d - Vocc @ (Vocc.T @ d)
                    nrm = np.linalg.norm(d)
                    if nrm < 1e-9:
                        continue
                    d /= nrm
                    new_c.append(w2.centers[m] + d * jitter * (1 + rng.random()))
                    new_s.append(w2.stds[m] * 0.7)
                    new_w.append(w2.weights[m] / n_split)
                    new_tags.append(w2.mode_tags[m])
                    new_q.append(w2.mode_quality[m])
            w2.centers = np.vstack([w2.centers, np.array(new_c)])
            w2.stds = np.concatenate([w2.stds, np.array(new_s)])
            w2.weights = np.concatenate([w2.weights * 0.75, np.array(new_w) * 3.0])
            w2.weights /= w2.weights.sum() / (1 - 0.005)
            w2.mode_tags = np.vstack([w2.mode_tags, np.array(new_tags)])
            w2.mode_quality = np.concatenate([w2.mode_quality, np.array(new_q)])
            n_splits += 1

        g = (step + 1) * K
        for c in sorted(cp):
            if g >= c:
                # Two probes, because refinement changes the question:
                #  - probe_world: the ORIGINAL support, comparable across tracks
                #  - w2: the generator's CURRENT support, which refinement grew
                out.append((int(c), t, headroom(hist[:t], probe_world, rng),
                            headroom(hist[:t], w2, rng)))
                cp.discard(c)
    print(f"  (refining: {n_splits} split rounds, "
          f"{len(w2.centers)} modes at end vs {M} at start)")
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    world = MixtureWorld.build(rng)
    tracks = {
        "naive": track_accept_all(world, "plain", RNG_SEED + 1),
        "high_temp": track_accept_all(world, "high_temp", RNG_SEED + 2),
        "random_latent": track_accept_all(world, "random_latent", RNG_SEED + 3),
        "novelty_online": track_online(world, RNG_SEED + 4, w_q=0, w_o=0, w_r=1,
                                       gate=False, maxmin=True),
        "ihd_no_ledger": track_online(world, RNG_SEED + 5, w_q=0.40, w_o=0.35,
                                      w_r=0.25, gate=True),
        "ihd_refining": track_refining(world, RNG_SEED, RNG_SEED + 6, guided=False),
        "ihd_refining_guided": track_refining(world, RNG_SEED, RNG_SEED + 7, guided=True),
    }
    payload = {}
    for name, tr in tracks.items():
        g = [r[0] for r in tr]
        n = [r[1] for r in tr]
        h = [r[2] for r in tr]
        hs = [r[3] for r in tr]
        h0, hs0 = h[0], hs[0]
        payload[name] = {"generations": g, "corpus_size": n, "headroom": h,
                         "self_headroom": hs,
                         "inability": [round(1 - x / h0, 4) for x in h],
                         "self_inability": [round(1 - x / hs0, 4) for x in hs]}
        print(f"{name:16s} headroom {h0:.2f} -> {h[-1]:.2f} "
              f"(inability {1 - h[-1] / h0:.2%}) | self {hs0:.2f} -> {hs[-1]:.2f} "
              f"(self-inability {1 - hs[-1] / hs0:.2%}) corpus {n[-1]}")
    with open(HERE / "figures" / "coverage_horizon.json", "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
