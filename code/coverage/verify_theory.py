"""
Numerical verification of the theory claims in PAPER.md.

Eight checks, each writing measured numbers into figures/summary_theory.json
(checks 7-8 reuse the sibling paper's slice-structured world):

  1. SUBMODULARITY. The Monte-Carlo coverage functional
         F_hat(S) = #{p in pool : dist(p, S) <= eps}
     is itself a (finite) coverage function -- each candidate x covers the
     fixed set N_eps(x) of pool points, and F_hat(S) = |U_{x in S} N_eps(x)| --
     hence monotone submodular EXACTLY, not just in expectation. We spot-check
     the diminishing-returns inequality over thousands of random nested chains
     S subset T, x not in T:  gain_S(x) >= gain_T(x). Expected violations: 0.

  2. GREEDY vs STREAM-GREEDY vs MAXMIN vs RANDOM on one shared candidate
     ground set: covered fraction of a held-out pool at budget k. Also the
     min-gap of each solution, to exhibit the covering/packing tradeoff in
     its purest (no-generator) form.

  3. THE (1 - 1/e) GUARANTEE against EXACT optima. On instances small enough
     to enumerate (|ground| = 30, k = 5 -> 142,506 subsets), compute OPT by
     brute force and verify F(greedy) >= (1 - 1/e) OPT on every instance.
     (Nemhauser-Wolsey-Fisher applies since F_hat is monotone submodular.)

  4. MC ESTIMATION ERROR (Hoeffding/Chernoff). For fixed S, the pool estimate
     of the covered fraction is a mean of iid Bernoullis, so
         P(|F_hat - F| > t) <= 2 exp(-2 P t^2).
     We check the implied 95% band  t_95 = sqrt(ln(2/0.05) / (2P))  empirically
     over 200 independent pools at several pool sizes P.

  5. ALLOCATION RULES at known budget n = 10,000 over known cells: equal
     per-cell, proportional to cell measure, sqrt-proportional, and greedy
     marginal coverage -- evaluated at small/medium/large eps to show which
     rule wins in which radius regime.

  6. THE REACHABILITY GAP (no-inverse-oracle consequence). We have an
     embedding oracle E: text -> R^D but no E^{-1}: we cannot decode a target
     point back into text, so the (1 - 1/e) guarantee holds only relative to
     the best subset of the PROPOSAL distribution's support, not of R^D. We
     quantify the gap by running identical greedy selection over (a) an
     oracle ground set drawn from the full latent mixture, (b) a
     proposal-limited ground set drawn from the base specs only, and (c) a
     refined-proposal ground set (one spec per latent mode -- what recursive
     refinement works toward), all evaluated on one held-out full-mixture
     pool.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from simulate_coverage import World, PoolTracker, blocked_min_dist, min_gap

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
SEED = 20260830


def coverage_value(cover_sets: np.ndarray, S: list[int]) -> int:
    """F_hat(S) with cover_sets a (n_cands, pool) boolean incidence matrix."""
    if not S:
        return 0
    return int(cover_sets[S].any(axis=0).sum())


# ---------------------------------------------------------------- check 1
def check_submodularity(rng: np.random.Generator, n_trials: int = 5000) -> dict:
    world = World.build("redteam", np.random.default_rng(SEED))
    world.rng = rng
    cands, _ = world.pool_from_specs(200, world.active_spec_ids())
    pool, _ = world.pool_from_specs(5000, world.active_spec_ids())
    eps = 1.0
    cover = cdist(cands, pool) <= eps          # (200, 5000) incidence
    n = len(cands)
    violations = 0
    worst = 0.0
    mono_violations = 0
    for _ in range(n_trials):
        perm = rng.permutation(n)
        a, b = sorted(rng.integers(1, n - 1, size=2).tolist())
        Ssmall, Sbig = list(perm[:a]), list(perm[:b])
        x = int(perm[-1])
        gS = coverage_value(cover, Ssmall + [x]) - coverage_value(cover, Ssmall)
        gT = coverage_value(cover, Sbig + [x]) - coverage_value(cover, Sbig)
        if gT > gS:
            violations += 1
            worst = max(worst, gT - gS)
        if gS < 0 or gT < 0:
            mono_violations += 1
    return {"n_trials": n_trials, "eps": eps,
            "diminishing_returns_violations": violations,
            "monotonicity_violations": mono_violations,
            "worst_violation": worst}


# ---------------------------------------------------------------- check 2
def check_selection_rules(rng: np.random.Generator) -> dict:
    world = World.build("redteam", np.random.default_rng(SEED))
    world.rng = rng
    ground, _ = world.pool_from_specs(2000, world.active_spec_ids())
    pool, _ = world.pool_from_specs(20000, world.active_spec_ids())
    eval_pool, _ = world.pool_from_specs(20000, world.active_spec_ids())
    eps, k = 1.0, 300
    cover = cdist(ground, pool) <= eps

    # full (lazy) greedy over the whole ground set
    def greedy(k: int) -> list[int]:
        covered = np.zeros(cover.shape[1], dtype=bool)
        S: list[int] = []
        gains = cover.sum(axis=1).astype(float)
        for _ in range(k):
            # lazy re-evaluation
            order = np.argsort(-gains)
            best, best_g = -1, -1.0
            for i in order:
                if i in S:
                    continue
                if gains[i] <= best_g:
                    break
                g = float((cover[i] & ~covered).sum())
                gains[i] = g
                if g > best_g:
                    best, best_g = int(i), g
            S.append(best)
            covered |= cover[best]
        return S

    # stream-greedy with K=4: the ground set arrives in random order in
    # batches of 4; pick the batch member with max marginal gain (what a
    # sequential generation pipeline can actually implement)
    def stream_greedy(k: int, K: int = 4) -> list[int]:
        order = rng.permutation(len(ground))
        covered = np.zeros(cover.shape[1], dtype=bool)
        S: list[int] = []
        for j in range(k):
            batch = order[j * K:(j + 1) * K]
            g = [(cover[i] & ~covered).sum() for i in batch]
            best = int(batch[int(np.argmax(g))])
            S.append(best)
            covered |= cover[best]
        return S

    def maxmin_gonzalez(k: int) -> list[int]:
        S = [int(rng.integers(len(ground)))]
        d = np.linalg.norm(ground - ground[S[0]], axis=1)
        for _ in range(k - 1):
            i = int(np.argmax(d))
            S.append(i)
            d = np.minimum(d, np.linalg.norm(ground - ground[i], axis=1))
        return S

    def random_sel(k: int) -> list[int]:
        return rng.choice(len(ground), size=k, replace=False).tolist()

    out = {"eps": eps, "k": k, "n_ground": len(ground)}
    for name, fn in [("greedy", greedy), ("stream_greedy_K4", stream_greedy),
                     ("maxmin", maxmin_gonzalez), ("random", random_sel)]:
        S = fn(k)
        X = ground[S]
        d_eval = blocked_min_dist(eval_pool, X)
        out[name] = {
            "covered_fraction_heldout": float((d_eval <= eps).mean()),
            "covered_fraction_train": coverage_value(cover, S) / cover.shape[1],
            "min_gap": min_gap(X),
        }
    return out


# ---------------------------------------------------------------- check 3
def check_greedy_bound(rng: np.random.Generator, n_instances: int = 10) -> dict:
    ratios = []
    for inst in range(n_instances):
        world = World.build("redteam", np.random.default_rng(SEED + inst))
        world.rng = np.random.default_rng(SEED + 400 + inst)
        ground, _ = world.pool_from_specs(30, world.active_spec_ids())
        pool, _ = world.pool_from_specs(2000, world.active_spec_ids())
        eps, k = 1.2, 5
        cover = cdist(ground, pool) <= eps
        # exact OPT by enumeration of all C(30,5) subsets, vectorized-ish
        best = 0
        for comb in itertools.combinations(range(30), k):
            v = int(cover[list(comb)].any(axis=0).sum())
            if v > best:
                best = v
        # plain greedy
        covered = np.zeros(cover.shape[1], dtype=bool)
        S: list[int] = []
        for _ in range(k):
            g = [(cover[i] & ~covered).sum() if i not in S else -1 for i in range(30)]
            i = int(np.argmax(g))
            S.append(i)
            covered |= cover[i]
        greedy_v = coverage_value(cover, S)
        ratios.append(greedy_v / best if best else 1.0)
    bound = 1.0 - 1.0 / np.e
    return {"n_instances": n_instances, "bound": bound,
            "ratios": [round(r, 4) for r in ratios],
            "min_ratio": round(min(ratios), 4),
            "bound_satisfied_all": bool(min(ratios) >= bound - 1e-12)}


# ---------------------------------------------------------------- check 4
def check_mc_error(rng: np.random.Generator) -> dict:
    world = World.build("redteam", np.random.default_rng(SEED))
    world.rng = rng
    S, _ = world.pool_from_specs(300, world.active_spec_ids())
    eps = 1.0
    truth_pool, _ = world.pool_from_specs(200_000, world.active_spec_ids())
    truth = float((blocked_min_dist(truth_pool, S) <= eps).mean())
    out = {"eps": eps, "truth_pool": 200_000, "true_covered_fraction": truth,
           "delta": 0.05, "pools": {}}
    R = 200
    for P in (500, 2000, 8000):
        errs = []
        for _ in range(R):
            pool, _ = world.pool_from_specs(P, world.active_spec_ids())
            est = float((blocked_min_dist(pool, S) <= eps).mean())
            errs.append(abs(est - truth))
        errs = np.array(errs)
        t95 = float(np.sqrt(np.log(2 / 0.05) / (2 * P)))
        out["pools"][str(P)] = {
            "hoeffding_t95": round(t95, 5),
            "empirical_mean_abs_err": round(float(errs.mean()), 5),
            "empirical_q95_abs_err": round(float(np.quantile(errs, 0.95)), 5),
            "fraction_within_hoeffding_band": round(float((errs <= t95).mean()), 4),
        }
    return out


# ---------------------------------------------------------------- check 5
def check_allocation(rng: np.random.Generator) -> dict:
    """Budget n over KNOWN cells (the M latent modes, weights known): this is
    the oracle-side planning question -- if you could allocate, how should
    you? Evaluated against a held-out pool from the full mixture."""
    world = World.build("redteam", np.random.default_rng(SEED))
    world.rng = rng
    M, n = world.M, 10_000
    w = world.weights
    pool, _ = world.pool_world(50_000)
    est_pool, _ = world.pool_world(20_000)   # greedy's working pool

    def draw_cell(m: int, count: int) -> np.ndarray:
        return world.centers[m] + world.rng.normal(size=(count, D_)) * world.stds[m]

    D_ = world.centers.shape[1]

    def alloc_counts(kind: str) -> np.ndarray:
        if kind == "equal":
            p = np.full(M, 1.0 / M)
        elif kind == "proportional":
            p = w / w.sum()
        elif kind == "sqrt":
            p = np.sqrt(w); p = p / p.sum()
        else:
            raise ValueError(kind)
        c = np.floor(p * n).astype(int)
        c[np.argsort(-(p * n - c))[: n - c.sum()]] += 1
        return c

    eps_grid = [0.6, 1.0, 1.6, 2.4]
    out = {"n": n, "M": M, "eps_grid": eps_grid, "rules": {}}
    for kind in ("equal", "proportional", "sqrt"):
        counts = alloc_counts(kind)
        X = np.vstack([draw_cell(m, c) for m, c in enumerate(counts) if c > 0])
        d = blocked_min_dist(pool, X)
        out["rules"][kind] = {str(e): float((d <= e).mean()) for e in eps_grid}

    # greedy marginal coverage: each step draw one fresh candidate per cell,
    # take the argmax marginal gain on the working pool (measured at eps=1.0)
    t0 = time.time()
    tracker = PoolTracker(est_pool.copy())
    eps_sel = 1.0
    chosen = np.empty((n, D_))
    picks_per_cell = np.zeros(M, dtype=int)
    for t in range(n):
        cands = np.vstack([draw_cell(m, 1) for m in range(M)])
        g = tracker.marginal_gains(cands, eps_sel)
        m = int(np.argmax(g))
        chosen[t] = cands[m]
        picks_per_cell[m] += 1
        tracker.update(cands[m])
    d = blocked_min_dist(pool, chosen)
    out["rules"]["greedy_marginal"] = {str(e): float((d <= e).mean()) for e in eps_grid}
    out["greedy_elapsed_sec"] = round(time.time() - t0, 1)
    out["greedy_picks_top5_cells"] = sorted(picks_per_cell.tolist(), reverse=True)[:5]
    out["greedy_cells_used"] = int((picks_per_cell > 0).sum())
    # correlation between greedy's allocation and each closed-form rule
    for kind in ("equal", "proportional", "sqrt"):
        c = alloc_counts(kind).astype(float)
        g = picks_per_cell.astype(float)
        out[f"corr_greedy_vs_{kind}"] = round(float(np.corrcoef(c, g)[0, 1]), 4)
    np.savez_compressed(FIGDIR / "allocation.npz",
                        picks_per_cell=picks_per_cell, weights=w,
                        equal=alloc_counts("equal"),
                        proportional=alloc_counts("proportional"),
                        sqrt=alloc_counts("sqrt"))
    return out


# ---------------------------------------------------------------- check 6
def check_reachability_gap(rng: np.random.Generator) -> dict:
    """Same greedy selector, three proposal distributions. The spread between
    (a) and (b) is the price of not having E^{-1} plus a narrow proposal
    distribution; (c) shows recursive refinement recovering most of it."""
    from simulate_coverage import Spec
    world = World.build("redteam", np.random.default_rng(SEED))
    world.rng = rng
    eps, k, n_ground = 1.0, 300, 2000
    eval_pool, _ = world.pool_world(20000)

    def greedy_cover(ground: np.ndarray) -> float:
        cover = cdist(ground, eval_pool) <= eps
        covered = np.zeros(cover.shape[1], dtype=bool)
        S: list[int] = []
        for _ in range(k):
            g = cover[:, ~covered].sum(axis=1)
            g[S] = -1
            i = int(np.argmax(g))
            S.append(i)
            covered |= cover[i]
        # held-out evaluation (a second full-mixture pool)
        d = blocked_min_dist(heldout, ground[S])
        return float((d <= eps).mean())

    heldout, _ = world.pool_world(20000)
    oracle_ground, _ = world.pool_world(n_ground)
    base_ground, _ = world.pool_from_specs(n_ground, world.active_spec_ids())
    # refined-proposal: one base spec per mode (what refinement works toward)
    world_ref = World.build("redteam", np.random.default_rng(SEED))
    world_ref.rng = np.random.default_rng(SEED + 31)
    world_ref.specs = [Spec(mode=m, offset=None, std_scale=1.0)
                       for m in range(world_ref.M)]
    refined_ground, _ = world_ref.pool_from_specs(n_ground, list(range(world_ref.M)))

    out = {"eps": eps, "k": k, "n_ground": n_ground,
           "modes_reached_base": len({s.mode for s in world.specs if s.active}),
           "modes_total": world.M}
    out["oracle_ground_covered"] = greedy_cover(oracle_ground)
    out["proposal_limited_covered"] = greedy_cover(base_ground)
    out["refined_proposal_covered"] = greedy_cover(refined_ground)
    out["reachability_gap"] = round(
        out["oracle_ground_covered"] - out["proposal_limited_covered"], 4)
    out["gap_recovered_by_refinement"] = round(
        (out["refined_proposal_covered"] - out["proposal_limited_covered"])
        / max(out["reachability_gap"], 1e-9), 4)
    return out


# ---------------------------------------------------------------- checks 7-8
# Slice-structured world (imported from the sibling paper's
# simulate_slices.py): a spec composes level directions additively and
# confines the generator to an m-dim slice. These two checks port the
# sibling's two structural results to the COVERAGE objective.
import sys as _sys
_sys.path.insert(0, str(HERE.parent))


def check_lattice_vs_reachable_dim(rng: np.random.Generator) -> dict:
    """Two worlds with IDENTICAL lattice size (L^A specs) but different rank
    of the level-direction span. Coverage saturation is governed by
    reachable_dim, not lattice size: the number of distinct prompts you can
    write says nothing about how much there is to cover."""
    from simulate_slices import SliceWorld, D_MANIFOLD

    def build_world(low_rank: bool, seed: int) -> SliceWorld:
        r = np.random.default_rng(seed)
        w = SliceWorld.build(r)
        if low_rank:
            # force every axis's level directions into one shared 2-dim
            # subspace: the lattice still has L^A distinct specs, but they
            # all land in a thin region
            Qs, _ = np.linalg.qr(r.normal(size=(D_MANIFOLD, 2)))
            for a in range(len(w.v)):
                w.v[a] = [Qs @ r.normal(size=2) * 1.1 for _ in w.v[a]]
        return w

    out = {"n_budget": 2000, "pool": 20000}
    eps = None
    for name, low_rank in (("full_rank", False), ("low_rank", True)):
        w = build_world(low_rank, SEED + 3)
        w.rng = np.random.default_rng(SEED + 77)
        r2 = np.random.default_rng(SEED + 78)
        pool = np.vstack([w.sample(w.random_spec(r2), 10)[0] for _ in range(2000)])
        S = np.vstack([w.sample(w.random_spec(r2), 1)[0] for _ in range(2000)])
        if eps is None:  # calibrate once, on the full-rank world
            d0 = blocked_min_dist(pool[:4000], S)
            eps = float(np.median(d0))
        d = blocked_min_dist(pool, S)
        allv = np.array([x for lv in w.v for x in lv])
        out[name] = {
            "lattice_size": w.lattice_size(),
            "level_dir_rank": int(np.linalg.matrix_rank(allv, tol=1e-8)),
            "reachable_dim": w.reachable_dim(),
            "covered_fraction_own_pool": float((d <= eps).mean()),
        }
    out["eps"] = round(eps, 4)
    return out


def check_refinement_direction(rng: np.random.Generator) -> dict:
    """Naive vs transversality-guided refinement, scored by what COVERAGE
    cares about. Nearest-neighbor distance CANNOT discriminate the two (we
    measured it: a sparse corpus in a 13-dim occupied span leaves every child
    draw ~equally far from its nearest corpus point whether the child moved
    in-span or off-span), so we use span-level quantities: the fraction of
    the children's displacement energy lying OUTSIDE the corpus's occupied
    eigenspace, and the growth in the union's effective (participation-ratio)
    dimension. In-span children thicken directions the corpus already spans
    -- density; off-span children add directions -- reach. Ports the
    sibling's negative result 'naive refinement buys density, not reach' to
    the coverage objective, where reach is what grows coverable volume."""
    from simulate_slices import SliceWorld
    from calculus import occupied_basis

    def effdim(X: np.ndarray) -> float:
        Xc = X - X.mean(axis=0, keepdims=True)
        lam = np.linalg.eigvalsh(np.cov(Xc.T))
        lam = np.clip(lam, 0, None)
        return float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-12))

    out = {}
    # local  = the naive move the sibling paper measured hurting: tight
    #          children displaced a small distance from the saturated parent
    # isotropic = unguided full-length displacement
    # transverse = displacement projected off the corpus occupied span
    for mode, guided, scale in (("local", False, 0.25),
                                ("isotropic", False, 1.0),
                                ("transverse", True, 1.0)):
        w = SliceWorld.build(np.random.default_rng(SEED + 5))
        w.rng = np.random.default_rng(SEED + 91)
        r2 = np.random.default_rng(SEED + 92)
        corpus = np.vstack([w.sample(w.random_spec(r2), 1)[0] for _ in range(1500)])
        pool = np.vstack([w.sample(w.random_spec(r2), 10)[0] for _ in range(1200)])
        eps = float(np.median(blocked_min_dist(pool, corpus)))
        Vocc = occupied_basis(corpus, energy_frac=0.95)
        w.split_level(0, 0, 4, corpus_ambient=corpus, guided=guided,
                      rng=np.random.default_rng(SEED + 93))
        child_levels = list(range(4, len(w.v[0])))
        for l in child_levels:   # apply the displacement scale for "local"
            w.v[0][l] = w.v[0][0] + (w.v[0][l] - w.v[0][0]) * scale
        draws = []
        for _ in range(600):
            spec = list(w.random_spec(r2))
            spec[0] = int(r2.choice(child_levels))
            draws.append(w.sample(tuple(spec), 1)[0])
        draws = np.vstack(draws)
        d_old = blocked_min_dist(draws, corpus)
        # displacement of child level directions vs parent, in ambient coords
        disp = np.array([(w.Qd @ (w.v[0][l] - w.v[0][0])) for l in child_levels])
        proj = disp @ Vocc
        offspan = 1.0 - (proj ** 2).sum(axis=1) / (disp ** 2).sum(axis=1)
        out[mode] = {
            "eps_coverage_radius": round(eps, 4),
            "occupied_span_dim_95pct": int(Vocc.shape[1]),
            "child_offspan_energy_frac": round(float(offspan.mean()), 4),
            "already_covered_by_old_corpus_frac": round(float((d_old <= eps).mean()), 4),
            "effdim_corpus": round(effdim(corpus), 2),
            "effdim_corpus_plus_children": round(effdim(np.vstack([corpus, draws])), 2),
        }
        out[mode]["effdim_gain"] = round(
            out[mode]["effdim_corpus_plus_children"] - out[mode]["effdim_corpus"], 3)
    out["offspan_ratio_transverse_vs_isotropic"] = round(
        out["transverse"]["child_offspan_energy_frac"]
        / max(out["isotropic"]["child_offspan_energy_frac"], 1e-9), 2)
    out["note"] = ("NN-distance to corpus does NOT discriminate the modes "
                   "(both ~1.04 in a prior version of this check): density-vs-"
                   "reach is a span property, invisible to per-point distances")
    return out


def main():
    FIGDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED + 999)
    results = {}
    for name, fn in [("submodularity", check_submodularity),
                     ("selection_rules", check_selection_rules),
                     ("greedy_bound", check_greedy_bound),
                     ("mc_error", check_mc_error),
                     ("allocation", check_allocation),
                     ("reachability_gap", check_reachability_gap),
                     ("lattice_vs_reachable_dim", check_lattice_vs_reachable_dim),
                     ("refinement_direction", check_refinement_direction)]:
        print(f"[{name}] ...", flush=True)
        t0 = time.time()
        results[name] = fn(rng)
        results[name]["_elapsed_sec"] = round(time.time() - t0, 1)
        print(json.dumps(results[name], indent=2)[:1200])
    with open(FIGDIR / "summary_theory.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote figures/summary_theory.json")


if __name__ == "__main__":
    main()
