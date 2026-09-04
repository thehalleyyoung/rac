"""
Coverage at a Finite Budget -- simulation harness.

The dual problem to the sibling paper (research/infinite_horizon_diversity):
there the horizon is unbounded and the objective is a packing one (keep the
min-gap large forever); here the budget is FIXED at n = 10,000 items and the
objective is a covering one: maximize the measure of the reachable behavior
space that lies within eps of at least one chosen item,

    F(S) = mu( U_{x in S} B(x, eps) ),

estimated by Monte Carlo against a reference pool drawn from the generator's
own reachable distribution. F is monotone submodular (verified numerically in
verify_theory.py), so greedy marginal-gain selection over a candidate stream
inherits the classical (1 - 1/e) machinery.

No LLM is called here. The (generator + embedder) pair is stood in for by an
explicit mixture world so that claims about coverage, packing, junk, and
refinement can be checked exactly at n = 10,000 without API spend. The world
mirrors the sibling paper's MixtureWorld (modes with skewed Dirichlet weights,
a rare diffuse junk component, latent per-mode quality, a noisy judge) with
one addition that this paper needs: an explicit SPEC layer between policy and
world. A spec is the policy-visible handle (the analogue of an elicited
axis-level combination); which latent mode a spec actually lands on is
oracle-only. Refinement operates on specs, never on modes, so the policy can
never "cheat" by reading ground truth.

Two domains are instantiated:
  redteam  -- red-team/safety evaluation prompt suites: many attack families
              (M = 60), full-dimensional mode scatter, skewed family weights.
  persona  -- synthetic user-persona corpora for dialogue-system testing:
              fewer archetypes (M = 40) on a low-intrinsic-dimension manifold
              (d_int = 8 inside D = 32), the way real persona text embeds.

Policies compared at n = 10,000 (K = 4 candidates per step except naive):
  naive            accept every first draw (iid from the reachable distribution)
  maxmin           packing: argmax min-distance to the accepted set
                   (the sibling paper's objective, run under a finite budget)
  coverage         greedy marginal eps-ball gain on an estimation pool
  gated_coverage   same + judge-quality and typicality gates
  gated_coverage_refine
                   gated coverage + recursive spec refinement: when a spec's
                   marginal gains saturate, it is split into finer child specs
                   and (sometimes) a genuinely new spec is minted -- the
                   simulation analogue of asking the generator to split a
                   saturated axis level / mint a cell-local axis. Refinement
                   EXPANDS the reachable distribution, so the estimation pool
                   is refreshed with draws from the new specs (the "honest
                   denominator" mechanism discussed in the paper).
  gated_coverage_refine_calc
                   same, plus CALCULUS-GUIDED conditioning: instead of
                   sampling specs uniformly, each spec carries a running
                   estimate of the expected marginal eps-ball gain of
                   conditioning on it (initialized from a small probe draw at
                   spec creation, updated from observed candidate gains), and
                   the K specs per step are drawn proportionally to that
                   promise (with an exploration floor). This is the coverage
                   adaptation of the sibling paper's generation calculus
                   (../infinite_horizon_diversity/calculus.py): under a
                   submodular coverage objective, the four-factor product
                   (spread x transversality x independence x headroom)
                   collapses into ONE estimable scalar -- expected marginal
                   coverage gain -- because the covered mask already accounts
                   for what the corpus spans (transversality), what other
                   axes contributed (independence, via greedy re-scoring
                   after every accept), and how much uncovered measure is
                   left (measure-weighted headroom). The ablation
                   refine-vs-refine_calc isolates what promise-guided
                   conditioning buys.

Everything is O(1) per accepted item in n: selection reads a fixed-size
estimation pool, min-distances use a bounded history subsample, and pool
covered-masks are updated incrementally by one (pool_size x D) distance
computation per accept.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"

RNG_SEED = 20260830
N = 10_000            # the finite budget
K = 4                 # candidates per step (naive uses 1)
D = 32                # embedding dimensionality
P_EST = 20_000        # estimation pool (policy-visible MC reference)
P_TRACK = 20_000      # held-out tracking pool (growth curves)
P_FINAL = 100_000     # held-out final evaluation pool
HISTORY_CAP = 1000    # bounded subsample for maxmin's min-distance queries
CURVE_EVERY = 250     # record covered-fraction curve every this many accepts

JUDGE_NOISE = 0.10
Q_GATE = 0.35         # judge-quality gate threshold
TYP_GATE_Z = 3.0      # typicality gate: reject if anchor z-score exceeds this

# refinement
REFINE_MIN_USES = 30      # a spec must be tried this often before it can saturate
REFINE_GAIN_THRESH = 0.5  # EMA of marginal gain (pool points) below this = saturated
REFINE_CHILDREN = 3
REFINE_NEW_SPEC_P = 0.5   # prob. a refinement also mints a spec on an unreached mode
REFINE_MAX_EVENTS = 150
REFRESH_PER_SPEC = 300    # est-pool draws appended per newly minted spec


# ----------------------------------------------------------------------------
# world
# ----------------------------------------------------------------------------

@dataclass
class Spec:
    """Policy-visible generation handle. `mode` and `offset` are hidden state
    the policy never reads; it addresses specs only by index."""
    mode: int
    offset: np.ndarray | None   # child specs live off-center inside their mode
    std_scale: float
    active: bool = True
    parent: int | None = None
    depth: int = 0


@dataclass
class World:
    name: str
    M: int
    centers: np.ndarray       # (M, D)
    stds: np.ndarray          # (M,)
    weights: np.ndarray       # (M,), sums to 1 (used only for iid world draws)
    mode_quality: np.ndarray  # (M,)
    junk_weight: float
    junk_std: float
    specs: list[Spec]
    n_base_specs: int
    rng: np.random.Generator

    @classmethod
    def build(cls, name: str, rng: np.random.Generator) -> "World":
        if name == "redteam":
            M, d_int, n_specs = 60, D, 25
            junk_weight, junk_std = 0.008, 3.0
            std_range = (0.10, 0.22)
        elif name == "persona":
            M, d_int, n_specs = 40, 8, 20
            junk_weight, junk_std = 0.008, 3.0
            std_range = (0.10, 0.22)
        else:
            raise ValueError(name)
        if d_int == D:
            centers = rng.normal(size=(M, D))
        else:
            # low-intrinsic-dimension structure: mode centers live on a random
            # d_int-dimensional linear patch of R^D (plus tiny ambient jitter),
            # the way embedded persona text concentrates near a low-dim manifold.
            basis, _ = np.linalg.qr(rng.normal(size=(D, d_int)))
            z = rng.normal(size=(M, d_int))
            centers = z @ basis.T + 0.02 * rng.normal(size=(M, D))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        centers *= 1.5
        stds = rng.uniform(*std_range, size=M)
        weights = rng.dirichlet(np.full(M, 0.6))
        mode_quality = rng.uniform(0.5, 0.9, size=M)
        # base specs land on a random subset of modes: the base generator does
        # NOT reach everything the latent space contains -- refinement is the
        # only way to grow reach, exactly as in the live pipeline.
        base_modes = rng.choice(M, size=n_specs, replace=False)
        specs = [Spec(mode=int(m), offset=None, std_scale=1.0) for m in base_modes]
        return cls(name=name, M=M, centers=centers, stds=stds, weights=weights,
                   mode_quality=mode_quality, junk_weight=junk_weight,
                   junk_std=junk_std, specs=specs, n_base_specs=n_specs, rng=rng)

    # -- sampling ------------------------------------------------------------

    def sample_spec(self, spec_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Draw one item per spec id. Returns (points, component_id) with
        component M meaning junk. Junk happens regardless of spec: the
        generator sometimes just produces off-manifold text."""
        n = len(spec_ids)
        pts = np.empty((n, D))
        comp = np.empty(n, dtype=int)
        junk = self.rng.random(n) < self.junk_weight
        for i, sid in enumerate(spec_ids):
            if junk[i]:
                pts[i] = self.rng.normal(size=D) * self.junk_std
                comp[i] = self.M
                continue
            sp = self.specs[sid]
            c = self.centers[sp.mode] + (sp.offset if sp.offset is not None else 0.0)
            pts[i] = c + self.rng.normal(size=D) * self.stds[sp.mode] * sp.std_scale
            comp[i] = sp.mode
        return pts, comp

    def pool_from_specs(self, n: int, spec_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        ids = self.rng.choice(np.array(spec_ids), size=n)
        return self.sample_spec(ids)

    def pool_world(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draws from the FULL latent space (all M modes, weighted): what the
        generator could reach under unbounded refinement. Oracle-only."""
        junk = self.rng.random(n) < self.junk_weight
        comp = self.rng.choice(self.M, size=n, p=self.weights)
        comp[junk] = self.M
        pts = np.empty((n, D))
        legit = ~junk
        c = comp[legit]
        pts[legit] = self.centers[c] + self.rng.normal(size=(legit.sum(), D)) * self.stds[c, None]
        pts[junk] = self.rng.normal(size=(junk.sum(), D)) * self.junk_std
        return pts, comp

    # -- oracle quality / judge ---------------------------------------------

    def quality_of(self, comp: np.ndarray) -> np.ndarray:
        q = np.where(comp < self.M, self.mode_quality[np.minimum(comp, self.M - 1)], 0.05)
        return q + self.rng.normal(size=len(comp)) * 0.05

    def judge(self, true_q: np.ndarray) -> np.ndarray:
        return true_q + self.rng.normal(size=len(true_q)) * JUDGE_NOISE

    # -- refinement ----------------------------------------------------------

    def refine_spec(self, sid: int) -> list[int]:
        """Split a saturated spec into REFINE_CHILDREN finer child specs
        (off-center, tighter) and possibly mint one new spec on a mode no spec
        reaches yet. Returns the ids of all newly created specs."""
        sp = self.specs[sid]
        sp.active = False
        new_ids = []
        m = sp.mode
        base = sp.offset if sp.offset is not None else np.zeros(D)
        for _ in range(REFINE_CHILDREN):
            direction = self.rng.normal(size=D)
            direction /= np.linalg.norm(direction)
            off = base + direction * 2.5 * self.stds[m] * sp.std_scale
            child = Spec(mode=m, offset=off, std_scale=sp.std_scale * 0.6,
                         parent=sid, depth=sp.depth + 1)
            self.specs.append(child)
            new_ids.append(len(self.specs) - 1)
        if self.rng.random() < REFINE_NEW_SPEC_P:
            reached = {s.mode for s in self.specs if s.active}
            unreached = [m2 for m2 in range(self.M) if m2 not in reached]
            if unreached:
                m2 = int(self.rng.choice(unreached))
                self.specs.append(Spec(mode=m2, offset=None, std_scale=1.0, depth=0))
                new_ids.append(len(self.specs) - 1)
        return new_ids

    def active_spec_ids(self) -> list[int]:
        return [i for i, s in enumerate(self.specs) if s.active]


# ----------------------------------------------------------------------------
# incremental pool machinery (all O(pool) per accept, O(1) in n)
# ----------------------------------------------------------------------------

class PoolTracker:
    """Holds a fixed reference pool and the running min-distance from each
    pool point to the accepted set. covered_fraction(eps) is then O(pool)."""

    def __init__(self, pts: np.ndarray):
        self.pts = pts
        self.min_d = np.full(len(pts), np.inf)

    def update(self, x: np.ndarray) -> None:
        d = np.linalg.norm(self.pts - x[None, :], axis=1)
        np.minimum(self.min_d, d, out=self.min_d)

    def covered_fraction(self, eps: float) -> float:
        return float((self.min_d <= eps).mean())

    def marginal_gains(self, cands: np.ndarray, eps: float) -> np.ndarray:
        """Number of pool points newly covered by each candidate: the exact
        greedy objective for the MC estimate of F."""
        uncovered = self.min_d > eps
        if not uncovered.any():
            return np.zeros(len(cands))
        d = cdist(cands, self.pts[uncovered])
        return (d <= eps).sum(axis=1).astype(float)

    def append(self, pts: np.ndarray, history: np.ndarray) -> None:
        """Pool refresh after refinement: new reference points enter with
        their min-distance to the accepted-so-far set (bounded: history is
        already capped by the caller)."""
        if len(history):
            d = cdist(pts, history).min(axis=1)
        else:
            d = np.full(len(pts), np.inf)
        self.pts = np.vstack([self.pts, pts])
        self.min_d = np.concatenate([self.min_d, d])


def blocked_min_dist(pool: np.ndarray, S: np.ndarray, block: int = 4096) -> np.ndarray:
    """min-distance from each pool point to the set S, via blocked GEMM."""
    S2 = (S ** 2).sum(axis=1)
    out = np.empty(len(pool))
    for i in range(0, len(pool), block):
        P = pool[i:i + block]
        d2 = (P ** 2).sum(axis=1)[:, None] - 2.0 * (P @ S.T) + S2[None, :]
        out[i:i + block] = np.sqrt(np.clip(d2.min(axis=1), 0.0, None))
    return out


def min_gap(S: np.ndarray, block: int = 2048) -> float:
    """min over points of distance to nearest OTHER point of S (packing radius)."""
    S2 = (S ** 2).sum(axis=1)
    best = np.inf
    for i in range(0, len(S), block):
        P = S[i:i + block]
        d2 = (P ** 2).sum(axis=1)[:, None] - 2.0 * (P @ S.T) + S2[None, :]
        for r in range(len(P)):
            d2[r, i + r] = np.inf
        best = min(best, float(np.sqrt(max(d2.min(), 0.0))))
    return best


def vendi_score_linear(X: np.ndarray) -> float:
    """Vendi Score (Friedman & Dieng 2022), linear kernel: O(n D^2 + D^3)."""
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    Mm = (Xn.T @ Xn) / Xn.shape[0]
    eigs = np.clip(np.linalg.eigvalsh(Mm), 1e-12, None)
    eigs = eigs / eigs.sum()
    return float(np.exp(-np.sum(eigs * np.log(eigs))))


# ----------------------------------------------------------------------------
# policies
# ----------------------------------------------------------------------------

@dataclass
class RunningAnchor:
    """Running centroid + typical-radius EMA for the typicality gate."""
    sum_vec: np.ndarray
    count: int = 0
    r_ema: float = 1.0
    alpha: float = 0.02

    def z_anchor(self, cands: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return np.zeros(len(cands))
        c = self.sum_vec / self.count
        return np.linalg.norm(cands - c[None, :], axis=1) / self.r_ema

    def update(self, x: np.ndarray) -> None:
        self.sum_vec += x
        self.count += 1
        r = np.linalg.norm(x - self.sum_vec / self.count)
        self.r_ema = max(r, 1e-6) if self.count == 1 else \
            (1 - self.alpha) * self.r_ema + self.alpha * max(r, 1e-6)


def run_policy(policy: str, world: World, eps_sel: float, eps_list: list[float],
               track_pool: PoolTracker, seed: int) -> dict:
    """One n = 10,000 run of one policy. `eps_sel` is the selection radius the
    greedy policies optimize; `eps_list` are the evaluation radii tracked."""
    rng = np.random.default_rng(seed)
    use_gate = policy in ("gated_coverage", "gated_coverage_refine",
                          "gated_coverage_refine_calc")
    use_cov = policy in ("coverage", "gated_coverage", "gated_coverage_refine",
                         "gated_coverage_refine_calc")
    use_refine = policy in ("gated_coverage_refine", "gated_coverage_refine_calc")
    use_calc = policy == "gated_coverage_refine_calc"
    n_cands = 1 if policy == "naive" else K

    # policy-visible estimation pool: cheap draws from the CURRENT reachable
    # distribution (base specs at start; refreshed on refinement)
    est_pts, _ = world.pool_from_specs(P_EST, world.active_spec_ids())
    est = PoolTracker(est_pts) if use_cov else None

    history = np.empty((N, D))
    comp_hist = np.empty(N, dtype=int)
    quality_log = np.empty(N)
    gain_log = np.zeros(N)
    anchor = RunningAnchor(sum_vec=np.zeros(D))

    # per-spec saturation bookkeeping (refine policies only). For the calc
    # policy, spec_gain_ema doubles as the PROMISE estimate: the expected
    # marginal eps-ball gain of conditioning on that spec (the coverage
    # calculus's single scalar; see module docstring).
    spec_uses: dict[int, int] = {}
    spec_gain_ema: dict[int, float] = {}
    refine_events: list[dict] = []

    def probe_spec(sid: int, n_probe: int = 8) -> float:
        """Score a spec BEFORE spending selection budget on it: mean marginal
        gain of a small probe draw. Probes cost n_probe cheap generations
        (bounded per refinement event), not accepted-budget items."""
        pts, _ = world.sample_spec(np.full(n_probe, sid))
        return float(est.marginal_gains(pts, eps_sel).mean())

    if use_calc:
        for sid in world.active_spec_ids():
            spec_gain_ema[sid] = probe_spec(sid)

    curve_x: list[int] = []
    curve_y: dict[float, list[float]] = {e: [] for e in eps_list}
    vendi_x: list[int] = []
    vendi_y: list[float] = []

    t0 = time.time()
    for t in range(N):
        active = world.active_spec_ids()
        if use_calc:
            # promise-weighted conditioning with an exploration floor: sample
            # specs proportionally to their estimated expected marginal gain,
            # mixed 80/20 with uniform so no spec's estimate can starve
            p = np.array([max(spec_gain_ema.get(s, 0.0), 0.0) for s in active])
            w = p / p.sum() if p.sum() > 0 else np.full(len(active), 1 / len(active))
            w = 0.8 * w + 0.2 / len(active)
            spec_ids = rng.choice(np.array(active), size=n_cands, p=w / w.sum())
        else:
            spec_ids = rng.choice(np.array(active), size=n_cands)
        cands, comps = world.sample_spec(spec_ids)
        true_q = world.quality_of(comps)
        judged_q = world.judge(true_q)

        if policy == "naive":
            i = 0
        elif policy == "maxmin":
            h = history[:t]
            if t > HISTORY_CAP:
                ref = h[rng.choice(t, size=HISTORY_CAP, replace=False)]
            else:
                ref = h
            nn = cdist(cands, ref).min(axis=1) if t else np.full(n_cands, np.inf)
            i = int(np.argmax(nn))
        else:  # coverage family
            gains = est.marginal_gains(cands, eps_sel)
            util = gains.copy()
            if use_gate:
                ok = judged_q >= Q_GATE
                if anchor.count > 50:
                    ok &= anchor.z_anchor(cands) <= TYP_GATE_Z
                if ok.any():
                    util = np.where(ok, util, -np.inf)
                # if nothing passes, fall back to best judged quality: the gate
                # is negative supervision only, never a reason to emit nothing
                else:
                    util = judged_q
            # tie-break equal gains by judged quality if gated, else first
            i = int(np.argmax(util + (1e-6 * judged_q if use_gate else 0.0)))

            # per-spec saturation signal: EMA over CANDIDATE gains, so a spec
            # can be declared exhausted even if its candidates stop winning
            if use_refine:
                for j in range(n_cands):
                    sid = int(spec_ids[j])
                    spec_uses[sid] = spec_uses.get(sid, 0) + 1
                    g0 = spec_gain_ema.get(sid, gains[j])
                    spec_gain_ema[sid] = 0.9 * g0 + 0.1 * gains[j]

        x = cands[i]
        history[t] = x
        comp_hist[t] = comps[i]
        quality_log[t] = true_q[i]
        if use_cov:
            gain_log[t] = est.marginal_gains(x[None, :], eps_sel)[0]
            est.update(x)
        track_pool_update = track_pool.update(x)  # noqa: F841 (in-place)
        anchor.update(x)

        # refinement check
        if use_refine and len(refine_events) < REFINE_MAX_EVENTS and (t + 1) % 25 == 0:
            sat = [sid for sid in world.active_spec_ids()
                   if spec_uses.get(sid, 0) >= REFINE_MIN_USES
                   and spec_gain_ema.get(sid, np.inf) < REFINE_GAIN_THRESH]
            if sat:
                # refine the most-saturated spec only (bounded work per step)
                sid = min(sat, key=lambda s: spec_gain_ema[s])
                new_ids = world.refine_spec(sid)
                refine_events.append({"t": t + 1, "spec": sid, "new": new_ids,
                                      "gain_ema": spec_gain_ema[sid]})
                # honest-denominator mechanism: the estimation pool must follow
                # the reachable distribution, so append draws from new specs
                fresh, _ = world.pool_from_specs(REFRESH_PER_SPEC * len(new_ids), new_ids)
                hcap = history[:t + 1]
                if len(hcap) > HISTORY_CAP:
                    hcap = hcap[rng.choice(len(hcap), size=HISTORY_CAP, replace=False)]
                est.append(fresh, hcap)
                if use_calc:
                    for sid2 in new_ids:
                        spec_gain_ema[sid2] = probe_spec(sid2)

        if (t + 1) % CURVE_EVERY == 0 or t == N - 1:
            curve_x.append(t + 1)
            for e in eps_list:
                curve_y[e].append(track_pool.covered_fraction(e))
        if (t + 1) % 500 == 0 or t == N - 1:
            vendi_x.append(t + 1)
            vendi_y.append(vendi_score_linear(history[:t + 1]))

    elapsed = time.time() - t0
    junk_hit = comp_hist == world.M
    return dict(
        policy=policy, elapsed_sec=elapsed, history=history, comp_hist=comp_hist,
        quality=quality_log, junk_hit=junk_hit, gain_log=gain_log,
        curve_x=np.array(curve_x),
        curve_y={e: np.array(v) for e, v in curve_y.items()},
        vendi_x=np.array(vendi_x), vendi_y=np.array(vendi_y),
        refine_events=refine_events,
        final_spec_ids=world.active_spec_ids(),
    )


# ----------------------------------------------------------------------------
# calibration + evaluation
# ----------------------------------------------------------------------------

def calibrate_eps(world: World, rng: np.random.Generator) -> tuple[list[float], float]:
    """Self-calibrating radii: delta = median distance from a held-out
    reachable draw to its nearest neighbor among n = 10,000 iid reachable
    draws. By construction, naive iid covers ~50% of the pool at eps = delta,
    which makes the three radii [0.6, 1.0, 1.6] x delta interpretable."""
    base = world.active_spec_ids()
    ref, _ = world.pool_from_specs(N, base)
    probe, _ = world.pool_from_specs(4000, base)
    d = blocked_min_dist(probe, ref)
    delta = float(np.median(d))
    return [round(0.6 * delta, 4), round(delta, 4), round(1.6 * delta, 4)], delta


def evaluate_final(res: dict, world: World, eps_list: list[float],
                   pool_base: np.ndarray, pool_world: np.ndarray) -> dict:
    S = res["history"]
    d_base = blocked_min_dist(pool_base, S)
    d_world = blocked_min_dist(pool_world, S)
    out = {
        "covered_base": {str(e): float((d_base <= e).mean()) for e in eps_list},
        "covered_world": {str(e): float((d_world <= e).mean()) for e in eps_list},
        "mean_quality": float(res["quality"].mean()),
        "junk_fraction": float(res["junk_hit"].mean()),
        "min_gap": min_gap(S),
        "final_vendi": float(res["vendi_y"][-1]),
        "n_refine_events": len(res["refine_events"]),
        "elapsed_sec": round(res["elapsed_sec"], 1),
    }
    if res["policy"] in ("gated_coverage_refine", "gated_coverage_refine_calc"):
        # the policy's own final reachable distribution (grown denominator)
        pool_own, _ = world.pool_from_specs(P_FINAL, res["final_spec_ids"])
        d_own = blocked_min_dist(pool_own, S)
        out["covered_own_reachable"] = {str(e): float((d_own <= e).mean())
                                        for e in eps_list}
        out["n_final_specs"] = len(res["final_spec_ids"])
        out["n_modes_reached_final"] = len({world.specs[i].mode
                                            for i in res["final_spec_ids"]})
    return out


POLICIES = ["naive", "maxmin", "coverage", "gated_coverage",
            "gated_coverage_refine", "gated_coverage_refine_calc"]


def run_domain(name: str, master_seed: int) -> dict:
    print(f"=== domain: {name} ===")
    summary: dict = {"domain": name}
    arrays: dict[str, np.ndarray] = {}

    # calibrate on a pristine world copy
    world0 = World.build(name, np.random.default_rng(master_seed))
    eps_list, delta = calibrate_eps(world0, np.random.default_rng(master_seed + 1))
    eps_sel = eps_list[1]
    summary["delta_nn_median"] = delta
    summary["eps_list"] = eps_list
    summary["eps_selection"] = eps_sel
    print(f"  calibrated delta={delta:.4f}  eps={eps_list}")

    # held-out pools, fixed across policies (fresh world copy, same seed ->
    # identical mode geometry; pools drawn from the BASE reachable specs)
    wpool = World.build(name, np.random.default_rng(master_seed))
    wpool.rng = np.random.default_rng(master_seed + 7)
    pool_base, comp_base = wpool.pool_from_specs(P_FINAL, wpool.active_spec_ids())
    pool_world, _ = wpool.pool_world(P_FINAL)
    track_base, _ = wpool.pool_from_specs(P_TRACK, wpool.active_spec_ids())

    summary["policies"] = {}
    for pi, policy in enumerate(POLICIES):
        # each policy gets an identical fresh world (same geometry, same base
        # specs) and its own tracker over the SAME held-out track pool
        w = World.build(name, np.random.default_rng(master_seed))
        w.rng = np.random.default_rng(master_seed + 100 + pi)
        tracker = PoolTracker(track_base.copy())
        print(f"  running {policy} ...", flush=True)
        res = run_policy(policy, w, eps_sel, eps_list, tracker,
                         seed=master_seed + 1000 + pi)
        ev = evaluate_final(res, w, eps_list, pool_base, pool_world)
        summary["policies"][policy] = ev
        print(f"    {res['elapsed_sec']:.0f}s  cov_base@mid={ev['covered_base'][str(eps_sel)]:.3f}"
              f"  cov_world@mid={ev['covered_world'][str(eps_sel)]:.3f}"
              f"  junk={ev['junk_fraction']:.4f}  minq={ev['mean_quality']:.3f}"
              f"  min_gap={ev['min_gap']:.3f}  vendi={ev['final_vendi']:.2f}")
        arrays[f"{policy}__curve_x"] = res["curve_x"]
        for e in eps_list:
            arrays[f"{policy}__curve_y_{e}"] = res["curve_y"][e]
        arrays[f"{policy}__vendi_x"] = res["vendi_x"]
        arrays[f"{policy}__vendi_y"] = res["vendi_y"]
        arrays[f"{policy}__quality"] = res["quality"]
        arrays[f"{policy}__junk_hit"] = res["junk_hit"].astype(np.int8)
        arrays[f"{policy}__gain_log"] = res["gain_log"]
        if policy in ("gated_coverage_refine", "gated_coverage_refine_calc"):
            arrays[f"{policy}__refine_ts"] = np.array(
                [ev2["t"] for ev2 in res["refine_events"]])
            summary.setdefault("refine_events_head", {})[policy] = \
                res["refine_events"][:6]

    np.savez_compressed(FIGDIR / f"sim_{name}.npz", **arrays)
    return summary


def main():
    FIGDIR.mkdir(exist_ok=True)
    t0 = time.time()
    out = {"seed": RNG_SEED, "N": N, "K": K, "D": D,
           "P_est": P_EST, "P_final": P_FINAL,
           "domains": {}}
    for name, seed in [("redteam", RNG_SEED), ("persona", RNG_SEED + 50_000)]:
        out["domains"][name] = run_domain(name, seed)
    out["total_elapsed_sec"] = round(time.time() - t0, 1)
    with open(FIGDIR / "summary_sim.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({d: {p: v["covered_base"] for p, v in s["policies"].items()}
                      for d, s in out["domains"].items()}, indent=2))
    print(f"total {out['total_elapsed_sec']}s")


if __name__ == "__main__":
    main()
