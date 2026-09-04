"""
Infinite-Horizon Diversity -- simulation harness.

This script does NOT call any LLM. It stands in for the (generator + embedder)
pair with an explicit, inspectable stochastic process, so that the paper's
claims about *asymptotic* behavior (drift, saturation, packing limits) can be
checked exactly and reproducibly, without spending API budget or waiting on
network latency for 10,000 sequential generations.

The real pipeline (OpenRouter "openai/gpt-5.6-luna" generator + a text
embedding model) is provided in real_pipeline.py with the identical policy
code (`select_candidate`, `RunningStats`) imported from here, so switching
from simulated to live poems is a one-line change of the `propose` callback,
not a rewrite of the algorithm.

Ground-truth model of the (generator, embedder) pair
-----------------------------------------------------
Embeddings live in R^d. A real instruction-following LLM asked "write a poem"
thousands of times does not sample uniformly over meaning-space: it has a
limited repertoire of openings, conceits, and forms it disproportionately
reaches for (well documented as "mode collapse" / low lexical entropy under
repeated sampling of instruction-tuned models). We model that repertoire as a
mixture of M anisotropic Gaussians ("style modes") with skewed (Dirichlet)
weights, plus one deliberately rare, high-variance "junk" component
representing the incoherent, off-manifold text that high-temperature or
tail-seeking sampling occasionally produces. Nothing about the *policies*
below is allowed to see which component a candidate came from -- that
information is oracle-only, used exclusively for after-the-fact evaluation
plots (mode coverage, junk fraction).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent

RNG_SEED = 20260830
D = 32                # embedding dimensionality
M = 40                # number of legitimate "style modes"
N = 10_000            # corpus size (poems)
K = 8                 # candidates proposed per step (best-of-K reranking pool)
HISTORY_CAP = 800      # bounded reference subsample for min-distance queries
JUNK_WEIGHT = 0.005    # prior probability mass on the incoherent/off-manifold mode
JUNK_STD = 6.0         # stdev of the junk component (mode centers have norm ~1.5)
MODE_STD_RANGE = (0.10, 0.22)
VENDI_EVERY = 200      # recompute Vendi Score every this many accepted poems

JUDGE_NOISE = 0.10     # stdev of the observable (LLM-judge) quality estimate

# --- symbolic layer: word/theme "tags" attached to each latent style mode ---
TAG_VOCAB_SIZE = 60    # stand-in for a lexicon of recurring words/images/themes
TAGS_PER_MODE = 3
LEDGER_ROUND = 50      # mine attractors every this many accepted poems
LEDGER_AVOID_TOP_K = 10
LEDGER_OVERSAMPLE = 4  # draw this many x K raw candidates before ledger filtering
LEDGER_PATH = str(HERE / "ledger" / "repulsion_ledger.jsonl")


@dataclass
class MixtureWorld:
    """The fixed (unknown-to-the-policy) generative process poems are drawn from."""
    centers: np.ndarray          # (M, D)
    stds: np.ndarray             # (M,)
    weights: np.ndarray          # (M,) sums to 1 - JUNK_WEIGHT
    mode_tags: np.ndarray        # (M, TAGS_PER_MODE) int, indices into a size-TAG_VOCAB_SIZE lexicon
    mode_quality: np.ndarray     # (M,) latent craft level of each style mode
    rng: np.random.Generator

    @classmethod
    def build(cls, rng: np.random.Generator) -> "MixtureWorld":
        centers = rng.normal(size=(M, D))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        centers *= 1.5
        stds = rng.uniform(*MODE_STD_RANGE, size=M)
        raw = rng.dirichlet(np.full(M, 0.6))  # skewed: some modes rare, some common
        weights = raw * (1.0 - JUNK_WEIGHT)

        # A small "popular tropes" sub-vocabulary is deliberately over-represented
        # across many modes' tag sets, standing in for the cliches (moonlight,
        # silence, autumn leaves, ...) an instruction-tuned model reaches for
        # regardless of which latent style it is nominally in.
        n_popular = TAG_VOCAB_SIZE // 4
        popular = np.arange(n_popular)
        rest = np.arange(n_popular, TAG_VOCAB_SIZE)
        mode_tags = np.empty((M, TAGS_PER_MODE), dtype=int)
        for m in range(M):
            n_pop_tags = rng.integers(1, TAGS_PER_MODE + 1)  # 1..TAGS_PER_MODE popular tags
            pop_tags = rng.choice(popular, size=n_pop_tags, replace=False)
            n_rest = TAGS_PER_MODE - n_pop_tags
            rest_tags = rng.choice(rest, size=n_rest, replace=False) if n_rest else np.array([], dtype=int)
            mode_tags[m] = np.concatenate([pop_tags, rest_tags])
        mode_quality = rng.uniform(0.5, 0.9, size=M)
        return cls(centers=centers, stds=stds, weights=weights, mode_tags=mode_tags,
                   mode_quality=mode_quality, rng=rng)

    @property
    def n_modes(self) -> int:
        """Read from `centers`, not the module constant: the recursive
        refinement track (coverage_horizon.track_refining) grows a world's
        mode set at run time, and every method here must follow it."""
        return len(self.centers)

    @property
    def junk_id(self) -> int:
        return self.n_modes

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (points (n,D), component_id (n,) where n_modes == junk)."""
        u = self.rng.random(n)
        junk_mask = u < JUNK_WEIGHT
        comp = np.zeros(n, dtype=int)
        # sample legitimate-mode assignment for the non-junk draws
        non_junk_idx = np.where(~junk_mask)[0]
        if len(non_junk_idx):
            probs = self.weights / self.weights.sum()
            comp[non_junk_idx] = self.rng.choice(self.n_modes, size=len(non_junk_idx), p=probs)
        pts = np.empty((n, D))
        if len(non_junk_idx):
            c = comp[non_junk_idx]
            pts[non_junk_idx] = (
                self.centers[c] + self.rng.normal(size=(len(non_junk_idx), D)) * self.stds[c, None]
            )
        if junk_mask.any():
            pts[junk_mask] = self.rng.normal(size=(junk_mask.sum(), D)) * JUNK_STD
        comp[junk_mask] = self.junk_id
        return pts, comp

    def quality_of(self, comp: np.ndarray) -> np.ndarray:
        """True latent quality per draw (junk ~ 0.05). Policies never see this
        directly; they see a noisy JUDGE_NOISE-corrupted version, standing in
        for an LLM judge scoring a rubric."""
        q = np.where(comp < self.n_modes,
                     self.mode_quality[np.minimum(comp, self.n_modes - 1)], 0.05)
        return q + self.rng.normal(size=len(comp)) * 0.05

    def judge(self, true_q: np.ndarray) -> np.ndarray:
        return true_q + self.rng.normal(size=len(true_q)) * JUDGE_NOISE

    def tags_of(self, comp: np.ndarray) -> list[set]:
        """Symbolic tag-set for each drawn component (empty set for junk)."""
        out = []
        for c in comp:
            out.append(set(self.mode_tags[c].tolist()) if c < self.n_modes else set())
        return out


class Ledger:
    """Append-only word/theme repulsion ledger.

    Mirrors evidence/ledger's append-only convention: every mining round
    APPENDS a record of what was found; nothing is ever rewritten or deleted.
    Only a bounded top-K slice of the accumulated counts is read back out to
    steer generation, so the ledger itself is free to grow without bound
    while its influence on any single prompt stays fixed-size -- the
    symbolic-layer analogue of HISTORY_CAP on the vector side.
    """

    def __init__(self, path: str):
        self.path = path
        self.counts: dict[int, int] = {}
        self.avoid_set: set[int] = set()
        self._fh = open(path, "w")

    def observe(self, tags: set):
        for t in tags:
            self.counts[t] = self.counts.get(t, 0) + 1

    def mine_round(self, round_idx: int, window_tag_sets: list[set]):
        """`window_tag_sets`: the tag-sets of the last LEDGER_ROUND accepted poems
        (the "50 sampled" the user described) -- attractors are whichever tags
        recur across an outsized share of that recent window, not just whatever
        is globally most common, so the avoid-list tracks *current* drift.
        """
        freq: dict[int, int] = {}
        for s in window_tag_sets:
            for t in s:
                freq[t] = freq.get(t, 0) + 1
        hot = sorted(freq.items(), key=lambda kv: -kv[1])[:LEDGER_AVOID_TOP_K]
        self.avoid_set = {t for t, c in hot if c >= max(3, len(window_tag_sets) // 8)}
        record = {
            "round": round_idx,
            "window_size": len(window_tag_sets),
            "hot_tags": [{"tag": int(t), "hits_in_window": int(c)} for t, c in hot],
            "avoid_set": sorted(int(t) for t in self.avoid_set),
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def vendi_score_linear(X: np.ndarray) -> float:
    """Vendi Score (Friedman & Dieng, 2022) under the linear/cosine kernel.

    For K = X_n X_n^T / n with X_n the *unit-normalized* embeddings, the
    nonzero eigenvalues of K equal those of (X_n^T X_n)/n, a D x D matrix.
    This makes the score O(n D^2 + D^3) instead of O(n^3), independent of
    corpus size for fixed embedding dimension -- exactly the kind of
    infinite-horizon-friendly evaluation this paper argues for.
    """
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = Xn.shape[0]
    M_ = (Xn.T @ Xn) / n
    eigs = np.linalg.eigvalsh(M_)
    eigs = np.clip(eigs, 1e-12, None)
    eigs = eigs / eigs.sum()
    entropy = -np.sum(eigs * np.log(eigs))
    return float(np.exp(entropy))


@dataclass
class RunningStats:
    """Tracks the two reference scales that non-dimensionalize the objective.

    r_n  : typical radius of the corpus around its own centroid (anchor scale)
    s_n  : typical nearest-neighbor gap within the corpus (repulsion scale)
    Both are updated online from a bounded random subsample, O(1) amortized.
    """
    sum_vec: np.ndarray
    Csum: np.ndarray             # sum of x_hat x_hat^T over accepted items (unit-normalized)
    count: int = 0
    r_ema: float = 1.0
    s_ema: float = 1.0
    alpha: float = 0.02

    @classmethod
    def init(cls, d: int) -> "RunningStats":
        return cls(sum_vec=np.zeros(d), Csum=np.zeros((d, d)))

    def orth_residual(self, cands: np.ndarray, energy_frac: float = 0.8) -> np.ndarray:
        """Fraction of each candidate's (unit) energy OUTSIDE the top eigenspace
        of the corpus second-moment matrix -- the O(D^2)-per-candidate, O(1)-in-n
        continuous form of 'approximately orthogonal to everything so far'.
        Greedily maximizing it flattens the Gram spectrum, which is exactly
        what raises the (linear-kernel) Vendi Score."""
        if self.count < 10:
            return np.zeros(len(cands))
        C = self.Csum / self.count
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        cum = np.cumsum(w) / max(w.sum(), 1e-12)
        j = int(np.searchsorted(cum, energy_frac)) + 1
        Xn = cands / np.linalg.norm(cands, axis=1, keepdims=True)
        proj = Xn @ V[:, :j]
        return 1.0 - (proj ** 2).sum(axis=1)

    def centroid(self) -> np.ndarray:
        if self.count == 0:
            return self.sum_vec.copy()
        return self.sum_vec / self.count

    def update_after_accept(self, x: np.ndarray, nn_dist: float):
        self.sum_vec += x
        xh = x / max(np.linalg.norm(x), 1e-12)
        self.Csum += np.outer(xh, xh)
        self.count += 1
        r = np.linalg.norm(x - self.centroid())
        if self.count == 1:
            # nn_dist is 0/undefined for the very first item; seeding s_ema
            # from it (old code: max(0, 1e-6) = 1e-6) made z_repulsion ~1e6 x
            # too large for the first ~100 steps, which silently turned the
            # blended policy into pure novelty-seeking during warm-up.
            self.r_ema = max(r, 1e-6)
            return
        if self.count == 2:
            self.s_ema = max(nn_dist, 1e-6)
        self.r_ema = (1 - self.alpha) * self.r_ema + self.alpha * max(r, 1e-6)
        self.s_ema = (1 - self.alpha) * self.s_ema + self.alpha * max(nn_dist, 1e-6)


def min_dist_to_history(cands: np.ndarray, history: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Approximate min distance from each candidate to the corpus so far.

    Uses a bounded random reference subsample once the corpus exceeds
    HISTORY_CAP -- the same amortized-cost move an approximate nearest
    neighbor index buys you at infinite horizon, made explicit here as a
    plain random projection of the reference set rather than a tree/ANN
    index, since D=32 and HISTORY_CAP=800 make brute force fast enough that
    a fancier index would only be optics.
    """
    n = history.shape[0]
    if n == 0:
        return np.full(cands.shape[0], np.inf)
    if n > HISTORY_CAP:
        idx = rng.choice(n, size=HISTORY_CAP, replace=False)
        ref = history[idx]
    else:
        ref = history
    d = cdist(cands, ref)
    return d.min(axis=1)


GATE_Z = 2.5  # typicality gate: reject candidates further than this many r_ema from centroid


def select_candidate(policy: str, cands: np.ndarray, history: np.ndarray,
                      stats: RunningStats, rng: np.random.Generator, lam: float = 0.5,
                      q_judged: np.ndarray | None = None) -> tuple[int, float]:
    """Returns (index of chosen candidate, its raw min-distance to history)."""
    nn = min_dist_to_history(cands, history, rng)
    if policy == "naive":
        return 0, nn[0]
    if policy == "novelty_only":
        i = int(np.argmax(nn))
        return i, nn[i]
    if policy == "spectral":
        # quality + spectral orthogonality + capped local repulsion, behind a
        # typicality gate. Weights sum to 1; the cap on the repulsion z-score
        # stops the unbounded junk reward that pure min-gap maximization pays.
        centroid = stats.centroid()
        anchor = np.linalg.norm(cands - centroid[None, :], axis=1)
        z_anchor = anchor / stats.r_ema
        orth = stats.orth_residual(cands)
        z_rep = np.minimum(nn / stats.s_ema, 2.0)
        q = q_judged if q_judged is not None else np.zeros(len(cands))
        U = 0.4 * q + 0.35 * orth + 0.25 * (z_rep / 2.0)
        if stats.count > 50:
            gated = z_anchor > GATE_Z
            if not gated.all():
                U = np.where(gated, -np.inf, U)
        i = int(np.argmax(U))
        return i, nn[i]
    if policy in ("anchor_repulsion", "gated_anchor_repulsion"):
        centroid = stats.centroid()
        anchor = np.linalg.norm(cands - centroid[None, :], axis=1)
        z_anchor = anchor / stats.r_ema
        z_repulsion = nn / stats.s_ema
        J = lam * z_anchor - (1 - lam) * z_repulsion
        if policy == "gated_anchor_repulsion" and stats.count > 50:
            # Hard rejection is negative supervision only: it may say "not this
            # one", never "this one is good" -- mirroring the forbidden/
            # dispreferred line in luthier/theory/negatives.py. Gated-out
            # candidates get +inf, and if the gate rejects everything we fall
            # back to the ungated argmin rather than refusing to produce.
            gated = z_anchor > GATE_Z
            if not gated.all():
                J = np.where(gated, np.inf, J)
        i = int(np.argmin(J))
        return i, nn[i]
    raise ValueError(policy)


def run_policy(policy: str, world: MixtureWorld, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    history = np.empty((N, D))
    comp_hist = np.empty(N, dtype=int)
    stats = RunningStats.init(D)
    use_ledger = policy == "spectral_ledger"
    ledger = Ledger(LEDGER_PATH) if use_ledger else None
    recent_tag_sets: list[set] = []

    anchor_dist_log = np.empty(N)
    nn_dist_log = np.empty(N)
    quality_log = np.empty(N)
    junk_hit = np.zeros(N, dtype=bool)
    ledger_rejected_log = np.zeros(N, dtype=int)
    vendi_x, vendi_y = [], []

    base_policy = "spectral" if use_ledger else policy
    n_cands = 1 if policy == "naive" else K
    t0 = time.time()
    for t in range(N):
        if use_ledger:
            raw_cands, raw_comps = world.sample(K * LEDGER_OVERSAMPLE)
            raw_tags = world.tags_of(raw_comps)
            overlap = np.array([len(ts & ledger.avoid_set) for ts in raw_tags])
            order = np.argsort(overlap, kind="stable")  # least-overlapping first
            keep = order[:K]
            ledger_rejected_log[t] = int((overlap[order[:K]] > 0).sum())
            cands, comps = raw_cands[keep], raw_comps[keep]
        else:
            cands, comps = world.sample(n_cands)

        true_q = world.quality_of(comps)
        judged_q = world.judge(true_q)
        i, nn = select_candidate(base_policy, cands, history[:t], stats, rng, q_judged=judged_q)
        x = cands[i]
        history[t] = x
        comp_hist[t] = comps[i]
        quality_log[t] = true_q[i]
        junk_hit[t] = comps[i] == world.junk_id
        centroid_before = stats.centroid()
        anchor_dist_log[t] = np.linalg.norm(x - centroid_before) if t > 0 else 0.0
        nn_dist_log[t] = nn if np.isfinite(nn) else 0.0
        stats.update_after_accept(x, nn if np.isfinite(nn) else 0.0)

        if use_ledger:
            tags = set(world.mode_tags[comps[i]].tolist()) if comps[i] < M else set()
            ledger.observe(tags)
            recent_tag_sets.append(tags)
            if len(recent_tag_sets) > LEDGER_ROUND:
                recent_tag_sets.pop(0)
            if (t + 1) % LEDGER_ROUND == 0:
                ledger.mine_round((t + 1) // LEDGER_ROUND, recent_tag_sets)

        if (t + 1) % VENDI_EVERY == 0 or t == N - 1:
            vendi_x.append(t + 1)
            vendi_y.append(vendi_score_linear(history[: t + 1]))

    if ledger is not None:
        ledger.close()

    elapsed = time.time() - t0
    mode_ids = comp_hist.copy()
    n_modes_covered = len(set(mode_ids[mode_ids != world.junk_id].tolist()))
    junk_rate_windowed = _windowed_mean(junk_hit.astype(float), window=500)

    return dict(
        policy=policy,
        elapsed_sec=elapsed,
        anchor_dist=anchor_dist_log,
        nn_dist=nn_dist_log,
        quality=quality_log,
        junk_hit=junk_hit,
        junk_rate_windowed=junk_rate_windowed,
        ledger_rejected=ledger_rejected_log,
        vendi_x=np.array(vendi_x),
        vendi_y=np.array(vendi_y),
        n_modes_covered=n_modes_covered,
        final_junk_fraction=float(junk_hit[-2000:].mean()),
    )


def _windowed_mean(a: np.ndarray, window: int) -> np.ndarray:
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.empty_like(a)
    for i in range(len(a)):
        lo = max(0, i - window + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def main():
    (HERE / "ledger").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    world = MixtureWorld.build(rng)
    results = {}
    for policy in ["naive", "novelty_only", "anchor_repulsion",
                   "gated_anchor_repulsion", "spectral", "spectral_ledger"]:
        print(f"running {policy} ...")
        results[policy] = run_policy(policy, world, seed=RNG_SEED + hash(policy) % 1000)
        r = results[policy]
        print(
            f"  done in {r['elapsed_sec']:.1f}s | modes_covered={r['n_modes_covered']}/{M} "
            f"| final_junk_fraction(last 2000)={r['final_junk_fraction']:.4f} "
            f"| final Vendi={r['vendi_y'][-1]:.2f} "
            f"| mean_quality={r['quality'].mean():.3f}"
        )

    np.savez(
        HERE / "figures" / "results.npz",
        **{
            f"{p}__{k}": v
            for p, r in results.items()
            for k, v in r.items()
            if isinstance(v, np.ndarray)
        },
    )
    summary = {
        p: {
            "elapsed_sec": r["elapsed_sec"],
            "n_modes_covered": r["n_modes_covered"],
            "final_junk_fraction_last2000": r["final_junk_fraction"],
            "final_vendi_score": float(r["vendi_y"][-1]),
            "final_anchor_dist_last500_mean": float(r["anchor_dist"][-500:].mean()),
            "final_nn_dist_last500_mean": float(r["nn_dist"][-500:].mean()),
            "mean_true_quality": float(r["quality"].mean()),
            "mean_true_quality_last2000": float(r["quality"][-2000:].mean()),
        }
        for p, r in results.items()
    }
    with open(HERE / "figures" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
