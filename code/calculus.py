"""
A calculus of diversity: which latent variable to condition on next, and
which of its values are most different.

The theorems in verify_slices.py say that conditioning on a prompt selects an
m-dimensional slice of a d-dimensional manifold, and that only TRANSVERSE
motion of that slice raises the reachable dimension (P4). That turns "how do
we stay diverse?" into a concrete, rankable question:

    among the latent variables we could condition on next, which one moves
    the slice most transversely to the span the corpus already occupies,
    per unit of budget?

Everything below is an estimator for that question. Each axis a with levels
L(a) gets a SEPARATION TENSOR: the embedded level descriptions, from which we
read off four quantities.

  spread(a)      mean pairwise distance between a's level embeddings.
                 How different are this axis's values from EACH OTHER? An
                 axis whose levels are near-synonyms cannot separate outputs
                 no matter how it is sampled.

  transversality(a)
                 fraction of the level-to-level variation that lies OUTSIDE
                 the corpus's occupied eigenspace. This is the P4 quantity:
                 an axis whose levels differ only along directions the corpus
                 already spans adds nothing to reachable dimension.

  independence(a)
                 1 - max canonical correlation between a's level-variation
                 subspace and every other axis's. An axis that merely restates
                 an existing axis is redundant however transverse it looks in
                 isolation.

  headroom(a)    how UNEVENLY the corpus has sampled a's levels so far, as a
                 normalized entropy deficit. An axis can be excellent and
                 already exhausted; this is the term that makes the ranking
                 change over time rather than being a fixed property.

The promise of an axis is their product-form score. High spread but zero
transversality = a distinction without a difference. High transversality but
zero headroom = a good axis already spent. The calculus is multiplicative on
purpose: an axis needs ALL of them, and any one being zero should zero the
score rather than be averaged away by the others.

Choosing VALUES, recursively. Once an axis is chosen, we do not sample its
levels uniformly -- we take the max-min subset of its level embeddings
(greedy farthest-point in level space), which is the packing problem again,
one level down. If the chosen levels are themselves saturated we recurse:
the axis's best level becomes a parent and the generator is asked to split it
(axes.AxisTree.refine), producing a child axis scored by this same calculus.
The recursion terminates when no candidate axis clears the promise floor,
which is the honest signal that this generator's reachable space is exhausted
at the current resolution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def occupied_basis(corpus: np.ndarray, energy_frac: float = 0.80) -> np.ndarray:
    """Orthonormal basis for the directions the corpus already spends its
    energy on: the top eigenvectors of the second-moment matrix carrying
    `energy_frac` of the spectrum."""
    if len(corpus) < 5:
        return np.zeros((corpus.shape[1], 0))
    Xn = _unit(corpus)
    C = (Xn.T @ Xn) / len(Xn)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    cum = np.cumsum(w) / max(w.sum(), 1e-12)
    j = int(np.searchsorted(cum, energy_frac)) + 1
    return V[:, :j]


def variation_subspace(level_embs: np.ndarray, keep: int = 3) -> np.ndarray:
    """Orthonormal basis for the directions along which an axis's levels
    differ from one another (centered PCA of the level embeddings)."""
    X = _unit(level_embs)
    X = X - X.mean(axis=0, keepdims=True)
    if len(X) < 2:
        return np.zeros((level_embs.shape[1], 0))
    U, S, _ = np.linalg.svd(X.T, full_matrices=False)
    k = min(keep, int((S > 1e-8).sum()))
    return U[:, :k]


def _principal_angles_maxcos(A: np.ndarray, Bm: np.ndarray) -> float:
    if A.shape[1] == 0 or Bm.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(A.T @ Bm, compute_uv=False)
    return float(np.clip(s.max(), 0.0, 1.0))


@dataclass
class AxisScore:
    name: str
    spread: float
    transversality: float
    independence: float
    headroom: float

    @property
    def promise(self) -> float:
        return self.spread * self.transversality * self.independence * self.headroom

    @property
    def promise_unspent(self) -> float:
        """Promise with the headroom term removed: how good this axis would be
        if it were fresh.

        Needed because headroom and quality answer different questions and the
        product conflates them when every axis is equally spent. Sampling
        levels uniformly -- which is what an unguided generator does -- drives
        every axis's headroom toward zero simultaneously, at which point the
        full product is dominated by noise in a term that is uniformly small,
        and the ranking stops being about audibility at all. `promise` remains
        the quantity that decides condition-vs-refine; this one is the quantity
        that says WHICH axis is worth conditioning on once you do."""
        return self.spread * self.transversality * self.independence

    def to_json(self) -> dict:
        return {"axis": self.name, "spread": round(self.spread, 4),
                "transversality": round(self.transversality, 4),
                "independence": round(self.independence, 4),
                "headroom": round(self.headroom, 4),
                "promise": round(self.promise, 5),
                "promise_unspent": round(self.promise_unspent, 5)}


def score_axes(axis_level_embs: dict[str, np.ndarray], corpus: np.ndarray,
               level_usage: dict[str, np.ndarray] | None = None,
               active_axis_embs: dict[str, np.ndarray] | None = None) -> list[AxisScore]:
    """Rank CANDIDATE axes by promise.

    `axis_level_embs[a]` is (n_levels, D). `active_axis_embs` are the axes
    already conditioning the corpus; independence is measured against THOSE,
    never against rival candidates. Scoring candidates against each other is
    wrong and was the estimator's first bug: two duplicate candidates each
    drove the other's independence to zero, so a pair of excellent-but-
    identical axes both scored below a useless near-synonym axis. Redundancy
    among candidates is a selection problem, handled greedily in
    `select_axis_set`, not a property of any single candidate.
    """
    Vocc = occupied_basis(corpus)
    active_subs = [variation_subspace(E) for E in (active_axis_embs or {}).values()]
    scores = []
    for a, E in axis_level_embs.items():
        En = _unit(E)
        n_lv = len(En)
        if n_lv < 2:
            continue
        d = np.linalg.norm(En[:, None, :] - En[None, :, :], axis=-1)
        spread = float(d[np.triu_indices(n_lv, 1)].mean())

        Va = variation_subspace(E)
        if Vocc.shape[1] == 0 or Va.shape[1] == 0:
            transv = 1.0
        else:
            # energy fraction of the axis's variation lying outside the
            # corpus's occupied span
            proj = Vocc.T @ Va
            transv = float(np.clip(1.0 - (proj ** 2).sum() / max(Va.shape[1], 1), 0.0, 1.0))

        indep = 1.0 - max((_principal_angles_maxcos(Va, Vb) for Vb in active_subs), default=0.0)
        indep = float(np.clip(indep, 0.0, 1.0))

        if level_usage and a in level_usage and level_usage[a].sum() > 0:
            p = level_usage[a].astype(float)
            p = p / p.sum()
            nz = p[p > 0]
            H = -(nz * np.log(nz)).sum() / np.log(n_lv)
            # unused levels are pure headroom; a perfectly even history is spent
            headroom = float(np.clip(1.0 - H, 0.0, 1.0)) + float((p == 0).mean())
            headroom = float(np.clip(headroom, 0.0, 1.0))
        else:
            headroom = 1.0

        scores.append(AxisScore(a, spread, transv, indep, headroom))
    return sorted(scores, key=lambda s: -s.promise)


def farthest_levels(level_embs: np.ndarray, k: int) -> list[int]:
    """Max-min subset of an axis's levels: the packing problem one level down.
    Seeded at the level furthest from the level-centroid, so the result is
    deterministic given the embeddings."""
    X = _unit(level_embs)
    n = len(X)
    k = min(k, n)
    centroid = X.mean(axis=0, keepdims=True)
    start = int(np.argmax(np.linalg.norm(X - centroid, axis=1)))
    chosen = [start]
    dmin = np.linalg.norm(X - X[start][None, :], axis=1)
    while len(chosen) < k:
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[j][None, :], axis=1))
    return chosen


def select_axis_set(axis_level_embs: dict[str, np.ndarray], corpus: np.ndarray,
                    k: int, level_usage: dict[str, np.ndarray] | None = None,
                    active_axis_embs: dict[str, np.ndarray] | None = None) -> list[str]:
    """Greedily pick k candidate axes, re-scoring after each pick so that a
    duplicate of an already-picked axis is penalized at the point where the
    redundancy actually exists. Submodular-style greedy: each pick conditions
    the next round's independence term."""
    active = dict(active_axis_embs or {})
    remaining = dict(axis_level_embs)
    picked: list[str] = []
    while remaining and len(picked) < k:
        ranked = score_axes(remaining, corpus, level_usage, active)
        if not ranked:
            break
        best = ranked[0].name
        picked.append(best)
        active[best] = remaining.pop(best)
    return picked


def next_condition(axis_level_embs: dict[str, np.ndarray], corpus: np.ndarray,
                   level_usage: dict[str, np.ndarray] | None = None,
                   n_levels: int = 3, promise_floor: float = 0.02,
                   active_axis_embs: dict[str, np.ndarray] | None = None) -> dict:
    """One step of the calculus: pick the axis to condition on next, and the
    most-different values of it to use. Returns a decision record; when the
    best promise falls under `promise_floor` the caller should RECURSE
    (refine an axis) rather than keep sampling -- that is the exhaustion
    signal, not an error."""
    ranked = score_axes(axis_level_embs, corpus, level_usage, active_axis_embs)
    if not ranked:
        return {"decision": "no_axes"}
    best = ranked[0]
    if best.promise < promise_floor:
        return {"decision": "refine",
                "reason": f"best promise {best.promise:.4f} < floor {promise_floor}",
                "exhausted_axis": best.name,
                "ranking": [s.to_json() for s in ranked]}
    idx = farthest_levels(axis_level_embs[best.name], n_levels)
    return {"decision": "condition", "axis": best.name, "level_indices": idx,
            "ranking": [s.to_json() for s in ranked]}



# ---------------------------------------------------------------------------
# Multimodal extension: the same calculus, computed in the REAL space
# ---------------------------------------------------------------------------
def realized_level_vectors(specs: list[dict], E: np.ndarray,
                           min_per_level: int = 2) -> dict[str, np.ndarray]:
    """Turn (spec, realized-output-embedding) pairs into per-axis "level
    vectors" measured in the space the artifact actually occupies.

    Nothing about the calculus changes when we go multimodal. `score_axes`
    asks for one vector per level of each axis and computes spread,
    transversality, independence and headroom from them; all that changes is
    where those vectors come from. In the text-only setting they are
    embeddings of the level DESCRIPTIONS -- what the level claims it will do.
    Here they are the centroid of the embeddings of everything actually
    PRODUCED under that level -- what the level did.

    That substitution is the whole multimodal extension. An axis whose level
    descriptions are far apart in text space but whose realized outputs
    coincide will score high on the text-side calculus and near zero here,
    and the difference between the two scores is exactly the proxy gap the
    paper measures elsewhere.

    Levels with fewer than `min_per_level` observations are dropped: a
    centroid of one sample is a sample, not a centroid, and would report the
    noise of a single generation as the effect of the level.
    """
    En = _unit(E)
    by_axis: dict[str, dict[str, list[int]]] = {}
    for i, sp in enumerate(specs):
        for a, lv in sp.items():
            by_axis.setdefault(a, {}).setdefault(str(lv), []).append(i)
    out: dict[str, np.ndarray] = {}
    for a, levels in by_axis.items():
        rows = [En[idx].mean(axis=0) for idx in levels.values()
                if len(idx) >= min_per_level]
        if len(rows) >= 2:
            out[a] = np.vstack(rows)
    return out


def realized_level_usage(specs: list[dict], min_per_level: int = 2
                         ) -> dict[str, np.ndarray]:
    """Usage counts aligned with `realized_level_vectors`, so the headroom
    term counts the same levels the spread term measured."""
    by_axis: dict[str, dict[str, int]] = {}
    for sp in specs:
        for a, lv in sp.items():
            by_axis.setdefault(a, {}).setdefault(str(lv), 0)
            by_axis[a][str(lv)] += 1
    return {a: np.array([c for c in lv.values() if c >= min_per_level])
            for a, lv in by_axis.items()
            if sum(1 for c in lv.values() if c >= min_per_level) >= 2}


def score_axes_realized(specs: list[dict], E: np.ndarray,
                        active_axis_embs: dict[str, np.ndarray] | None = None,
                        min_per_level: int = 2) -> list[AxisScore]:
    """The calculus of §4, evaluated in the real space.

    `E` are embeddings of the ARTIFACTS (CLAP/MERT for audio, CLIP for
    images), not of their prompts. The corpus that transversality is measured
    against is the same set of artifacts, so "directions the corpus already
    spends its energy on" means directions the SOUND or the PICTURES already
    occupy.
    """
    lv = realized_level_vectors(specs, E, min_per_level)
    if not lv:
        return []
    usage = realized_level_usage(specs, min_per_level)
    scored = score_axes(lv, E, usage, active_axis_embs)
    # rank by audibility; spent-ness is reported but does not reorder here
    return sorted(scored, key=lambda s: -s.promise_unspent)


def next_condition_realized(specs: list[dict], E: np.ndarray, n_levels: int = 3,
                            promise_floor: float = 0.02,
                            min_per_level: int = 2) -> dict:
    """Recursive orthogonalization in the real space.

    Identical control flow to `next_condition`: rank the axes, and if none
    clears the promise floor emit `refine` -- the signal that no available
    latent variable moves the ARTIFACT any more and the generator should be
    asked to subdivide one. The recursion terminates on evidence from the
    output, not from the prompt.
    """
    lv = realized_level_vectors(specs, E, min_per_level)
    if not lv:
        return {"decision": "insufficient_data",
                "reason": f"no axis has >=2 levels with >={min_per_level} samples"}
    usage = realized_level_usage(specs, min_per_level)
    ranked = sorted(score_axes(lv, E, usage, None), key=lambda s: -s.promise_unspent)
    if not ranked:
        return {"decision": "no_axes"}
    best = ranked[0]
    if max(s.promise for s in ranked) < promise_floor:
        return {"decision": "refine",
                "reason": f"best realized promise {best.promise:.4f} < floor {promise_floor}",
                "exhausted_axis": best.name,
                "ranking": [s.to_json() for s in ranked]}
    return {"decision": "condition", "axis": best.name,
            "level_indices": farthest_levels(lv[best.name], n_levels),
            "ranking": [s.to_json() for s in ranked]}


# ---------------------------------------------------------------------------
# Two objectives, two calculi
# ---------------------------------------------------------------------------
"""
Both objectives spend the SAME budget -- n generations -- and they ask for
different things with it:

    COVERAGE   "in n turns, reach as much of the space as possible"
    MAX-MIN    "in n turns, make no two items resemble each other"

Those are not two phrasings of one goal. Coverage is happy to place two items
close together if between them they reach a large region; max-min is happy to
leave most of the space empty as long as nothing collides. At n = 100 in a
space that needs 10,000 to fill, the first will cluster representatives across
many regions and the second will string points along the frontier.

The four factors above are written for MAX-MIN. The companion coverage paper
optimizes the other one, and the calculus has to change with it. The
differences are not cosmetic -- they point in opposite directions on two of
the four terms.

                    max-min / packing              coverage / covering
  the ask           no two items alike             reach as much as possible
                    in n turns                     in n turns
  objective         maximize the SMALLEST           maximize vol(union of
                    pairwise distance               eps-balls) over the
                                                    reachable measure
  governed by       the closest pair -- one bad     the bulk -- one bad
                    duplicate ruins the corpus      region costs only its
                                                    own measure
  submodular?       no (max-min is not)             yes, so greedy has a
                                                    (1 - 1/e) guarantee
  classical kin     k-center                        facility location
  spread(a)         MIN distance between level      MEAN distance, WEIGHTED
                    centroids: an axis is only      by how much measure each
                    as good as its two closest      level actually carries
                    levels, because those are
                    the ones that will collide
  headroom(a)       levels not yet USED             measure not yet COVERED:
                    (entropy deficit) -- an axis    a rare level that is
                    is spent when every level       under-sampled relative to
                    has been sampled                its share is headroom; a
                                                    common one that is fully
                                                    covered is not
  value selection   farthest_levels: max-min        coverage_levels: greedy
                    subset, the packing problem     marginal-gain subset,
                    one level down                  the covering problem one
                                                    level down
  failure mode      chases outliers; the most       ignores the frontier; will
                    novel candidate is often        happily leave the extremes
                    off-manifold junk (Thm 4)       empty if they carry little
                                                    measure

Transversality and independence are shared: both objectives need directions
the corpus does not already span, and neither benefits from an axis that
restates an active one.

The practical consequence, which both papers measure independently, is that
optimizing one does not deliver the other. k-center wins min-gap and finishes
LAST on coverage; the highest-Vendi selectors are among the worst covering
ones. An implementation that reaches for "diversity" without saying which of
these it means will get whichever one its distance function happens to encode.
"""


def coverage_levels(level_embs: np.ndarray, k: int, eps: float | None = None,
                    weights: np.ndarray | None = None) -> list[int]:
    """Greedy marginal-coverage subset of an axis's levels.

    The covering counterpart of `farthest_levels`. Instead of repeatedly
    taking the point furthest from everything chosen (which walks to the
    boundary), take the point that brings the most currently-uncovered levels
    within eps -- weighting each by how much of the corpus it represents, so
    a level carrying real mass beats an exotic one that carries none.
    """
    X = _unit(level_embs)
    n = len(X)
    k = min(k, n)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    if eps is None:
        eps = float(np.median(D[np.triu_indices(n, 1)])) if n > 1 else 1.0
    w = (np.ones(n) if weights is None else np.asarray(weights, dtype=float))
    w = w / max(w.sum(), 1e-12)
    covered = np.zeros(n, dtype=bool)
    chosen: list[int] = []
    while len(chosen) < k:
        gains = [(w[(D[i] <= eps) & ~covered].sum() if i not in chosen else -1.0)
                 for i in range(n)]
        i = int(np.argmax(gains))
        if gains[i] <= 0 and chosen:
            break
        chosen.append(i)
        covered |= D[i] <= eps
    return chosen


def score_axes_coverage(axis_level_embs: dict[str, np.ndarray], corpus: np.ndarray,
                        level_usage: dict[str, np.ndarray] | None = None,
                        active_axis_embs: dict[str, np.ndarray] | None = None
                        ) -> list[AxisScore]:
    """The calculus under the COVERING objective.

    Same four factors, two of them redefined per the table above:
      spread   -> measure-weighted MEAN separation (not the min), because
                  coverage is paid by the bulk rather than by the worst pair
      headroom -> under-coverage relative to a level's share of the measure,
                  so an under-sampled but common level reads as headroom while
                  an evenly-sampled rare one does not
    """
    Vocc = occupied_basis(corpus)
    active_subs = [variation_subspace(E) for E in (active_axis_embs or {}).values()]
    scores = []
    for a, E in axis_level_embs.items():
        En = _unit(E)
        n_lv = len(En)
        if n_lv < 2:
            continue
        cnt = (level_usage.get(a) if level_usage else None)
        w = (np.asarray(cnt, dtype=float) if cnt is not None and len(cnt) == n_lv
             else np.ones(n_lv))
        w = w / max(w.sum(), 1e-12)
        D = np.linalg.norm(En[:, None, :] - En[None, :, :], axis=-1)
        # measure-weighted mean separation: each pair counts for how much of
        # the corpus it actually stands between
        pw = np.outer(w, w)
        np.fill_diagonal(pw, 0.0)
        spread = float((D * pw).sum() / max(pw.sum(), 1e-12))

        Va = variation_subspace(E)
        if Vocc.shape[1] == 0 or Va.shape[1] == 0:
            transv = 1.0
        else:
            proj = Vocc.T @ Va
            transv = float(np.clip(1.0 - (proj ** 2).sum() / max(Va.shape[1], 1), 0.0, 1.0))

        indep = float(np.clip(
            1.0 - max((_principal_angles_maxcos(Va, Vb) for Vb in active_subs), default=0.0),
            0.0, 1.0))

        # under-coverage: total-variation distance between how often each level
        # was sampled and an equal share of the measure. Zero when the axis has
        # been sampled in proportion to what it represents.
        if cnt is not None and len(cnt) == n_lv and np.sum(cnt) > 0:
            target = np.full(n_lv, 1.0 / n_lv)
            headroom = float(np.clip(0.5 * np.abs(w - target).sum() * 2.0, 0.0, 1.0))
        else:
            headroom = 1.0
        scores.append(AxisScore(a, spread, transv, indep, headroom))
    return sorted(scores, key=lambda s: -s.promise_unspent)


# ---------------------------------------------------------------------------
# Recursive orthogonalization in a recursive latent space
# ---------------------------------------------------------------------------
def cell_path(spec: dict, axis_order: list[str], depth: int) -> tuple:
    """The candidate's address in the latent tree, to `depth` levels."""
    return tuple((a, str(spec.get(a))) for a in axis_order[:depth])


def recursive_orth_residual(cand: np.ndarray, cand_specs: list[dict],
                            corpus: np.ndarray, corpus_specs: list[dict],
                            axis_order: list[str], max_depth: int = 3,
                            energy_frac: float = 0.80, decay: float = 0.6,
                            min_cell: int = 4) -> np.ndarray:
    """Orthogonality measured at every level of the latent tree at once.

    Single-level orthogonalization asks one question: is this candidate
    pointing somewhere the corpus as a whole does not already point? That is
    the right question at depth 0 and the wrong one as soon as the axis tree
    has been refined, because a candidate can be unlike the corpus globally
    and still be the fourth near-identical member of its own cell. Global
    novelty and local novelty are different quantities, and refinement is
    precisely the operation that makes them diverge.

    So we recurse. At depth 0 the candidate is scored against the occupied
    span of the whole corpus. At depth 1 it is scored against the occupied
    span of only those corpus items sharing its value on the first axis; at
    depth 2, those sharing the first two; and so on down its own address in
    the tree. The score is a decaying-weighted sum, so being globally new
    counts for most, being new within your own neighbourhood counts for
    progressively less but never nothing.

    Cells with fewer than `min_cell` members are skipped rather than scored:
    an occupied basis estimated from two points describes those two points,
    not a region, and would make every third item in a fresh cell look
    maximally novel.
    """
    Xc = _unit(cand)
    out = np.zeros(len(Xc))
    wsum = 0.0
    for depth in range(max_depth + 1):
        w = decay ** depth
        if depth == 0:
            V = occupied_basis(corpus, energy_frac)
            r = (1.0 - ((Xc @ V) ** 2).sum(axis=1)) if V.shape[1] else np.ones(len(Xc))
            out += w * r
            wsum += w
            continue
        buckets: dict[tuple, list[int]] = {}
        for i, sp in enumerate(corpus_specs):
            buckets.setdefault(cell_path(sp, axis_order, depth), []).append(i)
        r = np.ones(len(Xc))
        touched = False
        for k, x in enumerate(Xc):
            idx = buckets.get(cell_path(cand_specs[k], axis_order, depth), [])
            if len(idx) < min_cell:
                continue
            V = occupied_basis(corpus[idx], energy_frac)
            if V.shape[1] == 0:
                continue
            r[k] = float(1.0 - ((x @ V) ** 2).sum())
            touched = True
        if touched:
            out += w * r
            wsum += w
    return out / max(wsum, 1e-12)


def recursive_axis_order(specs: list[dict], E: np.ndarray,
                         min_per_level: int = 2) -> list[str]:
    """The order in which the tree should be descended: most audible/visible
    axis first, so the deepest cells are the ones that actually matter.
    Built from the realized calculus, i.e. from what the artifacts did rather
    than from what the level names claim."""
    ranked = score_axes_realized(specs, E, min_per_level=min_per_level)
    return [s.name for s in ranked]

if __name__ == "__main__":
    # Self-test on synthetic axes with known ground truth, so the estimator's
    # behavior is checkable without any API calls.
    rng = np.random.default_rng(11)
    D = 32
    corpus = rng.normal(size=(500, D)) @ np.diag(np.linspace(3, 0.05, D))
    Vocc = occupied_basis(corpus)

    def axis_in_span(k=4, scale=1.0):
        base = rng.normal(size=D) * 0.2
        return base[None, :] + (rng.normal(size=(k, Vocc.shape[1])) @ Vocc.T) * scale

    def axis_transverse(k=4, scale=1.0):
        base = rng.normal(size=D) * 0.2
        V = rng.normal(size=(k, D))
        V -= (V @ Vocc) @ Vocc.T
        return base[None, :] + V * scale

    axes_test = {
        "redundant_in_span": axis_in_span(),
        "near_synonym_levels": axis_transverse(scale=0.02),
        "genuinely_new": axis_transverse(scale=1.0),
    }
    axes_test["duplicate_of_new"] = axes_test["genuinely_new"] + rng.normal(size=(4, D)) * 0.01

    print("-- ranking (no active axes) --")
    for s in score_axes(axes_test, corpus):
        print(json.dumps(s.to_json()))
    top = score_axes(axes_test, corpus)[0].name
    assert top in ("genuinely_new", "duplicate_of_new"), top

    print("-- greedy set of 2: the duplicate must NOT be picked second --")
    picked = select_axis_set(axes_test, corpus, k=2)
    print(picked)
    assert picked[0] in ("genuinely_new", "duplicate_of_new")
    assert not {"genuinely_new", "duplicate_of_new"} <= set(picked), \
        "greedy selected both an axis and its duplicate"

    print("-- exhaustion: everything already in-span => refine --")
    spent = {"a": axis_in_span(), "b": axis_in_span()}
    print(json.dumps(next_condition(spent, corpus)["decision"]))
    assert next_condition(spent, corpus)["decision"] == "refine"
    print("calculus self-test OK")
