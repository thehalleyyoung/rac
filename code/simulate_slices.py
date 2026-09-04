"""
The conditional-generation world: a prompt selects an m-dimensional slice.

This is the simulator the paper's central claim actually needs. `simulate.py`
draws every sample from one global mixture, so it cannot exhibit the effect
that matters: that a FIXED prompt confines the generator to a thin slice, and
that escaping the resulting horizon requires moving the prompt transversely
rather than sampling harder.

Structure
---------
Ambient R^AMBIENT. A reachable manifold of dimension d, spanned by Qd.
An axis lattice: A axes, each with L language-valued levels. Each level
(a, l) carries a latent direction v[a][l] in the d-dim coordinate system.
A spec s = (l_1, ..., l_A) composes ADDITIVELY:

    center(s) = sum_a v[a][l_a]

which is the standard bag-of-attributes model of how conditioning text
composes. Given a spec, the generator emits

    x = center(s) + F(s) z,   z ~ N(0, sigma^2 I_m),   m << d

so the conditional support is an m-dim slice through the manifold. The number
of reachable slice-centers is L^A -- combinatorial in the number of axes,
which is what makes an unbounded horizon conceivable at all -- but the
reachable DIMENSION is rank(span{v}) + m, which is what actually caps how
different two outputs can be. Those two quantities come apart, and most of
the paper lives in the gap.

Refinement splits a level into sub-levels, minting new v vectors. Whether
that helps depends entirely on where the new vectors point (theorem P4):
transverse to the corpus's occupied span raises reachable dimension; inside
it only raises density. `guided=True` uses the calculus to choose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

from calculus import occupied_basis
from simulate import HERE, vendi_score_linear

AMBIENT = 64
D_MANIFOLD = 24        # dimension of the reachable semantic manifold
M_SLICE = 3            # conditional dimension given a fixed prompt (m << d)
SIGMA = 0.16           # within-slice spread
A_AXES = 5
L_LEVELS = 4
JUNK_P = 0.004
JUNK_STD = 3.0
B = 10_000
K = 8
SEED = 20260830


@dataclass
class SliceWorld:
    Qd: np.ndarray                       # (AMBIENT, d) manifold frame
    v: list[list[np.ndarray]]            # v[a][l] in R^d
    Fslice: np.ndarray                   # (d, m) shared slice frame, in d-coords
    quality: list[list[float]]           # per (axis, level) craft contribution
    rng: np.random.Generator

    @classmethod
    def build(cls, rng):
        Qd, _ = np.linalg.qr(rng.normal(size=(AMBIENT, D_MANIFOLD)))
        v, quality = [], []
        for a in range(A_AXES):
            # each axis spans its own few directions in the d-dim space
            base = rng.normal(size=(D_MANIFOLD, 2))
            Qa, _ = np.linalg.qr(base)
            v.append([Qa @ rng.normal(size=2) * 1.1 for _ in range(L_LEVELS)])
            quality.append(list(rng.uniform(0.45, 0.9, size=L_LEVELS)))
        A0 = rng.normal(size=(D_MANIFOLD, M_SLICE))
        Fslice, _ = np.linalg.qr(A0)
        return cls(Qd=Qd, v=v, Fslice=Fslice, quality=quality, rng=rng)

    @property
    def n_axes(self) -> int:
        return len(self.v)

    def lattice_size(self) -> float:
        n = 1.0
        for lv in self.v:
            n *= len(lv)
        return n

    def reachable_dim(self) -> int:
        """rank(span of all level directions) + m -- the true ceiling on how
        different two outputs can be, regardless of lattice size."""
        allv = np.array([x for lv in self.v for x in lv])
        r = np.linalg.matrix_rank(allv, tol=1e-8)
        return int(min(D_MANIFOLD, r + M_SLICE))

    def center_d(self, spec: tuple[int, ...]) -> np.ndarray:
        c = np.zeros(D_MANIFOLD)
        for a, l in enumerate(spec):
            c += self.v[a][l]
        return c

    def spec_quality(self, spec: tuple[int, ...]) -> float:
        return float(np.mean([self.quality[a][l] for a, l in enumerate(spec)]))

    def sample(self, spec: tuple[int, ...], n: int):
        """Draw n outputs conditioned on one spec."""
        c = self.center_d(spec)
        z = self.rng.normal(size=(n, M_SLICE)) * SIGMA
        pts_d = c[None, :] + z @ self.Fslice.T
        X = pts_d @ self.Qd.T
        junk = self.rng.random(n) < JUNK_P
        if junk.any():
            X[junk] = self.rng.normal(size=(int(junk.sum()), AMBIENT)) * JUNK_STD
        q = np.full(n, self.spec_quality(spec)) + self.rng.normal(size=n) * 0.04
        q[junk] = 0.05
        return X, junk, q

    def random_spec(self, rng) -> tuple[int, ...]:
        return tuple(int(rng.integers(len(lv))) for lv in self.v)

    def split_level(self, axis: int, level: int, n_new: int,
                    corpus_ambient: np.ndarray | None, guided: bool, rng):
        """Refine one level into n_new sub-levels. The new directions are
        perturbations of the parent's; `guided` projects those perturbations
        OFF the corpus's occupied span, which is the calculus's
        transversality criterion applied at refinement time.

        The corpus arrives in AMBIENT coordinates but level directions live in
        the d-dim manifold coordinates, so the occupied basis is pulled back
        through Qd before it can be projected out."""
        Vocc = None
        if guided and corpus_ambient is not None and len(corpus_ambient) > 10:
            Vocc_amb = occupied_basis(corpus_ambient)          # (AMBIENT, k)
            if Vocc_amb.shape[1] > 0:
                Vd = self.Qd.T @ Vocc_amb                      # (d, k)
                Q, r = np.linalg.qr(Vd)
                keep = np.abs(np.diag(r)) > 1e-8
                Vocc = Q[:, keep] if keep.any() else None
        parent = self.v[axis][level]
        for _ in range(n_new):
            delta = rng.normal(size=D_MANIFOLD)
            if Vocc is not None and Vocc.shape[1] > 0:
                delta = delta - Vocc @ (Vocc.T @ delta)
            nrm = np.linalg.norm(delta)
            if nrm < 1e-9:
                continue
            delta = delta / nrm * 0.9
            self.v[axis].append(parent + delta)
            self.quality[axis].append(self.quality[axis][level])


def headroom(corpus: np.ndarray, world: SliceWorld, rng, probes: int = 60) -> float:
    """Median best-of-K gap a fresh COHERENT draw can achieve, where the draw
    is allowed to use any spec the world currently reaches."""
    if len(corpus) == 0:
        return float("inf")
    vals = []
    for _ in range(probes):
        X, junk, _ = world.sample(world.random_spec(rng), K)
        Xc = X[~junk]
        if len(Xc) == 0:
            continue
        vals.append(cdist(Xc, corpus).min(axis=1).max())
    return float(np.median(vals)) if vals else 0.0


CHECKPOINTS = np.unique(np.round(np.logspace(np.log10(5), np.log10(B), 22)).astype(int))


def run(policy: str, seed: int, *, refine_every: int = 0, guided: bool = False,
        gate: bool = True) -> dict:
    """policy in {fixed_prompt, random_spec, calculus_spec}."""
    rng = np.random.default_rng(seed)
    world = SliceWorld.build(np.random.default_rng(SEED))
    corpus, corpus_d, junk_hits, quals = [], [], [], []
    fixed = world.random_spec(rng)
    n_per_call = 1
    used_specs: list[tuple[int, ...]] = []
    track = []
    cp = set(CHECKPOINTS.tolist())
    g = 0
    r_ema, s_ema, sumv = 1.0, 1.0, np.zeros(AMBIENT)

    while g < B:
        if policy == "fixed_prompt":
            spec = fixed
        elif policy == "random_spec":
            spec = world.random_spec(rng)
        elif policy == "calculus_spec":
            # Propose a pool of specs; keep the one whose CENTER points most
            # outside the corpus's occupied span (transversality), breaking
            # ties toward under-used levels (headroom). This is the calculus
            # operating on specs rather than on axes.
            pool = [world.random_spec(rng) for _ in range(10)]
            if len(corpus) < 12:
                spec = pool[0]
            else:
                Vocc = occupied_basis(np.array(corpus))
                cs = np.array([world.center_d(s) @ world.Qd.T for s in pool])
                csn = cs / np.clip(np.linalg.norm(cs, axis=1, keepdims=True), 1e-12, None)
                if Vocc.shape[1] > 0:
                    resid = 1.0 - ((csn @ Vocc) ** 2).sum(axis=1)
                else:
                    resid = np.ones(len(pool))
                spec = pool[int(np.argmax(resid))]
        else:
            raise ValueError(policy)

        X, junk, q = world.sample(spec, K)
        g += K
        # selection within the slice: quality + gap, behind a typicality gate
        if len(corpus) == 0:
            i = 0
        else:
            C = np.array(corpus)
            nn = cdist(X, C).min(axis=1)
            centroid = sumv / len(corpus)
            anchor = np.linalg.norm(X - centroid[None, :], axis=1)
            U = 0.45 * q + 0.55 * np.minimum(nn / max(s_ema, 1e-6), 2.0) / 2.0
            if gate and len(corpus) > 40:
                bad = anchor > 2.5 * r_ema
                if not bad.all():
                    U = np.where(bad, -np.inf, U)
            i = int(np.argmax(U))
        x = X[i]
        gap = 0.0 if not corpus else float(cdist(x[None, :], np.array(corpus)).min())
        corpus.append(x)
        corpus_d.append(world.center_d(spec))
        junk_hits.append(bool(junk[i]))
        quals.append(float(q[i]))
        used_specs.append(spec)
        sumv += x
        r = float(np.linalg.norm(x - sumv / len(corpus)))
        n_c = len(corpus)
        if n_c == 1:
            r_ema = max(r, 1e-6)
        else:
            if n_c == 2:
                s_ema = max(gap, 1e-6)
            r_ema = 0.98 * r_ema + 0.02 * max(r, 1e-6)
            s_ema = 0.98 * s_ema + 0.02 * max(gap, 1e-6)

        if refine_every and n_c % refine_every == 0:
            from collections import Counter
            cnt = Counter()
            for s in used_specs[-refine_every:]:
                for a, l in enumerate(s):
                    cnt[(a, l)] += 1
            (ax, lv), _ = cnt.most_common(1)[0]
            world.split_level(ax, lv, 2, np.array(corpus), guided, rng)

        for c in sorted(cp):
            if g >= c:
                track.append({
                    "generations": int(c),
                    "n": n_c,
                    "headroom": headroom(np.array(corpus), world, rng),
                    "lattice": world.lattice_size(),
                    "reach_dim": world.reachable_dim(),
                })
                cp.discard(c)

    X = np.array(corpus)
    h0 = track[0]["headroom"]
    for r_ in track:
        r_["inability"] = round(1 - r_["headroom"] / h0, 4)
    return {
        "policy": policy, "refine_every": refine_every, "guided": guided,
        "n": len(corpus),
        "vendi": round(vendi_score_linear(X), 2),
        "mean_quality": round(float(np.mean(quals)), 3),
        "junk_fraction": round(float(np.mean(junk_hits)), 4),
        "final_reach_dim": world.reachable_dim(),
        "final_lattice": world.lattice_size(),
        "final_inability": track[-1]["inability"],
        "track": track,
    }


def main():
    runs = {
        "fixed_prompt": run("fixed_prompt", SEED + 1),
        "random_spec": run("random_spec", SEED + 2),
        "calculus_spec": run("calculus_spec", SEED + 3),
        "random_spec_refine": run("random_spec", SEED + 4, refine_every=120, guided=False),
        "calculus_spec_refine": run("calculus_spec", SEED + 5, refine_every=120, guided=False),
        "calculus_spec_refine_guided": run("calculus_spec", SEED + 6, refine_every=120, guided=True),
    }
    for k, r in runs.items():
        print(f"{k:28s} n={r['n']:5d} vendi={r['vendi']:6.2f} q={r['mean_quality']:.3f} "
              f"junk={r['junk_fraction']:.4f} reach_dim={r['final_reach_dim']:3d} "
              f"lattice={r['final_lattice']:.1e} inability={r['final_inability']:.3f}")
    with open(HERE / "figures" / "slice_world.json", "w") as f:
        json.dump(runs, f, indent=2)


if __name__ == "__main__":
    main()
