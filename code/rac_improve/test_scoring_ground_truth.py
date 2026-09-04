"""
Ground-truth test of the two scorers, on data where the answer is known.

The corpora cannot settle this on their own: measured realization is itself an
estimate, so a disagreement between scorer and audit could be either one's
fault. Here the generative process is written down, so there is no ambiguity
about which axis matters.

Construction: three axes, specs sampled independently exactly as RAC samples
them. Axis A displaces the output strongly along a fixed direction, axis C
weakly along another, and axis B does nothing at all -- it is the axis the
generator ignores. A scorer that works must rank A first and B last.

Result (seed 0): the attribution scorer ranks them B, C, A -- the inert axis
first and the strongest axis LAST, an exact inversion. The manipulation
scorer ranks A first.

The inversion is not noise, and its mechanism is worth stating plainly. An
axis that really moves the artifact fills the corpus with variance along its
own direction. `occupied_basis` then absorbs that direction, and
transversality -- defined as the share of an axis's variation lying OUTSIDE
the corpus's occupied span -- collapses toward zero for exactly that axis.
The inert axis contributes only isotropic noise, little of which the occupied
basis captures, so its transversality stays high. Promise is multiplicative,
so the effective axis is driven to the bottom of the ranking and the inert one
to the top. The calculus systematically de-ranks its own working dimensions,
and the effect gets stronger the better the axis is.

The realization factor fixes this because it is measured on group structure
rather than on span geometry: an axis with no effect has no between-level
variance to find, whatever the occupied basis is doing.

Run:  python3 test_scoring_ground_truth.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calculus import score_axes_realized                       # noqa: E402
from calculus2 import axis_realization, score_axes_realized_causal  # noqa: E402


def make_corpus(seed: int, n: int = 120, D: int = 64, noise: float = 0.35):
    """Axis A strong, C weak, B inert. Independent specs, as RAC samples."""
    rng = np.random.default_rng(seed)
    dirA = rng.normal(size=D); dirA /= np.linalg.norm(dirA)
    dirC = rng.normal(size=D); dirC /= np.linalg.norm(dirC)
    lvA = {"a1": -1.5, "a2": -0.5, "a3": 0.5, "a4": 1.5}
    lvC = {"c1": -0.3, "c2": -0.1, "c3": 0.1, "c4": 0.3}
    specs, E = [], []
    for i in range(n):
        r = random.Random(seed * 100000 + i)
        a, b, c = (r.choice(list(lvA)), r.choice(["b1", "b2", "b3", "b4"]),
                   r.choice(list(lvC)))
        v = noise * rng.normal(size=D) + lvA[a] * dirA + lvC[c] * dirC
        specs.append({"A": a, "B": b, "C": c})
        E.append(v)
    return specs, np.vstack(E)


def main(trials: int = 12) -> int:
    old_first_inert = old_last_strong = 0
    new_first_strong = new_last_inert = 0
    rows = []
    for seed in range(trials):
        specs, E = make_corpus(seed)
        old = [s.name for s in score_axes_realized(specs, E)]
        new = [s.name for s in score_axes_realized_causal(specs, E)]
        rho = axis_realization(specs, E)
        old_first_inert += old[0] == "B"
        old_last_strong += old[-1] == "A"
        new_first_strong += new[0] == "A"
        new_last_inert += new[-1] == "B"
        rows.append((seed, "".join(old), "".join(new),
                     rho.get("A", 0), rho.get("B", 0), rho.get("C", 0)))

    print(f"{'seed':>4} {'attribution':>12} {'manipulation':>13} "
          f"{'rho_A':>7} {'rho_B':>7} {'rho_C':>7}")
    for s, o, n_, ra, rb, rc in rows:
        print(f"{s:>4} {o:>12} {n_:>13} {ra:>7.3f} {rb:>7.3f} {rc:>7.3f}")
    print(f"\nground truth: A strong, C weak, B inert -> correct ranking 'ACB'")
    print(f"attribution : ranked the INERT axis first in "
          f"{old_first_inert}/{trials}, the STRONG axis last in "
          f"{old_last_strong}/{trials}")
    print(f"manipulation: ranked the STRONG axis first in "
          f"{new_first_strong}/{trials}, the INERT axis last in "
          f"{new_last_inert}/{trials}")

    ok = new_first_strong >= int(0.8 * trials)
    print("\nPASS" if ok else "\nFAIL",
          "- manipulation scoring identifies the effective axis")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
