"""
Which generation mode earns the coverage?

The covering arm is radius-adaptive: when a target sits in a dense region of the
reference its own k-NN radius is tight, and the prompt asks the generator to
imitate the local task family rather than to diversify, because diversification
pressure would push the item outside the small ball it has to land in. When the
radius is wide the prompt conditions on axes and shows the nearest own outputs
as negatives. The paper describes the mechanism; it does not say how the budget
divides between the two, and the division turns out to be lopsided: in the
24,000-call run, 19,967 items (83.2%) were generated in imitate mode and 4,026
(16.8%) under axis conditioning.

That raises an attribution question the coverage number alone cannot answer. If
imitation is producing the coverage, then the headline covering result is a
result about targeted nearest-neighbour imitation rather than about recursive
axis conditioning, and the two deserve to be named separately.

The measurement here decomposes coverage by mode, at matched item counts so the
comparison is not simply "more items cover more":

  * coverage of the reference achieved by an n-item sample drawn only from
    imitate-mode items, against an n-item sample drawn only from axis-mode
    items, over several n and several k;
  * which reference points each mode reaches, and how much each mode covers
    that the other does not, which is the question that decides whether the
    modes are redundant or complementary.

Usage:
    python3 mode_attribution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
COVERAGE = RESEARCH / "coverage"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(COVERAGE))

KS = [3, 5, 10]
SIZES = [500, 1000, 2000, 4000]
TRIALS = 5
N_REF = 4000       # reference sample; identical for both modes, so the
                   # comparison is unaffected and the estimate stays stable
RNG = np.random.default_rng(20260901)
CACHE = HERE / "emb_cache"


def embed_par(texts: list[str], tag: str, workers: int = 8) -> np.ndarray:
    """Parallel, cached embedding.

    `pipeline.embed` issues one HTTP request per batch and waits, which is
    about ten seconds per 96 texts here -- an hour for the corpora this
    analysis needs. The batches are independent, so they can be in flight at
    once; the result is cached because this is re-run whenever the decomposition
    is refined."""
    from concurrent.futures import ThreadPoolExecutor
    from pipeline import embed
    CACHE.mkdir(exist_ok=True)
    key = CACHE / f"{tag}_{len(texts)}.npy"
    if key.exists():
        return np.load(key)
    B = 96
    chunks = [texts[i:i + B] for i in range(0, len(texts), B)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(lambda c: embed(c, batch=B), chunks))
    E = unit(np.vstack(parts))
    np.save(key, E)
    return E


def unit(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def knn_radii(X: np.ndarray, k: int) -> np.ndarray:
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, k - 1]


def coverage_of(ref: np.ndarray, gen: np.ndarray, r_ref: np.ndarray) -> float:
    D = 1.0 - ref @ gen.T
    return float((D <= r_ref[:, None]).any(axis=1).mean())


def covered_mask(ref: np.ndarray, gen: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    D = 1.0 - ref @ gen.T
    return (D <= r_ref[:, None]).any(axis=1)


def main():
    from scalefree_h2h import load_dolly

    recs = [json.loads(l) for l in
            (COVERAGE / "fair_logs" / "ours_v4_10x.jsonl").read_text().splitlines()
            if l.strip()]
    imit = [r["text"] for r in recs if r.get("spec", {}).get("_mode") == "imitate"]
    axed = [r["text"] for r in recs if r.get("spec") and "_mode" not in r["spec"]]
    print(f"corpus {len(recs)}: imitate {len(imit)} ({len(imit)/len(recs):.1%}), "
          f"axis-conditioned {len(axed)} ({len(axed)/len(recs):.1%})", flush=True)

    ref_txt = load_dolly()
    if len(ref_txt) > N_REF:
        ref_txt = [ref_txt[i] for i in RNG.choice(len(ref_txt), N_REF, replace=False)]
    ref = embed_par(ref_txt, "dolly_ref")
    print(f"reference: {len(ref)} held-out items", flush=True)

    # match the imitate pool to the axis pool: the decomposition compares at
    # equal n anyway, and embedding all 20k buys nothing
    n_pool = min(len(imit), max(len(axed), max(SIZES)))
    imit = [imit[i] for i in RNG.choice(len(imit), n_pool, replace=False)]
    Ei = embed_par(imit, "imitate")
    Ea = embed_par(axed, "axis")
    print(f"embedded: imitate {Ei.shape}, axis {Ea.shape}", flush=True)

    radii = {k: knn_radii(ref, k) for k in KS}
    out = {"n_imitate": len(imit), "n_axis": len(axed), "by_size": []}

    for n in SIZES:
        if n > min(len(Ei), len(Ea)):
            continue
        row = {"n": n}
        for k in KS:
            ci, ca = [], []
            for _ in range(TRIALS):
                ii = RNG.choice(len(Ei), n, replace=False)
                ia = RNG.choice(len(Ea), n, replace=False)
                ci.append(coverage_of(ref, Ei[ii], radii[k]))
                ca.append(coverage_of(ref, Ea[ia], radii[k]))
            row[f"k{k}_imitate"] = round(float(np.mean(ci)), 4)
            row[f"k{k}_axis"] = round(float(np.mean(ca)), 4)
        out["by_size"].append(row)
        print(f"  n={n:>5}  " + "  ".join(
            f"k={k}: imitate {row[f'k{k}_imitate']:.4f} / axis {row[f'k{k}_axis']:.4f}"
            for k in KS), flush=True)

    # complementarity at the largest matched size, k = 5
    n = out["by_size"][-1]["n"]
    ii = RNG.choice(len(Ei), n, replace=False)
    ia = RNG.choice(len(Ea), n, replace=False)
    mi = covered_mask(ref, Ei[ii], radii[5])
    ma = covered_mask(ref, Ea[ia], radii[5])
    out["complementarity"] = {
        "n_matched": n,
        "imitate_only": round(float((mi & ~ma).mean()), 4),
        "axis_only": round(float((ma & ~mi).mean()), 4),
        "both": round(float((mi & ma).mean()), 4),
        "neither": round(float((~mi & ~ma).mean()), 4),
        "union": round(float((mi | ma).mean()), 4),
    }
    c = out["complementarity"]
    print(f"\nat matched n={n}, k=5:")
    print(f"  covered by imitate only : {c['imitate_only']:.1%}")
    print(f"  covered by axis only    : {c['axis_only']:.1%}")
    print(f"  covered by both         : {c['both']:.1%}")
    print(f"  covered by neither      : {c['neither']:.1%}")
    print(f"  union                   : {c['union']:.1%}")
    json.dump(out, open(HERE / "mode_attribution.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'mode_attribution.json'}")


if __name__ == "__main__":
    main()
