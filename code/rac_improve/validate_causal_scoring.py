"""
Does manipulation-scoring rank axes better than attribution-scoring?

There is a ground truth available for this question, and it does not come from
either scorer: `axis_realization.py` measures, on the rendered images, whether
an axis's commanded level is detectable at all -- by CLIP identification
against chance, and by permutation tests on pixel statistics that live outside
CLIP entirely. An axis that fails both is one the generator ignored, whatever
any scorer says about it.

So the test is a rank correlation. Score the same corpus both ways, and ask
which ranking agrees with measured realization. The old scorer's failure is
already known and specific -- it put "Copy and seriality logic" first, and that
axis is undetectable in the renders on both channels -- so the headline check
is simply where each scorer places the axes that demonstrably did nothing.

No new generation: this runs on corpora that already exist.

Usage:
    python3 validate_causal_scoring.py ../real/dalle_steer2_maxmin [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))

from calculus import score_axes_realized                      # noqa: E402
from calculus2 import (axis_realization, partial_level_vectors,  # noqa: E402
                       score_axes_realized_causal)


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else float("nan")


def load(run_dir: Path):
    recs = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()
            if l.strip()]
    img = run_dir / "images"
    recs = [r for r in recs if not (img / r["file"]).exists() or (img / r["file"]).exists()]
    emb = run_dir / "clip_emb.npy"
    if emb.exists():
        E = np.load(emb)
        recs = recs[:len(E)]
    else:                       # run still in flight: embed what is on disk
        sys.path.insert(0, str(HERE))
        from axis_realization import clip_encoder
        _, image_enc = clip_encoder()
        recs = [r for r in recs if (img / r["file"]).exists()]
        E = image_enc([img / r["file"] for r in recs])
    return [r["spec"] for r in recs], E, recs


def truth_from_audit(run_dir: Path) -> dict[str, dict]:
    """Measured realization per axis, from the pixel/CLIP audit."""
    sys.path.insert(0, str(HERE))
    from axis_realization import audit_run
    res = audit_run(run_dir)
    return {a["axis"]: a for a in res["axes"]}


def main(run_dir: Path):
    specs, E, recs = load(run_dir)
    truth = truth_from_audit(run_dir)

    old = score_axes_realized(specs, E)
    new = score_axes_realized_causal(specs, E)
    rho = axis_realization(specs, E)

    old_rank = {s.name: i for i, s in enumerate(old)}
    new_rank = {s.name: i for i, s in enumerate(new)}
    axes = [s.name for s in new if s.name in old_rank and s.name in truth]

    # ground-truth quality of an axis: does its level show in the render?
    # combine the two independent channels, each as a signed evidence score
    def gt(a: str) -> float:
        t = truth[a]
        clip_ev = (t["clip_id_lift"] - 1.0)                 # >0 beats chance
        lit_ev = t["literal_eta2"] * (1.0 if t["literal_eta2_p"] < 0.05 else 0.0)
        return clip_ev + 4.0 * lit_ev

    g = [gt(a) for a in axes]
    o = [-old_rank[a] for a in axes]        # higher = better rank
    n_ = [-new_rank[a] for a in axes]

    print(f"\n=== {run_dir.name} (n={len(specs)}) ===")
    print(f"{'axis':<40} {'old':>4} {'new':>4} {'rho':>6} {'CLIPlift':>9} "
          f"{'litη²':>7} {'p':>6}  realized?")
    for a in sorted(axes, key=lambda x: new_rank[x]):
        t = truth[a]
        real = ("yes" if (t["clip_id_p"] < 0.05 or t["literal_eta2_p"] < 0.05)
                else "NO")
        print(f"{a[:39]:<40} {old_rank[a]+1:>4} {new_rank[a]+1:>4} "
              f"{rho.get(a, 0.0):>6.3f} {t['clip_id_lift']:>9.2f} "
              f"{t['literal_eta2']:>7.3f} {t['literal_eta2_p']:>6.3f}  {real}")

    print(f"\nrank agreement with measured realization (Spearman):")
    print(f"  attribution (calculus.py)   {spearman(o, g):+.3f}")
    print(f"  manipulation (calculus2.py) {spearman(n_, g):+.3f}")

    inert = [a for a in axes
             if truth[a]["clip_id_p"] >= 0.05 and truth[a]["literal_eta2_p"] >= 0.05]
    if inert:
        print(f"\naxes with NO detectable effect in the renders ({len(inert)}):")
        for a in inert:
            print(f"  {a[:44]:<46} old rank {old_rank[a]+1}/{len(old)}"
                  f"  ->  new rank {new_rank[a]+1}/{len(new)}"
                  f"  (realization {rho.get(a, 0.0):.3f})")
        mo = float(np.mean([old_rank[a] for a in inert])) + 1
        mn = float(np.mean([new_rank[a] for a in inert])) + 1
        print(f"  mean rank of inert axes: {mo:.1f} -> {mn:.1f} "
              f"(higher is better; {len(axes)} axes total)")
    return {"run": run_dir.name, "n": len(specs),
            "spearman_attribution": spearman(o, g),
            "spearman_manipulation": spearman(n_, g),
            "inert_axes": inert,
            "old_rank": {a: old_rank[a] + 1 for a in axes},
            "new_rank": {a: new_rank[a] + 1 for a in axes},
            "realization": {a: round(rho.get(a, 0.0), 4) for a in axes}}


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]] or [RESEARCH / "real" / "dalle_steer2_maxmin"]
    out = [main(d.resolve()) for d in dirs]
    json.dump(out, open(HERE / "causal_scoring_validation.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'causal_scoring_validation.json'}")
