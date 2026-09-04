"""Does deduplicating an arm change where it lands on Dolly coverage?

Same reference half, same radii, same evaluated n as the published table; the
only change is that each corpus is collapsed to its distinct instructions first.
A keep-everything arm spends part of its evaluated sample on repeats that cover
nothing new, so the question is whether the ranking survives removing them.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scalefree_h2h import (KS, K_MAIN, SEED, coverage_auc, coverage_density,  # noqa: E402
                           embed_cached, knn_radii, load_alpaca, load_dolly,
                           load_personahub, load_wizardlm)

FAIR = HERE / "fair_logs"
N_EVAL = 450


def fair(arm: str) -> list[str]:
    return [json.loads(l)["text"].strip() for l in
            (FAIR / f"{arm}.jsonl").read_text().splitlines() if l.strip()]


def main():
    r2 = random.Random(20260830)
    d2 = load_dolly()
    r2.shuffle(d2)
    REF = embed_cached("dolly_eval", d2[3000:6000])
    print(f"reference: STEER-disjoint half, n={len(REF)}, "
          f"median k-NN radius (k={K_MAIN})={np.median(knn_radii(REF, K_MAIN)):.4f}\n")

    sources = {
        "ours-v4-covretr": lambda: fair("ours_v4"),
        "ours-keepall": lambda: fair("ours_keepall"),
        "ours-v3-orthcond": lambda: fair("ours_v3"),
        "ours-selective": lambda: fair("ours"),
        "self-instruct (reimpl)": lambda: fair("self_instruct"),
        "random-seed": lambda: fair("random_seed"),
        "Alpaca (52k)": load_alpaca,
        "PersonaHub (50k)": load_personahub,
        "WizardLM (143k)": load_wizardlm,
    }
    print(f"{'corpus':26s} {'mode':7s} {'n_avail':>7} {'n_eval':>6} "
          f"{'AUC':>7} {'prec':>6}")
    out = {}
    for name, loader in sources.items():
        raw = [t for t in loader() if t.strip()]
        for mode in ("as-is", "dedup"):
            texts = raw if mode == "as-is" else list(dict.fromkeys(raw))
            rng = random.Random(SEED)
            samp = rng.sample(texts, min(N_EVAL, len(texts)))
            G = embed_cached(f"resc_{name}_{mode}".replace(" ", "_"), samp)
            auc = coverage_auc(REF, G, KS)
            prec = coverage_density(REF, G, K_MAIN)["precision"]
            out.setdefault(name, {})[mode] = {"auc": auc, "precision": prec,
                                              "n_available": len(texts)}
            print(f"{name:26s} {mode:7s} {len(texts):7d} {len(samp):6d} "
                  f"{auc:7.4f} {prec:6.3f}")
    json.dump(out, open(HERE / "figures" / "summary_dedup_rescore.json", "w"), indent=2)
    print("\nwrote figures/summary_dedup_rescore.json")


if __name__ == "__main__":
    main()
