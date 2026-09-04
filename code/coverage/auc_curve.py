"""Coverage AUC as the scaled arm grows, recomputed every 1000 items.

The question the scaled run exists to answer is whether coverage keeps
accruing with budget or saturates. Saturation would mean the extra items are
refining territory already held, which is what the downstream tail analysis
implied and what the conditional-dimension argument predicts.

Reads the append-only arm file, so it can run repeatedly while generation
continues and will simply report more checkpoints each time.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scalefree_h2h import KS, coverage_density, embed_cached, load_dolly  # noqa: E402

STEP = 1000
ARM = HERE / "fair_logs" / "ours_v4_10x.jsonl"
OUT = HERE / "figures" / "auc_curve.json"


def main():
    texts = [json.loads(l)["text"] for l in ARM.read_text().splitlines() if l.strip()]
    print(f"arm has {len(texts)} items")
    r2 = random.Random(20260830)
    d2 = load_dolly()
    r2.shuffle(d2)
    EVAL = embed_cached("dolly_eval", d2[3000:6000])

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    rows = {int(k): v for k, v in prev.get("points", {}).items()}

    marks = list(range(STEP, len(texts) + 1, STEP))
    if marks and marks[-1] != len(texts):
        marks.append(len(texts))
    E_all = embed_cached("v4_10x_all", texts)

    print(f"\n{'items':>7} {'AUC':>8} {'delta':>8} {'per 1k items':>13}")
    last = None
    for n in marks:
        if n in rows:
            auc = rows[n]
        else:
            G = E_all[:n]
            auc = float(np.mean([coverage_density(EVAL, G, k)["coverage"] for k in KS]))
            rows[n] = round(auc, 4)
        d = "" if last is None else f"{auc - last:+8.4f}"
        per = f"{auc / (n / 1000):13.4f}"
        print(f"{n:7d} {auc:8.4f} {d:>8} {per}")
        last = auc

    OUT.write_text(json.dumps(
        {"arm": "ours_v4_10x", "reference": "dolly eval half (STEER-disjoint)",
         "ks": KS, "points": {str(k): v for k, v in sorted(rows.items())}}, indent=2))
    print(f"\nwrote {OUT.name}")
    print("baseline: v4 at 2,398 items scored 0.4441 (0.4752 deduplicated); "
          "Alpaca 52k scored 0.3722")


if __name__ == "__main__":
    main()
