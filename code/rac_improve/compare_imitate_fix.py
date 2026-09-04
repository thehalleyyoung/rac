"""
Did negatives and exhaustion detection reduce the imitate branch's duplication?

Two matched 1,200-call arms, identical seeds and targets, differing only in the
imitate branch:

  ours_v4_ctl   fair_fight.py       imitate prompt sees examples only
  ours_v4_fix   fair_fight_causal.py  imitate prompt also sees its own recent
                                    output as negatives, and a neighbourhood
                                    with >=8 calls and >50% duplicates is
                                    declared exhausted and handed to the
                                    conditioning branch

The published run's imitate branch duplicated on 65.4% of its calls against
0.000 for conditioning, so the quantity to compare is the imitate branch's own
duplicate rate, not the corpus rate: the fix also moves calls between branches,
and a corpus-level number would confound the two effects.

Reported per arm:
  mode mix          how the budget divided, which the exhaustion switch changes
  duplicate rate    per mode, since that is where the effect should appear
  template mass     share of the arm in its four most common six-word openings,
                    which is the failure the duplicate rate understates
  distinct yield    unique items per generator call, the thing a budget buys

Usage:
    python3 compare_imitate_fix.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "coverage" / "fair_logs"
ARMS = [("control", "ours_v4_ctl"), ("negatives + exhaustion", "ours_v4_fix")]


def load(tag: str) -> list[dict]:
    p = LOGS / f"{tag}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def stats(recs: list[dict]) -> dict:
    imit = [r["text"].strip() for r in recs
            if r.get("spec", {}).get("_mode") == "imitate"]
    axed = [r["text"].strip() for r in recs
            if r.get("spec") and "_mode" not in r["spec"]]
    allt = [r["text"].strip() for r in recs]

    def dup(xs):
        return 0.0 if not xs else 1 - len(set(xs)) / len(xs)

    def tmpl(xs):
        if not xs:
            return 0.0
        c = Counter(" ".join(x.split()[:6]).lower() for x in xs)
        return sum(v for _, v in c.most_common(4)) / len(xs)

    return {
        "n": len(recs), "n_imitate": len(imit), "n_axis": len(axed),
        "imitate_share": len(imit) / max(len(recs), 1),
        "dup_imitate": dup(imit), "dup_axis": dup(axed), "dup_all": dup(allt),
        "tmpl_imitate": tmpl(imit), "tmpl_all": tmpl(allt),
        "distinct_per_call": len(set(allt)) / max(len(recs), 1),
    }


def main():
    rows = []
    for label, tag in ARMS:
        recs = load(tag)
        if not recs:
            print(f"{label}: no log yet ({tag}.jsonl)")
            return
        rows.append((label, stats(recs)))

    w = 26
    print(f"{'':<{w}}" + "".join(f"{lab:>24}" for lab, _ in rows))
    def line(name, key, fmt="{:.3f}"):
        print(f"{name:<{w}}" + "".join(f"{fmt.format(s[key]):>24}" for _, s in rows))
    line("items generated", "n", "{:d}")
    line("imitate share", "imitate_share", "{:.1%}")
    line("  n imitate", "n_imitate", "{:d}")
    line("  n axis-conditioned", "n_axis", "{:d}")
    print()
    line("duplicate rate, imitate", "dup_imitate", "{:.3f}")
    line("duplicate rate, axis", "dup_axis", "{:.3f}")
    line("duplicate rate, corpus", "dup_all", "{:.3f}")
    print()
    line("top-4 openings, imitate", "tmpl_imitate", "{:.1%}")
    line("top-4 openings, corpus", "tmpl_all", "{:.1%}")
    line("distinct items per call", "distinct_per_call", "{:.3f}")

    c, f = rows[0][1], rows[1][1]
    print("\nchange in the imitate branch's own duplicate rate: "
          f"{c['dup_imitate']:.3f} -> {f['dup_imitate']:.3f}")
    if c["dup_imitate"] > 0:
        print(f"  relative reduction: {1 - f['dup_imitate']/c['dup_imitate']:+.1%}")
    print(f"distinct items per call: {c['distinct_per_call']:.3f} -> "
          f"{f['distinct_per_call']:.3f} "
          f"({f['distinct_per_call']/max(c['distinct_per_call'],1e-9)-1:+.1%})")
    json.dump({lab: s for lab, s in rows},
              open(HERE / "imitate_fix_comparison.json", "w"), indent=2)


if __name__ == "__main__":
    main()
