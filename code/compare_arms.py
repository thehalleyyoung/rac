"""
Cross-arm comparison at matched n, both domains, real gpt-5.6-luna corpora.

Every arm is evaluated on the SAME number of accepted items, because the arms
have different yields and comparing them at their natural sizes would confound
"more diverse" with "kept fewer". Where an arm's cost per accepted item
differs, that is reported separately rather than folded into the diversity
numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from metrics import (HERE, distinct_n, embed_vendi, embed_vendi_centered,
                     exact_dup_rate, load_run, mattr, ngram_vendi, nn_stats,
                     self_repetition)

ARMS = ["naive", "high_temp", "self_instruct", "evol_instruct", "persona",
        "ihd", "vision"]
LABEL = {
    "naive": "naive repeated prompting",
    "high_temp": "high temperature (T=1.6)",
    "self_instruct": "Self-Instruct (few-shot + ROUGE filter)",
    "evol_instruct": "Evol-Instruct (WizardLM operators)",
    "persona": "persona/attribute conditioning",
    "ihd": "RAC (ours)",
    "vision": "RAC + vision steering (ours)",
}


def evaluate(texts: list[str], E: np.ndarray) -> dict:
    d = exact_dup_rate(texts)
    uniq, seen = [], set()
    for j, t in enumerate(texts):
        if t not in seen:
            seen.add(t)
            uniq.append(j)
    Eu = E[np.array(uniq)] if len(uniq) >= 2 else E
    row = {
        "n": len(texts),
        "unique_texts": d["unique_texts"],
        "exact_dup_rate": d["exact_dup_rate"],
        "max_single_item_count": d["max_single_item_count"],
        "distinct_2": round(distinct_n(texts, 2), 5),
        "self_repetition_4": round(self_repetition(texts), 5),
        "mattr": round(mattr(texts), 5),
        "ngram_vendi_2": round(ngram_vendi(texts, 2), 2),
        "embed_vendi": round(embed_vendi(E), 3),
        "embed_vendi_centered": round(embed_vendi_centered(E), 3),
        "embed_vendi_centered_dedup": round(embed_vendi_centered(Eu), 3),
    }
    row.update({k: round(v, 5) for k, v in nn_stats(E).items()})
    return row


def cost_of(domain: str, arm: str) -> dict:
    p = HERE / "real" / f"{domain}_{arm}" / "run_summary.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text())
    gens = s.get("generations") or s.get("generations_this_session")
    n = s.get("n")
    out = {}
    if gens and n:
        out["generations_per_item"] = round(gens / max(n, 1), 2)
    for k in ("cost_usd", "cost_usd_this_session", "rejected_by_filter"):
        if k in s:
            out[k] = s[k]
    return out


def main():
    domains = sys.argv[1:] or ["dalle", "psychometric"]
    out = {}
    for domain in domains:
        avail = {}
        for arm in ARMS:
            try:
                texts, E = load_run(domain, arm)
            except FileNotFoundError:
                continue
            if len(texts) >= 50:
                avail[arm] = (texts, E)
        if not avail:
            continue
        n_match = min(len(t) for t, _ in avail.values())
        print(f"\n=== {domain}: {len(avail)} arms, matched at n={n_match} ===")
        rows = {}
        for arm, (texts, E) in avail.items():
            r = evaluate(texts[:n_match], E[:n_match])
            r["label"] = LABEL[arm]
            r["n_total_available"] = len(texts)
            r.update(cost_of(domain, arm))
            rows[arm] = r
            print(f"{arm:16s} dup={r['exact_dup_rate']:.3f} d2={r['distinct_2']:.4f} "
                  f"selfrep={r['self_repetition_4']:.4f} "
                  f"ngramV={r['ngram_vendi_2']:7.1f} "
                  f"embedVc={r['embed_vendi_centered']:7.2f} "
                  f"dedupVc={r['embed_vendi_centered_dedup']:7.2f} "
                  f"medNN={r['median_nn_cos_dist']:.4f}")
        out[domain] = {"matched_n": n_match, "arms": rows}
    with open(HERE / "figures" / "arm_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote figures/arm_comparison.json")


if __name__ == "__main__":
    main()
