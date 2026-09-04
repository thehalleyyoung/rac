"""
Head-to-head on exam items: our synthetic bank against human-written banks
and against our reimplementations of published synthetic-data methods.

Exam items are the right domain for THIS paper's objective. An instruction
corpus wants coverage, and a near-duplicate instruction is merely wasteful;
an item bank wants every item to earn its place, and two items that are too
close are "enemy items" -- seeing one answers the other, which is a test-
security and validity defect at any bank size. Lack of novelty in a single
item is itself the harm, which is exactly the regime this paper formalizes.

Baselines, in increasing order of how uncomfortable they should make us:

  mmlu         cais/mmlu 'all' test split, ~14,000 items. HUMAN-WRITTEN,
               drawn from real practice exams and textbooks across 57
               subjects. Not a synthetic-data method at all -- a ceiling.
  arc          allenai/ai2_arc, grade-school science exam questions, also
               human-written.
  self_instruct / evol_instruct / persona / high_temp / naive
               our faithful reimplementations (baselines.py) run on the same
               psychometric domain with the same generator.

The human banks are the interesting comparison. They were written by many
people over years, with editorial review, deliberately spanning 57 subjects.
Matching their per-item diversity from one small model in an afternoon would
be a real result; not matching it is the honest measure of the remaining gap.

Fairness rules, applied to every corpus including ours:
  * matched n, sampled uniformly at random from each corpus (not the first n,
    since several released corpora are ordered by generation batch and their
    heads are unrepresentative)
  * identical embedder (nomic-embed-text) and identical metric code
  * the instruction/prompt FIELD only -- never the response, since our corpus
    has no responses and including them would compare different objects
  * exact-duplicate rate reported alongside, and a dedup'd variant of every
    embedding metric, since duplication otherwise sets the scale
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from metrics import (distinct_n, embed_vendi, embed_vendi_centered,
                     exact_dup_rate, mattr, ngram_vendi, nn_stats,
                     self_repetition)
from pipeline import HERE, embed

PUB = HERE / "public_corpora"
CACHE = HERE / "figures" / "h2h_cache"
CACHE.mkdir(parents=True, exist_ok=True)
N_MATCH = 2000
SEED = 4242


def load_json_items(name: str) -> list[str]:
    p = PUB / f"{name}_items.json"
    if not p.exists():
        return []
    return [t for t in json.loads(p.read_text()) if len(t) >= 30]


def load_ours(arm: str = "ihd", domain: str = "psychometric") -> list[str]:
    p = HERE / "real" / f"{domain}_{arm}" / "corpus.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


SOURCES = {
    "MMLU (human-written, 14k)": lambda: load_json_items("mmlu"),
    "ARC (human-written)": lambda: load_json_items("arc"),
    "naive gpt-5.6-luna": lambda: load_ours("naive"),
    "high temperature": lambda: load_ours("high_temp"),
    "Self-Instruct (reimpl.)": lambda: load_ours("self_instruct"),
    "Evol-Instruct (reimpl.)": lambda: load_ours("evol_instruct"),
    "persona conditioning (reimpl.)": lambda: load_ours("persona"),
    "IHD (ours)": lambda: load_ours("ihd"),
}


def embed_cached(name: str, texts: list[str]) -> np.ndarray:
    key = CACHE / (name.replace(" ", "_").replace("/", "_").replace(":", "")
                   + f"_{len(texts)}.npy")
    if key.exists():
        return np.load(key)
    E = embed(texts, batch=96)
    np.save(key, E)
    return E


def main():
    n_match = int(sys.argv[1]) if len(sys.argv) > 1 else N_MATCH
    rng = random.Random(SEED)
    rows = {}
    for name, loader in SOURCES.items():
        texts = loader()
        if len(texts) < 50:
            print(f"skip {name}: only {len(texts)} items")
            continue
        avail = len(texts)
        k = min(n_match, avail)
        sample = rng.sample(texts, k)
        E = embed_cached(name, sample)
        d = exact_dup_rate(sample)
        uniq, seen = [], set()
        for j, t in enumerate(sample):
            if t not in seen:
                seen.add(t)
                uniq.append(j)
        Eu = E[np.array(uniq)] if len(uniq) >= 2 else E
        r = {
            "n_evaluated": k,
            "n_available": avail,
            "exact_dup_rate": d["exact_dup_rate"],
            "distinct_2": round(distinct_n(sample, 2), 5),
            "distinct_3": round(distinct_n(sample, 3), 5),
            "self_repetition_4": round(self_repetition(sample), 5),
            "mattr": round(mattr(sample), 5),
            "ngram_vendi_2": round(ngram_vendi(sample, 2), 2),
            "embed_vendi": round(embed_vendi(E), 3),
            "embed_vendi_centered": round(embed_vendi_centered(E), 3),
            "embed_vendi_centered_dedup": round(embed_vendi_centered(Eu), 3),
            "mean_chars": round(float(np.mean([len(t) for t in sample])), 1),
        }
        r.update({kk: round(vv, 5) for kk, vv in nn_stats(E).items()})
        rows[name] = r
        print(f"{name:34s} n={k:5d} dup={r['exact_dup_rate']:.4f} "
              f"d2={r['distinct_2']:.4f} selfrep={r['self_repetition_4']:.4f} "
              f"ngramV={r['ngram_vendi_2']:7.1f} "
              f"embedVc={r['embed_vendi_centered']:7.2f} "
              f"medNN={r['median_nn_cos_dist']:.4f}", flush=True)
    with open(HERE / "figures" / "head_to_head.json", "w") as f:
        json.dump({"matched_n": n_match, "sources": rows}, f, indent=2)
    print("\nwrote figures/head_to_head.json")


if __name__ == "__main__":
    main()
