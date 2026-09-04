"""
Coverage on real corpora, against a NEUTRAL denominator.

Why this script exists
----------------------
The live pilot (live_pilot.py, PAPER.md section 8) hit a denominator problem we
could diagnose but not fix at pilot scale: the reference pool was built from
the same prompt as the naive baseline, so the baseline was scored against the
distribution it was sampled from and the comparison was unfair by
construction. The parent project's `real/` directory now contains something
that fixes it: two 10,000-item corpora (dalle_naive, psychometric_naive)
produced by plain repeated prompting, with saved nomic-embed-text embeddings.
No selection policy shaped them, and they are large relative to any arm, so
they can serve as a policy-independent Monte-Carlo sample of the generator's
reachable space.

Two design points that the live pilot's failure taught us:

1. NO ARM MAY APPEAR IN ITS OWN DENOMINATOR. The naive corpus is itself one of
   the arms under test, so we split it: a random half becomes the reference
   measure, and the naive arm is scored using items drawn only from the OTHER
   half. Every arm is then scored against a fixed set of points that none of
   its own items belong to.

2. DUPLICATES MUST BE REMOVED FROM THE DENOMINATOR BEFORE ANYTHING ELSE.
   Naive psychometric prompting emits the same item thousands of times. A
   reference pool with a 72% exact-duplicate rate is mostly one point repeated,
   so "covered fraction" against it largely measures whether a method
   reproduces the single most common output. We therefore report every
   coverage number twice: against the raw pool and against the deduplicated
   pool, and we calibrate epsilon on the deduplicated pool (on the raw pool the
   median nearest-neighbor distance is exactly 0, which breaks calibration
   outright -- a concrete way duplicates corrupt distance-based diversity work).

What the denominator does and does not mean
-------------------------------------------
This reference measure is the reachable space OF UNCONDITIONED PROMPTING. It
answers "how much of what the model would say anyway does this method cover?"
That is a real question, and it is the one a fair comparison can actually
settle. It is NOT "how much of what the model COULD say": a method that
deliberately leaves the unconditioned region -- which is the entire point of
conditioning -- scores LOW here while being better, not worse. So we report
covered-fraction jointly with within-corpus diversity, and read the pair.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REAL = HERE.parent / "real"
FIGDIR = HERE / "figures"
SEED = 20260830

DOMAINS = {
    "dalle": ["naive", "high_temp", "self_instruct", "evol_instruct",
              "persona", "ihd", "vision"],
    "psychometric": ["naive", "high_temp", "self_instruct", "evol_instruct",
                     "persona", "ihd"],
}
ARM_DIR = {("dalle", a): f"dalle_{a}" for a in DOMAINS["dalle"]}
ARM_DIR.update({("psychometric", a): f"psychometric_{a}"
                for a in DOMAINS["psychometric"]})
ARM_DIR[("dalle", "vision")] = "dalle_vision"


def load_arm(domain: str, arm: str) -> tuple[list[str], np.ndarray] | None:
    d = REAL / ARM_DIR[(domain, arm)]
    if not (d / "embeddings.npy").exists():
        return None
    E = np.load(d / "embeddings.npy")
    texts = [json.loads(l)["text"] for l in open(d / "corpus.jsonl")]
    # some arms have more corpus rows than embedded rows; embeddings are a
    # row-aligned prefix, so truncate the texts to match rather than guess
    n = min(len(texts), len(E))
    E = E[:n]
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    return texts[:n], E


def dedup(texts: list[str], E: np.ndarray) -> tuple[list[str], np.ndarray, dict]:
    """Collapse EXACT duplicate texts, keeping first occurrence."""
    seen: dict[str, int] = {}
    keep = []
    for i, t in enumerate(texts):
        k = t.strip()
        if k not in seen:
            seen[k] = i
            keep.append(i)
    keep = np.array(keep)
    counts: dict[str, int] = {}
    for t in texts:
        counts[t.strip()] = counts.get(t.strip(), 0) + 1
    top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("", 0)
    stats = {
        "n_raw": len(texts),
        "n_unique": len(keep),
        "exact_duplicate_rate": round(1.0 - len(keep) / max(len(texts), 1), 4),
        "most_common_count": top[1],
        "most_common_text": top[0][:120],
    }
    return [texts[i] for i in keep], E[keep], stats


def blocked_min_dist(pool: np.ndarray, S: np.ndarray, block: int = 2048) -> np.ndarray:
    S2 = (S ** 2).sum(axis=1)
    out = np.empty(len(pool))
    for i in range(0, len(pool), block):
        P = pool[i:i + block]
        d2 = (P ** 2).sum(axis=1)[:, None] - 2.0 * (P @ S.T) + S2[None, :]
        out[i:i + block] = np.sqrt(np.clip(d2.min(axis=1), 0.0, None))
    return out


def self_nn(E: np.ndarray, block: int = 2048) -> np.ndarray:
    E2 = (E ** 2).sum(axis=1)
    out = np.empty(len(E))
    for i in range(0, len(E), block):
        P = E[i:i + block]
        d2 = (P ** 2).sum(axis=1)[:, None] - 2.0 * (P @ E.T) + E2[None, :]
        for r in range(len(P)):
            d2[r, i + r] = np.inf
        out[i:i + block] = np.sqrt(np.clip(d2.min(axis=1), 0.0, None))
    return out


def vendi(E: np.ndarray, center: bool) -> float:
    X = E - E.mean(axis=0, keepdims=True) if center else E
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    K = (X @ X.T) / len(X)
    e = np.clip(np.linalg.eigvalsh(K), 1e-12, None)
    e = e / e.sum()
    return round(float(np.exp(-(e * np.log(e)).sum())), 2)


def ngram(texts: list[str]) -> dict:
    import re
    toks = [re.findall(r"[a-z0-9']+", t.lower()) for t in texts]
    out = {}
    for n in (1, 2):
        g = [tuple(ts[i:i + n]) for ts in toks for i in range(len(ts) - n + 1)]
        out[f"distinct_{n}"] = round(len(set(g)) / max(len(g), 1), 4)
    return out


def analyze(domain: str, rng: np.random.Generator) -> dict:
    print(f"\n=== {domain} ===", flush=True)
    ref_raw_texts, ref_raw_E = load_arm(domain, "naive")
    ref_texts, ref_E, ref_stats = dedup(ref_raw_texts, ref_raw_E)
    print(f"  naive corpus: {ref_stats['n_raw']} raw, {ref_stats['n_unique']} unique "
          f"({100*ref_stats['exact_duplicate_rate']:.1f}% exact dups); "
          f"most common text appears {ref_stats['most_common_count']}x")

    # split the DEDUPED naive corpus: half is the denominator, half is the
    # pool the naive ARM is sampled from, so no arm is in its own denominator
    perm = rng.permutation(len(ref_E))
    half = len(perm) // 2
    ref_idx, naive_holdout_idx = perm[:half], perm[half:]
    REF = ref_E[ref_idx]
    REF_raw = ref_raw_E[rng.permutation(len(ref_raw_E))[:len(REF)]]

    nn = self_nn(REF)
    delta = float(np.median(nn))
    eps_list = [round(0.6 * delta, 4), round(delta, 4), round(1.6 * delta, 4)]
    raw_nn_median = float(np.median(self_nn(REF_raw)))
    print(f"  eps calibration on DEDUPED ref: delta={delta:.4f} -> {eps_list}")
    print(f"  (median self-NN on RAW ref pool = {raw_nn_median:.4f})")

    arms = {}
    loaded = {}
    for arm in DOMAINS[domain]:
        got = load_arm(domain, arm)
        if got is None:
            print(f"  {arm}: no embeddings, skipped")
            continue
        texts, E = got
        if arm == "naive":   # score naive from its held-out half only
            texts = [ref_texts[i] for i in naive_holdout_idx]
            E = ref_E[naive_holdout_idx]
        loaded[arm] = (texts, E)

    matched_n = min(len(E) for _, E in loaded.values())
    print(f"  matched n across {len(loaded)} arms = {matched_n}")

    # The complementary denominator. The naive-corpus reference answers "how
    # much of what the model would say ANYWAY does this method cover?". To ask
    # the other question -- "how much of what the model COULD say?" -- we need a
    # reference that includes the territory conditioning opens. We build it as
    # the deduped union of every arm, and score each arm LEAVE-ONE-OUT (its own
    # items are excluded from its denominator), which keeps the no-arm-in-its-
    # own-denominator rule that the naive-split enforces above. This is still
    # not the true reachable space -- nothing we can sample is -- but it is a
    # denominator no single policy authored.
    # STRATUM-BALANCED: each contributing corpus supplies the SAME number of
    # points. An unweighted union is dominated by whichever stratum has the
    # most points (here the 5k naive reference), which would smuggle the
    # unconditioned distribution back in as the de-facto denominator -- the
    # very bias this second denominator exists to avoid.
    union_parts = {}
    for a, (t, E) in loaded.items():
        Edd = dedup(t, E)[1]
        take = min(matched_n, len(Edd))
        union_parts[a] = Edd[rng.permutation(len(Edd))[:take]]
    union_parts["_refpool"] = REF[rng.permutation(len(REF))[:matched_n]]

    for arm, (texts, E) in loaded.items():
        dd_texts, dd_E, st = dedup(texts, E)
        sel = rng.permutation(len(E))[:matched_n]
        sel_dd = rng.permutation(len(dd_E))[:min(matched_n, len(dd_E))]
        Eraw, Edd = E[sel], dd_E[sel_dd]
        d_raw = blocked_min_dist(REF, Eraw)
        d_dd = blocked_min_dist(REF, Edd)
        d_rawref = blocked_min_dist(REF_raw, Eraw)
        # leave-one-out union denominator
        UNION = np.vstack([v for k, v in union_parts.items() if k != arm])
        d_union = blocked_min_dist(UNION, Edd)
        arms[arm] = {
            "n_available": len(E),
            "n_scored": int(matched_n),
            "n_scored_dedup": int(len(sel_dd)),
            "exact_duplicate_rate": st["exact_duplicate_rate"],
            "covered_dedup_ref": {str(e): round(float((d_dd <= e).mean()), 4)
                                  for e in eps_list},
            "covered_dedup_ref_no_arm_dedup": {
                str(e): round(float((d_raw <= e).mean()), 4) for e in eps_list},
            "covered_raw_ref": {str(e): round(float((d_rawref <= e).mean()), 4)
                                for e in eps_list},
            "covered_union_loo": {str(e): round(float((d_union <= e).mean()), 4)
                                  for e in eps_list},
            "n_union_points": int(len(UNION)),
            "median_self_nn": round(float(np.median(self_nn(Edd))), 4),
            "vendi_centered": vendi(Edd, True),
            "vendi_uncentered": vendi(Edd, False),
            **ngram(dd_texts[:matched_n]),
        }
        a = arms[arm]
        print(f"  {arm:15s} cov@mid(dedup)={a['covered_dedup_ref'][str(eps_list[1])]:.4f} "
              f"cov@mid(raw ref)={a['covered_raw_ref'][str(eps_list[1])]:.4f} "
              f"dup={a['exact_duplicate_rate']:.3f} d2={a['distinct_2']:.3f} "
              f"union_loo={a['covered_union_loo'][str(eps_list[1])]:.4f} "
              f"vendi_c={a['vendi_centered']:.1f} selfNN={a['median_self_nn']:.4f}")

    return {
        "domain": domain,
        "reference": {
            "source": f"{ARM_DIR[(domain,'naive')]}/ (unconditioned prompting)",
            **ref_stats,
            "n_reference_points": int(len(REF)),
            "split": "random half of the deduped naive corpus; the naive arm is "
                     "scored from the complementary half",
            "delta_median_self_nn_dedup": round(delta, 4),
            "median_self_nn_raw_pool": round(raw_nn_median, 4),
            "eps_list": eps_list,
        },
        "matched_n": int(matched_n),
        "arms": arms,
    }


def main():
    FIGDIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "domains": {}}
    for domain in DOMAINS:
        out["domains"][domain] = analyze(domain, rng)
    with open(FIGDIR / "summary_real.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote figures/summary_real.json")


if __name__ == "__main__":
    main()
