"""
Literal (n-gram) and latent (embedding) diversity, measured as n grows 5 -> 10,000.

The two levels answer different questions and can disagree, which is the point
of measuring both:

  LITERAL   Are the actual words different? A generator can be pushed into
            semantically distinct territory while still reaching for the same
            vocabulary and the same sentence shapes -- "different content,
            same voice". n-gram measures catch that; embeddings do not,
            because a good embedder is deliberately invariant to it.

  LATENT    Are the meanings different? Two items can share almost no
            vocabulary and still say the same thing. Embedding measures catch
            that; n-gram measures do not.

A method that raises one while leaving the other flat is doing something
worth naming, in either direction.

Literal measures implemented
----------------------------
  distinct_n      |unique n-grams| / |n-grams|, the standard lexical-diversity
                  measure. Falls as the corpus grows for ANY fixed generator,
                  since the vocabulary is finite -- so we always report it
                  against n, never as a single number.
  ttr / mattr     type-token ratio, and a moving-average TTR over a fixed
                  window, which is the length-invariant version (plain TTR
                  falls mechanically with corpus size and cannot be compared
                  across different n).
  self_repetition mean over items of the fraction of an item's 4-grams that
                  appeared in ANY earlier item. This is the literal analogue
                  of the min-gap: it asks how much of this item is already on
                  the page.
  ngram_vendi     Vendi Score over L2-normalized n-gram count vectors, giving
                  an "effective number of distinct wordings" directly
                  comparable to the embedding Vendi.
  coverage_growth |unique n-grams| as a function of n -- a Heaps'-law curve
                  whose exponent measures how fast new vocabulary arrives.

Latent measures: Vendi (linear kernel), median/min nearest-neighbour cosine
distance, and headroom against a held-out probe.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    total, uniq = 0, set()
    for t in texts:
        g = ngrams(tokens(t), n)
        total += len(g)
        uniq.update(g)
    return len(uniq) / max(total, 1)


def ttr(texts: list[str]) -> float:
    toks = [w for t in texts for w in tokens(t)]
    return len(set(toks)) / max(len(toks), 1)


def mattr(texts: list[str], window: int = 2000) -> float:
    """Moving-average type-token ratio. Plain TTR falls mechanically as the
    token count grows, so it cannot be compared between corpora of different
    size; MATTR fixes the window and is therefore comparable across n."""
    toks = [w for t in texts for w in tokens(t)]
    if len(toks) <= window:
        return len(set(toks)) / max(len(toks), 1)
    vals = []
    step = max(1, (len(toks) - window) // 200)
    for i in range(0, len(toks) - window + 1, step):
        vals.append(len(set(toks[i:i + window])) / window)
    return float(np.mean(vals))


def self_repetition(texts: list[str], n: int = 4) -> float:
    """Mean fraction of an item's n-grams already seen in earlier items."""
    seen: set[tuple[str, ...]] = set()
    fracs = []
    for t in texts:
        g = ngrams(tokens(t), n)
        if not g:
            continue
        gs = set(g)
        fracs.append(len(gs & seen) / len(gs))
        seen |= gs
    return float(np.mean(fracs)) if fracs else 0.0


def exact_dup_rate(texts: list[str]) -> dict:
    """Exact-duplicate statistics.

    Worth measuring separately from every embedding metric because it is the
    one failure that needs no geometry to detect and no interpretation to
    condemn -- and because when it is present it dominates all the other
    numbers. In our psychometric naive corpus a single item recurs 1,637
    times, which would silently set the scale of every distance statistic
    computed over that corpus.
    """
    from collections import Counter
    c = Counter(texts)
    redundant = len(texts) - len(c)
    top = c.most_common(1)[0] if c else ("", 0)
    return {
        "unique_texts": len(c),
        "exact_dup_rate": round(redundant / max(len(texts), 1), 5),
        "max_single_item_count": top[1],
    }


def unique_ngram_count(texts: list[str], n: int) -> int:
    uniq = set()
    for t in texts:
        uniq.update(ngrams(tokens(t), n))
    return len(uniq)


def _vendi_from_gram(S: np.ndarray) -> float:
    w = np.linalg.eigvalsh(S)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def ngram_vendi(texts: list[str], n: int = 2, max_features: int = 4000,
                sample: int = 1500, seed: int = 0) -> float:
    """Vendi over n-gram count vectors.

    Restricted to the `max_features` most frequent n-grams and (for large
    corpora) a random subsample of items: the score is computed from a
    feature-space Gram matrix, so cost is O(items x features + features^2)
    and the truncation keeps that bounded. Rare n-grams carry almost no
    weight in a linear kernel, so the truncation costs little.
    """
    rng = np.random.default_rng(seed)
    if len(texts) > sample:
        idx = rng.choice(len(texts), sample, replace=False)
        texts = [texts[i] for i in idx]
    counts = Counter()
    per_item = []
    for t in texts:
        g = Counter(ngrams(tokens(t), n))
        per_item.append(g)
        counts.update(g)
    vocab = [g for g, _ in counts.most_common(max_features)]
    if not vocab:
        return 0.0
    index = {g: i for i, g in enumerate(vocab)}
    X = np.zeros((len(per_item), len(vocab)))
    for r, g in enumerate(per_item):
        for k, v in g.items():
            j = index.get(k)
            if j is not None:
                X[r, j] = v
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    keep = (nrm.ravel() > 0)
    X = X[keep] / nrm[keep]
    if len(X) < 2:
        return float(len(X))
    return _vendi_from_gram((X.T @ X) / len(X))


def embed_vendi(E: np.ndarray) -> float:
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    return _vendi_from_gram((En.T @ En) / len(En))


def embed_vendi_centered(E: np.ndarray) -> float:
    """Vendi after removing the corpus mean direction.

    Same-domain text embeddings sit in a narrow cone -- pairwise cosine
    similarity around 0.9 is normal for a corpus that is all one genre -- so
    the uncentered Gram matrix is dominated by a single shared direction and
    the score compresses hard toward 1. That makes the ABSOLUTE uncentered
    number an artifact of the kernel rather than a property of the corpus.
    Centering removes the shared component and measures the spread that
    actually distinguishes items. We report both: uncentered for comparability
    with the literature, centered because it is the honest one.
    """
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    Ec = En - En.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(Ec, axis=1, keepdims=True)
    keep = nrm.ravel() > 1e-9
    if keep.sum() < 2:
        return 1.0
    Ec = Ec[keep] / nrm[keep]
    return _vendi_from_gram((Ec.T @ Ec) / len(Ec))


def nn_stats(E: np.ndarray, sample: int = 3000, seed: int = 0) -> dict:
    """Nearest-neighbour cosine distance stats, on a bounded subsample so the
    O(n^2) similarity matrix stays affordable at n = 10,000."""
    rng = np.random.default_rng(seed)
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    if len(En) > sample:
        En = En[rng.choice(len(En), sample, replace=False)]
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    nn_sim = S.max(axis=1)
    return {
        "median_nn_cos_dist": float(np.median(1 - nn_sim)),
        "min_nn_cos_dist": float((1 - nn_sim).min()),
        "p10_nn_cos_dist": float(np.percentile(1 - nn_sim, 10)),
        "max_pair_cos_sim": float(nn_sim.max()),
    }


CHECKPOINTS = [5, 10, 20, 40, 75, 150, 300, 600, 1000, 1750, 3000, 5000, 7500, 10000]


def curve(texts: list[str], E: np.ndarray, label: str) -> list[dict]:
    out = []
    for n in CHECKPOINTS:
        if n > len(texts):
            break
        T, Ee = texts[:n], E[:n]
        row = {
            "n": n,
            # literal
            "distinct_1": round(distinct_n(T, 1), 5),
            "distinct_2": round(distinct_n(T, 2), 5),
            "distinct_3": round(distinct_n(T, 3), 5),
            "mattr": round(mattr(T), 5),
            "self_repetition_4": round(self_repetition(T), 5),
            "unique_2grams": unique_ngram_count(T, 2),
            "ngram_vendi_2": round(ngram_vendi(T, 2), 3),
            # latent
            "embed_vendi": round(embed_vendi(Ee), 3),
            "embed_vendi_centered": round(embed_vendi_centered(Ee), 3),
        }
        row.update({k: round(v, 5) for k, v in nn_stats(Ee).items()})
        row.update(exact_dup_rate(T))
        # dedup'd latent view: with a 72% duplicate rate the raw embedding
        # metrics describe the duplication, not the corpus
        uniq_idx = []
        seen = set()
        for j, t in enumerate(T):
            if t not in seen:
                seen.add(t)
                uniq_idx.append(j)
        if len(uniq_idx) >= 2:
            Eu = Ee[np.array(uniq_idx)]
            row["embed_vendi_centered_dedup"] = round(embed_vendi_centered(Eu), 3)
        out.append(row)
        print(f"  [{label}] n={n:6d} distinct2={row['distinct_2']:.4f} "
              f"selfrep={row['self_repetition_4']:.4f} "
              f"ngramV={row['ngram_vendi_2']:7.2f} embedV={row['embed_vendi']:6.2f} "
              f"embedVc={row['embed_vendi_centered']:7.2f} "
              f"medNN={row['median_nn_cos_dist']:.4f}", flush=True)
    return out


def load_run(domain: str, arm: str):
    d = HERE / "real" / f"{domain}_{arm}"
    texts = []
    for line in (d / "corpus.jsonl").read_text().splitlines():
        if line.strip():
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    E = np.load(d / "embeddings.npy")
    m = min(len(texts), len(E))
    return texts[:m], E[:m]


def main():
    import sys
    runs = sys.argv[1:] or ["dalle_naive", "dalle_ihd",
                            "psychometric_naive", "psychometric_ihd"]
    out = {}
    for r in runs:
        domain, arm = r.rsplit("_", 1)
        try:
            texts, E = load_run(domain, arm)
        except FileNotFoundError:
            print(f"skipping {r}: not found")
            continue
        print(f"{r}: n={len(texts)}")
        out[r] = curve(texts, E, r)
    path = HERE / "figures" / "real_curves.json"
    prev = json.load(open(path)) if path.exists() else {}
    prev.update(out)
    with open(path, "w") as f:
        json.dump(prev, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
