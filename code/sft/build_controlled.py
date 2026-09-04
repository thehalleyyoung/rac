"""Subsets of ONE corpus that differ in coverage and in nothing else.

Same policy, generator, responder, style and size; the only thing that varies is
how the items are spread over the reference. Greedy-maximum coverage, greedy-
minimum coverage, and random give three points on that axis, which is the only
design here that can separate coverage from the properties that travel with it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data"
N = 800


def embed(texts, tag):
    c = D / f"cc_{tag}.npy"
    if c.exists():
        E = np.load(c)
        if len(E) == len(texts):
            return E
    import urllib.request
    v = []
    for i in range(0, len(texts), 64):
        b = [t[:2000] for t in texts[i:i + 64]]
        rq = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=json.dumps({"model": "nomic-embed-text", "input": b}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=600) as r:
            v.extend(json.loads(r.read())["embeddings"])
    E = np.asarray(v, dtype=np.float32)
    E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    np.save(c, E)
    return E


def greedy(E, ref, eps, n, maximise=True):
    covered = np.zeros(len(ref), dtype=bool)
    picked, avail = [], set(range(len(E)))
    for _ in range(n):
        best, best_gain = -1, (-1 if maximise else 10 ** 9)
        for j in list(avail)[:4000]:
            gain = int((~covered & ((ref @ E[j]) >= 1.0 - eps)).sum())
            if (maximise and gain > best_gain) or (not maximise and gain < best_gain):
                best, best_gain = j, gain
        if best < 0:
            break
        picked.append(best)
        avail.discard(best)
        covered |= (ref @ E[best]) >= 1.0 - eps
    return picked, covered.mean()


def main():
    rows = [json.loads(l) for l in (D / "v4_10x.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("response", "").strip()]
    print(f"scaled corpus: {len(rows)} items with responses")
    E = embed([r["instruction"] for r in rows], "v4_10x")

    held = json.load(open(D / "dolly_heldout.json"))
    R = embed([h["instruction"] for h in held], "dolly_q")
    S = R @ R.T
    np.fill_diagonal(S, -np.inf)
    eps = float(np.percentile(1.0 - S.max(axis=1), 60))
    print(f"radius eps = {eps:.4f}, reference = {len(R)} held-out queries\n")

    rng = random.Random(20260901)
    out = {}
    hi, chi = greedy(E, R, eps, N, maximise=True)
    lo, clo = greedy(E, R, eps, N, maximise=False)
    rd = rng.sample(range(len(rows)), N)
    crd = ((R @ E[np.array(rd)].T) >= 1.0 - eps).any(axis=1).mean()
    for tag, idx, cov in (("high", hi, chi), ("low", lo, clo), ("rand", rd, crd)):
        p = D / f"arm_C_v4_{tag}.jsonl"
        p.write_text("\n".join(json.dumps(rows[i]) for i in idx) + "\n")
        iw = np.mean([len(rows[i]["instruction"].split()) for i in idx])
        rw = np.mean([len(rows[i]["response"].split()) for i in idx])
        out[tag] = {"n": len(idx), "reference_covered": float(cov),
                    "instr_words": float(iw), "resp_words": float(rw)}
        print(f"  C_v4_{tag:5s} n={len(idx):4d}  covers {cov:.3f} of the reference  "
              f"instr {iw:.0f}w  resp {rw:.0f}w")
    json.dump(out, open(HERE / "figures" / "controlled.json", "w"), indent=2)
    print("\nAll three come from one corpus, so generator, style and length match;")
    print("coverage is the variable.")


if __name__ == "__main__":
    main()
