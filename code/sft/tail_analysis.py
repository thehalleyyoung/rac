"""Split downstream score by how far each query sits from the training data.

If coverage buys anything for fine-tuning, it should appear on the queries with
no near neighbour in the training set. Averaging hides that: the bulk of queries
are served by every corpus, so a corpus that fixes only the tail moves the mean
very little.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data"
N_POOL = 950


def emb(texts, tag):
    c = D / f"rt_{tag}.npy"
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


def main():
    import sys
    sys.path.insert(0, str(HERE))
    from dashboard import load

    held = json.load(open(D / "dolly_heldout.json"))
    Q = emb([h["instruction"] for h in held], "queries")
    rows = [r for r in load() if r.get("dolly_per_item")]
    if not rows:
        print("no runs with per-item scores yet")
        return
    rng = random.Random(20260901)

    print(f"{'arm':22s} {'mean':>7} {'near tercile':>13} {'far tercile':>12} "
          f"{'gap':>7}")
    out = {}
    for r in sorted(rows, key=lambda r: r["arm"]):
        arm = r["arm"].replace("x_", "").replace("p_", "")
        f = D / f"arm_x_{arm}.jsonl"
        if not f.exists():
            continue
        items = [json.loads(l)["instruction"] for l in
                 f.read_text().splitlines() if l.strip()]
        if len(items) < N_POOL:
            continue
        P = emb(rng.sample(items, N_POOL), f"pool_{arm}")
        nn = (Q @ P.T).max(axis=1)                  # query -> nearest train item
        s = np.asarray(r["dolly_per_item"], dtype=float)
        m = min(len(s), len(nn))
        s, d = s[:m], nn[:m]
        order = np.argsort(d)                       # ascending similarity = far first
        third = m // 3
        far = s[order[:third]].mean()
        near = s[order[-third:]].mean()
        out[arm] = {"mean": float(s.mean()), "near": float(near),
                    "far": float(far), "gap": float(near - far)}
        print(f"{arm:22s} {s.mean():7.4f} {near:13.4f} {far:12.4f} "
              f"{near - far:7.4f}")
    json.dump(out, open(HERE / "figures" / "tail.json", "w"), indent=2)
    print("\n'far' = the third of queries with no near neighbour in the training set.")
    print("A corpus that covers well should lose less on those than one that does not.")


if __name__ == "__main__":
    main()
