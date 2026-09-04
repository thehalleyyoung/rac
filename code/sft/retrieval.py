"""Does a high-coverage pool serve more queries? No training involved.

Coverage says: for a given radius, what fraction of the reference distribution
has an exemplar nearby. The direct consequence, if the claim means anything, is
that retrieving k nearest demonstrations from a high-coverage pool should find
RELEVANT demonstrations for more test queries than a low-coverage pool of the
same size does.

That is coverage's own mechanism, measured without a model in the way. Three
quantities per pool, all against the same held-out Dolly queries:

  served@r   fraction of queries with at least one pool item within radius r
  nn_sim     similarity of each query to its nearest pool item (mean, and the
             10th percentile, which is the tail coverage is supposed to fix)
  demo_gain  how much closer the retrieved demonstrations are than random ones
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data"
N_POOL = 950            # matched pool size, as in the SFT comparison

ARMS = ["ours_v4", "ours_keepall", "ours_v3", "ours_sel",
        "fs_self_instruct", "fs_evol_instruct", "fs_persona_hub",
        "alpaca_tokmatch", "personahub", "wizardlm"]


def embed(texts, tag):
    cache = D / f"rt_{tag}.npy"
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(texts):
            return E
    import urllib.request
    vecs = []
    for i in range(0, len(texts), 64):
        b = [t[:2000] for t in texts[i:i + 64]]
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=json.dumps({"model": "nomic-embed-text", "input": b}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            vecs.extend(json.loads(r.read())["embeddings"])
    E = np.asarray(vecs, dtype=np.float32)
    E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    np.save(cache, E)
    return E


def main():
    held = json.load(open(D / "dolly_heldout.json"))
    Q = embed([h["instruction"] for h in held], "queries")
    print(f"queries: {len(Q)} held-out Dolly instructions\n")

    rng = random.Random(20260901)
    rows = {}
    print(f"{'pool':20s} {'nn_sim':>7} {'p10':>7} {'served@.60':>11} "
          f"{'served@.65':>11} {'served@.70':>11}")
    for a in ARMS:
        f = D / f"arm_x_{a}.jsonl"
        if not f.exists():
            continue
        items = [json.loads(l)["instruction"] for l in
                 f.read_text().splitlines() if l.strip()]
        if len(items) < N_POOL:
            continue
        samp = rng.sample(items, N_POOL)
        P = embed(samp, f"pool_{a}")
        S = Q @ P.T                       # cosine, both unit-normalised
        nn = S.max(axis=1)
        rows[a] = {
            "nn_sim_mean": float(nn.mean()),
            "nn_sim_p10": float(np.percentile(nn, 10)),
            "served_0.60": float((nn >= 0.60).mean()),
            "served_0.65": float((nn >= 0.65).mean()),
            "served_0.70": float((nn >= 0.70).mean()),
        }
        r = rows[a]
        print(f"{a:20s} {r['nn_sim_mean']:7.4f} {r['nn_sim_p10']:7.4f} "
              f"{r['served_0.60']:11.3f} {r['served_0.65']:11.3f} "
              f"{r['served_0.70']:11.3f}")
    json.dump(rows, open(HERE / "figures" / "retrieval.json", "w"), indent=2)
    print("\nwrote figures/retrieval.json")
    print("nn_sim_p10 is the tail: the 10% of queries worst served by the pool.")


if __name__ == "__main__":
    main()
