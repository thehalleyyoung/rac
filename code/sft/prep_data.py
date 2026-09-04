"""Build the SFT arms: instruction sets, embeddings, and the coverage-greedy prune.

The pruning arm is the clean one: every item comes from Alpaca, so Alpaca's own
responses are used unchanged and no generation confound enters. The generated
arms need responses written by one responder, which gen_responses.py supplies.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PUB = ROOT / "public_corpora"
FAIR = ROOT / "coverage" / "fair_logs"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
SEED = 20260901


def alpaca() -> list[dict]:
    rows = json.loads((PUB / "alpaca_data.json").read_text())
    out = []
    for r in rows:
        instr = r["instruction"].strip()
        if r.get("input", "").strip():
            instr = instr + "\n\n" + r["input"].strip()
        if instr and r.get("output", "").strip():
            out.append({"instruction": instr, "response": r["output"].strip()})
    return out


def embed(texts: list[str], tag: str) -> np.ndarray:
    """Ollama nomic-embed-text, cached — the same geometry the paper reports in."""
    cache = OUT / f"emb_{tag}.npy"
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(texts):
            return E
    import urllib.request
    vecs, B = [], 64
    for i in range(0, len(texts), B):
        batch = [t[:2000] for t in texts[i:i + B]]
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=json.dumps({"model": "nomic-embed-text", "input": batch}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            vecs.extend(json.loads(r.read())["embeddings"])
        if (i // B) % 20 == 0:
            print(f"  embed {tag}: {i + len(batch)}/{len(texts)}", flush=True)
    E = np.asarray(vecs, dtype=np.float32)
    E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    np.save(cache, E)
    return E


def coverage_greedy(E: np.ndarray, k: int, eps: float) -> list[int]:
    """Maximise the number of pool points within eps of the chosen set.

    Lazy greedy over a subsampled reference pool; the marginal gain of a
    candidate is the count of not-yet-covered pool points inside its ball.
    """
    rng = np.random.default_rng(SEED)
    pool_idx = rng.choice(len(E), size=min(20000, len(E)), replace=False)
    P = E[pool_idx]
    covered = np.zeros(len(P), dtype=bool)
    chosen, block = [], 2048
    # upper bounds for lazy evaluation, refreshed in blocks
    ub = np.full(len(E), len(P), dtype=np.int64)
    taken = np.zeros(len(E), dtype=bool)
    for step in range(k):
        order = np.argsort(-ub)
        best, best_gain = -1, -1
        for cand in order[:block]:
            if taken[cand]:
                continue
            if ub[cand] <= best_gain:
                break
            gain = int((~covered & ((P @ E[cand]) >= 1.0 - eps)).sum())
            ub[cand] = gain
            if gain > best_gain:
                best, best_gain = int(cand), gain
        if best < 0:
            break
        taken[best] = True
        covered |= (P @ E[best]) >= 1.0 - eps
        chosen.append(best)
        if (step + 1) % 200 == 0:
            print(f"  greedy {step+1}/{k} covered={covered.mean():.3f}", flush=True)
    return chosen


def main():
    rng = random.Random(SEED)
    alp = alpaca()
    print(f"Alpaca: {len(alp)} items with responses")

    E = embed([r["instruction"] for r in alp], "alpaca")
    # radius: the 5th percentile of reference nearest-neighbour distance
    sub = E[rng.sample(range(len(E)), 4000)]
    S = sub @ sub.T
    np.fill_diagonal(S, -np.inf)
    eps = float(np.percentile(1.0 - S.max(axis=1), 60))
    print(f"selection radius eps = {eps:.4f}")

    k = 2398
    keep = coverage_greedy(E, k, eps)
    (OUT / "arm_alpaca_covsel.jsonl").write_text(
        "\n".join(json.dumps(alp[i]) for i in keep) + "\n")
    rand = rng.sample(range(len(alp)), k)
    (OUT / "arm_alpaca_random.jsonl").write_text(
        "\n".join(json.dumps(alp[i]) for i in rand) + "\n")
    (OUT / "arm_alpaca_full.jsonl").write_text(
        "\n".join(json.dumps(r) for r in alp) + "\n")
    json.dump({"eps": eps, "k": k, "n_alpaca": len(alp), "seed": SEED},
              open(OUT / "prune_meta.json", "w"), indent=2)
    print(f"wrote pruning arms: covsel={len(keep)} random={k} full={len(alp)}")

    # the generated arms carry instructions only; responses come later
    for name, f in [("ours_v4", "ours_v4.jsonl"), ("ours_keepall", "ours_keepall.jsonl"),
                    ("ours_v3", "ours_v3.jsonl"), ("ours_sel", "ours.jsonl"),
                    ("self_instruct", "self_instruct.jsonl"),
                    ("random_seed", "random_seed.jsonl")]:
        rows = [json.loads(l) for l in (FAIR / f).read_text().splitlines() if l.strip()]
        texts = [r["text"].strip() for r in rows if r.get("text", "").strip()]
        (OUT / f"instr_{name}.jsonl").write_text(
            "\n".join(json.dumps({"instruction": t}) for t in texts) + "\n")
        print(f"  {name}: {len(texts)} instructions")


if __name__ == "__main__":
    main()
