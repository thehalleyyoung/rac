"""
Is the reachable space bounded by the model that names it?

RAC elicits its axes from the same model that then generates the artifacts, and
recursion refines those axes rather than replacing them: a split subdivides a
dimension the first elicitation proposed, and expansion asks the same model what
its own outputs never vary. Every enlargement of the space is therefore drawn
from one model's account of what poems can differ in. The open-endedness claim
rests on recursion escaping that account, and nothing so far tests whether it
can. A dimension the proposer cannot conceive is not a dimension refinement will
ever reach.

The test holds the generator fixed and changes only who proposes. The same
elicitation prompt goes to three other model families; their axis sets condition
the same generator that the self-elicited set conditions, at the same budget.

    self    axes proposed by the generator itself (the method as published)
    cross   axes proposed by Llama 3.3 70B, Qwen 2.5 72B and DeepSeek

Scoring is in the channel the objective never optimizes -- prosody -- for the
reason it always is here: a semantic embedder is nearly blind to metre, line and
rhyme, so a difference measured there cannot have been produced by optimizing
the objective. Three questions, in order of how much they would matter:

  spread        does either arm's corpus occupy more prosodic room
  reach         does either arm put poems in prosodic regions the other never
                enters -- measured as the share of one arm's poems whose nearest
                neighbour in the OTHER arm is further than that arm's own
                internal nearest-neighbour scale
  vocabulary    do the two axis sets name the same dimensions in different words,
                or genuinely different dimensions

Reach is the one that speaks to the premise. Equal spread with mutual reach
means both proposers find room the other misses, and the bound is real but
symmetric. Equal spread with no reach means the two accounts are notational
variants and self-elicitation costs nothing.

Usage:
    python3 who_proposes_the_axes.py [n_per_arm]
"""
from __future__ import annotations

import collections
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

import pipeline                                        # noqa: E402
from pipeline import USAGE, chat, parse_json           # noqa: E402
from axes import SEED_AXIS_PROMPT                      # noqa: E402
from text_realization import prosodic_features         # noqa: E402
from poem_device_axis import gen, build                # noqa: E402
from poem_lexical_axis import craft                    # noqa: E402

SEED = 20260919
N_DEFAULT = 80
WORKERS = 8
K_AXES = 7
PROPOSERS = ["meta-llama/llama-3.3-70b-instruct",
             "qwen/qwen-2.5-72b-instruct",
             "deepseek/deepseek-chat"]


def elicit(model: str, k: int = K_AXES) -> dict[str, list[str]]:
    """Same prompt, same k, different proposer."""
    orig = pipeline.MODEL
    pipeline.MODEL = model
    try:
        for attempt in range(3):
            time.sleep(1.0 * attempt)
            try:
                raw = chat([{"role": "user", "content": SEED_AXIS_PROMPT.format(k=k)}],
                           temperature=1.0, max_tokens=6000, json_mode=True)
                obj = parse_json(raw)
                out = {}
                for a in obj.get("axes", []):
                    name = str(a.get("name", "")).strip()
                    lv = [str(x).strip() for x in a.get("levels", []) if str(x).strip()]
                    if name and len(lv) >= 3:
                        out[name] = lv[:6]
                if len(out) >= 4:
                    return out
            except Exception as e:
                last = e
        print(f"   elicitation failed for {model}", flush=True)
        return {}
    finally:
        pipeline.MODEL = orig


def prosody(texts: list[str]) -> np.ndarray:
    fs = [prosodic_features(t) for t in texts]
    names = sorted(fs[0])
    X = np.array([[f[n] for n in names] for f in fs], dtype=float)
    return X


def standardize(*arrays: np.ndarray) -> list[np.ndarray]:
    """One shared scaling, fitted on the pooled cloud.

    Standardizing each arm separately would erase exactly the difference the
    reach measure is looking for: an arm that occupies a wider range would be
    rescaled to look like one that does not.
    """
    P = np.vstack(arrays)
    mu, sd = P.mean(0), P.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return [(A - mu) / sd for A in arrays]


def nn_scale(X: np.ndarray) -> float:
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(1)))


def reach(A: np.ndarray, B: np.ndarray) -> float:
    """Share of A's poems whose nearest B-poem is further than B's own median
    nearest-neighbour spacing: A is standing where B never stands."""
    D = np.sqrt(np.maximum(((A[:, None] - B[None]) ** 2).sum(-1), 0))
    return float(np.mean(D.min(1) > nn_scale(B)))


def spread(X: np.ndarray) -> float:
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
    return float(D[np.triu_indices(len(X), 1)].mean())


def main(n: int) -> None:
    print("eliciting axes from each proposer (same prompt, same k)\n", flush=True)
    sets = {}
    for m in PROPOSERS:
        ax = elicit(m)
        if ax:
            sets[m.split("/")[-1]] = ax
            print(f"{m.split('/')[-1]:<26} {len(ax)} axes: "
                  f"{', '.join(list(ax)[:4])}", flush=True)
    if not sets:
        raise SystemExit("no cross-model axis set could be elicited")

    # one cross corpus per proposer, pooled; self arm reused from disk
    prev = [json.loads(l) for l in
            (HERE / "poem_device_arms.jsonl").read_text().splitlines() if l.strip()]
    self_texts = [r["text"] for r in prev if r["arm"] == "base"][:n]
    per = max(n // len(sets), 8)
    cross_rows = []
    for name, ax in sets.items():
        specs = build(ax, per, SEED)
        print(f"\ngenerating {per} poems from {name}'s axes", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(gen, specs))
        got = [{"proposer": name, "spec": s, "text": t}
               for t, s in zip(outs, specs) if t]
        cross_rows += got
        print(f"  {len(got)}/{per} written", flush=True)
    with open(HERE / "cross_proposer_arm.jsonl", "w") as fh:
        for r in cross_rows:
            fh.write(json.dumps(r) + "\n")

    cross_texts = [r["text"] for r in cross_rows]
    m = min(len(self_texts), len(cross_texts))
    S, C = standardize(prosody(self_texts[:m]), prosody(cross_texts[:m]))
    print(f"\nprosodic comparison at n = {m} per arm")
    print(f"  spread   self {spread(S):.3f}   cross {spread(C):.3f}")
    print(f"  reach    cross into regions self never enters: {reach(C, S):.0%}")
    print(f"           self  into regions cross never enters: {reach(S, C):.0%}")

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        cr = list(ex.map(craft, cross_texts[:m]))
    cg = [x for x in cr if x is not None]
    print(f"  craft    cross {np.mean(cg):.2f}   (self baseline 6.5-6.7)")

    print(f"\naxis vocabulary, self vs cross")
    self_axes = ["Claim posture", "Syntactic weather", "Register contract",
                 "Temporal stance", "Voice agency", "Address geometry",
                 "Lineation function"]
    print(f"  self : {', '.join(self_axes)}")
    for name, ax in sets.items():
        print(f"  {name[:12]:<12}: {', '.join(list(ax))}")
    json.dump({"axis_sets": sets,
               "spread_self": spread(S), "spread_cross": spread(C),
               "reach_cross_into_self": reach(C, S),
               "reach_self_into_cross": reach(S, C),
               "craft_cross": float(np.mean(cg))},
              open(HERE / "who_proposes.json", "w"), indent=2)
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
