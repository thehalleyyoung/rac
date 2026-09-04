"""
An axis that CORRELATES with the target space is not an axis that SPANS it.

Iteration 28 recovered three content dimensions from the items themselves --
deployment model, service abstraction level, data modality -- and conditioning
on them did not widen blueprint coverage. It moved the lump: the architectural
arm concentrates on cost (17 of 36 items), the content arm on data (13 of 40),
and the entropy difference is +0.020 with a 95% interval spanning zero.

The diagnosis is that those axes are content-ADJACENT, not blueprint-ALIGNED.
"Streaming, analytical, time-series data" is a real dimension of variation and
it pushes items into one blueprint area rather than across ten. Gap expansion
finds dimensions that are ABSENT; spanning a target partition needs an axis
whose LEVELS ARE that partition. This arm tests exactly that: one axis whose
five levels name blueprint areas the bank underuses.

Scoring cannot be the keyword classifier here. An axis whose levels name
blueprint areas will plant that vocabulary in the item, and a regex would then
confirm itself. Every item from every arm is instead classified by a judge that
sees one item at a time, is blind to arm and to any axis, and assigns the
single area the item PRIMARILY assesses.

Usage:
    python3 item_blueprint_axis.py
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

from pipeline import USAGE, chat, parse_json                    # noqa: E402
from item_content_axes import ARCH_AXES, BLUEPRINT, build, gen  # noqa: E402

SEED = 20260905
N = 40
WORKERS = 8

BLUEPRINT_AXIS = {
    "Blueprint content area": [
        "Identity and access management: roles, permissions, least privilege, federation",
        "Observability: metrics, logs, traces, alerting, audit trails",
        "Migration: rehosting, replatforming, cutover planning, on-premises estates",
        "Networking: addressing, routing, load balancing, DNS, content delivery",
        "Security operations: vulnerability handling, patching, key and certificate lifecycle",
    ]
}

JUDGE = """You are classifying one exam item for a cloud certification blueprint.

Which SINGLE blueprint area does this item primarily assess? Answer with exactly \
one of these labels:
compute, storage, networking, security, identity, data, reliability, cost, \
monitoring, migration

Judge what the item makes the candidate REASON about, not every technology it \
mentions in passing.

Item:
{item}

Return JSON only: {{"area": "..."}}"""


def classify(text: str) -> str | None:
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.0) + 1.5 * attempt)
        try:
            raw = chat([{"role": "user", "content": JUDGE.format(item=text[:2500])}],
                       temperature=0.0, max_tokens=2000, json_mode=True)
            a = str(parse_json(raw)["area"]).strip().lower()
            if a in BLUEPRINT:
                return a
        except Exception:
            pass
    return None


def entropy(counts: collections.Counter, k: int = 10) -> float:
    p = np.array([counts.get(a, 0) for a in BLUEPRINT], dtype=float)
    p = p / max(p.sum(), 1)
    return float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(k))


def main(n: int = N) -> None:
    prev = [json.loads(l) for l in
            (HERE / "item_content_arms.jsonl").read_text().splitlines() if l.strip()]
    arms: dict[str, list[str]] = collections.defaultdict(list)
    for r in prev:
        arms[r["arm"]].append(r["text"])

    specs = build({**ARCH_AXES, **BLUEPRINT_AXIS}, n, SEED)
    print(f"generating blueprint arm: {n} items, {len(ARCH_AXES) + 1} axes/item",
          flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(gen, specs))
    arms["blueprint (+1 aligned axis)"] = [t for t in outs if t]
    print(f"  {len(arms['blueprint (+1 aligned axis)'])}/{n} generated", flush=True)

    # One flat, shuffled pass so the judge cannot see arm structure.
    flat = [(a, t) for a, ts in arms.items() for t in ts]
    random.Random(SEED).shuffle(flat)
    print(f"\nclassifying {len(flat)} items blind", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        labs = list(ex.map(classify, [t for _, t in flat]))

    by: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (a, _), l in zip(flat, labs):
        if l:
            by[a][l] += 1

    print(f"\n{'arm':<30}{'n':>5}{'areas':>7}{'entropy':>9}   top area")
    ents = {}
    for a in arms:
        c = by[a]
        ents[a] = entropy(c)
        top = c.most_common(1)[0] if c else ("-", 0)
        print(f"{a:<30}{sum(c.values()):>5}{len(c):>7}{ents[a]:>9.3f}   "
              f"{top[0]} {top[1] / max(sum(c.values()), 1):.0%}")
        print(f"    {', '.join(f'{k}:{v}' for k, v in c.most_common())}")

    rng = np.random.default_rng(SEED)
    names = list(arms)
    base = names[0]
    lab_by = {a: [l for (aa, _), l in zip(flat, labs) if aa == a and l] for a in names}
    print(f"\nentropy vs '{base}' (5000 bootstrap):")
    for a in names[1:]:
        d = []
        for _ in range(5000):
            x = rng.choice(lab_by[base], len(lab_by[base]))
            y = rng.choice(lab_by[a], len(lab_by[a]))
            d.append(entropy(collections.Counter(y)) - entropy(collections.Counter(x)))
        d = np.array(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f" {star} {a:<30}{d.mean():>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"P(>0)={np.mean(d > 0):.3f}")

    json.dump({a: dict(by[a]) for a in arms},
              open(HERE / "item_blueprint_axis.json", "w"), indent=2)
    with open(HERE / "item_blueprint_arm.jsonl", "w") as fh:
        for t in arms["blueprint (+1 aligned axis)"]:
            fh.write(json.dumps({"arm": "blueprint", "text": t}) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N)
