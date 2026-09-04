"""
Name the whole partition, not a slice of it.

Iteration 30 left a conditional rule: an aligned axis controls the distribution
over the categories it NAMES, and whether that improves coverage depends on
whether the corpus was already spanning. Naming five of ten task categories
moved 67 points of probability mass onto those five (14% -> 81%) and changed
total spread not at all, because the five it did not name were driven out --
`decide` 27 -> 0, `plan` 23 -> 1, `explain` 11 -> 0.

That failure is a property of the SLICE, not of alignment. An axis enumerating
the whole partition has no unnamed half to trade away, and level balancing then
does the rest: each category is commanded equally often, so the corpus is
pushed toward uniform coverage whatever its starting distribution.

The prescription is therefore testable in both regimes at once, and it has to
hold in both to be worth stating:

  items          a lumpy corpus  (base entropy 0.700, dominant area 35%)
  instructions   a flat corpus   (base entropy 0.811, dominant task 27%)

A slice-axis helped the first and cost the second. A full-partition axis should
help the first and at worst leave the second alone. If it flattens items but
degrades instructions, the rule stays conditional and the prescription is
wrong.

Usage:
    python3 full_partition_axis.py [n_per_arm]
"""
from __future__ import annotations

import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from pipeline import USAGE                                    # noqa: E402
import item_content_axes as ITEM                              # noqa: E402
import instruction_alignment as INSTR                         # noqa: E402

SEED = 20260907
N_DEFAULT = 100
WORKERS = 8

# All ten blueprint areas, not the five underused ones.
ITEM_FULL = {
    "Blueprint content area": [
        "Compute: instances, containers, autoscaling, runtime sizing",
        "Storage: object, block and file storage, tiering, snapshots",
        "Networking: addressing, routing, load balancing, DNS, delivery",
        "Security operations: vulnerability handling, patching, key lifecycle",
        "Identity and access management: roles, permissions, least privilege",
        "Data platforms: databases, warehouses, streaming, query and schema",
        "Reliability: availability zones, failover, backup, recovery objectives",
        "Cost management: pricing models, budgets, chargeback, rightsizing",
        "Observability: metrics, logs, traces, alerting, audit trails",
        "Migration: rehosting, replatforming, cutover planning, legacy estates",
    ]
}


def ent(c: collections.Counter, keys: list[str]) -> float:
    p = np.array([c.get(k, 0) for k in keys], dtype=float)
    p = p / max(p.sum(), 1)
    q = p[p > 0]
    return float(-(q * np.log(q)).sum() / np.log(len(keys)))


def boot(a: list[str], b: list[str], keys: list[str], n: int = 8000):
    rng = np.random.default_rng(SEED)
    de, dm = [], []
    for _ in range(n):
        x = collections.Counter(rng.choice(a, len(a)))
        y = collections.Counter(rng.choice(b, len(b)))
        de.append(ent(y, keys) - ent(x, keys))
        dm.append(max(y.values()) / sum(y.values()) - max(x.values()) / sum(x.values()))
    return np.array(de), np.array(dm)


def show(name: str, labs: list[str], keys: list[str]) -> None:
    c = collections.Counter(labs)
    tot = max(sum(c.values()), 1)
    print(f"{name:<22}{tot:>5}{len(c):>7}{ent(c, keys):>9.3f}"
          f"{max(c.values()) / tot:>11.1%}   {c.most_common(1)[0][0]}")
    print(f"    {', '.join(f'{k}:{v}' for k, v in c.most_common())}")


def run_items(n: int) -> None:
    print("\n" + "=" * 68 + "\nITEMS -- lumpy baseline\n" + "=" * 68, flush=True)
    keys = list(ITEM.BLUEPRINT)
    prev = json.load(open(HERE / "item_blueprint_axis.json"))
    base_labs = [k for k, v in prev["base (7 architectural axes)"].items() for _ in range(v)]

    specs = ITEM.build({**ITEM.ARCH_AXES, **ITEM_FULL}, n, SEED)
    print(f"generating full-partition arm: {n} items, "
          f"{len(ITEM.ARCH_AXES) + 1} axes/item", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        texts = [t for t in ex.map(ITEM.gen, specs) if t]
    print(f"  {len(texts)}/{n} generated; classifying blind", flush=True)
    import item_blueprint_axis as BP
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        labs = [l for l in ex.map(BP.classify, texts) if l]

    print(f"\n{'arm':<22}{'n':>5}{'areas':>7}{'entropy':>9}{'max share':>11}   top")
    show("base (architectural)", base_labs, keys)
    show("full partition", labs, keys)
    de, dm = boot(base_labs, labs, keys)
    for lab, v in (("entropy", de), ("max share", dm)):
        lo, hi = np.percentile(v, [2.5, 97.5])
        s = "*" if lo > 0 or hi < 0 else " "
        print(f"{s} full - base, {lab:<11}{v.mean():>+8.3f}  [{lo:+.3f}, {hi:+.3f}]")
    json.dump({"base": dict(collections.Counter(base_labs)),
               "full": dict(collections.Counter(labs))},
              open(HERE / "full_partition_items.json", "w"), indent=2)


def run_instructions(n: int) -> None:
    print("\n" + "=" * 68 + "\nINSTRUCTIONS -- flat baseline\n" + "=" * 68, flush=True)
    keys = INSTR.TASKS
    prev = json.load(open(HERE / "instruction_alignment.json"))
    base_labs = [k for k, v in prev["base"].items() for _ in range(v)]

    full = {**INSTR.SHAPE_AXES, "Task category": [
        f"{t}: {INSTR.TASK_DESC[t]}" for t in INSTR.TASKS]}
    print(f"generating full-partition arm: {n} instructions, {len(full)} axes",
          flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        texts = [t for t in ex.map(INSTR.gen, INSTR.build(full, n, SEED)) if t]
    print(f"  {len(texts)}/{n} generated; classifying blind", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        labs = [l for l in ex.map(INSTR.classify, texts) if l]

    print(f"\n{'arm':<22}{'n':>5}{'tasks':>7}{'entropy':>9}{'max share':>11}   top")
    show("base (shape only)", base_labs, keys)
    show("full partition", labs, keys)
    de, dm = boot(base_labs, labs, keys)
    for lab, v in (("entropy", de), ("max share", dm)):
        lo, hi = np.percentile(v, [2.5, 97.5])
        s = "*" if lo > 0 or hi < 0 else " "
        print(f"{s} full - base, {lab:<11}{v.mean():>+8.3f}  [{lo:+.3f}, {hi:+.3f}]")
    json.dump({"base": dict(collections.Counter(base_labs)),
               "full": dict(collections.Counter(labs))},
              open(HERE / "full_partition_instructions.json", "w"), indent=2)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    run_items(n)
    run_instructions(n)
    print(f"\ntotal spend ${USAGE.cost_usd():.2f}")
