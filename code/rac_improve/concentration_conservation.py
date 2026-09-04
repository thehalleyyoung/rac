"""
Is concentration conserved, or merely moved?

§6.14 now says an intervention enumerating a proper subset of a measure's space
displaces concentration into the complement. That is a claim about MOVEMENT,
supported by watching a single statistic -- the top non-echo word share -- go up
in one place when an axis constrained another. A stronger and more surprising
reading is available: that the total tendency to repeat is roughly fixed, and
conditioning only decides where it lands.

The distinction matters for what the paper can claim. If total lexical
concentration is conserved across the four poem arms while its distribution over
word classes shifts, the rule is mechanical and predicts where the next
intervention will fail. If total concentration varies -- if some arms are simply
more repetitive overall -- then the rule is descriptive, and §6.14 should say
that conditioning relocates concentration WHEN it relocates it, not that
repetition is a fixed budget.

Two measures, both computed within a posture level and averaged, both blind to
which words an axis named:

  overlap    mean pairwise Jaccard over content words -- how much any two poems
             at the same posture share, the whole-vocabulary version of the
             top-word statistic
  top-share  the existing per-level maximum, split by whether the dominant word
             is a field noun (what the lexical axis constrained) or not

The four arms differ in exactly what they constrain: nothing, rhetorical
vehicles, per-posture devices, and concrete-noun field. Under conservation the
overlap is flat across all four.

Usage:
    python3 concentration_conservation.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from stereotypy_scan import words          # noqa: E402
from resample import bootstrap_ci           # noqa: E402

RNG = np.random.default_rng(20260913)

# The eight fields the lexical axis names, used only to CLASSIFY where a
# dominant word came from -- never to score an arm.
FIELD = re.compile(
    r"\b(kitchen|door|chair|lamp|drawer|shelf|floor|window|cup|bed|room|house|table|key"
    r"|engine|gear|piston|lathe|bolt|weld|motor|valve|belt|wire|steel|machine|crank"
    r"|root|leaf|leaves|seed|soil|branch|bloom|vine|grass|orchard|garden|stem|bark"
    r"|bone|blood|lung|nerve|skin|wound|pulse|vein|fever|spine|marrow|heart"
    r"|ledger|debt|contract|clause|invoice|coin|account|lien|payment|fee|deed"
    r"|cloud|rain|storm|wind|frost|sky|thunder|granite|snow|hail|stone"
    r"|bread|salt|knife|pot|flour|broth|oven|plate|spoon|butter|stew"
    r"|harbor|harbour|tide|hull|keel|rope|anchor|wave|shore|dock|mast|sail)\b")


def load() -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    f = HERE / "poem_device_arms.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    arms["base"] = [r for r in rows if r["arm"] == "base"]
    arms["vehicle"] = [r for r in rows if r["arm"] == "device"]
    for name, fn in (("refined", "poem_refined_arm.jsonl"),
                     ("lexical", "poem_lexical_arm.jsonl")):
        p = HERE / fn
        if p.exists():
            arms[name] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return arms


def by_level(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        out[str(r["spec"]["Claim posture"]).split(" -- ")[0]].append(r)
    return out


def overlap(rows: list[dict]) -> float:
    """Mean pairwise Jaccard over content words, echo stripped per poem."""
    sets = []
    for r in rows:
        echo = set().union(*[words(str(v)) for v in r["spec"].values()])
        sets.append(words(r["text"]) - echo)
    v = [len(a & b) / max(len(a | b), 1)
         for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(v)) if v else float("nan")


def top_share(rows: list[dict]) -> tuple[str, float]:
    df = collections.Counter()
    for r in rows:
        echo = set().union(*[words(str(v)) for v in r["spec"].values()])
        df.update(words(r["text"]) - echo)
    if not df:
        return "-", 0.0
    w, c = df.most_common(1)[0]
    return w, c / len(rows)


def boot(rows: list[dict], fn, n: int = 2000) -> tuple[float, float]:
    """Delegates to the shared guard, which refuses replacement for pairwise
    statistics and raises if the interval fails to bracket the estimate."""
    _, lo, hi = bootstrap_ci(rows, fn, kind="pairwise", n=n, seed=13)
    return lo, hi


def main() -> None:
    arms = load()
    print(f"{'arm':<10}{'n':>5}{'overlap':>10}{'95% CI':>20}"
          f"{'top-share':>11}{'field-noun tops':>17}")
    res = {}
    for name, rows in arms.items():
        lv = by_level(rows)
        ov = float(np.mean([overlap(v) for v in lv.values() if len(v) >= 8]))
        tops = [top_share(v) for v in lv.values() if len(v) >= 8]
        ts = float(np.mean([s for _, s in tops]))
        nf = sum(1 for w, _ in tops if FIELD.search(w))
        lo, hi = boot(rows, lambda rs: float(np.mean(
            [overlap(v) for v in by_level(rs).values() if len(v) >= 8])))
        res[name] = (ov, ts, lo, hi)
        print(f"{name:<10}{len(rows):>5}{ov:>10.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>20}{ts:>11.3f}"
              f"{f'{nf}/{len(tops)}':>17}")

    b = res["base"][0]
    print(f"\noverlap vs base:")
    for name in arms:
        if name == "base":
            continue
        ov, _, lo, hi = res[name]
        sep = "flat" if (lo <= b <= hi) else "DIFFERS"
        print(f"  {name:<10}{ov - b:>+9.4f}   base {b:.4f} "
              f"{'inside' if sep == 'flat' else 'outside'} this arm's CI  -> {sep}")

    spread = max(r[0] for r in res.values()) - min(r[0] for r in res.values())
    print(f"\noverlap spread across all four arms: {spread:.4f} "
          f"(top-share spread: "
          f"{max(r[1] for r in res.values()) - min(r[1] for r in res.values()):.3f})")
    json.dump({k: {"overlap": v[0], "top_share": v[1],
                   "ci": [v[2], v[3]]} for k, v in res.items()},
              open(HERE / "concentration_conservation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
