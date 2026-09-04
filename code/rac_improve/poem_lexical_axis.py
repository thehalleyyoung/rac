"""
The prediction the alignment rule makes, and the way to test it without cheating.

Iterations 32-35 converge on one statement: an intervention reduces a measured
concentration when it enumerates the space THE MEASURE IS DEFINED OVER, and
merely relocates it otherwise. The item device axis enumerated device forms and
was scored on device forms -- 76% to 24%. The poem axes enumerate rhetorical
means (vehicles, then per-posture devices) and are scored on LEXICAL choice,
which they constrain only indirectly -- 0.04, with the mass reappearing on a new
word.

That predicts something specific and falsifiable: an axis enumerating lexical
FIELD should move the lexical measure a great deal, where two rhetorical axes
moved it barely at all.

The obvious objection is circularity, and the design answers it. The metric
groups poems by their commanded CLAIM POSTURE and reports the share of that
group using its most frequent non-echo word. The lexical axis is orthogonal to
posture and level-balanced across it, so each posture level receives all eight
fields in equal measure. Commanding "botanical" makes botanical words common
WITHIN the botanical cells, which are spread evenly across every posture, and
the statistic being reported is computed across a posture's whole group. Nothing
in the axis tells the generator which word to reach for inside a posture; it
only prevents the whole posture from drawing on one field.

Two guards, because a lexical constraint is a plausible way to wreck a poem:
the posture contract is judged blind as before, and craft is judged blind too. A
drop in device share bought by flattening the poems is not a result.

Usage:
    python3 poem_lexical_axis.py [n_per_arm]
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

from pipeline import USAGE, chat, parse_json                 # noqa: E402
from stereotypy_scan import words                            # noqa: E402
from poem_device_axis import BASE_AXES, gen, honours, build  # noqa: E402

SEED = 20260912
N_DEFAULT = 80
WORKERS = 8

LEXICAL_AXIS = {
    "Lexical field": [
        "Draw the poem's concrete nouns from household and domestic objects",
        "Draw them from machinery, tools and industrial process",
        "Draw them from plants, soil and growing things",
        "Draw them from the body, illness and medicine",
        "Draw them from money, contracts and law",
        "Draw them from weather, sky and geology",
        "Draw them from food, cooking and the table",
        "Draw them from water, shipping and the coast",
    ]
}

CRAFT_JUDGE = """You are judging one poem for craft: control of line, image and \
syntax. Penalize cliche, filler and autopilot abstraction. Ignore subject matter \
and ignore whether you like it.

Poem:
{poem}

Return JSON only: {{"craft": <integer 0-10>}}"""


def craft(text: str) -> int | None:
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.0) + 1.5 * attempt)
        try:
            v = parse_json(chat([{"role": "user",
                                  "content": CRAFT_JUDGE.format(poem=text[:2500])}],
                                temperature=0.0, max_tokens=2000, json_mode=True))["craft"]
            if isinstance(v, (int, float)) and 0 <= v <= 10:
                return int(v)
        except Exception:
            pass
    return None


def score(rows: list[dict], key: str = "Claim posture") -> dict:
    """Top non-echo word share per posture level, echo stripped from the FULL
    spec rather than only the grouped label."""
    docs = [words(r["text"]) for r in rows]
    echo = [set().union(*[words(str(v)) for v in r["spec"].values()]) for r in rows]
    labs = [str(r["spec"][key]).split(" -- ")[0] for r in rows]
    out = {}
    for lv in sorted(set(labs)):
        idx = [i for i, l in enumerate(labs) if l == lv]
        if len(idx) < 12:
            continue
        df = collections.Counter()
        for i in idx:
            df.update(docs[i] - echo[i])
        if df:
            w, c = df.most_common(1)[0]
            out[lv] = {"device": w, "share": c / len(idx)}
    return out


def main(n: int) -> None:
    axes = {**BASE_AXES, **LEXICAL_AXIS}
    specs = build(axes, n, SEED)
    print(f"lexical arm: {n} poems, {len(axes)} axes", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(gen, specs))
    keep = [{"arm": "lexical", "spec": s, "text": t}
            for t, s in zip(outs, specs) if t]
    print(f"  {len(keep)}/{n} generated; judging posture and craft blind", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ok = list(ex.map(honours, [(r["text"], r["spec"]["Claim posture"])
                                   for r in keep]))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        cr = list(ex.map(craft, [r["text"] for r in keep]))
    good = [x for x in ok if x is not None]
    cg = [x for x in cr if x is not None]
    print(f"  posture exhibited: {sum(good)}/{len(good)} = "
          f"{sum(good)/max(len(good),1):.0%}")
    print(f"  craft: mean {np.mean(cg):.2f} (n={len(cg)})")

    prev = [json.loads(l) for l in
            (HERE / "poem_device_arms.jsonl").read_text().splitlines() if l.strip()]
    ref = (HERE / "poem_refined_arm.jsonl")
    refined = [json.loads(l) for l in ref.read_text().splitlines() if l.strip()] \
        if ref.exists() else []
    arms = {"base": [r for r in prev if r["arm"] == "base"],
            "vehicle": [r for r in prev if r["arm"] == "device"],
            "refined": refined,
            "lexical": keep}
    sc = {k: score(v) for k, v in arms.items() if v}

    print(f"\n{'level':<26}" + "".join(f"{k:>17}" for k in sc))
    for lv in sorted(sc["base"]):
        row = f"{lv[:25]:<26}"
        for k in sc:
            x = sc[k].get(lv)
            cell = "{} {:.0%}".format(x["device"], x["share"]) if x else "-"
            row += f"{cell:>17}"
        print(row)
    print(f"\n{'MEAN device share':<26}" +
          "".join(f"{np.mean([x['share'] for x in sc[k].values()]):>17.3f}" for k in sc))

    rng = np.random.default_rng(SEED)
    b, l_ = arms["base"], arms["lexical"]
    d = []
    for _ in range(3000):
        i = rng.integers(0, len(b), len(b)); j = rng.integers(0, len(l_), len(l_))
        vb = score([b[k] for k in i]); vl = score([l_[k] for k in j])
        if len(vb) >= 2 and len(vl) >= 2:
            d.append(np.mean([x["share"] for x in vl.values()])
                     - np.mean([x["share"] for x in vb.values()]))
    if d:
        d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
        s = "*" if lo > 0 or hi < 0 else " "
        print(f"\n{s} lexical - base: {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    with open(HERE / "poem_lexical_arm.jsonl", "w") as fh:
        for r, o, c in zip(keep, ok, cr):
            fh.write(json.dumps({**r, "exhibits": o, "craft": c}) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
