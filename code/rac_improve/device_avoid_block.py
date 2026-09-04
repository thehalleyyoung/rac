"""
An avoid-block that names the overused DEVICE, not the overused word.

Two results from this sequence point at the same missing piece. Static device
axes barely move poem stereotypy -- a general vehicle axis gives 0.04, and
per-posture refinement relocates the mass rather than reducing it -- because an
axis names a fixed set of alternatives chosen before the run, and concentration
settles wherever that set does not reach. The avoid-block does far better on the
one thing it can see: image instructions written from contracts alone put
"mixed-media" in 55% of them, and the same axes through the full pipeline put it
in 22%, because the avoid-block names whatever the corpus is overusing NOW.

But the avoid-block operates on words. The defect measured in iterations 32-36
is a DEVICE -- a coloured card standing in for every red herring, a liturgical
"By X, by Y" performing every ritual pronouncement. Banning the word leaves the
move intact: forbid "salt" and the next poem swears by ash.

So this gives the adaptive mechanism the right target. Every batch, a judge reads
recent poems and names the single rhetorical MOVE the corpus is leaning on, in
its own words; that move joins a growing block the generator must avoid. The
axis set never changes. What changes is a running description of the habit the
corpus has just formed.

    base            the seven elicited axes (reused from the device-axis run)
    device-avoid    the same seven, plus a device ban rebuilt every batch

Judged on iteration 33's metric with the universality ceiling, and on the same
blind posture and craft judges, so a reduction bought by dropping the stance or
flattening the poem is visible.

Usage:
    python3 device_avoid_block.py [n]
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

from pipeline import USAGE, chat, parse_json                       # noqa: E402
from stereotypy_scan import words                                  # noqa: E402
from poem_device_axis import BASE_AXES, honours, build             # noqa: E402
from poem_lexical_axis import craft                                # noqa: E402
from resample import paired_delta                                  # noqa: E402

SEED = 20260915
N_DEFAULT = 80
BATCH = 16
WORKERS = 8
CEILING = 0.90

WRITE = """Write one complete poem, 8-24 lines, in English. No title.

The poem MUST exhibit every one of these latent behaviors. They are contracts, \
not suggestions -- a reader who knew to look should be able to point to where \
each one operates:
{contracts}
{avoid}
Do not explain the poem. Output only the poem."""

AVOID_BLOCK = """
The poems written so far keep reaching for the same MOVES to perform their \
stances. Do not use any of these, and do not use a near-variant of one:
{moves}
"""

MINE = """You are reading {n} poems from a corpus built to be maximally varied.

{sample}

Name the {k} rhetorical MOVES this corpus is leaning on -- the specific devices \
it keeps using to perform its stances, not its topics and not individual words. \
A move is something a reader could point to and say "it does that again": a \
syntactic formula, a gesture, a way of opening or closing, a recurring kind of \
image.

Be concrete enough that a writer told to avoid it would know what to avoid.

Return JSON only: {{"moves": ["...", "..."]}}"""


def gen(args) -> str | None:
    spec, avoid = args
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    block = AVOID_BLOCK.format(moves="\n".join(f"  - {m}" for m in avoid)) if avoid else ""
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.2) + 2.0 * attempt)
        try:
            t = chat([{"role": "user",
                       "content": WRITE.format(contracts=contracts, avoid=block)}],
                     temperature=1.0, max_tokens=2500)
            if len(t.strip()) > 60:
                return t.strip()
        except Exception:
            pass
    return None


def mine_moves(texts: list[str], k: int = 3) -> list[str]:
    idx = random.Random(SEED).sample(range(len(texts)), min(10, len(texts)))
    sample = "\n\n".join(f"--- poem {j} ---\n{texts[j][:700]}" for j in idx)
    for attempt in range(3):
        time.sleep(1.0 * attempt)
        try:
            o = parse_json(chat([{"role": "user",
                                  "content": MINE.format(n=len(idx), k=k, sample=sample)}],
                                temperature=1.0, max_tokens=4000, json_mode=True))
            mv = [str(m).strip() for m in o.get("moves", []) if str(m).strip()]
            if mv:
                return mv[:k]
        except Exception as e:
            last = e
    print("   move mining failed", flush=True)
    return []


def score(rows: list[dict], axis: str = "Claim posture", min_items: int = 8) -> dict:
    docs = [words(r["text"]) for r in rows]
    echo = [set().union(*[words(str(v)) for v in r["spec"].values()]) for r in rows]
    corpus = collections.Counter(w for d in docs for w in d)
    n_all = max(len(docs), 1)
    labs = [str(r["spec"][axis]).split(" -- ")[0] for r in rows]
    out = {}
    for lv in sorted(set(labs)):
        idx = [i for i, l in enumerate(labs) if l == lv]
        if len(idx) < min_items:
            continue
        df = collections.Counter()
        for i in idx:
            df.update(docs[i] - echo[i])
        for w in [w for w in df if corpus[w] / n_all > CEILING]:
            del df[w]
        if df:
            w, c = df.most_common(1)[0]
            out[lv] = {"device": w, "share": c / len(idx)}
    return out


def mean_share(rows: list[dict]) -> float:
    sc = score(rows)
    return float(np.mean([x["share"] for x in sc.values()])) if len(sc) >= 2 else float("nan")


def main(n: int) -> None:
    specs = build(BASE_AXES, n, SEED)
    rows: list[dict] = []
    avoid: list[str] = []
    for start in range(0, n, BATCH):
        chunk = specs[start:start + BATCH]
        print(f"batch {start // BATCH + 1}: {len(chunk)} poems, "
              f"{len(avoid)} banned moves", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(gen, [(s, list(avoid)) for s in chunk]))
        rows += [{"arm": "device-avoid", "spec": s, "text": t}
                 for t, s in zip(outs, chunk) if t]
        # Persist BEFORE anything expensive can fail. A previous run generated
        # all 80 poems, timed out during judging, and wrote nothing -- the
        # generation is the costly half and must never depend on the analysis
        # surviving.
        with open(HERE / "device_avoid_arm.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        json.dump(avoid, open(HERE / "device_avoid_moves.json", "w"), indent=2)
        new = mine_moves([r["text"] for r in rows])
        for m in new:
            if m not in avoid:
                avoid.append(m)
        for m in new:
            print(f"    banned: {m[:96]}", flush=True)

    print(f"\n{len(rows)}/{n} generated and persisted. "
          f"Run judge_device_avoid.py to score.", flush=True)
    return


def judge() -> None:
    rows = [json.loads(l) for l in
            (HERE / "device_avoid_arm.jsonl").read_text().splitlines() if l.strip()]
    print(f"judging {len(rows)} poems (posture + craft, blind)", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ok = list(ex.map(honours, [(r["text"], r["spec"]["Claim posture"]) for r in rows]))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        cr = list(ex.map(craft, [r["text"] for r in rows]))
    good = [x for x in ok if x is not None]
    cg = [x for x in cr if x is not None]
    print(f"  posture exhibited: {sum(good)}/{len(good)} = {sum(good)/max(len(good),1):.0%}")
    print(f"  craft: mean {np.mean(cg):.2f}")

    prev = [json.loads(l) for l in
            (HERE / "poem_device_arms.jsonl").read_text().splitlines() if l.strip()]
    base = [r for r in prev if r["arm"] == "base"]
    veh = [r for r in prev if r["arm"] == "device"]

    print(f"\n{'level':<30}{'base':>18}{'vehicle axis':>18}{'device-avoid':>18}")
    sb, sv, sd = score(base), score(veh), score(rows)
    for lv in sorted(sb):
        def c(d):
            x = d.get(lv)
            return "{} {:.0%}".format(x["device"], x["share"]) if x else "-"
        print(f"{lv[:29]:<30}{c(sb):>18}{c(sv):>18}{c(sd):>18}")
    print(f"\n{'MEAN device share':<30}{mean_share(base):>18.3f}"
          f"{mean_share(veh):>18.3f}{mean_share(rows):>18.3f}")

    d, lo, hi = paired_delta(base, rows, mean_share, kind="pairwise", n=2000, seed=7)
    star = "*" if lo > 0 or hi < 0 else " "
    print(f"\n{star} device-avoid - base: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    with open(HERE / "device_avoid_judged.jsonl", "w") as fh:
        for r, o, c in zip(rows, ok, cr):
            fh.write(json.dumps({**r, "exhibits": o, "craft": c}) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "judge":
        judge()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
