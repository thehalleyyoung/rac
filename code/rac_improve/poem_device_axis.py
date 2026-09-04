"""
The device-axis fix, carried from test items to poems.

Iteration 32 found an axis level realized through one device -- the
psychometric red herring was nearly always a coloured card -- and fixed it with
an axis enumerating the forms the device could take: device entropy 0.425 to
0.933, contract honoured 100% in both arms. Iteration 33 showed the defect is
general and WORSE in poems, whose levels average 0.533 on the share of items
using their single most frequent non-echo word against 0.412 for the bank.

The worst is `Claim posture: ritual pronouncement`, and reading its 37 poems
shows exactly the psychometric shape. The posture is performed by one liturgical
formula with a fixed stock of props:

    "By ash, by antenna, by the names"
    "By the gathered breath, this name is consecrated"
    "By salt and smoke, I seal this house within"

A level names a stance; the generator picks one way to perform it; nothing in
the objective or the axis scoring notices it performs it identically every time.

The fix has to be more general here than it was for items. A red herring's
FORMS are a closed list, but a poem's stance can be delivered by any rhetorical
vehicle, and a device axis tied to one posture would not compose with the other
four. So the axis enumerates VEHICLES -- the syntactic and rhetorical means by
which any stance is carried -- which is orthogonal to which stance is being
carried, and therefore composes with every level of every posture axis.

  base     seven elicited poem axes
  device   the same seven plus one vehicle axis, eight levels

Scoring uses iteration 33's validated metric (share of a level's items using its
most frequent non-echo content word), and both arms are judged for whether the
commanded posture is still performed -- a fix that varies the vehicle while
losing the stance would be no fix.

Usage:
    python3 poem_device_axis.py [n_per_arm]
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

from pipeline import USAGE, chat, parse_json          # noqa: E402
from stereotypy_scan import stereotypy, words         # noqa: E402

SEED = 20260910
N_DEFAULT = 80
WORKERS = 8

BASE_AXES = {
    "Claim posture": [
        "Ritual pronouncement", "Provisional inquiry", "Flat assertion",
        "Confession under pressure", "Refusal to claim"],
    "Syntactic weather": [
        "Nested subordinate clauses", "Elliptical syntactic gaps",
        "Long accumulating parataxis", "Clipped declaratives",
        "Self-interrupting recursion"],
    "Register contract": [
        "Ceremonial elevation", "Conversational plainness",
        "Specialist terminology", "Documentary flatness",
        "Register collision within the line"],
    "Temporal stance": [
        "Immediate unfolding", "Retrospective reconstruction",
        "Anticipatory projection", "Cyclical recurrence",
        "Counterfactual time"],
    "Voice agency": [
        "Stable first-person witness", "Adopted dramatic persona",
        "Impersonal distributed voice", "Second-person conscription",
        "Voice that disclaims its own authority"],
    "Address geometry": [
        "Inward monologue", "Direct intimate address", "Public declaration",
        "Address to absent or dead", "Shifting audience"],
    "Lineation function": [
        "End-stopped grammatical closure", "Enjambment against syntax",
        "Visual field construction", "Breath-unit phrasing",
        "Fragmentary isolation"],
}

# Vehicles, not stances: the rhetorical means by which ANY posture is carried.
# Orthogonal to which posture is commanded, so it composes with all five levels
# of Claim posture rather than fixing one of them.
VEHICLE_AXIS = {
    "Rhetorical vehicle": [
        "Carry it through an imperative given to someone",
        "Carry it through a question the poem does not answer",
        "Carry it through negation -- saying what is not the case",
        "Carry it through a bare concrete object, with no statement about it",
        "Carry it through a reported speech act someone else performed",
        "Carry it through a conditional or hypothetical construction",
        "Carry it through an enumeration or inventory",
        "Carry it through a correction of something said earlier in the poem",
    ]
}

WRITE = """Write one complete poem, 8-24 lines, in English. No title.

The poem MUST exhibit every one of these latent behaviors. They are contracts, \
not suggestions -- a reader who knew to look should be able to point to where \
each one operates:
{contracts}

Do not explain the poem. Output only the poem."""

CONTRACT_JUDGE = """You are reading one poem.

Does this poem exhibit the following stance in how it makes its claims?

    {level}

Answer only about whether that stance is present, not about quality.

Poem:
{poem}

Return JSON only: {{"exhibits": true or false}}"""


def gen(spec: dict) -> str | None:
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.2) + 2.0 * attempt)
        try:
            t = chat([{"role": "user", "content": WRITE.format(contracts=contracts)}],
                     temperature=1.0, max_tokens=2500)
            if len(t.strip()) > 60:
                return t.strip()
        except Exception:
            pass
    return None


def honours(args) -> bool | None:
    text, level = args
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.0) + 1.5 * attempt)
        try:
            raw = chat([{"role": "user",
                         "content": CONTRACT_JUDGE.format(level=level, poem=text[:2500])}],
                       temperature=0.0, max_tokens=2000, json_mode=True)
            v = parse_json(raw)["exhibits"]
            if isinstance(v, bool):
                return v
        except Exception:
            pass
    return None


def build(axes: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    used = {a: collections.Counter() for a in axes}
    out = []
    for _ in range(n):
        s = {}
        for a, lv in axes.items():
            lo = min(used[a].get(x, 0) for x in lv)
            s[a] = rng.choice([x for x in lv if used[a].get(x, 0) == lo])
            used[a][s[a]] += 1
        out.append(s)
    return out


def score(texts: list[str], specs: list[dict], axis: str) -> dict:
    docs = [words(t) for t in texts]
    labs = [str(s[axis]) for s in specs]
    return stereotypy(docs, labs)


def main(n: int) -> None:
    arms = {"base": BASE_AXES, "device": {**BASE_AXES, **VEHICLE_AXIS}}
    store = {}
    for name, axes in arms.items():
        specs = build(axes, n, SEED)
        print(f"\narm '{name}': {n} poems, {len(axes)} axes", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(gen, specs))
        keep = [(t, s) for t, s in zip(outs, specs) if t]
        texts = [t for t, _ in keep]
        sp = [s for _, s in keep]
        print(f"  {len(texts)}/{n} generated; judging the posture contract blind",
              flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            ok = list(ex.map(honours, [(t, s["Claim posture"]) for t, s in keep]))
        good = [x for x in ok if x is not None]
        store[name] = (texts, sp, ok)
        print(f"  posture exhibited: {sum(good)}/{len(good)} = "
              f"{sum(good)/max(len(good),1):.0%}")

    print(f"\n{'axis':<22}{'base mean':>11}{'device mean':>13}{'delta':>9}")
    rows = {}
    for a in BASE_AXES:
        vb = score(*store["base"][:2], a)
        vd = score(*store["device"][:2], a)
        if len(vb) < 2 or len(vd) < 2:
            continue
        mb = float(np.mean([x["share"] for x in vb.values()]))
        md = float(np.mean([x["share"] for x in vd.values()]))
        rows[a] = (mb, md, vb, vd)
        print(f"{a[:21]:<22}{mb:>11.3f}{md:>13.3f}{md-mb:>+9.3f}")
    mb = float(np.mean([v[0] for v in rows.values()]))
    md = float(np.mean([v[1] for v in rows.values()]))
    print(f"{'ALL AXES':<22}{mb:>11.3f}{md:>13.3f}{md-mb:>+9.3f}")

    if "Claim posture" in rows:
        _, _, vb, vd = rows["Claim posture"]
        print("\nClaim posture, per level (device word, share of that level):")
        print(f"{'level':<32}{'base':>22}{'device':>22}")
        for lv in sorted(set(vb) | set(vd)):
            b = vb.get(lv, {}); d = vd.get(lv, {})
            print(f"{lv[:31]:<32}"
                  f"{(b.get('device') or '-') + f' {b.get(chr(115)+chr(104)+chr(97)+chr(114)+chr(101), 0):.0%}':>22}"
                  f"{(d.get('device') or '-') + f' {d.get(chr(115)+chr(104)+chr(97)+chr(114)+chr(101), 0):.0%}':>22}")

    rng = np.random.default_rng(SEED)
    tb, sb = store["base"][0], store["base"][1]
    td, sd = store["device"][0], store["device"][1]
    diffs = []
    for _ in range(2000):
        ib = rng.integers(0, len(tb), len(tb)); idd = rng.integers(0, len(td), len(td))
        vb = score([tb[i] for i in ib], [sb[i] for i in ib], "Claim posture")
        vd = score([td[i] for i in idd], [sd[i] for i in idd], "Claim posture")
        if len(vb) >= 2 and len(vd) >= 2:
            diffs.append(np.mean([x["share"] for x in vd.values()])
                         - np.mean([x["share"] for x in vb.values()]))
    if diffs:
        d = np.array(diffs); lo, hi = np.percentile(d, [2.5, 97.5])
        s = "*" if lo > 0 or hi < 0 else " "
        print(f"\n{s} Claim posture device share, device - base: "
              f"{d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    with open(HERE / "poem_device_arms.jsonl", "w") as fh:
        for nm, (ts, sp, ok) in store.items():
            for t, s_, o in zip(ts, sp, ok):
                fh.write(json.dumps({"arm": nm, "spec": s_, "exhibits": o,
                                     "text": t}) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
