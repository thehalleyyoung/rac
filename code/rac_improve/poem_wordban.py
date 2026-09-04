"""
Does a token ban raise concentration in a second domain?

The block ablation found the pipeline's own hard block -- structural bans plus a
list of the eight most overused words -- RAISING within-level concentration in
image instructions rather than lowering it, from 0.564 to 0.653, with the
mechanism visible in what replaced what: *light* was banned and *illumination*
became the dominant device on fifty level-slots. Banning a word leaves the
concept, and the generator reaches for the nearest synonym. A prohibition names
a subset by construction, so it relocates concentration exactly as an axis does.

That was measured in one domain. If it is a rule rather than an image-domain
quirk, banning the poem corpus's most overused words should make poem stereotypy
worse too, and the substitutions should be legible.

The words come from the base corpus by the same routine the pipeline uses, with
two exclusions. `you` and `your` are the two most overused, and the *Address
geometry* axis commands direct address at three of its five levels: banning them
would break a contract rather than test displacement. Pure function words are
dropped for the same reason the metric needs a universality ceiling. What remains
is the corpus's own stock: *still, hand, name, record, rain, let, say, silence* --
seven of which the stereotypy scan had already identified as the dominant device
of some level, `let` at 88% of ritual pronouncements among them.

    base       the seven elicited axes (reused)
    +wordban   the same seven, plus a ban on those eight words

Posture and craft are judged blind, and the ban text is stripped from the echo
set so that no arm is scored on vocabulary its own prompt supplied.

Usage:
    python3 poem_wordban.py [n]
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

from pipeline import USAGE, chat                                # noqa: E402
from stereotypy_scan import words                               # noqa: E402
from poem_device_axis import BASE_AXES, honours, build          # noqa: E402
from poem_lexical_axis import craft                             # noqa: E402
from resample import paired_delta                               # noqa: E402

SEED = 20260917
N_DEFAULT = 80
WORKERS = 8
CEILING = 0.90

BANNED = ["still", "hand", "name", "record", "rain", "let", "say", "silence"]

WRITE = """Write one complete poem, 8-24 lines, in English. No title.

The poem MUST exhibit every one of these latent behaviors. They are contracts, \
not suggestions -- a reader who knew to look should be able to point to where \
each one operates:
{contracts}
{ban}
Do not explain the poem. Output only the poem."""

BAN_BLOCK = """
The poems written so far overuse these words. Do not use any of them:
  {words}
"""


def gen(args) -> str | None:
    spec, ban = args
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    block = BAN_BLOCK.format(words=", ".join(ban)) if ban else ""
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.2) + 2.0 * attempt)
        try:
            t = chat([{"role": "user",
                       "content": WRITE.format(contracts=contracts, ban=block)}],
                     temperature=1.0, max_tokens=2500)
            if len(t.strip()) > 60:
                return t.strip()
        except Exception:
            pass
    return None


def score(rows: list[dict], extra: str = "") -> dict:
    docs = [words(r["text"]) for r in rows]
    ex = words(extra)
    echo = [set().union(*[words(str(v)) for v in r["spec"].values()]) | ex
            for r in rows]
    corpus = collections.Counter(w for d in docs for w in d)
    n_all = max(len(docs), 1)
    labs = [str(r["spec"]["Claim posture"]).split(" -- ")[0] for r in rows]
    out = {}
    for lv in sorted(set(labs)):
        idx = [i for i, l in enumerate(labs) if l == lv]
        if len(idx) < 8:
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


def mean_share(rows: list[dict], extra: str = "") -> float:
    sc = score(rows, extra)
    return float(np.mean([x["share"] for x in sc.values()])) if len(sc) >= 2 \
        else float("nan")


def main(n: int) -> None:
    specs = build(BASE_AXES, n, SEED)
    print(f"+wordban arm: {n} poems, banning {', '.join(BANNED)}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(gen, [(s, BANNED) for s in specs]))
    rows = [{"arm": "wordban", "spec": s, "text": t}
            for t, s in zip(outs, specs) if t]
    with open(HERE / "poem_wordban_arm.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  {len(rows)}/{n} written and persisted", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ok = list(ex.map(honours, [(r["text"], r["spec"]["Claim posture"])
                                   for r in rows]))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        cr = list(ex.map(craft, [r["text"] for r in rows]))
    good = [x for x in ok if x is not None]
    cg = [x for x in cr if x is not None]
    print(f"  posture exhibited: {sum(good)}/{len(good)} = "
          f"{sum(good)/max(len(good),1):.0%}   craft {np.mean(cg):.2f}")

    prev = [json.loads(l) for l in
            (HERE / "poem_device_arms.jsonl").read_text().splitlines() if l.strip()]
    base = [r for r in prev if r["arm"] == "base"]
    ban_txt = " ".join(BANNED)

    # Both arms scored on the vocabulary available to BOTH. Scoring base on its
    # own larger vocabulary is not a comparison: its dominant devices are the
    # very words the other arm was forbidden, so it would be credited for using
    # what the treatment removes.
    sb, sw = score(base, ban_txt), score(rows, ban_txt)
    print(f"\n{'level':<30}{'base':>20}{'+wordban':>20}")
    for lv in sorted(sb):
        def c(d):
            x = d.get(lv)
            return "{} {:.0%}".format(x["device"], x["share"]) if x else "-"
        print(f"{lv[:29]:<30}{c(sb):>20}{c(sw):>20}")
    mb, mw = mean_share(base, ban_txt), mean_share(rows, ban_txt)
    print(f"\n{'MEAN device share':<30}{mb:>20.3f}{mw:>20.3f}")

    d, lo, hi = paired_delta(base, rows, lambda rs: mean_share(rs, ban_txt),
                             kind="pairwise", n=2000, seed=11)
    star = "*" if lo > 0 or hi < 0 else " "
    print(f"\n{star} +wordban - base: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    hit = sum(1 for r in rows if any(b in r["text"].lower() for b in BANNED))
    print(f"  ban obeyed: {len(rows)-hit}/{len(rows)} poems avoid all eight words")
    with open(HERE / "poem_wordban_judged.jsonl", "w") as fh:
        for r, o, c in zip(rows, ok, cr):
            fh.write(json.dumps({**r, "exhibits": o, "craft": c}) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
