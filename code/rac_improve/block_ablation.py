"""
Which block actually suppresses the medium habit?

Iteration 38 compared contracts-only image instructions against the FULL
pipeline and credited the drop -- "mixed-media" in 55% of instructions against
22% -- to the avoid-block. That comparison isolates nothing. The pipeline's
prompt carries three blocks, and two of them were in the "full" arm and absent
from the "contracts only" arm:

    {contracts}   the axis spec
    {avoid}       mined semantic ATTRACTORS -- what a judge says the images keep
                  sharing, accumulated over the run
    {hard}        measured structural bans: no grid or tiling, no repeated
                  layout, no dominant hue, no readable text, and a list of the
                  eight most OVERUSED WORDS in the corpus so far

The word-level ban lives in `hard`, not in `avoid`, so attributing a word's
suppression to the avoid-block was wrong twice over. Iteration 39 supplied the
first evidence: an adaptive ban on rhetorical moves, which is what the
avoid-block is for, reproduced none of the effect.

This runs the ablation. Four arms over the same axes and the same prior corpus,
differing only in which blocks the prompt carries.

    contracts     the axis spec alone
    +avoid        plus mined attractors
    +hard         plus the structural and overused-word bans
    +both         the full prompt

Both derived blocks are built from the SAME prior corpus -- the 157
contracts-only instructions already on disk -- so the arms differ in what the
prompt says and in nothing else.

Usage:
    python3 block_ablation.py [n_per_arm]
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from pipeline import USAGE, chat, parse_json            # noqa: E402
from structural import overused_words                   # noqa: E402
from probe_image_seam import final_axes                 # noqa: E402
from image_stereotypy import build, score               # noqa: E402

MEDIA = {
    "photo": r"photograph|gelatin|film still|darkroom|lens|c-print",
    "collage": r"collage|montage|cut[- ]?paper|torn paper|assemblage",
    "painting": r"oil paint|acrylic|gouache|watercolou?r|fresco|tempera|canvas",
    "print": r"lithograph|etching|risograph|letterpress|woodcut|screen[- ]?print|silkscreen",
    "digital": r"digital|render|3d|cgi|vector|generative",
    "sculpture": r"sculpt|installation|diorama|relief|cast ",
    "textile": r"textile|weav|embroider|tapestry|quilt",
    "drawing": r"graphite|charcoal|ink drawing|pen and ink|pastel",
}


def medium_entropy(texts: list[str]) -> tuple[float, str, float]:
    """Spread over medium categories.

    Reported instead of the raw mixed-media rate because the hot-word ban
    contains "mixed" and "media" themselves: measuring that phrase in an arm
    that forbids those words scores compliance with the ban, not the breadth of
    the corpus. Which OTHER medium the generator reaches for is not dictated by
    the ban, so its spread is a fair comparison across arms.
    """
    c = collections.Counter()
    for t in texts:
        hit = [m for m, pat in MEDIA.items() if re.search(pat, t, re.I)]
        for m in hit:
            c[m] += 1
    if not c:
        return float("nan"), "-", float("nan")
    p = np.array([c.get(m, 0) for m in MEDIA], dtype=float)
    p = p / p.sum()
    q = p[p > 0]
    ent = float(-(q * np.log(q)).sum() / np.log(len(MEDIA)))
    top, n = c.most_common(1)[0]
    return ent, top, n / max(len(texts), 1)

SEED = 20260916
N_DEFAULT = 110
WORKERS = 8
RUN = RESEARCH / "real" / "dalle_steer7_maxmin"
PRIOR = HERE / "image_stereotypy_corpus.jsonl"

WRITE = """Write ONE image-generation instruction for a post-modern artwork. \
One paragraph, 30-70 words, concrete and visual.

Let these choices shape the work. Express each as something visible, never as \
abstract art-theory:
{contracts}
{avoid}{hard}
Name the medium, the composition, the palette, the surface, and the treatment. \
Output only the instruction."""

AVOID_TMPL = """
The set so far keeps doing these things. Avoid all of them:
{items}
"""

HARD_TMPL = """
Hard constraints, non-negotiable:
{items}
"""

MINE = """These art-generation instructions were written to differ from one another. \
Say what the resulting pictures would keep SHARING -- palettes, recurring objects, \
compositions, textures, gestures toward "post-modern".

{sample}

Return JSON only: {{"attractors": ["...", "..."]}} -- 4-8 concrete findings."""


def mine_attractors(texts: list[str]) -> list[str]:
    idx = random.Random(SEED).sample(range(len(texts)), min(14, len(texts)))
    sample = "\n\n".join(f"--- {j} ---\n{texts[j][:300]}" for j in idx)
    for attempt in range(3):
        time.sleep(1.0 * attempt)
        try:
            o = parse_json(chat([{"role": "user", "content": MINE.format(sample=sample)}],
                                temperature=1.0, max_tokens=4000, json_mode=True))
            a = [str(x).strip() for x in o.get("attractors", []) if str(x).strip()]
            if a:
                return a[:8]
        except Exception:
            pass
    return []


def build_hard(texts: list[str]) -> list[str]:
    """The pipeline's hard block, rebuilt from a prior corpus.

    The fixed structural bans plus the eight most overused words, with the
    template constants the pipeline itself excludes -- it drops create/use/
    using/make for the same reason the stereotypy metric needs a universality
    ceiling: a word in every instruction distinguishes nothing.
    """
    lines = [
        "Do NOT compose the work as a grid, tiling, array, or rows of repeated "
        "cells, panels, stamps, or posters. One continuous composition.",
        "No readable text, letters, numbers, or typography anywhere in the "
        "artwork, unless one of the contracts above explicitly requires text.",
    ]
    hot = [w for w, _ in overused_words(texts, min_df=0.4)][:12]
    hot = [w for w in hot if w not in ("create", "use", "using", "make")][:8]
    if hot:
        lines.append("Do not use or depict any of these overused elements: "
                     + ", ".join(hot) + ".")
    return lines, hot


def gen(args) -> str | None:
    spec, avoid, hard = args
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    ab = AVOID_TMPL.format(items="\n".join(f"  - {a}" for a in avoid)) if avoid else ""
    hb = HARD_TMPL.format(items="\n".join(f"  - {h}" for h in hard)) if hard else ""
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.2) + 2.0 * attempt)
        try:
            # 3000, not 600. At 600 the model reasons past the cap before
            # emitting anything and returns "", and it does so MORE OFTEN the
            # more blocks the prompt carries -- 104/92/75/34 of 110 across the
            # four arms. That is differential attrition correlated with the
            # treatment, which voids the comparison entirely, and it is the
            # third time in this work that a token cap consumed by reasoning
            # has looked like a refusal or an inability.
            t = chat([{"role": "user", "content": WRITE.format(
                contracts=contracts, avoid=ab, hard=hb)}], temperature=1.0,
                max_tokens=3000)
            if len(t.strip()) > 40:
                return t.strip()
        except Exception:
            pass
    return None


def main(n: int) -> None:
    prior = [json.loads(l)["text"] for l in PRIOR.read_text().splitlines() if l.strip()]
    print(f"prior corpus: {len(prior)} contracts-only instructions", flush=True)
    attractors = mine_attractors(prior)
    hard_lines, hot = build_hard(prior)
    print(f"mined {len(attractors)} attractors; banned words: {', '.join(hot)}\n",
          flush=True)

    axes = final_axes(RUN)
    specs = build(axes, n, SEED)
    arms = {"contracts": ([], []), "+avoid": (attractors, []),
            "+hard": ([], hard_lines), "+both": (attractors, hard_lines)}

    res = {}
    for name, (av, hd) in arms.items():
        print(f"arm '{name}': {n} instructions", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(gen, [(s, av, hd) for s in specs]))
        rows = [{"spec": s, "text": t} for t, s in zip(outs, specs) if t]
        with open(HERE / f"ablation_{name.strip('+')}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        shares = []
        for a in sorted(axes):
            sc = score(rows, a)
            if len(sc) >= 2:
                shares.append(float(np.mean([x["share"] for x in sc.values()])))
        ent, top, share = medium_entropy([r["text"] for r in rows])
        res[name] = {"n": len(rows), "device_share": float(np.mean(shares)) if shares else float("nan"),
                     "medium_entropy": ent, "top_medium": top, "top_share": share}
        print(f"  {len(rows)}/{n} written   device share "
              f"{np.mean(shares) if shares else float('nan'):.3f}   "
              f"medium entropy {ent:.3f} (top {top} {share:.0%})", flush=True)

    print(f"\n{'arm':<12}{'n':>5}{'device share':>15}{'medium entropy':>16}"
          f"{'top medium':>18}")
    for k, v in res.items():
        print(f"{k:<12}{v['n']:>5}{v['device_share']:>15.3f}"
              f"{v['medium_entropy']:>16.3f}"
              f"{f'{v[chr(116)+chr(111)+chr(112)+chr(95)+chr(109)+chr(101)+chr(100)+chr(105)+chr(117)+chr(109)]} {v[chr(116)+chr(111)+chr(112)+chr(95)+chr(115)+chr(104)+chr(97)+chr(114)+chr(101)]:.0%}':>18}")
    base = res["contracts"]
    print(f"\n{'arm':<12}{'delta device share':>20}{'delta medium entropy':>22}")
    for k, v in res.items():
        if k == "contracts":
            continue
        print(f"{k:<12}{v['device_share']-base['device_share']:>+20.3f}"
              f"{v['medium_entropy']-base['medium_entropy']:>+22.3f}")
    lo = min(v["n"] for v in res.values()); hi = max(v["n"] for v in res.values())
    print(f"\nwrite rate {lo}-{hi} of {n} per arm"
          + ("   -- comparable" if hi - lo <= 0.1 * n else
             "   -- DIFFERENTIAL ATTRITION, comparison not valid"))
    json.dump({"arms": res, "attractors": attractors, "hot_words": hot},
              open(HERE / "block_ablation.json", "w"), indent=2)
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
