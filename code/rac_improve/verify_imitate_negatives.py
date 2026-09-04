"""
Does showing the imitate branch its own recent output stop it repeating itself?

Imitation accounts for the whole of the covering run's duplicate rate: 65.4% of
its calls returned text already in the corpus, against 0.000 for the axis
branch. The diagnosis is that a dense target gets hit repeatedly, the same three
neighbours are shown each time, and nothing tells the generator what it already
wrote there — while the axis branch has always passed its own nearby outputs as
explicit negatives.

Re-running the 24,000-call experiment to test the fix costs hours and dollars.
This tests the mechanism instead, on the neighbourhoods where it actually
failed. Take the templates that collapsed in the published run, reconstruct the
prompt the generator saw, and ask for k completions two ways:

  control     the published prompt, examples only
  treated     the same prompt plus the items already produced in that
              neighbourhood, as explicit negatives

Then measure how many of the k completions are distinct, and how many collide
with the published items. If the diagnosis is right, control repeats and treated
does not; if both repeat, the cause is the dense-core instruction itself and
the fix does not address it.

Usage:
    python3 verify_imitate_negatives.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
COVERAGE = RESEARCH / "coverage"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(COVERAGE))

from pipeline import USAGE, chat  # noqa: E402

K = 12          # completions per condition per neighbourhood
N_HOODS = 5     # collapsed template families to test
WORKERS = 12

CONTROL = """Here are examples of the kind of instruction people give an AI assistant:

{examples}

Write ONE more instruction of exactly this kind -- same territory, same task family, \
same plain register -- just a different concrete case of it. Not a rephrasing of any \
example: a new instance of the same kind of ask.

One or two sentences, the way a person actually types, self-contained. Output only \
the instruction."""

TREATED = """Here are examples of the kind of instruction people give an AI assistant:

{examples}

You have already written these for this same neighbourhood. Do not produce any of \
them again, and do not produce a near-copy that differs only in a name, a number or \
one swapped noun:
{negatives}

Write ONE more instruction of exactly this kind -- same territory, same task family, \
same plain register -- just a different concrete case of it. Not a rephrasing of any \
example: a new instance of the same kind of ask.

One or two sentences, the way a person actually types, self-contained. Output only \
the instruction."""


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower()).rstrip(".?!")


def ask(prompt: str) -> str | None:
    try:
        return chat([{"role": "user", "content": prompt}], 1.0, 400).strip()
    except Exception:
        return None


def main():
    recs = [json.loads(l) for l in
            (COVERAGE / "fair_logs" / "ours_v4_10x.jsonl").read_text().splitlines()
            if l.strip()]
    imit = [r["text"].strip() for r in recs
            if r.get("spec", {}).get("_mode") == "imitate"]

    # the collapsed families, by six-word opening
    fams = Counter(" ".join(t.split()[:6]).lower() for t in imit)
    hoods = [k for k, _ in fams.most_common(N_HOODS)]

    rows = []
    for h in hoods:
        members = [t for t in imit if " ".join(t.split()[:6]).lower() == h]
        uniq = list(dict.fromkeys(members))
        examples = "\n".join(f"- {t[:140]}" for t in uniq[:3])
        negatives = "\n".join(f"- {t[:120]}" for t in uniq[:2]) or "- (nothing yet)"
        published = {norm(t) for t in members}

        prompts = ([CONTROL.format(examples=examples)] * K
                   + [TREATED.format(examples=examples, negatives=negatives)] * K)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(ask, prompts))
        ctl = [o for o in outs[:K] if o]
        trt = [o for o in outs[K:] if o]

        def stats(xs):
            n = max(len(xs), 1)
            nx = [norm(x) for x in xs]
            return (len(set(nx)) / n, sum(1 for x in nx if x in published) / n)

        cu, cc = stats(ctl)
        tu, tc = stats(trt)
        rows.append({"family": h, "n_published": len(members),
                     "control_distinct": round(cu, 3), "control_collide": round(cc, 3),
                     "treated_distinct": round(tu, 3), "treated_collide": round(tc, 3)})
        print(f"\n{h[:52]!r}  ({len(members)} published)")
        print(f"   control: {cu:.0%} distinct, {cc:.0%} collide with published")
        print(f"   treated: {tu:.0%} distinct, {tc:.0%} collide with published",
              flush=True)

    import numpy as np
    print(f"\n{'':<12}{'distinct':>10}{'collide':>10}")
    for lab, a, b in (("control", "control_distinct", "control_collide"),
                      ("treated", "treated_distinct", "treated_collide")):
        print(f"{lab:<12}{np.mean([r[a] for r in rows]):>10.1%}"
              f"{np.mean([r[b] for r in rows]):>10.1%}")
    print(f"\nspend ${USAGE.cost_usd():.2f}")
    json.dump(rows, open(HERE / "imitate_negatives.json", "w"), indent=2)


if __name__ == "__main__":
    main()
