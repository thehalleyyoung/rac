"""
Does the seed prompt elicit the axes, or hand them over?

Four model families were asked for seven axes of poetic variation with the same
prompt, and all four returned essentially the same six dimensions under
different names -- temporal stance, address, the poem's relation to its own
claim, syntactic weather, line-break function, register. That reads as
convergence on something real about the domain.

The prompt says this:

    go for axes a working poet would recognise as real craft decisions:
    temporal stance, who is being addressed, the poem's relationship to its own
    claim, syntactic weather, what the line break is doing, the register the
    poem refuses, and so on.

Six examples; six shared axes. The convergence may be agreement about poems, or
it may be four models returning the list they were shown. Those have opposite
consequences for the method: the first makes elicitation a discovery step, the
second makes it retrieval, and every recursion downstream then refines a space
the prompt author chose.

Two conditions, same models, same k:

    with     the published prompt
    without  identical except the six worked examples are removed, leaving the
             criteria (orthogonal, structural, all levels usable) and the
             instruction to avoid surface axes

Measured by how many of each returned axis set matches one of the six seeded
dimensions, judged by a keyword map fixed in advance rather than by reading the
results afterwards.

Usage:
    python3 does_the_prompt_supply_the_axes.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))

import pipeline                                   # noqa: E402
from pipeline import USAGE, chat, parse_json      # noqa: E402
from axes import SEED_AXIS_PROMPT                 # noqa: E402

K = 7
REPS = 3
MODELS = ["openai/gpt-5.6-luna", "meta-llama/llama-3.3-70b-instruct",
          "qwen/qwen-2.5-72b-instruct", "deepseek/deepseek-chat"]

EXAMPLES = ("go for axes a working poet would recognise as real craft "
            "decisions: temporal stance, who is being addressed, the poem's "
            "relationship to its own claim, syntactic weather, what the line "
            "break is doing, the register the poem refuses, and so on.")
REPLACEMENT = ("go for axes a working poet would recognise as real craft "
               "decisions rather than surface features of the finished text.")

# Fixed before the results were read: the six dimensions the prompt names.
SEEDED = {
    "temporal": r"tempor|time|tense|chronolog|when",
    "address": r"address|audience|reader|listener|apostroph|second person",
    "claim": r"claim|assert|truth|reality|epistem|knowledge|certain|stance toward",
    "syntax": r"syntact|syntax|clause|sentence|grammar|density|complexity",
    "line": r"line[- ]?break|lineation|enjamb|line function|line logic",
    "register": r"register|diction|tone|formality|idiom",
}


def elicit(model: str, prompt: str) -> list[str]:
    orig = pipeline.MODEL
    pipeline.MODEL = model
    try:
        for attempt in range(3):
            time.sleep(1.0 * attempt)
            try:
                obj = parse_json(chat([{"role": "user", "content": prompt}],
                                      temperature=1.0, max_tokens=6000,
                                      json_mode=True))
                names = [str(a.get("name", "")).strip()
                         for a in obj.get("axes", []) if a.get("name")]
                if len(names) >= 4:
                    return names
            except Exception:
                pass
        return []
    finally:
        pipeline.MODEL = orig


def seeded_hits(names: list[str]) -> tuple[int, list[str]]:
    hit = set()
    for n in names:
        for k, pat in SEEDED.items():
            if re.search(pat, n, re.I):
                hit.add(k)
    return len(hit), sorted(hit)


def main() -> None:
    base = SEED_AXIS_PROMPT.format(k=K)
    assert EXAMPLES.split(":")[1][:30].strip() in base, "example list not found"
    stripped = base.replace(EXAMPLES, REPLACEMENT)
    print(f"prompt with examples:    {len(base)} chars")
    print(f"prompt without examples: {len(stripped)} chars\n")

    rows = []
    for model in MODELS:
        for cond, p in (("with", base), ("without", stripped)):
            for r in range(REPS):
                names = elicit(model, p)
                if not names:
                    continue
                n_hit, which = seeded_hits(names)
                rows.append({"model": model.split("/")[-1], "cond": cond,
                             "rep": r, "axes": names, "seeded": n_hit,
                             "which": which})
                print(f"{model.split('/')[-1][:22]:<24}{cond:<9}"
                      f"{n_hit}/6 seeded   {', '.join(names)[:78]}", flush=True)

    print(f"\n{'model':<24}{'with':>10}{'without':>10}{'drop':>8}")
    import numpy as np
    for model in MODELS:
        m = model.split("/")[-1]
        w = [r["seeded"] for r in rows if r["model"] == m and r["cond"] == "with"]
        o = [r["seeded"] for r in rows if r["model"] == m and r["cond"] == "without"]
        if w and o:
            print(f"{m[:23]:<24}{np.mean(w):>10.2f}{np.mean(o):>10.2f}"
                  f"{np.mean(w)-np.mean(o):>+8.2f}")
    W = [r["seeded"] for r in rows if r["cond"] == "with"]
    O = [r["seeded"] for r in rows if r["cond"] == "without"]
    print(f"{'ALL':<24}{np.mean(W):>10.2f}{np.mean(O):>10.2f}"
          f"{np.mean(W)-np.mean(O):>+8.2f}")
    print(f"\nof six seeded dimensions, mean recovered per elicitation:")
    print(f"  with the examples in the prompt   {np.mean(W):.2f}")
    print(f"  with them removed                 {np.mean(O):.2f}")
    json.dump(rows, open(HERE / "prompt_supplies_axes.json", "w"), indent=2)
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main()
