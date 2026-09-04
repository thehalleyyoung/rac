"""
Generate our instruction-domain candidate pools (naive, high-temp, conditioned).

Design note: the method is propose-measure-SELECT, so generation and selection
are separable and we separate them here. This script only PROPOSES -- it emits
three candidate pools of equal generation budget:

  naive      unconditioned repeated prompting, T = 1.0
  high_temp  same prompt, T = 1.6
  cond       axis-conditioned: a spec (one level per elicited axis) per call,
             with a periodically-mined attractor ledger as negative constraints

Selection (coverage-greedy over the conditioned pool) happens offline in
instruction_h2h.py, which lets us report the axes contribution and the
selection contribution separately, and lets us state the generation budget
each arm consumed rather than hiding it behind matched n.

Generation is parallel because nothing here is sequential: no candidate's
prompt depends on another candidate's outcome (only on the axis set and the
ledger, both of which update on a fixed schedule). All output is append-only
JSONL, so the script is resumable.

Run:  python3 instruction_gen.py [n_per_arm]
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pipeline import chat, embed, parse_json, USAGE  # noqa: E402

OUT = HERE / "instruction_logs"
N_PER_ARM = int(sys.argv[1]) if len(sys.argv) > 1 else 750
WORKERS = 12
LEDGER_EVERY = 150
N_AXES = 7

NAIVE_PROMPT = (
    "Write one instruction or task prompt for an AI assistant. It should be "
    "self-contained and answerable without extra context. One to three "
    "sentences. Output only the instruction."
)

SEED_AXES = (
    "You are designing the dimensions of variation for a very large corpus "
    "of INSTRUCTIONS given to an AI assistant. Do not write any "
    "instructions.\n\n"
    "Propose {k} INDEPENDENT axes of variation. A good axis is close to "
    "orthogonal to the others, changes something structural about what the "
    "instruction DEMANDS rather than merely its topic, and has levels that "
    "are all genuinely usable.\n\n"
    "Avoid the obvious surface axes (subject area, length, difficulty "
    "alone). Reach for real task-design decisions: the cognitive operation "
    "required, what the output format has to be, who the assistant is "
    "asked to be accountable to, whether the task is under- or "
    "over-specified, what has to be REFUSED or held back, how many "
    "constraints interact, whether success is checkable, what the "
    "instruction assumes the assistant already knows.\n\n"
    'Return JSON only: {{"axes": [{{"name": "...", "why_orthogonal": "...", '
    '"levels": ["...", "...", "...", "..."]}}]}}\n'
    "Each axis needs 4-6 levels. Keep why_orthogonal under 15 words and each "
    "level name under 10 words."
)

COND_PROMPT = (
    "Write one instruction or task prompt for an AI assistant. It should be "
    "self-contained and answerable without extra context. One to three "
    "sentences.\n\n"
    "The instruction MUST exhibit every one of these latent design behaviors "
    "-- they are contracts, not suggestions:\n{contracts}\n\n"
    "{avoid}"
    "Output only the instruction."
)

MINE_PROMPT = (
    "Here are instructions from a corpus being built to be maximally diverse:\n\n"
    "{items}\n\n"
    "What do they have in COMMON? List the recurring task types, framings, "
    "topics, verbs, and sentence shapes -- the attractors the generator keeps "
    'falling back into. Return JSON only: {{"attractors": ["...", "..."]}} '
    "(6-10 short phrases, most dominant first)."
)


def append_jsonl(path: Path, recs: list[dict]):
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def gen_one(prompt: str, temperature: float, max_tokens: int = 1500) -> str | None:
    """Note on max_tokens: this model bills reasoning against the same budget,
    so a limit sized for the visible answer alone returns an EMPTY content
    field once the prompt gets complex (the ledger avoid-list did exactly
    that: cost kept accruing while accepted-item yield collapsed to ~2 per
    48-call batch). Size the budget for reasoning + answer, not the answer."""
    try:
        t = chat([{"role": "user", "content": prompt}],
                 temperature=temperature, max_tokens=max_tokens).strip()
        t = t.strip('"').strip()
        return t if len(t) >= 15 else None
    except Exception:
        return None


def run_plain(arm: str, temperature: float, n: int):
    path = OUT / f"{arm}.jsonl"
    have = read_jsonl(path)
    print(f"{arm}: {len(have)}/{n} present", flush=True)
    while len(have) < n:
        need = min(WORKERS * 4, n - len(have))
        with ThreadPoolExecutor(WORKERS) as ex:
            got = list(ex.map(lambda _: gen_one(NAIVE_PROMPT, temperature, 800),
                              range(need)))
        recs = [{"text": t, "arm": arm} for t in got if t]
        append_jsonl(path, recs)
        have.extend(recs)
        print(f"  {arm}: {len(have)}/{n}  (${USAGE.cost_usd():.3f})", flush=True)


def seed_axes() -> list[dict]:
    path = OUT / "axes.jsonl"
    have = read_jsonl(path)
    if have:
        return have[-1]["axes"]
    # a 7-axis reply with rationales overruns a small max_tokens and comes back
    # as truncated JSON; give it room and fall back to fewer axes if needed
    axes = []
    for k, mt in ((N_AXES, 4000), (N_AXES, 6000), (5, 4000)):
        try:
            obj = parse_json(chat([{"role": "user",
                                    "content": SEED_AXES.format(k=k)}],
                                  temperature=1.0, max_tokens=mt, json_mode=True))
            axes = [{"name": str(a["name"]),
                     "levels": [str(x) for x in a["levels"] if str(x).strip()]}
                    for a in obj["axes"] if len(a.get("levels", [])) >= 3]
            if len(axes) >= 4:
                break
        except (ValueError, KeyError) as e:
            print(f"  axis elicitation retry (k={k}, max_tokens={mt}): {e}")
    if len(axes) < 4:
        raise RuntimeError("axis elicitation failed")
    append_jsonl(path, [{"axes": axes}])
    print(f"axes: {[a['name'] for a in axes]}", flush=True)
    return axes


def run_conditioned(axes: list[dict], n: int):
    path = OUT / "cond.jsonl"
    ledger_path = OUT / "ledger.jsonl"
    have = read_jsonl(path)
    rng = random.Random(20260830 + len(have))
    ledger = read_jsonl(ledger_path)
    avoid = ledger[-1]["attractors"][:8] if ledger else []
    print(f"cond: {len(have)}/{n} present", flush=True)
    while len(have) < n:
        need = min(WORKERS * 4, n - len(have))
        specs = [{a["name"]: rng.choice(a["levels"]) for a in axes}
                 for _ in range(need)]
        avoid_txt = ""
        if avoid:
            avoid_txt = ("The corpus so far has been drifting toward these habits. "
                         "Yours must stay clear of all of them:\n"
                         + "\n".join(f"  - {a}" for a in avoid) + "\n\n")
        prompts = [COND_PROMPT.format(
            contracts="\n".join(f"  - {k}: {v}" for k, v in s.items()),
            avoid=avoid_txt) for s in specs]
        with ThreadPoolExecutor(WORKERS) as ex:
            got = list(ex.map(lambda p: gen_one(p, 1.0, 1500), prompts))
        recs = [{"text": t, "arm": "cond", "spec": s}
                for t, s in zip(got, specs) if t]
        append_jsonl(path, recs)
        have.extend(recs)
        print(f"  cond: {len(have)}/{n}  (${USAGE.cost_usd():.3f})", flush=True)
        if len(have) // LEDGER_EVERY > len(ledger):
            sample = [r["text"] for r in random.sample(have, min(50, len(have)))]
            try:
                obj = parse_json(chat(
                    [{"role": "user", "content": MINE_PROMPT.format(
                        items="\n".join(f"- {t}" for t in sample))}],
                    temperature=0.5, max_tokens=900, json_mode=True))
                attractors = [str(a) for a in obj.get("attractors", [])][:10]
                append_jsonl(ledger_path, [{"after_n": len(have),
                                            "attractors": attractors}])
                ledger = read_jsonl(ledger_path)
                avoid = attractors[:8]
                print(f"  ledger mined ({len(ledger)}): {avoid[:2]}", flush=True)
            except Exception as e:
                print("  ledger mining failed:", e)


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    axes = seed_axes()
    run_conditioned(axes, N_PER_ARM)
    run_plain("naive", 1.0, N_PER_ARM)
    run_plain("high_temp", 1.6, N_PER_ARM)

    # embed everything once, cached alongside
    for arm in ("cond", "naive", "high_temp"):
        recs = read_jsonl(OUT / f"{arm}.jsonl")
        ep = OUT / f"{arm}_emb.npy"
        if ep.exists() and len(np.load(ep)) == len(recs):
            continue
        E = embed([r["text"] for r in recs], batch=96)
        np.save(ep, E)
        print(f"embedded {arm}: {E.shape}", flush=True)

    usage = {"calls": USAGE.calls, "errors": USAGE.errors,
             "prompt_tokens": USAGE.prompt_tokens,
             "completion_tokens": USAGE.completion_tokens,
             "cost_usd": round(USAGE.cost_usd(), 4),
             "elapsed_sec": round(time.time() - t0, 1)}
    append_jsonl(OUT / "usage.jsonl", [usage])
    print(json.dumps(usage, indent=2))


if __name__ == "__main__":
    main()
