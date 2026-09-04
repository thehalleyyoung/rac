"""
The current best-regarded approaches to diverse synthetic data generation,
implemented as live arms against gpt-5.6-luna on the same two domains.

These are not strawmen. Each is the mechanism that made the corresponding
paper work, reproduced as faithfully as a single-domain reimplementation
allows, and each gets the same generator, the same embedder, and the same
budget accounting as our method.

  self_instruct   Self-Instruct / Alpaca. Maintain a pool; prompt with a few
                  randomly sampled exemplars from it; reject a candidate whose
                  max ROUGE-L overlap against the pool exceeds 0.7. The
                  few-shot exemplars are the diversity mechanism (they push
                  the model off its default mode) and the ROUGE filter is the
                  redundancy guard. Note this filter is LITERAL, not semantic:
                  it is exactly the thing our n-gram/embedding split predicts
                  will control lexical but not semantic redundancy.

  evol_instruct   WizardLM's Evol-Instruct. Sample an existing item and apply
                  one randomly chosen evolution operator (deepen, concretize,
                  add a constraint, increase reasoning, or mutate to a new
                  form). Diversity comes from a random walk over the corpus
                  rather than from conditioning.

  persona         Persona-Hub / AttrPrompt style attribute conditioning. A
                  large flat catalogue of personas and attributes is sampled
                  uniformly and used to condition generation. This is the
                  closest existing relative of our method and therefore the
                  most informative comparison: it has language-valued latent
                  conditioning but NO orthogonality selection, NO ledger, and
                  NO recursive refinement. The gap between this arm and ours
                  isolates exactly what those three components buy.

  high_temp       Plain sampling at high temperature -- the trivial baseline
                  that, in simulation, bought the highest raw diversity by
                  degrading the generator.

All arms are resumable and write to real/<domain>_<arm>/ exactly like
real_run.py, so metrics.py reads them without special-casing.
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pipeline import HERE, USAGE, chat, embed
from real_run import DOMAINS, Corpus, EMBED_BATCH, item_text, nn_cos_dist

WORKERS = 40
ROUGE_THRESHOLD = 0.7
N_EXEMPLARS = 3
HIGH_TEMP = 1.6

EVOL_OPS = [
    "Rewrite it so it demands one additional, genuinely harder step of reasoning.",
    "Rewrite it to be much more concrete and specific, replacing every general term.",
    "Add one constraint that meaningfully changes what a correct response looks like.",
    "Deepen it: keep the surface subject but make what it is really asking less obvious.",
    "Mutate it into a different form or structure while keeping its domain.",
]

PERSONAS = [
    "a conservator who restores damaged film stock", "a municipal zoning clerk",
    "a competitive memory athlete", "a marine acoustician", "a bankruptcy trustee",
    "a scent designer for department stores", "a glacier surveyor",
    "a puppet maker for political satire", "an actuary who prices flood risk",
    "a stage manager for touring opera", "a semiconductor failure analyst",
    "a rare-book forger turned authenticator", "a wildfire smoke-plume modeller",
    "a competitive sheepdog handler", "an archivist of defunct social networks",
    "a court stenographer", "a prosthetics fitter", "a bell foundry tuner",
    "a submarine cable splicer", "a food-safety inspector for street markets",
    "a typeface designer for road signage", "a seed-bank curator",
    "an avalanche forecaster", "a hospital sleep-lab technician",
]
ATTRS = [
    "obsessed with edge cases", "deeply sceptical of received wisdom",
    "trained in a different discipline first", "working under severe time pressure",
    "explaining to a hostile audience", "recovering from a public mistake",
    "writing for an archive nobody will read for fifty years",
    "convinced the standard framing is wrong", "new to the field and unimpressed",
    "at the end of a long career",
]


def lcs_len(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, the core of ROUGE-L.

    Rolling two-row DP: the full table for two 60-token items is small, but
    this is called against every pool member for every candidate, so the
    allocation savings matter."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    l = lcs_len(a, b)
    p, r = l / len(a), l / len(b)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def max_rouge_against(cand: str, pool_toks: list[list[str]], rng: random.Random,
                      cap: int = 120) -> float:
    """Self-Instruct compares against the whole pool; at n in the thousands
    that is O(n) LCS computations per candidate and dominates the loop, so we
    compare against a bounded random sample. This makes the filter weaker
    than the original at large n -- which is itself worth reporting, since the
    original's cost grows linearly in corpus size and ours does not."""
    ct = cand.lower().split()
    if not pool_toks:
        return 0.0
    ref = pool_toks if len(pool_toks) <= cap else [pool_toks[i] for i in
                                                   rng.sample(range(len(pool_toks)), cap)]
    return max(rouge_l(ct, r) for r in ref)


def run(domain: str, arm: str, target: int, budget: float):
    C = Corpus(domain, arm)
    rng = random.Random(99 + C.n)
    cfg = DOMAINS[domain]
    json_mode = domain == "psychometric"
    pool_toks = [t.lower().split() for t in C.texts]
    t0, n_start, gens, rejected = time.time(), C.n, 0, 0
    print(f"[{domain}/{arm}] resuming at n={C.n}, target {target}", flush=True)

    while C.n < target:
        if USAGE.cost_usd() > budget:
            print(f"[{domain}/{arm}] budget cap ${budget} hit at n={C.n}", flush=True)
            break
        batch = min(WORKERS, (target - C.n) * 2)
        prompts, temps, metas = [], [], []
        for _ in range(batch):
            if arm == "self_instruct":
                ex = (rng.sample(C.texts, min(N_EXEMPLARS, len(C.texts)))
                      if C.texts else [])
                block = "\n\n".join(f"Example {i+1}:\n{e}" for i, e in enumerate(ex))
                p = (f"{block}\n\nWrite ONE more, clearly different from the examples "
                     f"above.\n\n{cfg['naive_prompt']}" if ex else cfg["naive_prompt"])
                prompts.append(p); temps.append(1.0); metas.append({})
            elif arm == "evol_instruct":
                if C.texts:
                    seed = rng.choice(C.texts)
                    op = rng.choice(EVOL_OPS)
                    p = (f"Here is an existing item:\n\n{seed}\n\n{op}\n\n"
                         f"Output only the new item, in the same format.\n\n"
                         f"{cfg['naive_prompt']}")
                    metas.append({"op": op[:40]})
                else:
                    p = cfg["naive_prompt"]; metas.append({})
                prompts.append(p); temps.append(1.0)
            elif arm == "persona":
                persona, attr = rng.choice(PERSONAS), rng.choice(ATTRS)
                p = (f"Adopt the perspective of {persona}, {attr}.\n\n"
                     f"{cfg['naive_prompt']}")
                prompts.append(p); temps.append(1.0)
                metas.append({"persona": persona, "attr": attr})
            elif arm == "high_temp":
                prompts.append(cfg["naive_prompt"]); temps.append(HIGH_TEMP)
                metas.append({})
            else:
                raise ValueError(arm)

        results = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex_:
            max_tok = 2500 if json_mode else 900
            futs = {ex_.submit(chat, [{"role": "user", "content": p}], tp, max_tok, 4,
                               json_mode): i
                    for i, (p, tp) in enumerate(zip(prompts, temps))}
            for fu in as_completed(futs):
                try:
                    results.append((futs[fu], fu.result()))
                except Exception:
                    pass
        gens += len(prompts)
        parsed = [(i, item_text(domain, r)) for i, r in results]
        parsed = [(i, t) for i, t in parsed if t]
        if not parsed:
            continue

        accepted = []
        for i, t in parsed:
            if C.n + len(accepted) >= target:
                break
            if arm == "self_instruct":
                if max_rouge_against(t, pool_toks, rng) > ROUGE_THRESHOLD:
                    rejected += 1
                    continue
            accepted.append((i, t))
        if not accepted:
            continue

        embs = embed([t for _, t in accepted], batch=EMBED_BATCH)
        for (i, t), e in zip(accepted, embs):
            gap = float(nn_cos_dist(e[None, :], C, rng)[0]) if C.n else 1.0
            C.add(t, e, metas[i] if i < len(metas) else {},
                  {"arm": arm, "gap": round(gap, 5)})
            pool_toks.append(t.lower().split())

        if C.n % 200 < WORKERS:
            C.checkpoint()
            rate = (C.n - n_start) / max(time.time() - t0, 1e-9)
            print(f"[{domain}/{arm}] n={C.n} gens={gens} rej={rejected} "
                  f"${USAGE.cost_usd():.2f} {rate*60:.0f}/min", flush=True)

    C.checkpoint()
    summary = {"domain": domain, "arm": arm, "n": C.n, "generations": gens,
               "rejected_by_filter": rejected,
               "cost_usd": round(USAGE.cost_usd(), 4),
               "wall_clock_min": round((time.time() - t0) / 60, 1)}
    with open(C.dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    C.close()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000,
        float(sys.argv[4]) if len(sys.argv) > 4 else 3.0)
