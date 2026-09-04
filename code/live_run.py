"""
Live pilot: the full IHD stack against gpt-5.6-luna, two domains.

  python3 live_run.py poems  [n]
  python3 live_run.py items  [n]
  python3 live_run.py poems_naive [n]   (baseline: same model, plain prompt)

Per accepted item the stack costs a bounded number of LLM calls regardless of
corpus size: K parallel generations + 1 judge call, plus amortized O(1/25)
mining and refinement calls. All state that policies read is O(1)-bounded:
running centroid/second-moment, EMA scales, a bounded ledger slice, and a
bounded recent-window for exact min-distance (plus a bounded random subsample
of the older corpus).

Everything is append-only JSONL and resumable: kill it, rerun it, it
continues from where the corpus left off.
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pipeline import HERE, USAGE, chat, embed, parse_json
from axes import AxisTree, spec_text

K = 3                    # parallel candidates per step
MINE_EVERY = 25          # attractor mining cadence (accepted items)
MINE_SAMPLE = 50         # poems shown to the miner (or all, if fewer)
LEDGER_SLICE = 12        # attractors actively repelled against per prompt
RECENT_WINDOW = 200      # exact min-dist window; older corpus is subsampled
OLD_SUBSAMPLE = 300
SPEC_POOL = 12           # candidate specs scored for orthogonality per step
SATURATION_CHECK = 25    # cadence for the refine-on-saturation check
BUDGET_USD = 4.0         # hard spend cap for one invocation

GATE_SIM = 0.995         # near-duplicate guard (cosine) -- poems above this to any
                         # neighbor are rejected outright even in the soft domain
ITEM_FLOOR_SIM = 0.90    # hard floor for exam items: reject if cos-sim to any
                         # existing item exceeds this ("enemy item" radius)

POEM_PROMPT = """Write one complete poem, 8-24 lines, in English. No title unless asked.

The poem MUST exhibit every one of these latent behaviors. They are contracts, \
not suggestions -- a reader who knew to look should be able to point to where \
each one operates:
{contracts}

The corpus this poem joins has been drifting toward the following habits. Your \
poem must STAY CLEAR of all of them:
{avoid}

Do not explain the poem. Output only the poem."""

POEM_NAIVE_PROMPT = "Write one complete poem, 8-24 lines, in English. Output only the poem."

POEM_JUDGE_PROMPT = """You are judging candidate poems for a curated corpus. For each \
candidate, score:
  craft: 0-10 (control of line, image, syntax; penalize cliche, filler, \
autopilot abstraction)
  behaviors_met: for each required behavior, true/false -- did the poem \
actually EXHIBIT it (not merely mention it)?

Required behaviors:
{contracts}

Candidates:
{cands}

Return JSON only: {{"judgments": [{{"index": 0, "craft": 0, "behaviors_met": \
[true, false, ...], "weakest_moment": "..."}}]}}"""

ITEM_PROMPT = """Write one multiple-choice exam item for a professional certification in \
GENERAL CLOUD COMPUTING (architecture, storage, networking, security, cost, \
reliability -- practitioner level).

The item MUST exhibit these latent design behaviors:
{contracts}

The item bank has been drifting toward these habits; your item must avoid all \
of them:
{avoid}

Requirements: one stem, exactly 4 options (A-D), exactly one defensibly correct \
answer, plausible distractors that each encode a distinct misconception. Return \
JSON only:
{{"stem": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, \
"correct": "A", "distractor_misconceptions": {{"B": "...", "C": "...", "D": "..."}}}}"""

ITEM_JUDGE_PROMPT = """You are a psychometrician reviewing candidate exam items. For each, \
score:
  validity: 0-10 (exactly one defensible key, stem answerable without options, \
distractors plausible-but-wrong, no cueing, no trivia)
  behaviors_met: for each required design behavior, true/false.

Required behaviors:
{contracts}

Candidates:
{cands}

Return JSON only: {{"judgments": [{{"index": 0, "validity": 0, "behaviors_met": \
[true, ...], "fatal_flaw": "..." or null}}]}}"""

MINE_PROMPT = """Here are {n} items sampled from a growing corpus. Identify what they \
share: recurring words and images, recurring moves and structures, recurring \
stances -- anything a reader of the whole corpus would start to find \
predictable. Be concrete and name the pattern precisely (not "similar tone" \
but the actual tone).

{sample}

Return JSON only: {{"attractors": ["...", "..."]}} -- 5-12 findings, most \
pervasive first."""


class RunState:
    def __init__(self, tag: str):
        self.dir = HERE / "live" / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.dir / "corpus.jsonl"
        self.ledger_path = self.dir / "ledger.jsonl"
        self.axes_log = self.dir / "axes.jsonl"
        self.texts: list[str] = []
        self.embs: list[np.ndarray] = []
        self.specs: list[dict] = []
        self.qualities: list[float] = []
        self.nn_at_accept: list[float] = []
        self.attractors: list[str] = []
        if self.corpus_path.exists():
            for line in self.corpus_path.read_text().splitlines():
                rec = json.loads(line)
                self.texts.append(rec["text"])
                self.embs.append(np.array(rec["emb"]))
                self.specs.append(rec.get("spec", {}))
                self.qualities.append(rec.get("quality", 0.0))
                self.nn_at_accept.append(rec.get("nn_cos_dist", 1.0))
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                self.attractors.extend(json.loads(line).get("attractors", []))

    @property
    def n(self) -> int:
        return len(self.texts)

    def E(self) -> np.ndarray:
        return np.stack(self.embs) if self.embs else np.empty((0, 768))

    def append(self, text: str, emb: np.ndarray, spec: dict, quality: float,
               nn_cos_dist: float, meta: dict):
        rec = {"i": self.n, "text": text, "spec": spec, "quality": quality,
               "nn_cos_dist": nn_cos_dist, "ts": time.time(), **meta,
               "emb": [round(float(v), 6) for v in emb]}
        with open(self.corpus_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.texts.append(text)
        self.embs.append(emb)
        self.specs.append(spec)
        self.qualities.append(quality)
        self.nn_at_accept.append(nn_cos_dist)

    def append_mined(self, attractors: list[str], round_idx: int):
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps({"round": round_idx, "n_at_mining": self.n,
                                "ts": time.time(), "attractors": attractors}) + "\n")
        self.attractors.extend(attractors)


def nn_cos_dist(cand_embs: np.ndarray, state: RunState, rng: random.Random,
                exact: bool = False) -> np.ndarray:
    """1 - max cosine similarity to the corpus. Exact over the recent window
    (+ full corpus when `exact` or corpus small); bounded subsample of the
    older corpus otherwise."""
    if state.n == 0:
        return np.ones(len(cand_embs))
    E = state.E()
    if exact or state.n <= RECENT_WINDOW + OLD_SUBSAMPLE:
        ref = E
    else:
        old = E[:-RECENT_WINDOW]
        idx = rng.sample(range(len(old)), OLD_SUBSAMPLE)
        ref = np.vstack([old[idx], E[-RECENT_WINDOW:]])
    sims = cand_embs @ ref.T
    return 1.0 - sims.max(axis=1)


def orth_residual(cand_embs: np.ndarray, state: RunState, energy_frac: float = 0.8) -> np.ndarray:
    if state.n < 10:
        return np.zeros(len(cand_embs))
    E = state.E()
    C = (E.T @ E) / state.n
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    cum = np.cumsum(w) / max(w.sum(), 1e-12)
    j = int(np.searchsorted(cum, energy_frac)) + 1
    proj = cand_embs @ V[:, :j]
    return 1.0 - (proj ** 2).sum(axis=1)


def choose_spec(tree: AxisTree, state: RunState, rng: random.Random) -> dict:
    """Orthogonalization at the SPEC level: sample a pool of points in the
    latent lattice, embed their descriptions, take the one pointing most into
    the corpus's trailing eigenspace. This is 'ask for the latent, then pick
    the orthogonal one' -- no randomness is trusted to do the work on its own."""
    pool = [tree.sample_spec(rng) for _ in range(SPEC_POOL)]
    if state.n < 10:
        return rng.choice(pool)
    embs = embed([spec_text(s) for s in pool])
    orth = orth_residual(embs, state)
    return pool[int(np.argmax(orth))]


def mine(state: RunState, rng: random.Random, kind: str) -> list[str]:
    k = min(MINE_SAMPLE, state.n)
    sample_idx = rng.sample(range(state.n), k)
    excerpts = []
    for i in sample_idx:
        t = state.texts[i]
        excerpts.append(f"--- item {i} ---\n{t[:600]}")
    raw = chat([{"role": "user", "content": MINE_PROMPT.format(n=k, sample="\n".join(excerpts))}],
               temperature=0.3, max_tokens=800, json_mode=True)
    try:
        return [str(a) for a in parse_json(raw).get("attractors", [])][:12]
    except ValueError:
        return []


def avoid_block(state: RunState) -> str:
    if not state.attractors:
        return "  (nothing yet)"
    return "\n".join(f"  - {a}" for a in state.attractors[-LEDGER_SLICE:])


def run_poems(n_target: int, naive: bool = False):
    tag = "poems_naive" if naive else "poems"
    state = RunState(tag)
    rng = random.Random(7 + state.n)
    tree = None
    if not naive:
        tree = AxisTree(state.axes_log)
        if state.axes_log.exists() and state.axes_log.stat().st_size > 0:
            for line in state.axes_log.read_text().splitlines():
                rec = json.loads(line)
                if rec["event"] == "seed":
                    from axes import Axis
                    tree.axes = [Axis(a["name"], a["levels"],
                                      tuple(a["parent"]) if a["parent"] else None, a["depth"])
                                 for a in rec["axes"]]
                elif rec["event"] == "refine":
                    from axes import Axis
                    a = rec["axis"]
                    tree.axes.append(Axis(a["name"], a["levels"],
                                          tuple(a["parent"]) if a["parent"] else None, a["depth"]))
        if not tree.axes:
            print("seeding axis tree ...")
            tree.seed(k=7)
        print(f"axis tree: {len(tree.axes)} axes, |space| ~ {tree.space_size():.2e}")

    rejected_log = state.dir / "rejected.jsonl"
    t_start = time.time()
    while state.n < n_target:
        if USAGE.cost_usd() > BUDGET_USD:
            print(f"budget cap {BUDGET_USD} USD hit, stopping at n={state.n}")
            break
        step_t0 = time.time()
        if naive:
            spec, contracts = {}, "(none)"
            prompts = [POEM_NAIVE_PROMPT] * 1  # naive = single sample, no selection
        else:
            spec = choose_spec(tree, state, rng)
            contracts = spec_text(spec)
            prompts = [POEM_PROMPT.format(contracts=contracts, avoid=avoid_block(state))] * K

        with ThreadPoolExecutor(max_workers=K) as ex:
            cands = list(ex.map(
                lambda p: chat([{"role": "user", "content": p}], temperature=1.0, max_tokens=700),
                prompts))
        cands = [c.strip() for c in cands if len(c.strip()) > 40]
        if not cands:
            continue

        if naive:
            quality, behaviors_ok = 5.0, True
            chosen, ch_emb = cands[0], embed([cands[0]])[0]
            nn = float(nn_cos_dist(ch_emb[None, :], state, rng)[0])
        else:
            cand_block = "\n".join(f"--- candidate {i} ---\n{c}" for i, c in enumerate(cands))
            jraw = chat([{"role": "user", "content": POEM_JUDGE_PROMPT.format(
                contracts=contracts, cands=cand_block)}],
                temperature=0.0, max_tokens=700, json_mode=True)
            try:
                judgments = {j["index"]: j for j in parse_json(jraw)["judgments"]}
            except (ValueError, KeyError):
                judgments = {}
            embs = embed(cands)
            nnd = nn_cos_dist(embs, state, rng)
            orth = orth_residual(embs, state)
            best_i, best_u = None, -np.inf
            for i, c in enumerate(cands):
                j = judgments.get(i, {})
                craft = float(j.get("craft", 0))
                met = j.get("behaviors_met", [])
                n_missed = sum(1 for b in met if not b)
                if craft < 5 or n_missed > 1:
                    with open(rejected_log, "a") as f:
                        f.write(json.dumps({"n": state.n, "craft": craft,
                                            "missed": n_missed, "text": c[:200]}) + "\n")
                    continue
                if nnd[i] < 1 - GATE_SIM:
                    continue
                u = 0.4 * (craft / 10) + 0.35 * orth[i] + 0.25 * min(nnd[i] / 0.35, 1.0)
                if u > best_u:
                    best_i, best_u = i, u
            if best_i is None:
                continue
            chosen, ch_emb = cands[best_i], embs[best_i]
            quality = float(judgments.get(best_i, {}).get("craft", 0))
            nn = float(nnd[best_i])

        state.append(chosen, ch_emb, spec, quality, nn,
                     {"model": "openai/gpt-5.6-luna", "naive": naive})
        print(f"[{tag} {state.n}/{n_target}] q={quality:.1f} nn={nn:.3f} "
              f"cost=${USAGE.cost_usd():.2f} ({time.time() - step_t0:.1f}s)")

        if not naive and state.n % MINE_EVERY == 0:
            att = mine(state, rng, "poems")
            if att:
                state.append_mined(att, state.n // MINE_EVERY)
                print(f"  mined {len(att)} attractors, ledger={len(state.attractors)}")

        if not naive and state.n % SATURATION_CHECK == 0 and state.n >= 2 * SATURATION_CHECK:
            recent = state.nn_at_accept[-SATURATION_CHECK:]
            prev = state.nn_at_accept[:-SATURATION_CHECK]
            if np.median(recent) < 0.6 * np.median(prev):
                cell = _dominant_cell(state.specs[-SATURATION_CHECK:])
                print(f"  saturation detected (median nn {np.median(recent):.3f} "
                      f"vs {np.median(prev):.3f}); refining {cell[:80]} ...")
                ax = tree.refine(cell, state.attractors[-LEDGER_SLICE:])
                if ax:
                    print(f"  new axis: {ax.name} ({len(ax.levels)} levels), "
                          f"|space| ~ {tree.space_size():.2e}")

    _write_run_summary(state, tag, t_start)


def _dominant_cell(specs: list[dict]) -> str:
    from collections import Counter
    axis_counts: dict[str, Counter] = {}
    for s in specs:
        for a, level in s.items():
            axis_counts.setdefault(a, Counter())[level] += 1
    parts = []
    for a, c in axis_counts.items():
        level, k = c.most_common(1)[0]
        if k >= len(specs) * 0.4:
            parts.append(f"{a} = {level}")
    return "; ".join(parts) if parts else "(no dominant cell; general saturation)"


def run_items(n_target: int):
    state = RunState("items")
    rng = random.Random(23 + state.n)
    tree = AxisTree(state.axes_log)
    if not tree.axes:
        print("seeding item axis tree ...")
        seed_prompt = """You are designing dimensions of variation for a very large \
certification exam item bank (general cloud computing, practitioner level). \
Propose 6 INDEPENDENT axes of item-design variation (NOT topic lists): \
cognitive level demanded, scenario framing, what the distractors encode, \
quantitative vs qualitative reasoning, single-fact vs multi-constraint, \
perspective of the actor in the stem, etc. Each axis: 4-6 levels, all usable.
Return JSON only: {"axes": [{"name": "...", "why_orthogonal": "...", "levels": ["..."]}]}"""
        raw = chat([{"role": "user", "content": seed_prompt}], temperature=1.0,
                   max_tokens=1500, json_mode=True)
        from axes import Axis
        for a in parse_json(raw)["axes"]:
            levels = [str(x) for x in a["levels"] if str(x).strip()]
            if len(levels) >= 3:
                tree.axes.append(Axis(str(a["name"]), levels))
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})
    print(f"item axis tree: {len(tree.axes)} axes, |space| ~ {tree.space_size():.2e}")

    rejected_log = state.dir / "rejected.jsonl"
    floor_rejections = 0
    attempts_total = 0
    t_start = time.time()
    while state.n < n_target:
        if USAGE.cost_usd() > BUDGET_USD:
            print(f"budget cap hit, stopping at n={state.n}")
            break
        spec = choose_spec(tree, state, rng)
        contracts = spec_text(spec)
        prompt = ITEM_PROMPT.format(contracts=contracts, avoid=avoid_block(state))
        with ThreadPoolExecutor(max_workers=K) as ex:
            raws = list(ex.map(
                lambda _: chat([{"role": "user", "content": prompt}],
                               temperature=1.0, max_tokens=700, json_mode=True),
                range(K)))
        attempts_total += K
        items, texts = [], []
        for r in raws:
            try:
                it = parse_json(r)
                if not (isinstance(it, dict) and "stem" in it and "options" in it):
                    continue
                items.append(it)
                opts = it["options"]
                texts.append(it["stem"] + "\n" + "\n".join(
                    f"{k}. {v}" for k, v in sorted(opts.items())))
            except ValueError:
                continue
        if not items:
            continue

        cand_block = "\n".join(f"--- candidate {i} ---\n{t}" for i, t in enumerate(texts))
        jraw = chat([{"role": "user", "content": ITEM_JUDGE_PROMPT.format(
            contracts=contracts, cands=cand_block)}],
            temperature=0.0, max_tokens=700, json_mode=True)
        try:
            judgments = {j["index"]: j for j in parse_json(jraw)["judgments"]}
        except (ValueError, KeyError):
            judgments = {}

        embs = embed(texts)
        # HARD FLOOR: exact check against the full bank, never subsampled.
        nnd = nn_cos_dist(embs, state, rng, exact=True)
        orth = orth_residual(embs, state)
        best_i, best_u = None, -np.inf
        for i in range(len(items)):
            j = judgments.get(i, {})
            validity = float(j.get("validity", 0))
            met = j.get("behaviors_met", [])
            n_missed = sum(1 for b in met if not b)
            if nnd[i] < 1 - ITEM_FLOOR_SIM:
                floor_rejections += 1
                continue
            if validity < 6 or j.get("fatal_flaw") or n_missed > 1:
                with open(rejected_log, "a") as f:
                    f.write(json.dumps({"n": state.n, "validity": validity,
                                        "flaw": j.get("fatal_flaw"),
                                        "stem": items[i]["stem"][:150]}) + "\n")
                continue
            u = 0.4 * (validity / 10) + 0.35 * orth[i] + 0.25 * min(nnd[i] / 0.35, 1.0)
            if u > best_u:
                best_i, best_u = i, u
        if best_i is None:
            continue
        quality = float(judgments.get(best_i, {}).get("validity", 0))
        state.append(texts[best_i], embs[best_i], spec, quality, float(nnd[best_i]),
                     {"model": "openai/gpt-5.6-luna", "item_json": items[best_i]})
        print(f"[items {state.n}/{n_target}] v={quality:.1f} nn={nnd[best_i]:.3f} "
              f"floor_rej={floor_rejections} cost=${USAGE.cost_usd():.2f}")

        if state.n % MINE_EVERY == 0:
            att = mine(state, rng, "items")
            if att:
                state.append_mined(att, state.n // MINE_EVERY)
                print(f"  mined {len(att)} attractors")
        if state.n % SATURATION_CHECK == 0 and state.n >= 2 * SATURATION_CHECK:
            recent = state.nn_at_accept[-SATURATION_CHECK:]
            prev = state.nn_at_accept[:-SATURATION_CHECK]
            if np.median(recent) < 0.6 * np.median(prev):
                ax = tree.refine(_dominant_cell(state.specs[-SATURATION_CHECK:]),
                                 state.attractors[-LEDGER_SLICE:])
                if ax:
                    print(f"  new axis: {ax.name}")

    summary = _write_run_summary(state, "items", t_start)
    summary["floor_rejections"] = floor_rejections
    summary["attempts_total"] = attempts_total
    with open(state.dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _write_run_summary(state: RunState, tag: str, t_start: float) -> dict:
    E = state.E()
    sims = E @ E.T
    np.fill_diagonal(sims, -1)
    nn_sims = sims.max(axis=1) if state.n > 1 else np.array([0.0])
    from simulate import vendi_score_linear
    summary = {
        "tag": tag,
        "n": state.n,
        "wall_clock_s": time.time() - t_start,
        "cost_usd": USAGE.cost_usd(),
        "llm_calls": USAGE.calls,
        "llm_errors": USAGE.errors,
        "mean_quality": float(np.mean(state.qualities)) if state.qualities else 0,
        "vendi": vendi_score_linear(E) if state.n > 2 else 0,
        "max_pairwise_cos_sim": float(nn_sims.max()),
        "median_nn_cos_sim": float(np.median(nn_sims)),
        "n_attractors_mined": len(state.attractors),
    }
    with open(state.dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "poems"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if mode == "poems":
        run_poems(n)
    elif mode == "poems_naive":
        run_poems(n, naive=True)
    elif mode == "items":
        run_items(n)
    else:
        raise SystemExit(f"unknown mode {mode}")
