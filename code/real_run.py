"""
Full-scale real run: n = 5 -> 10,000 actual gpt-5.6-luna generations.

Two domains, deliberately chosen so that "too similar" means different things:

  dalle      DALL-E instructions for post-modern artworks. Diversity is the
             product itself: a prompt set that clusters produces an image set
             that clusters. Near-duplicates are waste.
  psychometric
             Psychometric test questions. Two items measuring the same
             construct with the same surface are redundant at best and, in an
             operational bank, a validity problem.

Two arms per domain, budget-aware and resumable:

  naive      one call, plain prompt, accept everything. This is the honest
             baseline: what you get from a strong model asked 10,000 times.
  ihd        the stack -- recursive elicited axes, spec-level orthogonalization,
             capped repulsion behind a typicality gate, append-only attractor
             ledger mined from 50 sampled items.

The IHD arm here runs WITHOUT the separate per-step judge call that the
smaller pilots use: at 10,000 items a judge call per step doubles cost for a
signal we already measured at pilot scale. Quality is instead defended by the
typicality gate and format validation. This is a stated cost compromise, not a
claim that judging is unnecessary.

Everything is append-only JSONL, checkpointed, and safe to kill and resume.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pipeline import HERE, USAGE, chat, embed, parse_json
from axes import Axis, AxisTree, spec_text

WORKERS = 48             # concurrent generator calls
K = 2                    # candidates per accepted item (IHD arm)
MINE_EVERY = 250         # attractor mining cadence (accepted items)
MINE_SAMPLE = 50         # items shown to the miner
LEDGER_SLICE = 10        # attractors actively repelled per prompt
SPEC_POOL = 6            # specs scored for orthogonality per step
REFINE_EVERY = 500       # saturation check cadence
RECENT_WINDOW = 400      # exact min-dist window
OLD_SUBSAMPLE = 600      # bounded subsample of the older corpus
EMBED_BATCH = 96
GATE_Z = 2.6
CHECKPOINT_EVERY = 100

DOMAINS = {
    "dalle": {
        "naive_prompt": (
            "Write one image-generation instruction for a post-modern artwork. "
            "One paragraph, 25-60 words, concrete and visual: name the medium, "
            "the composition, and the treatment. Output only the instruction."
        ),
        "ihd_prompt": (
            "Write one image-generation instruction for a post-modern artwork. "
            "One paragraph, 25-60 words, concrete and visual.\n\n"
            "The instruction MUST exhibit every one of these latent design "
            "behaviors -- they are contracts, not suggestions:\n{contracts}\n\n"
            "The instruction set so far has been drifting toward these habits. "
            "Yours must stay clear of all of them:\n{avoid}\n\n"
            "Output only the instruction."
        ),
        "seed_axes": (
            "You are designing the dimensions of variation for a very large set "
            "of image-generation instructions for POST-MODERN artworks. Do not "
            "write any instructions.\n\n"
            "Propose {k} INDEPENDENT axes of variation. A good axis is close to "
            "orthogonal to the others, changes something structural about the "
            "artwork rather than merely its subject, and has levels that are all "
            "genuinely usable.\n\n"
            "Avoid the obvious surface axes (subject matter, color palette, "
            "medium alone). Reach for the real post-modern operations: relation "
            "to an appropriated source, degree of self-reference, how the work "
            "treats its own materiality, what hierarchy it collapses, the "
            "register it quotes and the register it undercuts, how it handles "
            "seriality and the copy, what it does with text-in-image.\n\n"
            'Return JSON only: {{"axes": [{{"name": "...", "why_orthogonal": '
            '"...", "levels": ["...", "...", "...", "..."]}}]}}\n'
            "Each axis needs 4-6 levels. Keep why_orthogonal under 15 words and "
            "each level name under 10 words."
        ),
        "min_chars": 60,
    },
    "instruction": {
        # Head-to-head domain: the SAME artifact type as Alpaca / PersonaHub /
        # WizardLM's released corpora -- a standalone instruction or task
        # prompt for an assistant -- so our corpus can be compared against
        # theirs directly rather than against a reimplementation.
        "naive_prompt": (
            "Write one instruction or task prompt for an AI assistant. It should be "
            "self-contained and answerable without extra context. One to three "
            "sentences. Output only the instruction."
        ),
        "ihd_prompt": (
            "Write one instruction or task prompt for an AI assistant. It should be "
            "self-contained and answerable without extra context. One to three "
            "sentences.\n\n"
            "The instruction MUST exhibit every one of these latent design "
            "behaviors -- they are contracts, not suggestions:\n{contracts}\n\n"
            "The instruction set so far has been drifting toward these habits. "
            "Yours must stay clear of all of them:\n{avoid}\n\n"
            "Output only the instruction."
        ),
        "seed_axes": (
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
            'Return JSON only: {{"axes": [{{"name": "...", "why_orthogonal": '
            '"...", "levels": ["...", "...", "...", "..."]}}]}}\n'
            "Each axis needs 4-6 levels. Keep why_orthogonal under 15 words and "
            "each level name under 10 words."
        ),
        "min_chars": 25,
    },
    "psychometric": {
        "naive_prompt": (
            "Write one multiple-choice psychometric test question assessing "
            "general reasoning ability (verbal, numerical, logical, or spatial). "
            "Give a stem and exactly four options A-D with one defensibly "
            "correct answer.\n"
            'Return JSON only: {"stem": "...", "options": {"A": "...", "B": '
            '"...", "C": "...", "D": "..."}, "correct": "A", "construct": "..."}'
        ),
        "ihd_prompt": (
            "Write one multiple-choice psychometric test question assessing "
            "general reasoning ability.\n\n"
            "The item MUST exhibit every one of these latent design behaviors -- "
            "they are contracts, not suggestions:\n{contracts}\n\n"
            "The item bank has been drifting toward these habits; your item must "
            "avoid all of them:\n{avoid}\n\n"
            "Give a stem and exactly four options A-D with one defensibly "
            "correct answer and distractors that each encode a distinct "
            "misconception.\n"
            'Return JSON only: {{"stem": "...", "options": {{"A": "...", "B": '
            '"...", "C": "...", "D": "..."}}, "correct": "A", "construct": "..."}}'
        ),
        "seed_axes": (
            "You are designing the dimensions of variation for a very large "
            "psychometric item bank measuring general reasoning. Do not write "
            "any items.\n\n"
            "Propose {k} INDEPENDENT axes of ITEM-DESIGN variation (not a topic "
            "list). A good axis is close to orthogonal to the others and changes "
            "something structural about how the item works.\n\n"
            "Reach for real psychometric design decisions: the cognitive "
            "operation demanded, how the stem frames its scenario, what the "
            "distractors encode, whether the reasoning is convergent or "
            "eliminative, how many constraints must be held at once, what "
            "representation the examinee must build, how the item handles "
            "irrelevant detail.\n\n"
            'Return JSON only: {{"axes": [{{"name": "...", "why_orthogonal": '
            '"...", "levels": ["...", "...", "...", "..."]}}]}}\n'
            "Each axis needs 4-6 levels. Keep why_orthogonal under 15 words and "
            "each level name under 10 words."
        ),
        "min_chars": 80,
    },
}

MINE_PROMPT = """Here are {n} items sampled from a growing corpus. Identify what they \
SHARE: recurring words, images, structures, moves, framings -- anything a reader of the \
whole corpus would start to find predictable. Be concrete and name the pattern precisely \
(not "similar tone" but the actual tone; not "similar subjects" but which subjects).

{sample}

Return JSON only: {{"attractors": ["...", "..."]}} -- 6-12 findings, most pervasive first."""


def item_text(domain: str, raw: str) -> str | None:
    """Normalize a raw completion into the canonical text we embed and store."""
    raw = (raw or "").strip()
    if domain in ("dalle", "instruction"):
        raw = raw.strip('"').strip()
        return raw if len(raw) >= DOMAINS[domain]["min_chars"] else None
    try:
        obj = parse_json(raw)
    except ValueError:
        return None
    if not (isinstance(obj, dict) and obj.get("stem") and isinstance(obj.get("options"), dict)):
        return None
    opts = obj["options"]
    if set(opts) != {"A", "B", "C", "D"}:
        return None
    txt = str(obj["stem"]).strip() + "\n" + "\n".join(f"{k}. {opts[k]}" for k in "ABCD")
    return txt if len(txt) >= DOMAINS["psychometric"]["min_chars"] else None


class Corpus:
    def __init__(self, domain: str, arm: str):
        self.dir = HERE / "real" / f"{domain}_{arm}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.domain, self.arm = domain, arm
        self.path = self.dir / "corpus.jsonl"
        self.emb_path = self.dir / "embeddings.npy"
        self.ledger_path = self.dir / "ledger.jsonl"
        self.axes_path = self.dir / "axes.jsonl"
        self.texts: list[str] = []
        self.specs: list[dict] = []
        self.attractors: list[str] = []
        self.E: np.ndarray | None = None
        self._fh = None
        self._load()

    def _load(self):
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # a kill mid-write can leave one truncated final line;
                    # everything before it is intact, so skip and continue
                    continue
                if not r.get("text"):
                    continue
                self.texts.append(r["text"])
                self.specs.append(r.get("spec", {}))
        if self.emb_path.exists():
            try:
                E = np.load(self.emb_path)
            except (ValueError, OSError):
                E = np.empty((0, 768))     # corrupt/partial .npy: rebuild below
            self.E = E[: len(self.texts)]
        else:
            self.E = np.empty((0, 768))
        # The jsonl is written every item but embeddings only checkpoint every
        # CHECKPOINT_EVERY, so after a kill the jsonl legitimately runs ahead.
        # Re-embed the tail rather than discarding those items -- they cost
        # real money to generate and the text is right there on disk.
        missing = len(self.texts) - len(self.E)
        if missing > 0:
            print(f"  recovering {missing} un-embedded item(s) from a previous run",
                  flush=True)
            tail = embed(self.texts[len(self.E):], batch=EMBED_BATCH)
            self.E = tail if len(self.E) == 0 else np.vstack([self.E, tail])
            self._atomic_save()
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                try:
                    self.attractors.extend(json.loads(line).get("attractors", []))
                except json.JSONDecodeError:
                    continue
        self._fh = open(self.path, "a")

    def _atomic_save(self):
        """np.save straight onto emb_path would leave a half-written file if
        killed mid-write; write beside it and rename, which is atomic."""
        if self.E is None or len(self.E) == 0:
            return
        tmp = self.emb_path.with_suffix(".npy.tmp")
        # np.save(path, ...) appends ".npy" unless the name already ends in it,
        # so passing the tmp PATH silently wrote embeddings.npy.tmp.npy and the
        # rename then failed on a file that never existed. Hand it an open file
        # object instead, which numpy writes verbatim.
        with open(tmp, "wb") as f:
            np.save(f, self.E)
        os.replace(tmp, self.emb_path)

    @property
    def n(self) -> int:
        return len(self.texts)

    def add(self, text: str, emb: np.ndarray, spec: dict, meta: dict):
        self._fh.write(json.dumps({"i": self.n, "text": text, "spec": spec,
                                   "ts": round(time.time(), 2), **meta}) + "\n")
        self._fh.flush()   # the jsonl is the source of truth; never buffer it
        self.texts.append(text)
        self.specs.append(spec)
        self.E = emb[None, :] if self.E is None else np.vstack([self.E, emb[None, :]])

    def checkpoint(self):
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._atomic_save()

    def add_mined(self, attractors: list[str]):
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps({"n_at_mining": self.n, "ts": time.time(),
                                "attractors": attractors}) + "\n")
        self.attractors.extend(attractors)

    def close(self):
        try:
            self.checkpoint()
        finally:
            self._fh.close()


def nn_cos_dist(cands: np.ndarray, C: Corpus, rng: random.Random) -> np.ndarray:
    if C.n == 0:
        return np.ones(len(cands))
    E = C.E
    if C.n <= RECENT_WINDOW + OLD_SUBSAMPLE:
        ref = E
    else:
        old = E[:-RECENT_WINDOW]
        idx = rng.sample(range(len(old)), OLD_SUBSAMPLE)
        ref = np.vstack([old[idx], E[-RECENT_WINDOW:]])
    return 1.0 - (cands @ ref.T).max(axis=1)


class Spectral:
    """Running second moment + centroid for orthogonality and the gate."""

    def __init__(self, d: int):
        self.Csum = np.zeros((d, d))
        self.sum = np.zeros(d)
        self.n = 0
        self.r_ema = 1.0
        self.s_ema = 1.0
        self._V = None
        self._stale = 0

    def seed_from(self, E: np.ndarray):
        if E is None or len(E) == 0:
            return
        self.Csum = E.T @ E
        self.sum = E.sum(axis=0)
        self.n = len(E)
        c = self.sum / self.n
        self.r_ema = float(np.linalg.norm(E - c[None, :], axis=1).mean())
        self._stale = 10 ** 6

    def update(self, x: np.ndarray, gap: float):
        self.Csum += np.outer(x, x)
        self.sum += x
        self.n += 1
        r = float(np.linalg.norm(x - self.sum / self.n))
        if self.n == 1:
            self.r_ema = max(r, 1e-6)
        else:
            if self.n == 2:
                self.s_ema = max(gap, 1e-6)
            self.r_ema = 0.99 * self.r_ema + 0.01 * max(r, 1e-6)
            self.s_ema = 0.99 * self.s_ema + 0.01 * max(gap, 1e-6)
        self._stale += 1

    def basis(self, energy_frac: float = 0.8) -> np.ndarray:
        """Recomputed lazily -- a 768x768 eigendecomposition every step would
        dominate the loop, and the occupied subspace moves slowly."""
        if self.n < 12:
            return np.zeros((len(self.sum), 0))
        if self._V is None or self._stale >= 50:
            w, V = np.linalg.eigh(self.Csum / self.n)
            order = np.argsort(w)[::-1]
            w, V = w[order], V[:, order]
            cum = np.cumsum(w) / max(w.sum(), 1e-12)
            j = int(np.searchsorted(cum, energy_frac)) + 1
            self._V = V[:, :j]
            self._stale = 0
        return self._V

    def orth(self, cands: np.ndarray) -> np.ndarray:
        V = self.basis()
        if V.shape[1] == 0:
            return np.zeros(len(cands))
        return 1.0 - ((cands @ V) ** 2).sum(axis=1)

    def anchor_z(self, cands: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.zeros(len(cands))
        c = self.sum / self.n
        return np.linalg.norm(cands - c[None, :], axis=1) / max(self.r_ema, 1e-6)


def load_tree(C: Corpus, domain: str) -> AxisTree:
    tree = AxisTree(C.axes_path)
    if C.axes_path.exists() and C.axes_path.stat().st_size > 0:
        for line in C.axes_path.read_text().splitlines():
            rec = json.loads(line)
            if rec["event"] == "seed":
                tree.axes = [Axis(a["name"], a["levels"],
                                  tuple(a["parent"]) if a["parent"] else None, a["depth"])
                             for a in rec["axes"]]
            elif rec["event"] == "refine":
                a = rec["axis"]
                tree.axes.append(Axis(a["name"], a["levels"],
                                      tuple(a["parent"]) if a["parent"] else None, a["depth"]))
    if not tree.axes:
        print(f"  seeding axis tree for {domain} ...", flush=True)
        for attempt in range(3):
            raw = chat([{"role": "user",
                         "content": DOMAINS[domain]["seed_axes"].format(k=7)}],
                       temperature=1.0, max_tokens=4000, json_mode=True)
            try:
                obj = parse_json(raw)
                break
            except ValueError:
                if attempt == 2:
                    raise
        for a in obj["axes"]:
            levels = [str(x) for x in a["levels"] if str(x).strip()]
            if len(levels) >= 3:
                tree.axes.append(Axis(str(a["name"]), levels))
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})
    return tree


def run(domain: str, arm: str, target: int, budget_usd: float):
    C = Corpus(domain, arm)
    rng = random.Random(1234 + C.n)
    spec_stats = Spectral(768)
    spec_stats.seed_from(C.E)
    tree = load_tree(C, domain) if arm == "ihd" else None
    cfg = DOMAINS[domain]
    t0 = time.time()
    n_start = C.n
    gens = 0
    print(f"[{domain}/{arm}] resuming at n={C.n}, target {target}", flush=True)

    while C.n < target:
        if USAGE.cost_usd() > budget_usd:
            print(f"[{domain}/{arm}] budget cap ${budget_usd} reached at n={C.n}", flush=True)
            break

        if arm == "naive":
            batch = min(WORKERS, target - C.n)
            prompts = [cfg["naive_prompt"]] * batch
            specs = [{}] * batch
        else:
            # one spec per slot in the batch, each chosen by orthogonality
            n_slots = max(1, min(WORKERS // K, target - C.n))
            pool = [tree.sample_spec(rng) for _ in range(n_slots * SPEC_POOL)]
            if C.n >= 24:
                sp_emb = embed([spec_text(s) for s in pool], batch=EMBED_BATCH)
                sc = spec_stats.orth(sp_emb)
                order = np.argsort(sc)[::-1]
                chosen = [pool[i] for i in order[:n_slots]]
            else:
                chosen = pool[:n_slots]
            avoid = ("\n".join(f"  - {a}" for a in C.attractors[-LEDGER_SLICE:])
                     or "  (nothing yet)")
            prompts, specs = [], []
            for s in chosen:
                for _ in range(K):
                    prompts.append(cfg["ihd_prompt"].format(
                        contracts=spec_text(s), avoid=avoid))
                    specs.append(s)

        results: list[tuple[int, str]] = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            # gpt-5.6-luna spends output budget on internal reasoning BEFORE
            # emitting content, so a too-small cap returns an empty string
            # rather than a truncated one. The richer IHD prompts reason more:
            # at 700 tokens the psychometric IHD arm parsed 0/3, at 2500 it
            # parses 3/3. Structured domains therefore get a much larger cap.
            max_tok = 2500 if domain == "psychometric" else 900
            futs = {ex.submit(chat, [{"role": "user", "content": p}], 1.0, max_tok,
                              4, domain == "psychometric"): i
                    for i, p in enumerate(prompts)}
            for fu in as_completed(futs):
                i = futs[fu]
                try:
                    results.append((i, fu.result()))
                except Exception:
                    pass
        gens += len(prompts)
        parsed = [(i, item_text(domain, r)) for i, r in results]
        parsed = [(i, t) for i, t in parsed if t]
        if not parsed:
            continue

        texts = [t for _, t in parsed]
        embs = embed(texts, batch=EMBED_BATCH)

        if arm == "naive":
            for (i, t), e in zip(parsed, embs):
                if C.n >= target:
                    break
                gap = float(nn_cos_dist(e[None, :], C, rng)[0]) if C.n else 1.0
                C.add(t, e, {}, {"arm": arm, "gap": round(gap, 5)})
                spec_stats.update(e, gap)
        else:
            # group candidates by their spec slot and select one per slot
            slots: dict[int, list[int]] = {}
            for idx, (i, _) in enumerate(parsed):
                slots.setdefault(i // K, []).append(idx)
            for slot, idxs in sorted(slots.items()):
                if C.n >= target:
                    break
                cand_e = embs[idxs]
                nn = nn_cos_dist(cand_e, C, rng)
                orth = spec_stats.orth(cand_e)
                z = spec_stats.anchor_z(cand_e)
                U = 0.55 * orth + 0.45 * np.minimum(nn / max(spec_stats.s_ema, 1e-6), 2.0) / 2.0
                if C.n > 200:
                    bad = z > GATE_Z
                    if not bad.all():
                        U = np.where(bad, -np.inf, U)
                b = int(np.argmax(U))
                gi = idxs[b]
                C.add(texts[gi], cand_e[b], specs[parsed[gi][0]],
                      {"arm": arm, "gap": round(float(nn[b]), 5)})
                spec_stats.update(cand_e[b], float(nn[b]))

        if C.n % CHECKPOINT_EVERY < WORKERS:
            C.checkpoint()
            rate = (C.n - n_start) / max(time.time() - t0, 1e-9)
            eta = (target - C.n) / max(rate, 1e-9) / 60
            print(f"[{domain}/{arm}] n={C.n} gens={gens} ${USAGE.cost_usd():.2f} "
                  f"{rate * 60:.0f} items/min eta {eta:.0f}m", flush=True)

        if arm == "ihd" and C.n // MINE_EVERY > len(
                [1 for _ in open(C.ledger_path)] if C.ledger_path.exists() else []):
            k = min(MINE_SAMPLE, C.n)
            sample = rng.sample(range(C.n), k)
            blob = "\n".join(f"--- {i} ---\n{C.texts[i][:420]}" for i in sample)
            try:
                raw = chat([{"role": "user", "content": MINE_PROMPT.format(n=k, sample=blob)}],
                           temperature=0.3, max_tokens=900, json_mode=True)
                att = [str(a) for a in parse_json(raw).get("attractors", [])][:12]
                if att:
                    C.add_mined(att)
                    print(f"  [{domain}] mined {len(att)} attractors at n={C.n}", flush=True)
            except Exception as e:
                print(f"  mining failed: {e}", flush=True)

        if arm == "ihd" and C.n >= REFINE_EVERY and C.n % REFINE_EVERY < WORKERS:
            from collections import Counter
            cnt = Counter()
            for s in C.specs[-REFINE_EVERY:]:
                for a, lv in s.items():
                    cnt[(a, lv)] += 1
            if cnt:
                (ax, lv), _ = cnt.most_common(1)[0]
                try:
                    new = tree.refine(f"{ax} = {lv}", C.attractors[-LEDGER_SLICE:])
                    if new:
                        print(f"  [{domain}] refined -> {new.name} "
                              f"({len(new.levels)} levels), |space|~{tree.space_size():.1e}",
                              flush=True)
                except Exception as e:
                    print(f"  refine failed: {e}", flush=True)

    C.checkpoint()
    summary = {"domain": domain, "arm": arm, "n": C.n, "generations_this_session": gens,
               "cost_usd_this_session": round(USAGE.cost_usd(), 4),
               "llm_calls": USAGE.calls, "llm_errors": USAGE.errors,
               "wall_clock_min": round((time.time() - t0) / 60, 1),
               "n_attractors": len(C.attractors),
               "n_axes": len(tree.axes) if tree else 0}
    with open(C.dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    C.close()


if __name__ == "__main__":
    domain = sys.argv[1]
    arm = sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
    run(domain, arm, target, budget)
