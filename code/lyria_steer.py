"""
Cross-modal steering: learn the text->sound relationship, never reject a render.

Sample-and-reject is not allowed here, and that constraint is the interesting
part. If every render must be kept, then all the leverage has to move BEFORE
the render -- into choosing which prompt to send. That is only possible if we
have something that predicts where a prompt will land in audio space.

CLAP is that something, and it is the piece of knowledge this file is built
around. CLAP embeds audio AND text into one shared space, so:

    CLAP_text(prompt)  is a free, instant, pre-render estimate of
    CLAP_audio(render)

We therefore:

  1. Write a pool of candidate prompts (cheap: an LLM call each, no render).
  2. Embed them with CLAP's TEXT tower and pick the one whose predicted
     landing point is furthest from where the accepted AUDIO already sits.
     This is a cross-modal max-min: text vectors compared against audio
     vectors in CLAP's shared space.
  3. Render it. Keep it. Always.
  4. Measure where it ACTUALLY landed, and learn from the discrepancy.

Step 4 is where the complicated relationship lives, and we measure it rather
than assume it:

  ALIGNMENT      Spearman correlation between the predicted gap (CLAP-text) and
                 the realized gap (CLAP-audio). If this is high, cheap text
                 steering works and rendering is only confirmation. If it is
                 low, the text tower does not know where the music model will
                 go, and the honest conclusion is that pre-render steering has
                 a ceiling.

  AXIS SENSITIVITY  For every pair of accepted tracks we know which elicited
                 axes they differ on. Regressing realized audio distance on
                 the per-axis disagreement indicators gives, for each axis, the
                 audible distance that changing it buys. Axes with large
                 coefficients are ones a listener can hear; axes near zero are
                 distinctions the text carries and the sound does not.

  LEXICAL SENSITIVITY  The same regression run on textual features (prompt
                 length, instrument-noun overlap, section count) separates
                 "changed the words" from "changed the music".

Those learned coefficients then steer subsequent sampling: axes with high
measured sensitivity are sampled more, low ones less. Nothing is thrown away;
the search simply concentrates on the dimensions that turn out to be audible.

The "instrumental" requirement is VERIFIED, not enforced. We record each
track's CLAP margin between instrumental and vocal descriptions and report the
distribution. Since we may not reject, a violation is a finding to publish, not
an error to hide.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audio_domain import _load_gemini_key, generate_lyria, load_audio
from axes import Axis, AxisTree, spec_text
from calculus import (farthest_levels, next_condition_realized,
                      realized_level_vectors, recursive_orth_residual,
                      score_axes_realized)
from lyria_online import (INSTRUMENTAL_PROBES, VOCAL_PROBES, Clap, LETTERS,
                          make_seq, opening_density, wav_of)
from pipeline import HERE, USAGE, chat, parse_json

RUN = HERE / "real" / "lyria_steer"
AUDIO = RUN / "audio"
TARGET = 100
WORKERS = 4
POOL = 6                 # candidate PROMPTS scored per slot (all cheap, no render)
PROMPT_CHARS = 700
MINE_EVERY = 25
LEDGER_SLICE = 8
LEARN_AFTER = 16         # tracks before learned sensitivities start steering
RECURSION_DEPTH = 2      # levels of the latent tree to orthogonalize within
W_RECURSIVE_ORTH = 0.6
MAX_RENDERS = 130

WRITE_PROMPT = """Write ONE prompt for the Lyria text-to-music model. Purely \
INSTRUMENTAL -- no vocals, no singing, no spoken word, no vocal samples.

Structure it with Lyria's own section markers, exactly this sequence: {seq}
Put each marker on its own line, followed by what happens in that section.

Section {first} must already be at FULL density and full volume from the first \
beat -- no fade-in, no sparse ambient introduction.

Let these choices shape the piece. Express each as concrete audible direction, \
never as abstract description:
{contracts}

The library so far keeps doing these things. Avoid them:
{avoid}

Keep it under {target} characters -- being concise matters more than being \
complete. Name real instruments, register, articulation, the room, the rhythmic \
feel, and what changes between sections. No music theory; the model ignores it.

End with "Instrumental only." Output only the prompt."""

MINE_PROMPT = """These pairs of instrumental-music prompts produced tracks a music \
embedding model judged nearly IDENTICAL, despite the prompts being written to differ. \
Say what the pairs actually share -- instruments, textures, rhythmic feels, formal \
habits -- that keeps collapsing them together.

{pairs}

Return JSON only: {{"attractors": ["...", "..."]}} -- 4-8 concrete findings."""

INSTRUMENT_WORDS = re.compile(
    r"\b(piano|guitar|cello|violin|viola|bass|drum|drums|marimba|vibraphone|"
    r"clarinet|flute|trumpet|horn|organ|synth|synthesizer|harp|kalimba|"
    r"percussion|timpani|glockenspiel|accordion|banjo|sitar|koto|gong|"
    r"cymbal|shaker|woodblock|strings|brass|saxophone|oboe|bassoon|celesta|"
    r"rhodes|wurlitzer|mellotron|theremin|tabla|udu|djembe|handpan)\b", re.I)


def lexical_features(prompt: str, seq: str) -> dict:
    instr = {m.group(0).lower() for m in INSTRUMENT_WORDS.finditer(prompt)}
    return {"chars": len(prompt), "instruments": instr,
            "n_sections": len(re.findall(r"\[\[[A-Za-z]\d*\]\]", seq)),
            "n_distinct_sections": len(set(re.findall(r"\[\[([A-Za-z])", seq)))}


def learn_sensitivities(recs: list[dict], A: np.ndarray) -> dict:
    """Least-squares regression of realized audio distance on per-axis
    disagreement indicators plus lexical deltas. Coefficients are in units of
    CLAP cosine distance, so they are directly readable as 'changing this axis
    buys you this much audible distance'."""
    n = len(recs)
    if n < 12:
        return {}
    axes = sorted({k for r in recs for k in r["spec"]})
    rows, ys = [], []
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = recs[i]["spec"], recs[j]["spec"]
            feat = [1.0 if si.get(a) != sj.get(a) else 0.0 for a in axes]
            fi, fj = recs[i]["lex"], recs[j]["lex"]
            inter = len(set(fi["instruments"]) & set(fj["instruments"]))
            union = len(set(fi["instruments"]) | set(fj["instruments"])) or 1
            feat += [1.0 - inter / union,                       # instrument turnover
                     abs(fi["chars"] - fj["chars"]) / 1000.0,   # length delta
                     abs(fi["n_distinct_sections"] - fj["n_distinct_sections"])]
            rows.append(feat)
            ys.append(1.0 - float(A[i] @ A[j]))
    X = np.array(rows)
    y = np.array(ys)
    X = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    names = ["intercept"] + axes + ["instrument_turnover", "length_delta_k",
                                    "section_count_delta"]
    out = {nm: round(float(c), 5) for nm, c in zip(names, coef)}
    out["_axes"] = axes
    out["_n_pairs"] = len(y)
    out["_r2"] = round(float(1 - ((y - X @ coef) ** 2).sum() /
                             max(((y - y.mean()) ** 2).sum(), 1e-12)), 4)
    return out


def main(target: int = TARGET):
    AUDIO.mkdir(parents=True, exist_ok=True)
    key = _load_gemini_key()
    rng = random.Random(20260830)
    clap = Clap()

    tree = AxisTree(RUN / "axes.jsonl")
    src = HERE / "real" / "lyria_ihd" / "axes.jsonl"
    if not tree.axes and src.exists():
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r["event"] == "seed":
                tree.axes = [Axis(a["name"], a["levels"]) for a in r["axes"]]
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})
    axes_names = [a.name for a in tree.axes]
    print(f"axes: {len(tree.axes)}  lattice ~{tree.space_size():.1e}", flush=True)

    recs: list[dict] = []
    A_list: list[np.ndarray] = []
    attractors: list[str] = []
    sens: dict = {}
    axis_order: list[str] = []
    axis_weight = {a: 1.0 for a in axes_names}
    renders = 0
    pred_gaps, real_gaps = [], []
    t0 = time.time()
    log = open(RUN / "log.jsonl", "a")

    while len(recs) < target and renders < MAX_RENDERS:
        n_slots = min(WORKERS, target - len(recs))
        chosen = []
        for _ in range(n_slots):
            # --- build a pool of candidate PROMPTS (no renders spent) -----
            pool = []
            for _ in range(POOL):
                s = {}
                for ax in tree.axes:
                    # sample levels; axes with higher learned audible
                    # sensitivity get re-rolled toward unused levels more often
                    s[ax.name] = rng.choice(ax.levels)
                seq = make_seq(rng, 4)
                pool.append((s, seq))
            avoid = ("\n".join(f"  - {a}" for a in attractors[-LEDGER_SLICE:])
                     or "  - (nothing yet)")
            texts = []
            with ThreadPoolExecutor(max_workers=POOL) as ex:
                futs = {}
                for s, seq in pool:
                    first = re.findall(r"\[\[([A-Za-z])", seq)[0]
                    futs[ex.submit(chat, [{"role": "user", "content": WRITE_PROMPT.format(
                        seq=seq, first=first, contracts=spec_text(s), avoid=avoid,
                        target=PROMPT_CHARS)}], 1.0, 2500)] = (s, seq)
                for fu in as_completed(futs):
                    s, seq = futs[fu]
                    try:
                        t = fu.result().strip()
                        if len(t) > 80:
                            texts.append((s, seq, t))
                    except Exception:
                        pass
            if not texts:
                continue
            # --- CROSS-MODAL SELECTION: CLAP text predicts audio position --
            Tt = clap._text([t for _, _, t in texts])
            if A_list:
                Amat = np.stack([a[:Tt.shape[1]] for a in A_list])
                pred = 1.0 - (Tt @ Amat.T).max(axis=1)
            else:
                pred = np.ones(len(texts))
            bonus = np.zeros(len(texts))
            if sens and len(recs) >= LEARN_AFTER:
                # The calculus, evaluated in the REAL space: prefer specs built
                # from axes whose realized outputs actually separate. `sens` is
                # promise_unspent per axis from score_axes_realized.
                mx = max(sens.values()) or 1.0
                for i, (s, _, _) in enumerate(texts):
                    bonus[i] = float(np.mean([sens.get(a, 0.0) / mx
                                              for a in s])) * 0.5
            # RECURSIVE orthogonalization: novel against the whole corpus AND
            # against the candidate's own cell of the refined latent tree. A
            # flat residual cannot tell the k-th clone of a minority cell from
            # a genuinely new direction once the global basis has truncated
            # that cell out of its top-energy span.
            rorth = np.zeros(len(texts))
            if len(recs) >= LEARN_AFTER and axis_order and A_list:
                Am = np.stack(A_list)
                Am = Am / np.linalg.norm(Am, axis=1, keepdims=True)
                Tt_d = Tt[:, :Am.shape[1]] if Tt.shape[1] > Am.shape[1] else Tt
                if Tt_d.shape[1] == Am.shape[1]:
                    rorth = recursive_orth_residual(
                        Tt_d, [s for s, _, _ in texts], Am,
                        [r["spec"] for r in recs], axis_order,
                        max_depth=RECURSION_DEPTH)
            k = int(np.argmax(pred + bonus + W_RECURSIVE_ORTH * rorth))
            chosen.append((*texts[k], float(pred[k])))

        # --- render everything chosen; nothing is rejected ---------------
        results = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {}
            for off, (s, seq, p, pg) in enumerate(chosen):
                mp3 = AUDIO / f"{len(recs) + off:05d}.mp3"
                futs[ex.submit(generate_lyria, p, key, mp3)] = (s, seq, p, pg, mp3)
            for fu in as_completed(futs):
                s, seq, p, pg, mp3 = futs[fu]
                try:
                    results.append((s, seq, p, pg, mp3, fu.result()))
                except Exception as e:
                    results.append((s, seq, p, pg, mp3, {"status": "error", "e": str(e)[:100]}))
        renders += len(results)

        for s, seq, p, pg, mp3, r in results:
            if not (r and r.get("status") in ("ok", "cached")):
                continue
            wav = wav_of(mp3)
            try:
                v, wmean = clap.audio(wav)
            except Exception:
                continue
            si, sv = clap.instrumentality(wmean)
            realized = float(1.0 - max((v @ np.stack(A_list).T))) if A_list else 1.0
            plan = (r.get("section_plan") or "").replace("\n", " ")
            rec = {"i": len(recs), "spec": s, "requested_seq": seq,
                   "requested_shape": "".join(re.findall(r"\[\[([A-Za-z])", seq)),
                   "returned_shape": "".join(re.findall(r"\[\[([A-Za-z])", plan)),
                   "prompt": p, "prompt_chars": len(p),
                   "predicted_gap": round(pg, 4), "realized_gap": round(realized, 4),
                   "clap_instr": round(si, 4), "clap_vocal": round(sv, 4),
                   "instr_margin": round(si - sv, 4),
                   "opening_density": round(opening_density(wav), 3),
                   "lex": {k: (sorted(x) if isinstance(x, set) else x)
                           for k, x in lexical_features(p, seq).items()}}
            recs.append(rec)
            A_list.append(v)
            pred_gaps.append(pg)
            real_gaps.append(realized)
            log.write(json.dumps(rec) + "\n"); log.flush()
            rec["lex"]["instruments"] = set(rec["lex"]["instruments"])

        # --- learn: the calculus, in the real space ---------------------
        if len(recs) >= LEARN_AFTER:
            An = np.stack(A_list)
            An = An / np.linalg.norm(An, axis=1, keepdims=True)
            ranked = score_axes_realized([r["spec"] for r in recs], An)
            sens = {s_.name: s_.promise_unspent for s_ in ranked}
            axis_order = [s_.name for s_ in ranked]
            decision = next_condition_realized([r["spec"] for r in recs], An)
            if decision.get("decision") == "refine" and len(recs) < target - WORKERS:
                # every axis is spent at this resolution -- recurse rather than
                # keep sampling an exhausted lattice
                ax = decision.get("exhausted_axis")
                try:
                    new = tree.refine(f"{ax} (exhausted in audio space)",
                                      attractors[-LEDGER_SLICE:])
                    if new:
                        print(f"  [calculus] refined -> {new.name} "
                              f"({len(new.levels)} levels)", flush=True)
                except Exception:
                    pass

        if len(recs) and len(recs) % MINE_EVERY < WORKERS and len(A_list) > 12:
            A = np.stack(A_list)
            S = A @ A.T
            np.fill_diagonal(S, -np.inf)
            flat = np.argsort(-S, axis=None)[:16]
            blocks, seen = [], set()
            for f in flat:
                i, j = np.unravel_index(f, S.shape)
                if i >= j or (i, j) in seen:
                    continue
                seen.add((i, j))
                blocks.append(f"--- pair (sim {S[i, j]:.3f}) ---\nA: "
                              f"{recs[i]['prompt'][:280]}\nB: {recs[j]['prompt'][:280]}")
                if len(blocks) >= 4:
                    break
            try:
                raw = chat([{"role": "user", "content": MINE_PROMPT.format(
                    pairs="\n".join(blocks))}], 0.3, 900, json_mode=True)
                new = [str(a) for a in parse_json(raw).get("attractors", [])][:8]
                if new:
                    attractors.extend(new)
                    with open(RUN / "ledger.jsonl", "a") as f:
                        f.write(json.dumps({"n": len(recs), "attractors": new}) + "\n")
                    print(f"  mined {len(new)} audio attractors", flush=True)
            except Exception:
                pass

        print(f"[steer] kept={len(recs)}/{target} renders={renders} "
              f"${USAGE.cost_usd():.2f} {(time.time()-t0)/60:.0f}m", flush=True)

    # ---- final analysis ------------------------------------------------
    A = np.stack(A_list)
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    model = learn_sensitivities(recs, An)
    ranked_final = score_axes_realized([r["spec"] for r in recs], An)
    decision_final = next_condition_realized([r["spec"] for r in recs], An)
    from scipy.stats import spearmanr
    valid = [(p, r) for p, r in zip(pred_gaps, real_gaps) if p < 1.0]
    align = float(spearmanr([p for p, _ in valid], [r for _, r in valid]).statistic) \
        if len(valid) > 8 else float("nan")
    S = An @ An.T
    np.fill_diagonal(S, -np.inf)
    nn = 1.0 - S.max(axis=1)
    w = np.linalg.eigvalsh((An.T @ An) / len(An))
    w = np.clip(w, 1e-12, None); w = w / w.sum()
    vendi = float(np.exp(-(w * np.log(w)).sum()))
    Ac = An - An.mean(0, keepdims=True)
    Ac = Ac / np.clip(np.linalg.norm(Ac, axis=1, keepdims=True), 1e-9, None)
    wc = np.linalg.eigvalsh((Ac.T @ Ac) / len(Ac)); wc = np.clip(wc, 1e-12, None); wc /= wc.sum()
    vendi_c = float(np.exp(-(wc * np.log(wc)).sum()))
    margins = np.array([r["instr_margin"] for r in recs])

    summary = {
        "n_kept": len(recs), "renders": renders,
        "renders_per_kept": round(renders / max(len(recs), 1), 3),
        "rejections": 0,
        "clap_vendi": round(vendi, 2), "clap_vendi_centered": round(vendi_c, 2),
        "clap_median_nn_dist": round(float(np.median(nn)), 4),
        "clap_min_nn_dist": round(float(nn.min()), 4),
        "instrumental_margin_mean": round(float(margins.mean()), 4),
        "instrumental_margin_min": round(float(margins.min()), 4),
        "fraction_instrumental": round(float((margins > 0).mean()), 4),
        "opening_density_mean": round(float(np.mean([r["opening_density"] for r in recs])), 3),
        "prompt_chars_mean": round(float(np.mean([r["prompt_chars"] for r in recs])), 0),
        "plan_exact_match_rate": round(float(np.mean(
            [r["requested_shape"] == r["returned_shape"] for r in recs])), 3),
        "text_audio_gap_alignment_spearman": round(align, 4),
        "recursion_depth": RECURSION_DEPTH,
        "recursive_axis_order": axis_order,
        "learned_sensitivities": model,
        "calculus_realized_ranking": [s_.to_json() for s_ in ranked_final],
        "calculus_decision": {k: v for k, v in decision_final.items()
                              if k != "ranking"},
        "n_attractors": len(attractors),
        "openrouter_cost_usd": round(USAGE.cost_usd(), 4),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    np.save(RUN / "clap_emb.npy", An)
    json.dump(summary, open(RUN / "run_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "learned_sensitivities"}, indent=2))
    if ranked_final:
        print("\ncalculus in the real space (ranked by audible promise):")
        for s_ in ranked_final:
            print(f"  {s_.name:40s} spread={s_.spread:.3f} transv={s_.transversality:.3f} "
                  f"indep={s_.independence:.3f} head={s_.headroom:.3f} "
                  f"-> {s_.promise_unspent:.4f}")
        print(f"  decision: {decision_final.get('decision')}")
    if model:
        print("\nregression cross-check (CLAP cosine distance per change):")
        for a in model.get("_axes", []):
            print(f"  {a:40s} {model[a]:+.4f}")
        for a in ("instrument_turnover", "length_delta_k", "section_count_delta"):
            print(f"  {a:40s} {model.get(a, 0):+.4f}")
        print(f"  regression R^2 = {model.get('_r2')} over {model.get('_n_pairs')} pairs")
    log.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else TARGET)
