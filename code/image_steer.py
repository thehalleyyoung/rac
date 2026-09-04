"""
Cross-modal steering for images: the audio loop, one modality over.

Identical structure to lyria_steer.py, and deliberately so -- the point of the
calculus is that it is modality-agnostic once you can embed the artifact. CLIP
plays the role CLAP played: a shared text/image space, so CLIP_text(prompt) is
a free pre-render estimate of CLIP_image(render), and selection can happen over
prompts rather than over renders.

No rejection. Every image generated is kept. All the leverage is in choosing
which prompt to send, and all the learning is about which natural-language
axes actually move the picture.

Two objectives, run as separate arms over the same machinery, because they
want different things from the same budget of n renders:

  maxmin    "in n turns, make no two images resemble each other" -- pick the
            candidate whose predicted landing point is FURTHEST from the
            nearest already-rendered image. Frontier-seeking.
  coverage  "in n turns, reach as much of the space as possible" -- pick the
            candidate that brings the most currently-uncovered reference
            points within eps. Bulk-filling.

Running both on one generator, one embedder, and one budget is the cleanest
available test of the claim that these are different problems: if they were
the same, the two arms would converge to the same corpus.

The validity gate is verified, never enforced (we may not reject): each image
gets a CLIP margin between "a post-modern artwork" descriptions and generic
photo descriptions, reported as a distribution.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from axes import Axis, AxisTree, spec_text
from calculus import (coverage_levels, farthest_levels, next_condition_realized,
                      recursive_axis_order, recursive_orth_residual,
                      score_axes_coverage, score_axes_realized)
from pipeline import HERE, USAGE, chat, parse_json
from render_images import _load_openai_key, render_one

_load_openai_key()

TARGET = 60
WORKERS = 6
POOL = 6
QUALITY = "high"
MINE_EVERY = 20
LEDGER_SLICE = 8
LEARN_AFTER = 14
RECURSION_DEPTH = 2      # levels of the latent tree to orthogonalize within
W_RECURSIVE_ORTH = 0.6
EPS_QUANTILE = 0.10      # coverage radius, as a quantile of reference distances

ON_PROBES = [
    "a post-modern artwork",
    "a work of contemporary conceptual art",
    "an art piece using appropriation and collage",
]
OFF_PROBES = [
    "a plain photograph of an everyday object",
    "a stock photo",
    "a screenshot of a user interface",
]

WRITE_PROMPT = """Write ONE image-generation instruction for a post-modern artwork. \
One paragraph, 30-70 words, concrete and visual.

Let these choices shape the work. Express each as something visible, never as \
abstract art-theory:
{contracts}

The set so far keeps doing these things. Avoid all of them:
{avoid}

Name the medium, the composition, the palette, the surface, and the treatment. \
Output only the instruction."""

MINE_PROMPT = """These pairs of art-generation instructions produced images that an \
image embedding model judged nearly IDENTICAL, despite the instructions being written \
to differ. Say what the resulting pictures keep sharing -- palettes, recurring objects, \
compositions, textures, gestures toward "post-modern".

{pairs}

Return JSON only: {{"attractors": ["...", "..."]}} -- 4-8 concrete findings."""


class Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.on = self.text(ON_PROBES)
        self.off = self.text(OFF_PROBES)

    @staticmethod
    def _t(out):
        import torch
        if isinstance(out, torch.Tensor):
            return out
        for a in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(out, a, None)
            if v is not None:
                return v if v.dim() == 2 else v.mean(dim=1)
        raise TypeError(type(out))

    def text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            i = self.proc(text=[t[:300] for t in texts], return_tensors="pt",
                          padding=True, truncation=True)
            f = self._t(self.model.get_text_features(**i))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()

    def image(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), 8):
                imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
                inp = self.proc(images=imgs, return_tensors="pt")
                f = self._t(self.model.get_image_features(**inp))
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.numpy())
        return np.vstack(out)

    def artness(self, v: np.ndarray) -> tuple[float, float]:
        v = v / max(np.linalg.norm(v), 1e-9)
        return float((self.on @ v).mean()), float((self.off @ v).mean())


def load_axes(tree: AxisTree) -> AxisTree:
    src = HERE / "real" / "dalle_ihd" / "axes.jsonl"
    if not tree.axes and src.exists():
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r["event"] == "seed":
                tree.axes = [Axis(a["name"], a["levels"]) for a in r["axes"]]
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})
    return tree


def run(objective: str, target: int = TARGET):
    assert objective in ("maxmin", "coverage")
    RUN = HERE / "real" / f"dalle_steer_{objective}"
    IMG = RUN / "images"
    IMG.mkdir(parents=True, exist_ok=True)
    rng = random.Random(90210 if objective == "maxmin" else 11223)
    clip = Clip()
    tree = load_axes(AxisTree(RUN / "axes.jsonl"))
    print(f"[{objective}] axes={len(tree.axes)} lattice~{tree.space_size():.1e}", flush=True)

    # reference pool for the coverage objective: a large, policy-independent
    # sample of what the generator reaches unprompted. Coverage is only
    # meaningful against a measure, and this is the honest one available.
    ref = None
    if objective == "coverage":
        rp = HERE / "real" / "dalle_naive" / f"clip_image_emb_{QUALITY}.npy"
        if not rp.exists():
            rp = HERE / "real" / "dalle_naive" / "clip_image_emb_medium.npy"
        if rp.exists():
            ref = np.load(rp)
            ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
            d = 1.0 - ref @ ref.T
            eps = float(np.quantile(d[np.triu_indices(len(ref), 1)], EPS_QUANTILE))
            print(f"[coverage] reference pool {ref.shape}, eps={eps:.4f} "
                  f"(q{EPS_QUANTILE:.2f} of its own pairwise distances)", flush=True)
        else:
            print("[coverage] no reference pool; falling back to self-coverage",
                  flush=True)
            eps = 0.25
    else:
        eps = None

    recs: list[dict] = []
    E: list[np.ndarray] = []
    attractors: list[str] = []
    sens: dict = {}
    axis_order: list[str] = []
    covered = np.zeros(len(ref), dtype=bool) if ref is not None else None
    renders = 0
    pred, real = [], []
    t0 = time.time()
    log = open(RUN / "log.jsonl", "a")

    while len(recs) < target:
        n_slots = min(WORKERS, target - len(recs))
        chosen = []
        for _ in range(n_slots):
            pool = [tree.sample_spec(rng) for _ in range(POOL)]
            avoid = ("\n".join(f"  - {a}" for a in attractors[-LEDGER_SLICE:])
                     or "  - (nothing yet)")
            texts = []
            with ThreadPoolExecutor(max_workers=POOL) as ex:
                futs = {ex.submit(chat, [{"role": "user", "content": WRITE_PROMPT.format(
                    contracts=spec_text(s), avoid=avoid)}], 1.0, 900): s for s in pool}
                for fu in as_completed(futs):
                    s = futs[fu]
                    try:
                        t = fu.result().strip().strip('"')
                        if len(t) > 40:
                            texts.append((s, t))
                    except Exception:
                        pass
            if not texts:
                continue
            Tt = clip.text([t for _, t in texts])
            # ---- the two objectives diverge exactly here ----------------
            if objective == "maxmin":
                score = (1.0 - (Tt @ np.stack(E).T).max(axis=1)) if E else np.ones(len(texts))
            else:
                if ref is not None:
                    d = 1.0 - Tt @ ref.T                     # (cand, ref)
                    score = ((d <= eps) & ~covered[None, :]).sum(axis=1).astype(float)
                    score = score / max(len(ref), 1)
                else:
                    score = (1.0 - (Tt @ np.stack(E).T).max(axis=1)) if E else np.ones(len(texts))
            bonus = np.zeros(len(texts))
            if sens and len(recs) >= LEARN_AFTER:
                mx = max(sens.values()) or 1.0
                for i, (s, _) in enumerate(texts):
                    bonus[i] = float(np.mean([sens.get(a, 0.0) / mx for a in s]))
            # RECURSIVE orthogonalization: novel globally AND novel inside the
            # candidate's own cell of the (recursively refined) latent tree.
            # A flat residual cannot separate the k-th clone of a minority cell
            # from a genuinely new direction once the global basis has
            # truncated that cell away.
            rorth = np.zeros(len(texts))
            if len(recs) >= LEARN_AFTER and axis_order:
                A = np.stack(E); A = A / np.linalg.norm(A, axis=1, keepdims=True)
                rorth = recursive_orth_residual(
                    Tt, [s for s, _ in texts], A, [r["spec"] for r in recs],
                    axis_order, max_depth=RECURSION_DEPTH)
            k = int(np.argmax(score / max(score.max(), 1e-9)
                              + 0.5 * bonus + W_RECURSIVE_ORTH * rorth))
            chosen.append((*texts[k], float(score[k])))

        results = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {}
            for off, (s, p, sc) in enumerate(chosen):
                idx = len(recs) + off
                futs[ex.submit(render_one, idx, p, IMG, QUALITY)] = (idx, s, p, sc)
            for fu in as_completed(futs):
                idx, s, p, sc = futs[fu]
                try:
                    ok = fu.result()
                except Exception:
                    ok = False
                results.append((idx, s, p, sc, ok))
        renders += len(results)

        new_paths, new_meta = [], []
        for idx, s, p, sc, ok in sorted(results):
            f = IMG / f"{idx:05d}.png"
            if ok and f.exists():
                new_paths.append(f)
                new_meta.append((s, p, sc))
        if not new_paths:
            continue
        V = clip.image(new_paths)
        for v, (s, p, sc), f in zip(V, new_meta, new_paths):
            on, off_ = clip.artness(v)
            realized = float(1.0 - max(v @ np.stack(E).T)) if E else 1.0
            rec = {"i": len(recs), "spec": s, "prompt": p, "file": f.name,
                   "predicted_score": round(sc, 5), "realized_gap": round(realized, 4),
                   "clip_on": round(on, 4), "clip_off": round(off_, 4),
                   "art_margin": round(on - off_, 4)}
            recs.append(rec); E.append(v)
            pred.append(sc); real.append(realized)
            if covered is not None:
                covered |= (1.0 - ref @ v) <= eps
            log.write(json.dumps(rec) + "\n"); log.flush()

        if len(recs) >= LEARN_AFTER:
            A = np.stack(E)
            A = A / np.linalg.norm(A, axis=1, keepdims=True)
            scorer = score_axes_realized if objective == "maxmin" else None
            if objective == "maxmin":
                ranked = score_axes_realized([r["spec"] for r in recs], A)
            else:
                from calculus import realized_level_usage, realized_level_vectors
                lv = realized_level_vectors([r["spec"] for r in recs], A)
                usage = realized_level_usage([r["spec"] for r in recs])
                ranked = score_axes_coverage(lv, A, usage) if lv else []
            sens = {s_.name: s_.promise_unspent for s_ in ranked}
            # descend the tree most-discriminative-axis-first, so the deepest
            # cells are the ones that actually separate the artifacts
            axis_order = [s_.name for s_ in ranked]

        if len(recs) and len(recs) % MINE_EVERY < WORKERS and len(E) > 10:
            A = np.stack(E)
            S = A @ A.T
            np.fill_diagonal(S, -np.inf)
            blocks, seen = [], set()
            for f in np.argsort(-S, axis=None)[:16]:
                i, j = np.unravel_index(f, S.shape)
                if i >= j or (i, j) in seen:
                    continue
                seen.add((i, j))
                blocks.append(f"--- pair (sim {S[i, j]:.3f}) ---\nA: {recs[i]['prompt'][:260]}"
                              f"\nB: {recs[j]['prompt'][:260]}")
                if len(blocks) >= 4:
                    break
            try:
                raw = chat([{"role": "user", "content": MINE_PROMPT.format(
                    pairs="\n".join(blocks))}], 0.3, 800, json_mode=True)
                new = [str(a) for a in parse_json(raw).get("attractors", [])][:8]
                if new:
                    attractors.extend(new)
                    with open(RUN / "ledger.jsonl", "a") as fh:
                        fh.write(json.dumps({"n": len(recs), "attractors": new}) + "\n")
                    print(f"  mined {len(new)} visual attractors", flush=True)
            except Exception:
                pass

        print(f"[{objective}] kept={len(recs)}/{target} renders={renders} "
              f"${USAGE.cost_usd():.2f} {(time.time()-t0)/60:.0f}m", flush=True)

    A = np.stack(E)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    S = A @ A.T
    np.fill_diagonal(S, -np.inf)
    nn = 1.0 - S.max(axis=1)
    Ac = A - A.mean(0, keepdims=True)
    Ac = Ac / np.clip(np.linalg.norm(Ac, axis=1, keepdims=True), 1e-9, None)
    w = np.linalg.eigvalsh((Ac.T @ Ac) / len(Ac)); w = np.clip(w, 1e-12, None); w /= w.sum()
    vendi_c = float(np.exp(-(w * np.log(w)).sum()))
    from scipy.stats import spearmanr
    v = [(p, r) for p, r in zip(pred, real) if r < 1.0]
    align = float(spearmanr([a for a, _ in v], [b for _, b in v]).statistic) if len(v) > 8 else float("nan")
    summary = {
        "objective": objective, "n": len(recs), "renders": renders, "rejections": 0,
        "quality": QUALITY,
        "clip_vendi_centered": round(vendi_c, 2),
        "clip_median_nn_dist": round(float(np.median(nn)), 4),
        "clip_min_nn_dist": round(float(nn.min()), 4),
        "coverage_of_reference": (round(float(covered.mean()), 4)
                                  if covered is not None else None),
        "coverage_eps": round(eps, 4) if eps else None,
        "art_margin_mean": round(float(np.mean([r["art_margin"] for r in recs])), 4),
        "fraction_on_concept": round(float(np.mean(
            [r["art_margin"] > 0 for r in recs])), 4),
        "pred_vs_realized_spearman": round(align, 4),
        "recursion_depth": RECURSION_DEPTH,
        "recursive_axis_order": axis_order,
        "calculus_ranking": [s_.to_json() for s_ in (
            score_axes_realized([r["spec"] for r in recs], A) if objective == "maxmin"
            else score_axes_coverage(*_cov_args(recs, A)))],
        "openrouter_cost_usd": round(USAGE.cost_usd(), 4),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    np.save(RUN / "clip_emb.npy", A)
    json.dump(summary, open(RUN / "run_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "calculus_ranking"}, indent=2))
    log.close()


def _cov_args(recs, A):
    from calculus import realized_level_usage, realized_level_vectors
    return (realized_level_vectors([r["spec"] for r in recs], A), A,
            realized_level_usage([r["spec"] for r in recs]))


if __name__ == "__main__":
    obj = sys.argv[1] if len(sys.argv) > 1 else "maxmin"
    run(obj, int(sys.argv[2]) if len(sys.argv) > 2 else TARGET)
