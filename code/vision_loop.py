"""
The vision-steered arm: close the loop in the space the product lives in.

Everything else in this paper steers on TEXT. For an image-generation
instruction set that is a proxy, and §7 shows the proxy is weak: instructions
that are lexically and semantically varied still render to pictures that look
alike, because the image model contributes its own mode structure downstream
of ours.

This arm therefore adds two capabilities and uses both to steer:

  text -> vision      we can RENDER an instruction, so a candidate's true
                      position is observable, not merely predictable.
  vision embedding    CLIP gives us the rendered image's coordinates, so the
                      same orthogonalization machinery works on pictures.
  vision judge        a VLM looks at the rendered image and scores it against
                      the instruction and against a craft rubric, which is the
                      quality term the text-only pipeline could only guess at.

The important structural change is WHERE the feedback enters. Rendering is
expensive, so we cannot render every candidate. Instead we render a bounded
sample of accepted items, embed and judge them, and feed two things back into
the TEXT-side loop:

  1. a vision-side occupied basis, projected onto the text embeddings of the
     instructions that produced those images. This gives a linear map from
     "directions the IMAGES are crowding into" to "directions in instruction
     space", so the text-side orthogonality term can push away from visual
     redundancy it cannot itself see.
  2. mined VISUAL attractors from the judge -- what the pictures keep doing --
     appended to the same append-only ledger as the textual ones, and
     therefore repelled against in the next prompts.

This is the paper's mechanism applied one level down: the ledger already
repels against what the model keeps SAYING; here it also repels against what
the model keeps SHOWING.
"""
from __future__ import annotations

import base64
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pipeline import HERE, USAGE, chat, embed, parse_json
from real_run import Corpus, DOMAINS, EMBED_BATCH, item_text, nn_cos_dist
from render_images import render_one, _load_openai_key

_load_openai_key()

WORKERS = 12
K = 2
RENDER_EVERY = 10        # render 1 in this many accepted items
VISION_ROUND = 40        # re-fit the vision basis / mine visual attractors
VISION_SAMPLE = 12       # images shown to the vision judge per round
LEDGER_SLICE = 10
GATE_Z = 2.6
W_TEXT_ORTH = 0.35
W_VIS_ORTH = 0.30
W_GAP = 0.35

VISION_JUDGE_PROMPT = """You are looking at {n} images that were generated from \
independently written instructions for POST-MODERN artworks. They are supposed to be \
different from one another.

Two tasks.

1. VISUAL ATTRACTORS: name what these images KEEP DOING -- recurring palettes, recurring \
objects, recurring compositions, recurring textures, recurring gestures toward \
"post-modern". Be concrete and specific (not "bold colors" but which colors; not \
"collage" but what is being collaged and how it is arranged).

2. CRAFT: rate the SET on how visually distinct its members are from each other, 0-10, \
where 0 means near-identical and 10 means each image is its own visual world.

Return JSON only:
{{"visual_attractors": ["...", "..."], "set_distinctness": 0, "most_redundant_pair": "..."}}"""


def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def vision_judge(paths: list[Path], model: str = "openai/gpt-5.6-luna") -> dict:
    """Multimodal call: images go in as data: URIs on the OpenAI-compatible
    content-parts schema that OpenRouter accepts."""
    content = [{"type": "text", "text": VISION_JUDGE_PROMPT.format(n=len(paths))}]
    for p in paths:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_png(p)}"}})
    raw = chat([{"role": "user", "content": content}], temperature=0.2,
               max_tokens=1200, json_mode=True)
    try:
        return parse_json(raw)
    except ValueError:
        return {}


def _as_tensor(out):
    """transformers 5.x returns a ModelOutput from get_image_features where
    4.x returned a plain tensor; accept either."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v if v.dim() == 2 else v.mean(dim=1)
    raise TypeError(f"cannot extract image features from {type(out)}")


def clip_encoder():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)

    def enc(paths: list[Path]) -> np.ndarray:
        with torch.no_grad():
            imgs = [Image.open(p).convert("RGB") for p in paths]
            inp = proc(images=imgs, return_tensors="pt")
            f = _as_tensor(model.get_image_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()
    return enc


def occupied(E: np.ndarray, frac: float = 0.8) -> np.ndarray:
    if len(E) < 6:
        return np.zeros((E.shape[1], 0))
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    w, V = np.linalg.eigh((En.T @ En) / len(En))
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    j = int(np.searchsorted(np.cumsum(w) / max(w.sum(), 1e-12), frac)) + 1
    return V[:, :j]


def run(target: int = 240, budget_usd: float = 6.0, image_budget: int = 120):
    """Vision-steered DALL-E instruction generation."""
    domain = "dalle"
    C = Corpus(domain, "vision")
    rng = random.Random(555 + C.n)
    cfg = DOMAINS[domain]
    img_dir = C.dir / "images"
    img_dir.mkdir(exist_ok=True)
    enc = clip_encoder()

    # persisted vision state
    vs_path = C.dir / "vision_state.json"
    state = json.loads(vs_path.read_text()) if vs_path.exists() else {
        "rendered": [], "visual_attractors": [], "distinctness": []}
    rendered: list[int] = state["rendered"]
    visual_attractors: list[str] = state["visual_attractors"]

    Vimg = np.zeros((512, 0))      # CLIP-space occupied basis
    Mmap = None                    # text-space image of that basis
    t0 = time.time()
    n_images = len(list(img_dir.glob("*.png")))

    from axes import AxisTree, Axis, spec_text
    tree = AxisTree(C.axes_path)
    if not tree.axes:
        raw = chat([{"role": "user", "content": cfg["seed_axes"].format(k=7)}],
                   temperature=1.0, max_tokens=4000, json_mode=True)
        for a in parse_json(raw)["axes"]:
            lv = [str(x) for x in a["levels"] if str(x).strip()]
            if len(lv) >= 3:
                tree.axes.append(Axis(str(a["name"]), lv))
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})

    print(f"[vision] start n={C.n} images={n_images}", flush=True)
    while C.n < target and USAGE.cost_usd() < budget_usd:
        n_slots = max(1, min(WORKERS // K, target - C.n))
        specs = [tree.sample_spec(rng) for _ in range(n_slots)]
        avoid_txt = C.attractors[-LEDGER_SLICE:]
        avoid_vis = visual_attractors[-LEDGER_SLICE:]
        avoid = "\n".join(f"  - {a}" for a in avoid_txt)
        if avoid_vis:
            avoid += "\n" + "\n".join(
                f"  - [seen in the RENDERED IMAGES] {a}" for a in avoid_vis)
        prompts, pspec = [], []
        for s in specs:
            for _ in range(K):
                prompts.append(cfg["ihd_prompt"].format(
                    contracts=spec_text(s), avoid=avoid or "  (nothing yet)"))
                pspec.append(s)

        got = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(chat, [{"role": "user", "content": p}], 1.0, 700): i
                    for i, p in enumerate(prompts)}
            for fu in as_completed(futs):
                try:
                    got.append((futs[fu], fu.result()))
                except Exception:
                    pass
        parsed = [(i, item_text(domain, r)) for i, r in got]
        parsed = [(i, t) for i, t in parsed if t]
        if not parsed:
            continue
        texts = [t for _, t in parsed]
        E = embed(texts, batch=EMBED_BATCH)

        slots: dict[int, list[int]] = {}
        for idx, (i, _) in enumerate(parsed):
            slots.setdefault(i // K, []).append(idx)
        Etxt = C.E if C.E is not None and len(C.E) else None
        Vtxt = occupied(Etxt) if Etxt is not None else np.zeros((E.shape[1], 0))
        for slot, idxs in sorted(slots.items()):
            if C.n >= target:
                break
            cand = E[idxs]
            nn = nn_cos_dist(cand, C, rng)
            o_txt = (1.0 - ((cand @ Vtxt) ** 2).sum(axis=1)
                     if Vtxt.shape[1] else np.zeros(len(cand)))
            # vision-side orthogonality, expressed in text space via Mmap
            if Mmap is not None and Mmap.shape[1]:
                o_vis = 1.0 - ((cand @ Mmap) ** 2).sum(axis=1)
            else:
                o_vis = np.zeros(len(cand))
            U = (W_TEXT_ORTH * o_txt + W_VIS_ORTH * o_vis
                 + W_GAP * np.minimum(nn / 0.25, 1.0))
            if C.n > 60 and Etxt is not None:
                c = Etxt.mean(axis=0)
                z = np.linalg.norm(cand - c[None, :], axis=1)
                bad = z > GATE_Z * float(np.linalg.norm(Etxt - c[None, :], axis=1).mean())
                if not bad.all():
                    U = np.where(bad, -np.inf, U)
            b = int(np.argmax(U))
            gi = idxs[b]
            C.add(texts[gi], cand[b], pspec[parsed[gi][0]],
                  {"arm": "vision", "gap": round(float(nn[b]), 5),
                   "o_txt": round(float(o_txt[b]), 4),
                   "o_vis": round(float(o_vis[b]), 4)})

            if C.n % RENDER_EVERY == 0 and n_images < image_budget:
                idx = C.n - 1
                if render_one(idx, C.texts[idx], img_dir):
                    rendered.append(idx)
                    n_images += 1

        # ---- vision feedback round ----
        if len(rendered) >= 6 and len(rendered) % (VISION_ROUND // RENDER_EVERY) == 0:
            paths = [img_dir / f"{i:05d}.png" for i in rendered]
            paths = [p for p in paths if p.exists()]
            if len(paths) >= 6:
                Ei = enc(paths)
                Vimg = occupied(Ei)
                # map the crowded IMAGE directions into TEXT space: regress the
                # instruction embeddings on the image-space coordinates, so the
                # text-side score can penalize visually-redundant directions.
                idxs = [i for i in rendered if (img_dir / f"{i:05d}.png").exists()]
                Et = C.E[np.array(idxs)]
                A = Ei @ Vimg                      # (nimg, k) image coords
                # least-squares text-space direction for each image direction
                Mraw, *_ = np.linalg.lstsq(A, Et, rcond=None)   # (k, dtext)
                Q, R = np.linalg.qr(Mraw.T)
                keep = np.abs(np.diag(R)) > 1e-8
                Mmap = Q[:, keep] if keep.any() else None
                sample = random.sample(paths, min(VISION_SAMPLE, len(paths)))
                try:
                    vj = vision_judge(sample)
                    va = [str(x) for x in vj.get("visual_attractors", [])][:10]
                    if va:
                        visual_attractors.extend(va)
                        C.add_mined([f"[visual] {a}" for a in va])
                    state["distinctness"].append(
                        {"n": C.n, "set_distinctness": vj.get("set_distinctness")})
                    print(f"  [vision] n={C.n} judge distinctness="
                          f"{vj.get('set_distinctness')} attractors={len(va)}", flush=True)
                except Exception as e:
                    print(f"  vision judge failed: {e}", flush=True)
                state["rendered"] = rendered
                state["visual_attractors"] = visual_attractors
                vs_path.write_text(json.dumps(state, indent=2))

        if C.n % 40 < WORKERS:
            C.checkpoint()
            print(f"[vision] n={C.n} imgs={n_images} ${USAGE.cost_usd():.2f} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)

    C.checkpoint()
    state["rendered"] = rendered
    state["visual_attractors"] = visual_attractors
    vs_path.write_text(json.dumps(state, indent=2))
    summary = {"arm": "vision", "n": C.n, "images": n_images,
               "cost_usd_openrouter": round(USAGE.cost_usd(), 4),
               "visual_attractors": len(visual_attractors),
               "distinctness_track": state["distinctness"],
               "wall_clock_min": round((time.time() - t0) / 60, 1)}
    with open(C.dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    C.close()


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 240,
        float(sys.argv[2]) if len(sys.argv) > 2 else 6.0,
        int(sys.argv[3]) if len(sys.argv) > 3 else 120)
