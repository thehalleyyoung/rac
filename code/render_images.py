"""
Render DALL-E instructions to actual images, then embed the images with CLIP.

Why this exists. Every diversity number elsewhere in this paper is measured on
TEXT -- either the literal tokens or a text embedding. For an
image-generation instruction set, neither is the product. The product is the
pictures. A set of instructions can be lexically varied and semantically
varied and still render to two hundred images that look like each other,
because the image model has its own mode structure that composes with the
text model's, and because much of what varies in the text ("post-modern",
"appropriated source") lands in the same visual place.

So we render the first N instructions and measure diversity a third time, in
a VISION embedding space (CLIP image encoder), which is the space the
end-user's experience actually lives in. Three numbers per corpus:

    literal (n-gram)  ->  latent-text (nomic)  ->  latent-vision (CLIP)

and the interesting result is where they disagree.

Cost control: gpt-image-1-mini at low quality, 1024x1024, one image per
instruction, hard-capped and fully resumable -- every image is written to
disk immediately and an existing file is never re-requested.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMG_URL = "https://api.openai.com/v1/images/generations"
IMG_MODEL = "gpt-image-1-mini"
SIZE = "1024x1024"
QUALITY = "medium"   # "low" was used for the first 199-image run; medium is 4x the image tokens
WORKERS = 10
MAX_IMAGES = 100


def _load_openai_key():
    if os.environ.get("OPENAI_API_KEY"):
        return
    for rc in (Path.home() / ".zshrc", Path.home() / ".bashrc"):
        if not rc.is_file():
            continue
        for line in rc.read_text().splitlines():
            m = re.match(r'\s*export\s+OPENAI_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                os.environ["OPENAI_API_KEY"] = m.group(1)
                return


_load_openai_key()

_lock = threading.Lock()
_stats = {"ok": 0, "fail": 0, "skip": 0}


def render_one(idx: int, prompt: str, out_dir: Path, quality: str = QUALITY) -> bool:
    path = out_dir / f"{idx:05d}.png"
    if path.exists() and path.stat().st_size > 1000:
        with _lock:
            _stats["skip"] += 1
        return True
    key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({"model": IMG_MODEL, "prompt": prompt[:3800],
                       "size": SIZE, "quality": quality, "n": 1}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                IMG_URL, data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.load(r)
            b64 = payload["data"][0].get("b64_json")
            if not b64:
                raise RuntimeError("no b64_json in response")
            tmp = path.with_suffix(".png.part")
            tmp.write_bytes(base64.b64decode(b64))
            os.replace(tmp, path)
            with _lock:
                _stats["ok"] += 1
            return True
        except Exception as e:
            if attempt == 2:
                with _lock:
                    _stats["fail"] += 1
                print(f"  [{idx}] failed: {str(e)[:160]}", flush=True)
                return False
            time.sleep(2 ** attempt + 1)
    return False


def render(run: str = "dalle_naive", limit: int = MAX_IMAGES,
           quality: str = QUALITY, subdir: str | None = None):
    """`subdir` keeps quality tiers apart, so a medium-quality cross-arm
    comparison is never contaminated by the earlier low-quality run."""
    src = HERE / "real" / run / "corpus.jsonl"
    out_dir = HERE / "real" / run / (subdir or f"images_{quality}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        try:
            prompts.append(json.loads(line)["text"])
        except (json.JSONDecodeError, KeyError):
            continue
        if len(prompts) >= limit:
            break
    print(f"rendering {len(prompts)} images from {run} -> {out_dir}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(render_one, i, p, out_dir, quality): i
                for i, p in enumerate(prompts)}
        done = 0
        for fu in as_completed(futs):
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(prompts)} ok={_stats['ok']} "
                      f"skip={_stats['skip']} fail={_stats['fail']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    n_files = len(list(out_dir.glob("*.png")))
    print(f"done: {n_files} images on disk, {json.dumps(_stats)}", flush=True)
    with open(HERE / "real" / run / f"render_summary_{quality}.json", "w") as f:
        json.dump({"model": IMG_MODEL, "size": SIZE, "quality": quality,
                   "requested": len(prompts), "on_disk": n_files, **_stats}, f, indent=2)


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


def clip_embed(run: str = "dalle_naive", subdir: str = "images_medium",
               tag: str = "medium"):
    """Embed rendered images with the CLIP vision encoder."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    out_dir = HERE / "real" / run / subdir
    files = sorted(out_dir.glob("*.png"))
    if not files:
        print("no images to embed")
        return
    print(f"embedding {len(files)} images with CLIP ...", flush=True)
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)
    vecs, idxs = [], []
    B = 16
    with torch.no_grad():
        for i in range(0, len(files), B):
            chunk = files[i:i + B]
            imgs = [Image.open(f).convert("RGB") for f in chunk]
            inp = proc(images=imgs, return_tensors="pt")
            f = _as_tensor(model.get_image_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
            vecs.append(f.numpy())
            idxs.extend(int(p.stem) for p in chunk)
            print(f"  {min(i + B, len(files))}/{len(files)}", flush=True)
    E = np.vstack(vecs)
    np.save(HERE / "real" / run / f"clip_image_emb_{tag}.npy", E)
    with open(HERE / "real" / run / f"clip_index_{tag}.json", "w") as f:
        json.dump(idxs, f)
    print("saved CLIP image embeddings", E.shape, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    run = sys.argv[2] if len(sys.argv) > 2 else "dalle_naive"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_IMAGES
    quality = sys.argv[4] if len(sys.argv) > 4 else QUALITY
    sub = f"images_{quality}"
    if cmd == "render":
        render(run, limit, quality, sub)
    elif cmd == "embed":
        clip_embed(run, sub, quality)
    elif cmd == "both":
        render(run, limit, quality, sub)
        clip_embed(run, sub, quality)
