"""
Audio-domain diversity: prompts, and the music they actually produce.

Same structure as vision_compare.py, one level further down the chain. For
each arm we hold the prompt corpus and the rendered tracks side by side and
measure:

    literal (n-gram over prompts)
  -> latent text (nomic over prompts)
  -> CLAP over audio   (caption-aligned; "how would you describe this track")
  -> MERT over audio   (self-supervised music representation; pitch, harmony,
                        timbre structure that CLAP's caption space discards)
  -> section plan      (Lyria reports its own form, e.g. [[A0]][[B1]][[C2]];
                        distinct plans are a discrete, model-reported measure
                        of formal variety that needs no embedding at all)

Two audio embedders rather than one because they disagree, and the
disagreement is informative: CLAP can call two tracks similar because both are
describable as "warm ambient piano" while MERT hears that one never leaves a
single chord and the other modulates twice.

Both are window models, so a track vector is [mean, std, mean |first
difference|] over windows -- a static drone and a piece that develops through
three sections must not collide, and mean-pooling alone would collide them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from metrics import (distinct_n, embed_vendi_centered, exact_dup_rate,
                     ngram_vendi, nn_stats, self_repetition)

HERE = Path(__file__).resolve().parent
ARMS = ["naive", "ihd"]
LABEL = {"naive": "naive prompting", "ihd": "IHD (ours)"}


def load(arm: str):
    d = HERE / "real" / f"lyria_{arm}"
    if not (d / "corpus.jsonl").exists():
        return None
    texts = [json.loads(l)["text"] for l in (d / "corpus.jsonl").read_text().splitlines()
             if l.strip()]
    Et = np.load(d / "embeddings.npy")
    files = sorted((d / "audio").glob("*.mp3")) if (d / "audio").exists() else []
    idx = [int(f.stem) for f in files]
    plans = []
    for f in files:
        pf = f.with_suffix(".plan.txt")
        plans.append(pf.read_text().strip() if pf.exists() else "")
    return texts, Et, files, idx, plans


def wavify(files: list[Path]) -> list[Path]:
    """Lyria returns mpeg; soundfile needs wav. ffmpeg once, then cached."""
    import subprocess
    out = []
    for f in files:
        w = f.with_suffix(".wav")
        if not w.exists() or w.stat().st_size < 10_000:
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(f),
                            "-ac", "1", "-ar", "48000", str(w)], check=False)
        if w.exists():
            out.append(w)
    return out


def plan_stats(plans: list[str]) -> dict:
    """Lyria's own section labels, e.g. '[[A0]] [[B1]] [[C2]] [[D3]]'.
    We count distinct LETTER sequences (A,B,C,D...) -- the shape of the form,
    ignoring the index suffixes."""
    shapes = []
    for p in plans:
        letters = re.findall(r"\[\[([A-Za-z])\d*\]\]", p)
        if letters:
            shapes.append("".join(letters))
    if not shapes:
        return {}
    from collections import Counter
    c = Counter(shapes)
    n_sections = [len(s) for s in shapes]
    n_unique_per = [len(set(s)) for s in shapes]
    return {
        "n_with_plan": len(shapes),
        "distinct_form_shapes": len(c),
        "form_shape_diversity": round(len(c) / len(shapes), 4),
        "mean_sections": round(float(np.mean(n_sections)), 2),
        "mean_distinct_sections": round(float(np.mean(n_unique_per)), 2),
        "most_common_form": c.most_common(1)[0][0],
        "most_common_count": c.most_common(1)[0][1],
    }


def main():
    from audio_domain import clap_embed, mert_embed
    data = {}
    for arm in ARMS:
        got = load(arm)
        if got:
            data[arm] = got
    if not data:
        print("no lyria corpora found")
        return
    n_audio = min(len(g[2]) for g in data.values())
    if n_audio < 5:
        print(f"only {n_audio} rendered tracks per arm; rendering may still be running")
    rows = {}
    for arm, (texts, Et, files, idx, plans) in data.items():
        files, idx, plans = files[:n_audio], idx[:n_audio], plans[:n_audio]
        keep = [i for i in idx if i < len(texts)]
        T = [texts[i] for i in keep]
        ET = Et[np.array(keep)] if keep else Et[:0]
        wavs = wavify(files)
        print(f"[{arm}] embedding {len(wavs)} tracks ...", flush=True)
        Ec = clap_embed(wavs)
        Em = mert_embed(wavs)
        np.save(HERE / "real" / f"lyria_{arm}" / "clap_emb.npy", Ec)
        np.save(HERE / "real" / f"lyria_{arm}" / "mert_emb.npy", Em)
        r = {
            "label": LABEL[arm], "n_prompts": len(T), "n_tracks": len(wavs),
            "prompt_mean_chars": round(float(np.mean([len(t) for t in T])), 1),
            "prompt_exact_dup": exact_dup_rate(T)["exact_dup_rate"],
            "prompt_distinct_2": round(distinct_n(T, 2), 4),
            "prompt_self_rep_4": round(self_repetition(T), 4),
            "prompt_ngram_vendi": round(ngram_vendi(T, 2), 1),
            "prompt_vendi_centered": round(embed_vendi_centered(ET), 2),
            "prompt_median_nn": round(nn_stats(ET)["median_nn_cos_dist"], 4),
            "clap_vendi_centered": round(embed_vendi_centered(Ec), 2),
            "clap_median_nn": round(nn_stats(Ec)["median_nn_cos_dist"], 4),
            "mert_vendi_centered": round(embed_vendi_centered(Em), 2),
            "mert_median_nn": round(nn_stats(Em)["median_nn_cos_dist"], 4),
        }
        r.update({f"form_{k}": v for k, v in plan_stats(plans).items()})
        # does prompt geometry predict audio geometry?
        if len(ET) == len(Ec) and len(ET) > 2:
            Tn = ET / np.linalg.norm(ET, axis=1, keepdims=True)
            iu = np.triu_indices(len(Tn), 1)
            for tag, E in (("clap", Ec), ("mert", Em)):
                An = E / np.linalg.norm(E, axis=1, keepdims=True)
                r[f"prompt_{tag}_sim_corr"] = round(
                    float(np.corrcoef((Tn @ Tn.T)[iu], (An @ An.T)[iu])[0, 1]), 4)
        rows[arm] = r
        print(f"  {arm}: promptV={r['prompt_vendi_centered']} "
              f"CLAP={r['clap_vendi_centered']} MERT={r['mert_vendi_centered']} "
              f"forms={r.get('form_distinct_form_shapes')}", flush=True)
    with open(HERE / "figures" / "audio_compare.json", "w") as f:
        json.dump({"n_tracks": n_audio, "arms": rows}, f, indent=2)
    print("wrote figures/audio_compare.json")
    if len(rows) == 2:
        for k in ("prompt_vendi_centered", "clap_vendi_centered", "mert_vendi_centered"):
            a, b = rows["naive"][k], rows["ihd"][k]
            print(f"  {k}: naive {a} -> IHD {b}  ({b / a:.2f}x)")


if __name__ == "__main__":
    main()
