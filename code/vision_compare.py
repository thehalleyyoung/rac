"""
Cross-arm diversity in the space the product actually occupies.

Every method in this paper is a text-side method: it reads embeddings of
instructions and selects among them. Section 7.6 showed that text-embedding
similarity predicts rendered-image similarity at only r = 0.155, which means
none of these methods can see whether they are succeeding at the thing a user
of an image-generation prompt set cares about.

So we render the same number of prompts from every collapse-minimizing arm at
the same quality tier and measure diversity three ways on the identical items:

    literal (n-gram)  ->  latent text (nomic)  ->  rendered image (CLIP)

and ask whether the ranking survives the trip. A method that wins on text and
loses on pixels is optimizing the proxy, not the product.

Quality is held fixed across arms (set by VISION_TAG, default the `high`
tier) because image quality changes the CLIP geometry -- a low-quality render
is blurrier and therefore lands closer to other low-quality renders, which
would inflate or deflate diversity for reasons having nothing to do with the
prompts. Each tier lives in its own directory and they are never mixed. Having
two full tiers (medium and high) also lets us check that the ranking is a
property of the methods rather than of the renderer settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from metrics import (distinct_n, embed_vendi_centered, exact_dup_rate,
                     ngram_vendi, nn_stats, self_repetition)

HERE = Path(__file__).resolve().parent
ARMS = ["naive", "high_temp", "self_instruct", "evol_instruct", "persona",
        "ihd", "vision"]
LABEL = {
    "naive": "naive repeated prompting",
    "high_temp": "high temperature (T=1.6)",
    "self_instruct": "Self-Instruct (ROUGE filter)",
    "evol_instruct": "Evol-Instruct (WizardLM)",
    "persona": "persona conditioning",
    "ihd": "IHD (ours)",
    "vision": "IHD + vision steering (ours)",
}
TAG = os.environ.get("VISION_TAG", "high")   # quality tier to analyse


def load_arm(arm: str):
    d = HERE / "real" / f"dalle_{arm}"
    emb_p = d / f"clip_image_emb_{TAG}.npy"
    idx_p = d / f"clip_index_{TAG}.json"
    if not emb_p.exists():
        return None
    Ei = np.load(emb_p)
    idx = json.load(open(idx_p))
    texts = [json.loads(l)["text"] for l in (d / "corpus.jsonl").read_text().splitlines()
             if l.strip()]
    Et = np.load(d / "embeddings.npy")
    keep = [i for i in idx if i < len(texts) and i < len(Et)]
    sel = [idx.index(i) for i in keep]
    return ([texts[i] for i in keep], Et[np.array(keep)], Ei[np.array(sel)])


def main():
    rows = {}
    avail = {}
    for arm in ARMS:
        got = load_arm(arm)
        if got and len(got[0]) >= 20:
            avail[arm] = got
    if not avail:
        print("no CLIP embeddings yet; run render_images.py embed first")
        return
    n = min(len(t) for t, _, _ in avail.values())
    print(f"comparing {len(avail)} arms at matched n={n} ({TAG} quality)\n")
    for arm, (texts, Et, Ei) in avail.items():
        T, ET, EI = texts[:n], Et[:n], Ei[:n]
        r = {
            "label": LABEL[arm], "n": n,
            "exact_dup_rate": exact_dup_rate(T)["exact_dup_rate"],
            "distinct_2": round(distinct_n(T, 2), 4),
            "self_repetition_4": round(self_repetition(T), 4),
            "ngram_vendi_2": round(ngram_vendi(T, 2), 1),
            "text_vendi_centered": round(embed_vendi_centered(ET), 2),
            "text_median_nn": round(nn_stats(ET)["median_nn_cos_dist"], 4),
            "image_vendi_centered": round(embed_vendi_centered(EI), 2),
            "image_median_nn": round(nn_stats(EI)["median_nn_cos_dist"], 4),
        }
        # per-arm text/image agreement: does this arm's text geometry predict
        # its own image geometry?
        Tn = ET / np.linalg.norm(ET, axis=1, keepdims=True)
        In = EI / np.linalg.norm(EI, axis=1, keepdims=True)
        iu = np.triu_indices(n, 1)
        r["text_image_sim_corr"] = round(
            float(np.corrcoef((Tn @ Tn.T)[iu], (In @ In.T)[iu])[0, 1]), 4)
        rows[arm] = r
        print(f"{arm:15s} d2={r['distinct_2']:.4f} textV={r['text_vendi_centered']:7.2f} "
              f"imgV={r['image_vendi_centered']:7.2f} "
              f"imgNN={r['image_median_nn']:.4f} r={r['text_image_sim_corr']:+.3f}")

    # does the text ranking survive the trip to pixels?
    arms = list(rows)
    tv = np.array([rows[a]["text_vendi_centered"] for a in arms])
    iv = np.array([rows[a]["image_vendi_centered"] for a in arms])
    from scipy.stats import spearmanr
    rho = float(spearmanr(tv, iv).statistic)
    best_text = arms[int(np.argmax(tv))]
    best_img = arms[int(np.argmax(iv))]
    print(f"\ntext-vs-image ranking agreement across arms: Spearman rho = {rho:.3f}")
    print(f"best on text: {best_text}   best on image: {best_img}")
    out = {"matched_n": n, "quality": TAG, "arms": rows,
           "rank_agreement_spearman": rho,
           "best_text_arm": best_text, "best_image_arm": best_img}
    with open(HERE / "figures" / "vision_compare.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote figures/vision_compare.json")


if __name__ == "__main__":
    main()
