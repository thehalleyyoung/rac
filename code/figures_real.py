"""Figures for the real gpt-5.6-luna runs: literal, latent, and vision diversity."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})

STYLE = {
    "dalle_naive": ("#888888", "DALL-E instr., naive"),
    "dalle_ihd": ("#1f77b4", "DALL-E instr., RAC"),
    "psychometric_naive": ("#c49a3a", "psychometric, naive"),
    "psychometric_ihd": ("#2ca02c", "psychometric, RAC"),
}


def load():
    with open(FIG / "real_curves.json") as f:
        return json.load(f)


def fig_real_curves():
    C = load()
    panels = [
        ("distinct_2", "distinct-2 (literal)", "(a) literal: vocabulary exhausting", True),
        ("self_repetition_4", "fraction of 4-grams already seen",
         "(b) literal: self-repetition", True),
        ("ngram_vendi_2", "n-gram Vendi (effective wordings)",
         "(c) literal: effective distinct wordings", True),
        ("embed_vendi_centered", "mean-centered embedding Vendi",
         "(d) latent: effective distinct meanings", True),
        ("median_nn_cos_dist", "median nearest-neighbour cosine distance",
         "(e) latent: room between neighbours", True),
        ("embed_vendi", "uncentered embedding Vendi",
         "(f) latent: uncentered (kernel-compressed)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6))
    for ax, (key, ylab, title, logx) in zip(axes.ravel(), panels):
        for run, (c, lab) in STYLE.items():
            if run not in C:
                continue
            rows = C[run]
            xs = [r["n"] for r in rows if key in r]
            ys = [r[key] for r in rows if key in r]
            if not xs:
                continue
            ax.plot(xs, ys, "o-", ms=2.8, lw=1.3, color=c, label=lab)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel("corpus size $n$")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0, 0].legend(fontsize=6.8, loc="lower left")
    fig.suptitle("Real gpt-5.6-luna corpora, $n = 5 \\to 10{,}000$: "
                 "literal and latent diversity move differently", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_real_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_literal_vs_latent():
    """The decoupling, stated directly: normalize each series to its n=5 value."""
    C = load()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, (dom, title) in zip(axes, [("dalle", "DALL-E instructions"),
                                        ("psychometric", "psychometric items")]):
        for arm, ls in [("naive", "-"), ("ihd", "--")]:
            run = f"{dom}_{arm}"
            if run not in C:
                continue
            rows = C[run]
            xs = [r["n"] for r in rows]
            lit = np.array([r["distinct_2"] for r in rows])
            lat = np.array([r["embed_vendi_centered"] for r in rows])
            ax.plot(xs, lit / lit[0], ls, color="#d62728", lw=1.4,
                    label=f"{arm}: literal (distinct-2)")
            ax.plot(xs, lat / lat[0], ls, color="#1f77b4", lw=1.4,
                    label=f"{arm}: latent (centered Vendi)")
        ax.set_xscale("log")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("corpus size $n$")
        ax.set_ylabel("value relative to $n=5$")
        ax.set_title(title)
        ax.legend(fontsize=6.5)
    fig.suptitle("Literal diversity falls while latent diversity rises", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_literal_vs_latent.png", bbox_inches="tight")
    plt.close(fig)


def fig_vision():
    """Text-embedding vs CLIP-image-embedding diversity on the rendered set."""
    from metrics import embed_vendi, embed_vendi_centered, nn_stats, load_run
    run = "dalle_naive"
    d = HERE / "real" / run
    # prefer the high-quality render tier
    emb_p = d / "clip_image_emb_high.npy"
    idx_p = d / "clip_index_high.json"
    if not emb_p.exists():
        emb_p, idx_p = d / "clip_image_emb.npy", d / "clip_index.json"
    if not emb_p.exists():
        print("no CLIP embeddings yet; skipping fig12")
        return
    Ei = np.load(emb_p)
    idx = json.load(open(idx_p))
    texts, Et = load_run(*run.rsplit("_", 1))
    Et = Et[np.array(idx)]

    ns = [n for n in [5, 10, 20, 40, 75, 150, 200] if n <= len(Ei)]
    rows = []
    for n in ns:
        rows.append({
            "n": n,
            "text_vendi_c": embed_vendi_centered(Et[:n]),
            "image_vendi_c": embed_vendi_centered(Ei[:n]),
            "text_nn": nn_stats(Et[:n])["median_nn_cos_dist"],
            "image_nn": nn_stats(Ei[:n])["median_nn_cos_dist"],
        })
    with open(FIG / "vision_curves.json", "w") as f:
        json.dump(rows, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    ax = axes[0]
    ax.plot([r["n"] for r in rows], [r["text_vendi_c"] for r in rows], "o-",
            color="#1f77b4", lw=1.4, ms=3.5, label="text embedding (nomic)")
    ax.plot([r["n"] for r in rows], [r["image_vendi_c"] for r in rows], "s-",
            color="#d62728", lw=1.4, ms=3.5, label="image embedding (CLIP)")
    ax.set_xscale("log")
    ax.set_xlabel("number of rendered instructions")
    ax.set_ylabel("mean-centered Vendi")
    ax.set_title("(a) effective distinct items")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot([r["n"] for r in rows], [r["text_nn"] for r in rows], "o-",
            color="#1f77b4", lw=1.4, ms=3.5)
    ax.plot([r["n"] for r in rows], [r["image_nn"] for r in rows], "s-",
            color="#d62728", lw=1.4, ms=3.5)
    ax.set_xscale("log")
    ax.set_xlabel("number of rendered instructions")
    ax.set_ylabel("median nearest-neighbour cosine distance")
    ax.set_title("(b) room between neighbours")

    ax = axes[2]
    n = len(Ei)
    Tn = Et[:n] / np.linalg.norm(Et[:n], axis=1, keepdims=True)
    In = Ei / np.linalg.norm(Ei, axis=1, keepdims=True)
    St = (Tn @ Tn.T)[np.triu_indices(n, 1)]
    Si = (In @ In.T)[np.triu_indices(n, 1)]
    ax.scatter(St, Si, s=4, alpha=0.15, color="#444444")
    r = float(np.corrcoef(St, Si)[0, 1])
    ax.set_xlabel("pairwise cosine sim, TEXT embedding")
    ax.set_ylabel("pairwise cosine sim, IMAGE embedding")
    ax.set_title(f"(c) do they agree?  Pearson $r$ = {r:.3f}")
    fig.suptitle(f"Text vs vision diversity on {n} rendered post-modern artworks "
                 f"(gpt-image-1-mini + CLIP)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig12_vision.png", bbox_inches="tight")
    plt.close(fig)
    print(f"vision: text/image pairwise-similarity correlation r = {r:.3f}")
    with open(FIG / "vision_summary.json", "w") as f:
        json.dump({"n_images": n, "pairwise_sim_correlation": r,
                   "text_vendi_centered": rows[-1]["text_vendi_c"],
                   "image_vendi_centered": rows[-1]["image_vendi_c"],
                   "text_median_nn": rows[-1]["text_nn"],
                   "image_median_nn": rows[-1]["image_nn"]}, f, indent=2)


def fig_contact_sheet(k: int = 32):
    """Figure 13 is the two-panel naive-vs-steered sheet built by
    make_contact_sheet.py; delegate so a rerun here cannot overwrite it with a
    single-policy sheet."""
    from make_contact_sheet import sheet
    sheet()




ARM_LABEL = {"naive": "naive", "high_temp": "high temp.",
             "evol_instruct": "Evol-Instruct", "self_instruct": "Self-Instruct",
             "persona": "persona", "ihd": "RAC", "vision": "RAC + vision"}


def fig_arms():
    """Cross-arm comparison at matched n, both domains."""
    with open(FIG / "arm_comparison.json") as f:
        A = json.load(f)
    order = ["naive", "high_temp", "evol_instruct", "self_instruct", "persona",
             "ihd", "vision"]
    cols = {"naive": "#999999", "high_temp": "#d62728", "evol_instruct": "#8c564b",
            "self_instruct": "#ff7f0e", "persona": "#9467bd",
            "ihd": "#1f77b4", "vision": "#2ca02c"}
    metrics = [("distinct_2", "distinct-2 (literal)"),
               ("embed_vendi_centered", "centered embedding Vendi (latent)"),
               ("median_nn_cos_dist", "median NN cosine distance"),
               ("self_repetition_4", "4-gram self-repetition (lower better)")]
    doms = [d for d in ["dalle", "psychometric"] if d in A]
    fig, axes = plt.subplots(len(doms), 4, figsize=(14, 3.5 * len(doms)),
                             squeeze=False)
    for r, dom in enumerate(doms):
        arms = A[dom]["arms"]
        present = [a for a in order if a in arms]
        for c, (key, lab) in enumerate(metrics):
            ax = axes[r][c]
            vals = [arms[a][key] for a in present]
            ax.barh(np.arange(len(present)), vals,
                    color=[cols[a] for a in present])
            ax.set_yticks(np.arange(len(present)))
            ax.set_yticklabels([ARM_LABEL[a] for a in present], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel(lab, fontsize=7.5)
            if c == 0:
                ax.set_ylabel(f"{dom}\n(n={A[dom]['matched_n']})", fontsize=9)
    fig.suptitle("Competitive comparison on real gpt-5.6-luna corpora, matched $n$",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "fig14_arms.png", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    fig_real_curves()
    fig_literal_vs_latent()
    fig_vision()
    fig_contact_sheet()
    fig_arms()
    print("real-data figures written")
