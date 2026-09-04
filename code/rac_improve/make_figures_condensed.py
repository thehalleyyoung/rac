"""Two figures for the condensed paper.

fig22_out_of_objective.png
    The three out-of-objective separations in one frame. Left: every image
    corpus on pixel statistics (real values, no overlap between groups).
    Right: the ratio summary for all three artifact domains and both
    statistics, with the count of favourable resamples beside each row.

fig23_completeness.png
    The completeness rule. Left: partition-coverage interventions, effect on
    normalized entropy over the target partition. Right: within-level device
    interventions, effect on the dominant device's share. Both split by
    whether the axis enumerates the whole space the measure is defined over
    or only a subset of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
FIG = RESEARCH / "figures"

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})

INK = "#222222"
OURS = "#1f77b4"
BASE = "#d62728"
GREY = "#999999"
GOOD = "#2a7f4f"
BAD = "#b0522a"

BASELINES = {"dalle_evol_instruct", "dalle_high_temp", "dalle_naive",
             "dalle_persona", "dalle_self_instruct"}
PRETTY = {
    "dalle_evol_instruct": "Evol-Instruct", "dalle_high_temp": "high temperature",
    "dalle_naive": "naive", "dalle_persona": "Persona-Hub",
    "dalle_self_instruct": "Self-Instruct",
}


def out_of_objective():
    d = json.load(open(HERE / "optical_consistent.json"))
    ours = {k: v for k, v in d.items() if k not in BASELINES}
    base = {k: v for k, v in d.items() if k in BASELINES}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 3.9),
                                 gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- left: the image corpora, real values ------------------------------
    a = np.array([v["spread"] for v in ours.values()])
    b = np.array([v["spread"] for v in base.values()])
    rng = np.random.default_rng(7)
    ax.scatter(a, rng.normal(1.0, 0.05, len(a)), s=48, color=OURS,
               zorder=3, label=f"RAC arms ($n$ = {len(a)})")
    ax.scatter(b, rng.normal(0.0, 0.05, len(b)), s=48, color=BASE,
               marker="s", zorder=3, label=f"published baselines ($n$ = {len(b)})")
    ax.axvspan(b.max(), a.min(), color=GREY, alpha=0.16, zorder=0)
    ax.annotate("no corpus of ours\noverlaps any baseline",
                ((b.max() + a.min()) / 2, 1.53), ha="center", va="top",
                fontsize=8, color=INK)
    ax.annotate(f"group mean {b.mean():.2f}", (b.mean(), -0.42), ha="center",
                fontsize=8, color=BASE)
    ax.annotate(f"group mean {a.mean():.2f}", (a.mean(), 0.55), ha="center",
                fontsize=8, color=OURS)
    ax.plot([b.mean(), b.mean()], [-0.30, 0.28], color=BASE, ls="--", lw=1.1)
    ax.plot([a.mean(), a.mean()], [0.70, 1.28], color=OURS, ls="--", lw=1.1)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["baselines", "ours"])
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlim(2.2, 5.05)
    ax.set_xlabel("optical spread  (mean pairwise distance, 9 pixel statistics)")
    ax.set_title("Images: a channel CLIP never sees", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)

    # ---- right: ratio summary across the three domains ---------------------
    rows = [
        ("images \u00b7 optical spread", 1.534, 1.38, 1.68, "55 / 55 pairs"),
        ("images \u00b7 local crowding", 1.369, 1.25, 1.48, "54 / 55 pairs"),
        ("poems \u00b7 prosodic spread", 2.25, None, None, "200 / 200 draws"),
        ("poems \u00b7 local crowding", 2.18, None, None, "200 / 200 draws"),
        ("items \u00b7 structural spread", 1.61, None, None, "50 / 50 draws"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, r, lo, hi, note) in zip(ys, rows):
        if lo is not None:
            bx.plot([lo, hi], [y, y], color=OURS, lw=2.2, solid_capstyle="round")
        bx.scatter([r], [y], s=54, color=OURS, zorder=3)
        bx.annotate(f"{r:.2f}\u00d7", (r, y + 0.17), ha="center", fontsize=8.5,
                    color=INK)
        bx.annotate(note, (3.02, y), ha="right", va="center", fontsize=7.5,
                    color=GREY)
    bx.axvline(1.0, color=BASE, lw=1.2)
    bx.annotate("parity", (1.0, -0.66), ha="center", fontsize=7.5, color=BASE)
    bx.set_yticks(ys); bx.set_yticklabels([r[0] for r in rows], fontsize=8)
    bx.set_xlim(0.85, 3.05)
    bx.set_xticks([1.0, 1.5, 2.0, 2.5])
    bx.set_ylim(-0.8, len(rows) - 0.35)
    bx.set_xlabel("RAC \u00f7 strongest comparison, at matched generator budget")
    bx.set_title("Three domains, three channels the loop never optimizes",
                 fontsize=9.5)
    bx.grid(axis="y", visible=False)

    fig.tight_layout()
    out = FIG / "fig22_out_of_objective.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def completeness():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 3.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---- left: partition coverage, delta normalized entropy ---------------
    part = [  # label, delta, lo, hi, complete?
        ("instructions · names 5 of 10 tasks", -0.014, -0.080, 0.058, False),
        ("item bank · names 5 of 10 areas", 0.117, 0.056, 0.183, False),
        ("instructions · names all 10 tasks", 0.146, 0.069, 0.225, True),
        ("item bank · names all 10 areas", 0.219, 0.138, 0.297, True),
    ]
    ys = np.arange(len(part))
    for y, (lab, dd, lo, hi, comp) in zip(ys, part):
        c = GOOD if comp else BAD
        ax.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round")
        ax.scatter([dd], [y], s=52, color=c, zorder=3)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([p[0] for p in part], fontsize=8)
    ax.set_xlabel("Δ normalized entropy over the target partition (n = 100/arm)")
    ax.set_title("Between categories: does the corpus span its partition?",
                 fontsize=9.5)
    ax.set_ylim(-0.7, len(part) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.annotate("names the whole space", (0.30, 3.34), fontsize=8, color=GOOD,
                ha="center")
    ax.annotate("names a subset", (0.30, 1.34), fontsize=8, color=BAD,
                ha="center")

    # ---- right: within-level device concentration -------------------------
    dev = [  # label, delta in dominant-device share, lo, hi, complete?
        ("poems · lexical field (nouns only)", 0.076, -0.076, 0.204, False),
        ("poems · adaptive ban on 12 moves", -0.012, -0.136, 0.104, False),
        ("poems · per-stance refinement", -0.068, -0.188, 0.093, False),
        ("poems · rhetorical vehicles", -0.088, -0.213, 0.053, False),
        ("item bank · all 8 device forms", -0.507, -0.622, -0.389, True),
    ]
    ys = np.arange(len(dev))
    for y, (lab, dd, lo, hi, comp) in zip(ys, dev):
        c = GOOD if comp else BAD
        bx.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round")
        bx.scatter([dd], [y], s=52, color=c, zorder=3)
    bx.axvline(0, color=INK, lw=1.0)
    bx.set_yticks(ys)
    bx.set_yticklabels([d[0] for d in dev], fontsize=8)
    bx.set_xlabel("Δ share of the level's single most frequent device  (lower is better)")
    bx.set_title("Within a level: is the level realized the same way each time?",
                 fontsize=9.5)
    bx.set_ylim(-0.7, len(dev) - 0.3)
    bx.grid(axis="y", visible=False)
    bx.annotate("names the whole space", (-0.36, 4.32), fontsize=8, color=GOOD,
                ha="center")
    bx.annotate("names a subset", (-0.36, 3.30), fontsize=8, color=BAD,
                ha="center")

    fig.tight_layout()
    out = FIG / "fig23_completeness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    out_of_objective()
    completeness()
