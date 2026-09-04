"""Figure: what expansion buys, in the two domains, at matched budget.

Left: the three image arms on the metrics they are scored by. Right: the poem
corpus's form spread, baseline against expanded, on the prosodic features the
§6.14 audit showed the baseline holding nearly constant.

The point of putting them side by side is the asymmetry: the same mechanism,
the same prompt structure, a large effect in one domain and none in the other.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
FIG = RESEARCH / "figures"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
GREY, COOL, WARM, INK = "#999999", "#1f77b4", "#ff7f0e", "#222222"

ARMS = [("baseline", "dalle_steer2_maxmin"),
        ("+ expansion", "dalle_steer3_maxmin"),
        ("+ expansion\n+ manipulation", "dalle_steer4_maxmin")]
FEATS = ["n_lines", "syl_per_line_sd", "type_token", "rhyme_density",
         "line_chars_mean", "syl_per_line_mean"]


def main(poem_n: int | None = None):
    from text_realization import prosodic_features

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---- images: three arms, normalized to the baseline -------------------
    S = {lab: json.load(open(RESEARCH / "real" / t / "run_summary.json"))
         for lab, t in ARMS}
    mets = [("clip_vendi_centered", "Vendi"),
            ("clip_median_nn_dist", "median NN"),
            ("clip_min_nn_dist", "min NN\n(the objective)")]
    x = np.arange(len(mets))
    base = [S["baseline"][k] for k, _ in mets]
    for i, (lab, _) in enumerate(ARMS):
        vals = [S[lab][k] / b for (k, _), b in zip(mets, base)]
        axes[0].bar(x + (i - 1) * 0.27, vals, 0.25,
                    color=[GREY, COOL, WARM][i], label=lab.replace("\n", " "))
    axes[0].axhline(1.0, color=INK, lw=1.1, ls="--")
    axes[0].set_xticks(x); axes[0].set_xticklabels([n for _, n in mets])
    axes[0].set_ylabel("relative to baseline")
    axes[0].set_ylim(0.8, 1.15)
    axes[0].set_title("Images: expansion changes nothing measurable\n"
                      "(n = 60 renders per arm, one seed)", fontsize=9.5)
    axes[0].legend(fontsize=7.5, loc="upper left")

    # ---- poems: form spread, baseline vs expanded -------------------------
    def cvs(tag, n):
        recs = [json.loads(l) for l in
                (RESEARCH / "live" / tag / "corpus.jsonl").read_text().splitlines()
                if l.strip()][:n]
        F = [prosodic_features(r["text"]) for r in recs]
        out = {}
        for k in FEATS:
            v = np.array([f.get(k, 0.0) for f in F])
            out[k] = float(v.std() / max(abs(v.mean()), 1e-9))
        return out, len(recs)

    gap_all = [l for l in (RESEARCH / "live" / "poems_gap" / "corpus.jsonl"
                           ).read_text().splitlines() if l.strip()]
    n = poem_n or len(gap_all)
    a, na = cvs("poems", n)
    b, nb = cvs("poems_gap", n)
    y = np.arange(len(FEATS))
    names = [f.replace("_", " ") for f in FEATS]
    axes[1].barh(y - 0.19, [a[f] for f in FEATS], 0.36, color=GREY, label="baseline")
    axes[1].barh(y + 0.19, [b[f] for f in FEATS], 0.36, color=WARM, label="+ expansion")
    axes[1].set_yticks(y); axes[1].set_yticklabels(names)
    axes[1].set_xlabel("coefficient of variation across the corpus")
    axes[1].set_title(f"Poems: every form dimension opens up\n"
                      f"(matched n = {min(na, nb)})", fontsize=9.5)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "fig20_expansion.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig20_expansion.png (poem matched n={min(na, nb)})")
    for f in FEATS:
        print(f"  {f:<20} baseline {a[f]:.3f}   expanded {b[f]:.3f}   "
              f"ratio {b[f]/max(a[f],1e-9):.2f}x")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)


# ---------------------------------------------------------------- fig 21
def fig21_optical_spread():
    """Optical spread of every image corpus, measured outside CLIP entirely."""
    import json
    vals = json.load(open(HERE / "optical_spread.json"))
    BASE = {"dalle_naive", "dalle_high_temp", "dalle_self_instruct",
            "dalle_evol_instruct", "dalle_persona"}
    NICE = {"dalle_naive": "naive", "dalle_high_temp": "high temperature",
            "dalle_self_instruct": "Self-Instruct", "dalle_evol_instruct": "Evol-Instruct",
            "dalle_persona": "Persona-Hub", "dalle_ihd": "RAC",
            "dalle_vision": "RAC + vision", "dalle_steer_maxmin": "RAC max-min",
            "dalle_steer_coverage": "RAC coverage", "dalle_steer2_maxmin": "RAC max-min (v2)",
            "dalle_steer2_coverage": "RAC coverage (v2)",
            "dalle_steer3_maxmin": "RAC + expansion",
            "dalle_steer4_maxmin": "RAC + expansion, manip."}
    items = sorted(vals.items(), key=lambda kv: kv[1])
    names = [NICE.get(k, k) for k, _ in items]
    v = [x for _, x in items]
    cols = ["#999999" if k in BASE else "#1f77b4" for k, _ in items]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.barh(names, v, color=cols, height=0.66)
    b = [x for k, x in vals.items() if k in BASE]
    o = [x for k, x in vals.items() if k not in BASE]
    ax.axvline(np.mean(b), color="#999999", lw=1.2, ls="--")
    ax.axvline(np.mean(o), color="#1f77b4", lw=1.2, ls="--")
    ax.set_xlabel("mean pairwise distance in standardized pixel-statistic space")
    ax.set_title("Optical spread, measured outside CLIP entirely\n"
                 f"every RAC corpus exceeds every baseline ({np.mean(o)/np.mean(b):.2f}x on the means)",
                 fontsize=9.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="ours"),
                       Patch(color="#999999", label="published baselines")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig21_optical_spread.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig21_optical_spread.png")
