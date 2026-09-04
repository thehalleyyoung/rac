"""Scale-free head-to-head against released instruction corpora."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
d = json.load(open(FIG / "summary_scalefree_h2h_cleanref.json"))
rows = d["rows"]
ks = d["ks"]

LABEL = {
    "fair: ours-v4-covretr":          ("RAC-coverage, retrieval-aimed (2.4k)", True),
    "Alpaca (Self-Instruct, 52k)":    ("Alpaca (52k)", False),
    "fair: ours-selective":           ("RAC-coverage, selective 1-of-8 (304)", True),
    "PersonaHub (50k)":               ("PersonaHub (50k)", False),
    "WizardLM Evol-Instruct (143k)":  ("WizardLM Evol-Instruct (143k)", False),
    "fair: self-instruct":            ("Self-Instruct, reimpl. (1.8k)", False),
    "fair: ours-keepall":             ("RAC, conditioning keep-all (1.3k)", True),
    "fair: random-seed":              ("few-shot from seeds (2.4k)", False),
    "fair: ours-v3-orthcond":         ("RAC, orthogonalized conditioning (1.2k)", True),
    "ours: axis-conditioned":         ("axis-conditioned, unseeded (450)", True),
    "ours: naive":                    ("naive prompting (450)", False),
    "ours: high-temp":                ("high temperature (450)", False),
}
def short(name):
    return LABEL[name][0]
def is_ours(name):
    return LABEL[name][1]

order = sorted(rows, key=lambda k: -rows[k]["coverage_auc_over_k"])
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

# (a) coverage against k for the leading corpora
for name in order[:5]:
    r = rows[name]
    ys = [r[f"k{k}"]["coverage"] for k in ks]
    axes[0].plot(ks, ys, marker="o", lw=2 if is_ours(name) else 1.4,
                 color="#1f77b4" if is_ours(name) else "#999999",
                 label=short(name).split(" (")[0], zorder=3 if is_ours(name) else 2)
axes[0].set_xscale("log"); axes[0].set_xticks(ks)
axes[0].set_xticklabels([str(k) for k in ks]); axes[0].minorticks_off()
axes[0].set_xlabel("k (reference-side k-NN radius)")
axes[0].set_ylabel("coverage of held-out human reference")
axes[0].set_title("(a) Coverage at every radius")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# (b) AUC over k, all corpora
names = order
vals = [rows[n]["coverage_auc_over_k"] for n in names]
cols = ["#1f77b4" if is_ours(n) else "#999999" for n in names]
axes[1].barh(range(len(names))[::-1], vals, color=cols)
axes[1].set_yticks(range(len(names))[::-1])
axes[1].set_yticklabels([short(n) for n in names], fontsize=7)
axes[1].set_xlabel("coverage AUC over k")
axes[1].set_title("(b) All twelve corpora, matched n = 450")
axes[1].grid(alpha=0.3, axis="x")

# (c) coverage against precision -- diversity without leaving the reference
for name in names:
    r = rows[name]
    axes[2].scatter(r["k5"]["precision"], r["coverage_auc_over_k"], s=70,
                    color="#1f77b4" if is_ours(name) else "#999999", zorder=3)
for name, dx, dy in [("fair: ours-v4-covretr", -8, -14), ("Alpaca (Self-Instruct, 52k)", 7, -3),
                     ("fair: ours-selective", -10, 8), ("PersonaHub (50k)", 7, -3)]:
    r = rows[name]
    axes[2].annotate(short(name).split(" (")[0], (r["k5"]["precision"], r["coverage_auc_over_k"]),
                     textcoords="offset points", xytext=(dx, dy), fontsize=7,
                     ha="right" if dx < 0 else "left")
axes[2].set_xlabel("precision at k = 5 (items inside the reference manifold)")
axes[2].set_ylabel("coverage AUC over k")
axes[2].set_title("(c) Coverage bought without leaving the manifold")
axes[2].grid(alpha=0.3)

fig.suptitle("Scale-free coverage of a held-out human-written reference "
             "(blue = ours, grey = released corpora and baselines)", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "fig_h2h_scalefree.png", dpi=150, bbox_inches="tight")
print("wrote figures/fig_h2h_scalefree.png")
