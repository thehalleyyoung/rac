"""Render all paper figures from the saved simulation + pilot outputs."""
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
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})

POLICY_LABELS = {
    "naive": "naive (i.i.d.)",
    "novelty_only": "novelty-only (max-min)",
    "anchor_repulsion": "anchor+repulsion (ungated)",
    "gated_anchor_repulsion": "anchor+repulsion (gated)",
    "spectral": "spectral (Q + orth + gap)",
    "spectral_ledger": "spectral + ledger",
}
COLORS = {
    "naive": "#888888", "novelty_only": "#d62728", "anchor_repulsion": "#ff7f0e",
    "gated_anchor_repulsion": "#9467bd", "spectral": "#1f77b4", "spectral_ledger": "#2ca02c",
}


def smooth(a, w=200):
    if len(a) < w:
        return a
    c = np.cumsum(np.insert(a.astype(float), 0, 0.0))
    return (c[w:] - c[:-w]) / w


def fig1_poetry():
    z = np.load(FIG / "results.npz")
    policies = list(POLICY_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))

    ax = axes[0, 0]
    for p in policies:
        y = smooth(z[f"{p}__nn_dist"])
        ax.plot(np.arange(len(y)) + 100, y, label=POLICY_LABELS[p], color=COLORS[p], lw=1.2)
    ax.set_xlabel("corpus size $n$")
    ax.set_ylabel("min-gap of accepted item (200-item mean)")
    ax.set_title("(a) local novelty as the corpus grows")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[0, 1]
    for p in policies:
        ax.plot(z[f"{p}__vendi_x"], z[f"{p}__vendi_y"], color=COLORS[p], lw=1.2)
    ax.set_xlabel("corpus size $n$")
    ax.set_ylabel("Vendi Score (linear kernel)")
    ax.set_title("(b) effective number of distinct styles")

    ax = axes[1, 0]
    for p in policies:
        y = smooth(z[f"{p}__junk_hit"].astype(float), 500)
        ax.plot(np.arange(len(y)) + 250, y, color=COLORS[p], lw=1.2)
    ax.set_xlabel("corpus size $n$")
    ax.set_ylabel("junk fraction (500-item window)")
    ax.set_title("(c) the junk tax of unguarded novelty")

    ax = axes[1, 1]
    for p in policies:
        y = smooth(z[f"{p}__quality"], 500)
        ax.plot(np.arange(len(y)) + 250, y, color=COLORS[p], lw=1.2)
    ax.set_xlabel("corpus size $n$")
    ax.set_ylabel("true latent quality (500-item mean)")
    ax.set_title("(d) quality under each policy")

    fig.suptitle("Poetry world, $n=10{,}000$: six acquisition policies", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_poetry_policies.png", bbox_inches="tight")
    plt.close(fig)


def fig2_tradeoff():
    with open(FIG / "summary.json") as f:
        s = json.load(f)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for p, r in s.items():
        ax.scatter(r["final_nn_dist_last500_mean"], r["mean_true_quality"],
                   s=120 * np.sqrt(r["final_vendi_score"]) / 4,
                   color=COLORS[p], zorder=3)
        junk = r["final_junk_fraction_last2000"]
        label = POLICY_LABELS[p] + (f"  (junk {junk:.0%})" if junk > 0.005 else "")
        ax.annotate(label, (r["final_nn_dist_last500_mean"], r["mean_true_quality"]),
                    textcoords="offset points", xytext=(8, -3), fontsize=7)
    ax.set_xlabel("late-corpus min-gap (novelty maintained at $n=10{,}000$)")
    ax.set_ylabel("mean true quality over the whole corpus")
    ax.set_title("Quality vs maintained novelty (marker area = final Vendi Score)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig3_exam():
    z = np.load(FIG / "exam_results.npz")
    with open(FIG / "exam_summary.json") as f:
        s = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    ax = axes[0]
    for name, color in [("first_legal", "#d62728"), ("tightest_legal", "#ff7f0e"),
                        ("first_legal_gated", "#9467bd"), ("tightest_legal_gated", "#1f77b4")]:
        att = z[f"{name}__attempts"]
        acc = att[att[:, 2] == 1]
        w = 50
        if len(acc) > w:
            xs = acc[w - 1:, 0]
            ys = np.convolve(acc[:, 1], np.ones(w) / w, mode="valid")
            ax.plot(xs, ys, color=color, lw=1.2,
                    label=name.replace("_", " "))
    ax.set_yscale("log")
    ax.set_xlabel("bank size")
    ax.set_ylabel("generator calls per accepted item (50-item mean)")
    ax.set_title("(a) cost of the next legal item")
    ax.legend(fontsize=7)

    ax = axes[1]
    for name, color in [("first_legal", "#d62728"), ("farthest_legal", "#ff7f0e"),
                        ("tightest_legal", "#8c564b")]:
        junk = z[f"{name}__junk"].astype(float)
        w = 100
        ys = np.convolve(junk, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(ys)) + w // 2, ys, color=color, lw=1.2,
                label=name.replace("_", " ") + " (ungated)")
    ax.set_xlabel("bank position")
    ax.set_ylabel("junk fraction (100-item window)")
    ax.set_title("(b) what fills the bank after saturation")
    ax.legend(fontsize=7)

    ax = axes[2]
    names = list(s)
    real = [s[n]["real_bank_size"] for n in names]
    junk = [s[n]["junk_in_bank"] for n in names]
    ypos = np.arange(len(names))
    ax.barh(ypos, real, color="#1f77b4", label="real items")
    ax.barh(ypos, junk, left=real, color="#d62728", label="junk items")
    ax.set_yticks(ypos)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    ax.set_xlabel("final bank composition")
    ax.set_title("(c) real vs fake capacity")
    ax.legend(fontsize=7)

    fig.suptitle("Exam-item world: hard novelty floor $\\delta$, finite packing", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_exam.png", bbox_inches="tight")
    plt.close(fig)


def fig4_live():
    runs = {}
    for tag in ["poems", "poems_naive", "items"]:
        p = HERE / "live" / tag / "corpus.jsonl"
        if p.exists():
            recs = [json.loads(l) for l in p.read_text().splitlines()]
            if recs:
                runs[tag] = recs
    if not runs:
        print("no live runs found; skipping fig4")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    ax = axes[0]
    for tag, color, label in [("poems", "#1f77b4", "IHD stack"),
                              ("poems_naive", "#888888", "naive prompting")]:
        if tag not in runs:
            continue
        E = np.array([r["emb"] for r in runs[tag]])
        sims = E @ E.T
        np.fill_diagonal(sims, -1)
        # running max-similarity: for item i, most similar EARLIER item
        run_max = [sims[i, :i].max() if i > 0 else 0.0 for i in range(len(E))]
        ax.plot(np.arange(len(E)), run_max, color=color, lw=1.0, label=label)
    ax.set_xlabel("poem index")
    ax.set_ylabel("cosine sim to nearest earlier poem")
    ax.set_title("(a) live poems: drift toward self-similarity")
    ax.legend(fontsize=7)

    ax = axes[1]
    from simulate import vendi_score_linear
    for tag, color, label in [("poems", "#1f77b4", "IHD stack"),
                              ("poems_naive", "#888888", "naive prompting")]:
        if tag not in runs:
            continue
        E = np.array([r["emb"] for r in runs[tag]])
        xs = list(range(5, len(E) + 1, 5))
        ys = [vendi_score_linear(E[:m]) for m in xs]
        ax.plot(xs, ys, color=color, lw=1.2, label=label)
    ax.set_xlabel("corpus size")
    ax.set_ylabel("Vendi Score")
    ax.set_title("(b) live poems: effective diversity")
    ax.legend(fontsize=7)

    ax = axes[2]
    if "items" in runs:
        E = np.array([r["emb"] for r in runs["items"]])
        q = [r.get("quality", 0) for r in runs["items"]]
        sims = E @ E.T
        np.fill_diagonal(sims, -1)
        nn_sim = sims.max(axis=1)
        ax.scatter(nn_sim, q, s=18, color="#2ca02c", alpha=0.8)
        ax.axvline(0.90, color="#d62728", ls="--", lw=1, label="enemy-item floor (0.90)")
        ax.set_xlabel("cosine sim to nearest other item")
        ax.set_ylabel("judge validity (0-10)")
        ax.set_title("(c) live exam items: floor respected, validity kept")
        ax.legend(fontsize=7)
    fig.suptitle("Live pilots: gpt-5.6-luna + nomic-embed-text", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_live.png", bbox_inches="tight")
    plt.close(fig)


def fig5_scaling():
    """T2/T7 panels: measured decay slopes + best-of-K gain, from fresh draws."""
    from scipy.spatial.distance import cdist
    from scipy import stats as sstats
    rng = np.random.default_rng(4242)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    ax = axes[0]
    for k_int, color in [(3, "#d62728"), (8, "#ff7f0e"), (32, "#1f77b4")]:
        Q, _ = np.linalg.qr(rng.normal(size=(32, k_int)))
        ns = np.array([250, 500, 1000, 2000, 4000, 8000])
        means = []
        for n in ns:
            pts = rng.normal(size=(n, k_int)) @ Q.T
            fresh = rng.normal(size=(200, k_int)) @ Q.T
            means.append(cdist(fresh, pts).min(axis=1).mean())
        slope = sstats.linregress(np.log(ns), np.log(means)).slope
        ax.loglog(ns, means, "o-", color=color, ms=3, lw=1.1,
                  label=f"$k={k_int}$: slope {slope:.3f} (theory $-1/{k_int}$ = {-1/k_int:.3f})")
    ax.set_xlabel("corpus size $n$")
    ax.set_ylabel("expected min-gap of a fresh draw")
    ax.set_title("(a) novelty decays like $n^{-1/k}$")
    ax.legend(fontsize=7)

    ax = axes[1]
    Ks = np.array([1, 2, 4, 8, 16, 32, 64])
    for k_int, color in [(3, "#d62728"), (8, "#ff7f0e")]:
        Q, _ = np.linalg.qr(rng.normal(size=(32, k_int)))
        pts = rng.normal(size=(2000, k_int)) @ Q.T
        gains = []
        for K in Ks:
            g = []
            for _ in range(200):
                cands = rng.normal(size=(K, k_int)) @ Q.T
                g.append(cdist(cands, pts).min(axis=1).max())
            gains.append(np.mean(g))
        gains = np.array(gains) / gains[0]
        ax.loglog(Ks, gains, "o-", color=color, ms=3, lw=1.1, label=f"measured, $k={k_int}$")
        ax.loglog(Ks, Ks ** (1 / k_int), "--", color=color, lw=0.9, alpha=0.6,
                  label=f"$K^{{1/{k_int}}}$")
    ax.set_xlabel("candidates per step $K$")
    ax.set_ylabel("min-gap gain over $K=1$")
    ax.set_title("(b) oversampling buys only $K^{1/k}$")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG / "fig5_scaling.png", bbox_inches="tight")
    plt.close(fig)


def fig6_horizon():
    with open(FIG / "coverage_horizon.json") as f:
        H = json.load(f)
    styles = {
        "naive": ("#888888", "naive (accept all)"),
        "high_temp": ("#d62728", "high temperature"),
        "random_latent": ("#8c564b", "random latent / persona"),
        "novelty_online": ("#ff7f0e", "online max-min"),
        "ihd_no_ledger": ("#1f77b4", "IHD selection"),
        "ihd_refining": ("#2ca02c", "IHD + recursive refinement"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for k, (c, lab) in styles.items():
        if k not in H:
            continue
        ax.semilogx(H[k]["generations"], H[k]["headroom"], "o-", ms=2.6, lw=1.2,
                    color=c, label=lab)
    ax.set_xlabel("total generations consumed")
    ax.set_ylabel("novelty headroom (median best-of-$K$ gap)")
    ax.set_title("(a) headroom against the same probe")
    ax.legend(fontsize=6.5)

    ax = axes[1]
    for k, (c, lab) in styles.items():
        if k not in H:
            continue
        ax.semilogx(H[k]["generations"], H[k]["inability"], "o-", ms=2.6, lw=1.2, color=c)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("total generations consumed")
    ax.set_ylabel("inability to be novel  $1 - h(g)/h(5)$")
    ax.set_title("(b) the coverage horizon closing")
    fig.suptitle("Coverage horizon: 5 $\\to$ 10,000 generations", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_horizon.png", bbox_inches="tight")
    plt.close(fig)


def fig7_budget():
    with open(FIG / "budget_comparison.json") as f:
        BC = json.load(f)
    R = BC["results"]
    baselines = ["naive", "high_temp", "random_latent", "dedup_post", "fps_post",
                 "novelty_online"]
    ours = ["ihd_full", "ihd_no_gate", "ihd_no_orth", "ihd_no_quality",
            "ihd_no_repulsion", "ihd_no_ledger"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8))

    ax = axes[0]
    for grp, color, mark in [(baselines, "#888888", "o"), (ours, "#1f77b4", "s")]:
        xs = [R[k]["mean_quality"] for k in grp if k in R]
        ys = [R[k]["vendi"] for k in grp if k in R]
        ax.scatter(xs, ys, c=color, marker=mark, s=42, zorder=3)
        for k in grp:
            if k in R:
                ax.annotate(k.replace("ihd_", "").replace("_", " "),
                            (R[k]["mean_quality"], R[k]["vendi"]),
                            textcoords="offset points", xytext=(5, 3), fontsize=6)
    ax.set_xlabel("mean true quality")
    ax.set_ylabel("Vendi Score")
    ax.set_title("(a) quality vs diversity at $B=10{,}000$")

    ax = axes[1]
    names = [k for k in baselines + ours if k in R]
    ypos = np.arange(len(names))
    ax.barh(ypos, [R[k]["junk_fraction"] for k in names],
            color=["#888888" if k in baselines else "#1f77b4" for k in names])
    ax.set_yticks(ypos)
    ax.set_yticklabels([k.replace("_", " ") for k in names], fontsize=6.5)
    ax.set_xlabel("junk fraction of final corpus")
    ax.set_title("(b) off-manifold contamination")

    ax = axes[2]
    ax.barh(ypos, [R[k]["n"] for k in names],
            color=["#888888" if k in baselines else "#1f77b4" for k in names])
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels([k.replace("_", " ") for k in names], fontsize=6.5)
    ax.set_xlabel("corpus size kept from the same 10,000 generations")
    ax.set_title("(c) yield")
    fig.suptitle("Budget-matched comparison and ablations", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_budget.png", bbox_inches="tight")
    plt.close(fig)


def fig8_slices():
    with open(FIG / "slice_theory_checks.json") as f:
        S = json.load(f)
    sweep = S["depth_sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for c, color in [("3.0", "#1f77b4"), ("30.0", "#d62728")]:
        if c not in sweep:
            continue
        row = sweep[c]
        xs = sorted(int(k) for k in row)
        ys = [row[str(x)] for x in xs]
        ax.semilogx(xs, ys, "o-", ms=3.5, lw=1.3, color=color,
                    label=f"prompt-switch cost $c={float(c):.0f}$")
        bi = int(np.argmax(ys))
        ax.scatter([xs[bi]], [ys[bi]], s=110, facecolors="none", edgecolors=color, lw=1.6)
    ax.set_xlabel("depth: samples per prompt $n$")
    ax.set_ylabel("fraction of the $d$-manifold $\\epsilon$-covered")
    ax.set_title("Breadth vs depth at fixed budget\n(circled = optimum)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_breadth_depth.png", bbox_inches="tight")
    plt.close(fig)


def fig9_slice_world():
    with open(FIG / "slice_world.json") as f:
        S = json.load(f)
    styles = {
        "fixed_prompt": ("#d62728", "fixed prompt (one slice)"),
        "random_spec": ("#888888", "random spec"),
        "calculus_spec": ("#ff7f0e", "calculus-chosen spec"),
        "random_spec_refine": ("#9467bd", "random spec + refinement"),
        "calculus_spec_refine": ("#1f77b4", "calculus + refinement"),
        "calculus_spec_refine_guided": ("#2ca02c", "calculus + guided refinement"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))

    ax = axes[0]
    for k, (c, lab) in styles.items():
        tr = S[k]["track"]
        ax.semilogx([r["generations"] for r in tr], [r["inability"] for r in tr],
                    "o-", ms=2.6, lw=1.2, color=c, label=lab)
    ax.set_xlabel("total generations consumed")
    ax.set_ylabel("inability to be novel")
    ax.set_title("(a) the horizon, with and without refinement")
    ax.legend(fontsize=6.2, loc="upper left")

    ax = axes[1]
    # the three refinement tracks land on identical reachable dimensions, so
    # they are drawn with different dashes and small vertical offsets --
    # otherwise the last one plotted hides the other two entirely.
    dashes = {"random_spec_refine": (4, 2), "calculus_spec_refine": (2, 2),
              "calculus_spec_refine_guided": None}
    offs = {"random_spec_refine": -0.14, "calculus_spec_refine": 0.0,
            "calculus_spec_refine_guided": 0.14}
    for k, (c, lab) in styles.items():
        tr = S[k]["track"]
        y = np.array([r["reach_dim"] for r in tr], dtype=float) + offs.get(k, 0.0)
        ln, = ax.semilogx([r["generations"] for r in tr], y, "o-", ms=2.6,
                          lw=1.4, color=c, label=lab)
        d = dashes.get(k)
        if d:
            ln.set_dashes(d)
    ax.set_xlabel("total generations consumed")
    ax.set_ylabel("reachable dimension  rank(span $v$) + $m$")
    ax.set_title("(b) refinement raises the ceiling")
    ax.legend(fontsize=6.0, loc="center left")

    ax = axes[2]
    names = list(styles)
    ypos = np.arange(len(names))
    ax.barh(ypos, [S[k]["vendi"] for k in names],
            color=[styles[k][0] for k in names])
    ax.set_yticks(ypos)
    ax.set_yticklabels([styles[k][1] for k in names], fontsize=6.5)
    ax.set_xlabel("final Vendi Score")
    ax.set_title("(c) diversity actually achieved")
    fig.suptitle("Conditional world: a prompt selects an $m$-dimensional slice "
                 "($d=24$, $m=3$)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig9_slice_world.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_poetry()
    fig2_tradeoff()
    fig3_exam()
    fig4_live()
    fig5_scaling()
    fig6_horizon()
    fig7_budget()
    fig8_slices()
    fig9_slice_world()
    print("figures written to", FIG)
