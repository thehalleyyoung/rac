"""Figures for the coverage-at-a-finite-budget paper. Reads only artifacts
produced by simulate_coverage.py, verify_theory.py, and live_pilot.py."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

POLICIES = ["naive", "maxmin", "coverage", "gated_coverage",
            "gated_coverage_refine", "gated_coverage_refine_calc"]
LABEL = {"naive": "naive iid", "maxmin": "max-min (packing)",
         "coverage": "coverage-greedy", "gated_coverage": "gated coverage",
         "gated_coverage_refine": "gated coverage + refine",
         "gated_coverage_refine_calc": "RAC-coverage"}
COLOR = {"naive": "#888888", "maxmin": "#d62728", "coverage": "#1f77b4",
         "gated_coverage": "#2ca02c", "gated_coverage_refine": "#9467bd",
         "gated_coverage_refine_calc": "#e377c2"}
DOMAIN_TITLE = {"redteam": "red-team eval suite", "persona": "user-persona corpus"}


def load_sim():
    with open(FIG / "summary_sim.json") as f:
        summary = json.load(f)
    data = {d: np.load(FIG / f"sim_{d}.npz") for d in summary["domains"]}
    # Only plot policies actually present in the artifacts, so a partial or
    # in-progress run still produces every figure it has data for.
    present = [p for p in POLICIES
               if all(f"{p}__curve_x" in data[d] for d in data)]
    missing = [p for p in POLICIES if p not in present]
    if missing:
        print(f"note: policies absent from artifacts, omitted from figures: {missing}")
    return summary, data, present


def fig_coverage_growth(summary, data, POL):
    domains = list(summary["domains"])
    fig, axes = plt.subplots(len(domains), 3, figsize=(13, 7), sharex=True)
    for r, dom in enumerate(domains):
        eps_list = summary["domains"][dom]["eps_list"]
        for c, e in enumerate(eps_list):
            ax = axes[r, c]
            for p in POL:
                ax.plot(data[dom][f"{p}__curve_x"], data[dom][f"{p}__curve_y_{e}"],
                        color=COLOR[p], label=LABEL[p], lw=1.6)
            ax.set_title(f"{DOMAIN_TITLE[dom]}, $\\varepsilon$={e}", fontsize=10)
            ax.grid(alpha=0.3)
            if r == len(domains) - 1:
                ax.set_xlabel("items accepted $n$")
            if c == 0:
                ax.set_ylabel("covered fraction of held-out pool")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Coverage growth at a finite budget (n = 10,000), tracked on a held-out 20k reachable pool")
    fig.tight_layout()
    fig.savefig(FIG / "fig_coverage_growth.png", dpi=150)
    plt.close(fig)


def fig_tradeoff(summary, POL):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, dom in zip(axes, summary["domains"]):
        s = summary["domains"][dom]
        eps_mid = str(s["eps_list"][1])
        for p in POL:
            v = s["policies"][p]
            ax.scatter(v["min_gap"], v["covered_base"][eps_mid], s=90,
                       color=COLOR[p], label=LABEL[p], zorder=3)
            ax.annotate(f"junk {100*v['junk_fraction']:.1f}%",
                        (v["min_gap"], v["covered_base"][eps_mid]),
                        textcoords="offset points", xytext=(7, -3), fontsize=7)
        ax.set_xlabel("min-gap of chosen set (packing objective)")
        ax.set_ylabel(f"covered fraction @ $\\varepsilon$={eps_mid} (coverage objective)")
        ax.set_title(DOMAIN_TITLE[dom])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Covering and packing pull apart: no policy wins both")
    fig.tight_layout()
    fig.savefig(FIG / "fig_tradeoff.png", dpi=150)
    plt.close(fig)


def fig_allocation():
    with open(FIG / "summary_theory.json") as f:
        th = json.load(f)
    alloc = th["allocation"]
    npz = np.load(FIG / "allocation.npz")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    rules = ["equal", "proportional", "sqrt", "greedy_marginal"]
    rl = {"equal": "equal per cell", "proportional": "$\\propto$ measure",
          "sqrt": "$\\propto \\sqrt{measure}$", "greedy_marginal": "greedy marginal"}
    eps_grid = alloc["eps_grid"]
    width = 0.2
    xs = np.arange(len(eps_grid))
    for i, rule in enumerate(rules):
        vals = [alloc["rules"][rule][str(e)] for e in eps_grid]
        axes[0].bar(xs + (i - 1.5) * width, vals, width, label=rl[rule])
    axes[0].set_xticks(xs, [f"$\\varepsilon$={e}" for e in eps_grid])
    axes[0].set_ylabel("covered fraction of full mixture (50k pool)")
    axes[0].set_title(f"Allocation of n = {alloc['n']:,} across M = {alloc['M']} known cells")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    order = np.argsort(-npz["weights"])
    axes[1].plot(npz["weights"][order] * alloc["n"], label="$\\propto$ measure", lw=1.5)
    axes[1].plot(npz["picks_per_cell"][order], label="greedy marginal picks", lw=1.5)
    axes[1].axhline(alloc["n"] / alloc["M"], color="k", ls="--", lw=1, label="equal per cell")
    axes[1].set_xlabel("cell (sorted by measure, descending)")
    axes[1].set_ylabel("items allocated")
    axes[1].set_yscale("log")
    axes[1].set_title("What greedy actually allocates: measure-aware but saturating")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_allocation.png", dpi=150)
    plt.close(fig)


def fig_denominator(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, dom in zip(axes, summary["domains"]):
        s = summary["domains"][dom]
        eps_mid = str(s["eps_list"][1])
        p = s["policies"]["gated_coverage_refine"]
        base_naive = s["policies"]["naive"]["covered_base"][eps_mid]
        bars = {
            "base reachable pool\n(fixed denominator)": p["covered_base"][eps_mid],
            "own final reachable pool\n(denominator grew)": p["covered_own_reachable"][eps_mid],
            "full latent space\n(oracle pool)": p["covered_world"][eps_mid],
        }
        naive_world = s["policies"]["naive"]["covered_world"][eps_mid]
        ax.bar(range(len(bars)), list(bars.values()),
               color=["#1f77b4", "#9467bd", "#ff7f0e"])
        ax.set_xticks(range(len(bars)), list(bars.keys()), fontsize=8)
        ax.axhline(base_naive, color="#888888", ls="--", lw=1.2,
                   label=f"naive on base pool ({base_naive:.2f})")
        ax.axhline(naive_world, color="#888888", ls=":", lw=1.2,
                   label=f"naive on full space ({naive_world:.2f})")
        ax.set_ylabel(f"covered fraction @ $\\varepsilon$={eps_mid}")
        ax.set_title(f"{DOMAIN_TITLE[dom]}: refine policy, three denominators")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("The honest-denominator problem: refinement grows the thing being covered")
    fig.tight_layout()
    fig.savefig(FIG / "fig_denominator.png", dpi=150)
    plt.close(fig)


def fig_spectrum(summary, data, POL):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, dom in zip(axes[:2], summary["domains"]):
        for p in POL:
            ax.plot(data[dom][f"{p}__vendi_x"], data[dom][f"{p}__vendi_y"],
                    color=COLOR[p], label=LABEL[p], lw=1.5)
        ax.set_xlabel("items accepted $n$")
        ax.set_ylabel("Vendi score (linear kernel)")
        ax.set_title(DOMAIN_TITLE[dom])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    # diminishing marginal gains of the accepted items (submodularity in the wild)
    dom = list(summary["domains"])[0]
    for p in [q for q in ("coverage", "gated_coverage", "gated_coverage_refine",
                          "gated_coverage_refine_calc") if q in POL]:
        g = data[dom][f"{p}__gain_log"]
        w = 200
        smooth = np.convolve(g, np.ones(w) / w, mode="valid")
        axes[2].plot(np.arange(len(smooth)) + w // 2, smooth, color=COLOR[p],
                     label=LABEL[p], lw=1.4)
    if "gated_coverage_refine__refine_ts" in data[dom]:
        for i, t in enumerate(data[dom]["gated_coverage_refine__refine_ts"]):
            axes[2].axvline(t, color="#9467bd", alpha=0.12, lw=0.8,
                            label="refinement event" if i == 0 else None)
    axes[2].set_xlabel("items accepted $n$")
    axes[2].set_ylabel("marginal gain (pool pts, 200-item mean)")
    axes[2].set_yscale("log")
    axes[2].set_title("Diminishing returns + refinement resets", fontsize=10)
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_spectrum_vendi.png", dpi=150)
    plt.close(fig)


def fig_live():
    if not (FIG / "summary_live.json").exists():
        print("live pilot artifacts missing; skipping fig_live")
        return
    with open(FIG / "summary_live.json") as f:
        live = json.load(f)
    npz = np.load(FIG / "live_curves.npz")
    eps_list = live["eps_list"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    styles = ["--", "-", ":"]
    for e, ls in zip(eps_list, styles):
        for name, color in (("naive", "#888888"), ("method", "#1f77b4")):
            y = npz[f"{name}_{e}"]
            axes[0].plot(np.arange(1, len(y) + 1), y, ls=ls, color=color,
                         label=f"{name} @ $\\varepsilon$={e}", lw=1.6)
    axes[0].set_xlabel("items accepted")
    axes[0].set_ylabel(f"covered fraction of {live['pool_size']}-item reachable pool")
    axes[0].set_title("Live pilot (gpt-5.6-luna): coverage growth")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    import json as _json
    corpus = [_json.loads(l) for l in
              (HERE / "live_logs" / "corpus.jsonl").read_text().splitlines()]
    gains = [r["marginal_gain"] for r in corpus]
    axes[1].bar(range(1, len(gains) + 1), gains, color="#1f77b4", width=0.9)
    axes[1].set_xlabel("accepted item index")
    axes[1].set_ylabel(f"marginal gain (pool points newly within $\\varepsilon$={live['eps_selection']})")
    axes[1].set_title("Per-item marginal coverage gain (method run)")
    axes[1].grid(alpha=0.3, axis="y")

    # the actual result: each policy covers its OWN proposal stratum
    strat = live["covered_fraction_by_stratum_at_eps_sel"]
    labels = ["unconditioned\nstratum (240)", "axis-conditioned\nstratum (120)",
              "pooled (360)"]
    keys = ["unconditioned_stratum", "conditioned_stratum", "full_pool"]
    x = np.arange(len(labels))
    axes[2].bar(x - 0.2, [strat["naive"][k] for k in keys], 0.4,
                label="naive", color="#888888")
    axes[2].bar(x + 0.2, [strat["method"][k] for k in keys], 0.4,
                label="method", color="#1f77b4")
    axes[2].set_xticks(x, labels, fontsize=8)
    axes[2].set_ylabel(f"covered fraction @ $\\varepsilon$={live['eps_selection']}")
    axes[2].set_title("Each policy covers its own proposal stratum", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig_live_pilot.png", dpi=150)
    plt.close(fig)


def fig_literal_vs_latent():
    """Literal (n-gram) vs latent (embedding) diversity as n grows, plus the
    centered/uncentered Vendi gap. Live-pilot data only."""
    path = FIG / "summary_live.json"
    if not path.exists():
        print("live artifacts missing; skipping fig_literal_vs_latent")
        return
    with open(path) as f:
        live = json.load(f)
    rows = live.get("diversity_vs_n") or []
    if not rows:
        print("no diversity_vs_n rows; skipping fig_literal_vs_latent")
        return
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    ax.plot(ns, [r["distinct_2"] for r in rows], "o-", color="#d62728",
            label="distinct-2 (literal)")
    ax.plot(ns, [r["distinct_1"] for r in rows], "s--", color="#ff7f0e",
            label="distinct-1 (literal)")
    ax.set_xlabel("items accepted")
    ax.set_ylabel("distinct n-gram ratio")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ns, [r["vendi_centered"] for r in rows], "^-", color="#1f77b4",
             label="Vendi, mean-centered (latent)")
    ax2.plot(ns, [r["vendi_uncentered"] for r in rows], "v:", color="#9467bd",
             label="Vendi, uncentered (latent)")
    ax2.set_ylabel("Vendi score")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax.set_title("Literal diversity falls while latent diversity rises", fontsize=10)

    ax = axes[1]
    cone = live.get("cone_diagnostics", {})
    div = live.get("diversity_two_level", {})
    names = [k for k in ("naive", "method", "pool_unconditioned") if k in div]
    x = np.arange(len(names))
    unc = [div[n]["embedding_vendi_uncentered"] for n in names]
    cen = [div[n]["embedding_vendi_centered"] for n in names]
    ax.bar(x - 0.2, unc, 0.4, label="uncentered", color="#9467bd")
    ax.bar(x + 0.2, cen, 0.4, label="mean-centered", color="#1f77b4")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Vendi score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    sub = (f"pairwise cos mean {cone.get('pairwise_cosine_mean')}, "
           f"p95 {cone.get('pairwise_cosine_p95')}")
    ax.set_title(f"Kernel artifact: the shared mean direction\n({sub})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_literal_vs_latent.png", dpi=150)
    plt.close(fig)


def main():
    summary, data, POL = load_sim()
    fig_coverage_growth(summary, data, POL)
    fig_tradeoff(summary, POL)
    fig_denominator(summary)
    fig_spectrum(summary, data, POL)
    if (FIG / "summary_theory.json").exists():
        fig_allocation()
    else:
        print("theory artifacts missing; skipping fig_allocation")
    fig_live()
    fig_literal_vs_latent()
    fig_benchmark()
    print("figures written:", sorted(p.name for p in FIG.glob("fig_*.png")))




def fig_benchmark():
    """Budget-matched selection benchmark on real corpora (section 9.5)."""
    path = FIG / "summary_benchmark.json"
    if not path.exists():
        print("benchmark artifacts missing; skipping fig_benchmark")
        return
    with open(path) as f:
        bm = json.load(f)
    doms = list(bm["domains"])
    fig, axes = plt.subplots(1, len(doms) + 1, figsize=(5 * (len(doms) + 1), 4.6))
    TITLE = {"dalle": "DALL-E instructions", "psychometric": "psychometric items"}
    allpts = []
    for ax, dom in zip(axes[:len(doms)], doms):
        r = bm["domains"][dom]
        e = str(r["eps_selection"])
        items = sorted(r["methods"].items(), key=lambda kv: kv[1]["covered"][e])
        names = [k for k, _ in items]
        vals = [v["covered"][e] for _, v in items]
        cols = ["#1f77b4" if "ours" in n else "#999999" for n in names]
        ax.barh(range(len(names)), vals, color=cols)
        ax.set_yticks(range(len(names)),
                      [n.replace(" (ours)", "\n(ours)") for n in names], fontsize=8)
        ax.set_xlabel(f"covered fraction @ $\\varepsilon$={e}")
        ax.set_title(f"{TITLE.get(dom, dom)}  (k={r['k_budget']}, matched budget)",
                     fontsize=10)
        ax.grid(alpha=0.3, axis="x")
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
        for n, v in r["methods"].items():
            allpts.append((dom, n, v["min_gap"], v["covered"][e]))

    ax = axes[-1]
    marks = {"dalle": "o", "psychometric": "s"}
    for dom, n, gap, cov in allpts:
        ax.scatter(gap, cov, s=70, marker=marks.get(dom, "o"),
                   color="#1f77b4" if "ours" in n else "#d62728",
                   alpha=0.85, zorder=3)
    ax.set_xlabel("min-gap (packing objective)")
    ax.set_ylabel("covered fraction (covering objective)")
    ax.set_title("Packing and covering are different objectives\n"
                 "(blue = ours, red = baselines; circle = DALL-E, square = psych)",
                 fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_benchmark.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
