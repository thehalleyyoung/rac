"""Pull every finished run off the volume and render the current picture.

Safe to run at any time: it reads results only, so it can be called while the
sweep is still going and will simply show fewer points.
"""
from __future__ import annotations

import json
import subprocess
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "results.json"

LABEL = {"alpaca_covsel": "coverage-selected", "alpaca_random": "random",
         "alpaca_full": "full Alpaca pool"}
COLOR = {"alpaca_covsel": "#1f77b4", "alpaca_random": "#999999",
         "alpaca_full": "#c1583a"}


def fetch() -> list[dict]:
    """modal volume get is the only dependency-free way in from the host."""
    out = subprocess.run(
        ["modal", "volume", "ls", "rac-sft-vol", "results"],
        capture_output=True, text=True, timeout=180)
    names = [l.strip().split("/")[-1] for l in out.stdout.splitlines()
             if l.strip().endswith(".json")]
    rows = []
    tmp = HERE / ".dl"
    tmp.mkdir(exist_ok=True)
    for n in names:
        f = tmp / n
        subprocess.run(["modal", "volume", "get", "--force", "rac-sft-vol",
                        f"results/{n}", str(f)], capture_output=True, timeout=180)
        if f.exists():
            try:
                rows.append(json.loads(f.read_text()))
            except Exception:
                pass
    CACHE.write_text(json.dumps(rows, indent=2))
    return rows


def load() -> list[dict]:
    try:
        return fetch()
    except Exception as e:
        print(f"(live fetch failed: {str(e)[:70]}; using cache)")
        return json.loads(CACHE.read_text()) if CACHE.exists() else []


def agg(rows: list[dict], metric: str, x: str = "n_items") -> dict:
    """{arm: [(x, mean, spread, n_seeds), ...]} sorted by x.

    The claim under test is that a better-chosen corpus needs FEWER DATAPOINTS,
    so item count is the axis. Training tokens are available as a secondary
    view, but they conflate item count with item length: coverage-greedy happens
    to pick items about 6% shorter than random, which would shift its curve left
    for a reason that has nothing to do with coverage.
    """
    by = {}
    for r in rows:
        if metric not in r:
            continue
        by.setdefault(r["arm"], {}).setdefault(r["n_items"], []).append(
            (r[x], r[metric]))
    out = {}
    for arm, per_n in by.items():
        pts = []
        for n, vals in sorted(per_n.items()):
            ys = [v for _, v in vals]
            pts.append((st.mean([t for t, _ in vals]), st.mean(ys),
                        st.stdev(ys) if len(ys) > 1 else 0.0, len(ys)))
        out[arm] = pts
    return out


def x_to_target(pts, target: float) -> float | None:
    """Interpolate on the log x-axis for the first crossing of target."""
    import math
    for (t0, y0, _, _), (t1, y1, _, _) in zip(pts, pts[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            f = (target - y0) / (y1 - y0)
            return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
    return None


def text(metric: str = "superni_macro", x: str = "n_items") -> str:
    rows = load()
    lines = [f"runs complete: {len(rows)}", ""]
    a = agg(rows, metric, x)
    if not a:
        return "\n".join(lines + [f"(no runs have reported {metric} yet)"])
    tok_per = {}
    for r in rows:
        if metric in r:
            tok_per.setdefault(r["arm"], {})[r["n_items"]] = r["train_tokens"]
    lines.append(f"{'arm':20s} {'items':>6} {'seeds':>5}  {metric:>16}  {'(tokens)':>9}")
    for arm in ("alpaca_covsel", "alpaca_random", "alpaca_full"):
        for (xv, m, sd, k) in a.get(arm, []):
            n = int(xv)
            lines.append(f"{LABEL.get(arm, arm):20s} {n:6d} {k:5d}  "
                         f"{m:.4f} ±{sd:.4f}  {tok_per.get(arm, {}).get(n, 0):9d}")
    full = a.get("alpaca_full") or []
    if full and len(full) > 1:
        ceiling = max(y for _, y, _, _ in full)
        lines += ["", f"ceiling (best full-pool point): {ceiling:.4f}"]
        for arm in ("alpaca_covsel", "alpaca_random"):
            t = x_to_target(a.get(arm, []), ceiling * 0.95)
            lines.append(f"  {LABEL[arm]:20s} ITEMS to 95% of ceiling: "
                         + (f"{t:,.0f}" if t else "not reached"))
    return "\n".join(lines)


def chart(metric: str = "superni_macro", out: str = "figures/sweep.png",
          x: str = "n_items") -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load()
    a = agg(rows, metric, x)
    if not a:
        return None
    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    base = next((r[metric] for r in rows if r.get("arm", "").startswith("base")
                 and metric in r), None)
    if base is not None:
        ax.axhline(base, color="#000000", lw=1.2, ls="--", alpha=0.7)
        ax.text(0.99, base, f" base, no SFT ({base:.3f})", transform=
                ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    for arm, pts in a.items():
        if not pts or arm.startswith("base"):
            continue
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        e = [p[2] for p in pts]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, lw=1.8,
                    color=COLOR.get(arm, "#555"), label=LABEL.get(arm, arm))
    ax.set_xscale("log")
    ax.set_xlabel("training items" if x == "n_items" else "training tokens")
    ax.set_xscale("log")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Data efficiency: how many items to reach a given breadth")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / out, dpi=150)
    return out


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "superni_macro"
    xaxis = sys.argv[2] if len(sys.argv) > 2 else "n_items"
    print(text(m, xaxis))
    p = chart(m, x=xaxis)
    print(f"\nchart: {p}" if p else "\n(no chart yet)")
