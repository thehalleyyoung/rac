"""
Does per-axis obedience really fall as contracts-per-prompt rises?

§6.15 states a contract budget: the generator honours an axis less faithfully
when the prompt carries more contracts. That claim was measured on images, and
the obvious way to port it to poems -- run a 7-contract arm against an
11-contract arm and compare per-axis realization -- cannot answer it.

The reason is not sample size in the usual sense. Realization has to be a
PARTIAL effect (§4.4): the variance an axis moves once every other axis is
adjusted for. Adjusting observationally needs a dummy column per level per
other axis, and at n=50 with nine to eleven axes that design matrix has 33-46
columns for at most 50 rows. The regression interpolates, the residuals are
identically zero, and every partial estimate is an artifact. Subset
conditioning makes it strictly worse, because "<absent>" adds a level to every
axis it drops. The marginal estimate is available but useless here for the
opposite reason: fewer contracts means axes co-occur less, so marginal effects
rise mechanically through reduced confounding -- precisely the bias that
motivated partial scoring in the first place.

So this measures the claim the way the claim is phrased: by MANIPULATION. Hold
a base spec fixed, set one axis to each of its levels, and read how far the
poem moves. There is no adjustment set, so there is no design matrix and no
degrees-of-freedom problem; the other axes are constant by construction rather
than regressed away. Running the same probe at two contract loads asks exactly
whether the load attenuates the effect of setting an axis.

Effects are read on PROSODY, not on the embedding, so that "the axis moved the
poem" is not graded by the instrument the objective is written in.

Usage:
    python3 probe_contract_load.py [runs...]
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from pipeline import USAGE, chat            # noqa: E402
from text_realization import prosodic_features, eta_squared  # noqa: E402

LOADS = (7, 11)
N_BASES = 4
N_PERM = 2000
WORKERS = 12
SEED = 20260902

# The probe prompt carries the contracts and nothing else. No avoid-block and
# no local negatives: those are corpus-state dependent, and a probe has to hold
# everything except the manipulated axis fixed.
PROBE_PROMPT = """Write one complete poem, 8-24 lines, in English. No title unless asked.

The poem MUST exhibit every one of these latent behaviors. They are contracts, \
not suggestions -- a reader who knew to look should be able to point to where \
each one operates:
{contracts}

Do not explain the poem. Output only the poem."""


def final_axes(run: Path) -> dict[str, list[str]]:
    """Replay the axis log to the axis set the run finished with."""
    axes: dict[str, list[str]] = {}
    for line in (run / "axes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for a in d.get("axes", []) or []:
            axes[a["name"]] = list(a["levels"])
        a = d.get("axis")
        if isinstance(a, dict) and a.get("name"):
            axes[a["name"]] = list(a["levels"])
    return axes


def spec_text(spec: dict[str, str]) -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in spec.items())


def generate(spec: dict[str, str]) -> str | None:
    """Retry with backoff.

    Without this, a third of generations came back empty and the losses were
    concentrated in whichever axes happened to be probed first -- the whole job
    list hits the pool at once, the opening burst gets rate-limited, and the
    result is an axis-correlated hole in the design that looks like an
    axis-specific effect. Retrying is not politeness to the API; it is what
    keeps the missingness from aliasing onto the thing being measured.
    """
    last = None
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.5) + 2.0 * attempt)
        try:
            t = chat([{"role": "user", "content":
                       PROBE_PROMPT.format(contracts=spec_text(spec))}],
                     temperature=1.0, max_tokens=2500)
            if len(t.strip()) > 40:
                return t.strip()
            last = "short output"
        except Exception as e:
            last = repr(e)
    print(f"   generate failed after 4 tries: {last}", flush=True)
    return None


def effect(texts: list[str], labels: list[str], bases: list[int],
           n_perm: int = 0) -> tuple[float, float]:
    """Null-corrected between-level variance share, centered within base.

    Centering within base is what makes this a manipulation estimate: only
    variation produced by SETTING the axis contributes, and any level of the
    base spec is differenced out.
    """
    F = [prosodic_features(t) for t in texts]
    names = sorted(F[0])
    M = np.array([[f[n] for n in names] for f in F])
    M = (M - M.mean(0)) / np.clip(M.std(0), 1e-9, None)
    for b in set(bases):            # remove the base main effect
        m = np.array([x == b for x in bases])   # bases may be tuples
        M[m] -= M[m].mean(0)
    k, n = len(set(labels)), len(labels)
    if k < 2 or n <= k:
        return float("nan"), float("nan")
    eta = eta_squared(M, labels)
    null = (k - 1) / (n - 1)
    rho = float(np.clip((eta - null) / max(1.0 - null, 1e-9), 0.0, 1.0))
    if not n_perm:
        return rho, float("nan")
    # Permute levels WITHIN base: the null is "the commanded level carries no
    # information", with the base structure left intact.
    rng = np.random.default_rng(SEED)
    lab = np.array(labels)
    masks = [np.array([x == b for x in bases]) for b in set(bases)]
    hits = 0
    for _ in range(n_perm):
        q = lab.copy()
        for m in masks:
            q[m] = rng.permutation(q[m])
        if eta_squared(M, list(q)) >= eta:
            hits += 1
    return rho, float((hits + 1) / (n_perm + 1))


def main(runs: list[Path]) -> None:
    rng = random.Random(SEED)
    axes = final_axes(runs[0])
    targets = [a for a, lv in sorted(axes.items()) if len(lv) >= 3]
    others = sorted(axes)
    print(f"{len(axes)} axes; probing {len(targets)} targets at loads {LOADS}\n")

    jobs, meta = [], []
    for tgt in targets:
        pool = [a for a in others if a != tgt]
        for bi in range(N_BASES):
            base = {a: rng.choice(axes[a]) for a in pool}
            order = rng.sample(pool, len(pool))
            for load in LOADS:
                keep = order[:max(load - 1, 0)]
                for lv in axes[tgt]:
                    spec = {tgt: lv} | {a: base[a] for a in keep}
                    jobs.append(spec)
                    meta.append((tgt, load, bi, lv))

    print(f"{len(jobs)} generations queued", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(generate, jobs))

    # Persist every generation before any analysis: a probe whose outputs are
    # not on disk cannot be re-analysed without paying for it twice.
    with open(HERE / "probe_generations.jsonl", "w") as fh:
        for o, (tgt, load, bi, lv) in zip(outs, meta):
            fh.write(json.dumps({"axis": tgt, "load": load, "base": bi,
                                 "level": lv, "text": o}) + "\n")
    n_ok = sum(1 for o in outs if o)
    print(f"{n_ok}/{len(outs)} generations returned text")

    rows = []
    print(f"\n{'axis':<34}{'levels':>7}{'rho@7':>9}{'rho@11':>9}{'delta':>9}"
          f"{'rho(all)':>10}{'p':>8}")
    for tgt in targets:
        cell = {}
        for load in LOADS:
            sel = [(o, m) for o, m in zip(outs, meta)
                   if o and m[0] == tgt and m[1] == load]
            if len(sel) < 6:
                cell[load] = float("nan"); continue
            cell[load], _ = effect([o for o, _ in sel], [m[3] for _, m in sel],
                                   [m[2] for _, m in sel])
        d = cell[7] - cell[11]
        # pooled over both loads, with a permutation p: is the axis realized AT ALL?
        allsel = [(o, m) for o, m in zip(outs, meta) if o and m[0] == tgt]
        rho_all, p_all = effect([o for o, _ in allsel], [m[3] for _, m in allsel],
                                [(m[1], m[2]) for _, m in allsel], n_perm=N_PERM)
        rows.append({"axis": tgt, "levels": len(axes[tgt]), "n": len(allsel),
                     "rho_7": cell[7], "rho_11": cell[11], "delta": d,
                     "rho_all": rho_all, "p": p_all})
        print(f"{tgt[:33]:<34}{len(axes[tgt]):>7}{cell[7]:>9.4f}"
              f"{cell[11]:>9.4f}{d:>+9.4f}{rho_all:>10.4f}{p_all:>8.4f}")

    good = [r for r in rows if np.isfinite(r["delta"])]
    up = sum(1 for r in good if r["delta"] > 0)
    print(f"\n{'MEAN':<34}{'':>7}{np.nanmean([r['rho_7'] for r in good]):>9.4f}"
          f"{np.nanmean([r['rho_11'] for r in good]):>9.4f}"
          f"{np.nanmean([r['delta'] for r in good]):>+9.4f}")
    print(f"{up}/{len(good)} axes obeyed better at load 7")
    sig = [r for r in rows if np.isfinite(r["p"]) and r["p"] < 0.05]
    print(f"axes realized under manipulation (p<0.05): {len(sig)}/"
          f"{sum(1 for r in rows if np.isfinite(r['p']))}"
          f"   mean rho = {np.nanmean([r['rho_all'] for r in rows]):.4f}")
    print(f"spend ${USAGE.cost_usd():.2f}")
    json.dump(rows, open(HERE / "probe_contract_load.json", "w"), indent=2)


if __name__ == "__main__":
    rs = [Path(a) for a in sys.argv[1:]] or [RESEARCH / "live" / "poems_ctl_exp_s11"]
    main([r.resolve() for r in rs])
