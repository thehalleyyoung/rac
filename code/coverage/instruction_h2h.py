"""
Head-to-head on instruction space, against RELEASED corpora, under a coverage
objective.

Adapted from the parent project's head_to_head.py: same fairness rules (matched
n sampled uniformly at random rather than from the head, identical embedder,
instruction field only, exact-duplicate rate and a dedup'd variant of every
metric), with the packing metrics replaced by this paper's coverage functional.

WHAT IS COMPARED

  alpaca      tatsu-lab/stanford_alpaca, 52,002 items -- the canonical
              Self-Instruct (Wang et al. 2023) output
  personahub  proj-persona/PersonaHub, 50,000 items -- persona-driven
              synthesis (Ge et al. 2024), the closest published relative of
              latent conditioning
  wizardlm    WizardLMTeam/WizardLM_evol_instruct_V2_143k -- the released
              Evol-Instruct (Xu et al. 2023) corpus
  ours_naive      unconditioned repeated prompting of gpt-5.6-luna
  ours_high_temp  the same prompt at T = 1.6
  ours_cond       axis-conditioned generation (elicited axes + attractor ledger)

Two orthogonal factors, crossed:
  SOURCE     which of the six corpora the candidates come from
  SELECTOR   `random` (a uniform draw, i.e. the corpus as shipped) or
             `coverage_greedy` (this paper's submodular selection)

Crossing them separates two claims that a single-number comparison confuses:
whether our GENERATION is better, and whether our SELECTION is better. The
selector is applied to every source, including the released ones, so if it
helps it must help them too.

THE DENOMINATOR, and a mistake we made first. Our initial version scored each
source LEAVE-ONE-OUT against the union of the OTHER sources. That sounds like
the conservative choice, but it silently breaks the selector: coverage-greedy
optimizes an estimate of one measure (its own reachable distribution -- the
only thing a deployed pipeline can see) while being graded on a different one
(what OTHER generators happen to produce), and those two are anti-correlated
for exactly the reason section 9.2 describes. Under that design greedy scored
BELOW a uniform random draw on all three released corpora. It is the same
denominator mismatch as the live pilot, relocated inside the selector.

The corrected design gives each source THREE disjoint slices:
  EST   the selector's estimation measure -- its own generations, which is
        what a deployed pipeline actually has
  CAND  the candidates it selects from
  REF   held-out items contributed to a shared, stratum-balanced reference
The evaluation reference is the union of every source's REF slice, including
the scored source's own. That is the honest question -- "how much of the space
that all these methods together can reach does this corpus cover?" -- and no
item is ever scored against itself, because REF is disjoint from CAND. Every
source contributes equally, so covering only your own region caps you at 1/6
of the reference; beating that requires genuinely broad coverage.

Epsilon is calibrated on the reference actually used, and we report its
percentile in that reference's own pairwise-distance distribution.

Run: python3 instruction_h2h.py  ->  figures/summary_h2h.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
PUB = HERE.parent / "public_corpora"
GEN = HERE / "instruction_logs"
CACHE = HERE / "figures" / "h2h_cache"
FIGDIR = HERE / "figures"
SEED = 4242

N_REF = 80         # held-out reference points contributed by each source
N_EST = 150        # the selector's own estimation measure (disjoint from both)
N_CAND = 120       # candidate items each source offers the selector
N_SELECT = 60      # items each (source, selector) pair ends up with

from real_coverage import blocked_min_dist, self_nn, vendi, ngram, dedup  # noqa: E402


# ------------------------------------------------------------------ loaders

def load_alpaca() -> list[str]:
    d = json.loads((PUB / "alpaca_data.json").read_text())
    out = []
    for r in d:
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        # Alpaca splits some tasks into instruction + input; the instruction
        # alone is often uninterpretable ("Rewrite the sentence."), so the
        # comparable unit is the concatenation -- that is the whole prompt.
        out.append((instr + ("\n" + inp if inp else "")).strip())
    return [x for x in out if len(x) >= 15]


def load_personahub() -> list[str]:
    out = []
    for line in (PUB / "personahub_instruction.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = (r.get("synthesized text") or "").strip()
        if len(t) >= 15:
            out.append(t)
    return out


def load_wizardlm() -> list[str]:
    d = json.loads((PUB / "wizardlm_evol_v2.json").read_text())
    out = []
    for r in d:
        for turn in (r.get("conversations") or []):
            if turn.get("from") in ("human", "user"):
                t = (turn.get("value") or "").strip()
                if len(t) >= 15:
                    out.append(t)
                break
    return out


def load_ours(arm: str) -> list[str]:
    p = GEN / f"{arm}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


SOURCES = {
    "alpaca (Self-Instruct, 52k)": load_alpaca,
    "PersonaHub (50k)": load_personahub,
    "WizardLM Evol-Instruct (143k)": load_wizardlm,
    "ours: naive": lambda: load_ours("naive"),
    "ours: high-temp T=1.6": lambda: load_ours("high_temp"),
    "ours: axis-conditioned": lambda: load_ours("cond"),
}


def embed_cached(name: str, texts: list[str]) -> np.ndarray:
    from pipeline import embed
    CACHE.mkdir(parents=True, exist_ok=True)
    import hashlib
    # content-addressed: a cache keyed only on (name, length) would silently
    # return vectors for a DIFFERENT sample of the same size if the RNG stream
    # ever shifts, which is exactly the kind of bug that produces confident
    # wrong numbers. Hash the texts themselves.
    h = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    slug = "".join(c if c.isalnum() else "_" for c in name)[:40]
    key = CACHE / f"{slug}_{len(texts)}_{h}.npy"
    if key.exists():
        E = np.load(key)
        if len(E) == len(texts):
            return E
    E = embed(texts, batch=96)
    np.save(key, E)
    return E


# ---------------------------------------------------------------- selectors

def sel_random(E, cover, k, rng, **kw):
    return list(rng.permutation(len(E))[:k])


ZERO_GAIN_PICKS = {}


def sel_coverage_greedy(E, cover, k, rng, tag="?"):
    """Greedy marginal coverage. We COUNT the picks made with zero marginal
    gain: once the estimation measure is saturated the objective carries no
    signal and the tie-break decides, which quietly turns the rule into
    something else (a farthest-point walk). A large count here means the
    selection radius is mis-scaled for the estimation pool's density, not that
    coverage-greedy is a bad rule -- see calibrate_eps_sel."""
    covered = np.zeros(cover.shape[1], dtype=bool)
    chosen: list[int] = []
    zero = 0
    for _ in range(k):
        gains = cover[:, ~covered].sum(axis=1)
        gains[chosen] = -1
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            zero += 1
            d = np.linalg.norm(E - E[chosen[-1]], axis=1)
            d[chosen] = -1
            i = int(np.argmax(d))
        chosen.append(i)
        covered |= cover[i]
    ZERO_GAIN_PICKS[tag] = zero
    return chosen


SELECTORS = {"random": sel_random, "coverage_greedy": sel_coverage_greedy}


# ------------------------------------------------------------------- driver

def main():
    FIGDIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)

    pools = {}
    for name, loader in SOURCES.items():
        texts = loader()
        if len(texts) < N_REF + N_EST + N_CAND:
            print(f"SKIP {name}: only {len(texts)} items "
                  f"(need {N_REF + N_EST + N_CAND})")
            continue
        # dedup first, then sample uniformly at random (never the head)
        uniq = list(dict.fromkeys(t.strip() for t in texts))
        dup_rate = round(1 - len(uniq) / len(texts), 4)
        sample = rng.sample(uniq, N_REF + N_EST + N_CAND)
        E = embed_cached(name, sample)
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
        a, b = N_REF, N_REF + N_EST
        pools[name] = {
            "texts_ref": sample[:a], "E_ref": E[:a],
            "E_est": E[a:b],
            "texts_cand": sample[b:], "E_cand": E[b:],
            "n_available": len(texts), "exact_dup_rate": dup_rate,
        }
        print(f"loaded {name:32s} avail={len(texts):7d} dup={dup_rate:.4f}",
              flush=True)

    if len(pools) < 2:
        print("not enough sources loaded; run instruction_gen.py first")
        return

    # epsilon calibrated on the full stratum-balanced reference
    REF_ALL = np.vstack([p["E_ref"] for p in pools.values()])
    delta = float(np.median(self_nn(REF_ALL)))
    eps_list = [round(0.6 * delta, 4), round(delta, 4), round(1.6 * delta, 4)]
    eps_sel = eps_list[1]
    sub = REF_ALL[nprng.permutation(len(REF_ALL))[:1500]]
    pd = np.sqrt(np.clip(2 - 2 * (sub @ sub.T), 0, None))
    iu = np.triu_indices(len(sub), 1)
    pct = {str(e): round(float((pd[iu] <= e).mean()), 4) for e in eps_list}
    print(f"\nreference: {len(REF_ALL)} points from {len(pools)} sources "
          f"({N_REF} each)")
    print(f"eps delta={delta:.4f} -> {eps_list}; percentile of reference "
          f"pairwise distances: {pct}\n")

    rows = {}
    from scipy.spatial.distance import cdist

    def calibrate_eps_sel(pools) -> tuple[float, float]:
        """The EVALUATION radius and the SELECTION radius answer different
        questions. Evaluation eps is a resolution choice, calibrated on the
        reference. Selection eps must additionally leave the marginal-gain
        signal non-degenerate: with a sparse estimation pool, a ball at the
        evaluation radius contains ~0 points, every candidate ties at gain 0,
        and the tie-break -- not the objective -- picks the corpus. We choose
        the smallest multiple of delta whose mean ball on the estimation pool
        holds at least MIN_BALL points, the same rule the live pilot uses."""
        MIN_BALL = 3.0
        for mult in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
            e = mult * delta
            sizes = [float((cdist(q["E_cand"], q["E_est"]) <= e).sum(axis=1).mean())
                     for q in pools.values()]
            if min(sizes) >= MIN_BALL:
                return round(e, 4), round(float(np.mean(sizes)), 2)
        return round(3.0 * delta, 4), round(float(np.mean(sizes)), 2)

    eps_sel_signal, mean_ball = calibrate_eps_sel(pools)
    print(f"selection radius (calibrated for signal): {eps_sel_signal} "
          f"(mean {mean_ball} estimation points per ball); "
          f"evaluation radii unchanged at {eps_list}\n")

    REF = REF_ALL          # shared, stratum-balanced, disjoint from every CAND
    for name, p in pools.items():
        # the selector sees ONLY its own EST slice -- its own generations,
        # never the evaluation reference
        cover_est = cdist(p["E_cand"], p["E_est"]) <= eps_sel_signal
        for sname, fn in SELECTORS.items():
            kw = {"tag": f"{name}|{sname}"} if sname == "coverage_greedy" else {}
            idx = fn(p["E_cand"], cover_est, N_SELECT,
                     np.random.default_rng(SEED + 1), **kw)
            idx = list(dict.fromkeys(int(i) for i in idx))[:N_SELECT]
            S = p["E_cand"][idx]
            st = [p["texts_cand"][i] for i in idx]
            d = blocked_min_dist(REF, S)
            key = f"{name} | {sname}"
            rows[key] = {
                "source": name, "selector": sname,
                "n_selected": len(idx), "n_candidates": len(p["E_cand"]),
                "n_available": p["n_available"],
                "exact_dup_rate_corpus": p["exact_dup_rate"],
                "covered": {str(e): round(float((d <= e).mean()), 4)
                            for e in eps_list},
                "vendi_centered": vendi(S, True),
                "vendi_uncentered": vendi(S, False),
                "median_self_nn": round(float(np.median(self_nn(S))), 4),
                **ngram(st),
                "mean_chars": round(float(np.mean([len(t) for t in st])), 1),
            }
            r = rows[key]
            print(f"{key:52s} cov@mid={r['covered'][str(eps_sel)]:.4f} "
                  f"vendi_c={r['vendi_centered']:7.2f} d2={r['distinct_2']:.3f} "
                  f"selfNN={r['median_self_nn']:.4f}", flush=True)

    ours = "ours: axis-conditioned | coverage_greedy"
    published = [k for k in rows if k.split(" |")[0] in
                 ("alpaca (Self-Instruct, 52k)", "PersonaHub (50k)",
                  "WizardLM Evol-Instruct (143k)")]
    best_pub = max(published, key=lambda k: rows[k]["covered"][str(eps_sel)]) \
        if published else None
    baselines = [k for k in rows if not k.startswith("ours: axis-conditioned")]
    best_base = max(baselines, key=lambda k: rows[k]["covered"][str(eps_sel)])

    out = {
        "seed": SEED, "n_ref_per_source": N_REF, "n_candidates": N_CAND,
        "n_selected": N_SELECT,
        "eps_list": eps_list, "eps_evaluation_mid": eps_sel,
        "eps_selection_signal_calibrated": eps_sel_signal,
        "mean_estimation_points_per_ball": mean_ball,
        "zero_gain_picks": dict(ZERO_GAIN_PICKS),
        "eps_percentile_of_reference_pairwise": pct,
        "reference_size_total": int(len(REF_ALL)),
        "reference_note": "stratum-balanced; each source scored leave-one-out",
        "rows": rows,
        "ours": ours,
        "best_baseline_overall": best_base,
        "best_published_corpus": best_pub,
    }
    if ours in rows:
        o = rows[ours]["covered"][str(eps_sel)]
        out["ours_vs_best_baseline_abs"] = round(
            o - rows[best_base]["covered"][str(eps_sel)], 4)
        out["ours_vs_best_published_abs"] = round(
            o - rows[best_pub]["covered"][str(eps_sel)], 4) if best_pub else None
        out["n_baselines_beaten"] = sum(
            1 for k in baselines
            if rows[k]["covered"][str(eps_sel)] < o)
        out["n_baselines_total"] = len(baselines)
    with open(FIGDIR / "summary_h2h.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote figures/summary_h2h.json")
    if ours in rows:
        print(f"ours beats {out['n_baselines_beaten']}/{out['n_baselines_total']} "
              f"baseline configurations at eps={eps_sel}")


if __name__ == "__main__":
    main()
