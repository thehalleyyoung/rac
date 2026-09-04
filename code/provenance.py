"""
Provenance: which procedure produced which number.

This project's method changed several times while the experiments were
running, and results from earlier procedures are still worth keeping -- they
are real measurements, and some of them are the reason the procedure changed.
What is NOT acceptable is presenting a number produced by an older procedure
as though the current framework produced it.

So every result file gets an explicit record of the procedure that made it:
which script, with what policy, under what constraints, and what is different
about it relative to the current framework. The paper then cites the
procedure alongside the number.

The three procedures used across this project, oldest first:

  P1  simulation-only
      Synthetic mixture worlds; no LLM in the loop. Selection over sampled
      candidates with anchor/repulsion/orthogonality terms. Establishes the
      asymptotics and the theorems; makes no claim about real generators.

  P2  real-text, select-and-reject
      gpt-5.6-luna generations, nomic-embed-text embeddings, K candidates per
      slot with the losers DISCARDED. Diversity measured on text (n-gram and
      embedding). This is the procedure behind the 43k-generation corpora, the
      seven-arm comparison, and the MMLU head-to-head.
      Differs from current: rejection sampling is permitted; steering and
      measurement both happen in text space; the artifact is never rendered.

  P3  cross-modal steering, no rejection
      The current framework. Selection happens over PROMPTS before rendering,
      using the shared text/artifact space of a multimodal embedder (CLAP for
      audio, CLIP for images) as a pre-render estimate of where the artifact
      will land. Every render is kept -- no rejection at any point. The
      calculus is evaluated in the REAL space (score_axes_realized), on the
      artifacts rather than on level descriptions, and recursion is triggered
      by exhaustion measured in that space.
      Differs from P2: no rejection; steering signal is cross-modal; the
      objective is measured on the rendered artifact, not on its prompt.

Numbers from P1 and P2 remain in the paper. They are labelled.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
REAL = HERE / "real"

PROCEDURES = {
    "P1": {
        "name": "simulation-only",
        "llm": None,
        "selection": "over sampled candidates",
        "rejection": True,
        "measured_in": "synthetic embedding space",
        "artifact_rendered": False,
        "calculus": "text/level-description space (score_axes)",
    },
    "P2": {
        "name": "real-text, select-and-reject",
        "llm": "openai/gpt-5.6-luna",
        "selection": "over K rendered-as-text candidates, losers discarded",
        "rejection": True,
        "measured_in": "text (n-gram + nomic-embed-text)",
        "artifact_rendered": False,
        "calculus": "text/level-description space (score_axes)",
    },
    "P2.5": {
        "name": "cross-modal gate WITH rejection (superseded)",
        "llm": "openai/gpt-5.6-luna",
        "selection": "render, then accept/reject on CLAP validity + novelty",
        "rejection": True,
        "measured_in": "rendered artifact (CLAP audio)",
        "artifact_rendered": True,
        "calculus": "bandit-style axis credit, not the calculus proper",
    },
    "P3": {
        "name": "cross-modal steering, no rejection",
        "llm": "openai/gpt-5.6-luna",
        "selection": "over PROMPTS pre-render, via shared text/artifact embedding",
        "rejection": False,
        "measured_in": "rendered artifact (CLAP/MERT audio, CLIP image)",
        "artifact_rendered": True,
        "calculus": "real space (score_axes_realized / score_axes_coverage)",
    },
}

# Every result artifact this project produces, mapped to the procedure that
# made it. Anything not listed here is unlabelled and must not be cited.
RESULTS = {
    # ---- P1: simulation ------------------------------------------------
    "figures/summary.json": ("P1", "simulate.py", "poetry world, 6 policies, n=10,000"),
    "figures/exam_summary.json": ("P1", "simulate_exam.py", "hard-floor bank packing"),
    "figures/budget_comparison.json": ("P1", "experiment_budget.py",
                                       "budget-matched baselines + ablations"),
    "figures/coverage_horizon.json": ("P1", "coverage_horizon.py",
                                      "inability-to-be-novel, 5 -> 10,000"),
    "figures/slice_world.json": ("P1", "simulate_slices.py",
                                 "conditional world, d=24 m=3"),
    "figures/theory_checks.json": ("P1", "verify_theory.py", "8 theorem checks"),
    "figures/slice_theory_checks.json": ("P1", "verify_slices.py",
                                         "7 conditional-dimension checks"),
    # ---- P2: real text, rejection allowed -------------------------------
    "figures/real_curves.json": ("P2", "metrics.py",
                                 "literal+latent diversity, n=5..10,000"),
    "figures/arm_comparison.json": ("P2", "compare_arms.py",
                                    "7 arms at matched n, text metrics"),
    "figures/head_to_head.json": ("P2", "head_to_head.py",
                                  "vs MMLU, matched n=1000"),
    "figures/mureka_text.json": ("P2", "audio_domain.py prompts",
                                 "instrumental PROMPT diversity only, no audio"),
    "figures/vision_summary.json": ("P2", "figures_real.py",
                                    "text vs CLIP on 199 low-quality renders"),
    "figures/vision_compare.json": ("P2", "vision_compare.py VISION_TAG=high",
                                    "7 arms x 100 HIGH-quality renders of P2 corpora, "
                                    "matched n=96; a medium-quality n=48 pass exists "
                                    "and differs in BOTH quality and n, so the two are "
                                    "not a clean quality ablation"),
    "figures/audio_compare.json": ("P2", "audio_compare.py",
                                   "CLAP/MERT on renders of P2 prompt corpora"),
    "figures/lyria_length_probe.json": ("P2", "lyria_calibrate.py",
                                        "prompt length vs contract compliance"),
    # ---- P3: cross-modal steering, no rejection -------------------------
    "real/lyria_steer/run_summary.json": ("P3", "lyria_steer.py",
                                          "100 tracks, CLAP-steered, 0 rejections"),
    "real/dalle_steer_maxmin/run_summary.json": ("P3", "image_steer.py maxmin",
                                                 "CLIP-steered max-min, 0 rejections"),
    "real/dalle_steer_coverage/run_summary.json": ("P3", "image_steer.py coverage",
                                                   "CLIP-steered coverage, 0 rejections"),
    "real/lyria_online/run_summary.json": ("P2.5", "lyria_online.py",
                                           "CLAP gate WITH rejection -- superseded by "
                                           "lyria_steer.py; kept for the rejection-cost "
                                           "comparison"),
}


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def build() -> dict:
    rev = git_rev()
    out = {"generated": datetime.now().isoformat(timespec="seconds"),
           "git_rev": rev, "procedures": PROCEDURES, "results": {}}
    missing = []
    for rel, (proc, script, note) in RESULTS.items():
        p = HERE / rel
        if not p.exists():
            missing.append(rel)
            continue
        st = p.stat()
        out["results"][rel] = {
            "procedure": proc, "script": script, "note": note,
            "bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    out["missing_not_yet_produced"] = missing
    (FIG).mkdir(exist_ok=True)
    with open(FIG / "provenance.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def report():
    o = build()
    print(f"provenance @ {o['git_rev']}  ({len(o['results'])} result files)\n")
    by = {}
    for rel, r in o["results"].items():
        by.setdefault(r["procedure"], []).append((rel, r))
    for proc in sorted(by):
        meta = PROCEDURES.get(proc, {})
        print(f"[{proc}] {meta.get('name', proc)}"
              f"  rejection={meta.get('rejection')}  measured_in={meta.get('measured_in')}")
        for rel, r in sorted(by[proc]):
            print(f"    {rel:52s} {r['script']:28s} {r['note']}")
        print()
    if o["missing_not_yet_produced"]:
        print("not yet produced (do not cite):")
        for m in o["missing_not_yet_produced"]:
            print(f"    {m}")


if __name__ == "__main__":
    report()
