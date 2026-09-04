"""
Is the boundary about enumerability, or about the domain?

Iteration 34 drew a line: stereotypy is diagnosable wherever a level is realized
in language, but cheaply fixable only where the ways of realizing it can be
written down. The evidence was a size gap. Enumerating the FORMS a red herring
can take moved the item bank's dominant device from 76% to 24%; enumerating
general rhetorical VEHICLES moved poems by 0.04, leaving `ritual pronouncement`
reaching for "let" in 69% of its poems.

But those two axes differ in more than domain. The item axis enumerated the
realization space OF THE TARGETED CONSTRUCT. The poem axis enumerated vehicles
in general -- orthogonal to every stance, and therefore specific to none. A
general axis is the weaker instrument by construction, so the size gap does not
yet show that poetic stance is unenumerable. It may only show that I did not
enumerate it.

This closes that gap. Rather than one vehicle axis beside `Claim posture`, each
of its five levels is REFINED into five ways that particular stance is
performed -- twenty-five sub-levels, level-balanced. Ritual pronouncement
becomes invocation, blessing, oath, anathema, consecration-by-naming. This is
RAC's own refinement mechanism (§5.3) applied at device grain rather than by
saturation, and it is the construct-specific instrument the item domain got.

Scored on the PARENT posture so the numbers compare directly with iteration 34,
using iteration 33's validated metric, with the same blind contract judge.

Usage:
    python3 poem_refined_devices.py [n_per_arm]
"""
from __future__ import annotations

import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from pipeline import USAGE                                  # noqa: E402
from stereotypy_scan import stereotypy, words               # noqa: E402
from poem_device_axis import BASE_AXES, gen, honours, build  # noqa: E402

SEED = 20260911
N_DEFAULT = 80
WORKERS = 8

# Each posture refined into five ways THAT posture is performed.
REFINED = {
    "Ritual pronouncement": [
        "as an invocation that lists what it swears by",
        "as a blessing conferred on someone present",
        "as an oath binding the speaker to a future act",
        "as an anathema or curse pronounced against something",
        "as a consecration performed by naming a thing correctly"],
    "Provisional inquiry": [
        "as a question the poem declines to answer",
        "as a hypothesis offered and then qualified",
        "as competing explanations weighed against each other",
        "as an admission that the evidence is insufficient",
        "as a revision of a claim made earlier in the poem"],
    "Flat assertion": [
        "as an entry in a record or log",
        "as a measurement reported without comment",
        "as a rule stated in the present tense",
        "as a caption naming what is in view",
        "as a verdict delivered in one clause"],
    "Confession under pressure": [
        "as an answer given to an unnamed interrogator",
        "as a disclosure the speaker interrupts and resumes",
        "as an accounting of what the speaker failed to do",
        "as a plea entered on the speaker's own behalf",
        "as something overheard being told to a third party"],
    "Refusal to claim": [
        "as an explicit declining to say",
        "as a description that stops short of the conclusion",
        "as deferral to someone else's authority to say it",
        "as a list of what cannot be established",
        "as silence marked in the poem's own structure"],
}

REFINED_AXIS = {"Claim posture": [f"{k} -- {d}" for k, ds in REFINED.items()
                                  for d in ds]}


def parent(level: str) -> str:
    return level.split(" -- ")[0]


def main(n: int) -> None:
    axes = {**BASE_AXES, **REFINED_AXIS}
    specs = build(axes, n, SEED)
    print(f"refined arm: {n} poems, {len(axes)} axes, "
          f"{len(REFINED_AXIS['Claim posture'])} posture sub-levels", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(gen, specs))
    keep = [(t, s) for t, s in zip(outs, specs) if t]
    texts = [t for t, _ in keep]
    print(f"  {len(texts)}/{n} generated; judging the posture contract blind",
          flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ok = list(ex.map(honours, [(t, parent(s["Claim posture"])) for t, s in keep]))
    good = [x for x in ok if x is not None]
    print(f"  posture exhibited: {sum(good)}/{len(good)} = "
          f"{sum(good)/max(len(good),1):.0%}")

    docs = [words(t) for t in texts]
    labs = [parent(s["Claim posture"]) for _, s in keep]
    obs = stereotypy(docs, labs)

    prev = {}
    f = HERE / "poem_device_arms.jsonl"
    if f.exists():
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for arm in ("base", "device"):
            t = [r["text"] for r in rows if r["arm"] == arm]
            l_ = [str(r["spec"]["Claim posture"]) for r in rows if r["arm"] == arm]
            prev[arm] = stereotypy([words(x) for x in t], l_)

    print(f"\n{'level':<30}{'base':>18}{'vehicle':>18}{'refined':>18}")
    for lv in sorted(REFINED):
        def cell(d):
            x = d.get(lv)
            return f"{(x['device'] or '-')} {x['share']:.0%}" if x else "-"
        print(f"{lv[:29]:<30}{cell(prev.get('base', {})):>18}"
              f"{cell(prev.get('device', {})):>18}{cell(obs):>18}")

    def mean(d):
        return float(np.mean([x["share"] for x in d.values()])) if d else float("nan")
    mb, mv, mr = mean(prev.get("base", {})), mean(prev.get("device", {})), mean(obs)
    print(f"\n{'MEAN device share':<30}{mb:>18.3f}{mv:>18.3f}{mr:>18.3f}")
    print(f"{'vs base':<30}{'':>18}{mv-mb:>+18.3f}{mr-mb:>+18.3f}")

    rng = np.random.default_rng(SEED)
    base_rows = [json.loads(l) for l in
                 (HERE / "poem_device_arms.jsonl").read_text().splitlines() if l.strip()]
    bt = [r["text"] for r in base_rows if r["arm"] == "base"]
    bl = [str(r["spec"]["Claim posture"]) for r in base_rows if r["arm"] == "base"]
    bd = [words(x) for x in bt]
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(bd), len(bd)); j = rng.integers(0, len(docs), len(docs))
        vb = stereotypy([bd[k] for k in i], [bl[k] for k in i])
        vr = stereotypy([docs[k] for k in j], [labs[k] for k in j])
        if len(vb) >= 2 and len(vr) >= 2:
            diffs.append(mean(vr) - mean(vb))
    if diffs:
        d = np.array(diffs); lo, hi = np.percentile(d, [2.5, 97.5])
        s = "*" if lo > 0 or hi < 0 else " "
        print(f"\n{s} refined - base: {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    with open(HERE / "poem_refined_arm.jsonl", "w") as fh:
        for (t, s_), o in zip(keep, ok):
            fh.write(json.dumps({"arm": "refined", "spec": s_,
                                 "exhibits": o, "text": t}) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
