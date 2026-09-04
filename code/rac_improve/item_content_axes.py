"""
Does an item bank need CONTENT axes, or is naming the domain enough?

Reading the bank says it does not span its domain. Three items drawn at random
are all cloud backup and availability-zone problems, and across the corpus
"cost" appears in 80% of items, "recovery" in 60%, "storage" in 50%. The seven
elicited axes explain why: `Cognitive operation demanded`, `Scenario framing`,
`Distractor encoding`, `Constraint structure`, `Reasoning representation`,
`Actor perspective`, `Target difficulty` -- every one describes item
ARCHITECTURE, none describes subject matter. The elicitation could not have
proposed otherwise: its prompt asks for axes of item-design variation and says
"NOT topic lists".

This matters more for exam items than the analogous gap does elsewhere. A bank
is built to a BLUEPRINT -- content domain crossed with cognitive level -- and a
bank that varies its forms while repeating one content area is unusable however
well-formed its items are. Test information is not the only thing that has to
span; content has to span too, and no amount of distractor variety substitutes.

The generation prompt already names the domain broadly ("architecture, storage,
networking, security, cost, reliability"). If naming a domain were enough, the
bank would already span it. So the arms differ only in whether the three
content dimensions the gap judge recovered are CONDITIONED ON:

  base       the seven architectural axes
  content    the same seven plus deployment model, service abstraction level,
             and data modality

Coverage is scored by a blind keyword classifier over ten standard blueprint
areas, never by the axis levels themselves -- scoring arm `content` by the
levels it was commanded with would be circular.

Usage:
    python3 item_content_axes.py [n_per_arm]
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))

from pipeline import USAGE, chat, parse_json          # noqa: E402

SEED = 20260904
N_DEFAULT = 40
WORKERS = 8

ARCH_AXES = {
    "Cognitive operation demanded": [
        "Recall a definition or stated fact", "Apply a rule to a new case",
        "Analyze relationships, causes, or consequences",
        "Evaluate alternatives against stated criteria",
        "Diagnose a fault from symptoms"],
    "Scenario framing": [
        "Decontextualized question with direct conditions",
        "Short static operational vignette",
        "Comparative case set requiring transfer across two situations",
        "Incident narrative unfolding over time",
        "Planning brief with stated budget and constraints"],
    "Distractor encoding": [
        "Common misconception or inverted principle",
        "Technically relevant but irrelevant-to-the-question option",
        "Near-neighbor option that is valid in a different condition",
        "Correct action at the wrong scope or time",
        "Overgeneralized best practice"],
    "Constraint structure": [
        "One decisive condition", "Two independent conditions",
        "Several conjunctive conditions that must all hold",
        "Competing conditions requiring a trade-off",
        "A condition plus an explicit exclusion"],
    "Reasoning representation": [
        "Purely qualitative reasoning",
        "Ordinal comparison such as higher, lower, sooner, or more suitable",
        "Explicit arithmetic on stated quantities",
        "Probability, expected-value, or uncertainty reasoning",
        "Threshold or limit comparison against a stated bound"],
    "Actor perspective": [
        "Business or service owner", "Platform or operations engineer",
        "Security, risk, or compliance reviewer",
        "Finance or cost-management owner", "End user or application developer"],
    "Target difficulty": [
        "Very easy: one stated fact, no distractor competition",
        "Easy: one rule applied directly",
        "Moderate: two conditions, one attractive distractor",
        "Hard: several interacting conditions and a near-neighbor distractor",
        "Very hard: trade-off under uncertainty with two defensible-looking options"],
}

# Recovered by the gap judge from the items themselves, shown the artifacts and
# told which axes already exist.
CONTENT_AXES = {
    "Deployment and locality model": [
        "Single-site or single-availability-zone deployment",
        "Multi-availability-zone deployment within one region",
        "Multi-region public-cloud deployment",
        "Hybrid or multicloud deployment",
        "Edge, disconnected, or intermittently connected deployment"],
    "Cloud service abstraction level": [
        "Infrastructure resources such as VMs, disks, and networks",
        "Managed infrastructure services such as databases, queues, and file systems",
        "Platform services such as managed runtimes and container platforms",
        "Serverless functions and event-driven application services",
        "SaaS or fully managed business applications"],
    "Data and processing modality": [
        "Relational transactional data", "Object and unstructured data",
        "File and shared-content data", "Key-value, document, or graph data",
        "Streaming, analytical, time-series, or warehouse data"],
}

ITEM_PROMPT = """Write one multiple-choice exam item for a professional certification in \
GENERAL CLOUD COMPUTING (architecture, storage, networking, security, cost, \
reliability -- practitioner level).

The item MUST exhibit these latent design behaviors:
{contracts}

Requirements: one stem, exactly 4 options (A-D), exactly one defensibly correct \
answer, plausible distractors that each encode a distinct misconception. Return \
JSON only:
{{"stem": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, \
"correct": "A", "distractor_misconceptions": {{"B": "...", "C": "...", "D": "..."}}}}"""

# Ten standard blueprint areas. Deliberately written against cloud
# certification vocabulary rather than against the axis levels, so that neither
# arm can score well merely by echoing the words it was conditioned on.
BLUEPRINT = {
    "compute": r"\b(vm|virtual machine|instance type|autoscal|container|kubernetes|cluster node|cpu|serverless function|lambda)\b",
    "storage": r"\b(object storage|block storage|disk|volume|bucket|archive tier|file share|snapshot)\b",
    "networking": r"\b(vpc|subnet|load balancer|dns|cdn|bandwidth|latency|gateway|peering|firewall rule|routing)\b",
    "security": r"\b(encrypt|key management|kms|certificate|vulnerabilit|patch|threat|malware|zero trust|hardening)\b",
    "identity": r"\b(iam|identity|role|permission|least privilege|mfa|federation|single sign|service account|policy attach)\b",
    "data": r"\b(database|sql|nosql|query|schema|index|warehouse|analytic|stream|etl|partition key|replica lag)\b",
    "reliability": r"\b(availability zone|failover|rpo|rto|backup|disaster recovery|redundan|resilien|sla|uptime)\b",
    "cost": r"\b(cost|price|pricing|budget|spend|reserved instance|savings plan|per month|billing|chargeback)\b",
    "monitoring": r"\b(monitor|metric|log|alert|trace|observab|dashboard|audit trail|telemetry)\b",
    "migration": r"\b(migrat|lift and shift|rehost|replatform|cutover|on-premises|datacenter move|legacy)\b",
}


def areas(text: str) -> set[str]:
    t = text.lower()
    return {a for a, p in BLUEPRINT.items() if re.search(p, t)}


def gen(spec: dict) -> str | None:
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    last = None
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.2) + 2.0 * attempt)
        try:
            raw = chat([{"role": "user",
                         "content": ITEM_PROMPT.format(contracts=contracts)}],
                       temperature=1.0, max_tokens=3000, json_mode=True)
            o = parse_json(raw)
            opts = o["options"]
            txt = o["stem"] + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
            if len(txt) > 80:
                return txt
            last = "short"
        except Exception as e:
            last = repr(e)
    print(f"   gen failed: {last}", flush=True)
    return None


def build(axes: dict, n: int, seed: int) -> list[dict]:
    """Least-used-level balancing, the rule ported in iteration 22."""
    rng = random.Random(seed)
    used = {a: Counter() for a in axes}
    specs = []
    for _ in range(n):
        s = {}
        for a, lv in axes.items():
            lo = min(used[a].get(x, 0) for x in lv)
            s[a] = rng.choice([x for x in lv if used[a].get(x, 0) == lo])
            used[a][s[a]] += 1
        specs.append(s)
    return specs


def report(name: str, texts: list[str]) -> dict:
    cov = [areas(t) for t in texts]
    hit = Counter(a for c in cov for a in c)
    n = len(texts)
    # document frequency of the most common content word: the concentration
    # that reading the corpus made visible
    STOP = set("the a an and or of to in for with that this is are be as on at by from which "
               "must not all any each per than then when if it its their have has can could "
               "should would may might one two three which what item option".split())
    df = Counter()
    for t in texts:
        df.update({w for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in STOP})
    top = df.most_common(1)[0]
    p = np.array([hit.get(a, 0) for a in BLUEPRINT], dtype=float)
    p = p / max(p.sum(), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(BLUEPRINT)))
    out = {"arm": name, "n": n, "areas_covered": len(hit),
           "areas_per_item": float(np.mean([len(c) for c in cov])),
           "blueprint_entropy": ent,
           "top_term": top[0], "top_term_df": top[1] / n,
           "area_counts": {a: hit.get(a, 0) / n for a in BLUEPRINT}}
    print(f"\n=== {name} (n={n}) ===")
    print(f"  blueprint areas covered : {out['areas_covered']}/10")
    print(f"  blueprint entropy       : {ent:.3f}   (1.0 = uniform over 10 areas)")
    print(f"  most common content word: '{top[0]}' in {top[1]/n:.0%} of items")
    print("  " + "  ".join(f"{a}:{hit.get(a,0)/n:.0%}" for a in BLUEPRINT))
    return out


def main(n: int) -> None:
    arms = {"base (7 architectural axes)": ARCH_AXES,
            "content (+3 recovered axes)": {**ARCH_AXES, **CONTENT_AXES}}
    results, store = [], {}
    for name, axes in arms.items():
        specs = build(axes, n, SEED)
        print(f"\ngenerating {name}: {len(specs)} items, "
              f"{len(axes)} axes/item", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(gen, specs))
        texts = [t for t in outs if t]
        print(f"  {len(texts)}/{len(specs)} generated", flush=True)
        store[name] = [{"spec": s, "text": t} for s, t in zip(specs, outs) if t]
        results.append(report(name, texts))

    with open(HERE / "item_content_arms.jsonl", "w") as fh:
        for name, rows in store.items():
            for r in rows:
                fh.write(json.dumps({"arm": name, **r}) + "\n")
    json.dump(results, open(HERE / "item_content_axes.json", "w"), indent=2)
    b, c = results
    print(f"\n{'metric':<28}{'base':>10}{'content':>10}{'delta':>10}")
    for k, lab in (("areas_covered", "blueprint areas /10"),
                   ("areas_per_item", "areas per item"),
                   ("blueprint_entropy", "blueprint entropy"),
                   ("top_term_df", "top-term doc freq")):
        print(f"{lab:<28}{b[k]:>10.3f}{c[k]:>10.3f}{c[k]-b[k]:>+10.3f}")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
