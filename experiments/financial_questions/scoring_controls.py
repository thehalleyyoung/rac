"""Anchored rubric and predeclared control audit; no automatic financial judge."""
from answer_records import DIMENSIONS, fields, number, require, sha, text
from case_manifest import digest

ANCHORS = {
    "task_completion": ["Does not answer the requested task.", "Misses most material requirements.",
                        "Answers part of the task but misses a material requirement.",
                        "Meets material requirements with a minor omission.", "Meets every declared task criterion."],
    "evidence_fidelity": ["Fabricates or contradicts required evidence.", "Uses materially unsupported evidence.",
                          "Mixes supported claims with an unresolved material evidence gap.",
                          "Supports material claims with only minor citation defects.",
                          "Binds all material claims to eligible evidence and identifies its limits."],
    "financial_reasoning": ["Conclusion depends on a fundamental reasoning error.", "Contains a major analytical error.",
                            "Provides useful analysis but misses a material assumption or comparison.",
                            "Reasoning is sound with a minor omission.",
                            "Reasoning and comparisons follow from the supplied evidence and explicit assumptions."],
    "premise_and_uncertainty": ["Treats a contradicted or invented premise as fact.", "Ignores a material uncertainty.",
                                "Acknowledges uncertainty but leaves a material premise unchecked.",
                                "Checks material premises and qualifies the conclusion with minor omissions.",
                                "Revises contradicted premises, separates assertions from observations and states limits."],
    "retail_usefulness": ["Gives an unsupported action or unusable answer.", "Offers evasive or irrelevant next steps.",
                          "Offers a partly useful next step without a material condition.",
                          "Gives a supported next step or wait condition with a minor omission.",
                          "Gives a clear supported next step, activation/invalidation or justified wait condition; grants no effect permission."]}


def audit_controls(controls, judges, records):
    """Compare supplied judgments with frozen fixture expectations, retaining failures."""
    require(isinstance(controls, list) and bool(controls), "control roster required")
    roster = {j["judge_id"]: j for j in judges}
    require(len(roster) == len(judges) and len({j["family"].casefold().strip() for j in judges}) >= 2,
            "two declared judge families required")
    keys = set()
    for c in controls:
        fields(c, "control_id case_id answer_sha256 polarity bounds rationale", "scoring control")
        require(text(c["control_id"]) and c["control_id"] not in keys, "duplicate/missing control ID")
        keys.add(c["control_id"])
        require(text(c["case_id"]) and sha(c["answer_sha256"]) and text(c["rationale"]), "invalid control identity")
        require(c["polarity"] in {"positive", "negative"}, "unknown control polarity")
        require(isinstance(c["bounds"], dict) and bool(c["bounds"]) and set(c["bounds"]) <= set(DIMENSIONS), "invalid dimension bounds")
        for bounds in c["bounds"].values():
            require(isinstance(bounds, list) and len(bounds) == 2 and all(number(v) for v in bounds)
                    and 0 <= bounds[0] <= bounds[1] <= 4, "invalid control interval")
    require({c["polarity"] for c in controls} == {"positive", "negative"}, "positive and deliberately bad controls required")
    require(len({(c["case_id"], c["answer_sha256"]) for c in controls}) == len(controls), "duplicate control answer binding")
    declared = {(c["case_id"], c["answer_sha256"]): c for c in controls}
    results, seen, errors = [], set(), []
    for i, row in enumerate(records):
        key = (row.get("case_id"), row.get("answer_sha256"))
        judge_id = row.get("judge_id")
        if key not in declared or judge_id not in roster or (key, judge_id) in seen:
            errors.append({"record": i, "reason": "unknown/duplicate control judgment"})
            continue
        seen.add((key, judge_id))
        control = declared[key]
        scores = row.get("scores")
        valid = isinstance(scores, dict) and set(scores) == set(DIMENSIONS) and all(number(v) and v <= 4 for v in scores.values())
        valid = valid and all(row.get(k) == roster[judge_id][k] for k in ("family", "model", "configuration_sha256"))
        passed = valid and all(lo <= scores[d] <= hi for d, (lo, hi) in control["bounds"].items())
        results.append({"control_id": control["control_id"], "judge_id": judge_id, "passed": passed,
                        "scores": scores, "bounds": control["bounds"]})
    for key, c in declared.items():
        for judge_id in sorted(roster):
            if (key, judge_id) not in seen:
                errors.append({"control_id": c["control_id"], "judge_id": judge_id, "reason": "missing control judgment"})
    return {"schema_version": "financial-scoring-controls/v1", "controls_sha256": digest(controls),
            "judges_sha256": digest(judges), "passed": not errors and all(r["passed"] for r in results),
            "results": results, "errors": errors,
            "limitations": "Predeclared control discrimination only; no calibrated scale, population ranking or measured trader uplift."}
