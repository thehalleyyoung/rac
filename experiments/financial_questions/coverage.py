"""Coverage and declared-policy checks, without market-capability filtering."""
import collections
import copy

from case_manifest import (ASSETS, ASSIGNABLE, INSTRUMENTS, STRATA, ContractError,
                           DOMAIN_ASSETS, assign, digest, normalized_prompt, validate_case)

POLICY_VERSION = "financial-benchmark-policy/v1"
AXES = ("asset_class", "instrument", "task", "stratum", "evidence_mode")
FILTERS = set(AXES) | {"family_id", "source_name", "premise_mode"}
UNKNOWNS = "<unknown>"


def validate_policy(policy):
    required = {"schema_version", "policy_id", "status", "task_registry",
                "required_source_intents", "total", "quotas", "max_duplicate_excess",
                "task_compatibility", "required_realization_axes", "assignment_rules"}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ContractError("policy fields do not match the versioned contract")
    if policy["schema_version"] != POLICY_VERSION:
        raise ContractError("unsupported policy schema")
    if not isinstance(policy["policy_id"], str) or not policy["policy_id"].strip():
        raise ContractError("policy_id is required")
    if policy["status"] not in ("development_proposal", "owner_selected"):
        raise ContractError("declare development_proposal or owner_selected policy status")
    for key in ("task_registry", "required_source_intents", "required_realization_axes"):
        rows = policy[key]
        if not isinstance(rows, list) or any(not isinstance(x, str) or not x.strip() for x in rows) or len(rows) != len(set(rows)):
            raise ContractError("policy " + key + " must contain unique strings")
    if not policy["task_registry"]:
        raise ContractError("policy task registry is empty")
    if not set(policy["required_realization_axes"]) <= set(AXES):
        raise ContractError("unsupported realization axis")
    if type(policy["max_duplicate_excess"]) is not int or policy["max_duplicate_excess"] < 0:
        raise ContractError("invalid duplicate allowance")
    if not isinstance(policy["quotas"], list):
        raise ContractError("quotas must be an array")

    def bounds(row, total=False):
        if not isinstance(row, dict) or set(row) - {"where", "min", "max", "exact"}:
            raise ContractError("unknown quota field")
        if total and "where" in row:
            raise ContractError("total cannot have a conditional filter")
        if not total and ("where" not in row or not isinstance(row["where"], dict) or not row["where"]):
            raise ContractError("conditional quota needs a nonempty where")
        if "exact" in row and ("min" in row or "max" in row):
            raise ContractError("exact quota cannot also specify min/max")
        values = {k: v for k, v in row.items() if k != "where"}
        if not values or any(type(v) is not int or v < 0 for v in values.values()):
            raise ContractError("quota bounds must be nonnegative integers")
        if row.get("min", 0) > row.get("max", float("inf")):
            raise ContractError("inverted quota bounds")
        for key, value in row.get("where", {}).items():
            if key not in FILTERS or not isinstance(value, str) or not value.strip():
                raise ContractError("unknown quota axis or invalid value")
            registry = {"asset_class": ASSETS, "instrument": INSTRUMENTS,
                        "stratum": STRATA, "task": set(policy["task_registry"])}
            if key in registry and value not in registry[key]:
                raise ContractError("quota value outside registry: " + key)

    bounds(policy["total"], total=True)
    for row in policy["quotas"]:
        bounds(row)
    if not isinstance(policy["assignment_rules"], list):
        raise ContractError("assignment_rules must be an array")
    for rule in policy["assignment_rules"]:
        if not isinstance(rule, dict) or set(rule) != {"where", "set", "only_missing"}:
            raise ContractError("assignment rule requires where, set, only_missing")
        if not isinstance(rule["where"], dict) or set(rule["where"]) - FILTERS:
            raise ContractError("unknown assignment filter")
        if any(not isinstance(v, str) or not v for v in rule["where"].values()):
            raise ContractError("assignment filter values must be strings")
        if not isinstance(rule["set"], dict) or not rule["set"] or set(rule["set"]) - ASSIGNABLE:
            raise ContractError("assignment rule sets protected or unknown fields")
        if type(rule["only_missing"]) is not bool:
            raise ContractError("assignment rule only_missing must be boolean")
    compatibility = policy["task_compatibility"]
    if not isinstance(compatibility, dict) or set(compatibility) - set(policy["task_registry"]):
        raise ContractError("invalid task compatibility registry")
    for task, row in compatibility.items():
        if not isinstance(row, dict) or set(row) != {"asset_classes", "instruments"}:
            raise ContractError("compatibility needs asset_classes and instruments")
        for key, registry in [("asset_classes", ASSETS), ("instruments", INSTRUMENTS)]:
            values = row[key]
            if not isinstance(values, list) or any(not isinstance(x, str) or x not in registry for x in values):
                raise ContractError("invalid compatibility values")


def counts(cases, axis):
    return dict(sorted(collections.Counter(
        case.get(axis) or UNKNOWNS for case in cases).items()))


def source_name(case):
    return case["provenance"][0]["source_name"]


def apply_policy_assignments(cases, policy):
    """Frozen design rules can fill generated requirements without run-time annotation."""
    validate_policy(policy)
    result = copy.deepcopy(cases)
    for index, rule in enumerate(policy["assignment_rules"]):
        for case in result:
            if matching(case, rule["where"]):
                fields = {k: v for k, v in rule["set"].items()
                          if not rule["only_missing"] or case[k] in (None, "", [])}
                assign(case, fields, "policy_rule:" + digest(policy) + ":" + str(index))
    return result


def matching(case, where):
    return all((source_name(case) if key == "source_name" else case.get(key)) == value
               for key, value in where.items())


def observed_labels(row):
    labels = dict(row)
    if "asset_class" not in labels and labels.get("domain") in DOMAIN_ASSETS:
        labels["asset_class"] = DOMAIN_ASSETS[labels["domain"]]
    if "task" not in labels and isinstance(labels.get("realized_behavior"), str):
        labels["task"] = labels["realized_behavior"].split("/", 1)[0]
    return labels


def requested_vs_realized(cases):
    result = {axis: {"comparisons": 0, "matches": 0, "mismatches": [],
                     "missing_assigned": 0, "missing_inferred": 0}
              for axis in AXES}
    for case in cases:
        for inference in case["inferred_labels"]:
            labels = observed_labels(inference["labels"])
            for axis in AXES:
                row = result[axis]
                if case.get(axis) is None:
                    row["missing_assigned"] += 1
                if labels.get(axis) is None:
                    row["missing_inferred"] += 1
                if case.get(axis) is None or labels.get(axis) is None:
                    continue
                row["comparisons"] += 1
                if case[axis] == labels[axis]:
                    row["matches"] += 1
                else:
                    row["mismatches"].append({"case_id": case["case_id"],
                                               "assigned": case[axis], "inferred": labels[axis],
                                               "producer": inference["producer"]})
    for row in result.values():
        row["match_rate"] = row["matches"] / row["comparisons"] if row["comparisons"] else None
    return result


def selection_coverage(planned, selected, candidates=None):
    """Compare the emitted selection to its original slot plan, not new quotas."""
    if not isinstance(planned, list) or any(not isinstance(x, dict) or not isinstance(x.get("id"), str) for x in planned):
        raise ContractError("generated run specs must be objects with IDs")
    planned_ids = [x["id"] for x in planned]
    if len(set(planned_ids)) != len(planned_ids):
        raise ContractError("generated run repeats a planned slot")
    selected_ids = [x.get("spec_id") or x["requested_spec"].get("id") for x in selected]
    if any(not isinstance(x, str) for x in selected_ids):
        raise ContractError("selected generator records need spec IDs")
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - set(planned_ids):
        raise ContractError("selected run duplicates or invents a planned slot")

    def specs(rows):
        result = []
        for spec in rows:
            item = dict(spec)
            item.setdefault("asset_class", DOMAIN_ASSETS.get(item.get("domain")))
            result.append(item)
        return result

    before = specs(planned)
    after = specs([x["requested_spec"] for x in selected])
    missing = sorted(set(planned_ids) - set(selected_ids))
    axes = AXES + ("source_id", "premise_mode")
    result = {"planned": len(planned), "selected": len(selected), "complete": not missing,
              "unfilled_slot_ids": missing, "planned_distribution": {},
              "selected_distribution": {}, "distribution_delta": {}}
    for axis in axes:
        p, s = counts(before, axis), counts(after, axis)
        result["planned_distribution"][axis] = p
        result["selected_distribution"][axis] = s
        result["distribution_delta"][axis] = {key: s.get(key, 0) - p.get(key, 0)
                                                 for key in sorted(set(p) | set(s))}
    if candidates is not None:
        if not isinstance(candidates, list) or any(not isinstance(x, dict) for x in candidates):
            raise ContractError("all-candidates must be an array of objects")
        result["candidate_decisions"] = dict(sorted(collections.Counter(
            x.get("decision", UNKNOWNS) for x in candidates).items()))
        result["rejection_reasons"] = dict(sorted(collections.Counter(
            reason for x in candidates for reason in x.get("rejections", [])).items()))
    return result


def coverage_report(cases, policy, selections=None):
    validate_policy(policy)
    selections = selections or {}
    structural, readiness = [], []
    for case in cases:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        for row in validate_case(case, tasks=set(policy["task_registry"])):
            structural.append(dict(row, case_id=case_id))
        for row in validate_case(case, freeze=True, tasks=set(policy["task_registry"])):
            readiness.append(dict(row, case_id=case_id))
    # Malformed structure is never made acceptable by audit mode.
    if structural:
        raise ContractError("invalid portable case contract: " + str(structural[:5]))
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ContractError("case IDs collide across inputs")
    failures = list(readiness)
    if not cases:
        failures.append({"code": "empty_batch"})
    families = counts(cases, "family_id")
    missing = sorted(set(policy["required_source_intents"]) - set(families))
    for family in missing:
        failures.append({"code": "missing_required_source_intent", "family_id": family})
    quotas = []
    for row in [dict(policy["total"], where={})] + policy["quotas"]:
        observed = sum(matching(case, row["where"]) for case in cases)
        passed = (observed == row["exact"] if "exact" in row else
                  row.get("min", 0) <= observed <= row.get("max", float("inf")))
        quotas.append(dict(row, observed=observed, passed=passed))
        if not passed:
            failures.append({"code": "quota_not_met", "quota": quotas[-1]})
    duplicates = collections.defaultdict(list)
    for case in cases:
        duplicates[normalized_prompt(case["prompt"])].append(case["case_id"])
        rule = policy["task_compatibility"].get(case["task"])
        if rule:
            for field, allowed in [("asset_class", rule["asset_classes"]),
                                   ("instrument", rule["instruments"])]:
                if case[field] is not None and case[field] not in allowed:
                    failures.append({"code": "incompatible_task_axis", "case_id": case["case_id"],
                                     "field": field, "value": case[field]})
        for axis in policy["required_realization_axes"]:
            labels = [observed_labels(row["labels"]) for row in case["inferred_labels"]]
            generated = any(p["source_format"] == "generator_selected" for p in case["provenance"])
            # Absence of classifier output is not a successful realization check.
            if (generated or labels) and (not labels or any(
                row.get(axis) is None or row.get(axis) != case.get(axis) for row in labels
            )):
                failures.append({"code": "required_realization_mismatch", "case_id": case["case_id"], "axis": axis})
    duplicate_groups = [{"normalized_prompt_sha256": digest(text), "case_ids": group}
                        for text, group in sorted(duplicates.items()) if len(group) > 1]
    duplicate_excess = sum(len(g["case_ids"]) - 1 for g in duplicate_groups)
    if duplicate_excess > policy["max_duplicate_excess"]:
        failures.append({"code": "duplicate_allowance_exceeded", "observed": duplicate_excess})
    for name, selection in selections.items():
        if not selection["complete"]:
            failures.append({"code": "upstream_selection_incomplete", "source_name": name,
                             "missing": selection["unfilled_slot_ids"]})
    axes = {axis: counts(cases, axis) for axis in AXES}
    for axis, values in [("asset_class", ASSETS), ("instrument", INSTRUMENTS), ("stratum", STRATA)]:
        axes[axis] = {key: axes[axis].get(key, 0) for key in sorted(set(axes[axis]) | values | {UNKNOWNS})}
    matrix = collections.Counter((case["asset_class"] or UNKNOWNS,
                                  case["instrument"] or UNKNOWNS) for case in cases)
    return {
        "schema_version": "financial-benchmark-coverage/v1", "policy_id": policy["policy_id"],
        "policy_status": policy["status"], "case_count": len(cases),
        "counts": axes, "by_source": dict(sorted(collections.Counter(source_name(c) for c in cases).items())),
        "by_family": families, "family_count": len(families),
        "asset_instrument": [{"asset_class": a, "instrument": i, "count": n}
                             for (a, i), n in sorted(matrix.items())],
        "missing_source_intents": missing, "quotas": quotas,
        "identity_status": counts(cases, "identity_status"),
        "unresolved_identity_case_ids": [c["case_id"] for c in cases if c["identity_status"] != "resolved"],
        "contract_availability": dict(sorted(collections.Counter(c["contract_context"]["availability"] for c in cases).items())),
        "exact_normalized_duplicates": duplicate_groups, "duplicate_excess": duplicate_excess,
        "semantic_duplicates": "not inferred; explicit family provenance is reported separately",
        "requested_vs_realized": requested_vs_realized(cases),
        "inference_coverage": {"with_labels": sum(bool(c["inferred_labels"]) for c in cases),
                               "without_labels": sum(not c["inferred_labels"] for c in cases)},
        "generator_selections": selections, "freeze_ready": not failures,
        "freeze_failures": failures,
        "interpretation": "Design and artifact checks only; labels are not financial truth, actual listings, GTE coverage or judge validity.",
    }


def coverage_markdown(report):
    lines = ["# Case coverage audit", "", "Cases: " + str(report["case_count"]),
             "Policy: `" + report["policy_id"] + "` (" + report["policy_status"] + ")",
             "Freeze ready: " + str(report["freeze_ready"]).lower(), ""]
    for axis, rows in report["counts"].items():
        lines += ["## " + axis, "", "| Value | Count |", "| --- | ---: |"]
        lines += ["| " + key.replace("|", "\\|") + " | " + str(n) + " |" for key, n in rows.items()]
        lines.append("")
    lines += ["## Open requirements", "",
              "Missing source intents: " + str(len(report["missing_source_intents"])),
              "Unresolved/ambiguous/unavailable identities: " + str(len(report["unresolved_identity_case_ids"])),
              "Exact normalized duplicate excess: " + str(report["duplicate_excess"]), "",
              "| Freeze failure | Count |", "| --- | ---: |"]
    reasons = collections.Counter(row["code"] for row in report["freeze_failures"])
    lines += ["| " + code + " | " + str(n) + " |" for code, n in sorted(reasons.items())]
    lines += ["", "Full case IDs, conditional quotas, family provenance, and requested/realized differences are in coverage.json.",
              "", report["interpretation"], ""]
    return "\n".join(lines)
