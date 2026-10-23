"""Lossless adapters for development questions; no provider/model calls."""
import copy

from case_manifest import ContractError, DOMAIN_ASSETS, assign, blank_case, digest


def require_rows(rows):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ContractError("input must be an array of objects")
    ids = [r.get("id") for r in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ContractError("input record IDs must be nonempty and unique")


def reference_cases(rows, config, namespace, sha):
    require_rows(rows)
    cells = {r["source_id"]: r for r in config["reference_cells"]}
    result = []
    for row in rows:
        case = blank_case(namespace, row["id"], row.get("prompt"),
                          namespace + ":" + row["id"], sha, "reference_questions")
        cell = cells.get(row["id"])
        if cell is None or row["prompt"] != cell["reference_prompt"]:
            raise ContractError("reference intent/prompt differs from the pinned registry: " + row["id"])
        category = DOMAIN_ASSETS.get(row.get("category"))
        if category != DOMAIN_ASSETS.get(cell["category"]):
            raise ContractError("reference category conflicts with registry: " + row["id"])
        assign(case, {"asset_class": category, "task": cell["task"]},
               "reference_registry:" + digest(config))
        # The supplied category says spot; this is a design label, not a listing.
        if row.get("category") == "onchain spot":
            assign(case, {"instrument": "spot"}, "source_category")
        case["source_contract"] = {"raw_source_record": copy.deepcopy(row)}
        result.append(case)
    return result


def rq01_cases(provenance, datasets, namespace, hashes):
    if not isinstance(provenance, dict) or not isinstance(provenance.get("rows"), list):
        raise ContractError("RQ01 provenance requires rows")
    result, seen = [], set()
    for link in provenance["rows"]:
        if not isinstance(link, dict):
            raise ContractError("invalid RQ01 provenance row")
        record_id, dataset = link.get("caseId"), link.get("dataset")
        if record_id in seen:
            raise ContractError("duplicate RQ01 case ID")
        seen.add(record_id)
        rows = datasets.get(dataset)
        if rows is None:
            raise ContractError("RQ01 references a dataset not supplied")
        require_rows(rows)
        row = next((r for r in rows if r["id"] == record_id), None)
        if row is None or row.get("userPrompt") != link.get("prompt"):
            raise ContractError("RQ01 case missing or exact prompt mismatch: " + str(record_id))
        family = link.get("sourceFamily")
        if not isinstance(family, str) or not family:
            raise ContractError("RQ01 source family missing")
        case = blank_case(namespace, record_id, row["userPrompt"],
                          namespace + ":" + family, hashes["provenance"], "rq01_eval")
        case["provenance"].append({"source_name": namespace,
                                    "source_record_id": record_id,
                                    "source_format": dataset,
                                    "source_sha256": hashes[dataset]})
        assign(case, {"success_criteria": copy.deepcopy(row.get("gradingCriteria", []))},
               "source_eval_criteria")
        case["source_contract"] = {"eval_row": copy.deepcopy(row),
                                     "rq01_link": copy.deepcopy(link)}
        # No ticker-derived asset/instrument/task/stratum guesses for legacy rows.
        result.append(case)
    return result


def generated_cases(rows, namespace, sha):
    require_rows(rows)
    result = []
    for row in rows:
        if row.get("decision", "selected") != "selected":
            raise ContractError("adapter takes selected output, not silently filtered candidates")
        spec = row.get("requested_spec")
        if not isinstance(spec, dict):
            raise ContractError("generated question requires its requested_spec")
        source_id = spec.get("source_id") or row.get("source_id")
        family = namespace + ":" + (source_id or spec.get("id", ""))
        if family == namespace + ":":
            raise ContractError("generated question has no source family or spec ID")
        case = blank_case(namespace, row["id"], row.get("prompt"), family, sha, "generator_selected")
        fields = {k: copy.deepcopy(spec[k]) for k in
                  ("asset_class", "instrument", "task", "stratum", "challenge_tags",
                   "premise_mode", "success_criteria", "evidence_mode") if k in spec}
        if "asset_class" not in fields and spec.get("domain") in DOMAIN_ASSETS:
            fields["asset_class"] = DOMAIN_ASSETS[spec["domain"]]
        if "as_of" in spec:
            fields["cutoff"] = spec["as_of"]
        assign(case, fields, "generator_requested_spec")
        labels = copy.deepcopy(row.get("label", {}))
        if not isinstance(labels, dict):
            raise ContractError("generator label must be an object")
        if row.get("realized_behavior") is not None:
            labels["realized_behavior"] = row["realized_behavior"]
        if labels:
            case["inferred_labels"].append({"producer": "source_generator_classifier",
                                             "status": "unvalidated_model_inference",
                                             "labels": labels})
        case["source_contract"] = {"requested_spec": copy.deepcopy(spec),
                                     "raw_source_record": copy.deepcopy(row)}
        result.append(case)
    return result
