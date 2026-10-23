"""Portable benchmark inputs. Design assignments are not market observations."""
import copy
import datetime as dt
import hashlib
import json
import re
import unicodedata

VERSION = "financial-benchmark-case/v1"
ASSETS = {"equities", "commodities", "crypto_majors", "crypto_onchain", "mixed"}
INSTRUMENTS = {"spot", "perps", "options", "prediction_markets", "mixed"}
STRATA = {"ordinary", "challenge"}
IDENTITIES = {"unresolved", "ambiguous", "resolved", "unavailable"}
IDENTITY_POLICIES = {"resolve_at_run", "provided", "clarify_unknown"}
MODES = {"controlled_evidence", "end_to_end"}
HEALTH = {"sufficient", "partial", "stale", "conflicting", "unavailable"}
ASSIGNABLE = {
    "asset_class", "instrument", "task", "stratum", "challenge_tags", "cutoff",
    "timezone", "identity_status", "identity_policy", "premise_mode",
    "premise_truth_status", "success_criteria", "evidence_mode",
    "data_health_expectation", "family_id",
}
REQUIRED_FOR_FREEZE = {
    "asset_class", "instrument", "task", "stratum", "cutoff", "timezone",
    "identity_policy", "success_criteria", "evidence_mode",
}
DOMAIN_ASSETS = {"onchain_spot": "crypto_onchain", "onchain spot": "crypto_onchain",
                 "crypto majors": "crypto_majors", "crypto_majors": "crypto_majors",
                 "commodities": "commodities", "equities": "equities"}


class ContractError(ValueError):
    pass


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(text):
    def nonfinite(value):
        raise ContractError("nonfinite JSON number: " + value)
    return json.loads(text, object_pairs_hook=_object, parse_constant=nonfinite)


def timestamp(value):
    if not isinstance(value, str):
        raise ContractError("timestamp must be a timezone-aware ISO string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("invalid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError("timestamp must include an offset")
    return parsed


def normalized_prompt(text):
    """Exact normalized text matching, not a semantic equivalence detector."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def blank_case(namespace, record_id, prompt, family_id, source_sha256, source_format):
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", namespace):
        raise ContractError("source namespace must be a stable lowercase identifier")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ContractError("source record ID must be a nonempty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ContractError("source prompt must be a nonempty string")
    return {
        "schema_version": VERSION,
        "case_id": namespace + ":" + record_id,
        "source_id": record_id,
        "family_id": family_id,
        "prompt": prompt,
        "asset_class": None, "instrument": None, "task": None,
        "stratum": None, "challenge_tags": [], "cutoff": None,
        "timezone": None, "identity_status": "unresolved", "identity_policy": None,
        "premise_mode": None, "premise_truth_status": "unknown",
        "success_criteria": [], "evidence_mode": None,
        "data_health_expectation": None,
        "contract_context": {"availability": "unverified", "instrument_id": None,
                             "venue": None, "expiry": None, "strike": None,
                             "resolution_rules": None},
        "external_effects_allowed": False,
        "assigned_requirements": {}, "assignment_history": [],
        "inferred_labels": [], "verified_evidence": [],
        "provenance": [{"source_name": namespace, "source_record_id": record_id,
                        "source_format": source_format, "source_sha256": source_sha256}],
        "source_contract": {},
    }


def assign(case, fields, origin):
    unknown = set(fields) - ASSIGNABLE
    if unknown:
        raise ContractError("cannot assign protected/unknown fields: " + ", ".join(sorted(unknown)))
    if not isinstance(origin, str) or not origin.strip():
        raise ContractError("assignments need a named origin")
    for name, value in fields.items():
        if case[name] != value or name not in case["assigned_requirements"]:
            case["assignment_history"].append({"field": name,
                                               "previous": copy.deepcopy(case[name]),
                                               "assigned": copy.deepcopy(value),
                                               "origin": origin})
        case[name] = copy.deepcopy(value)
        case["assigned_requirements"][name] = origin


def apply_annotations(cases, document):
    if set(document) != {"version", "owner", "assignments"} or document["version"] != 1:
        raise ContractError("annotation document requires version=1, owner, assignments")
    if not isinstance(document["owner"], str) or not document["owner"].strip():
        raise ContractError("annotation document requires an accountable owner")
    if not isinstance(document["assignments"], dict):
        raise ContractError("assignments must be an object keyed by case ID")
    by_id = {case["case_id"]: case for case in cases}
    unknown = set(document["assignments"]) - set(by_id)
    if unknown:
        raise ContractError("annotation refers to missing cases: " + ", ".join(sorted(unknown)))
    origin = "design_assignment:" + document["owner"] + ":" + digest(document)
    result = copy.deepcopy(cases)
    by_id = {case["case_id"]: case for case in result}
    for case_id, fields in document["assignments"].items():
        if not isinstance(fields, dict):
            raise ContractError("per-case assignments must be an object")
        assign(by_id[case_id], fields, origin)
    return result


def validate_case(case, freeze=False, tasks=None):
    """Return structural/design issues; this cannot establish financial truth."""
    issues = []

    def issue(code, field):
        issues.append({"code": code, "field": field})

    if not isinstance(case, dict):
        return [{"code": "case_is_not_an_object", "field": "case"}]
    known = set(blank_case("schema", "schema", "Schema?", "schema:family", "a" * 64, "schema"))
    for field in sorted(set(case) - known):
        issue("unknown_case_field", field)
    for field in sorted(known - set(case)):
        issue("missing_case_field", field)
    if case.get("schema_version") != VERSION:
        issue("unsupported_version", "schema_version")
    for field in ("case_id", "source_id", "family_id", "prompt"):
        if not isinstance(case.get(field), str) or not case[field].strip():
            issue("missing_or_invalid_identity", field)
    if case.get("external_effects_allowed") is not False:
        issue("external_effect_authority_forbidden", "external_effects_allowed")
    enums = {"asset_class": ASSETS, "instrument": INSTRUMENTS, "stratum": STRATA,
             "identity_status": IDENTITIES, "identity_policy": IDENTITY_POLICIES,
             "evidence_mode": MODES, "data_health_expectation": HEALTH,
             "premise_truth_status": {"unknown", "supported", "contradicted", "hypothetical"}}
    for field, values in enums.items():
        if case.get(field) is not None and (not isinstance(case[field], str) or case[field] not in values):
            issue("invalid_enum", field)
    if case.get("task") is not None:
        if not isinstance(case["task"], str) or not case["task"].strip():
            issue("invalid_task", "task")
        elif tasks is not None and case["task"] not in tasks:
            issue("unregistered_task", "task")
    for field in ("success_criteria", "challenge_tags"):
        value = case.get(field)
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
            issue("invalid_string_list", field)
    if case.get("stratum") == "ordinary" and case.get("challenge_tags"):
        issue("ordinary_case_has_challenge_tags", "challenge_tags")
    if case.get("stratum") == "challenge" and not case.get("challenge_tags"):
        issue("challenge_case_needs_tags", "challenge_tags")
    for field in ("cutoff",):
        if case.get(field) is not None:
            try:
                timestamp(case[field])
            except ContractError:
                issue("invalid_timestamp", field)
    if case.get("timezone") is not None:
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(case["timezone"])
        except (KeyError, TypeError, ValueError):
            issue("invalid_timezone", "timezone")
    if freeze:
        for field in sorted(REQUIRED_FOR_FREEZE):
            if case.get(field) in (None, "", []):
                issue("missing_freeze_requirement", field)
    provenance = case.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        issue("missing_provenance", "provenance")
    else:
        for row in provenance:
            if not isinstance(row, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_sha256", ""))):
                issue("invalid_source_hash", "provenance")
    if not isinstance(case.get("assigned_requirements"), dict):
        issue("missing_assignment_provenance", "assigned_requirements")
    elif freeze:
        for field in REQUIRED_FOR_FREEZE:
            if not case["assigned_requirements"].get(field):
                issue("missing_assignment_provenance", field)
    if not isinstance(case.get("assignment_history"), list):
        issue("invalid_assignment_history", "assignment_history")
    labels = case.get("inferred_labels")
    if not isinstance(labels, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("labels"), dict)
        or not isinstance(row.get("producer"), str)
        or row.get("status") != "unvalidated_model_inference" for row in labels
    ):
        issue("invalid_inferred_labels", "inferred_labels")
    context = case.get("contract_context", {})
    if not isinstance(context, dict):
        issue("invalid_contract_context", "contract_context")
        context = {}
    if context.get("availability") not in ("unverified", "verified", "unavailable"):
        issue("invalid_contract_context", "contract_context")
    elif context["availability"] != "verified" and any(
        context.get(k) is not None for k in ("instrument_id", "venue", "expiry", "strike", "resolution_rules")
    ):
        issue("unverified_contract_asserts_listing", "contract_context")
    if case.get("identity_status") == "resolved" and (context.get("availability") != "verified" or not context.get("instrument_id")):
        issue("resolved_identity_needs_contract", "identity_status")
    if case.get("identity_policy") == "provided" and case.get("identity_status") != "resolved":
        issue("provided_identity_is_not_resolved", "identity_policy")
    evidence = case.get("verified_evidence", [])
    if not isinstance(evidence, list):
        issue("invalid_evidence", "verified_evidence")
        evidence = []
    for item in evidence:
        if not isinstance(item, dict):
            issue("invalid_evidence", "verified_evidence")
            continue
        if item.get("kind") not in {"provider_capture", "official_document", "calculation"}:
            issue("unsupported_evidence_kind", "verified_evidence")
        for field in ("evidence_id", "source_ref", "producer"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                issue("missing_evidence_provenance", field)
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("content_sha256", ""))):
            issue("invalid_evidence_hash", "verified_evidence")
        for field in ("observed_at", "published_at", "retrieved_at", "known_at"):
            if field not in item:
                issue("missing_evidence_time_field", field)
            elif item[field] is not None:
                try:
                    time = timestamp(item[field])
                    if field in {"observed_at", "published_at", "known_at"} and case.get("cutoff") is not None and time > timestamp(case["cutoff"]):
                        issue("evidence_after_cutoff", field)
                except ContractError:
                    issue("invalid_evidence_time", field)
        if freeze and (not item.get("known_at") or not item.get("retrieved_at")):
            issue("missing_evidence_eligibility_times", "verified_evidence")
    if (context.get("availability") == "verified" or case.get("premise_truth_status") in {"supported", "contradicted"}) and not evidence:
        issue("asserted_fact_without_evidence", "verified_evidence")
    return issues
