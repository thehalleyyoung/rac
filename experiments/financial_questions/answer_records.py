"""Offline RQ01 answer/evidence handoff. No network, effects or LLM invocation."""
import argparse
import hashlib
from pathlib import Path
import re
import sys

from benchmark_cases import read_rows, verify, write_json
from case_manifest import ContractError, digest, timestamp

DIMENSIONS = ("task_completion", "evidence_fidelity", "financial_reasoning",
              "premise_and_uncertainty", "retail_usefulness")
STATUSES = {"completed", "timeout", "error", "refused"}


def require(condition, message):
    if not condition:
        raise ContractError(message)


def fields(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names.split()),
            "invalid " + label + " fields")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def number(value):
    return type(value) in (int, float) and 0 <= value < float("inf")


def load_bundle(path):
    result = verify(Path(path))
    require(result["state"] == "frozen", "answer runs require a verified frozen bundle")
    return result, read_rows(Path(path) / "cases.jsonl")


def validate_plan(plan, bundle, cases):
    fields(plan, "schema_version run_id bundle_id systems judges read_tools max_attempts evidence_policy", "run plan")
    require(plan["schema_version"] == "financial-answer-plan/v1", "unsupported run plan")
    require(text(plan["run_id"]) and plan["bundle_id"] == bundle["bundle_id"], "run/bundle identity mismatch")
    require(type(plan["max_attempts"]) is int and 1 <= plan["max_attempts"] <= 10, "invalid retry bound")
    systems, judges = {}, {}
    for row in plan["systems"]:
        fields(row, "system_id model configuration_sha256", "system")
        require(all(text(row[k]) for k in ("system_id", "model")) and sha(row["configuration_sha256"]), "invalid system version")
        require(row["system_id"] not in systems, "duplicate system")
        systems[row["system_id"]] = row
    require(bool(systems), "no systems declared")
    for row in plan["judges"]:
        fields(row, "judge_id family model configuration_sha256", "judge")
        require(all(text(row[k]) for k in ("judge_id", "family", "model")) and sha(row["configuration_sha256"]), "invalid judge version")
        require(row["judge_id"] not in judges, "duplicate judge")
        judges[row["judge_id"]] = row
    require(len({j["family"].casefold().strip() for j in judges.values()}) >= 2, "declare at least two judge families")
    require(isinstance(plan["read_tools"], list) and all(text(v) for v in plan["read_tools"])
            and len(set(plan["read_tools"])) == len(plan["read_tools"]), "invalid read-tool roster")
    policy = plan["evidence_policy"]
    modes = {c["evidence_mode"] for c in cases}
    require(isinstance(policy, dict) and set(policy) == modes, "declare exactly the scheduled evidence modes")
    for mode in sorted(modes):
        item = policy[mode]
        if mode == "controlled_evidence":
            fields(item, "packet_sha256_by_case", "controlled policy")
            require(set(item["packet_sha256_by_case"]) == {c["case_id"] for c in cases if c["evidence_mode"] == mode}, "missing controlled packet")
            require(all(sha(v) for v in item["packet_sha256_by_case"].values()), "invalid controlled packet hash")
        elif item.get("kind") == "replay":
            fields(item, "kind capture_manifest_sha256", "replay policy")
            require(sha(item["capture_manifest_sha256"]), "replay needs a capture manifest hash")
        else:
            fields(item, "kind start end", "synchronization policy")
            require(item["kind"] == "synchronized" and timestamp(item["start"]) <= timestamp(item["end"]), "invalid synchronized window")
    return systems, judges


def evidence_packet(evidence, case, plan):
    require(isinstance(evidence, list), "evidence must be a list")
    ids = set()
    cutoff = timestamp(case["cutoff"])
    policy = plan["evidence_policy"][case["evidence_mode"]]
    for item in evidence:
        fields(item, "evidence_id kind producer source_ref instrument_id observed_at published_at known_at retrieved_at content content_sha256 capture_manifest_sha256", "evidence")
        require(all(text(item[k]) for k in ("evidence_id", "producer", "source_ref")), "missing evidence identity")
        require(item["kind"] in {"provider_capture", "official_document", "calculation", "synthetic_control"}, "unknown evidence kind")
        require(item["instrument_id"] is None or text(item["instrument_id"]), "invalid evidence instrument")
        require(item["evidence_id"] not in ids, "duplicate evidence ID")
        ids.add(item["evidence_id"])
        require(isinstance(item["content"], str) and hashlib.sha256(item["content"].encode()).hexdigest() == item["content_sha256"], "evidence bytes/hash mismatch")
        known, retrieved = timestamp(item["known_at"]), timestamp(item["retrieved_at"])
        require(known <= cutoff and known <= retrieved, "evidence known after cutoff/retrieval")
        for key in ("observed_at", "published_at"):
            if item[key] is not None:
                require(timestamp(item[key]) <= cutoff and timestamp(item[key]) <= retrieved, "evidence observed/published after cutoff/retrieval")
        if case["evidence_mode"] == "end_to_end":
            if policy["kind"] == "replay":
                require(item["capture_manifest_sha256"] == policy["capture_manifest_sha256"], "capture manifest mismatch")
            else:
                require(item["observed_at"] is not None and timestamp(policy["start"]) <= timestamp(item["observed_at"]) <= timestamp(policy["end"]), "evidence outside synchronized observation window")
    if case["evidence_mode"] == "controlled_evidence":
        # Order is part of the declared packet, not silently canonicalized away.
        require(digest(evidence) == policy["packet_sha256_by_case"][case["case_id"]], "controlled evidence packet differs")
    return ids


def validate_attempt(row, case, plan, systems):
    fields(row, "schema_version run_id plan_sha256 bundle_id case_id case_sha256 system_id request_id attempt status model configuration_sha256 started_at finished_at answer evidence evidence_refs tool_trace data_health metrics error", "answer attempt")
    require(row["schema_version"] == "financial-answer-attempt/v1", "unsupported answer version")
    require(row["run_id"] == plan["run_id"] and row["plan_sha256"] == digest(plan)
            and row["bundle_id"] == plan["bundle_id"] and row["case_sha256"] == digest(case), "answer identity/plan/case mismatch")
    require(row["system_id"] in systems, "undeclared system")
    system = systems[row["system_id"]]
    require(all(row[k] == system[k] for k in ("model", "configuration_sha256")), "system version mismatch")
    require(text(row["request_id"]) and type(row["attempt"]) is int and 1 <= row["attempt"] <= plan["max_attempts"], "invalid attempt identity")
    require(row["status"] in STATUSES, "unknown operational status")
    start, end = timestamp(row["started_at"]), timestamp(row["finished_at"])
    require(start <= end, "attempt clock regression")
    if row["status"] == "completed":
        require(text(row["answer"]) and row["error"] is None, "completed answer missing text or contains operational error")
    else:
        require(row["answer"] is None and text(row["error"]), "operational failure needs an error and no fabricated answer")
    ids = evidence_packet(row["evidence"], case, plan)
    require(isinstance(row["evidence_refs"], list) and all(v in ids for v in row["evidence_refs"]), "answer references absent evidence")
    fields(row["data_health"], "state missing_required", "data health")
    require(row["data_health"]["state"] in {"sufficient", "partial", "stale", "conflicting", "unavailable"}
            and type(row["data_health"]["missing_required"]) is bool, "invalid data health")
    require(isinstance(row["tool_trace"], list), "missing tool trace")
    calls, traced = set(), set()
    for call in row["tool_trace"]:
        fields(call, "call_id tool_name access_mode request response evidence_ids status started_at finished_at", "tool trace")
        require(text(call["call_id"]) and call["call_id"] not in calls, "duplicate/missing tool call")
        calls.add(call["call_id"])
        require(call["access_mode"] == "read_only" and call["tool_name"] in plan["read_tools"], "undeclared or effect-capable tool activity")
        require(isinstance(call["request"], str) and isinstance(call["response"], str), "tool request/result bytes absent")
        require(call["status"] in {"ok", "error", "timeout"}, "invalid tool status")
        require(start <= timestamp(call["started_at"]) <= timestamp(call["finished_at"]) <= end, "tool call outside attempt")
        require(isinstance(call["evidence_ids"], list) and all(v in ids for v in call["evidence_ids"]), "tool references absent evidence")
        for evidence_id in call["evidence_ids"]:
            item = next(e for e in row["evidence"] if e["evidence_id"] == evidence_id)
            require(timestamp(item["retrieved_at"]) <= timestamp(call["finished_at"]), "evidence retrieved after tool finished")
            require(item["content"] == call["response"], "captured evidence bytes differ from tool response")
        traced.update(call["evidence_ids"])
    if case["evidence_mode"] == "end_to_end":
        require(traced == ids, "end-to-end evidence lacks an actual tool trace")
        if row["status"] == "completed":
            require(bool(calls), "completed end-to-end answer has no retrieval trace")
    require(all(timestamp(e["retrieved_at"]) <= end for e in row["evidence"]), "evidence retrieved after answer")
    require(not (row["status"] == "completed" and not ids and not row["data_health"]["missing_required"]), "empty evidence cannot claim complete data")
    fields(row["metrics"], "latency_ms input_tokens output_tokens cost_usd", "metrics")
    require(number(row["metrics"]["latency_ms"]), "invalid latency")
    for key in ("input_tokens", "output_tokens"):
        require(row["metrics"][key] is None or (type(row["metrics"][key]) is int and row["metrics"][key] >= 0), "invalid token count")
    cost = row["metrics"]["cost_usd"]
    require(cost is None or (isinstance(cost, str) and re.fullmatch(r"\d+(\.\d+)?", cost) is not None), "cost must be a nonnegative decimal string or unknown")


def validate_judge(row, answer, plan, judges, case):
    fields(row, "schema_version run_id plan_sha256 case_id system_id attempt answer_sha256 judge_id family model configuration_sha256 scores reason evidence_refs critical_flags", "judge record")
    require(row["schema_version"] == "financial-answer-judge/v1", "unsupported judge version")
    require(row["run_id"] == plan["run_id"] and row["plan_sha256"] == digest(plan)
            and row["answer_sha256"] == digest(answer), "judge answer/plan mismatch")
    require(row["judge_id"] in judges, "undeclared judge")
    require(all(row[k] == judges[row["judge_id"]][k] for k in ("family", "model", "configuration_sha256")), "judge version/family mismatch")
    require(isinstance(row["scores"], dict) and set(row["scores"]) == set(DIMENSIONS), "five content dimensions required")
    require(all(number(v) and v <= 4 for v in row["scores"].values()), "scores must be finite 0–4 numbers")
    require(text(row["reason"]), "judge rationale missing")
    ids = {e["evidence_id"] for e in answer["evidence"]}
    require(isinstance(row["evidence_refs"], list) and all(v in ids for v in row["evidence_refs"]), "judge cites absent evidence")
    require(isinstance(row["critical_flags"], list) and all(text(v) for v in row["critical_flags"]), "invalid semantic flags")
    if case["stratum"] == "ordinary" and answer["data_health"]["missing_required"]:
        require(row["scores"]["task_completion"] < 4, "missing required ordinary evidence cannot earn full completion")


def report_run(bundle_path, plan, attempts, judgements):
    bundle, cases = load_bundle(bundle_path)
    systems, judges = validate_plan(plan, bundle, cases)
    require(isinstance(attempts, list) and isinstance(judgements, list), "records must be lists")
    by_case = {c["case_id"]: c for c in cases}
    errors, rows, slots, request_ids = [], {}, {}, {}
    for index, row in enumerate(attempts):
        try:
            require(isinstance(row, dict) and row.get("case_id") in by_case, "unknown case")
            validate_attempt(row, by_case[row["case_id"]], plan, systems)
            key = (row["case_id"], row["system_id"], row["attempt"])
            require(key not in rows, "duplicate attempt")
            slot = key[:2]
            require(row["request_id"] not in request_ids or request_ids[row["request_id"]] == slot, "request ID reused across scheduled answers")
            request_ids[row["request_id"]] = slot
            rows[key] = row
            slots.setdefault(slot, []).append(row)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            errors.append({"kind": "attempt", "index": index, "message": str(exc)})
    terminal = {}
    for case in cases:
        for system_id in sorted(systems):
            slot = (case["case_id"], system_id)
            ordered = sorted(slots.get(slot, []), key=lambda r: r["attempt"])
            if not ordered:
                errors.append({"kind": "missing_slot", "case_id": slot[0], "system_id": slot[1]})
                continue
            valid = [r["attempt"] for r in ordered] == list(range(1, len(ordered) + 1))
            valid = valid and len({r["request_id"] for r in ordered}) == 1
            valid = valid and all(r["status"] != "completed" for r in ordered[:-1])
            valid = valid and all(timestamp(a["finished_at"]) <= timestamp(b["started_at"]) for a, b in zip(ordered, ordered[1:]))
            if not valid:
                errors.append({"kind": "attempt_sequence", "case_id": slot[0], "system_id": slot[1]})
            terminal[slot] = ordered[-1]
    scored, seen = {}, set()
    for index, row in enumerate(judgements):
        try:
            key = (row["case_id"], row["system_id"], row["attempt"])
            answer = rows.get(key)
            require(answer is not None and answer["status"] == "completed" and terminal[key[:2]] is answer, "judge does not address a terminal completed answer")
            validate_judge(row, answer, plan, judges, by_case[key[0]])
            judge_key = key + (row["judge_id"],)
            require(judge_key not in seen, "duplicate judge record")
            seen.add(judge_key)
            scored.setdefault(key[:2], []).append(row)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            errors.append({"kind": "judge", "index": index, "message": str(exc)})
    answers = []
    counts = {}
    for case in cases:
        for system_id in sorted(systems):
            slot = (case["case_id"], system_id)
            answer = terminal.get(slot)
            results = sorted(scored.get(slot, []), key=lambda r: r["judge_id"])
            if answer and answer["status"] == "completed" and {r["judge_id"] for r in results} != set(judges):
                errors.append({"kind": "missing_judges", "case_id": slot[0], "system_id": slot[1]})
            mode = case["evidence_mode"]
            group = counts.setdefault(mode, {}).setdefault(system_id, {"planned": 0, "terminal": {}, "attempts": {}})
            group["planned"] += 1
            status = answer["status"] if answer else "missing"
            group["terminal"][status] = group["terminal"].get(status, 0) + 1
            for r in slots.get(slot, []):
                group["attempts"][r["status"]] = group["attempts"].get(r["status"], 0) + 1
            answers.append({"case_id": slot[0], "family_id": case["family_id"], "system_id": system_id,
                            "evidence_mode": mode, "status": status,
                            "ordinary_completion_failure": bool(answer and case["stratum"] == "ordinary" and answer["data_health"]["missing_required"]),
                            "judge_records": results,
                            "disagreement_range": {d: max(r["scores"][d] for r in results) - min(r["scores"][d] for r in results) for d in DIMENSIONS} if len(results) >= 2 else None})
    return {"schema_version": "financial-answer-report/v1", "run_id": plan["run_id"],
            "bundle_id": bundle["bundle_id"], "plan_sha256": digest(plan),
            "ready_for_pilot_interpretation": not errors, "protocol_errors": errors,
            "counts_by_mode_and_system": counts, "answers": answers,
            "raw_attempt_count": len(attempts), "raw_judge_count": len(judgements),
            "semantic_flags_are_unverified": True, "scalar_score": None,
            "limits": ["Artifact consistency only; no provider/model authenticity or judge calibration.",
                       "Declared read-tool roster is not a runtime sandbox or trading authorization.",
                       "No cross-mode ranking, calibrated weights, independence or population inference."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--judges", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.out.exists(), "output already exists; preserve prior runs")
        plan, attempts, judges = read_rows(args.plan), read_rows(args.attempts), read_rows(args.judges)
        report = report_run(args.bundle, plan, attempts, judges)
        args.out.mkdir(parents=True, exist_ok=False)
        for name, value in (("plan.json", plan), ("attempts.json", attempts), ("judges.json", judges), ("report.json", report)):
            write_json(args.out / name, value)
        write_json(args.out / "receipt.json", {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                "files": {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in ("plan.json", "attempts.json", "judges.json", "report.json")}})
        print("ready" if report["ready_for_pilot_interpretation"] else "blocked")
        return 0 if report["ready_for_pilot_interpretation"] else 2
    except (ContractError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
