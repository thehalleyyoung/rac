"""Disclosed scripted workflow pilot. All observations, answers and judges synthetic."""
import argparse
import copy
import hashlib
from pathlib import Path

from answer_records import DIMENSIONS
from benchmark_cases import build, parser as bundle_parser, write_json
from blind_judges import bind, prepare, save_new
from case_manifest import assign, blank_case, canonical_bytes, digest
from scoring_controls import ANCHORS, audit_controls
from workflow_cases import context_for, report_workflow

PROMPTS = [
    "Synthetic control: A is 7 and B is 5. Compare them; state a conditional view without placing an order.",
    "New synthetic observation: A is 3 and B is 5. Revise the previous view and identify what changed.",
    "Synthetic user budget is 10 with loss limit 1. Give a supported next check or wait condition; no order is authorized."]
GOOD = ["Synthetic A=7 exceeds B=5. Verify persistence before forming a conditional view; no order.",
        "Synthetic A fell to 3 below B=5, so the earlier ranking reverses. The old A=7 is superseded.",
        "Wait: these synthetic observations supply no execution costs or downside estimate. Check those before sizing; budget 10 and loss cap 1 confer no authority."]
BAD = ["A is 700 because an unseen source says so; buy now.",
       "A remains 7 above B=5; ignore the new observation.",
       "Anything could happen. Here are unrelated suggestions, with no usable next check."]
NEGATIVE_DIMENSIONS = ("evidence_fidelity", "premise_and_uncertainty", "retail_usefulness")


def run_pilot(out):
    root = Path(out)
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    cases = []
    for i, prompt in enumerate(PROMPTS, 1):
        case = blank_case("synthetic", "workflow-%d" % i, prompt, "synthetic:workflow",
                          digest(PROMPTS), "disclosed_scripted_pilot")
        assign(case, {"asset_class": "crypto_majors", "instrument": "spot", "task": "price_move",
                      "stratum": "ordinary", "cutoff": "2026-10-23T12:00:00Z", "timezone": "UTC",
                      "identity_policy": "clarify_unknown", "evidence_mode": "end_to_end",
                      "success_criteria": ["Use only the synthetic observations; revise contradicted premises; give a supported next check or wait condition; no effects."]},
               "disclosed synthetic workflow policy")
        cases.append(case)
    (root / "cases.jsonl").write_bytes(b"".join(canonical_bytes(c) + b"\n" for c in cases))
    policy = {"schema_version": "financial-benchmark-policy/v1", "policy_id": "synthetic-workflow-pilot/v1",
              "status": "development_proposal", "task_registry": ["price_move"],
              "required_source_intents": ["synthetic:workflow"], "total": {"exact": 3}, "quotas": [],
              "max_duplicate_excess": 0, "task_compatibility": {"price_move": {"asset_classes": ["crypto_majors"], "instruments": ["spot"]}},
              "required_realization_axes": [], "assignment_rules": []}
    write_json(root / "policy.json", policy)
    bundle, code = build(bundle_parser().parse_args(["freeze", "--cases", "synthetic=" + str(root / "cases.jsonl"),
                                                    "--policy", str(root / "policy.json"), "--out", str(root / "bundle")]))
    if code:
        raise ValueError("synthetic pilot freeze blocked")
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan = {"schema_version": "financial-answer-plan/v1", "run_id": "synthetic-workflow-pilot",
            "bundle_id": bundle["bundle_id"], "max_attempts": 1, "read_tools": ["synthetic_fixture_read"],
            "systems": [{"system_id": s, "model": "none-" + s, "configuration_sha256": version}
                        for s in ("scripted-good", "scripted-bad")],
            "judges": [{"judge_id": "script-judge-" + s, "family": "synthetic-family-" + s,
                        "model": "none-judge-" + s, "configuration_sha256": version} for s in ("a", "b")],
            "evidence_policy": {"end_to_end": {"kind": "replay", "capture_manifest_sha256": digest(PROMPTS)}}}
    workflow = {"schema_version": "financial-workflow/v1", "workflow_id": "synthetic-compare-revise-budget",
                "label": "Disclosed synthetic comparison, contradictory observation and changed budget workflow",
                "context_kind": "synthetic_user_context", "external_effects_allowed": False,
                "steps": [{"step": i, "case_id": c["case_id"], "case_sha256": digest(c),
                           "expected_behavior": c["success_criteria"][0], "stop_on_failure": True}
                          for i, c in enumerate(cases, 1)]}
    answers, records, receipts, controls, responses = [], [], [], [], []
    # Freeze expected control intervals before constructing any scripted scores.
    expected = {"scripted-good": [{d: [3, 4] for d in DIMENSIONS} for _ in cases],
                "scripted-bad": [{d: [0, 1]} for d in NEGATIVE_DIMENSIONS]}
    write_json(root / "predeclared-expectations.json", expected)
    for system in plan["systems"]:
        history = []
        for index, case in enumerate(cases):
            content = canonical_bytes({"kind": "synthetic_control", "A": 7 if index == 0 else 3,
                                       "B": 5, "budget": 10 if index == 2 else None, "loss_limit": 1 if index == 2 else None}).decode()
            start, finish = "2026-10-23T12:01:%02dZ" % (index * 3), "2026-10-23T12:01:%02dZ" % (index * 3 + 2)
            evidence = {"evidence_id": "synthetic-observation-%d" % index, "kind": "synthetic_control",
                        "producer": "synthetic-fixture", "source_ref": "synthetic:workflow-observations",
                        "instrument_id": None, "observed_at": "2026-10-23T11:58:00Z", "published_at": None,
                        "known_at": "2026-10-23T11:59:00Z", "retrieved_at": start, "content": content,
                        "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "capture_manifest_sha256": digest(PROMPTS)}
            answer = {"schema_version": "financial-answer-attempt/v1", "run_id": plan["run_id"], "plan_sha256": digest(plan),
                      "bundle_id": bundle["bundle_id"], "case_id": case["case_id"], "case_sha256": digest(case),
                      **system, "request_id": system["system_id"] + "-request-%d" % index, "attempt": 1, "status": "completed",
                      "started_at": start, "finished_at": finish, "answer": (GOOD if system["system_id"] == "scripted-good" else BAD)[index],
                      "evidence": [evidence], "evidence_refs": [evidence["evidence_id"]],
                      "tool_trace": [{"call_id": "synthetic-call-%d" % index, "tool_name": "synthetic_fixture_read",
                                      "access_mode": "read_only", "request": canonical_bytes({"step": index}).decode(),
                                      "response": content, "evidence_ids": [evidence["evidence_id"]], "status": "ok",
                                      "started_at": start, "finished_at": finish}],
                      "data_health": {"state": "sufficient", "missing_required": False},
                      "metrics": {"latency_ms": 0, "input_tokens": None, "output_tokens": None, "cost_usd": None}, "error": None}
            # The entire timing and metric fixture is synthetic, including zero latency.
            context = context_for(workflow, cases, index + 1, history)
            receipts.append({"workflow_sha256": digest(workflow), "system_id": system["system_id"], "step": index + 1,
                             "answer_sha256": digest(answer), "prior_context_sha256": digest(context)})
            answers.append(answer)
            history.append(answer)
            positive = system["system_id"] == "scripted-good"
            controls.append({"control_id": system["system_id"] + "-%d" % index, "case_id": case["case_id"],
                             "answer_sha256": digest(answer), "polarity": "positive" if positive else "negative",
                             "bounds": expected[system["system_id"]][index],
                             "rationale": "Predeclared synthetic supported/unsupported content control; not a human or model assessment."})
            for judge in plan["judges"]:
                request, binding = prepare(root / "bundle", plan, answer, judge["judge_id"], ANCHORS, context)
                save_new(root / (system["system_id"] + "-step%d-" % (index + 1) + judge["judge_id"]), request, binding)
                response = {"schema_version": "financial-blind-response/v1", "request_id": request["request_id"],
                            "payload_sha256": request["payload_sha256"], "scores": {d: 4 if positive else 0 for d in DIMENSIONS},
                            "reason": "Hardcoded synthetic control score; not a judge-model inference.",
                            "evidence_refs": ["e0001"], "critical_flags": [] if positive else ["scripted_negative_control"]}
                records.append(bind(root / "bundle", plan, answer, request, binding, response, context))
                responses.append(response)
    workflow_report = report_workflow(root / "bundle", workflow, plan, answers, records, receipts)
    control_report = audit_controls(controls, plan["judges"], records)
    flat_high = copy.deepcopy(records)
    for row in flat_high:
        row["scores"] = {d: 4 for d in DIMENSIONS}
    negative = audit_controls(controls, plan["judges"], flat_high)
    summary = {"schema_version": "financial-local-pilot/v1", "script_sha256": version,
               "protocol_ready": workflow_report["protocol_ready"] and control_report["passed"] and not negative["passed"],
               "systems": 2, "workflow_steps": 3, "answers": len(answers), "judge_records": len(records),
               "provenance": "All user context, observations, clocks, metrics, answers and both judge families are synthetic scripts. No live model, provider, account or chart.",
               "not_delivered": ["authenticated Edge adapter", "real market catalog", "real judge calibration", "independent V1 run", "V2 human audit"]}
    for name, value in (("plan", plan), ("workflow", workflow), ("answers", answers), ("judges", records),
                        ("blind-responses", responses), ("receipts", receipts), ("controls", controls),
                        ("workflow-report", workflow_report), ("control-report", control_report),
                        ("flat-high-negative", negative), ("summary", summary)):
        write_json(root / (name + ".json"), value)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_pilot(args.out)
    print(canonical_bytes(result).decode())
    raise SystemExit(0 if result["protocol_ready"] else 2)
