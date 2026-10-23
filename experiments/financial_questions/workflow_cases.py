"""Ordered, read-only workflow receipts over unchanged RQ01 cases and answers."""
from answer_records import fields, load_bundle, report_run, require, text
from case_manifest import digest, timestamp


def validate_workflow(workflow, cases):
    fields(workflow, "schema_version workflow_id label context_kind external_effects_allowed steps", "workflow")
    require(workflow["schema_version"] == "financial-workflow/v1", "unsupported workflow")
    require(text(workflow["workflow_id"]) and text(workflow["label"]), "missing workflow identity")
    require(workflow["context_kind"] == "synthetic_user_context" and workflow["external_effects_allowed"] is False,
            "workflow must use synthetic user context with no effect authority")
    require(isinstance(workflow["steps"], list) and len(workflow["steps"]) >= 2, "workflow needs multiple steps")
    by_case = {c["case_id"]: c for c in cases}
    seen, previous = set(), None
    for i, step in enumerate(workflow["steps"], 1):
        fields(step, "step case_id case_sha256 expected_behavior stop_on_failure", "workflow step")
        require(type(step["step"]) is int and step["step"] == i, "workflow steps must be contiguous and ordered")
        require(step["case_id"] in by_case and step["case_id"] not in seen, "unknown/duplicate workflow case")
        seen.add(step["case_id"])
        case = by_case[step["case_id"]]
        require(step["case_sha256"] == digest(case), "workflow case changed")
        require(text(step["expected_behavior"]) and step["stop_on_failure"] is True, "workflow needs criteria and a stop rule")
        require(previous is None or timestamp(previous) <= timestamp(case["cutoff"]), "workflow cutoff goes backward")
        previous = case["cutoff"]
    require(seen == set(by_case), "workflow must account for exactly its frozen bundle")
    return by_case


def context_for(workflow, cases, step_number, history):
    """Build the prior transcript; never include a future turn or system metadata."""
    by_case = validate_workflow(workflow, cases)
    require(type(step_number) is int and 1 <= step_number <= len(workflow["steps"]), "invalid next workflow step")
    require(isinstance(history, list) and len(history) == step_number - 1, "history must contain every prior step only")
    transcript = []
    system = None
    for step, answer in zip(workflow["steps"], history):
        require(answer["case_id"] == step["case_id"] and answer["case_sha256"] == step["case_sha256"], "history case/order mismatch")
        require(answer["status"] == "completed", "stop after failed prerequisite")
        require(system is None or answer["system_id"] == system, "history crosses systems")
        system = answer["system_id"]
        transcript.extend([{"role": "user", "content": by_case[step["case_id"]]["prompt"]},
                           {"role": "assistant", "content": answer["answer"]}])
    return transcript


def report_workflow(bundle_path, workflow, plan, attempts, judgements, receipts):
    _, cases = load_bundle(bundle_path)
    validate_workflow(workflow, cases)
    base = report_run(bundle_path, plan, attempts, judgements)
    errors = list(base["protocol_errors"])
    if errors:
        return {"schema_version": "financial-workflow-report/v1", "workflow_sha256": digest(workflow),
                "protocol_ready": False, "protocol_errors": errors, "steps": [], "answer_report": base,
                "limitations": "Invalid underlying records retained; workflow admission was not reached."}
    terminal = {}
    for answer in attempts:
        key = (answer["system_id"], answer["case_id"])
        if key not in terminal or answer["attempt"] > terminal[key]["attempt"]:
            terminal[key] = answer
    receipt_map = {}
    for receipt in receipts:
        fields(receipt, "workflow_sha256 system_id step answer_sha256 prior_context_sha256", "workflow receipt")
        key = (receipt["system_id"], receipt["step"])
        require(key not in receipt_map, "duplicate workflow receipt")
        require(receipt["workflow_sha256"] == digest(workflow), "workflow receipt version mismatch")
        require(receipt["system_id"] in {s["system_id"] for s in plan["systems"]} and
                type(receipt["step"]) is int and 1 <= receipt["step"] <= len(workflow["steps"]), "unknown workflow receipt slot")
        receipt_map[key] = receipt
    outcomes = []
    for system in plan["systems"]:
        history, stopped = [], False
        for step in workflow["steps"]:
            answer = terminal.get((system["system_id"], step["case_id"]))
            receipt = receipt_map.get((system["system_id"], step["step"]))
            if answer is None or receipt is None:
                errors.append({"kind": "workflow_missing_step", "system_id": system["system_id"], "step": step["step"]})
                stopped = True
                continue
            # A stopped chain has no deliverable context; refused outcomes remain counted.
            context = None if stopped else context_for(workflow, cases, step["step"], history)
            valid = receipt["answer_sha256"] == digest(answer) and receipt["prior_context_sha256"] == digest(context)
            if history:
                valid = valid and timestamp(history[-1]["finished_at"]) <= timestamp(answer["started_at"])
            valid = valid and (not stopped or answer["status"] == "refused")
            if not valid:
                errors.append({"kind": "workflow_context_or_order", "system_id": system["system_id"], "step": step["step"]})
            stopped = stopped or answer["status"] != "completed"
            history.append(answer)
            outcomes.append({"system_id": system["system_id"], "step": step["step"], "status": answer["status"],
                             "answer_sha256": digest(answer), "after_failed_prerequisite": context is None})
    return {"schema_version": "financial-workflow-report/v1", "workflow_sha256": digest(workflow),
            "protocol_ready": not errors, "protocol_errors": errors, "steps": outcomes,
            "answer_report": base,
            "limitations": "Receipt consistency only; no proof of model context consumption, semantic correction or real user workflow execution."}
