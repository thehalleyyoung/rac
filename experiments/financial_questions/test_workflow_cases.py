"""Synthetic ordered-workflow controls, no live models/accounts."""
import copy
from pathlib import Path
import tempfile
import unittest

from answer_records import load_bundle
from case_manifest import ContractError, digest
from test_answer_records import control, judge_control
from workflow_cases import context_for, report_workflow, validate_workflow


def workflow_control(cases):
    return {"schema_version": "financial-workflow/v1", "workflow_id": "synthetic-compare-revise",
            "label": "Synthetic compare then revise workflow", "context_kind": "synthetic_user_context",
            "external_effects_allowed": False,
            "steps": [{"step": i, "case_id": c["case_id"], "case_sha256": digest(c),
                       "expected_behavior": "Use supplied evidence; disclose uncertainty; never transact.",
                       "stop_on_failure": True} for i, c in enumerate(cases, 1)]}


def receipt_control(workflow, cases, answers):
    rows, history = [], []
    for i, answer in enumerate(answers, 1):
        context = context_for(workflow, cases, i, history)
        rows.append({"workflow_sha256": digest(workflow), "system_id": answer["system_id"], "step": i,
                     "answer_sha256": digest(answer), "prior_context_sha256": digest(context)})
        history.append(answer)
    return rows


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bundle, self.plan, self.answers, self.judges = control(Path(self.tmp.name), second=True)
        _, self.cases = load_bundle(self.bundle)
        self.answers[1].update(started_at="2026-10-23T12:01:03Z", finished_at="2026-10-23T12:01:05Z")
        call = self.answers[1]["tool_trace"][0]
        call.update(started_at="2026-10-23T12:01:03Z", finished_at="2026-10-23T12:01:04Z")
        self.judges = [judge_control(a, j, self.plan) for a in self.answers for j in self.plan["judges"]]
        self.workflow = workflow_control(self.cases)
        self.receipts = receipt_control(self.workflow, self.cases, self.answers)

    def tearDown(self):
        self.tmp.cleanup()

    def report(self):
        return report_workflow(self.bundle, self.workflow, self.plan, self.answers, self.judges, self.receipts)

    def test_two_step_positive_keeps_all_steps_and_judges(self):
        report = self.report()
        self.assertTrue(report["protocol_ready"], report["protocol_errors"])
        self.assertEqual(len(report["steps"]), 2)
        self.assertEqual(len(report["answer_report"]["answers"]), 2)

    def test_second_context_is_exact_prior_turn_without_future_or_metadata(self):
        context = context_for(self.workflow, self.cases, 2, self.answers[:1])
        self.assertEqual(context, [{"role": "user", "content": self.cases[0]["prompt"]},
                                   {"role": "assistant", "content": self.answers[0]["answer"]}])
        with self.assertRaisesRegex(ContractError, "every prior step only"):
            context_for(self.workflow, self.cases, 2, self.answers)

    def test_changed_prior_answer_cannot_reuse_context_receipt(self):
        self.answers[0]["answer"] = "Changed prior statement"
        self.assertFalse(self.report()["protocol_ready"])

    def test_reordered_missing_and_duplicate_receipts_refuse(self):
        self.receipts[1]["prior_context_sha256"] = digest([])
        self.assertFalse(self.report()["protocol_ready"])
        self.receipts.pop()
        self.assertFalse(self.report()["protocol_ready"])
        self.receipts.append(copy.deepcopy(self.receipts[0]))
        with self.assertRaisesRegex(ContractError, "duplicate workflow receipt"):
            self.report()

    def test_temporal_overlap_fails_even_with_valid_answer_records(self):
        self.answers[1]["started_at"] = "2026-10-23T12:01:01Z"
        self.receipts[1]["answer_sha256"] = digest(self.answers[1])
        self.judges = [judge_control(a, j, self.plan) for a in self.answers for j in self.plan["judges"]]
        report = self.report()
        self.assertTrue(report["answer_report"]["ready_for_pilot_interpretation"])
        self.assertFalse(report["protocol_ready"])

    def test_failure_stops_later_completed_output_and_retains_refusal(self):
        self.answers[0].update(status="timeout", answer=None, error="synthetic timeout")
        self.receipts[0]["answer_sha256"] = digest(self.answers[0])
        self.receipts[1]["prior_context_sha256"] = digest(None)
        self.judges = [judge_control(self.answers[1], j, self.plan) for j in self.plan["judges"]]
        self.assertFalse(self.report()["protocol_ready"])
        self.answers[1].update(status="refused", answer=None, error="blocked prerequisite")
        self.receipts[1]["answer_sha256"] = digest(self.answers[1])
        self.judges = []
        self.assertTrue(self.report()["protocol_ready"])
        self.assertEqual([s["status"] for s in self.report()["steps"]], ["timeout", "refused"])

    def test_authority_changed_cases_and_backward_cutoff_refuse(self):
        self.workflow["external_effects_allowed"] = True
        with self.assertRaisesRegex(ContractError, "no effect authority"):
            validate_workflow(self.workflow, self.cases)
        self.workflow["external_effects_allowed"] = False
        self.workflow["steps"][0]["case_sha256"] = "a" * 64
        with self.assertRaisesRegex(ContractError, "case changed"):
            validate_workflow(self.workflow, self.cases)


if __name__ == "__main__":
    unittest.main()
