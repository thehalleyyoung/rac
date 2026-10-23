"""Synthetic controls for projection/binding, not judge validity."""
import copy
from pathlib import Path
import stat
import tempfile
import unittest

from answer_records import DIMENSIONS, load_bundle, report_run
from blind_judges import bind, prepare, save_new
from case_manifest import ContractError, canonical_bytes, digest
from test_answer_records import control


def rubric_control():
    return {d: ["Synthetic anchor " + str(i) for i in range(5)] for d in DIMENSIONS}


def response_control(request, score=3):
    return {"schema_version": "financial-blind-response/v1", "request_id": request["request_id"],
            "payload_sha256": request["payload_sha256"], "scores": {d: score for d in DIMENSIONS},
            "reason": "Synthetic scripted judgment, no measured model quality.",
            "evidence_refs": [e["ref"] for e in request["payload"]["evidence"]], "critical_flags": []}


class BlindJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bundle, self.plan, self.answers, _ = control(self.root)
        self.answer = self.answers[0]
        self.request, self.binding = prepare(self.bundle, self.plan, self.answer, "judge-a", rubric_control())

    def tearDown(self):
        self.tmp.cleanup()

    def test_positive_two_families_roundtrip_reaches_existing_report(self):
        rows = []
        for j in self.plan["judges"]:
            req, binding = prepare(self.bundle, self.plan, self.answer, j["judge_id"], rubric_control())
            rows.append(bind(self.bundle, self.plan, self.answer, req, binding, response_control(req)))
        self.assertTrue(report_run(self.bundle, self.plan, self.answers, rows)["ready_for_pilot_interpretation"])
        self.assertEqual(rows[0]["evidence_refs"], self.answer["evidence_refs"])

    def test_payload_has_no_system_metadata_and_preserves_original_content(self):
        payload = self.request["payload"]
        self.assertEqual(payload["evidence"][0]["content"], self.answer["evidence"][0]["content"])
        self.assertEqual(payload["cutoff"], "2026-10-23T12:00:00Z")
        for forbidden in ("system_id", "configuration_sha256", "model", "metrics", "tool_trace", "plan_sha256"):
            self.assertNotIn(forbidden, payload)
        self.assertNotIn(self.answer["system_id"], canonical_bytes(payload).decode())

    def test_identity_in_answer_or_evidence_is_refused_not_redacted(self):
        for target in ("answer", "evidence"):
            answer = copy.deepcopy(self.answer)
            if target == "answer":
                answer["answer"] += " I am " + answer["model"]
            else:
                answer["evidence"][0]["source_ref"] = "https://example.test/" + answer["system_id"]
            with self.assertRaisesRegex(ContractError, "identity appears"):
                prepare(self.bundle, self.plan, answer, "judge-a", rubric_control())

    def test_rehashed_substituted_payload_is_still_refused(self):
        self.request["payload"]["answer"] = "Injected alternative answer"
        self.request["payload_sha256"] = digest(self.request["payload"])
        self.binding["request_sha256"] = digest(self.request)
        with self.assertRaisesRegex(ContractError, "projected payload"):
            bind(self.bundle, self.plan, self.answer, self.request, self.binding, response_control(self.request))

    def test_cross_request_or_payload_response_refused(self):
        for key in ("request_id", "payload_sha256"):
            response = response_control(self.request)
            response[key] = "b" * 64
            with self.assertRaisesRegex(ContractError, "different request/payload"):
                bind(self.bundle, self.plan, self.answer, self.request, self.binding, response)

    def test_changed_answer_and_plan_cannot_reuse_binding(self):
        self.answer["answer"] = "Changed answer"
        with self.assertRaisesRegex(ContractError, "binding changed"):
            bind(self.bundle, self.plan, self.answer, self.request, self.binding, response_control(self.request))

    def test_unknown_alias_and_invalid_score_and_extra_metadata_fail(self):
        for patch in ({"evidence_refs": ["foreign"]}, {"scores": {d: 5 for d in DIMENSIONS}}, {"system_id": "leaked"}):
            response = {**response_control(self.request), **patch}
            with self.assertRaises(ContractError):
                bind(self.bundle, self.plan, self.answer, self.request, self.binding, response)

    def test_unknown_judge_failed_answer_and_incomplete_rubric_refused(self):
        with self.assertRaises(ContractError):
            prepare(self.bundle, self.plan, self.answer, "undeclared", rubric_control())
        with self.assertRaises(ContractError):
            prepare(self.bundle, self.plan, self.answer, "judge-a", {})
        self.answer.update(status="timeout", answer=None, error="synthetic timeout")
        with self.assertRaisesRegex(ContractError, "only completed"):
            prepare(self.bundle, self.plan, self.answer, "judge-a", rubric_control())

    def test_output_private_and_existing_directory_never_overwritten(self):
        out = self.root / "judge-input"
        save_new(out, self.request, self.binding)
        self.assertEqual(stat.S_IMODE((out / "private-binding.json").stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            save_new(out, self.request, self.binding)

    def test_opaque_request_ids_do_not_reuse_private_source_hashes(self):
        other, _ = prepare(self.bundle, self.plan, self.answer, "judge-a", rubric_control())
        self.assertNotEqual(other["request_id"], self.request["request_id"])
        self.assertEqual(other["payload"], self.request["payload"])


if __name__ == "__main__":
    unittest.main()
