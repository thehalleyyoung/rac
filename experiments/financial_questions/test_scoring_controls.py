import copy
import unittest

from answer_records import DIMENSIONS
from case_manifest import ContractError
from scoring_controls import ANCHORS, audit_controls


def scoring_control():
    judges = [{"judge_id": "synthetic-a", "family": "synthetic-a", "model": "none-a", "configuration_sha256": "a" * 64},
              {"judge_id": "synthetic-b", "family": "synthetic-b", "model": "none-b", "configuration_sha256": "b" * 64}]
    controls = [{"control_id": "supported", "case_id": "synthetic:case", "answer_sha256": "a" * 64,
                 "polarity": "positive", "bounds": {d: [3, 4] for d in DIMENSIONS}, "rationale": "Synthetic supported answer."},
                {"control_id": "fabricated", "case_id": "synthetic:case", "answer_sha256": "b" * 64,
                 "polarity": "negative", "bounds": {"evidence_fidelity": [0, 1]}, "rationale": "Synthetic invented observation."}]
    records = [{"case_id": c["case_id"], "answer_sha256": c["answer_sha256"], **j,
                "scores": {d: (4 if c["polarity"] == "positive" else 0) for d in DIMENSIONS}}
               for c in controls for j in judges]
    return controls, judges, records


class ScoringTests(unittest.TestCase):
    def test_five_anchored_dimensions_and_positive_negative_controls(self):
        self.assertEqual(set(ANCHORS), set(DIMENSIONS))
        self.assertTrue(all(len(v) == 5 for v in ANCHORS.values()))
        self.assertTrue(audit_controls(*scoring_control())["passed"])

    def test_flat_high_and_flat_abstention_scores_fail_discrimination(self):
        for value in (0, 4):
            controls, judges, records = scoring_control()
            for r in records:
                r["scores"] = {d: value for d in DIMENSIONS}
            self.assertFalse(audit_controls(controls, judges, records)["passed"])

    def test_missing_duplicate_foreign_and_wrong_family_results_are_retained_as_failures(self):
        controls, judges, records = scoring_control()
        for rows in (records[:-1], records + records[:1], [{**records[0], "family": "foreign"}] + records[1:]):
            self.assertFalse(audit_controls(controls, judges, rows)["passed"])

    def test_changed_answer_binding_and_out_of_range_score_fail(self):
        controls, judges, records = scoring_control()
        records[0]["answer_sha256"] = "f" * 64
        self.assertFalse(audit_controls(controls, judges, records)["passed"])
        records[1]["scores"]["task_completion"] = float("nan")
        self.assertFalse(audit_controls(controls, judges, records)["passed"])

    def test_positive_only_and_bad_bounds_refuse(self):
        controls, judges, records = scoring_control()
        with self.assertRaisesRegex(ContractError, "deliberately bad"):
            audit_controls(controls[:1], judges, records[:2])
        controls[0]["bounds"]["task_completion"] = [4, 3]
        with self.assertRaisesRegex(ContractError, "interval"):
            audit_controls(controls, judges, records)

    def test_fractional_legacy_scores_remain_compatible(self):
        controls, judges, records = scoring_control()
        records[0]["scores"] = {d: 3.5 for d in DIMENSIONS}
        self.assertTrue(audit_controls(controls, judges, records)["passed"])


if __name__ == "__main__":
    unittest.main()
