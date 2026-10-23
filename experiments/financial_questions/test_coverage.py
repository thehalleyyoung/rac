import copy
import json
from pathlib import Path
import unittest

from case_manifest import ContractError
from coverage import (apply_policy_assignments, coverage_markdown, coverage_report, selection_coverage,
                      validate_policy)
from test_case_manifest import HERE, complete_case


def policy(total=1):
    return {"schema_version": "financial-benchmark-policy/v1", "policy_id": "synthetic-controls/v1",
            "status": "development_proposal", "task_registry": ["news", "options_analysis"],
            "required_source_intents": ["synthetic:family"], "total": {"exact": total},
            "quotas": [], "max_duplicate_excess": 0,
            "task_compatibility": {"news": {"asset_classes": ["equities", "crypto_majors"],
                                            "instruments": ["spot", "perps"]},
                                   "options_analysis": {"asset_classes": ["equities"],
                                                        "instruments": ["options"]}},
            "required_realization_axes": [], "assignment_rules": []}


class CoverageTests(unittest.TestCase):
    def test_complete_control_passes_but_missing_category_or_intent_blocks_freeze(self):
        case = complete_case()
        p = policy()
        self.assertTrue(coverage_report([case], p)["freeze_ready"])
        p["quotas"] = [{"where": {"asset_class": "commodities"}, "min": 1}]
        report = coverage_report([case], p)
        self.assertFalse(report["freeze_ready"])
        self.assertEqual(report["counts"]["asset_class"]["commodities"], 0)
        self.assertIn("quota_not_met", [x["code"] for x in report["freeze_failures"]])
        p["required_source_intents"].append("missing:intent")
        self.assertEqual(coverage_report([case], p)["missing_source_intents"], ["missing:intent"])

    def test_conditional_instrument_task_quotas_and_compatibility(self):
        a, b = complete_case("a"), complete_case("b")
        b.update(instrument="options", task="options_analysis")
        p = policy(2)
        p["quotas"] = [{"where": {"asset_class": "equities", "instrument": "options",
                                   "task": "options_analysis"}, "exact": 1}]
        self.assertTrue(coverage_report([a, b], p)["freeze_ready"])
        b["asset_class"] = "crypto_majors"
        report = coverage_report([a, b], p)
        self.assertFalse(report["freeze_ready"])
        self.assertIn("incompatible_task_axis", [x["code"] for x in report["freeze_failures"]])

    def test_duplicate_rows_are_preserved_with_both_sources_and_reported(self):
        a, b = complete_case("a"), complete_case("b")
        b["prompt"] = "  " + a["prompt"].upper() + "\n"
        b["provenance"][0]["source_name"] = "another_source"
        report = coverage_report([a, b], policy(2))
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["duplicate_excess"], 1)
        self.assertEqual(report["by_source"], {"another_source": 1, "synthetic": 1})
        self.assertEqual(report["exact_normalized_duplicates"][0]["case_ids"], [a["case_id"], b["case_id"]])
        self.assertFalse(report["freeze_ready"])
        self.assertEqual(a["prompt"], "What evidence is available for a?")

    def test_requested_realized_mismatches_are_diagnostic_unless_policy_requires_match(self):
        case = complete_case()
        case["inferred_labels"] = [{"producer": "synthetic_classifier",
                                    "status": "unvalidated_model_inference",
                                    "labels": {"task": "options_analysis", "domain": "crypto_majors"}}]
        p = policy()
        report = coverage_report([case], p)
        self.assertTrue(report["freeze_ready"])
        self.assertEqual(report["requested_vs_realized"]["task"]["match_rate"], 0)
        self.assertIsNone(report["requested_vs_realized"]["instrument"]["match_rate"])
        self.assertEqual(report["requested_vs_realized"]["instrument"]["missing_inferred"], 1)
        self.assertEqual(case["task"], "news")
        p["required_realization_axes"] = ["task", "instrument"]
        report = coverage_report([case], p)
        self.assertFalse(report["freeze_ready"])
        self.assertEqual(sum(x["code"] == "required_realization_mismatch" for x in report["freeze_failures"]), 2)
        case["inferred_labels"] = []
        case["provenance"][0]["source_format"] = "generator_selected"
        report = coverage_report([case], p)
        self.assertFalse(report["freeze_ready"])
        self.assertEqual(report["inference_coverage"]["without_labels"], 1)
        self.assertEqual(sum(x["code"] == "required_realization_mismatch" for x in report["freeze_failures"]), 2)

    def test_filtering_or_resampling_that_changes_strata_is_visible_and_fails_quotas(self):
        planned = [{"id": "s1", "stratum": "ordinary", "task": "news"},
                   {"id": "s2", "stratum": "challenge", "task": "news"}]
        selected = [{"spec_id": "s1", "requested_spec": dict(planned[0])},
                    {"spec_id": "s2", "requested_spec": dict(planned[1], stratum="ordinary")}]
        selection = selection_coverage(planned, selected)
        self.assertTrue(selection["complete"])
        self.assertEqual(selection["distribution_delta"]["stratum"], {"challenge": -1, "ordinary": 1})
        a, b = complete_case("a"), complete_case("b")
        p = policy(2)
        p["quotas"] = [{"where": {"stratum": "challenge"}, "exact": 1}]
        self.assertFalse(coverage_report([a, b], p, {"generated": selection})["freeze_ready"])
        incomplete = selection_coverage(planned, selected[:1])
        report = coverage_report([a], policy(), {"generated": incomplete})
        self.assertIn("upstream_selection_incomplete", [x["code"] for x in report["freeze_failures"]])
        self.assertEqual(incomplete["unfilled_slot_ids"], ["s2"])

    def test_frozen_rules_assign_requirements_without_per_question_human_annotation(self):
        case = complete_case()
        case.update(stratum=None, timezone=None, identity_policy=None, evidence_mode=None,
                    success_criteria=[], premise_mode="claim_check")
        p = policy()
        p["assignment_rules"] = [{"where": {}, "only_missing": True,
                                   "set": {"stratum": "ordinary", "timezone": "UTC",
                                           "identity_policy": "resolve_at_run", "evidence_mode": "end_to_end",
                                           "success_criteria": ["Verify the claim against available evidence."]}},
                                  {"where": {"premise_mode": "claim_check"}, "only_missing": False,
                                   "set": {"stratum": "challenge", "challenge_tags": ["premise_verification"]}}]
        result = apply_policy_assignments([case], p)
        self.assertTrue(coverage_report(result, p)["freeze_ready"])
        self.assertIsNone(case["stratum"])
        self.assertEqual(result[0]["stratum"], "challenge")
        self.assertEqual(result[0]["premise_truth_status"], "unknown")
        self.assertEqual(result[0]["identity_status"], "unresolved")
        self.assertTrue(result[0]["assigned_requirements"]["stratum"].startswith("policy_rule:"))

    def test_bad_policy_and_case_errors_cannot_silently_relax_the_run(self):
        mutations = [lambda p: p.update(gte_supported_only=True),
                     lambda p: p["quotas"].append({"where": {"instrument": "optionz"}, "min": 1}),
                     lambda p: p.update(total={"min": -1}),
                     lambda p: p["quotas"].append({"where": {"task": "news"}, "min": 2, "max": 1}),
                     lambda p: p["assignment_rules"].append({"where": {}, "set": {"external_effects_allowed": True}, "only_missing": False})]
        for mutate in mutations:
            p = policy()
            mutate(p)
            with self.assertRaises(ContractError):
                validate_policy(p)
        with self.assertRaises(ContractError):
            coverage_report([{}], policy())
        with self.assertRaises(ContractError):
            coverage_report([complete_case(), complete_case()], policy(2))
        self.assertFalse(coverage_report([], policy(0))["freeze_ready"])

    def test_development_policy_requires_all_32_families_and_does_not_assign_missing_labels(self):
        p = json.loads((HERE / "development-policy.json").read_text())
        validate_policy(p)
        self.assertEqual(len(p["required_source_intents"]), 32)
        self.assertEqual(p["status"], "development_proposal")
        self.assertEqual(p["assignment_rules"], [])
        self.assertEqual(len(p["task_registry"]), 27)

    def test_unknown_labels_remain_visible_in_rendered_markdown_tables(self):
        case = complete_case()
        case["instrument"] = None
        report = coverage_report([case], policy())
        text = coverage_markdown(report)
        self.assertIn("| &lt;unknown&gt; | 1 |", text)
        self.assertEqual(report["counts"]["instrument"]["<unknown>"], 1)


if __name__ == "__main__":
    unittest.main()
