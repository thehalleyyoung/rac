import copy
import json
from pathlib import Path
import unittest

from case_adapters import generated_cases, reference_cases, rq01_cases
from case_manifest import (ContractError, apply_annotations, assign, blank_case,
                           canonical_bytes, digest, parse_json, validate_case)

HERE = Path(__file__).resolve().parent
HASH = "a" * 64
CUTOFF = "2026-10-23T12:00:00Z"


def complete_case(record_id="control"):
    """Synthetic case contract, not a provider observation or answer label."""
    case = blank_case("synthetic", record_id, "What evidence is available?",
                      "synthetic:family", HASH, "synthetic_test")
    assign(case, {"asset_class": "equities", "instrument": "spot", "task": "news",
                  "stratum": "ordinary", "cutoff": CUTOFF, "timezone": "UTC",
                  "identity_policy": "resolve_at_run", "evidence_mode": "end_to_end",
                  "success_criteria": ["Identify the evidence or disclose its absence."]},
           "synthetic_test_policy")
    return case


class CaseContractTests(unittest.TestCase):
    def test_complete_synthetic_case_passes_and_missing_metadata_only_blocks_freeze(self):
        case = complete_case()
        self.assertEqual(validate_case(case, freeze=True, tasks={"news"}), [])
        draft = blank_case("synthetic", "draft", "A reference question?",
                           "synthetic:f", HASH, "test")
        self.assertEqual(validate_case(draft), [])
        self.assertIn({"code": "missing_freeze_requirement", "field": "asset_class"},
                      validate_case(draft, freeze=True))

    def test_external_authority_is_rejected_for_every_instrument(self):
        for instrument in ["spot", "perps", "options", "prediction_markets"]:
            case = complete_case()
            case["instrument"] = instrument
            self.assertEqual(validate_case(case, freeze=True), [])
            case["external_effects_allowed"] = True
            self.assertIn("external_effect_authority_forbidden",
                          [x["code"] for x in validate_case(case, freeze=True)])

    def test_unresolved_stays_unresolved_and_fake_listing_fails(self):
        case = complete_case()
        before = copy.deepcopy(case)
        self.assertEqual(validate_case(case, freeze=True), [])
        self.assertEqual(case, before)
        self.assertEqual(case["identity_status"], "unresolved")
        case["contract_context"]["instrument_id"] = "invented-option"
        self.assertIn("unverified_contract_asserts_listing",
                      [x["code"] for x in validate_case(case, freeze=True)])

    def test_claim_check_and_classifier_do_not_make_a_premise_false(self):
        row = {"id": "q1", "prompt": "Did BTC fall?", "requested_spec": {
            "id": "slot1", "domain": "crypto_majors", "task": "price_move",
            "premise_mode": "claim_check", "as_of": CUTOFF},
            "label": {"asset_class": "equities", "task": "news", "valid": True}}
        case = generated_cases([row], "generated", HASH)[0]
        self.assertEqual(case["task"], "price_move")
        self.assertEqual(case["asset_class"], "crypto_majors")
        self.assertEqual(case["inferred_labels"][0]["labels"]["task"], "news")
        self.assertEqual(case["premise_truth_status"], "unknown")
        self.assertIsNone(case["stratum"])
        self.assertIsNone(case["instrument"])
        self.assertEqual(case["verified_evidence"], [])

    def test_annotations_are_explicit_preserve_history_and_cannot_change_prompt_or_authority(self):
        case = complete_case()
        document = {"version": 1, "owner": "measurement-owner", "assignments": {
            case["case_id"]: {"stratum": "challenge", "challenge_tags": ["missing_evidence"]}}}
        updated = apply_annotations([case], document)[0]
        self.assertEqual(case["stratum"], "ordinary")
        self.assertEqual(updated["stratum"], "challenge")
        self.assertEqual(updated["assignment_history"][-2]["previous"], "ordinary")
        self.assertEqual(validate_case(updated, freeze=True), [])
        for field, value in [("prompt", "changed"), ("external_effects_allowed", True),
                             ("verified_evidence", [{"kind": "model_claim"}])]:
            bad = copy.deepcopy(document)
            bad["assignments"][case["case_id"]] = {field: value}
            with self.assertRaises(ContractError):
                apply_annotations([case], bad)
        document["assignments"]["missing:id"] = {}
        with self.assertRaises(ContractError):
            apply_annotations([case], document)

    def test_evidence_provenance_and_times_are_distinct_and_cutoff_is_enforced(self):
        case = complete_case()
        case["verified_evidence"] = [{
            "kind": "provider_capture", "evidence_id": "synthetic-capture",
            "source_ref": "fixture:test", "producer": "synthetic_fixture",
            "content_sha256": HASH, "observed_at": "2026-10-23T11:00:00Z",
            "published_at": None, "retrieved_at": "2026-10-24T00:00:00Z",
            "known_at": "2026-10-23T11:01:00Z"}]
        self.assertEqual(validate_case(case, freeze=True), [])
        case["verified_evidence"][0]["known_at"] = "2026-10-24T00:00:00Z"
        self.assertIn({"code": "evidence_after_cutoff", "field": "known_at"},
                      validate_case(case, freeze=True))
        case["verified_evidence"][0]["kind"] = "model_assertion"
        self.assertIn("unsupported_evidence_kind", [x["code"] for x in validate_case(case)])

    def test_supported_or_contradicted_premise_needs_evidence(self):
        for status in ["supported", "contradicted"]:
            case = complete_case()
            case["premise_truth_status"] = status
            self.assertIn("asserted_fact_without_evidence", [x["code"] for x in validate_case(case)])

    def test_schema_rejects_bad_values_and_json_ambiguities(self):
        for field, value, code in [("instrument", ["spot"], "invalid_enum"),
                                   ("cutoff", "2026-10-23T12:00:00", "invalid_timestamp"),
                                   ("timezone", "not/a/timezone", "invalid_timezone"),
                                   ("task", "unlisted", "unregistered_task")]:
            case = complete_case()
            case[field] = value
            self.assertIn(code, [i["code"] for i in validate_case(case, tasks={"news"})])
        for text in ['{"id":1,"id":2}', '{"value":NaN}', '{"value":Infinity}']:
            with self.assertRaises(ContractError):
                parse_json(text)
        with self.assertRaises(ValueError):
            canonical_bytes({"bad": float("nan")})
        self.assertEqual(validate_case([])[0]["code"], "case_is_not_an_object")
        bad = complete_case()
        bad["contract_context"] = []
        self.assertIn("invalid_contract_context", [i["code"] for i in validate_case(bad)])
        bad["undeclared_permission"] = True
        self.assertIn("unknown_case_field", [i["code"] for i in validate_case(bad)])


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "axes.json").read_text())
        cls.references = json.loads((HERE / "four-categories-32.json").read_text())

    def test_all_32_reference_texts_and_intents_preserved_without_guessed_metadata(self):
        rows = reference_cases(self.references, self.config, "reference32", HASH)
        self.assertEqual(len(rows), 32)
        self.assertEqual([r["prompt"] for r in rows], [r["prompt"] for r in self.references])
        self.assertEqual(len({r["source_id"] for r in rows}), 32)
        self.assertEqual({r["asset_class"] for r in rows},
                         {"equities", "commodities", "crypto_majors", "crypto_onchain"})
        self.assertEqual(sum(r["instrument"] == "spot" for r in rows), 8)
        self.assertTrue(all(r["stratum"] is None and r["cutoff"] is None for r in rows))
        self.assertTrue(all(not validate_case(r, tasks=self.config["tasks"]) for r in rows))
        bad = copy.deepcopy(self.references)
        bad[0]["prompt"] += " changed"
        with self.assertRaises(ContractError):
            reference_cases(bad, self.config, "reference32", HASH)

    def test_rq01_provenance_selects_exact_rows_but_never_drops_a_missing_row(self):
        row = {"id": "golden-rq01-one", "userPrompt": "What happened?",
               "gradingCriteria": ["Check the premise."], "executionMode": "live_read_only"}
        link = {"caseId": row["id"], "sourceFamily": "news", "candidateId": "c1",
                "prompt": row["userPrompt"], "dataset": "golden-user-prompts.jsonl"}
        provenance = {"rows": [link]}
        inputs = {link["dataset"]: [row, {"id": "not-rq01", "userPrompt": "Elsewhere?"}]}
        rows = rq01_cases(provenance, inputs, "rq01", {"provenance": HASH, link["dataset"]: HASH})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt"], row["userPrompt"])
        self.assertEqual(rows[0]["success_criteria"], row["gradingCriteria"])
        self.assertIsNone(rows[0]["asset_class"])
        self.assertIsNone(rows[0]["instrument"])
        self.assertEqual(rows[0]["family_id"], "rq01:news")
        inputs[link["dataset"]][0]["userPrompt"] += " changed"
        with self.assertRaises(ContractError):
            rq01_cases(provenance, inputs, "rq01", {"provenance": HASH, link["dataset"]: HASH})

    def test_both_generator_formats_preserve_requested_and_realized_metadata(self):
        base = {"id": "g1", "prompt": "How is the market?", "requested_spec": {
            "id": "s1", "source_id": "source1", "asset_class": "commodities",
            "instrument": "options", "task": "options_analysis", "as_of": CUTOFF}}
        raw = copy.deepcopy(base)
        raw.update(decision="selected", label={"task": "news", "instrument": "spot"})
        exported = copy.deepcopy(base)
        exported["realized_behavior"] = "news/rumor_check"
        for row in [raw, exported]:
            case = generated_cases([row], "generated", HASH)[0]
            self.assertEqual(case["instrument"], "options")
            self.assertEqual(case["task"], "options_analysis")
            self.assertEqual(case["source_contract"]["raw_source_record"], row)
            self.assertTrue(case["inferred_labels"])
            self.assertEqual(case["verified_evidence"], [])
        raw["decision"] = "rejected"
        with self.assertRaises(ContractError):
            generated_cases([raw], "generated", HASH)

    def test_duplicate_source_ids_fail_instead_of_overwriting(self):
        with self.assertRaises(ContractError):
            reference_cases(self.references + self.references[:1], self.config, "reference32", HASH)
        with self.assertRaises(ContractError):
            blank_case("../private", "id", "Question?", "family", HASH, "test")


if __name__ == "__main__":
    unittest.main()
