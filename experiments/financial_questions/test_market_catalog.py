import copy
import hashlib
import json
import unittest

from case_manifest import ContractError
from market_catalog import MARKET_FIELDS, audit_catalog


def catalog_control():
    """Seventeen synthetic markets, never real provider listings."""
    entries = []
    for i in range(17):
        row = {"market_id": "synthetic-market-%02d" % i,
               "asset_class": ("equities", "commodities", "crypto_majors", "crypto_onchain")[i % 4],
               "instrument": "spot", "venue": "synthetic-venue", "base": "SYNTH%02d" % i,
               "quote": "SYNTHUSD", "settlement": "SYNTHUSD", "expiry": None, "strike": None,
               "resolution_rules": None, "identity_status": "resolved", "snapshot_id": "synthetic-listing",
               "source_row": i, "valid_from": "2026-10-23T10:00:00Z", "valid_until": None}
        entries.append(row)
    content = json.dumps([{k: e[k] for k in MARKET_FIELDS.split()} for e in entries])
    catalog = {"schema_version": "financial-market-catalog/v1", "catalog_id": "synthetic-17/v1",
               "external_effects_allowed": False, "entries": entries,
               "snapshots": [{"snapshot_id": "synthetic-listing", "kind": "synthetic_control",
                              "source_ref": "synthetic:listing", "observed_at": "2026-10-23T10:00:00Z",
                              "known_at": "2026-10-23T11:00:00Z", "retrieved_at": "2026-10-23T11:01:00Z",
                              "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}]}
    policy = {"schema_version": "financial-catalog-policy/v1", "minimum_markets": 17,
              "asset_minimums": {a: 1 for a in ("equities", "commodities", "crypto_majors", "crypto_onchain")},
              "instrument_minimums": {"spot": 1}, "allow_synthetic": True, "cutoff": "2026-10-23T12:00:00Z"}
    return catalog, policy


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog, self.policy = catalog_control()

    def resnapshot(self):
        content = json.dumps([{k: e[k] for k in MARKET_FIELDS.split()} for e in self.catalog["entries"]])
        self.catalog["snapshots"][0].update(content=content, content_sha256=hashlib.sha256(content.encode()).hexdigest())

    def test_seventeen_synthetic_markets_pass_only_disclosed_fixture_policy(self):
        result = audit_catalog(self.catalog, self.policy)
        self.assertTrue(result["ready"])
        self.assertEqual(result["eligible_markets"], 17)
        self.assertEqual(result["evidence_kinds"], ["synthetic_control"])
        self.policy["allow_synthetic"] = False
        self.assertFalse(audit_catalog(self.catalog, self.policy)["ready"])

    def test_sixteen_markets_and_missing_asset_or_instrument_stratum_fail(self):
        self.catalog["entries"].pop()
        self.assertFalse(audit_catalog(self.catalog, self.policy)["ready"])
        self.policy["instrument_minimums"]["options"] = 1
        result = audit_catalog(self.catalog, self.policy)
        self.assertIn("options", str(result["coverage_gaps"]))

    def test_unresolved_duplicate_or_relabelled_original_does_not_count(self):
        for change in ({"identity_status": "unresolved"}, {"market_id": "synthetic-market-01"},
                       {"base": "FAKE"}, {"source_row": -1}):
            catalog = copy.deepcopy(self.catalog)
            catalog["entries"][0].update(change)
            self.assertFalse(audit_catalog(catalog, self.policy)["ready"])

    def test_tampered_snapshot_or_future_knowledge_refused(self):
        self.catalog["snapshots"][0]["content"] += " "
        with self.assertRaisesRegex(ContractError, "bytes/hash"):
            audit_catalog(self.catalog, self.policy)
        self.catalog, _ = catalog_control()
        self.catalog["snapshots"][0]["known_at"] = "2026-10-24T00:00:00Z"
        with self.assertRaisesRegex(ContractError, "cutoff eligible"):
            audit_catalog(self.catalog, self.policy)

    def test_expired_identity_and_unbacked_option_are_rejected(self):
        self.catalog["entries"][0]["valid_until"] = self.policy["cutoff"]
        self.assertFalse(audit_catalog(self.catalog, self.policy)["ready"])
        self.catalog["entries"][1]["instrument"] = "options"
        self.assertFalse(audit_catalog(self.catalog, self.policy)["ready"])

    def test_no_authority_or_sixteen_market_policy_is_accepted(self):
        self.catalog["external_effects_allowed"] = True
        with self.assertRaises(ContractError):
            audit_catalog(self.catalog, self.policy)
        self.catalog["external_effects_allowed"] = False
        self.policy["minimum_markets"] = 16
        with self.assertRaisesRegex(ContractError, "more than 16"):
            audit_catalog(self.catalog, self.policy)

    def test_backed_duplicate_contract_under_another_id_is_rejected(self):
        self.catalog["entries"][1]["base"] = self.catalog["entries"][0]["base"]
        self.resnapshot()
        result = audit_catalog(self.catalog, self.policy)
        self.assertIn("duplicate resolved contract", str(result["invalid_entries"]))

    def test_original_option_and_prediction_rows_require_current_terms(self):
        row = self.catalog["entries"][0]
        for instrument, phrase in (("options", "option terms missing"),
                                   ("prediction_markets", "rules missing")):
            row["instrument"] = instrument
            self.resnapshot()
            self.assertIn(phrase, str(audit_catalog(self.catalog, self.policy)["invalid_entries"]))
        row.update(instrument="options", expiry="2026-10-24T12:00:00Z", strike="100")
        self.resnapshot()
        self.assertTrue(audit_catalog(self.catalog, self.policy)["ready"])
        row["expiry"] = self.policy["cutoff"]
        self.resnapshot()
        self.assertIn("option expired", str(audit_catalog(self.catalog, self.policy)["invalid_entries"]))


if __name__ == "__main__":
    unittest.main()
