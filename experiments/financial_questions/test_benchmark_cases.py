import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark_cases import build, parser, verify, write_json
from case_manifest import ContractError, canonical_bytes, digest
from test_case_manifest import CUTOFF, HERE, complete_case
from test_coverage import policy


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / "cases.jsonl"
        self.policy_path = self.root / "policy.json"
        self.input.write_bytes(canonical_bytes(complete_case()) + b"\n")
        write_json(self.policy_path, policy())

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, command, out):
        return parser().parse_args([command, "--cases", "synthetic=" + str(self.input),
                                    "--policy", str(self.policy_path), "--out", str(out)])

    def test_real_cli_freezes_and_verifies_a_complete_offline_synthetic_control(self):
        out = self.root / "frozen"
        command = [sys.executable, str(HERE / "benchmark_cases.py"), "freeze", "--cases",
                   "synthetic=" + str(self.input), "--policy", str(self.policy_path), "--out", str(out)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["state"], "frozen")
        check = subprocess.run([sys.executable, str(HERE / "benchmark_cases.py"), "verify", str(out)],
                               capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertTrue(json.loads(check.stdout)["verified"])

    def test_same_inputs_emit_byte_identical_bundles_and_existing_output_is_preserved(self):
        first, second = self.root / "first", self.root / "second"
        with patch("socket.socket", side_effect=AssertionError("offline command attempted network")):
            build(self.args("freeze", first))
            build(self.args("freeze", second))
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                         {p.name: p.read_bytes() for p in second.iterdir()})
        with self.assertRaises(ContractError):
            build(self.args("freeze", first))
        self.assertTrue(verify(first)["verified"])

    def test_incomplete_audit_is_identical_and_verifiable_across_hash_seeds(self):
        case = complete_case()
        case.update(instrument=None, timezone=None, cutoff=None, stratum=None)
        case["assigned_requirements"] = {}
        self.input.write_bytes(canonical_bytes(case) + b"\n")
        outputs = []
        for seed in ["17", "91"]:
            out = self.root / ("hash-seed-" + seed)
            env = dict(os.environ, PYTHONHASHSEED=seed)
            run = subprocess.run([sys.executable, str(HERE / "benchmark_cases.py"), "audit", "--cases",
                                  "synthetic=" + str(self.input), "--policy", str(self.policy_path),
                                  "--out", str(out)], capture_output=True, text=True, env=env)
            self.assertEqual(run.returncode, 0, run.stderr)
            outputs.append({p.name: p.read_bytes() for p in out.iterdir()})
            check = subprocess.run([sys.executable, str(HERE / "benchmark_cases.py"), "verify", str(out)],
                                   capture_output=True, text=True, env=dict(env, PYTHONHASHSEED="123"))
            self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(outputs[0], outputs[1])

    def test_audit_keeps_incomplete_case_but_freeze_exits_two_with_rejected_artifacts(self):
        case = complete_case()
        case["instrument"] = None
        self.input.write_bytes(canonical_bytes(case) + b"\n")
        audit, code = build(self.args("audit", self.root / "audit"))
        self.assertEqual(code, 0)
        self.assertEqual(audit["state"], "audit")
        self.assertFalse(audit["freeze_ready"])
        frozen, code = build(self.args("freeze", self.root / "rejected"))
        self.assertEqual(code, 2)
        self.assertEqual(frozen["state"], "freeze_rejected")
        self.assertEqual((self.root / "rejected/cases.jsonl").read_bytes(), self.input.read_bytes())
        self.assertEqual(verify(self.root / "rejected")["state"], "freeze_rejected")
        run = subprocess.run([sys.executable, str(HERE / "benchmark_cases.py"), "freeze", "--cases",
                              "synthetic=" + str(self.input), "--policy", str(self.policy_path),
                              "--out", str(self.root / "rejected-cli")], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)

    def test_portable_artifact_tampering_is_detected(self):
        out = self.root / "bundle"
        build(self.args("freeze", out))
        (out / "cases.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            verify(out)

    def test_audit_cannot_be_relabelled_frozen_without_matching_command_and_readiness(self):
        out = self.root / "audit-state"
        build(self.args("audit", out))
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["state"] = "frozen"
        manifest["bundle_id"] = digest({k: v for k, v in manifest.items() if k != "bundle_id"})
        write_json(out / "manifest.json", manifest)
        with self.assertRaisesRegex(ContractError, "command/state"):
            verify(out)

    def test_live_generation_format_can_freeze_using_only_predeclared_rules(self):
        generated = self.root / "questions.json"
        write_json(generated, [{"id": "g1", "prompt": "What happened in this synthetic case?",
                               "decision": "selected", "label": {"task": "news"},
                               "requested_spec": {"id": "s1", "source_id": "source1",
                                                  "asset_class": "equities", "instrument": "spot",
                                                  "task": "news", "as_of": CUTOFF}}])
        p = policy()
        p["required_source_intents"] = ["generated:source1"]
        p["assignment_rules"] = [{"where": {}, "only_missing": True,
                                   "set": {"stratum": "ordinary", "timezone": "UTC",
                                           "identity_policy": "resolve_at_run", "evidence_mode": "end_to_end",
                                           "success_criteria": ["Identify the supplied synthetic evidence."]}}]
        write_json(self.policy_path, p)
        args = parser().parse_args(["freeze", "--generated", "generated=" + str(generated),
                                     "--policy", str(self.policy_path), "--out", str(self.root / "generated")])
        result, code = build(args)
        self.assertEqual(code, 0)
        self.assertTrue(result["freeze_ready"])
        self.assertTrue(verify(self.root / "generated")["verified"])

    def test_reimport_and_annotation_preserve_original_text_and_source_hash(self):
        case = complete_case()
        case["instrument"] = None
        self.input.write_bytes(canonical_bytes(case) + b"\n")
        annotations = self.root / "annotations.json"
        write_json(annotations, {"version": 1, "owner": "synthetic-policy-author",
                                 "assignments": {case["case_id"]: {"instrument": "spot"}}})
        args = self.args("freeze", self.root / "annotated")
        args.annotations = annotations
        result, code = build(args)
        self.assertEqual(code, 0)
        saved = json.loads((args.out / "cases.jsonl").read_text())
        self.assertEqual(saved["prompt"], case["prompt"])
        self.assertEqual(saved["provenance"], case["provenance"])
        self.assertTrue(saved["assignment_history"])
        self.assertTrue(verify(args.out)["verified"])

    def test_bad_inputs_and_colliding_names_do_not_emit_partial_success(self):
        self.input.write_text('{"id": "a", "id": "b"}\n')
        with self.assertRaises(ContractError):
            build(self.args("audit", self.root / "bad"))
        self.assertFalse((self.root / "bad").exists())
        self.input.write_bytes(canonical_bytes(complete_case()) + b"\n")
        args = self.args("audit", self.root / "collision")
        args.cases.append("synthetic=" + str(self.input))
        with self.assertRaises(ContractError):
            build(args)
        self.assertFalse(args.out.exists())

    def test_reference_import_works_in_one_command_without_model_dependencies(self):
        args = parser().parse_args(["audit", "--reference", "reference32=" + str(HERE / "four-categories-32.json"),
                                     "--out", str(self.root / "references")])
        result, code = build(args)
        self.assertEqual(code, 0)
        self.assertEqual(result["cases"], 32)
        report = json.loads((args.out / "coverage.json").read_text())
        self.assertEqual(report["missing_source_intents"], [])
        self.assertFalse(report["freeze_ready"])
        self.assertEqual(report["counts"]["instrument"]["<unknown>"], 24)
        self.assertTrue(verify(args.out)["verified"])


if __name__ == "__main__":
    unittest.main()
