"""Offline import, audit, freeze and verification of portable benchmark cases."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from case_adapters import generated_cases, reference_cases, rq01_cases
from case_manifest import ContractError, apply_annotations, canonical_bytes, digest, parse_json
from coverage import (apply_policy_assignments, coverage_markdown, coverage_report,
                      selection_coverage, validate_policy)

HERE = Path(__file__).resolve().parent
ARTIFACTS = {"cases.jsonl", "coverage.json", "coverage.md", "policy.json",
             "config.json", "inputs.json", "source-manifest.json"}
CODE_FILES = ("case_manifest.py", "case_adapters.py", "coverage.py", "benchmark_cases.py")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        return [parse_json(line) for line in text.splitlines() if line.strip()]
    return parse_json(text)


def source_argument(value):
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise ContractError("source arguments must be NAME=PATH") from exc
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or not path:
        raise ContractError("invalid source namespace or empty path")
    return name, Path(path)


def input_descriptor(name, kind, files):
    return {"source_name": name, "format": kind,
            "files": [{"role": role, "filename": path.name, "sha256": file_hash(path),
                       "bytes": path.stat().st_size} for role, path in sorted(files.items())]}


def load_sources(args, config):
    cases, inputs, selections, names = [], [], {}, set()
    for kind in ("reference", "rq01", "generated", "generated_run", "cases"):
        for value in getattr(args, kind, []) or []:
            name, path = source_argument(value)
            if name in names:
                raise ContractError("source namespace reused: " + name)
            names.add(name)
            if kind == "reference":
                files = {"questions": path}
                rows = reference_cases(read_rows(path), config, name, file_hash(path))
            elif kind == "rq01":
                files = {"provenance": path / "rq01-variant-provenance.json",
                         "golden-user-prompts.jsonl": path / "golden-user-prompts.jsonl",
                         "regression-control.jsonl": path / "regression-control.jsonl"}
                hashes = {role: file_hash(p) for role, p in files.items()}
                rows = rq01_cases(read_rows(files["provenance"]),
                                 {role: read_rows(p) for role, p in files.items() if role != "provenance"},
                                 name, hashes)
            elif kind in {"generated", "generated_run"}:
                questions = path / "questions.json" if kind == "generated_run" else path
                files = {"questions": questions}
                raw = read_rows(questions)
                rows = generated_cases(raw, name, file_hash(questions))
                if kind == "generated_run":
                    files["specs"] = path / "specs.json"
                    all_candidates = path / "all-candidates.json"
                    if all_candidates.exists():
                        files["all_candidates"] = all_candidates
                    selections[name] = selection_coverage(
                        read_rows(files["specs"]), raw,
                        read_rows(all_candidates) if all_candidates.exists() else None)
            else:
                files = {"portable_cases": path}
                rows = read_rows(path)
                if not isinstance(rows, list):
                    raise ContractError("portable cases input must contain a list of records")
            inputs.append(input_descriptor(name, kind, files))
            cases.extend(rows)
    if not inputs:
        raise ContractError("supply at least one reference, rq01, generated, generated-run or cases source")
    return cases, sorted(inputs, key=lambda row: row["source_name"]), selections


def write_json(path, value):
    Path(path).write_bytes(canonical_bytes(value) + b"\n")


def build(args):
    if args.out.exists():
        raise ContractError("output already exists; use a new directory to preserve prior runs")
    config = read_rows(args.config)
    policy = read_rows(args.policy)
    validate_policy(policy)
    cases, inputs, selections = load_sources(args, config)
    # Validate imported structure before policy rules can inspect or modify it.
    coverage_report(cases, dict(policy, assignment_rules=[]), selections)
    cases = apply_policy_assignments(cases, policy)
    if args.annotations:
        if any(row["source_name"] == "design_assignments" for row in inputs):
            raise ContractError("design_assignments is reserved when --annotations is used")
        cases = apply_annotations(cases, read_rows(args.annotations))
        inputs.append(input_descriptor("design_assignments", "explicit_design_annotations",
                                       {"annotations": args.annotations}))
    cases.sort(key=lambda row: row["case_id"])
    report = coverage_report(cases, policy, selections)
    state = ("frozen" if report["freeze_ready"] else "freeze_rejected") if args.command == "freeze" else "audit"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "cases.jsonl").write_bytes(b"".join(canonical_bytes(c) + b"\n" for c in cases))
    write_json(args.out / "coverage.json", report)
    (args.out / "coverage.md").write_text(coverage_markdown(report), encoding="utf-8")
    write_json(args.out / "policy.json", policy)
    write_json(args.out / "config.json", config)
    write_json(args.out / "inputs.json", sorted(inputs, key=lambda row: row["source_name"]))
    write_json(args.out / "source-manifest.json", {"files": {name: file_hash(HERE / name) for name in CODE_FILES},
                                                   "meaning": "source hashes identify the compiler, not an approved benchmark or deployed runner"})
    manifest = {"schema_version": "financial-benchmark-bundle/v1", "state": state,
                "command": args.command, "case_count": len(cases), "policy_id": policy["policy_id"],
                "policy_status": policy["status"], "external_effects_allowed": False,
                "selection_provenance": selections,
                "files": {name: file_hash(args.out / name) for name in sorted(ARTIFACTS)}}
    manifest["bundle_id"] = digest(manifest)
    write_json(args.out / "manifest.json", manifest)
    return {"state": state, "cases": len(cases), "freeze_ready": report["freeze_ready"],
            "freeze_failure_count": len(report["freeze_failures"]), "bundle_id": manifest["bundle_id"]}, (2 if state == "freeze_rejected" else 0)


def verify(path):
    manifest = read_rows(path / "manifest.json")
    expected_fields = {"schema_version", "state", "command", "case_count", "policy_id",
                       "policy_status", "external_effects_allowed", "selection_provenance", "files", "bundle_id"}
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ContractError("invalid bundle manifest fields")
    if manifest["schema_version"] != "financial-benchmark-bundle/v1" or manifest["state"] not in {"audit", "frozen", "freeze_rejected"}:
        raise ContractError("unsupported bundle version/state")
    if (manifest["command"] == "audit" and manifest["state"] != "audit") or (
        manifest["command"] == "freeze" and manifest["state"] not in {"frozen", "freeze_rejected"}
    ) or manifest["command"] not in {"audit", "freeze"}:
        raise ContractError("bundle command/state disagree")
    if set(manifest["files"]) != ARTIFACTS or manifest["external_effects_allowed"] is not False:
        raise ContractError("invalid artifact set or effect authority")
    unsigned = {k: v for k, v in manifest.items() if k != "bundle_id"}
    if digest(unsigned) != manifest["bundle_id"]:
        raise ContractError("bundle ID mismatch")
    for name, expected in manifest["files"].items():
        if file_hash(path / name) != expected:
            raise ContractError("artifact hash mismatch: " + name)
    cases = read_rows(path / "cases.jsonl")
    policy = read_rows(path / "policy.json")
    report = coverage_report(cases, policy, manifest["selection_provenance"])
    if report != read_rows(path / "coverage.json") or coverage_markdown(report) != (path / "coverage.md").read_text(encoding="utf-8"):
        raise ContractError("coverage differs from a fresh recomputation")
    if manifest["case_count"] != len(cases) or manifest["policy_id"] != policy["policy_id"] or manifest["policy_status"] != policy["status"]:
        raise ContractError("bundle metadata disagrees with its cases/policy")
    if (manifest["state"] == "frozen" and not report["freeze_ready"]) or (manifest["state"] == "freeze_rejected" and report["freeze_ready"]):
        raise ContractError("bundle readiness does not match its state")
    return {"verified": True, "state": manifest["state"], "cases": len(cases),
            "bundle_id": manifest["bundle_id"],
            "meaning": "artifact consistency, not source authenticity or financial correctness"}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("audit", "freeze"):
        p = sub.add_parser(command)
        for name in ("reference", "rq01", "generated", "generated-run", "cases"):
            p.add_argument("--" + name, action="append", metavar="NAME=PATH")
        p.add_argument("--config", type=Path, default=HERE / "axes.json")
        p.add_argument("--policy", type=Path, default=HERE / "development-policy.json")
        p.add_argument("--annotations", type=Path)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("bundle", type=Path)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "verify":
            result, code = verify(args.bundle), 0
        else:
            result, code = build(args)
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
