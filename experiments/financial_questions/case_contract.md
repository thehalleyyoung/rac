# Portable cases and distribution audit

RQ01 research implementation, not a production benchmark or adoption decision.
The CLI uses the Python standard library and makes no model, provider, account
or network calls. The existing generator and its CLI remain unchanged.

## What this slice does

`benchmark_cases.py` imports the 32 supplied reference questions, RQ01-linked
eval additions, selected generator outputs, or previously exported portable
cases. It preserves every imported question and its provenance. It audits a
declared distribution and can freeze a batch only when that contract passes.
It never generates a replacement question, drops a row, changes a quota,
deduplicates inputs, or selects on GTE's performance or available tools.

The case format is `financial-benchmark-case/v1`. A case contains:

| Fields | Meaning |
| --- | --- |
| `case_id`, `source_id`, `family_id`, `prompt` | Stable identity, explicit family linkage and exact original question |
| `asset_class`, `instrument`, `task`, `stratum`, `challenge_tags` | Assigned design requirements; missing values remain null |
| `cutoff`, `timezone`, `evidence_mode`, `data_health_expectation` | Declared temporal/source policy, not proof that evidence was captured |
| `identity_status`, `identity_policy`, `contract_context` | Resolution state, what the runner must do, and verified versus unverified contract metadata |
| `premise_mode`, `premise_truth_status` | Question framing versus independently supported truth status; `claim_check` never implies false |
| `success_criteria` | Case-specific grading requirements or explicitly assigned policy criteria |
| `assigned_requirements`, `assignment_history` | Where an assignment came from, including overwritten values and policy/annotation hashes |
| `inferred_labels` | Classifier output, explicitly unvalidated and separate from assigned requirements |
| `verified_evidence` | Declared capture/document/calculation provenance with separate observation, publication, retrieval and known times |
| `external_effects_allowed` | Always false, including spot and perps |
| `provenance`, `source_contract` | Input hashes and original metadata/rows, retained for inspection |

Evidence validation checks the declared provenance and times. It does not fetch
or authenticate the underlying content, establish financial truth, or make a
model label an observation. A resolved identity or supported/contradicted
premise needs evidence records. Observation, publication and known times must
not exceed the case cutoff; retrieval may be later for an explicitly captured
historical record. This does not prove that a live run met its source policy.

An unresolved identity may be frozen when `identity_policy` explicitly requires
resolution at run time or clarification. Freezing never supplies a missing
listing. `provided` identity requires a resolved, evidence-backed contract.

## Input adapters

- `--reference NAME=FILE`: the supplied reference format. Source ID, exact text
  and category must match the supplied axes registry. Tasks come from the
  authored registry, not inference. The explicitly named onchain-spot category
  supplies a spot design label; other instruments remain unknown.
- `--rq01 NAME=DIRECTORY`: reads `rq01-variant-provenance.json` and the two eval
  JSONL files in that directory. Imports exactly the linked rows, checks exact
  text and source read-only declarations, and retains grading criteria. It does
  not infer asset/instrument/task labels from legacy question wording.
- `--generated NAME=FILE`: accepts selected raw `questions.json` records or the
  exported question-bank format, both with `requested_spec`. Requested axes are
  retained separately from classifier labels/realized behavior. A selected file
  alone cannot prove whether an upstream generation plan was completed.
- `--generated-run NAME=DIRECTORY`: also reads `specs.json` and, when present,
  `all-candidates.json`. Reports unfilled slots, rejection reasons and initial
  versus selected distributions, including resampling changes.
- `--cases NAME=FILE`: reads portable cases without rewriting their original
  IDs or provenance. The new input file hash is recorded in the bundle.

Namespaces must be unique. Duplicate source IDs, missing linked rows, changed
reference text, malformed JSON and unknown schema fields fail rather than
silently reducing the input. Exact normalized duplicate questions retain all
their cases and source links; the report lists their groups. Semantic
equivalence is not inferred. Explicit family assignments remain the unit for
later clustered analysis, not a claim that all related prompts are identical.

## Policy and automated assignments

`development-policy.json` is an explicit development proposal: all 32 reference
families, presence of the four assets and four instruments, ordinary/challenge
coverage, task compatibility and no exact-duplicate excess. It does not encode
agreed population weights or require equal occupancy of all 16 design cells.
Change the policy explicitly when the measurement owner selects quotas.

Quotas support `where` filters and either `exact` or `min`/`max` bounds. Filters
can combine assets, instruments, tasks, strata, family, source, evidence mode
and premise framing. A quota miss is reported without reallocating mass.
Requested/inferred agreement is diagnostic unless a policy explicitly names
`required_realization_axes`; missing generated labels then fail the gate too.
Classifier agreement still does not establish valid questions or truth.

Predeclared `assignment_rules` fill requirements automatically, without human
annotation during a run. Each rule has `where`, `set`, and `only_missing`.
Rules are ordered, versioned with the policy, and retained in assignment
history. They cannot rewrite prompts, source IDs, evidence or effect authority.
An optional `--annotations` document supports explicit legacy development
assignments (`version: 1`, `owner`, `assignments` keyed by case ID). This is not
required for runtime generation. It must never be presented as independent
human ground truth or added after inspecting held-out system results.

The shipped development policy has no assignment rules: unresolved metadata in
the retained inputs is intentionally visible. The synthetic control policy
demonstrates an automatic ordinary/challenge assignment from framing, without
claiming a premise is true or false. It is a pipeline control, not a suitable
financial answer rubric or representative benchmark.

## Commands

From the research workspace root, audit the retained inputs in one command:

```sh
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/benchmark_cases.py audit \
  --reference reference32=./worktrees/rac-retail-question-distribution/experiments/financial_questions/four-categories-32.json \
  --rq01 rq01=./worktrees/idea-watch-rule-parity/packages/rust/quant-rig-evals/evals \
  --generated-run generated=./output/financial-question-generator-v2/reference-final-v29 \
  --out ./output/new-case-audit
```

The output directory must be new. This command imports 126 retained development
questions (32 + 63 + 31) for the recorded inputs; those counts are not a target
distribution or a claim that metadata and generation are complete.

Replace `audit` with `freeze` and choose a new output directory to require
completion. The retained inputs are expected to fail strict freeze, including
the historical unfilled generator slot. No gate is relaxed to produce green.

A positive synthetic control uses no paid generation or question annotations:

```sh
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/benchmark_cases.py freeze \
  --generated-run control=./worktrees/rac-retail-question-distribution/experiments/financial_questions/case_controls \
  --policy ./worktrees/rac-retail-question-distribution/experiments/financial_questions/case_controls/policy.json \
  --out ./output/new-synthetic-case-freeze
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/benchmark_cases.py verify \
  ./output/new-synthetic-case-freeze
```

## Artifacts, exits and reproducibility

Each successful audit or attempted freeze emits `cases.jsonl`, `coverage.json`,
`coverage.md`, `policy.json`, `config.json`, `inputs.json`,
`source-manifest.json` and `manifest.json`. The manifest records all artifact
hashes, compiler-source hashes, input hashes, policy status and a deterministic
bundle ID. It contains no execution timestamp or absolute input path, so the
same named inputs, code and configuration produce byte-identical bundles.

- Exit 0: audit emitted, or freeze succeeded. Check `state`; an audit is never
  a frozen benchmark, even if ready.
- Exit 2: freeze rejected. All cases and reasons remain in the output;
  `state=freeze_rejected`, never `frozen`.
- Exit 1: malformed input/policy, output already exists or another contract/IO
  error. No successful manifest is claimed.

`verify` checks hashes, bundle identity and fresh coverage recomputation. It
detects accidental artifact mutation; it is not a signature or proof that an
input provider, author, market observation or model is authentic. Use a trusted
compiler/source manifest and independent run administration for those claims.
The source files are identified by hash; they are not vendored into each bundle.

## Requirements and owning assertions

| Requirement | Evidence to inspect |
| --- | --- |
| Versioned contract and separate assertions/inferences/evidence | `test_case_manifest.py`, including claim-check and cutoff negatives |
| Three input adapters and original text/provenance | Adapter tests, real 126-record audit and external preservation check |
| Coverage, conditional quotas, missing intents, unknowns, duplicates/families | `test_coverage.py` and emitted `coverage.json` |
| Requested versus realized and acceptance-conditioned drift | Realized-axis and resampling/stratum tests; retained incomplete run selection report |
| Audit versus strict freeze; no authority escalation | CLI rejection tests and all-instrument authority negatives |
| No per-question annotation needed to run | Generated-format positive control using frozen assignment rules |
| Portable artifacts, deterministic replay and integrity | Byte-identical bundle, subprocess CLI and tampering tests |
| Compatibility with existing generator/legacy interfaces | Existing generator and legacy tests; new code does not alter their CLI |

Run new tests with the standard-library interpreter:

```sh
python3 -m unittest discover -s ./worktrees/rac-retail-question-distribution/experiments/financial_questions -p 'test_case_manifest.py' -v
python3 -m unittest discover -s ./worktrees/rac-retail-question-distribution/experiments/financial_questions -p 'test_coverage.py' -v
python3 -m unittest discover -s ./worktrees/rac-retail-question-distribution/experiments/financial_questions -p 'test_benchmark_cases.py' -v
```

The existing full generator suite also needs its previously recorded NumPy
environment. Test results, commits and the requirement-by-requirement completion
audit are recorded in the local RQ01 card and output receipts.

Remaining benchmark work is separate: approved population weights and grading
rubrics, verified catalog/data capture, synchronized retrieval, the headless
answer adapter, independent judges, human calibration and third-party execution.
This slice neither implements those systems nor makes the existing cases a
valid universal benchmark. Rollback is to stop consuming this additive format;
the older generator outputs and product authority contracts are unchanged.
