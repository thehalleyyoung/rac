# Answer records and pilot-readiness checks

RQ01 research consumer, implemented locally. It makes no network, model,
provider or account calls. It does not implement the live Edge adapter or
claim that declared provider/model identities are authentic.

## Handoff

`answer_records.py` consumes an independently verified **frozen** case bundle
from `benchmark_cases.py`, a declared run plan, answer attempts, and judge
records. An audit or rejected-freeze bundle cannot be used. The original case
format, generator and audit commands remain unchanged.

The plan binds the bundle ID, run ID, system model/configuration hashes,
read-tool roster, maximum attempts and judge model/configuration/family roster.
At least two declared families are required. Each attempt and judgement binds
the exact plan hash, case, system and attempt; each judgement also binds the
complete answer-record hash. Family labels are declarations, not evidence of
independence. A changed plan requires newly bound records and is a different
research run, not a silent correction of the old run.

Every case × declared system is scheduled. The consumer retains all supplied
attempts, including invalid records, and reports missing slots. Attempts start
at 1, remain contiguous, share a request ID for retries, and have ordered
clocks. A completed attempt cannot be followed by a retry. Final timeouts,
errors and refusals stay in the denominator; no judge score is invented for
them. A missing slot or invalid record blocks readiness. A fully accounted
operational failure does not prevent reporting that failure rate.

## Evidence and tool results

Evidence carries exact UTF-8 content and its SHA256, source/producer/identity,
observation, publication, known and retrieval times. Observation/publication/
known times cannot exceed the case cutoff; retrieval can occur later for an
eligible historical capture, but cannot precede knowledge or follow the
answer/tool completion. Tool traces retain exact request/result strings,
call identity, status, times and evidence links. Each linked capture must
match the tool response bytes. End-to-end evidence must be linked to a trace;
a completed end-to-end answer needs a retrieval attempt even if data is absent.

Controlled mode requires the same predeclared packet hash per case for all
systems. End-to-end mode requires either an explicit capture-manifest hash
for replay or a declared observation window. These checks verify declared
record consistency. They do not authenticate a provider, attest remote capture
time, independently establish market truth, or prove a live synchronized run.
Synthetic fixtures use `kind=synthetic_control` and never become market data.
No chart or media is produced by this slice.

The tool allowlist is supplied by the run administrator. Recorded write or
undeclared activity fails the protocol. This consumer does not execute tools
or enforce a runtime sandbox, and a caller labelling a tool read-only does not
grant it trading authority. A live adapter still needs an actual permission
boundary, authenticated trace capture and an independently reviewed roster.

## Judging and reporting

The five fixed score keys are `task_completion`, `evidence_fidelity`,
`financial_reasoning`, `premise_and_uncertainty`, and `retail_usefulness`.
Scores must be finite 0–4 numbers. Reasons, evidence references and individual
semantic flags are retained. Every terminal completed answer needs all
declared judges. Report per-dimension maximum-minus-minimum disagreement;
there is no calibrated threshold, scalar, cross-mode ranking or automatic
zeroing from a semantic allegation. Latency, tokens and decimal USD costs
remain recorded metrics, with unknown tokens/cost represented by null.

An ordinary case with `missing_required=true` is reported as a completion
failure and cannot receive full completion marks. An explicit unavailable-data
challenge can receive full marks under its frozen criteria. This is a narrow
consistency guard: the consumer does not prove that a judge applied those
criteria correctly or detect an adapter lying about `missing_required`.
Proactivity/actionability are task-specific rubric work, not new universal
axes or a claim that this consumer measures trader uplift.

Counts stay separate by evidence mode and system. Family IDs are retained for
later clustered uncertainty; this slice computes no confidence intervals or
population estimates. Raw inputs and hashes remain available in output even
when readiness fails. Readiness means this protocol can be inspected; it does
not establish judge validity, financial correctness, system freeze attestation,
third-party independence or adoption approval.

## Run and verify

From the portfolio root, after preparing the declared input records:

```sh
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/answer_records.py \
  --bundle ./output/rq01-answer-records/control-inputs/bundle \
  --plan ./output/rq01-answer-records/control-inputs/plan.json \
  --attempts ./output/rq01-answer-records/control-inputs/attempts.json \
  --judges ./output/rq01-answer-records/control-inputs/judges.json \
  --out ./output/new-answer-control-report
```

The example is a **synthetic scripted pipeline control**, including both
families' scripted scores. It is not an Edge run, market capture or LLM judge.
The owning tests' `control()` factory reproduces its inputs without paid calls.

Outputs are `plan.json`, `attempts.json`, `judges.json`, `report.json` and a
hash receipt identifying this consumer source. An existing directory is never
overwritten. Exit 0 means protocol-ready; exit 2 emits a blocked report with
all records retained; exit 1 means invalid plan/bundle/input or IO failure.
Repeat the CLI into a new directory and compare all bytes for deterministic
replay. A hash is not a signature: a malicious writer can rewrite and rehash
an entire bundle. Independent custody and append-only capture are future work.

```sh
python3 -m unittest discover \
  -s ./worktrees/rac-retail-question-distribution/experiments/financial_questions \
  -p 'test_answer_records.py' -v
```

Positive/negative controls cover exact tool/evidence bytes, cutoff clocks,
identity/version binding, missing slots, retry failures, two-family records,
ordinary versus unavailable cases, malformed scores, controlled packet parity,
synchronized/replay policy, separate mode counts, CLI replay and raw retention.
The full existing generator/case suite also needs its recorded NumPy environment.

Remaining slices: real read-only Edge adapter and permissions, provider/catalog
resolution, authentic capture manifests, judge requests and blind payloads,
task-specific rubrics and applicability, workflow cases, calibrated scoring,
independent execution and the later blinded human audit. Do not interpret
synthetic readiness as delivery of these features. Rollback is to stop consuming
this additive protocol; historical case bundles and product writers are intact.
