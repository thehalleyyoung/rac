# Blind judge handoff

`blind_judges.py` is an offline research adapter over the existing frozen-case
and answer-record contracts. `prepare` validates the inputs, selects task,
criteria, cutoff, mode, health, answer and evidence through an explicit allowlist,
and assigns opaque evidence references. It preserves evidence content and source
times. System/model/run identifiers, metrics and tool metadata stay private.
Known identifiers in the retained content cause refusal rather than redaction.
This cannot detect every semantic clue to authorship or remove provider bias.
Evidence and answers remain untrusted text; a prompt instruction is not a model
sandbox or proof of resistance to prompt injection.

`prepare` returns a judge request and a separate private source binding. Only
`request.json` goes to a judge. The output directory is mode 0700 and files 0600;
existing directories cannot be overwritten. Keep `private-binding.json` local.
The payload hash binds the rubric and exact projected content; the random request
ID distinguishes otherwise identical requests without exposing a source hash.
Use a frozen rubric selected before scoring. Hashes establish local consistency,
not authenticated custody or resistance to a writer replacing all source files.

`bind` checks original case/plan/answer/request hashes, regenerates the projection,
rejects cross-request responses and unknown evidence aliases, and translates an
accepted response into the unchanged `financial-answer-judge/v1` record. At least
two declared families and complete failure accounting remain the existing
answer-report consumer's responsibility. Call `report_run` on all attempts after
binding; preparing one answer does not prove terminality or complete scheduling.
Finite decimal 0–4 scores remain compatible with historical records. No score
weights, critical-flag zeroing or calibration threshold is introduced.

The CLI prepares requests only; it performs no network or paid model call:

```sh
python3 experiments/financial_questions/blind_judges.py --bundle <frozen-bundle> \
  --plan <plan.json> --answer <single-answer.json> --judge <declared-judge-id> \
  --rubric <five-dimension-anchor-map.json> --out <new-directory>
```

Ten synthetic controls cover positive two-family report integration, original
evidence bytes, known identity leakage, payload substitution after rehashing,
foreign response IDs/hashes, changed answers, unknown aliases, invalid scores,
unknown judge, failed answer, incomplete rubric, output permissions and unique
request IDs. They establish protocol behavior, not financial answer quality,
independent judges, live Edge permission enforcement or independent V1 readiness.
