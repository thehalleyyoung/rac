# Ordered workflow receipts

`workflow_cases.py` adds an offline workflow wrapper without changing frozen
case, plan, answer or judge wire versions. A workflow owns exactly its frozen
bundle and records ordered steps, case hashes, expected behavior and a stop rule.
Cases must have nondecreasing cutoffs. User context is explicitly synthetic and
effects remain forbidden.

`context_for` constructs a transcript from every prior completed turn, excluding
future turns and system metadata. Use it only after validating the underlying
records; it is not an authentication boundary. `report_workflow` first uses the
existing answer report, then checks every system/step receipt, exact answer and
prior-context hashes, and nonoverlapping step clocks. After a timeout, error or
refusal, later steps must be recorded as refusals with a null prior context. They
remain visible in the denominator. A final success cannot erase an earlier
failure. The workflow report retains the complete underlying per-answer report.

Seven synthetic controls reach two-step reporting, exact prior context, future
context refusal, changed prior answers, missing/duplicate/substituted receipts,
overlapping execution clocks, timeout/refusal accounting and forbidden authority.
This proves ordered receipt consistency, not that a live system consumed the
context or revised a financial conclusion correctly. Real docs.gte.xyz workflow
mapping, cross-step semantic rubrics and authenticated execution remain separate.
Rollback stops consuming this additive wrapper; old readers remain unchanged.
