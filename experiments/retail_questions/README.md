# RQ01: retail question generation

This exploration adapts RAC at base `62c0404c1c96689cad2f921c1e7345065516b2d3`. It imports the existing `coverage_levels` and `farthest_levels` primitives from `code/calculus.py`. The experiment supplies retail axes, a provider adapter, validity/style screening and source-family quotas. It does not run the poem-specific pipeline or implement the complete recursive refinement algorithm from the paper.

`prompts.json` is the user's supplied 12-example development reference, SHA256 `fd111e039d3aec3351c24a181a142c3e1b8bae52412e52daf614bdf97694a9e1`. Eight examples carry `history` and four carry `buy/sell`. These tags and family frequencies are preserved by construction, not inferred to represent all retail traders.

## Run

Install NumPy in an isolated Python environment. Set `OPENAI_API_KEY` in the process environment. From the workspace root:

```sh
python -m unittest discover -s worktrees/rac-retail-question-distribution/experiments/retail_questions -p 'test_*.py' -v
python worktrees/rac-retail-question-distribution/experiments/retail_questions/run.py --out output/rac-retail-question-generation/new-run
```

Each run requires a new output directory and stops rather than silently replacing artifacts. The optional `--key-file` parses only an `OPENAI_API_KEY` assignment; it never sources shell files or records the credential path/value. API traffic uses the official endpoint, with Responses storage disabled. Question content still goes to the model provider. No account data, private repo source or credentials are sent in prompts.

## Generation and filtering

1. Probe the filter with all 12 reference questions as positives and six synthetic negatives. Stop before generation if validity controls fail.
2. Generate a simple 24-question baseline, two per family. Screen it independently.
3. Embed descriptions of retail evidence/framing axes. RAC coverage chooses representative levels; RAC packing orders additional levels.
4. Generate 24 development questions conditioned on those axes. Collect rejected examples and reasons. Freeze this feedback, the filter prompt, model snapshot, thresholds and code hash before final generation.
5. Generate eight final candidates per family. A separate call screens question validity, actual family and naturalness. There is no GTE answer or performance feedback.
6. Require `valid`, style at least 4/5 and the correct family. Reject exact source copies and questions over 40 words. Embed candidates; reject cosine similarity at least 0.94 to any source example. Within each family, RAC coverage selects representative candidate levels, followed by packing until the quota is filled. A 0.90 cosine cap rejects near-duplicates among selected questions. Thresholds are engineering choices, not calibrated semantic truth.

The initial goal is five selected questions per family, 60 total. A shortfall remains visible; the script does not relax filters to manufacture success. All candidates, labels, reasons, raw requests/responses, embeddings and costs are retained. A baseline comparison reports rates and wording length. Different candidate counts and generation instructions prevent a causal attribution of changes to RAC alone.

`farthest_levels` may repeat an index when all remaining vectors have zero distance. This adapter de-duplicates its ordering and appends remaining candidate IDs so the downstream duplicate gate actually checks them; baseline code remains untouched.

## Bounds and interpretation

The provider has a USD 5 estimated-cost ceiling, a conservative preflight token reservation, 500,000 total tokens and 80 requests per run. Model prices checked on 22 October 2026: GPT-4.1 mini input/cached/output USD 0.40/0.10/1.60 per million; embeddings USD 0.02 per million. Estimates depend on those published rates; billing is not independently reconciled. Model snapshot `gpt-4.1-mini-2025-04-14` generates and screens in separate calls. This leaves correlated model bias. Embeddings use `text-embedding-3-small`, 256 dimensions. The recorded seed is reserved for deterministic local sampling; provider generations are stochastic and not seed-reproducible.

The filter asks whether a question is a legitimate evaluable financial-intelligence request, not whether its premise is true or the answer easy. The later answer judge must check false premises, market timestamp, sources, uncertainty and the actual answer. A passed question filter establishes none of these.

All examples are seen development material. A third party would need to freeze the code/config and independently generate fresh private questions after GTE's evaluated version is frozen. This run is not that independent evaluation. No human annotations are required for this version; later human audits should estimate filter precision and distribution fit before calling it representative.

Sources: [RAC](https://github.com/thehalleyyoung/rac), [generation model and prices](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [embedding model and prices](https://developers.openai.com/api/docs/models/text-embedding-3-small).

## Single command and measured run

For a fresh run, use `python experiments/retail_questions/generate.py --out <new-output-directory>` from this RAC worktree, with `OPENAI_API_KEY` set. It executes the development/control stage, conditional-family refinement, quota repair and shuffled paired audit, sequentially. A shared local ledger enforces the USD 5 / 500,000 token / 80-call bounds across those stages. It stops with artifacts if the target bank remains incomplete. No annotation or intervention is required during a successful run. The single-command wrapper was compiled and its shared-budget contract tested; the live evidence below is from its stages executed individually during development, not a fresh end-to-end wrapper run.

Measured sequence on 22 October 2026: `run-01` stopped on an API JSON-format contract; `run-02` stopped on a filter family-schema failure; `run-03` passed 12 positive/six negative controls, produced 24 baseline and 24 development questions, then stopped because the first final request returned 48 rather than 96 questions. `run-04` used smaller batches and produced 59 selected from 96 generic-axis candidates. `run-05` used family-conditional axes and produced 56 from 96; its hold/sell and catalyst families were under quota. `run-06` generated 16 further candidates only for those two families, using retained failure reasons. It selected 60 of 112 combined candidates, five per family, with the original 40 history / 20 buy/sell proportions. No filter threshold was loosened. The other 52 comprise 14 family/style failures, one semantic duplicate and 37 eligible reserve questions.

The independent-call validity filter marked all 112 final-pool candidates financially valid. That is an automated label, not measured precision. Inspection found weaknesses still present in the selected set: unsupported catalyst premises, a statement fragment, unfamiliar or dated asset identifiers, and questions whose depth/venue/time context needs validation. Keep these as research findings. A passing filter does not make the bank publication-ready. A future no-annotation version should take a versioned instrument catalog and an as-of/event manifest as generation inputs, and distinguish deliberate false-premise tests from accidentally invented premises.

The condition-blinded same-model audit sampled two selected questions per family against both simple-baseline questions. It preferred tuned voice in 9/12 pairs (baseline 3), and tuned task variety in 8/12 (baseline 2, tie 2). Mean question length was 11.83 words, versus references 11.42 and baseline 14.42. The audit sometimes confused changed tickers with a changed task family; its explanations are retained. These are directional diagnostics, not a statistical or independent validation. No causal claim about RAC alone is supported.

Run artifacts are outside the repository in `./output/rac-retail-question-generation/` relative to the portfolio workspace. The measured source snapshots for the successful conditional and quota-repair stages are retained beside their outputs. The local card/ledger carry exact commits and final receipts.
