# RQ01: randomized financial-intelligence questions

This is an isolated research generator. Its primary input is `four-categories-32.json`, SHA256 `15c4273242fd1c2e09f02718e0afc65e7efa004ed9e92eb0918042c1614b0afe`. The file has eight examples in each category: onchain spot, crypto majors, commodities and equities. It replaces the earlier 12-example distribution for this experiment. Reference questions and their intent decomposition are design inputs, not answer labels or verified market observations.

The implementation imports RAC's actual `coverage_levels` primitive from `./rac/code/calculus.py`, through the preceding RQ01 provider adapter. Clean RAC source snapshot: `62c0404c1c96689cad2f921c1e7345065516b2d3`. The previous research control is `6e89f85d9961c04afa8cb31601eaf26763f2b37a`. The adapter uses the official model API rather than importing RAC's poem-specific provider pipeline. No GTE calls, trade execution or private account reads occur.

## Run it

Requirements: Python with NumPy, an OpenAI API credential in `OPENAI_API_KEY`, and an explicit benchmark as-of time. From the isolated RAC worktree:

```sh
# Inspect a deterministic plan without any model/API call.
python experiments/financial_questions/engine.py plan --out <new-plan-directory> --count 64 --seed 12345 --profile four_category_reference --as-of 2026-10-22T00:00:00Z

# Generate, filter, select and refill without annotation or UI interaction.
python experiments/financial_questions/engine.py run --out <new-run-directory> --count 64 --seed 12345 --profile four_category_reference --as-of 2026-10-22T00:00:00Z

# Explore additional compatible tasks while preserving equal category mass.
python experiments/financial_questions/engine.py run --out <another-new-directory> --count 32 --profile four_category_explore --as-of 2026-10-22T00:00:00Z

# Hold background settings fixed and vary one axis at a time.
python experiments/financial_questions/engine.py probe --out <new-probe-directory> --seed 12345 --as-of 2026-10-22T00:00:00Z

# Replay stored requests/responses offline, checking the protocol matches.
python experiments/financial_questions/engine.py run --out <new-replay-directory> --count 64 --seed 12345 --profile four_category_reference --as-of 2026-10-22T00:00:00Z --replay <recorded-run-directory>
```

Omitting `--seed` draws a fresh 64-bit seed from system entropy and records it. Supplying a seed reproduces the compatible proposal pools for the same code/configuration/as-of. Exact selected specs additionally depend on the saved embedding response; LLM question text is stochastic. A replay requires the same command parameters and frozen generation protocol. It makes no provider calls. New directories are required; existing runs are never overwritten.

`questions.json` is the selected bank; `specs.json` retains requested conditions; `all-candidates.json` retains rejected and unused candidates, judge labels and reasons. `summary.json` states whether every requested slot was filled and lists any missing IDs. An incomplete bank is a partial research result, not a successful fixed-size benchmark. Callers must check `complete` before using it as a full form. `--fidelity-mode strict` additionally requires the exact sampled reasoning variant; default intent mode permits only variants in the source cell's declared intent set. Both record exact requested-versus-realized fidelity. Probe mode always uses strict matching.

The optional `--controls-from` reuses one recorded control response only when its entire API request is identical. This avoids paying for the same development probe again; provenance is saved and it is not counted as fresh live evidence. Fresh runs normally omit it. `--key-file` parses a single credential assignment for local use; it never sources shell code or records credentials in request artifacts.

## The axes and their boundaries

| Layer | What varies | Compatibility boundary |
| --- | --- | --- |
| Category | Onchain spot, crypto majors, commodities, equities | Four reference categories have equal mass. |
| Source intent | All 32 starting questions | At count 64 each receives two slots; at count 32 each receives one. |
| Parent task | Token diligence, funding/OI, technical setup, flows, relative value, earnings/outlook and other retail operations | The reference profile preserves its intent decomposition. The exploration profile samples additional category-compatible tasks. |
| Reasoning variant | Up to six structural information needs per parent task | Source cells allow only declared related variants; parent/child pairs are schema-enforced. |
| Asset scope | Single asset, pair, basket, thematic universe | Uses only the supplied category catalog; a narrow MSTR/BTC relationship exception is explicit. No automatic chain/address resolution. |
| Premise status | Open query, explicit hypothetical, attributed claim check | No verified market facts supplied. No invented actual schedules or price moves. |
| Time | Current, trailing, historical-event, prospective window; separate trade holding period | Historical event studies and current book snapshots have explicit constraints. |
| Trader context | Synthetic budget, loss limit, leverage, concentration/liquidity constraint, risk appetite | Context is labeled synthetic; it is not an observed account or an effect permission. |
| Evidence | Ordinary, primary-source, conflicting-evidence, missing/stale-data; compatible source surface | Technical setups use price data, weather uses suitable forecasts, holders/permissions use onchain evidence. |
| Answer form | Plain, bullets, ranking, table | The requested question must actually ask for the selected format. |
| Voice/difficulty | Casual/plain/skeptical; basic/intermediate/advanced | These are soft controls with model diagnostics, not independently validated traits. |
| Intent structure | One request or a tightly linked followup | Linked requests must stay within the same underlying task. |

There are 25 parent tasks and 151 listed reasoning levels in this design registry. These counts are vocabulary size, not a proof of 151 independent capabilities or a claim of unbounded usable questions. Many axes are conditional or correlated. The reference profile activates a subset; the broader profile does not inherit all original 32 intent quotas.

## How randomization and RAC interact

The reference profile randomizes the order of exact source-intent strata. The exploration profile allocates category quotas, then samples compatible task strata. Premise quotas are assigned by randomized bipartite matching so impossible cross-products are not generated. When rounded premise quotas cannot fit a very small selected task mix, the manifest explicitly records conditional sampling instead of pretending the original marginal held.

Each slot gets eight independently sampled compatible proposals. Their text descriptions encode the reasoning contract, evidence operation, premise type and output shape; asset names and decorative wording are excluded from the spec embedding so changing a ticker alone does not earn novelty. RAC coverage finds representative proposals. Weighted random selection then favors underused task/variant cells with a bounded semantic-gap term. Root source/category quotas stay fixed; lower-axis frequencies are affected by this policy and must be measured, not described as IID samples from a retail population.

The generator produces two phrasings per selected spec. A separate call sees opaque candidate IDs and the taxonomy, but does not see target specs. It classifies the actual behavior, category, premise type, literal instruments, format and other traits. Strict JSON Schema constrains behavior to a qualified parent/variant ID, avoiding invalid combinations such as borrowing a token-diligence variant under trade selection. A valid JSON response still requires semantic checks.

Question gates require financial validity, source/task fit, natural wording, no effects, no unsupported market premise, correct parent/category/premise/format and literal assigned-symbol binding. Unknown catalog names, invented symbol extractions and narrow explicit schedule/percentage assertions are rejected by code. These checks do not establish the truth of a market premise. The default intent gate accepts a child variant only within the source cell's declared intent set; exact-variant fidelity is still reported. It must not be mistaken for exact realization of every sampled child.

RAC representative selection and a bounded novelty preference choose among eligible phrasings. Similarity at least 0.90 to a selected question is rejected. Selection favors underrepresented **realized** behaviors as well as semantic gap. Rejected slots receive bounded retries with recorded failure reasons and compatible subordinate-axis resampling; source intent/category/task remain fixed. The loop does not silently relax gates, invent data or fill a missing slot with an unrelated task.

## What the catalog does not establish

Onchain entries come from user-supplied symbols/names: SI, autoton, ABYSS, claudia, STONK, NIL, GRASS, DRV and PRL. Their chains/contracts were not supplied. The generator must leave that unresolved and must not invent a verified listing or choose a same-symbol token as if its identity were known. DRV's Hyperliquid-spot wording is a source-provided hint, not a verified live listing.

Commodity identifiers denote gold, silver, crude, natural gas, copper, Brent and WTI references. They do not specify a futures contract, broker product or leveraged venue. SPX is an index reference, not a single stock. MSTR/BTC is a reference relationship to investigate, not a supplied current holdings observation. A production operator should provide a versioned instrument/venue/entity catalog and data/event manifests before claiming answerability or live coverage. None is silently supplied by model prose.

## Evidence, reproducibility and remaining limits

Unit tests exercise same-seed replay, different-seed variation, all 32 intent quotas, four-category balancing, incompatible pair/source rejection, literal symbol binding, malformed judge responses, duplicate rejection and controlled-intervention invariants. The `probe` command changes premise status or answer form over two fixed backgrounds. It checks realized model labels; it does not by itself prove a causal semantic effect, because model generation noise and judge errors remain.

The model-discovery pass proposes missing axes and refinements, but proposals are not automatically promoted into the registry. It saw a partial design description and no measured saturation corpus. Its claims about saturated cells are not evidence of saturation. We adopted distinct holding-period/source-surface controls and rejected redundant or unsupported interpretations. This is a bounded RAC adaptation with hierarchical expansion and realization diagnostics, not a complete reproduction of the paper's recursive mining/refinement system.

No human question labels are needed during a run. Humans still design the policy, choose/reference the input distribution and inspect development failures between versions. The generator and blind judge use the same model snapshot (`gpt-4.1-mini-2025-04-14`), leaving correlated bias. Embeddings use `text-embedding-3-small`, 256 dimensions. The default provider cap is USD 5 estimated, 500,000 tokens and 80 requests; an optional shared `RQ01_BUDGET_LEDGER` combines sequential experiments under the same ceiling. No concurrent processes should write that ledger. API response storage is disabled; question text still reaches the provider. Costs are estimates using the documented model rates, not reconciled invoices.

For an independent benchmark, freeze the code/configuration/catalog and the evaluated GTE build before an outside operator selects fresh private seeds and generates questions. The development seeds/questions here are already seen. This repository does not implement the GTE answer adapter or establish answer-judge quality, financial truth, universal acceptance or trading value.

Sources: clean RAC `./rac/paper/source.md` §§5.1–5.4 and `./rac/code/calculus.py`; [RAC](https://github.com/thehalleyyoung/rac); [official structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs); [generation model](https://developers.openai.com/api/docs/models/gpt-4.1-mini); [embedding model](https://developers.openai.com/api/docs/models/text-embedding-3-small).

## Retry and rendering policy added during exploration

The current retry policy keeps source intent, parent task and category fixed, but resamples compatible subordinate settings after a failed candidate batch. Every replacement is written to `resampling-history.json`; each candidate carries its own `requested_spec`. This avoids confusing a failed original spec with a later successful revision. Premise/style/variant marginals after rejection can differ from initial proposal quotas, and the final bank is acceptance-conditioned. It is not IID data or a claim about the retail population.

A small deterministic renderer appends an explicit bullet/table/ranking request when the model omitted it. It preserves the raw stem in `raw_prompt`; requests/responses retain the original model output. Format-control results therefore measure the whole generator pipeline, not an independently learned LLM capability. Instrument names are masked in reference intent/voice hints, and source IDs are kept out of generation contracts, so the original STONK example does not override a randomly assigned token. Full source IDs stay in local output provenance. Literal matching binds extracted identifiers to question text; an LLM-extracted name alone is insufficient.

`resume --resume-run <prior-run>` rechecks inherited selected questions against current gates, retains old candidates and regenerates only unfilled slots in a new directory. It records prior hashes and is a development continuation, not a new independent benchmark. `--rounds` accepts 1–3 and never loosens question gates. Incomplete `run`/`resume` commands write their artifacts and exit 2. Offline replay preserves that status, including an unfilled slot; it does not manufacture a full bank.

The diagnostic probe uses a separate compact blind classification call for parent task, premise status and requested answer form. It saves `probe-candidates.json` and `probe-summary.json`. It is not the complete validity filter and does not export an approved benchmark bank. `--probe-backgrounds 1` permits a bounded single-background diagnostic; the default is two. No causal generalization follows from one background or one realization.

For tuning beyond the defaults, provide a versioned `--axes` JSON. Profiles may specify `source_weights` over reference IDs and `task_weights_by_category` over compatible parent tasks; unknown/incompatible keys fail instead of silently widening the universe. The broad profile also has explicit category weights. The fixture, config hash, seed, as-of and actual accepted marginals must accompany any result.

## Measured development receipt, 22 October 2026

The final current sampler policy is `financial-axes-v2.10-four-categories`. Twenty V2 engineering tests and five preceding RQ01 control tests passed, zero filtered. Across 100 offline seeds, 6,400 proposed specs included 6,302 unique specifications; every 64-slot reference plan preserved 16 slots per category and two per source intent. This measures specification sampling, not semantic uniqueness of generated questions.

The paid development sequence ended with 31 selected questions from a 138-candidate continuation pool: seven onchain spot questions and eight in each other category. It covered 31 of the 32 source intents, 13 parent tasks and 27 model-classified reasoning behaviors. The remaining slot is the `oc-stonk` long-selection intent. It failed on combinations of claimed premises, instrument binding, excessive length and parent-task drift; no replacement outside that intent was silently inserted. The bank is incomplete, and current run/resume commands return exit 2 in that state. Initial strict matching produced 19/32 from 98 candidates; later intent-mode generation, compatible resampling and format rendering reached 31. Seeds, instructions, policies and candidate budgets changed, so this is not a controlled estimate of an improvement caused by one component.

The one-background intervention diagnostic generated 12 question candidates over six specifications. Blind labels matched premise and answer form on 12/12, while the intended parent task survived on 8/12. Format suffixes were rendered by code, so the form result is not evidence of a learned latent capability. The current production policy restricts full-analysis/event-study/long-selection and other task/premise combinations accordingly. That last restriction has offline controls; a fresh paid full-bank run of the tightened final policy was not executed.

Saved-response replay of the final continuation, using its frozen config, reproduced exactly the same selected IDs/question strings and the same incomplete/exit-2 status without provider calls. Repeated control responses were explicitly cached only for byte-identical requests. A failed early attempt dropped unknown invalid-case enum labels, and later JSON mode allowed an impossible parent/variant combination; the strict qualified behavior schema fixed that contract. These failures remain in their artifact directories. A source-name anchoring failure led to masked reference hints and a regression test preserving output source provenance.

Total V2 provider usage, including failed controls, exploratory generation, discovery and the probe: 79 successful requests, 471,464 tokens, estimated USD 0.23532794, within the declared USD 5 / 500,000-token / 80-request exploration bounds. This excludes the earlier V1 trial. No further paid work was run after the diagnostic. Artifacts and exact commands are recorded under `./output/financial-question-generator-v2/` relative to the portfolio workspace, with the local card/ledger as the evidence and decision record. No push, PR, merge or deployment.
