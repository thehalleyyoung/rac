# Public candle capture and replay

RQ01 research adapter for one bounded public read. It does not call Edge,
authenticate an account, run a model, or access a trading endpoint.

## Fixed read boundary

The collector sends only `POST https://api.hyperliquid.xyz/info` with
`type=candleSnapshot`. The request names BTC or ETH **perps**, interval `1m`,
and an explicit aligned observation window of 1–30 closed minutes. The CLI
takes an exclusive end time and sends `endTime=end-1`. It accepts no arbitrary
URL, request type, headers, credentials, account ID, interval or symbol.

This is a narrow implementation of the provider's
[candle snapshot contract](https://hyperliquid.gitbook.io/Hyperliquid-docs/for-developers/api/info-endpoint#candle-snapshot),
checked 23 October 2026. It is not a general catalog resolver or a statement
that all asset × instrument combinations are available. Sparse or incomplete
windows fail; there is no silent interpolation or substitution.

The standard-library TLS connection verifies certificates, bypasses environment
proxy forwarding, and follows no redirects. Timeout is 10 seconds and response
read size is capped at 262,145 bytes (one byte over the 262,144-byte acceptance
limit). No retry occurs automatically. Each attempted capture gets a new
directory; an existing directory is never overwritten. A failed attempt retains
the request, received response bytes and failure manifest. The bound is a socket
timeout, not an independently enforced total wall-clock deadline against an
arbitrarily slow server. TLS settings are unchanged.

## Records and clocks

Files are `request.json`, `response.bin`, and `manifest.json`. The manifest
retains provider/destination, method, request, response status/content type,
request start, retrieval time, response size, original byte hashes and capture
identity. A successful capture requires every expected candle in original
order with matching symbol/interval and exact open/close milliseconds. OHLCV
must remain strings; Decimal is used only to check validity, never to rewrite
them. No normalized or generated candle replaces the original response.

`known_at` is conservatively the actual retrieval time. A historical candle
timestamp does not establish that the benchmark knew it then. Export refuses a
case cutoff before retrieval, even if the observation window ended earlier.
Publication time remains unknown rather than inferred. A capture of an already
closed window is historical replay, not a future holdout.

`verify` checks the stored request/response bytes, manifest identity, clock
relationships, original values and complete observation window. `check-response`
starts from that verified capture and compares a separately supplied response
against every original timestamp, string OHLCV and other candle field for the
same requested instrument/interval/window. JSON layout may differ, values may
not. An apparently valid synthetic response with changed volume is rejected.
Checksums provide local integrity, not independent provider attestation: someone
who replaces all files and recomputes every hash can manufacture a new capture.
Independent custody/append-only publication remains future work.

## Existing answer-record handoff

`export` emits the existing evidence shape plus a read-only replay tool trace.
The evidence keeps original raw response content and retrieval/known times;
the replay call has its own actual start/finish clocks. `replay=true` is explicit.
Its manifest SHA is suitable for the existing end-to-end declared replay policy;
the same evidence packet can instead be pinned by the controlled-mode policy.
The output contains no answer or judge score. A replay trace is not a claim that
Edge or a live provider was called during that answering attempt.

## Commands

Example using the already recorded, actual public window:

```sh
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/public_candles.py verify \
  ./output/rq01-public-capture/BTC-1m-20261023T1754-1757
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/public_candles.py check-response \
  ./output/rq01-public-capture/BTC-1m-20261023T1754-1757 \
  --request ./output/rq01-public-capture/BTC-1m-20261023T1754-1757/request.json \
  --response ./output/rq01-public-capture/BTC-1m-20261023T1754-1757/response.bin
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/public_candles.py export \
  ./output/rq01-public-capture/BTC-1m-20261023T1754-1757 \
  --cutoff 2026-10-23T18:00:00Z --out ./output/new-public-evidence-export.json
```

Only `capture` uses the network. Choose a new output directory and a recent,
fully closed window still available from the provider:

```sh
python3 ./worktrees/rac-retail-question-distribution/experiments/financial_questions/public_candles.py capture \
  --coin BTC --start-ms 1792778040000 --end-ms 1792778220000 \
  --out ./output/new-public-capture-attempt
```

The example timestamps describe 23 October 2026, 17:54–17:57 UTC. They are not
automatically updated and may eventually be unavailable. The provider bounds
historical availability; a failed future replay request must stay a failure.
Capture exits 0 on success or 2 with a retained failed attempt; invalid input,
verification/export failure or IO errors exit 1. Export uses exclusive create.

## Evidence and limits

Thirteen synthetic controls exercise actual loopback HTTP, fixed request shape,
credential absence, redirects, size/clock/data failures, every original decimal,
altered-response refusal and integration into the existing answer/judge report.
They are labelled synthetic in the tests; they do not prove live market reads.
The separately retained actual HTTPS receipt contains three original BTC candles,
retrieved at 17:59:04.899 UTC on 23 October 2026. A disclosed scripted report
replays that real input; its answer and two judge-family records remain synthetic.
The report computes no scalar or live answer-quality result.

There is no chart/paint/History evidence in this slice. Future visible charts
must independently satisfy the workspace's HTTP-response, original-value,
painted-candle, animation-clock and saved-History/privacy requirements.

Clean source inspection at monorepo ffdd017f126a5b7a15e2539d502d858f47b364c1
(`quant-service-core/src/chat_request.rs` and `ui_stream.rs`) found ordinary chat
inputs/UI tool events, not a checked benchmark-only effect restriction or a full
original provider-byte/time contract. This collector therefore does not call the
chat route. Authenticated Edge capture, independent catalog identity, blind model
judges, workflow/rubric coverage, calibration and independent administration remain
open. No canonical authority/writer/wire is changed. Rollback is to stop consuming
this additive adapter while retaining historical captures, cases and answers.
