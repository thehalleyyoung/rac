"""RQ01 bounded public candle capture. No account, chat, model or trading API."""
import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import http.client
from pathlib import Path
import sys

from answer_records import fields, require
from benchmark_cases import read_rows, write_json
from case_manifest import canonical_bytes, digest, parse_json, timestamp

HOST = "api.hyperliquid.xyz"
PATH = "/info"
ENDPOINT = "https://" + HOST + PATH
MINUTE = 60_000
MAX_CANDLES = 30
MAX_BYTES = 262_144
TIMEOUT = 10


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def raw_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def request_for(coin, start_ms, end_ms):
    require(coin in {"BTC", "ETH"}, "collector supports only explicit BTC/ETH perp candles")
    require(type(start_ms) is int and type(end_ms) is int and 0 <= start_ms < end_ms,
            "invalid observation window")
    require(start_ms % MINUTE == 0 and end_ms % MINUTE == 0, "one-minute window must be aligned")
    require(end_ms - start_ms <= MAX_CANDLES * MINUTE, "observation window exceeds bounded capture")
    return {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1m",
                                              "startTime": start_ms, "endTime": end_ms - 1}}


def validate_request(request):
    fields(request, "type req", "public request")
    fields(request["req"], "coin interval startTime endTime", "candle request")
    r = request["req"]
    require(type(r["endTime"]) is int, "invalid end timestamp")
    require(request == request_for(r["coin"], r["startTime"], r["endTime"] + 1),
            "only fixed candleSnapshot/1m request is allowed")
    return r


def validate_candles(raw, request):
    r = validate_request(request)
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_BYTES, "invalid response size")
    candles = parse_json(raw.decode("utf-8"))
    require(isinstance(candles, list), "provider response is not a candle list")
    expected = list(range(r["startTime"], r["endTime"] + 1, MINUTE))
    require(len(candles) == len(expected), "incomplete or extra candle response")
    for candle, start in zip(candles, expected):
        fields(candle, "t T s i o h l c v n", "provider candle")
        require(type(candle["t"]) is int and type(candle["T"]) is int
                and candle["t"] == start and candle["T"] == start + MINUTE - 1,
                "candle timestamps are missing, reordered or outside the window")
        require(candle["s"] == r["coin"] and candle["i"] == r["interval"], "mislabeled candle instrument/interval")
        values = {}
        for key in ("o", "h", "l", "c", "v"):
            require(isinstance(candle[key], str), "provider OHLCV must retain original decimal strings")
            try:
                values[key] = Decimal(candle[key])
            except InvalidOperation as exc:
                raise ValueError("invalid OHLCV decimal") from exc
            require(values[key].is_finite() and values[key] >= 0, "nonfinite or negative OHLCV")
        require(min(values[k] for k in ("o", "h", "l", "c")) > 0, "nonpositive price")
        require(values["l"] <= min(values["o"], values["c"]) <= max(values["o"], values["c"]) <= values["h"], "invalid candle price range")
        require(type(candle["n"]) is int and candle["n"] >= 0, "invalid trade count")
    return candles


def fetch(request):
    """Only fixed TLS host/path; http.client follows no redirects or env proxies."""
    validate_request(request)
    connection = http.client.HTTPSConnection(HOST, timeout=TIMEOUT)
    try:
        connection.request("POST", PATH, body=canonical_bytes(request),
                           headers={"Content-Type": "application/json", "Accept-Encoding": "identity"})
        response = connection.getresponse()
        raw = response.read(MAX_BYTES + 1)
        return response.status, response.getheader("Content-Type"), raw
    finally:
        connection.close()


def capture(request, out):
    """Retain attempted request and response even if content/status validation fails."""
    r = validate_request(request)
    started = now()
    require(r["endTime"] < int(timestamp(started).timestamp() * 1000), "capture requires a fully closed window")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    request_raw = canonical_bytes(request)
    (out / "request.json").write_bytes(request_raw)
    status, content_type, raw, error = None, None, b"", None
    try:
        status, content_type, raw = fetch(request)
    except (OSError, http.client.HTTPException) as exc:
        error = type(exc).__name__
    retrieved = now()
    (out / "response.bin").write_bytes(raw)
    candles = []
    try:
        require(error is None, "transport failed: " + str(error))
        require(status == 200, "provider returned non-200; redirects are never followed")
        require(content_type is not None and content_type.split(";")[0].strip().lower() == "application/json", "response is not application/json")
        require(timestamp(started) <= timestamp(retrieved), "capture clock regressed")
        candles = validate_candles(raw, request)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        error = str(exc)
    manifest = {"schema_version": "public-candle-capture/v1", "state": "captured" if error is None else "capture_failed",
                "provider": "hyperliquid", "endpoint": ENDPOINT, "method": "POST", "request": request,
                "request_sha256": raw_hash(request_raw), "response_sha256": raw_hash(raw),
                "response_bytes": len(raw), "http_status": status, "content_type": content_type,
                "request_started_at": started, "retrieved_at": retrieved, "known_at": retrieved,
                "candle_count": len(candles), "error": error,
                "meaning": "Public HTTPS observation; local consistency, not independent attestation or forward holdout."}
    manifest["capture_id"] = digest(manifest)
    write_json(out / "manifest.json", manifest)
    return manifest


def verify_capture(path):
    path = Path(path)
    m = read_rows(path / "manifest.json")
    fields(m, "schema_version state provider endpoint method request request_sha256 response_sha256 response_bytes http_status content_type request_started_at retrieved_at known_at candle_count error meaning capture_id", "capture manifest")
    require(m["schema_version"] == "public-candle-capture/v1" and m["state"] == "captured" and m["error"] is None, "capture did not succeed")
    require(m["provider"] == "hyperliquid" and m["endpoint"] == ENDPOINT and m["method"] == "POST", "capture destination/method mismatch")
    require(m["capture_id"] == digest({k: v for k, v in m.items() if k != "capture_id"}), "capture identity mismatch")
    request_raw, raw = (path / "request.json").read_bytes(), (path / "response.bin").read_bytes()
    require(request_raw == canonical_bytes(m["request"]) and raw_hash(request_raw) == m["request_sha256"], "captured request bytes differ")
    require(raw_hash(raw) == m["response_sha256"] and len(raw) == m["response_bytes"], "captured response bytes differ")
    require(m["http_status"] == 200 and m["content_type"].split(";")[0].strip().lower() == "application/json", "invalid captured HTTP result")
    require(timestamp(m["request_started_at"]) <= timestamp(m["retrieved_at"]) and m["known_at"] == m["retrieved_at"], "capture clock/known-time mismatch")
    r = validate_request(m["request"])
    require(r["endTime"] < int(timestamp(m["request_started_at"]).timestamp() * 1000), "capture window was not closed before request")
    candles = validate_candles(raw, m["request"])
    require(len(candles) == m["candle_count"], "capture count mismatch")
    return m, raw, candles


def check_response(path, raw, request):
    """Compare independently supplied replay HTTP bytes against pinned observations."""
    m, original, candles = verify_capture(path)
    require(request == m["request"], "replay request instrument/interval/window differs")
    actual = validate_candles(raw, request)
    require(actual == candles, "replay response differs from original provider timestamps/decimal OHLCV")
    return {"matches": True, "capture_id": m["capture_id"], "candles": len(candles),
            "capture_response_sha256": raw_hash(original), "replay_response_sha256": raw_hash(raw)}


def export_replay(path, cutoff):
    started = now()
    m, raw, candles = verify_capture(path)
    require(timestamp(m["known_at"]) <= timestamp(cutoff), "capture was not known at the case cutoff")
    require(timestamp(m["retrieved_at"]) <= timestamp(started), "capture retrieval is in the replay future")
    body = raw.decode("utf-8")
    evidence_id = "hyperliquid-" + m["capture_id"]
    evidence = {"evidence_id": evidence_id, "kind": "provider_capture", "producer": "hyperliquid public candleSnapshot",
                "source_ref": ENDPOINT, "instrument_id": "hyperliquid:perp:" + m["request"]["req"]["coin"],
                "observed_at": iso(candles[-1]["T"]), "published_at": None,
                "known_at": m["known_at"], "retrieved_at": m["retrieved_at"], "content": body,
                "content_sha256": m["response_sha256"],
                "capture_manifest_sha256": raw_hash((Path(path) / "manifest.json").read_bytes())}
    finished = now()
    require(timestamp(started) <= timestamp(finished), "replay clock regressed")
    trace = {"call_id": "replay-" + m["capture_id"], "tool_name": "public_candle_replay", "access_mode": "read_only",
             "request": (Path(path) / "request.json").read_text(), "response": body,
             "evidence_ids": [evidence_id], "status": "ok", "started_at": started, "finished_at": finished}
    return {"evidence": [evidence], "tool_trace": [trace], "capture_id": m["capture_id"],
            "replay": True, "meaning": "Actual stored public response replay; no answer/model/judge execution."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--coin", choices=["BTC", "ETH"], required=True)
    p.add_argument("--start-ms", type=int, required=True)
    p.add_argument("--end-ms", type=int, required=True, help="exclusive aligned endpoint")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("capture", type=Path)
    p = sub.add_parser("check-response")
    p.add_argument("capture", type=Path)
    p.add_argument("--response", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p = sub.add_parser("export")
    p.add_argument("capture", type=Path)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "capture":
            m = capture(request_for(args.coin, args.start_ms, args.end_ms), args.out)
            print(m["state"] + " " + m["capture_id"])
            return 0 if m["state"] == "captured" else 2
        if args.command == "verify":
            m, _, candles = verify_capture(args.capture)
            print("verified " + m["capture_id"] + " candles=" + str(len(candles)))
        elif args.command == "check-response":
            print(check_response(args.capture, args.response.read_bytes(), read_rows(args.request)))
        else:
            result = export_replay(args.capture, args.cutoff)
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(canonical_bytes(result).decode() + "\n")
            print("exported " + result["capture_id"])
        return 0
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
