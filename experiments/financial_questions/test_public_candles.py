"""Loopback/synthetic protocol controls; these fixtures are not market evidence."""
import copy
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import public_candles as pc
from answer_records import report_run
from benchmark_cases import write_json
from case_manifest import ContractError, canonical_bytes, digest
from test_answer_records import control, judge_control

START = int(dt.datetime(2026, 10, 23, 11, 50, tzinfo=dt.timezone.utc).timestamp() * 1000)
REQUEST = pc.request_for("BTC", START, START + 2 * pc.MINUTE)
CLOCKS = ["2026-10-23T11:55:00Z", "2026-10-23T11:55:01Z"]


def synthetic_candles():
    return [{"t": START + i * pc.MINUTE, "T": START + (i + 1) * pc.MINUTE - 1,
             "s": "BTC", "i": "1m", "o": "100.00000000000000001", "h": "103.0000",
             "l": "99.0000", "c": "102.0000" if i == 0 else "99.5000",
             "v": "1.0000000000000000001", "n": 7} for i in range(2)]


class Loopback:
    def __init__(self, raw=None, status=200, content_type="application/json"):
        self.raw = canonical_bytes(synthetic_candles()) if raw is None else raw
        self.status, self.content_type, self.requests = status, content_type, []

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                owner.requests.append((self.command, self.path, dict(self.headers), self.rfile.read(length)))
                self.send_response(owner.status)
                self.send_header("Content-Type", owner.content_type)
                self.send_header("Content-Length", str(len(owner.raw)))
                if owner.status == 302:
                    self.send_header("Location", "https://unapproved.example/steal")
                self.end_headers()
                self.wfile.write(owner.raw)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def connection(self, host, timeout):
        assert host == pc.HOST and timeout == pc.TIMEOUT
        return http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=timeout)

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class PublicCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "capture"

    def tearDown(self):
        self.tmp.cleanup()

    def capture_fixture(self, raw=None, status=200, content_type="application/json"):
        with Loopback(raw, status, content_type) as server:
            with patch.object(pc.http.client, "HTTPSConnection", side_effect=server.connection), patch.object(pc, "now", side_effect=CLOCKS):
                manifest = pc.capture(copy.deepcopy(REQUEST), self.path)
        return manifest, server

    def test_actual_loopback_http_exchange_preserves_exact_bytes_and_fixed_request(self):
        raw = json.dumps(synthetic_candles(), indent=2).encode() + b"\n"
        manifest, server = self.capture_fixture(raw)
        self.assertEqual(manifest["state"], "captured")
        self.assertEqual((self.path / "response.bin").read_bytes(), raw)
        self.assertEqual(len(server.requests), 1)
        method, path, headers, body = server.requests[0]
        self.assertEqual((method, path, body), ("POST", "/info", canonical_bytes(REQUEST)))
        self.assertFalse(any(k.lower() in {"authorization", "cookie"} for k in headers))
        m, actual, candles = pc.verify_capture(self.path)
        self.assertEqual(actual, raw)
        self.assertEqual(candles, synthetic_candles())
        self.assertEqual(m["known_at"], CLOCKS[-1])
        self.assertEqual(candles[0]["o"], "100.00000000000000001")

    def test_arbitrary_kind_coin_interval_destination_fields_and_budget_reject_before_network(self):
        with patch.object(pc.http.client, "HTTPSConnection", side_effect=AssertionError("must not connect")):
            for request in [dict(REQUEST, type="order"), dict(REQUEST, endpoint="https://other"),
                            dict(REQUEST, authorization="secret"),
                            {"type": "candleSnapshot", "req": dict(REQUEST["req"], coin="unresolved-token")},
                            {"type": "candleSnapshot", "req": dict(REQUEST["req"], interval="1h")}]:
                with self.subTest(request=request), self.assertRaises((ContractError, TypeError)):
                    pc.fetch(request)
        for start, end in [(START + 1, START + pc.MINUTE), (START, START + 31 * pc.MINUTE),
                           (True, START), (START, START)]:
            with self.assertRaises(ContractError):
                pc.request_for("BTC", start, end)

    def test_future_or_open_window_is_refused_without_creating_attempt(self):
        with patch.object(pc, "now", return_value="2026-10-23T11:50:30Z"), patch.object(pc, "fetch") as fetch:
            with self.assertRaisesRegex(ContractError, "fully closed"):
                pc.capture(REQUEST, self.path)
        fetch.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_redirect_is_retained_as_failure_and_never_followed(self):
        manifest, server = self.capture_fixture(b"redirect-body", 302)
        self.assertEqual(len(server.requests), 1)
        self.assertEqual(manifest["state"], "capture_failed")
        self.assertEqual((self.path / "response.bin").read_bytes(), b"redirect-body")
        with self.assertRaisesRegex(ContractError, "did not succeed"):
            pc.verify_capture(self.path)

    def test_oversized_body_is_bounded_retained_and_not_exported(self):
        manifest, _ = self.capture_fixture(b"x" * (pc.MAX_BYTES + 10))
        self.assertEqual(manifest["state"], "capture_failed")
        self.assertEqual((self.path / "response.bin").stat().st_size, pc.MAX_BYTES + 1)
        self.assertIn("response size", manifest["error"])

    def test_transport_failure_has_a_retained_attempt_and_no_provider_observation(self):
        with patch.object(pc, "fetch", side_effect=TimeoutError("local synthetic timeout")), patch.object(pc, "now", side_effect=CLOCKS):
            m = pc.capture(REQUEST, self.path)
        self.assertEqual(m["state"], "capture_failed")
        self.assertEqual(m["candle_count"], 0)
        self.assertEqual((self.path / "response.bin").read_bytes(), b"")
        self.assertIsNone(m["http_status"])

    def test_bad_content_type_http_error_json_duplicates_and_nonfinite_values_fail(self):
        for index, (raw, status, content_type) in enumerate([
            (b"[]", 503, "application/json"), (b"[]", 200, "text/html"),
            (b"[NaN]", 200, "application/json"), (b'{"a":1,"a":2}', 200, "application/json")]):
            self.path = self.root / ("bad-" + str(index))
            m, _ = self.capture_fixture(raw, status, content_type)
            self.assertEqual(m["state"], "capture_failed")

    def test_instrument_interval_window_gap_order_and_ohlcv_negatives(self):
        original = synthetic_candles()
        mutations = [lambda c: c[0].update(s="ETH"), lambda c: c[0].update(i="5m"),
                     lambda c: c[0].update(t=START + 1), lambda c: c[0].update(T=START + pc.MINUTE),
                     lambda c: c.reverse(), lambda c: c.pop(), lambda c: c.append(c[0]),
                     lambda c: c[0].update(o=100.0), lambda c: c[0].update(v="NaN"),
                     lambda c: c[0].update(l="104"), lambda c: c[0].update(n=True)]
        for mutate in mutations:
            candles = copy.deepcopy(original)
            mutate(candles)
            with self.assertRaises((ContractError, ValueError)):
                pc.validate_candles(canonical_bytes(candles), REQUEST)

    def test_mutated_raw_capture_request_manifest_or_clock_cannot_verify(self):
        self.capture_fixture()
        originals = {p.name: p.read_bytes() for p in self.path.iterdir()}
        for name in ("request.json", "response.bin", "manifest.json"):
            (self.path / name).write_bytes(originals[name] + b" ")
            if name == "manifest.json":
                # JSON whitespace does not change manifest meaning or original raw values.
                self.assertEqual(pc.verify_capture(self.path)[0]["state"], "captured")
            else:
                with self.assertRaises(ContractError):
                    pc.verify_capture(self.path)
            (self.path / name).write_bytes(originals[name])
        m = json.loads(originals["manifest.json"])
        m["known_at"] = "2026-10-23T11:49:00Z"
        m["capture_id"] = digest({k: v for k, v in m.items() if k != "capture_id"})
        write_json(self.path / "manifest.json", m)
        with self.assertRaisesRegex(ContractError, "known-time"):
            pc.verify_capture(self.path)

    def test_exact_replay_positive_and_validly_labeled_synthetic_response_negative(self):
        self.capture_fixture()
        original = synthetic_candles()
        self.assertTrue(pc.check_response(self.path, json.dumps(original, indent=2).encode(), REQUEST)["matches"])
        original[0]["v"] = "9.0000000000000000001"
        with self.assertRaisesRegex(ContractError, "differs from original"):
            pc.check_response(self.path, canonical_bytes(original), REQUEST)
        with self.assertRaisesRegex(ContractError, "request instrument"):
            pc.check_response(self.path, canonical_bytes(original), pc.request_for("ETH", START, START + 2 * pc.MINUTE))

    def test_export_never_backdates_known_time_to_the_observation_window(self):
        self.capture_fixture()
        with self.assertRaisesRegex(ContractError, "not known"):
            pc.export_replay(self.path, "2026-10-23T11:52:00Z")
        with patch.object(pc, "now", side_effect=["2026-10-23T12:01:00Z", "2026-10-23T12:01:01Z"]):
            result = pc.export_replay(self.path, "2026-10-23T12:00:00Z")
        self.assertTrue(result["replay"])
        e, t = result["evidence"][0], result["tool_trace"][0]
        self.assertEqual(e["known_at"], CLOCKS[-1])
        self.assertEqual(e["observed_at"], "2026-10-23T11:51:59.999Z")
        self.assertEqual(e["content"].encode(), (self.path / "response.bin").read_bytes())
        self.assertEqual(t["response"], e["content"])
        self.assertEqual(t["access_mode"], "read_only")

    def test_export_reaches_existing_answer_and_two_judge_report_consumer(self):
        self.capture_fixture()
        with patch.object(pc, "now", side_effect=["2026-10-23T12:01:00Z", "2026-10-23T12:01:01Z"]):
            exported = pc.export_replay(self.path, "2026-10-23T12:00:00Z")
        root = self.root / "answers"
        root.mkdir()
        bundle, plan, attempts, _ = control(root)
        plan["read_tools"] = ["public_candle_replay"]
        plan["evidence_policy"]["end_to_end"]["capture_manifest_sha256"] = exported["evidence"][0]["capture_manifest_sha256"]
        a = attempts[0]
        a.update(plan_sha256=digest(plan), evidence=exported["evidence"], tool_trace=exported["tool_trace"],
                 evidence_refs=[exported["evidence"][0]["evidence_id"]])
        judges = [judge_control(a, j, plan) for j in plan["judges"]]
        r = report_run(bundle, plan, attempts, judges)
        self.assertTrue(r["ready_for_pilot_interpretation"], r["protocol_errors"])
        self.assertEqual(len(r["answers"][0]["judge_records"]), 2)
        self.assertIsNone(r["scalar_score"])

    def test_existing_capture_and_export_are_not_overwritten_and_cli_verifies_offline(self):
        self.capture_fixture()
        before = (self.path / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            pc.capture(REQUEST, self.path)
        self.assertEqual((self.path / "manifest.json").read_bytes(), before)
        result = subprocess.run([sys.executable, str(Path(pc.__file__)), "verify", str(self.path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = self.root / "export.json"
        command = [sys.executable, str(Path(pc.__file__)), "export", str(self.path), "--cutoff", "2026-10-23T12:00:00Z", "--out", str(out)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        data = out.read_bytes()
        self.assertEqual(subprocess.run(command, capture_output=True, text=True).returncode, 1)
        self.assertEqual(out.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
