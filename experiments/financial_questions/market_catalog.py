"""Offline admission of declared market catalogs, not provider authentication."""
import hashlib
from collections import Counter

from answer_records import fields, require, sha, text
from case_manifest import ASSETS, INSTRUMENTS, digest, parse_json, timestamp

MARKET_FIELDS = "market_id asset_class instrument venue base quote settlement expiry strike resolution_rules"


def audit_catalog(catalog, policy):
    fields(policy, "schema_version minimum_markets asset_minimums instrument_minimums allow_synthetic cutoff", "catalog policy")
    require(policy["schema_version"] == "financial-catalog-policy/v1", "unsupported catalog policy")
    require(type(policy["minimum_markets"]) is int and policy["minimum_markets"] > 16, "catalog must require more than 16 markets")
    require(type(policy["allow_synthetic"]) is bool, "synthetic permission must be explicit")
    cutoff = timestamp(policy["cutoff"])
    for key, allowed in (("asset_minimums", ASSETS - {"mixed"}), ("instrument_minimums", INSTRUMENTS - {"mixed"})):
        require(isinstance(policy[key], dict) and set(policy[key]) <= allowed and
                all(type(v) is int and v > 0 for v in policy[key].values()), "invalid catalog quota")
    require(set(policy["asset_minimums"]) == ASSETS - {"mixed"}, "declare all four asset-class minima")
    fields(catalog, "schema_version catalog_id external_effects_allowed snapshots entries", "market catalog")
    require(catalog["schema_version"] == "financial-market-catalog/v1" and text(catalog["catalog_id"]), "invalid catalog identity")
    require(catalog["external_effects_allowed"] is False, "catalog cannot grant effect authority")
    require(isinstance(catalog["snapshots"], list) and isinstance(catalog["entries"], list), "catalog rows must be lists")
    snapshots = {}
    for snapshot in catalog["snapshots"]:
        fields(snapshot, "snapshot_id kind source_ref observed_at known_at retrieved_at content content_sha256", "listing snapshot")
        require(text(snapshot["snapshot_id"]) and snapshot["snapshot_id"] not in snapshots, "duplicate/missing listing snapshot")
        require(snapshot["kind"] in {"provider_capture", "synthetic_control"} and text(snapshot["source_ref"]), "invalid listing provenance")
        require(isinstance(snapshot["content"], str) and sha(snapshot["content_sha256"]) and
                hashlib.sha256(snapshot["content"].encode()).hexdigest() == snapshot["content_sha256"], "listing bytes/hash mismatch")
        observed, known, retrieved = [timestamp(snapshot[k]) for k in ("observed_at", "known_at", "retrieved_at")]
        require(observed <= known <= cutoff and known <= retrieved, "listing knowledge not cutoff eligible")
        rows = parse_json(snapshot["content"])
        require(isinstance(rows, list), "listing adapter requires an original JSON array")
        snapshots[snapshot["snapshot_id"]] = (snapshot, rows)
    errors, eligible, seen, resolved_keys = [], [], set(), set()
    for index, row in enumerate(catalog["entries"]):
        try:
            fields(row, MARKET_FIELDS + " identity_status snapshot_id source_row valid_from valid_until", "catalog entry")
            require(all(text(row[k]) for k in ("market_id", "venue", "base", "quote", "settlement")), "missing stable market identity/units")
            require(row["market_id"] not in seen, "duplicate market ID")
            seen.add(row["market_id"])
            require(row["identity_status"] == "resolved", "unresolved identity cannot count as coverage")
            require(row["asset_class"] in ASSETS - {"mixed"} and row["instrument"] in INSTRUMENTS - {"mixed"}, "invalid market axes")
            require(timestamp(row["valid_from"]) <= cutoff and
                    (row["valid_until"] is None or cutoff < timestamp(row["valid_until"])), "market not valid at cutoff")
            require(row["snapshot_id"] in snapshots and type(row["source_row"]) is int and row["source_row"] >= 0, "unknown listing row")
            snapshot, original = snapshots[row["snapshot_id"]]
            require(row["source_row"] < len(original), "listing row out of bounds")
            require(original[row["source_row"]] == {k: row[k] for k in MARKET_FIELDS.split()}, "catalog differs from original listing row")
            require(snapshot["kind"] != "synthetic_control" or policy["allow_synthetic"], "synthetic listing excluded by run policy")
            if row["instrument"] == "options":
                require(text(row["expiry"]) and text(row["strike"]), "option terms missing")
                require(cutoff < timestamp(row["expiry"]), "option expired at cutoff")
            if row["instrument"] == "prediction_markets":
                require(text(row["resolution_rules"]) and text(row["expiry"]), "prediction-market rules missing")
                require(cutoff < timestamp(row["expiry"]), "prediction market expired at cutoff")
            identity = tuple(row[k] for k in ("venue", "instrument", "base", "quote", "settlement", "expiry", "strike", "resolution_rules"))
            require(identity not in resolved_keys, "duplicate resolved contract under different IDs")
            resolved_keys.add(identity)
            eligible.append(row)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"entry": index, "market_id": row.get("market_id") if isinstance(row, dict) else None, "reason": str(exc)})
    counts = {"assets": dict(Counter(r["asset_class"] for r in eligible)),
              "instruments": dict(Counter(r["instrument"] for r in eligible))}
    gaps = []
    if len(eligible) < policy["minimum_markets"]:
        gaps.append({"axis": "total", "required": policy["minimum_markets"], "actual": len(eligible)})
    for axis, key in (("assets", "asset_minimums"), ("instruments", "instrument_minimums")):
        for label, minimum in policy[key].items():
            if counts[axis].get(label, 0) < minimum:
                gaps.append({"axis": axis, "label": label, "required": minimum, "actual": counts[axis].get(label, 0)})
    return {"schema_version": "financial-catalog-audit/v1", "catalog_sha256": digest(catalog),
            "policy_sha256": digest(policy), "ready": not errors and not gaps,
            "eligible_markets": len(eligible), "counts": counts, "invalid_entries": errors, "coverage_gaps": gaps,
            "evidence_kinds": sorted({s[0]["kind"] for s in snapshots.values()}),
            "limitations": "Declared snapshot consistency only; no independent listing attestation, normalization adapter or low-cap classification verification."}
