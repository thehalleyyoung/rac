"""
Fetch real, human-authored exam-item banks as head-to-head baselines.

We compare our synthetic psychometric bank against item banks that humans
wrote and that the field actually uses. This is a harder and more meaningful
bar than any synthetic baseline: MMLU's items were drawn from real practice
exams and textbooks across 57 subjects, and ARC's from grade-school science
exams. If a synthetic bank matches or beats human-written banks on
per-item diversity, that is worth knowing; if it does not, the gap is the
honest measure of how far automatic item generation still has to go.

Downloaded via HuggingFace's datasets-server JSON rows endpoint, because
this environment has no pyarrow and the canonical distributions are parquet.
The endpoint paginates at 100 rows per request, so this is deliberately
polite: sequential, with a small delay, and fully resumable (an existing
output file is never refetched).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "public_corpora"
OUT.mkdir(exist_ok=True)
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def fetch_rows(dataset: str, config: str, split: str, limit: int) -> list[dict]:
    got = []
    offset = 0
    while len(got) < limit:
        q = urllib.parse.urlencode({
            "dataset": dataset, "config": config, "split": split,
            "offset": offset, "length": min(PAGE, limit - len(got))})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{ROWS_URL}?{q}", timeout=90) as r:
                    payload = json.load(r)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  giving up at offset {offset}: {str(e)[:120]}")
                    return got
                time.sleep(2 ** attempt)
        rows = payload.get("rows", [])
        if not rows:
            break
        got.extend(x["row"] for x in rows)
        offset += len(rows)
        if len(got) % 1000 < PAGE:
            print(f"  {dataset}/{config}: {len(got)}", flush=True)
        time.sleep(0.15)
    return got


def mmlu(limit: int = 14000) -> list[str]:
    """MMLU 'all' test split: 14,042 human-written multiple-choice items."""
    rows = fetch_rows("cais/mmlu", "all", "test", limit)
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        ch = r.get("choices") or []
        if q and len(ch) == 4:
            out.append(q + "\n" + "\n".join(
                f"{k}. {str(c).strip()}" for k, c in zip("ABCD", ch)))
    return out


def arc(limit: int = 7000) -> list[str]:
    out = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for split in ("train", "test", "validation"):
            rows = fetch_rows("allenai/ai2_arc", cfg, split, limit // 2)
            for r in rows:
                q = (r.get("question") or "").strip()
                ch = (r.get("choices") or {})
                texts = ch.get("text") or []
                labels = ch.get("label") or []
                if q and len(texts) == 4:
                    out.append(q + "\n" + "\n".join(
                        f"{l}. {t}" for l, t in zip(labels, texts)))
            if len(out) >= limit:
                return out[:limit]
    return out


TARGETS = {"mmlu": mmlu, "arc": arc}


def main():
    which = sys.argv[1:] or list(TARGETS)
    for name in which:
        path = OUT / f"{name}_items.json"
        if path.exists():
            n = len(json.loads(path.read_text()))
            print(f"{name}: already have {n} items")
            continue
        print(f"fetching {name} ...", flush=True)
        items = TARGETS[name]()
        path.write_text(json.dumps(items))
        print(f"{name}: {len(items)} items -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
