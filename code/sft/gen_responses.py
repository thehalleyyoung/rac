"""Write responses for every instruction arm with one responder.

Fairness turns on this: if our instructions get gpt-5.6-luna answers and Alpaca
keeps its text-davinci-003 answers, a win could be response quality rather than
instruction diversity. Every arm compared on tokens is answered here by the same
model at the same settings, Alpaca included.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = "openai/gpt-5.6-luna"
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]
LOCK = threading.Lock()
SPEND = {"in": 0, "out": 0, "calls": 0}

SYS = ("You are a helpful assistant. Answer the user's instruction directly and "
       "completely. Do not restate the instruction, do not add pleasantries, and "
       "do not explain that you are an AI. Keep the answer as long as the task "
       "genuinely requires and no longer.")


def one(instr: str, retries: int = 4) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": instr[:4000]}],
        "temperature": 0.7, "max_tokens": 700,
    }).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=body,
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            u = d.get("usage", {})
            with LOCK:
                SPEND["in"] += u.get("prompt_tokens", 0)
                SPEND["out"] += u.get("completion_tokens", 0)
                SPEND["calls"] += 1
            return (d["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            if a == retries - 1:
                print(f"  fail: {str(e)[:90]}", flush=True)
                return None
            import time
            time.sleep(2 * (a + 1))
    return None


def run(arm: str, instrs: list[str], workers: int = 16) -> None:
    """Append-only and resumable: a kill mid-run costs nothing already paid for."""
    out = DATA / f"{arm}.jsonl"
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["instruction"]] = r
    todo = [i for i in instrs if i not in done]
    print(f"{arm}: {len(done)} done, {len(todo)} to write", flush=True)
    if not todo:
        return
    fh = open(out, "a")
    from concurrent.futures import ThreadPoolExecutor
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for instr, resp in zip(todo, ex.map(one, todo)):
            if resp:
                with LOCK:
                    fh.write(json.dumps({"instruction": instr, "response": resp}) + "\n")
                    fh.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {arm}: {n}/{len(todo)}  calls={SPEND['calls']} "
                      f"tok_out={SPEND['out']}", flush=True)
    fh.close()


def dedupe(path) -> int:
    """Collapse repeated instructions; a resumed or duplicated writer can append
    the same key twice, and an SFT arm must not silently gain copies."""
    import json as _j
    if not path.exists():
        return 0
    seen, keep = set(), []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = _j.loads(line)
        if r["instruction"] in seen:
            continue
        seen.add(r["instruction"])
        keep.append(line)
    path.write_text("\n".join(keep) + "\n")
    return len(keep)


def main():
    which = sys.argv[1:] or ["ours_v4", "self_instruct", "random_seed",
                             "ours_keepall", "ours_v3", "ours_sel", "alpaca_tokmatch"]
    rng = random.Random(20260901)
    for arm in which:
        if arm == "alpaca_tokmatch":
            # enough Alpaca to reach the largest token budget, nested so every
            # smaller budget is a prefix of this one
            rows = [json.loads(l) for l in
                    (DATA / "arm_alpaca_full.jsonl").read_text().splitlines() if l.strip()]
            idx = list(range(len(rows)))
            rng.shuffle(idx)
            instrs = [rows[i]["instruction"] for i in idx[:26000]]
        else:
            f = DATA / f"instr_{arm}.jsonl"
            if not f.exists():
                print(f"skip {arm}: no instruction file")
                continue
            instrs = [json.loads(l)["instruction"] for l in
                      f.read_text().splitlines() if l.strip()]
        run(arm, instrs)
        print(f"  running spend: {SPEND['calls']} calls, "
              f"{SPEND['in']} in / {SPEND['out']} out tokens", flush=True)


if __name__ == "__main__":
    main()
