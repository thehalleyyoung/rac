"""One screen showing every experiment at once.

Reads the run logs directly rather than the volume, so it is instant and works
while everything is still in flight.
"""
from __future__ import annotations

import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_ROUGE = 0.1419          # untrained Qwen2.5-0.5B on held-out Dolly

ROW = re.compile(r"^\s+(\S+)\s+seed=(\d+).*?rougeL=([\d.]+)"
                 r"(?:.*?cap=([\d.]+))?(?:.*?len=\s*(\d+))?")
SEM = re.compile(r"sem=([\d.]+)")
LR = re.compile(r"^\s+lr=([\d.e-]+)\s+ep=(\d+) rougeL=([\d.]+).*?len=\s*(\d+)")


def read(log: str):
    p = HERE / log
    return p.read_text().splitlines() if p.exists() else []


def bar(v, lo, hi, w=22):
    if v != v:
        return " " * w
    f = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    n = int(round(f * w))
    return "█" * n + "·" * (w - n)


def arms(log: str, strip: str):
    out = {}
    for l in read(log):
        m = ROW.match(l)
        if not m or "lr=" in l:
            continue
        a = m.group(1).replace(strip, "", 1)
        sem = SEM.search(l)
        out.setdefault(a, []).append(
            (float(m.group(3)), float(sem.group(1)) if sem else float("nan"),
             int(m.group(5) or 0)))
    return out


def block(title, log, strip):
    a = arms(log, strip)
    n = sum(len(v) for v in a.values())
    print(f"\n{title}  ({n} runs)")
    if not a:
        print("   (no results yet)")
        return
    lo, hi = 0.12, 0.22
    print(f"   {'arm':20s} {'rougeL':>7} {'±':>6} {'sem':>6} {'len':>4}  "
          f"{'vs base':>8}  {'':22}")
    for arm, v in sorted(a.items(), key=lambda kv: -st.mean([x[0] for x in kv[1]])):
        r = [x[0] for x in v]
        s = [x[1] for x in v if x[1] == x[1]]
        m = st.mean(r)
        sd = st.stdev(r) if len(r) > 1 else 0.0
        mark = "  ours" if arm.startswith("ours") else ""
        print(f"   {arm:20s} {m:7.4f} {sd:6.4f} "
              f"{(st.mean(s) if s else float('nan')):6.3f} "
              f"{st.mean([x[2] for x in v]):4.0f}  {m - BASE_ROUGE:+8.4f}  "
              f"{bar(m, lo, hi)}{mark}")


def lrblock():
    rows = []
    for l in read("lrsweep.log"):
        m = LR.match(l)
        if m:
            rows.append((float(m.group(1)), int(m.group(2)),
                         float(m.group(3)), int(m.group(4))))
    print(f"\nLEARNING-RATE SWEEP on a baseline arm  ({len(rows)}/8)")
    if not rows:
        print("   (no results yet)")
        return
    print(f"   {'lr':>8} {'ep':>3} {'rougeL':>7} {'len':>4}  {'vs base':>8}")
    for lr, ep, v, L in sorted(rows, key=lambda r: -r[2]):
        print(f"   {lr:8g} {ep:3d} {v:7.4f} {L:4d}  {v - BASE_ROUGE:+8.4f}")
    best = max(rows, key=lambda r: r[2])
    print(f"   best: lr={best[0]:g} ep={best[1]} -> {best[2]:.4f}")


def main():
    print("=" * 78)
    print(f"SFT EXPERIMENT BOARD    base model, no SFT: {BASE_ROUGE:.4f}")
    print("=" * 78)
    lrblock()
    block("CROSS-CORPUS (as generated, markdown intact)", "cross.log", "x_")
    block("CROSS-CORPUS (prose-normalized)", "prose.log", "p_")
    block("SCALE CHECK (Qwen2.5-1.5B)", "scale.log", "p_")
    block("FINAL (tuned, 5 seeds)", "final.log", "p_")
    print()


if __name__ == "__main__":
    main()
