"""Package every corpus as an SFT arm: deduplicated instruction/response pairs.

All arms are answered by the same responder, so a difference between them is a
difference in the instructions, which is the thing under test.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent / "data"

ARMS = {
    # ours, from the fair-fight budget
    "ours_v4": "RAC-coverage, retrieval-aimed",
    "ours_keepall": "RAC, conditioning keep-all",
    "ours_v3": "RAC, orthogonalized conditioning",
    "ours_sel": "RAC, selective 1-of-8",
    # published methods, re-run faithfully on the same generator
    "fs_self_instruct": "Self-Instruct (faithful, gpt-5.6-luna)",
    "fs_evol_instruct": "Evol-Instruct (faithful, gpt-5.6-luna)",
    "fs_persona_hub": "Persona-Hub (faithful, gpt-5.6-luna)",
    # released instruction sets, answered by our responder
    "alpaca_tokmatch": "Alpaca instructions (released)",
    "personahub": "PersonaHub instructions (released)",
    "wizardlm": "WizardLM instructions (released)",
}


def main():
    rows_out = {}
    for name, label in ARMS.items():
        f = D / f"{name}.jsonl"
        if not f.exists():
            print(f"  {name:20s} MISSING")
            continue
        seen, keep = set(), []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i, resp = r.get("instruction", "").strip(), r.get("response", "").strip()
            if not i or not resp or i in seen:
                continue
            seen.add(i)
            keep.append({"instruction": i, "response": resp})
        out = D / f"arm_x_{name}.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in keep) + "\n")
        chars = sum(len(r["instruction"]) for r in keep) // max(len(keep), 1)
        rows_out[name] = len(keep)
        print(f"  {name:20s} {len(keep):6d} pairs  mean instr {chars:4d} chars   {label}")
    n = min(v for v in rows_out.values() if v >= 900)
    print(f"\nlargest matched n across arms with >=900 pairs: {n}")
    return rows_out


if __name__ == "__main__":
    main()
