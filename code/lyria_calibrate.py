"""
Calibrating natural-language axes against AUDIBLE diversity.

Every other experiment in this paper steers on text and hopes the artifact
follows. Here we close that loop for music: we learn, from rendered audio,
WHICH natural-language axes actually move the sound, then regenerate using
only those, and measure audible diversity before and after.

Three phases.

  A. LENGTH AND CONTRACT PROBE. How much instruction can Lyria actually
     satisfy? We vary prompt length and measure two things that are checkable
     rather than assumed:
       plan compliance -- Lyria echoes its own section plan as [[A0]] [[B1]]
                          ...; we request a letter sequence and check whether
                          the returned sequence matches. This is an exact,
                          binary contract test, not a vibe.
       CLAP adherence  -- cosine between the CLAP *text* embedding of the
                          prompt and the CLAP *audio* embedding of the render.
                          CLAP is audio-text aligned, so this measures whether
                          the audio is actually the thing the prompt asked for.
     The longest length at which both hold is the honest ceiling on how much
     detail is worth writing.

  B. AXIS AUDIBILITY. For each elicited axis we compute an effect size:
     mean pairwise audio distance between tracks that DIFFER on that axis,
     minus the mean between tracks that SHARE its level. An axis with a large
     positive effect is one a listener could actually hear; an axis near zero
     is a distinction the text carries and the audio does not. This is the
     calculus's transversality criterion, measured in the output space rather
     than assumed from the level descriptions.

  C. CALIBRATED REGENERATION. Regenerate using only the audible axes, at the
     calibrated length, in Lyria's native section format, with an explicit
     instruction against the sparse-opening attractor mined in phase B. Then
     compare audible diversity to the uncalibrated corpus.

Section format matters and was determined empirically, not guessed. Given the
same content, "[0:00-0:25] ... [0:25-1:00] ..." returned the plan
[[A0]][[A1]][[A2]][[A3]] -- four sections all labelled A, i.e. the requested
structure was ignored -- while "[[A0]] ... [[B1]] ... [[C2]] ... [[A3]]"
returned exactly [[A0]][[B1]][[C2]][[A3]]. Lyria consumes its own marker
format and ignores wall-clock timestamps, so we write markers.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audio_domain import (LYRIA_MODEL, _load_gemini_key, clap_embed,
                          generate_lyria, load_audio, mert_embed)
from pipeline import HERE, USAGE, chat, parse_json

OUT = HERE / "real" / "lyria_calib"
OUT.mkdir(parents=True, exist_ok=True)
LENGTHS = [250, 500, 900, 1400, 2000]
PER_LENGTH = 6
WORKERS = 4

LETTERS = "ABCDEFG"

LENGTH_PROMPT = """Write ONE prompt for the Lyria text-to-music model. Purely \
INSTRUMENTAL -- no vocals, no singing, no spoken word.

Structure it using Lyria's own section markers, exactly this sequence: {seq}
Write each marker on its own, followed by what happens in that section.

The piece must NOT fade in: section {first} should already be at full density \
and full volume from the first beat.

Make it {target} characters (+/- 15%). Every clause must be something audible: \
name real instruments, register, articulation, the room, the rhythmic feel, and \
what changes between sections. Do not write music theory or abstract \
description -- the model ignores that.

End with "Instrumental only." Output only the prompt."""


def requested_shape(seq: str) -> str:
    return "".join(re.findall(r"\[\[([A-Za-z])\d*\]\]", seq))


def returned_shape(plan: str) -> str:
    return "".join(re.findall(r"\[\[([A-Za-z])\d*\]\]", plan or ""))


def make_seq(rng: random.Random, n_sec: int) -> str:
    """A section sequence like '[[A0]] [[B1]] [[C2]] [[A3]]' -- with a repeat
    sometimes, so the contract tests recall of an earlier section too."""
    shape = [LETTERS[0]]
    for i in range(1, n_sec):
        if i > 1 and rng.random() < 0.3:
            shape.append(rng.choice(shape))          # return to an earlier one
        else:
            shape.append(LETTERS[len(set(shape))])
    return " ".join(f"[[{c}{i}]]" for i, c in enumerate(shape))


def wav_of(mp3: Path) -> Path:
    w = mp3.with_suffix(".wav")
    if not w.exists() or w.stat().st_size < 10_000:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3),
                        "-ac", "1", "-ar", "48000", str(w)], check=False)
    return w


def opening_density(wav: Path, secs: float = 3.0) -> float:
    """Loudness of the first `secs` relative to the track's own median."""
    y = load_audio(wav, 48000)
    f = int(0.1 * 48000)
    m = len(y) // f
    rms = np.sqrt((y[:m * f].reshape(m, f) ** 2).mean(axis=1))
    ref = np.median(rms[rms > 0]) if (rms > 0).any() else 1e-9
    return float(rms[:int(secs * 10)].mean() / max(ref, 1e-9))


def clap_text_audio(prompts: list[str], wavs: list[Path]) -> np.ndarray:
    """Cosine between CLAP text and CLAP audio embeddings, per item."""
    import torch
    from transformers import ClapModel, AutoProcessor
    name = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(name).eval()
    proc = AutoProcessor.from_pretrained(name)
    out = []
    with torch.no_grad():
        for p, w in zip(prompts, wavs):
            ti = proc(text=[p[:900]], return_tensors="pt", padding=True)
            tf = model.get_text_features(**ti)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            y = load_audio(w, 48000)
            win = y[:48000 * 10] if len(y) >= 48000 * 10 else np.pad(y, (0, 48000 * 10 - len(y)))
            mid = y[len(y) // 2: len(y) // 2 + 48000 * 10]
            wins = [win] + ([mid] if len(mid) == 48000 * 10 else [])
            ai = proc(audio=wins, sampling_rate=48000, return_tensors="pt")
            af = model.get_audio_features(**ai)
            if not isinstance(af, torch.Tensor):
                af = getattr(af, "audio_embeds", af)
            af = af / af.norm(dim=-1, keepdim=True)
            out.append(float((tf @ af.T).mean()))
    return np.array(out)


# ---------------------------------------------------------------- phase A
def phase_a():
    key = _load_gemini_key()
    rng = random.Random(11)
    recs_path = OUT / "length_probe.jsonl"
    done = set()
    if recs_path.exists():
        for l in recs_path.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["id"])
    jobs = []
    for L in LENGTHS:
        for j in range(PER_LENGTH):
            jid = f"L{L}_{j}"
            if jid in done:
                continue
            n_sec = 3 if L < 600 else (4 if L < 1200 else 5)
            seq = make_seq(rng, n_sec)
            jobs.append((jid, L, seq))
    print(f"phase A: {len(jobs)} prompts to write/render", flush=True)
    fh = open(recs_path, "a")
    for jid, L, seq in jobs:
        first = requested_shape(seq)[0]
        try:
            p = chat([{"role": "user", "content": LENGTH_PROMPT.format(
                seq=seq, target=L, first=first)}], 1.0, 3000).strip()
        except Exception as e:
            print(f"  {jid}: prompt failed {e}"); continue
        mp3 = OUT / f"{jid}.mp3"
        r = generate_lyria(p, key, mp3)
        ok = bool(r and r.get("status") in ("ok", "cached"))
        rec = {"id": jid, "target_len": L, "actual_len": len(p),
               "requested_seq": seq, "requested_shape": requested_shape(seq),
               "prompt": p, "rendered": ok,
               "returned_plan": (r or {}).get("section_plan", ""),
               "returned_shape": returned_shape((r or {}).get("section_plan", ""))}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"  {jid}: len={len(p)} req={rec['requested_shape']} "
              f"got={rec['returned_shape']} ok={ok}", flush=True)
    fh.close()
    analyse_a()


def analyse_a():
    recs = [json.loads(l) for l in (OUT / "length_probe.jsonl").read_text().splitlines()
            if l.strip()]
    recs = [r for r in recs if r["rendered"]]
    if not recs:
        print("no rendered length-probe items"); return
    wavs, prompts = [], []
    for r in recs:
        mp3 = OUT / f"{r['id']}.mp3"
        if mp3.exists():
            wavs.append(wav_of(mp3)); prompts.append(r["prompt"])
    print(f"phase A: scoring {len(wavs)} renders (CLAP text-audio) ...", flush=True)
    adh = clap_text_audio(prompts, wavs)
    by = defaultdict(list)
    for r, a, w in zip(recs, adh, wavs):
        by[r["target_len"]].append({
            "plan_match": r["requested_shape"] == r["returned_shape"],
            "plan_len_match": len(r["requested_shape"]) == len(r["returned_shape"]),
            "n_distinct_req": len(set(r["requested_shape"])),
            "n_distinct_got": len(set(r["returned_shape"])),
            "clap": float(a), "open": opening_density(w),
            "actual_len": r["actual_len"]})
    out = {}
    print(f"\n{'target':>7} {'n':>3} {'chars':>6} {'plan==':>7} {'|plan|==':>8} "
          f"{'CLAP':>6} {'open3s':>7}")
    for L in sorted(by):
        v = by[L]
        row = {"n": len(v),
               "mean_chars": round(float(np.mean([x["actual_len"] for x in v])), 0),
               "plan_exact_match": round(float(np.mean([x["plan_match"] for x in v])), 3),
               "plan_length_match": round(float(np.mean([x["plan_len_match"] for x in v])), 3),
               "mean_distinct_sections_req": round(float(np.mean([x["n_distinct_req"] for x in v])), 2),
               "mean_distinct_sections_got": round(float(np.mean([x["n_distinct_got"] for x in v])), 2),
               "clap_adherence": round(float(np.mean([x["clap"] for x in v])), 4),
               "opening_density_3s": round(float(np.mean([x["open"] for x in v])), 3)}
        out[L] = row
        print(f"{L:>7} {row['n']:>3} {row['mean_chars']:>6.0f} "
              f"{row['plan_exact_match']:>7.2f} {row['plan_length_match']:>8.2f} "
              f"{row['clap_adherence']:>6.3f} {row['opening_density_3s']:>7.2f}")
    json.dump(out, open(HERE / "figures" / "lyria_length_probe.json", "w"), indent=2)
    print("wrote figures/lyria_length_probe.json")


# ---------------------------------------------------------------- phase B
def phase_b(arm: str = "ihd"):
    """Which elicited axes are actually AUDIBLE?"""
    d = HERE / "real" / f"lyria_{arm}"
    recs = [json.loads(l) for l in (d / "corpus.jsonl").read_text().splitlines() if l.strip()]
    files = sorted((d / "audio").glob("*.mp3"))
    idx = [int(f.stem) for f in files]
    wavs = [wav_of(f) for f in files]
    specs = [recs[i]["spec"] for i in idx if i < len(recs)]
    n = min(len(specs), len(wavs))
    specs, wavs = specs[:n], wavs[:n]
    if n < 12:
        print(f"phase B needs more tracks (have {n})"); return
    print(f"phase B: {n} tracks, embedding ...", flush=True)
    for tag, fn in (("clap", clap_embed), ("mert", mert_embed)):
        cache = d / f"{tag}_emb.npy"
        if cache.exists() and np.load(cache).shape[0] == n:
            E = np.load(cache)
        else:
            E = fn(wavs); np.save(cache, E)
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        D = 1.0 - En @ En.T
        axes = sorted({k for s in specs for k in s})
        rows = {}
        for ax in axes:
            same, diff = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = specs[i].get(ax), specs[j].get(ax)
                    if a is None or b is None:
                        continue
                    (same if a == b else diff).append(D[i, j])
            if len(same) < 5 or len(diff) < 5:
                continue
            s_m, d_m = float(np.mean(same)), float(np.mean(diff))
            pooled = float(np.sqrt((np.var(same) + np.var(diff)) / 2)) or 1e-9
            rows[ax] = {"mean_dist_same_level": round(s_m, 4),
                        "mean_dist_diff_level": round(d_m, 4),
                        "effect": round(d_m - s_m, 4),
                        "cohens_d": round((d_m - s_m) / pooled, 3),
                        "n_same": len(same), "n_diff": len(diff)}
        ranked = sorted(rows.items(), key=lambda kv: -kv[1]["effect"])
        print(f"\n  [{tag}] axis audibility (higher = more audible):")
        for ax, r in ranked:
            print(f"    {ax:38s} effect={r['effect']:+.4f} d={r['cohens_d']:+.2f}")
        json.dump({"embedder": tag, "n_tracks": n, "axes": rows,
                   "ranked": [a for a, _ in ranked]},
                  open(HERE / "figures" / f"lyria_axis_audibility_{tag}.json", "w"), indent=2)
    print("\nwrote figures/lyria_axis_audibility_*.json")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "a"
    if which == "a":
        phase_a()
    elif which == "analyse_a":
        analyse_a()
    elif which == "b":
        phase_b()
