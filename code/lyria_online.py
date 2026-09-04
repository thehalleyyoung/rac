"""
Online CLAP-in-the-loop generation: 100 instrumentals, maximally spread in
AUDIO space, every one of which still qualifies as instrumental.

This is the paper's full stack closed in the output space rather than in text.
Nothing here trusts the prompt: after every render we listen (with CLAP),
check that the track is still what we asked for, measure where it landed, and
update what we believe about which natural-language axes actually move the
sound.

The loop, per accepted track:

  1. CHOOSE a spec. Sample a pool of points in the elicited axis lattice and
     keep the one that is both (a) pointing out of the region the accepted
     CLAP embeddings already occupy, and (b) built from axis levels that have
     historically PAID OFF in audible distance. (b) is the learned part.

  2. WRITE the prompt with an LLM, in Lyria's native [[A0]] [[B1]] section
     format (empirically the only structural format it honours -- wall-clock
     timestamps are ignored), at the calibrated length, with an explicit
     instruction against the sparse-opening attractor.

  3. RENDER with Lyria, decode, and embed the audio with CLAP.

  4. GATE on validity, in audio space, not on trust: CLAP is audio-text
     aligned, so we can ask the audio directly whether it is instrumental.
     A track is admitted only if its similarity to instrumental descriptions
     exceeds its similarity to vocal descriptions by a margin. This is the
     typicality gate of the main paper, instantiated where it can actually be
     checked -- and it is negative supervision only: passing it means "not
     disqualified", never "good".

  5. GATE on novelty: reject a track whose nearest accepted neighbour in CLAP
     space is closer than a floor. This is the max-min objective, enforced on
     what the listener hears rather than on what the prompt said.

  6. LEARN. Credit every (axis, level) used by an accepted track with the
     audible novelty it achieved, and debit the levels of rejected-as-too-
     similar tracks. Future specs sample levels in proportion to a softmax of
     that credit, so the search concentrates on the axes that turn out to be
     audible and abandons the ones that are only lexically different.

  7. MINE. Periodically take the closest surviving pairs in CLAP space, ask
     what their specs have in common, and append the answer to the ledger as
     a textual constraint on subsequent prompts.

Because the gate is applied after rendering, rejected tracks still cost a
render. We log those costs explicitly rather than reporting only the accepted
set, since a method that achieves spread by throwing away four fifths of its
generations is not free.
"""
from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audio_domain import _load_gemini_key, generate_lyria, load_audio
from axes import Axis, AxisTree, spec_text
from pipeline import HERE, USAGE, chat, parse_json

RUN = HERE / "real" / "lyria_online"
AUDIO = RUN / "audio"
TARGET = 100
WORKERS = 4                # parallel renders per wave
POOL = 8                   # candidate specs scored per slot
PROMPT_CHARS = 900         # set by the length probe; see lyria_calibrate.py
NOVELTY_FLOOR = 0.16       # min CLAP cosine distance to the accepted set
INSTR_MARGIN = 0.010       # CLAP(instrumental) - CLAP(vocal) must exceed this
MINE_EVERY = 25
LEDGER_SLICE = 8
TEMP = 0.6                 # softmax temperature on learned axis-level credit
MAX_RENDERS = 260          # hard cap on Lyria calls

INSTRUMENTAL_PROBES = [
    "instrumental music with no vocals",
    "an instrumental piece, no singing, no voice",
    "purely instrumental, no lyrics",
]
VOCAL_PROBES = [
    "a song with a singer, lead vocals and lyrics",
    "singing voice, vocal melody, words being sung",
    "a cappella voices, choir, spoken word",
]

WRITE_PROMPT = """Write ONE prompt for the Lyria text-to-music model. Purely \
INSTRUMENTAL -- no vocals, no singing, no spoken word, no vocal samples.

Structure it with Lyria's own section markers, exactly this sequence: {seq}
Put each marker on its own and follow it with what happens in that section.

Section {first} must already be at FULL density and full volume from the first \
beat -- no fade-in, no sparse ambient introduction.

Let these choices shape the piece. Express each as concrete audible direction, \
never as abstract description:
{contracts}

The library so far keeps doing these things. Avoid them:
{avoid}

About {target} characters. Name real instruments, register, articulation, the \
room, the rhythmic feel, and what actually changes between sections. Do not \
write music theory -- the model ignores it.

End with "Instrumental only." Output only the prompt."""

MINE_PROMPT = """These pairs of instrumental-music prompts produced tracks that a \
music embedding model judged to be nearly IDENTICAL, even though the prompts were \
written to be different. Say what the pairs actually share -- the instruments, \
textures, rhythmic feels, or formal habits that keep collapsing together despite \
different wording.

{pairs}

Return JSON only: {{"attractors": ["...", "..."]}} -- 4-8 concrete findings."""


class Clap:
    """CLAP audio + text encoder, loaded once and reused."""

    def __init__(self):
        import torch
        from transformers import ClapModel, AutoProcessor
        self.torch = torch
        name = "laion/clap-htsat-unfused"
        self.model = ClapModel.from_pretrained(name).eval()
        self.proc = AutoProcessor.from_pretrained(name)
        self.instr = self._text(INSTRUMENTAL_PROBES)
        self.vocal = self._text(VOCAL_PROBES)

    @staticmethod
    def _tensor(out):
        """transformers 5.x returns ModelOutput where 4.x returned a tensor."""
        import torch
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("text_embeds", "audio_embeds", "pooler_output",
                     "last_hidden_state"):
            v = getattr(out, attr, None)
            if v is not None:
                return v if v.dim() == 2 else v.mean(dim=1)
        raise TypeError(f"cannot extract features from {type(out)}")

    def _text(self, texts):
        with self.torch.no_grad():
            i = self.proc(text=texts, return_tensors="pt", padding=True)
            f = self._tensor(self.model.get_text_features(**i))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()

    def audio(self, wav: Path) -> np.ndarray:
        """Track vector: mean/std/temporal-difference over 10s windows, so a
        static piece and a developing one do not collide."""
        y = load_audio(wav, 48000)
        w, h = 48000 * 10, 48000 * 5
        wins = [y[i:i + w] for i in range(0, max(len(y) - w + 1, 1), h)] or [
            np.pad(y, (0, max(0, w - len(y))))[:w]]
        vecs = []
        with self.torch.no_grad():
            for i in range(0, len(wins), 8):
                inp = self.proc(audio=wins[i:i + 8], sampling_rate=48000,
                                return_tensors="pt")
                f = self._tensor(self.model.get_audio_features(**inp))
                f = f / f.norm(dim=-1, keepdim=True)
                vecs.append(f.numpy())
        W = np.vstack(vecs)
        agg = np.concatenate([W.mean(0), W.std(0),
                              np.abs(np.diff(W, axis=0)).mean(0) if len(W) > 1
                              else np.zeros(W.shape[1])])
        return agg / max(np.linalg.norm(agg), 1e-9), W.mean(0)

    def instrumentality(self, wmean: np.ndarray) -> tuple[float, float]:
        v = wmean / max(np.linalg.norm(wmean), 1e-9)
        return float((self.instr @ v).mean()), float((self.vocal @ v).mean())


def wav_of(mp3: Path) -> Path:
    w = mp3.with_suffix(".wav")
    if not w.exists() or w.stat().st_size < 10_000:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3),
                        "-ac", "1", "-ar", "48000", str(w)], check=False)
    return w


def opening_density(wav: Path, secs: float = 3.0) -> float:
    y = load_audio(wav, 48000)
    f = int(0.1 * 48000)
    m = len(y) // f
    if m == 0:
        return 0.0
    rms = np.sqrt((y[:m * f].reshape(m, f) ** 2).mean(axis=1))
    ref = np.median(rms[rms > 0]) if (rms > 0).any() else 1e-9
    return float(rms[:int(secs * 10)].mean() / max(ref, 1e-9))


LETTERS = "ABCDEF"


def make_seq(rng: random.Random, n_sec: int) -> str:
    shape = ["A"]
    for i in range(1, n_sec):
        if i > 1 and rng.random() < 0.3:
            shape.append(rng.choice(shape))
        else:
            shape.append(LETTERS[len(set(shape))])
    return " ".join(f"[[{c}{i}]]" for i, c in enumerate(shape))


class Credit:
    """Learned audible payoff of each (axis, level)."""

    def __init__(self):
        self.sum = defaultdict(float)
        self.n = defaultdict(int)

    def update(self, spec: dict, reward: float):
        for a, l in spec.items():
            self.sum[(a, l)] += reward
            self.n[(a, l)] += 1

    def score(self, spec: dict) -> float:
        vals = [self.sum[(a, l)] / self.n[(a, l)]
                for a, l in spec.items() if self.n[(a, l)] > 0]
        return float(np.mean(vals)) if vals else 0.0

    def table(self) -> dict:
        return {f"{a} = {l}": {"mean_reward": round(self.sum[(a, l)] / self.n[(a, l)], 4),
                               "n": self.n[(a, l)]}
                for (a, l) in sorted(self.n, key=lambda k: -self.sum[k] / self.n[k])
                if self.n[(a, l)] > 0}


def main(target: int = TARGET):
    AUDIO.mkdir(parents=True, exist_ok=True)
    key = _load_gemini_key()
    rng = random.Random(4242)
    clap = Clap()

    tree = AxisTree(RUN / "axes.jsonl")
    src = HERE / "real" / "lyria_ihd" / "axes.jsonl"
    if not tree.axes and src.exists():
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r["event"] == "seed":
                tree.axes = [Axis(a["name"], a["levels"]) for a in r["axes"]]
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})
    print(f"axes: {len(tree.axes)}, lattice ~{tree.space_size():.1e}", flush=True)

    log = open(RUN / "log.jsonl", "a")
    accepted: list[dict] = []
    E: list[np.ndarray] = []
    credit = Credit()
    attractors: list[str] = []
    renders = 0
    rej_vocal = rej_close = rej_fail = 0
    t0 = time.time()

    # resume
    if (RUN / "log.jsonl").exists():
        for line in (RUN / "log.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("accepted"):
                w = AUDIO / f"{r['i']:05d}.wav"
                if w.exists():
                    v, _ = clap.audio(w)
                    accepted.append(r); E.append(v)
                    credit.update(r["spec"], r["reward"])
        print(f"resumed with {len(accepted)} accepted", flush=True)

    while len(accepted) < target and renders < MAX_RENDERS:
        n_slots = min(WORKERS, target - len(accepted))
        specs = []
        for _ in range(n_slots):
            pool = [tree.sample_spec(rng) for _ in range(POOL)]
            if len(E) >= 8:
                A = np.stack(E)
                # spec-level score: learned audible credit, softmax-weighted
                sc = np.array([credit.score(s) for s in pool])
                if sc.std() > 1e-9:
                    p = np.exp((sc - sc.max()) / TEMP)
                    p = p / p.sum()
                    specs.append(pool[int(rng.choices(range(len(pool)), weights=p)[0])])
                else:
                    specs.append(rng.choice(pool))
            else:
                specs.append(pool[0])

        avoid = ("\n".join(f"  - {a}" for a in attractors[-LEDGER_SLICE:])
                 or "  - (nothing yet)")
        prompts = []
        for s in specs:
            seq = make_seq(rng, 4)
            first = re.findall(r"\[\[([A-Za-z])", seq)[0]
            try:
                p = chat([{"role": "user", "content": WRITE_PROMPT.format(
                    seq=seq, first=first, contracts=spec_text(s), avoid=avoid,
                    target=PROMPT_CHARS)}], 1.0, 3000).strip()
            except Exception:
                p = None
            prompts.append((s, seq, p))

        idx0 = len(accepted) + rej_vocal + rej_close + rej_fail
        results = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {}
            for k, (s, seq, p) in enumerate(prompts):
                if not p:
                    continue
                mp3 = AUDIO / f"cand_{idx0 + k:05d}.mp3"
                futs[ex.submit(generate_lyria, p, key, mp3)] = (k, s, seq, p, mp3)
            for fu in as_completed(futs):
                k, s, seq, p, mp3 = futs[fu]
                try:
                    results.append((k, s, seq, p, mp3, fu.result()))
                except Exception as e:
                    results.append((k, s, seq, p, mp3, {"status": "error", "e": str(e)[:120]}))
        renders += len(results)

        for k, s, seq, p, mp3, r in sorted(results):
            if not (r and r.get("status") in ("ok", "cached")):
                rej_fail += 1
                continue
            wav = wav_of(mp3)
            try:
                v, wmean = clap.audio(wav)
            except Exception:
                rej_fail += 1
                continue
            si, sv = clap.instrumentality(wmean)
            gap = float(1.0 - max(v @ np.stack(E).T)) if E else 1.0
            plan = (r.get("section_plan") or "").replace("\n", " ")
            rec = {"i": len(accepted), "spec": s, "requested_seq": seq,
                   "returned_plan": plan, "prompt_chars": len(p),
                   "clap_instr": round(si, 4), "clap_vocal": round(sv, 4),
                   "instr_margin": round(si - sv, 4), "gap": round(gap, 4),
                   "opening_density": round(opening_density(wav), 3),
                   "prompt": p}
            # (4) validity gate, measured on the audio
            if si - sv < INSTR_MARGIN:
                rej_vocal += 1
                rec.update(accepted=False, reason="not_instrumental")
                credit.update(s, -0.05)
                log.write(json.dumps(rec) + "\n"); log.flush()
                continue
            # (5) novelty gate, measured on the audio
            if gap < NOVELTY_FLOOR:
                rej_close += 1
                rec.update(accepted=False, reason="too_similar")
                credit.update(s, -gap)
                log.write(json.dumps(rec) + "\n"); log.flush()
                continue
            final = AUDIO / f"{len(accepted):05d}.mp3"
            mp3.replace(final)
            wav_of(final)
            rec["reward"] = round(gap, 4)
            rec.update(accepted=True, file=final.name)
            accepted.append(rec); E.append(v)
            credit.update(s, gap)
            log.write(json.dumps(rec) + "\n"); log.flush()

        if len(accepted) and len(accepted) % MINE_EVERY < WORKERS and len(E) > 12:
            A = np.stack(E)
            S = A @ A.T
            np.fill_diagonal(S, -np.inf)
            pairs = np.dstack(np.unravel_index(np.argsort(-S, axis=None)[:12], S.shape))[0]
            seen, blocks = set(), []
            for i, j in pairs:
                if i >= j or (i, j) in seen:
                    continue
                seen.add((i, j))
                blocks.append(f"--- pair (sim {S[i, j]:.3f}) ---\nA: "
                              f"{accepted[i]['prompt'][:300]}\nB: {accepted[j]['prompt'][:300]}")
                if len(blocks) >= 4:
                    break
            try:
                raw = chat([{"role": "user", "content": MINE_PROMPT.format(
                    pairs="\n".join(blocks))}], 0.3, 900, json_mode=True)
                new = [str(a) for a in parse_json(raw).get("attractors", [])][:8]
                if new:
                    attractors.extend(new)
                    with open(RUN / "ledger.jsonl", "a") as f:
                        f.write(json.dumps({"n": len(accepted), "attractors": new}) + "\n")
                    print(f"  mined {len(new)} audio attractors", flush=True)
            except Exception:
                pass

        print(f"[online] accepted={len(accepted)}/{target} renders={renders} "
              f"rej(vocal={rej_vocal} close={rej_close} fail={rej_fail}) "
              f"${USAGE.cost_usd():.2f} {(time.time()-t0)/60:.0f}m", flush=True)

    A = np.stack(E) if E else np.zeros((0, 1))
    summary = {
        "target": target, "accepted": len(accepted), "renders": renders,
        "renders_per_accept": round(renders / max(len(accepted), 1), 2),
        "rejected_not_instrumental": rej_vocal, "rejected_too_similar": rej_close,
        "rejected_failed": rej_fail,
        "novelty_floor": NOVELTY_FLOOR, "instr_margin": INSTR_MARGIN,
        "prompt_chars_mean": round(float(np.mean([a["prompt_chars"] for a in accepted])), 0)
        if accepted else 0,
        "opening_density_mean": round(float(np.mean([a["opening_density"] for a in accepted])), 3)
        if accepted else 0,
        "clap_instr_margin_mean": round(float(np.mean([a["instr_margin"] for a in accepted])), 4)
        if accepted else 0,
        "plan_exact_match_rate": round(float(np.mean([
            "".join(re.findall(r"\[\[([A-Za-z])", a["requested_seq"])) ==
            "".join(re.findall(r"\[\[([A-Za-z])", a["returned_plan"]))
            for a in accepted])), 3) if accepted else 0,
        "axis_level_credit": credit.table(),
        "n_attractors": len(attractors),
        "openrouter_cost_usd": round(USAGE.cost_usd(), 4),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    np.save(RUN / "clap_emb.npy", A)
    json.dump(summary, open(RUN / "run_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "axis_level_credit"}, indent=2))
    log.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else TARGET)
