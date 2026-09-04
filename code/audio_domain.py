"""
Third domain: Mureka instrumental prompts, judged in AUDIO embedding space.

This is the hardest test of the paper's thesis, because it has the longest
chain between what we steer and what we measure:

    latent axes -> long text prompt -> Mureka instrumental -> minutes of audio
                                                              -> embedding

Every link composes its own mode structure, and the DALL-E result (§7:
text/image pairwise-similarity correlation r = 0.155) says those structures
are largely unrelated. Text-side diversity is therefore an even weaker proxy
here than it was for images, and the point of this module is to measure the
end of the chain directly.

Prompts are written as long as the endpoint allows -- 1024 characters, which
the live API enforces even though the docs' 2000-char figure belongs to the
SONG endpoint -- because a long prompt is the only place the elicited latent
axes can actually be spent. A 20-word prompt cannot carry seven contracts.

Three embeddings, because they disagree and the disagreement is the finding:

  CLAP    (laion/clap-htsat-unfused) audio-text aligned; a "semantic" view,
          close to how a person would describe the track in words. Trained on
          10-second windows.
  MERT    (m-a-p/MERT-v1-95M) self-supervised MUSIC representation; hidden
          states carry pitch/harmony/timbre structure that CLAP's caption-
          aligned space discards. Also windowed (5s at 24kHz here).
  long    Both of the above are window models, and an instrumental is minutes
          long. We aggregate windows into a single track vector with mean,
          standard deviation, and a first-order temporal-difference term, so
          that two tracks with identical average timbre but different
          ARRANGEMENT (one static, one developing) do not collide. Aggregating
          by mean alone is the standard shortcut and it throws away exactly
          the axis a composer would call form.

Running this needs MUREKA_API_KEY. Without it, `prompts` still works (it is
pure text generation) and everything downstream refuses cleanly rather than
inventing audio.
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pipeline import HERE, USAGE, chat, embed, parse_json
from axes import Axis, AxisTree, spec_text
from real_run import Corpus, EMBED_BATCH

# --- Lyria 3 (Google Gemini API) ---------------------------------------
# Synchronous generateContent, audio returned inline as base64 -- no task
# queue, no polling, ~9s per clip. Replaces the Mureka path, which was
# asynchronous, single-concurrency, and whose quota we exhausted at n=2.
LYRIA_MODEL = "lyria-3-pro-preview"
LYRIA_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent?key={key}")
GEMINI_KEYFILE = Path("/Users/halley/Documents/newsuno/compositionplans/keys/geminiapikey.txt")
# Lyria's token limit is ~1M, so nothing external caps prompt length. The cap
# below is a claim about the MODEL, not the API: a text-to-music model honours
# concrete musical direction (instruments, texture, register, articulation,
# form, how it ends) and quietly ignores paragraphs of abstract compositional
# theory. Writing 1600-character walls of contracts would inflate our prompt
# diversity while changing the audio far less than the text metrics suggest --
# manufacturing exactly the proxy gap this paper warns about. So the target is
# 2-4 sentences of specific, performable direction.
MAX_PROMPT_CHARS = 480
TARGET_PROMPT_CHARS = 300

# --- legacy Mureka constants, kept so old corpora remain loadable ------
MUREKA_BASE = "https://api.mureka.ai"
GEN_PATH = "/v1/instrumental/generate"
QUERY_PATH = "/v1/instrumental/query/{task_id}"
MUREKA_MODEL = "auto"
POLL_INTERVAL = 12
POLL_TIMEOUT = 600

SEED_AXES_PROMPT = """You are designing the dimensions of variation for a very large \
library of INSTRUMENTAL music generation prompts. Do not write any prompts.

Propose {k} INDEPENDENT axes of variation. A good axis is close to orthogonal to the \
others, changes something structural about the MUSIC rather than merely its genre \
label, and has levels that are all genuinely usable.

Avoid the obvious surface axes (genre alone, mood alone, tempo alone). Reach for real \
compositional decisions: how the form develops over time, what the rhythmic \
relationship between layers is, how harmony moves or refuses to move, what the timbral \
palette's centre of gravity is, how density changes across the track, what the piece's \
relationship to a pulse is, how it begins and how it ends.

Return JSON only: {{"axes": [{{"name": "...", "why_orthogonal": "...", \
"levels": ["...", "...", "...", "..."]}}]}}
Each axis needs 4-6 levels. Keep why_orthogonal under 15 words, each level under 10 words."""

PROMPT_TEMPLATE = """Write ONE prompt for a text-to-music model. The result must be \
PURELY INSTRUMENTAL -- no vocals, no singing, no spoken word, no vocal samples.

Let these choices shape the piece. Do not name them or restate them abstractly; \
express each one as concrete, audible musical direction:
{contracts}

The library so far keeps doing these things. Avoid them:
{avoid}

Write {target_chars}-{max_chars} characters, 2-4 sentences. Be specific and \
performable: name actual instruments, the room or recording character, register, \
articulation, the rhythmic feel, roughly how it develops, and how it ends. Do NOT \
write abstract music theory or compositional philosophy -- a text-to-music model \
ignores that. Every clause should be something you could actually hear. End with \
"Instrumental only." Output only the prompt."""

MINE_PROMPT = """Here are {n} instrumental-music prompts sampled from a growing library. \
Identify what they SHARE: recurring instruments, recurring adjectives, recurring formal \
shapes, recurring harmonic or rhythmic moves, recurring ways of ending. Be concrete \
(not "atmospheric" but which atmosphere and built from what).

{sample}

Return JSON only: {{"attractors": ["...", "..."]}} -- 6-12 findings, most pervasive first."""


def _load_gemini_key() -> str | None:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    if GEMINI_KEYFILE.is_file():
        v = GEMINI_KEYFILE.read_text().strip()
        if v:
            os.environ["GEMINI_API_KEY"] = v
            return v
    return None


def generate_lyria(prompt: str, key: str, out_path: Path) -> dict | None:
    """One synchronous Lyria call; audio comes back inline as base64 mpeg."""
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return {"status": "cached", "path": str(out_path)}
    body = json.dumps({"contents": [{"parts": [{"text": prompt[:MAX_PROMPT_CHARS]}]}]}).encode()
    url = LYRIA_URL.format(model=LYRIA_MODEL, key=key)
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.load(r)
            cands = payload.get("candidates") or []
            if not cands:
                return {"status": "no_candidates",
                        "feedback": str(payload.get("promptFeedback"))[:160]}
            parts = (cands[0].get("content") or {}).get("parts")
            if not parts:
                # a filtered or refused generation returns a candidate with a
                # finishReason and no content at all -- report it rather than
                # raising a KeyError that looks like a transport failure
                return {"status": "no_content",
                        "finish_reason": cands[0].get("finishReason"),
                        "feedback": str(payload.get("promptFeedback"))[:160]}
            plan = "".join(pt.get("text", "") for pt in parts)
            for part in parts:
                d = part.get("inlineData")
                if not d or not d.get("data"):
                    continue
                raw = base64.b64decode(d["data"])
                tmp = out_path.with_suffix(".part")
                tmp.write_bytes(raw)
                os.replace(tmp, out_path)
                # lyria-3-pro emits its own section plan, e.g. "[[A0]] [[B1]]
                # [[C2]] [[D3]]" -- a model-reported description of the FORM it
                # chose, and the one structural signal available without
                # analysing the waveform. Keep it.
                if plan.strip():
                    out_path.with_suffix(".plan.txt").write_text(plan.strip())
                return {"status": "ok", "path": str(out_path),
                        "mime": d.get("mimeType"), "bytes": len(raw),
                        "section_plan": plan.strip()[:80]}
            return {"status": "no_audio"}
        except Exception as e:
            if attempt == 3:
                return {"status": "error", "error": str(e)[:200]}
            time.sleep(2 ** attempt + 1)
    return None


def _load_mureka_key() -> str | None:
    if os.environ.get("MUREKA_API_KEY"):
        return os.environ["MUREKA_API_KEY"]
    for rc in (Path.home() / ".zshrc", Path.home() / ".bashrc"):
        if not rc.is_file():
            continue
        for line in rc.read_text().splitlines():
            m = re.match(r'\s*export\s+MUREKA_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                os.environ["MUREKA_API_KEY"] = m.group(1)
                return m.group(1)
    keyfile = Path("/Users/halley/Documents/newsuno/compositionplans/keys/mureka_api_key.txt")
    if keyfile.is_file():
        v = keyfile.read_text().strip()
        if v:
            os.environ["MUREKA_API_KEY"] = v
            return v
    env = HERE.parents[1] / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("MUREKA_API_KEY="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["MUREKA_API_KEY"] = v
                return v
    return None


def _post(path: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(
        MUREKA_BASE + path, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _get(path: str, key: str) -> dict:
    req = urllib.request.Request(MUREKA_BASE + path,
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def generate_instrumental(prompt: str, key: str, out_path: Path) -> dict | None:
    """Submit one instrumental job, poll to completion, download the audio."""
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return {"status": "cached", "path": str(out_path)}
    task = _post(GEN_PATH, {"model": MUREKA_MODEL,
                            "prompt": prompt[:MAX_PROMPT_CHARS]}, key)
    tid = task.get("id") or task.get("task_id")
    if not tid:
        return None
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        st = _get(QUERY_PATH.format(task_id=tid), key)
        status = st.get("status", "")
        if status in ("failed", "timeouted", "cancelled"):
            return {"status": status, "task_id": tid}
        if status in ("succeeded", "success", "completed"):
            songs = st.get("choices") or st.get("songs") or []
            # Prefer WAV: this environment's librosa is unusable (numba
            # requires NumPy <= 2.4, we have 2.5), so decoding goes through
            # soundfile, which reads WAV/FLAC natively and MP3 only via
            # optional backends. WAV also avoids lossy artifacts polluting
            # the very spectral features we are about to embed.
            url = None
            for s in songs:
                url = (s.get("wav_url") or s.get("flac_url")
                       or s.get("url") or s.get("audio_url"))
                if url:
                    break
            if not url:
                return {"status": "no_url", "task_id": tid}
            tmp = out_path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, out_path)
            return {"status": "ok", "task_id": tid, "path": str(out_path),
                    "duration": st.get("duration")}
    return {"status": "timeout", "task_id": tid}


# ---------------------------------------------------------------------------
# audio embeddings
# ---------------------------------------------------------------------------
def load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Mono float32 at target_sr, via soundfile + scipy.

    Deliberately not librosa: this environment's numba requires NumPy <= 2.4
    and NumPy 2.5 is installed, so importing librosa.core.audio raises. The
    only thing we needed from it was load-and-resample, which soundfile and
    scipy.signal.resample_poly do exactly and without the dependency.
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
    return y


def _feat_tensor(out):
    """transformers 5.x returns a ModelOutput where 4.x returned a tensor."""
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("audio_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v if v.dim() == 2 else v.mean(dim=1)
    raise TypeError(f"cannot extract audio features from {type(out)}")


def _windows(y: np.ndarray, sr: int, win_s: float, hop_s: float):
    w, h = int(win_s * sr), int(hop_s * sr)
    if len(y) < w:
        yield np.pad(y, (0, w - len(y)))
        return
    for i in range(0, len(y) - w + 1, h):
        yield y[i:i + w]


def _aggregate(W: np.ndarray) -> np.ndarray:
    """Window vectors -> one track vector.

    mean captures the average character; std captures how much the track
    CHANGES; the mean absolute first difference captures how fast it changes.
    A mean-only aggregate makes a static drone and a piece that develops
    through three distinct sections look identical, which is precisely the
    distinction a diversity measure over music must not lose.
    """
    if len(W) == 1:
        return np.concatenate([W[0], np.zeros_like(W[0]), np.zeros_like(W[0])])
    mean = W.mean(axis=0)
    std = W.std(axis=0)
    dif = np.abs(np.diff(W, axis=0)).mean(axis=0)
    return np.concatenate([mean, std, dif])


def clap_embed(paths: list[Path]) -> np.ndarray:
    import torch
    from transformers import ClapModel, AutoProcessor
    name = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(name).eval()
    proc = AutoProcessor.from_pretrained(name)
    out = []
    for p in paths:
        y = load_audio(p, 48000)
        wins = list(_windows(y, 48000, 10.0, 5.0))
        vecs = []
        with torch.no_grad():
            for i in range(0, len(wins), 8):
                # transformers 5.x renamed the CLAP processor kwarg
                # `audios` -> `audio` and raises on the old name.
                inp = proc(audio=wins[i:i + 8], sampling_rate=48000,
                           return_tensors="pt")
                f = _feat_tensor(model.get_audio_features(**inp))
                f = f / f.norm(dim=-1, keepdim=True)
                vecs.append(f.numpy())
        out.append(_aggregate(np.vstack(vecs)))
        print(f"  CLAP {p.name}: {len(wins)} windows", flush=True)
    return np.array(out)


def mert_embed(paths: list[Path]) -> np.ndarray:
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    name = "m-a-p/MERT-v1-95M"
    model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
    out = []
    for p in paths:
        y = load_audio(p, 24000)
        wins = list(_windows(y, 24000, 5.0, 2.5))
        vecs = []
        with torch.no_grad():
            for w in wins:
                inp = fe(w, sampling_rate=24000, return_tensors="pt")
                h = model(**inp, output_hidden_states=True).hidden_states
                # mid-stack layers carry musical structure; the last layer is
                # closest to the SSL pretext task and generalizes worse
                sel = torch.stack(h[5:9]).mean(dim=0).mean(dim=1).squeeze(0)
                vecs.append((sel / sel.norm()).numpy())
        out.append(_aggregate(np.vstack(vecs)))
        print(f"  MERT {p.name}: {len(wins)} windows", flush=True)
    return np.array(out)


# ---------------------------------------------------------------------------
def run_prompts(arm: str, target: int, budget: float,
                target_chars: int = TARGET_PROMPT_CHARS):
    """Generate the instrumental PROMPTS (no audio; needs no Mureka key)."""
    C = Corpus("lyria", arm)
    rng = random.Random(77 + C.n)
    tree = AxisTree(C.axes_path)
    if C.axes_path.exists() and C.axes_path.stat().st_size:
        for line in C.axes_path.read_text().splitlines():
            rec = json.loads(line)
            if rec["event"] == "seed":
                tree.axes = [Axis(a["name"], a["levels"]) for a in rec["axes"]]
    if not tree.axes and arm == "ihd":
        raw = chat([{"role": "user", "content": SEED_AXES_PROMPT.format(k=7)}],
                   temperature=1.0, max_tokens=4000, json_mode=True)
        for a in parse_json(raw)["axes"]:
            lv = [str(x) for x in a["levels"] if str(x).strip()]
            if len(lv) >= 3:
                tree.axes.append(Axis(str(a["name"]), lv))
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})
        print(f"  seeded {len(tree.axes)} music axes", flush=True)

    naive_prompt = (
        "Write ONE prompt for a text-to-music model. The result must be PURELY "
        "INSTRUMENTAL -- no vocals, no singing, no spoken word. Write "
        f"{target_chars}-{MAX_PROMPT_CHARS} characters, 2-4 sentences: name actual "
        "instruments, the recording character, register, rhythmic feel, roughly how "
        "it develops, and how it ends. Every clause should be something you could "
        'actually hear. End with "Instrumental only." Output only the prompt.'
    )
    t0 = time.time()
    while C.n < target and USAGE.cost_usd() < budget:
        n_slots = min(12, target - C.n)
        prompts, specs = [], []
        for _ in range(n_slots):
            if arm == "ihd":
                s = tree.sample_spec(rng)
                avoid = ("\n".join(f"  - {a}" for a in C.attractors[-10:])
                         or "  (nothing yet)")
                prompts.append(PROMPT_TEMPLATE.format(
                    contracts=spec_text(s), avoid=avoid,
                    target_chars=target_chars, max_chars=MAX_PROMPT_CHARS))
                specs.append(s)
            else:
                prompts.append(naive_prompt)
                specs.append({})
        res = []
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(chat, [{"role": "user", "content": p}], 1.0, 1600): i
                    for i, p in enumerate(prompts)}
            for fu in as_completed(futs):
                try:
                    res.append((futs[fu], fu.result()))
                except Exception:
                    pass
        texts, keep = [], []
        for i, r in res:
            t = (r or "").strip().strip('"')
            if 120 <= len(t) <= MAX_PROMPT_CHARS + 200:
                texts.append(t[:MAX_PROMPT_CHARS])
                keep.append(i)
        if not texts:
            continue
        E = embed(texts, batch=EMBED_BATCH)
        for t, e, i in zip(texts, E, keep):
            if C.n >= target:
                break
            C.add(t, e, specs[i], {"arm": arm, "chars": len(t)})
        C.checkpoint()
        print(f"[mureka/{arm}] n={C.n} ${USAGE.cost_usd():.2f} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if arm == "ihd" and C.n and C.n % 25 == 0:
            k = min(25, C.n)
            samp = rng.sample(range(C.n), k)
            blob = "\n".join(f"--- {i} ---\n{C.texts[i][:500]}" for i in samp)
            try:
                raw = chat([{"role": "user",
                             "content": MINE_PROMPT.format(n=k, sample=blob)}],
                           temperature=0.3, max_tokens=900, json_mode=True)
                att = [str(a) for a in parse_json(raw).get("attractors", [])][:12]
                if att:
                    C.add_mined(att)
                    print(f"  mined {len(att)} musical attractors", flush=True)
            except Exception:
                pass
    lens = [len(t) for t in C.texts]
    summary = {"domain": "mureka", "arm": arm, "n": C.n,
               "mean_prompt_chars": round(float(np.mean(lens)), 1) if lens else 0,
               "cost_usd": round(USAGE.cost_usd(), 4),
               "n_axes": len(tree.axes), "n_attractors": len(C.attractors)}
    with open(C.dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    C.close()


def run_audio(arm: str, limit: int = 40, workers: int = 4):
    key = _load_gemini_key()
    if not key:
        print("No Gemini API key -- cannot render audio.", file=sys.stderr)
        return 2
    C = Corpus("lyria", arm)
    out_dir = C.dir / "audio"
    out_dir.mkdir(exist_ok=True)
    todo = [(i, C.texts[i]) for i in range(min(limit, C.n))]
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(generate_lyria, t, key,
                          out_dir / f"{i:05d}.mp3"): i for i, t in todo}
        for fu in as_completed(futs):
            r = fu.result()
            if r and r.get("status") in ("ok", "cached"):
                ok += 1
            print(f"  audio {futs[fu]}: {r}", flush=True)
    print(f"rendered {ok}/{len(todo)}", flush=True)
    return 0


def run_embed(arm: str):
    C = Corpus("lyria", arm)
    files = sorted((C.dir / "audio").glob("*.mp3")) if (C.dir / "audio").exists() else []
    if not files:
        print("no audio to embed")
        return 2
    print(f"embedding {len(files)} tracks", flush=True)
    Ec = clap_embed(files)
    np.save(C.dir / "clap_emb.npy", Ec)
    Em = mert_embed(files)
    np.save(C.dir / "mert_emb.npy", Em)
    json.dump([int(p.stem) for p in files], open(C.dir / "audio_index.json", "w"))
    print("saved CLAP", Ec.shape, "MERT", Em.shape, flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prompts"
    arm = sys.argv[2] if len(sys.argv) > 2 else "ihd"
    if cmd == "prompts":
        run_prompts(arm, int(sys.argv[3]) if len(sys.argv) > 3 else 120,
                    float(sys.argv[4]) if len(sys.argv) > 4 else 3.0)
    elif cmd == "audio":
        sys.exit(run_audio(arm, int(sys.argv[3]) if len(sys.argv) > 3 else 40))
    elif cmd == "embed":
        sys.exit(run_embed(arm))
