"""
Real pipeline: gpt-5.6-luna (OpenRouter) + nomic-embed-text (local Ollama).

The mechanism, in one paragraph
-------------------------------
We do NOT ask the model for a poem and then filter. We ask the model to
declare, in advance, the LATENT AXES along which poems can differ; then we
choose a point in that combinatorial axis-space that is approximately
ORTHOGONAL to the region the corpus already occupies; then we require the
generated text to exhibit those latent behaviors. The axis space is itself
grown recursively: whenever the corpus saturates a region of the current
space, we ask the model to split the saturated cell into finer sub-axes,
which raises the dimension of the space rather than resampling harder inside
an exhausted one. Attractors -- the words, images, moves, and structures that
keep recurring -- are mined periodically by showing the model a random sample
of 50 accepted poems and asking what they have in common; the answer is
appended to a JSONL ledger and becomes an explicit repulsion constraint on
subsequent prompts.

Why "orthogonalize", not "randomize"
-------------------------------------
Forcing randomness (raise temperature, inject a random seed word) buys
variance in the SURFACE and leaves the underlying distribution's mode
structure untouched; it also degrades quality monotonically, because
temperature has no way to distinguish "surprising" from "wrong". Forcing
approximate orthogonality instead asks: which DIRECTION in behavior-space is
this corpus not yet spending its energy on? That question has a bounded,
computable answer (the trailing eigenspace of the corpus second-moment
matrix), it improves rather than degrades quality when combined with a
quality term, and -- crucially -- it stays meaningful as n grows, whereas
"be random" gets no harder to satisfy and no more useful.

Cost/scale note
---------------
Everything here is O(1) per generated item in corpus size n:
  * spec proposal      -- reads a fixed-size slice of the ledger, not the corpus
  * orthogonality      -- D x D second-moment matrix, updated by rank-1 outer product
  * min-distance       -- bounded reference subsample + exact check against a
                          bounded recent window
  * attractor mining   -- 50 sampled poems every LEDGER_ROUND accepts
so the marginal cost of poem 10,000 equals that of poem 100.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = "openai/gpt-5.6-luna"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def _load_env():
    """Non-interactive shells never source ~/.bashrc, so read the export directly."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    for rc in (Path.home() / ".bashrc", Path.home() / ".zshrc"):
        if not rc.is_file():
            continue
        for line in rc.read_text().splitlines():
            m = re.match(r'\s*export\s+OPENROUTER_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                os.environ["OPENROUTER_API_KEY"] = m.group(1)
                return


_load_env()


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.errors = 0

    def add(self, u: dict):
        with self.lock:
            self.calls += 1
            self.prompt_tokens += int(u.get("prompt_tokens", 0))
            self.completion_tokens += int(u.get("completion_tokens", 0))

    def cost_usd(self) -> float:
        return self.prompt_tokens * 0.2e-6 + self.completion_tokens * 1.2e-6


USAGE = Usage()


def chat(messages: list[dict], temperature: float = 1.0, max_tokens: int = 900,
         retries: int = 4, json_mode: bool = False) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                CHAT_URL, data=data,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.load(r)
            if "error" in payload and not payload.get("choices"):
                raise RuntimeError(str(payload["error"])[:300])
            USAGE.add(payload.get("usage") or {})
            return payload["choices"][0]["message"]["content"] or ""
        except Exception as e:  # transport, 429, malformed body -- all retryable here
            last = e
            with USAGE.lock:
                USAGE.errors += 1
            time.sleep(min(2 ** attempt + random.random(), 20))
    raise RuntimeError(f"chat failed after {retries} attempts: {last}")


def embed(texts: list[str], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        req = urllib.request.Request(
            EMBED_URL,
            data=json.dumps({"model": EMBED_MODEL, "input": chunk}).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    out.extend(json.load(r)["embeddings"])
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    E = np.array(out, dtype=np.float64)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)


def parse_json(text: str) -> dict | list:
    """gpt-5.6-luna occasionally wraps JSON in prose or a fence even under
    json_mode; recover the outermost balanced object/array rather than failing."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not parse JSON from: {text[:200]!r}")
