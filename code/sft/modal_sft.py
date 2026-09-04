"""SFT sample-efficiency sweep on Modal.

One function trains a small base model on n items drawn from one arm and scores
it on IFEval. The training text is `instruction\\n\\nresponse` with loss on the
response only, which is exactly the continuation IFEval's prompts ask for, so no
chat template mismatch sits between the model and the metric.
"""
import json
import os

import modal

APP = modal.App("rac-sft")
VOL = modal.Volume.from_name("rac-sft-vol", create_if_missing=True)
VOLDIR = "/vol"

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "datasets==3.1.0",
        "numpy<2",
        "sentencepiece",
        "protobuf",
        "lm-eval==0.4.5",
        "sentence-transformers==3.3.1",
        "langdetect",
        "immutabledict",
        "nltk",
        "absl-py",
    )
    .env({"HF_HOME": f"{VOLDIR}/hf", "TOKENIZERS_PARALLELISM": "false"})
)

BASE = "Qwen/Qwen2.5-0.5B"


@APP.function(image=IMAGE, gpu="A10G", volumes={VOLDIR: VOL}, timeout=60 * 150,
              secrets=[modal.Secret.from_name("hf-token")])
def train_eval(arm: str, n_items: int, seed: int, epochs: int = 3,
               base: str = BASE, with_ifeval: bool = False,
               lr: float = 1e-5, with_superni: bool = False) -> dict:
    import random
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rows = [json.loads(l) for l in
            open(f"{VOLDIR}/arms/{arm}.jsonl").read().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("response", "").strip()]
    rng = random.Random(seed)
    if n_items < len(rows):
        rows = rng.sample(rows, n_items)

    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.pad_token or tok.eos_token

    MAXLEN = 768

    class SFT(Dataset):
        def __init__(self, rows):
            self.ex = []
            self.tokens = 0
            for r in rows:
                p = r["instruction"].strip() + "\n\n"
                a = r["response"].strip() + tok.eos_token
                pi = tok(p, add_special_tokens=False)["input_ids"]
                ai = tok(a, add_special_tokens=False)["input_ids"]
                ids = (pi + ai)[:MAXLEN]
                lab = ([-100] * len(pi) + ai)[:MAXLEN]
                if len(ids) < 8:
                    continue
                self.ex.append((ids, lab))
                self.tokens += len(ids)

        def __len__(self):
            return len(self.ex)

        def __getitem__(self, i):
            ids, lab = self.ex[i]
            return {"input_ids": ids, "labels": lab}

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        }

    ds = SFT(rows)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    outdir = f"/tmp/{arm}_{n_items}_{seed}"
    args = TrainingArguments(
        output_dir=outdir, num_train_epochs=epochs, per_device_train_batch_size=8,
        gradient_accumulation_steps=2, learning_rate=lr, warmup_ratio=0.03,
        lr_scheduler_type="cosine", logging_steps=25, save_strategy="no",
        bf16=True, report_to=[], seed=seed,
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)

    res = run_dolly(outdir)
    if with_superni:
        res.update(run_superni(outdir))
    if with_ifeval:
        res.update(run_ifeval(outdir))
    out = {"arm": arm, "n_items": len(ds), "train_tokens": ds.tokens,
           "seed": seed, "epochs": epochs, "lr": lr, **res}
    os.makedirs(f"{VOLDIR}/results", exist_ok=True)
    tag = f"{arm}_{n_items}_{seed}"
    if (epochs, lr) != (3, 1e-5):
        tag += f"_e{epochs}_lr{lr:g}"
    with open(f"{VOLDIR}/results/{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    VOL.commit()
    return out


def run_ifeval(model_dir: str) -> dict:
    """IFEval through lm-eval-harness, greedy, no chat template."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model_dir, batch_size=16, dtype="bfloat16")
    r = lm_eval.simple_evaluate(model=lm, tasks=["ifeval"], num_fewshot=0,
                                limit=None, bootstrap_iters=0)
    m = r["results"]["ifeval"]
    return {k: float(v) for k, v in m.items()
            if isinstance(v, (int, float)) and "stderr" not in k}




SUPERNI_FAMILIES = 60      # held-out task families
SUPERNI_PER_TASK = 20      # instances scored per family


@APP.function(image=IMAGE, gpu="A10G", volumes={VOLDIR: VOL}, timeout=60 * 30)
def build_superni(seed: int = 20260901) -> dict:
    """Cache a held-out slice of Super-NaturalInstructions.

    Coverage is a claim about reaching many regions, so the eval has to be many
    task families scored separately rather than one aggregate over a single
    skill. Families are sampled once and reused by every arm.
    """
    import random
    from datasets import load_dataset

    ds = load_dataset("Muennighoff/natural-instructions", split="test",
                      streaming=True)
    by_task, rng = {}, random.Random(seed)
    for r in ds:
        t = r.get("task_name") or r.get("task")
        if not t:
            continue
        b = by_task.setdefault(t, [])
        if len(b) < SUPERNI_PER_TASK:
            b.append({"definition": r.get("definition", ""),
                      "inputs": r.get("inputs", ""),
                      "targets": r.get("targets", "")})
        if len(by_task) > 400 and all(len(v) >= SUPERNI_PER_TASK
                                      for v in list(by_task.values())[:400]):
            break
    tasks = sorted([t for t, v in by_task.items() if len(v) >= SUPERNI_PER_TASK])
    rng.shuffle(tasks)
    keep = {t: by_task[t] for t in tasks[:SUPERNI_FAMILIES]}
    os.makedirs(f"{VOLDIR}/eval", exist_ok=True)
    with open(f"{VOLDIR}/eval/superni.json", "w") as f:
        json.dump(keep, f)
    VOL.commit()
    return {"families": len(keep),
            "instances": sum(len(v) for v in keep.values())}


def run_superni(model_dir: str) -> dict:
    """Per-family accuracy, then the aggregates that reward breadth.

    A corpus that reached many regions should clear the bar on many families,
    so the headline is the macro average over families and the count above a
    threshold -- both of which a corpus good at one thing cannot win.
    """
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = json.load(open(f"{VOLDIR}/eval/superni.json"))
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16).cuda().eval()

    def norm(s: str) -> str:
        s = s.strip().lower().split("\n")[0]
        return re.sub(r"[^a-z0-9 ]+", "", s).strip()

    per_family = {}
    for name, rows in tasks.items():
        prompts = [f"{r['definition'].strip()}\n\n{r['inputs'].strip()}\n\n"
                   for r in rows]
        golds = [r["targets"] for r in rows]
        hits = 0
        for i in range(0, len(prompts), 8):
            batch = prompts[i:i + 8]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=768).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j, o in enumerate(out):
                gen = tok.decode(o[enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                g = golds[i + j]
                gs = g if isinstance(g, list) else [g]
                if any(norm(x) and norm(x) in norm(gen) for x in gs):
                    hits += 1
        per_family[name] = hits / len(rows)

    vals = sorted(per_family.values())
    n = len(vals)
    return {
        "superni_macro": sum(vals) / n,
        "superni_families_over_0.1": sum(v > 0.10 for v in vals) / n,
        "superni_families_over_0.25": sum(v > 0.25 for v in vals) / n,
        "superni_median_family": vals[n // 2],
        "superni_n_families": n,
        "superni_per_family": per_family,
    }




@APP.function(image=IMAGE, volumes={VOLDIR: VOL}, timeout=60 * 20)
def put_dolly(payload: str) -> int:
    os.makedirs(f"{VOLDIR}/eval", exist_ok=True)
    with open(f"{VOLDIR}/eval/dolly_heldout.json", "w") as f:
        f.write(payload)
    VOL.commit()
    return len(json.loads(payload))


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def run_dolly(model_dir: str) -> dict:
    """Answer held-out human instructions, scored against human answers.

    This is the thing instruction tuning actually teaches, on the very
    distribution coverage is computed against: 3,000 Dolly instructions that no
    part of any pipeline read. A base model continues text instead of answering,
    so unlike the zero-shot benchmarks there is real headroom for data to fill.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = json.load(open(f"{VOLDIR}/eval/dolly_heldout.json"))
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16).cuda().eval()

    f1s, f1s_cap, lens, gens, refs = [], [], [], [], []
    for i in range(0, len(rows), 8):
        batch = rows[i:i + 8]
        prompts = [r["instruction"].strip() + "\n\n" for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, o in enumerate(out):
            gen = tok.decode(o[enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            ref = batch[j]["response"].strip()
            gens.append(gen or " ")
            refs.append(ref)
            g, r = gen.lower().split(), ref.lower().split()
            lens.append(len(g))
            def f1(gg, rr):
                if not gg or not rr:
                    return 0.0
                l = _lcs(gg, rr)
                if l == 0:
                    return 0.0
                pr, rc = l / len(gg), l / len(rr)
                return 2 * pr * rc / (pr + rc)
            f1s.append(f1(g[:400], r[:400]))
            cap = min(len(r), 64)
            f1s_cap.append(f1(g[:cap], r[:cap]))
    # semantic similarity ignores the markdown scaffolding that ROUGE-L
    # punishes, so a content gain and a formatting gain can be told apart
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as _np
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                 device="cuda")
        A = st.encode(gens, batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False)
        B = st.encode(refs, batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False)
        sem = float(_np.mean((A * B).sum(axis=1)))
    except Exception as _e:
        sem = float("nan")

    n = len(f1s)
    # ROUGE-L rewards matching the reference LENGTH as well as its content, and
    # every arm trains toward the same target length, so an arm that happens to
    # answer at Dolly's length gains for a reason unrelated to what it covers.
    # The truncated variant clips both sides to the same budget, which removes
    # most of that channel; a gain that survives it is not a length artifact.
    return {"dolly_semantic": sem,
            "dolly_per_item": f1s,
            "dolly_rougeL": sum(f1s) / n,
            "dolly_rougeL_lencap": sum(f1s_cap) / n,
            "dolly_mean_gen_len": sum(lens) / n,
            "dolly_n": n}


@APP.function(image=IMAGE, gpu="A10G", volumes={VOLDIR: VOL}, timeout=60 * 30)
def base_eval(base: str = BASE) -> dict:
    """Score the untrained base model.

    If a fine-tuned run cannot beat this, the experiment is measuring the base
    model rather than the training data, and no comparison between corpora
    means anything.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = "/tmp/base"
    AutoTokenizer.from_pretrained(base).save_pretrained(d)
    AutoModelForCausalLM.from_pretrained(base).save_pretrained(d)
    r = run_superni(d)
    if os.path.exists(f"{VOLDIR}/eval/dolly_heldout.json"):
        r.update(run_dolly(d))
    r.pop("superni_per_family", None)
    os.makedirs(f"{VOLDIR}/results", exist_ok=True)
    with open(f"{VOLDIR}/results/BASE.json", "w") as f:
        json.dump({"arm": "base (no SFT)", "n_items": 0, "train_tokens": 0,
                   "seed": -1, **r}, f, indent=2)
    VOL.commit()
    return r


@APP.function(image=IMAGE, volumes={VOLDIR: VOL}, timeout=60 * 30)
def put_arm(name: str, payload: str) -> int:
    os.makedirs(f"{VOLDIR}/arms", exist_ok=True)
    with open(f"{VOLDIR}/arms/{name}.jsonl", "w") as f:
        f.write(payload)
    VOL.commit()
    return payload.count("\n")


@APP.function(image=IMAGE, gpu="A10G", volumes={VOLDIR: VOL}, timeout=1800)
def smoke() -> dict:
    import importlib.metadata as md
    import torch, transformers, lm_eval  # noqa: F401
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "lm_eval": md.version("lm_eval"), "bf16": torch.cuda.is_bf16_supported()}


@APP.function(image=IMAGE, volumes={VOLDIR: VOL})
def collect() -> list:
    import glob
    out = []
    for p in sorted(glob.glob(f"{VOLDIR}/results/*.json")):
        out.append(json.load(open(p)))
    return out


@APP.function(image=IMAGE, gpu="A10G", volumes={VOLDIR: VOL}, timeout=60 * 90)
def icl_eval(arm: str, k: int = 4, n_pool: int = 950, seed: int = 0,
             n_queries: int = 250,
             base: str = BASE) -> dict:
    """Few-shot prompting with demonstrations retrieved from the corpus.

    This is how a large share of production systems actually consume a corpus:
    not by fine-tuning on it, but by retrieving the nearest examples to each
    query and putting them in the prompt. Coverage says a high-coverage pool has
    a relevant demonstration for more queries, and the retrieval numbers confirm
    it; the open question is whether better demonstrations produce better
    answers. No training happens here, so nothing but the pool differs.
    """
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in
            open(f"{VOLDIR}/arms/{arm}.jsonl").read().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("response", "").strip()]
    rng = random.Random(seed)
    if n_pool < len(rows):
        rows = rng.sample(rows, n_pool)
    held = json.load(open(f"{VOLDIR}/eval/dolly_heldout.json"))[:n_queries]

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    P = st.encode([r["instruction"] for r in rows], normalize_embeddings=True,
                  batch_size=128, show_progress_bar=False)
    Q = st.encode([h["instruction"] for h in held], normalize_embeddings=True,
                  batch_size=128, show_progress_bar=False)
    order = np.argsort(-(Q @ P.T), axis=1)[:, :k]

    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16).cuda().eval()

    def build(i):
        shots = "".join(
            f"{rows[j]['instruction'].strip()}\n\n{rows[j]['response'].strip()}\n\n"
            for j in order[i][::-1])
        return shots + held[i]["instruction"].strip() + "\n\n"

    import time as _t
    t0 = _t.time()
    print(f"[{arm}] pool={len(rows)} queries={len(held)} k={k}", flush=True)
    f1s, sims = [], []
    gens, refs = [], []
    BS = 16
    for i in range(0, len(held), BS):
        if i and i % (BS * 4) == 0:
            print(f"[{arm}] {i}/{len(held)}  {_t.time()-t0:.0f}s", flush=True)
        batch = [build(j) for j in range(i, min(i + BS, len(held)))]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1280).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, o in enumerate(out):
            gen = tok.decode(o[enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            # a few-shot continuation runs on into the next fake example
            gen = gen.split("\n\n")[0].strip()
            ref = held[i + j]["response"].strip()
            gens.append(gen or " ")
            refs.append(ref)
            g, r = gen.lower().split(), ref.lower().split()
            if not g or not r:
                f1s.append(0.0)
                continue
            l = _lcs(g[:180], r[:180])
            if l == 0:
                f1s.append(0.0)
                continue
            pr, rc = l / len(g[:180]), l / len(r[:180])
            f1s.append(2 * pr * rc / (pr + rc))
    A = st.encode(gens, normalize_embeddings=True, batch_size=64,
                  show_progress_bar=False)
    B = st.encode(refs, normalize_embeddings=True, batch_size=64,
                  show_progress_bar=False)
    out = {"arm": arm, "k": k, "n_pool": min(n_pool, len(rows)),
           "icl_rougeL": float(np.mean(f1s)),
           "icl_semantic": float(np.mean((A * B).sum(axis=1))),
           "icl_per_item": [float(x) for x in f1s]}
    os.makedirs(f"{VOLDIR}/results_icl", exist_ok=True)
    with open(f"{VOLDIR}/results_icl/{arm}_k{k}.json", "w") as f:
        json.dump(out, f)
    VOL.commit()
    o = dict(out)
    o.pop("icl_per_item")
    return o


@APP.local_entrypoint()
def icl(k: int = 4):
    """Retrieval-augmented prompting across every pool, no training."""
    arms = ["x_ours_v4", "x_ours_keepall", "x_ours_v3", "x_fs_self_instruct",
            "x_fs_evol_instruct", "x_fs_persona_hub", "x_alpaca_tokmatch",
            "x_personahub", "x_wizardlm"]
    print(f"retrieval-augmented prompting, k={k} demonstrations, pools of 950")
    for r in icl_eval.starmap([(a, k) for a in arms], order_outputs=False):
        print(f"  {r['arm']:22s} rougeL={r['icl_rougeL']:.4f} "
              f"sem={r['icl_semantic']:.4f}")


@APP.local_entrypoint()
def check():
    print(smoke.remote())


@APP.local_entrypoint()
def upload(names: str = ""):
    """Push arm files (instruction/response jsonl) into the volume."""
    from pathlib import Path
    d = Path(__file__).resolve().parent / "data"
    want = [n.strip() for n in names.split(",") if n.strip()]
    files = [d / f"{n}.jsonl" for n in want] if want else sorted(d.glob("arm_*.jsonl"))
    for f in files:
        if not f.exists():
            print(f"  missing {f.name}"); continue
        n = put_arm.remote(f.stem.replace("arm_", ""), f.read_text())
        print(f"  uploaded {f.stem} ({n} rows)")


@APP.local_entrypoint()
def prune(seeds: str = "0,1,2"):
    """The pruning experiment: coverage-selected 2.4k vs random 2.4k vs all 52k.

    Every item and every response comes from Alpaca, so nothing here can be
    explained by a stronger responder writing our answers.
    """
    jobs = []
    for sd in [int(x) for x in seeds.split(",")]:
        jobs += [("alpaca_covsel", 2398, sd), ("alpaca_random", 2398, sd),
                 ("alpaca_full", 51974, sd)]
    print(f"launching {len(jobs)} runs")
    for r in train_eval.starmap(jobs, order_outputs=False):
        print(f"  {r['arm']:16s} seed={r['seed']} n={r['n_items']:5d} "
              f"tok={r['train_tokens']:9d} "
              f"strict_prompt={r.get('prompt_level_strict_acc,none', float('nan')):.4f} "
              f"strict_inst={r.get('inst_level_strict_acc,none', float('nan')):.4f}")


@APP.local_entrypoint()
def superni_cache():
    print(build_superni.remote())


@APP.local_entrypoint()
def sweep(seeds: str = "0,1,2"):
    """Tokens-to-target needs a curve per arm, so budget is swept inside each.

    The full-Alpaca arm is the ceiling the two 2.4k arms are trying to reach;
    the horizontal distance between the curves at a fixed score is the number
    the experiment exists to produce.
    """
    grid = {
        "alpaca_covsel": [300, 600, 1200, 2398],
        "alpaca_random": [300, 600, 1200, 2398],
        "alpaca_full":   [2398, 5000, 12000, 25000, 51974],
    }
    jobs = [(arm, n, sd, 3, BASE, n == max(ns))
            for sd in [int(x) for x in seeds.split(",")]
            for arm, ns in grid.items() for n in ns]
    print(f"launching {len(jobs)} runs across {len(grid)} arms")
    for r in train_eval.starmap(jobs, order_outputs=False):
        print(f"  {r['arm']:14s} seed={r['seed']} n={r['n_items']:5d} "
              f"rougeL={r.get('dolly_rougeL', float('nan')):.4f} "
              f"cap={r.get('dolly_rougeL_lencap', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):3.0f} "
              f"macro={r.get('superni_macro', float('nan')):.4f}")


@APP.local_entrypoint()
def base():
    r = base_eval.remote()
    print(f"BASE MODEL (no SFT): macro={r['superni_macro']:.4f} "
          f"fam>0.1={r['superni_families_over_0.1']:.3f} "
          f"median_family={r['superni_median_family']:.3f}")


@APP.local_entrypoint()
def hpsweep():
    """Find a configuration in which 2,398 items beats 300.

    If no learning rate or epoch budget separates them, the setup has no
    sensitivity to data quantity and the design, not the tuning, is what needs
    changing.
    """
    jobs = []
    for lr in (1e-5, 5e-5, 2e-4):
        for ep in (2, 4):
            for n in (300, 2398):
                jobs.append(("alpaca_full", n, 0, ep, BASE, False, lr))
    print(f"launching {len(jobs)} runs: lr x epochs x (300 vs 2398 items)")
    got = {}
    for r in train_eval.starmap(jobs, order_outputs=False):
        got[(r["lr"], r["epochs"], r["n_items"])] = r["superni_macro"]
        print(f"  lr={r['lr']:<7g} ep={r['epochs']} n={r['n_items']:5d} "
              f"macro={r['superni_macro']:.4f}")
    print("\n  lr      epochs   n=300    n=2398   delta")
    for lr in (1e-5, 5e-5, 2e-4):
        for ep in (2, 4):
            a, b = got.get((lr, ep, 300)), got.get((lr, ep, 2398))
            if a is not None and b is not None:
                print(f"  {lr:<7g} {ep:<8d} {a:.4f}   {b:.4f}   {b-a:+.4f}")


@APP.local_entrypoint()
def dolly_cache():
    """Upload the held-out Dolly half: instructions no pipeline ever read."""
    import json as _j
    import random as _r
    from pathlib import Path as _P
    pub = _P(__file__).resolve().parent.parent / "public_corpora" / "dolly15k.jsonl"
    rows = []
    for line in pub.read_text().splitlines():
        if not line.strip():
            continue
        d = _j.loads(line)
        i, resp, ctx = (d.get("instruction", "").strip(),
                        d.get("response", "").strip(), d.get("context", "").strip())
        if i and resp and not ctx:          # no-context items are answerable alone
            rows.append({"instruction": i, "response": resp})
    r = _r.Random(20260830)
    r.shuffle(rows)
    held = rows[3000:4500]                  # inside the STEER-disjoint half
    print(f"held-out Dolly items: {len(held)} (from {len(rows)} context-free)")
    print("uploaded:", put_dolly.remote(_j.dumps(held)))


@APP.local_entrypoint()
def probe():
    """Does the held-out-instruction eval have headroom the benchmarks lack?

    Three runs decide it: the untrained model, a small corpus and a large one.
    If base is far below both and the large corpus beats the small one, the
    setup can finally see the training data.
    """
    b = base_eval.remote()
    print(f"  base (no SFT)      rougeL={b.get('dolly_rougeL', float('nan')):.4f} "
          f"len={b.get('dolly_mean_gen_len', 0):.0f}  macro={b['superni_macro']:.4f}")
    for r in train_eval.starmap(
            [("alpaca_full", 300, 0, 3, BASE, False, 1e-5),
             ("alpaca_full", 2398, 0, 3, BASE, False, 1e-5)], order_outputs=False):
        print(f"  n={r['n_items']:5d} SFT        "
              f"rougeL={r.get('dolly_rougeL', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):.0f}  "
              f"macro={r['superni_macro']:.4f}")


@APP.local_entrypoint()
def cross(n: int = 950, seeds: str = "0,1,2"):
    """Every corpus at a matched item count, answered by the same responder.

    Subset selection inside one pool was a small perturbation; these corpora
    differ by construction, so if instruction choice moves downstream quality at
    all, this is where it shows.
    """
    arms = ["x_ours_v4", "x_ours_keepall", "x_ours_v3",
            "x_fs_self_instruct", "x_fs_evol_instruct", "x_fs_persona_hub",
            "x_alpaca_tokmatch", "x_personahub", "x_wizardlm"]
    jobs = [(a, n, sd) for sd in [int(x) for x in seeds.split(",")] for a in arms]
    print(f"launching {len(jobs)} runs at matched n={n}")
    for r in train_eval.starmap(jobs, order_outputs=False):
        print(f"  {r['arm']:22s} seed={r['seed']} n={r['n_items']:5d} "
              f"rougeL={r.get('dolly_rougeL', float('nan')):.4f} "
              f"cap={r.get('dolly_rougeL_lencap', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):3.0f}")


@APP.local_entrypoint()
def prose(n: int = 950, seeds: str = "0,1,2"):
    """Same comparison on prose-normalized responses.

    If markdown was the confound, stripping it should lift every arm and
    reorder them; if the ordering survives, the formatting story is wrong.
    """
    arms = ["p_ours_v4", "p_ours_keepall", "p_ours_v3",
            "p_fs_self_instruct", "p_fs_evol_instruct", "p_fs_persona_hub",
            "p_alpaca_tokmatch", "p_personahub", "p_wizardlm"]
    jobs = [(a, n, sd) for sd in [int(x) for x in seeds.split(",")] for a in arms]
    print(f"launching {len(jobs)} prose-normalized runs at n={n}")
    for r in train_eval.starmap(jobs, order_outputs=False):
        print(f"  {r['arm']:22s} seed={r['seed']} "
              f"sem={r.get('dolly_semantic', float('nan')):.4f} "
              f"rougeL={r.get('dolly_rougeL', float('nan')):.4f} "
              f"cap={r.get('dolly_rougeL_lencap', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):3.0f}")


@APP.local_entrypoint()
def lrsweep_dolly():
    """Tune on the metric that responds, using a baseline arm rather than ours.

    The earlier sweep was run on a benchmark where training was harmful, so it
    chose the smallest learning rate. Tuning on a baseline corpus keeps the
    chosen setting from being one that happens to favour us.
    """
    jobs = [("p_fs_self_instruct", 950, 0, ep, BASE, False, lr)
            for lr in (1e-5, 3e-5, 1e-4, 3e-4) for ep in (3, 6)]
    print(f"launching {len(jobs)} runs on a BASELINE arm (prose-normalized)")
    best = (None, -1)
    for r in train_eval.starmap(jobs, order_outputs=False):
        v = r.get("dolly_rougeL", float("nan"))
        print(f"  lr={r['lr']:<7g} ep={r['epochs']} rougeL={v:.4f} "
              f"sem={r.get('dolly_semantic', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):3.0f}")
        if v == v and v > best[1]:
            best = ((r["lr"], r["epochs"]), v)
    print(f"\n  best: lr={best[0][0]:g} epochs={best[0][1]} rougeL={best[1]:.4f}")
    print("  (base model, no SFT: 0.1419)")


@APP.local_entrypoint()
def final(model: str = BASE, n: int = 950, seeds: str = "0,1,2,3,4",
          lr: float = 1e-5, epochs: int = 3):
    """The comparison run at the tuned setting, five seeds, prose-normalized.

    Five seeds rather than three because the spread between arms is small
    enough that three cannot separate them; the base model is included as a
    control line so "did training help at all" is answerable per arm.
    """
    arms = ["p_ours_v4", "p_ours_keepall", "p_ours_v3",
            "p_fs_self_instruct", "p_fs_evol_instruct", "p_fs_persona_hub",
            "p_alpaca_tokmatch", "p_personahub", "p_wizardlm"]
    jobs = [(a, n, sd, epochs, model, False, lr)
            for sd in [int(x) for x in seeds.split(",")] for a in arms]
    print(f"launching {len(jobs)} runs  model={model} n={n} lr={lr:g} ep={epochs}")
    for r in train_eval.starmap(jobs, order_outputs=False):
        print(f"  {r['arm']:22s} seed={r['seed']} "
              f"sem={r.get('dolly_semantic', float('nan')):.4f} "
              f"rougeL={r.get('dolly_rougeL', float('nan')):.4f} "
              f"cap={r.get('dolly_rougeL_lencap', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):3.0f}")


@APP.local_entrypoint()
def scale_check(n: int = 950):
    """Does a 1.5B model separate corpora that a 0.5B cannot?

    Two arms far apart on the 0.5B ordering, one seed each, as a cheap test of
    whether model scale is what the comparison was missing.
    """
    big = "Qwen/Qwen2.5-1.5B"
    jobs = [(a, n, 0, 3, big, False, 1e-5)
            for a in ("p_ours_keepall", "p_fs_persona_hub")]
    print(f"launching {len(jobs)} runs on {big}")
    for r in train_eval.starmap(jobs, order_outputs=False):
        print(f"  {r['arm']:22s} rougeL={r.get('dolly_rougeL', float('nan')):.4f} "
              f"sem={r.get('dolly_semantic', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):3.0f}")


@APP.local_entrypoint()
def tail(n: int = 950):
    """Per-item scores, so the queries furthest from the training data can be
    looked at separately from the bulk.

    Coverage is a claim about not leaving regions empty. A mean over all queries
    is dominated by the ones every corpus serves; if coverage buys anything for
    fine-tuning, it should show on the queries with no near neighbour in the
    training set rather than on the average.
    """
    arms = ["x_ours_v4", "x_ours_keepall", "x_alpaca_tokmatch", "x_fs_evol_instruct"]
    for r in train_eval.starmap([(a, n, 0) for a in arms], order_outputs=False):
        pi = r.get("dolly_per_item") or []
        print(f"  {r['arm']:22s} mean={r.get('dolly_rougeL', float('nan')):.4f} "
              f"n_items={len(pi)}")


@APP.local_entrypoint()
def budget(seeds: str = "0,1,2"):
    """Matched GENERATION BUDGET, not matched item count.

    Every arm here cost the same 2,400 generator calls. Subsampling them to a
    common n throws away the thing a practitioner actually buys: how many usable
    items that budget returned. Retrieval-aimed conditioning returns 1,848 from
    the budget where plain conditioning returns 1,068, and the tail analysis
    says the extra items are what serve the queries a smaller corpus misses.
    """
    full = {"x_ours_v4": 1848, "x_ours_keepall": 1068, "x_ours_v3": 992,
            "x_fs_self_instruct": 8374, "x_fs_evol_instruct": 2188,
            "x_fs_persona_hub": 2109}
    jobs = [(a, n, sd) for sd in [int(x) for x in seeds.split(",")]
            for a, n in full.items()]
    print(f"launching {len(jobs)} runs at each arm's FULL budget-matched size")
    for r in train_eval.starmap(jobs, order_outputs=False):
        print(f"  {r['arm']:22s} seed={r['seed']} n={r['n_items']:5d} "
              f"rougeL={r.get('dolly_rougeL', float('nan')):.4f} "
              f"len={r.get('dolly_mean_gen_len', 0):3.0f}")


@APP.local_entrypoint()
def icl_k():
    """Does the advantage grow with the number of retrieved demonstrations?

    If coverage works by supplying a relevant demonstration, more slots should
    help a covering pool more than a clustered one, whose extra slots fill with
    the same few regions.
    """
    arms = ["x_ours_v4", "x_ours_keepall", "x_alpaca_tokmatch", "x_ours_v3"]
    for r in icl_eval.starmap([(a, k) for k in (1, 2, 8) for a in arms],
                              order_outputs=False):
        print(f"  k={r['k']} {r['arm']:22s} rougeL={r['icl_rougeL']:.4f} "
              f"sem={r['icl_semantic']:.4f}")


@APP.local_entrypoint()
def icl_ladder():
    """Coverage ladder from ONE policy: nested prefixes of the scaled corpus.

    Every other property is held fixed -- same generator, responder, style,
    policy -- so coverage and item count are the only things that move.
    """
    arms = [f"L_v4_10x_{n}" for n in (500, 1000, 2000, 4000)]
    for r in icl_eval.starmap([(a, 4, 100000) for a in arms], order_outputs=False):
        print(f"  {r['arm']:22s} pool={r['n_pool']:5d} "
              f"rougeL={r['icl_rougeL']:.4f} sem={r['icl_semantic']:.4f}")


@APP.local_entrypoint()
def icl_controlled():
    """Coverage held as the ONLY variable: same pool, same n, different spread.

    Three 800-item subsets drawn from one corpus -- greedy-maximum coverage,
    greedy-minimum coverage, and random. Identical source, size, generator,
    responder and style. If ICL tracks coverage here, it is coverage doing the
    work rather than a property that happens to travel with it.
    """
    arms = ["C_v4_high", "C_v4_low", "C_v4_rand"]
    for r in icl_eval.starmap([(a, 4, 100000) for a in arms], order_outputs=False):
        print(f"  {r['arm']:22s} pool={r['n_pool']:5d} "
              f"rougeL={r['icl_rougeL']:.4f} sem={r['icl_semantic']:.4f}")
