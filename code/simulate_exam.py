"""
Exam-item-bank domain: hard min-distance floor, finite packing, saturation.

Domain contrast with poetry (simulate.py):
  * poetry  -- diversity is a SOFT objective. A near-duplicate poem is waste,
               not breach; the objective trades anchor vs repulsion smoothly.
  * exams   -- diversity has a HARD floor. Two operational items closer than
               delta are "enemy items": seeing one answers the other, which is
               a test-security failure at ANY bank size. The floor is a
               constraint, never a term in a weighted sum.

Structural fact that drives everything: an item model (template) exposes only
a few manipulable slots (numbers, names, distractor order, surface story), so
each mode is a LOW-INTRINSIC-DIMENSION disk embedded in embedding space. The
delta-packing number of a k-dim disk of radius R scales like (R/delta)^k --
small and finite for k ~ 3 -- so a hard floor implies a hard ceiling on bank
size, and the empirical question is when each acquisition policy hits it and
what the attempts-per-accepted-item cost curve looks like on the way.

Floor checking uses the FULL history (no HISTORY_CAP subsample): a sampled
check can only certify "probably no duplicate", which is not a security
guarantee. The asymmetry between the two domains' distance queries (soft
objective may subsample; hard floor must be exact) is itself a finding.

FINDING BAKED IN FROM THE FIRST RUN OF THIS SCRIPT: without a typicality
gate, the ungated strategies sailed straight past the ~450-item packing
capacity to the full 2000-item target -- because once the legitimate item
space is delta-saturated, the only candidates that still clear the floor are
the rare incoherent/off-manifold ("junk") generations. The hard novelty
constraint was being satisfied by broken items. Gated variants therefore
exist not as an optimization but as the thing that makes the reported bank
size MEAN anything; per-accept junk is logged for the ungated runs to show
exactly when the bank stops being real.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from simulate import RNG_SEED, D

HERE = Path(__file__).resolve().parent

M_EXAM = 25            # item models (templates)
K_INTRINSIC = 3        # manipulable slots per template
MODE_STD = 0.15
DELTA = 0.20           # enemy-item radius (hard floor)
N_TARGET = 2000        # items requested
BATCH = 16             # candidates drawn per generator call
A_MAX = 64             # max candidates tried for one slot before declaring failure
FAIL_STOP = 50         # consecutive failures = bank exhausted
JUNK_WEIGHT = 0.002    # flawed/incoherent items (assume upstream QA catches most)
JUNK_STD = 4.0
GATE_MULT = 2.5        # typicality gate: reject anchor dist > GATE_MULT * running radius


class ExamWorld:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        centers = rng.normal(size=(M_EXAM, D))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        self.centers = centers * 1.5
        # per-mode orthonormal frame spanning the template's manipulable slots
        self.frames = np.empty((M_EXAM, D, K_INTRINSIC))
        for m in range(M_EXAM):
            A = rng.normal(size=(D, K_INTRINSIC))
            Q, _ = np.linalg.qr(A)
            self.frames[m] = Q
        raw = rng.dirichlet(np.full(M_EXAM, 0.5))
        self.weights = raw * (1.0 - JUNK_WEIGHT)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        u = self.rng.random(n)
        junk = u < JUNK_WEIGHT
        comp = np.zeros(n, dtype=int)
        idx = np.where(~junk)[0]
        if len(idx):
            probs = self.weights / self.weights.sum()
            comp[idx] = self.rng.choice(M_EXAM, size=len(idx), p=probs)
        pts = np.empty((n, D))
        for i in idx:
            m = comp[i]
            z = self.rng.normal(size=K_INTRINSIC) * MODE_STD
            pts[i] = self.centers[m] + self.frames[m] @ z
        jidx = np.where(junk)[0]
        if len(jidx):
            pts[jidx] = self.rng.normal(size=(len(jidx), D)) * JUNK_STD
        comp[junk] = M_EXAM
        return pts, comp


def run_exam(strategy: str, world: ExamWorld, seed: int, gated: bool) -> dict:
    """strategy in {first_legal, farthest_legal, tightest_legal}."""
    rng = np.random.default_rng(seed)
    bank = np.empty((N_TARGET, D))
    comp_hist = np.empty(N_TARGET, dtype=int)
    attempts_log = []           # (bank_size_before, attempts_spent, success)
    accepted = 0
    consec_fail = 0
    sum_vec = np.zeros(D)
    r_ema = 1.0
    r_warm: list[float] = []   # plain mean during warm-up; EMA after
    GATE_START = 50
    t0 = time.time()

    while accepted < N_TARGET and consec_fail < FAIL_STOP:
        attempts = 0
        legal_pool: list[tuple[np.ndarray, int, float]] = []
        while attempts < A_MAX:
            cands, comps = world.sample(BATCH)
            attempts += BATCH
            if gated and accepted > GATE_START:
                centroid = sum_vec / accepted
                anchor = np.linalg.norm(cands - centroid[None, :], axis=1)
                ok = anchor <= GATE_MULT * r_ema
                cands, comps = cands[ok], comps[ok]
                if len(cands) == 0:
                    continue
            if accepted == 0:
                nn = np.full(len(cands), np.inf)
            else:
                nn = cdist(cands, bank[:accepted]).min(axis=1)  # exact, full history
            for j in np.where(nn >= DELTA)[0]:
                legal_pool.append((cands[j], comps[j], float(nn[j])))
            if legal_pool and strategy == "first_legal":
                break
        chosen = None
        if legal_pool:
            if strategy == "first_legal":
                chosen = legal_pool[0]
            elif strategy == "farthest_legal":
                chosen = max(legal_pool, key=lambda c: c[2])
            elif strategy == "tightest_legal":
                chosen = min(legal_pool, key=lambda c: c[2])
        if chosen is None:
            consec_fail += 1
            attempts_log.append((accepted, attempts, 0))
            continue
        consec_fail = 0
        x, c, nnd = chosen
        bank[accepted] = x
        comp_hist[accepted] = c
        attempts_log.append((accepted, attempts, 1))
        # anchor distance measured against the centroid BEFORE adding x --
        # measuring after made the first item distance-0 from itself and
        # seeded the gate's scale near zero, which then rejected everything
        # once the gate activated (first run: bank died at 21 items).
        if accepted > 0:
            r = float(np.linalg.norm(x - sum_vec / accepted))
            if accepted < GATE_START:
                r_warm.append(r)
                r_ema = float(np.mean(r_warm))
            else:
                r_ema = 0.98 * r_ema + 0.02 * max(r, 1e-6)
        sum_vec += x
        accepted += 1

    elapsed = time.time() - t0
    attempts_arr = np.array(attempts_log)
    comp = comp_hist[:accepted].copy()
    junk_mask = comp == M_EXAM
    return dict(
        strategy=strategy,
        gated=gated,
        bank_size=accepted,
        real_bank_size=int((~junk_mask).sum()),
        junk_in_bank=int(junk_mask.sum()),
        exhausted=consec_fail >= FAIL_STOP,
        elapsed_sec=elapsed,
        attempts=attempts_arr,
        comp=comp,
        junk_mask=junk_mask,
        templates_used=len(set(comp[~junk_mask].tolist())),
    )


def main():
    rng = np.random.default_rng(RNG_SEED + 777)
    world = ExamWorld(rng)
    runs = [
        ("first_legal", False),
        ("farthest_legal", False),
        ("tightest_legal", False),
        ("first_legal", True),
        ("farthest_legal", True),
        ("tightest_legal", True),
    ]
    out = {}
    for strat, gated in runs:
        name = f"{strat}{'_gated' if gated else ''}"
        print(f"running {name} ...")
        r = run_exam(strat, world, seed=RNG_SEED + 13 * len(strat) + (1000 if gated else 0), gated=gated)
        out[name] = r
        total_gen = int(r["attempts"][:, 1].sum())
        print(
            f"  bank={r['bank_size']} (real={r['real_bank_size']}, junk={r['junk_in_bank']}) "
            f"exhausted={r['exhausted']} templates={r['templates_used']}/{M_EXAM} "
            f"total_generations={total_gen} gen_per_item={total_gen / max(r['bank_size'], 1):.1f} "
            f"in {r['elapsed_sec']:.1f}s"
        )

    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    np.savez(
        figdir / "exam_results.npz",
        **{f"{s}__attempts": r["attempts"] for s, r in out.items()},
        **{f"{s}__comp": r["comp"] for s, r in out.items()},
        **{f"{s}__junk": r["junk_mask"] for s, r in out.items()},
    )
    summary = {
        s: {
            "bank_size": r["bank_size"],
            "real_bank_size": r["real_bank_size"],
            "junk_in_bank": r["junk_in_bank"],
            "exhausted": bool(r["exhausted"]),
            "templates_used": r["templates_used"],
            "total_generator_calls": int(r["attempts"][:, 1].sum()),
            "gen_per_accepted_item": float(r["attempts"][:, 1].sum() / max(r["bank_size"], 1)),
        }
        for s, r in out.items()
    }
    with open(figdir / "exam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
