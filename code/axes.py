"""
The recursive latent-behavior space, elicited from the generator itself.

An AXIS is a named dimension of poetic variation with a finite set of LEVELS
(e.g. axis "temporal stance" with levels {retrospective, anticipatory,
frozen-present, cyclic}). A SPEC is one level chosen per axis: a point in the
product space. The size of the space is the product of the level counts, so
adding one 4-level axis multiplies capacity by 4 -- combinatorial growth is
what makes "infinite horizon" plausible at all, since no fixed-size axis set
could support unbounded distinct output.

RECURSION. A cell is a region of spec-space. When a cell saturates (its
poems stop being mutually distant), we do not sample it harder. We ask the
generator to SPLIT the axis most responsible for the collapse into finer
sub-levels, or to propose a new axis that is meaningful *specifically inside
that cell*. The space therefore refines exactly where it is being exhausted
and stays coarse elsewhere -- the same adaptive-refinement logic a quadtree
uses, applied to a semantic space the model itself defines.

The refinement is genuinely open-ended: axis levels can be split again and
again, so the reachable spec-space is not bounded by anything fixed at
design time. What IS bounded is the per-step cost of choosing a spec, which
reads only the current axis set and a fixed-size ledger slice.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from pipeline import chat, parse_json

SEED_AXIS_PROMPT = """You are designing the dimensions of a very large, deliberately \
diverse corpus of poems. Do not write any poems.

Propose {k} INDEPENDENT axes of poetic variation. A good axis:
  - is close to ORTHOGONAL to the others (knowing a poem's level on one axis \
should tell you as little as possible about its level on another)
  - changes something structural or stance-level about the poem, not just its topic
  - has levels that are all genuinely usable, none a degenerate or joke option

Avoid the obvious surface axes (subject matter, length, rhyme-or-not) as your \
primary dimensions -- go for axes a working poet would recognise as real craft \
decisions: temporal stance, who is being addressed, the poem's relationship to \
its own claim, syntactic weather, what the line break is doing, the register the \
poem refuses, and so on.

Return JSON only:
{{"axes": [{{"name": "...", "why_orthogonal": "...", "levels": ["...", "...", "...", "..."]}}]}}
Each axis needs 4-6 levels. Keep "why_orthogonal" under 15 words and each level \
name under 10 words."""

SPLIT_PROMPT = """You are refining the dimensions of a large diverse poem corpus.

The following region of the design space has been SATURATED -- poems generated \
inside it have stopped being meaningfully different from each other:

  {cell}

Recent poems from this saturated region all share these tendencies (mined from \
the corpus itself):
  {attractors}

Do ONE of these, whichever genuinely opens more room:
  (a) split ONE of the levels above into 3-5 finer sub-levels that force real \
difference, or
  (b) propose ONE NEW axis that only becomes meaningful inside this region and \
is close to orthogonal to the axes already listed.

Return JSON only:
{{"action": "split" | "new_axis",
  "axis_name": "...",
  "parent_level": "..."  (only for split, else null),
  "levels": ["...", "...", "..."],
  "rationale": "..."}}"""


@dataclass
class Axis:
    name: str
    levels: list[str]
    parent: tuple[str, str] | None = None   # (axis_name, level) this refines
    depth: int = 0

    def to_json(self) -> dict:
        return {"name": self.name, "levels": self.levels,
                "parent": list(self.parent) if self.parent else None, "depth": self.depth}


class AxisTree:
    """The evolving latent-behavior space. Append-only history, live view."""

    def __init__(self, log_path: Path):
        self.axes: list[Axis] = []
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "a")

    # ---- construction -------------------------------------------------
    def seed(self, k: int = 7) -> None:
        obj = None
        for attempt in range(3):
            raw = chat([{"role": "user", "content": SEED_AXIS_PROMPT.format(k=k)}],
                       temperature=1.0, max_tokens=4000, json_mode=True)
            try:
                obj = parse_json(raw)
                break
            except ValueError:
                if attempt == 2:
                    raise
        for a in obj["axes"]:
            levels = [str(x) for x in a["levels"] if str(x).strip()]
            if len(levels) >= 3:
                self.axes.append(Axis(name=str(a["name"]), levels=levels))
        self._log({"event": "seed", "axes": [a.to_json() for a in self.axes]})

    def refine(self, cell_desc: str, attractors: list[str]) -> Axis | None:
        """Ask the generator to open room inside a saturated cell."""
        raw = chat([{"role": "user", "content": SPLIT_PROMPT.format(
            cell=cell_desc, attractors="; ".join(attractors[:12]) or "(none mined yet)")}],
            temperature=1.0, max_tokens=900, json_mode=True)
        try:
            obj = parse_json(raw)
        except ValueError:
            return None
        levels = [str(x) for x in obj.get("levels", []) if str(x).strip()]
        if len(levels) < 2:
            return None
        name = str(obj.get("axis_name") or "refinement")
        parent_level = obj.get("parent_level")
        if obj.get("action") == "split" and parent_level:
            parent = (name, str(parent_level))
            new_name = f"{name}:{parent_level}"
            depth = 1 + max((a.depth for a in self.axes if a.name == name), default=0)
        else:
            parent, new_name, depth = None, name, 0
        if any(a.name == new_name for a in self.axes):
            return None
        ax = Axis(name=new_name, levels=levels, parent=parent, depth=depth)
        self.axes.append(ax)
        self._log({"event": "refine", "axis": ax.to_json(),
                   "rationale": str(obj.get("rationale", ""))[:400],
                   "triggering_cell": cell_desc, "attractors": attractors[:12]})
        return ax

    # ---- sampling -----------------------------------------------------
    def space_size(self) -> float:
        n = 1.0
        for a in self.axes:
            n *= len(a.levels)
        return n

    def sample_spec(self, rng: random.Random, active_only: bool = True) -> dict[str, str]:
        """A spec is one level per applicable axis. A refinement axis applies
        only when its parent level was actually chosen -- that conditionality
        is what makes the space a TREE rather than a flat product, and is why
        refining a saturated cell does not inflate cost everywhere else."""
        spec: dict[str, str] = {}
        for a in self.axes:
            if a.parent is not None:
                pname, plevel = a.parent
                if spec.get(pname) != plevel:
                    continue
            spec[a.name] = rng.choice(a.levels)
        return spec

    def _log(self, rec: dict):
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def spec_text(spec: dict[str, str]) -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in spec.items())
