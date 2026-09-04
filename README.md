# Recursive Axis Conditioning for Diverse Synthetic Data Generation

Site: https://thehalleyyoung.github.io/rac/ · Paper: [`paper/paper.pdf`](paper/paper.pdf)
· Companion: [Proxy Embeddings](https://thehalleyyoung.github.io/proxy-embeddings/)

A synthetic corpus is judged by a measure, and the measure is chosen before the
corpus is. **Recursive Axis Conditioning** (RAC) is a generation loop that
improves a *chosen* measure: the generator is asked to name the language-valued
axes along which its own outputs can differ, those axes are ranked, their
most-different levels are taken, and an axis that runs out of transverse
variation is split into finer sub-axes. The measure enters at exactly two points
— how an axis is scored, and how one of *K* candidates is selected — and
everything else is shared. The two measures we take furthest:

- **Coverage** — *in n turns, reach as much of the space as possible.*
- **Max-min (packing)** — *in n turns, make no two items resemble each other.*

They pull in different directions, and the paper measures how far: **greedy
k-center wins min-gap in both domains and finishes last on coverage** — a sixth
of random selection — while the two highest-Vendi selectors are the two worst
covering ones.

This is the method paper. Everything about the *embedding* the loop optimizes
being a proxy for the artifact — text embeddings against rendered images,
embedder dependence in audio, conditioning that does not survive the render, and
defects no embedding can see — is the companion paper,
[*Proxy Embeddings*](https://thehalleyyoung.github.io/proxy-embeddings/), which
extends the method across that gap.

## Headline measurements

All from real generations, ~$40 of API spend. Numbers are labelled by the
procedure that produced them (`data/maxmin/provenance.json`).

| finding | number |
|---|---|
| coverage of a held-out human reference, matched *n* = 450 | **1st of 12 corpora**, 0.4441 vs Alpaca's 0.3722, on 1/20 the budget |
| naive psychometric bank, exact-duplicate rate | **73.6%** (one item ×2,726 in 10,000) |
| same, with axis conditioning | 0.0% |
| usable capacity at a blind-adjudicated enemy radius | RAC **100%**, human MMLU 94.9%, naive prompting **15.3%** |
| out-of-objective separation, poems (prosody) | **2.25×** naive, 200/200 draws, budget-matched |
| out-of-objective separation, exam items (architecture) | **1.61×**, 50/50 draws, after deduplication |
| out-of-objective separation, images (pixel statistics) | **1.53×** [1.38, 1.68], *U* = 55/55, *p* = 0.0002 |
| ours vs best synthetic baseline (centered Vendi, exam items) | **+25%** at matched *n* = 1,999 |
| naming a whole 10-way partition vs a 5-way slice of it | entropy **+0.219** [+0.138, +0.297] vs +0.117 |
| enumerating every device form of one construct | dominant form **76% → 24%**, contract held at 100% |
| k-center: min-gap rank / coverage rank | **1st / last** |
| coverage AUC → retrieval quality / in-context quality / fine-tuning | *r* = 0.97 / 0.82 / **−0.17** |

## What's here

```
paper/      source.md (the manuscript), build_paper.py, figures/, and the built
            paper.md, paper.tex, paper.pdf
index.html  the GitHub Pages site: the paper with every figure embedded
build_site.py, site.css, preview.sh
code/       every script needed to reproduce the numbers (see code/README.md);
            code/rac_improve/ holds the axis-level experiments of §8
data/       result JSONs, provenance registry, mined ledgers
```

`paper/source.md` is the manuscript. Figures are referenced as `{{FIG:name.png}}`
tokens and numbered by order of first appearance.

    python3 paper/build_paper.py     # paper.md, paper.tex, paper.pdf
    python3 build_site.py            # index.html from paper/paper.md
    ./preview.sh                     # open the site locally

The page embeds every figure as base64 and loads no external assets, so it
renders locally exactly as it does on GitHub Pages.

## The short version

Conditioned on a fixed prompt, a language model's output concentrates on a
submanifold of dimension *m* far below the dimension *d* of the space it could
reach. Fifteen numerical checks confirm the consequences: novelty at fixed prompt
decays as *n*<sup>−1/m</sup> rather than *n*<sup>−1/d</sup>; a single prompt
ε-covers a vanishing ε<sup>d−m</sup> fraction; only prompt motion *transverse* to
the occupied span raises the ceiling; and the reachability gap between what
conditioning can propose and what a selector could pick from is 17.4 points that
no selection algorithm can close.

We assume an embedding oracle and **no inverse**. We can compute exactly where the
next item should land and have no way to decode that point into text. Everything
follows from that: the system must propose, measure and select rather than solve,
and the only steering handles are language-valued.

Three lessons transfer past the method. **Name the measure**, because covering
and packing are not one problem and each one's characteristic tool damages the
other's score. **Score your control variables by manipulation, not attribution**
— the obvious observational estimator is not merely noisy but inverted,
penalizing an axis in proportion to how well it works (`code/calculus.py`, and
the ground-truth test in §5.3). And **name the whole space, or the repetition
moves** — an axis enumerating a proper subset of the space a measure is defined
over displaces concentration into the complement rather than removing it (§8.3,
seven controlled comparisons with blind judges).

## Reproducing

```bash
pip install numpy scipy matplotlib transformers torch soundfile pillow
python3 code/verify_theory.py     # 8/8 theorem checks
python3 code/verify_slices.py     # 7/7 conditional-dimension checks
python3 code/calculus.py          # calculus self-test, both objectives
```

Live runs need `OPENROUTER_API_KEY`, a local Ollama with `nomic-embed-text`, and
optionally `OPENAI_API_KEY` (images) and a Gemini key (Lyria audio). See
`code/README.md`.

## Licence

Code MIT. Paper CC BY 4.0. Generated corpora are model outputs and are released
alongside the code for replication.
