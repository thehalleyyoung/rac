# Code

Everything needed to reproduce the paper. Two families of scripts, matching the
two objectives, over a shared core.

## Shared core
| file | what it is |
|---|---|
| `calculus.py` | the calculus of diversity: `score_axes` (max-min), `score_axes_coverage`, `score_axes_realized` (evaluated in the artifact space), `recursive_orth_residual` (orthogonalization at every level of the latent tree), `farthest_levels` vs `coverage_levels` |
| `axes.py` | the recursive latent lattice: elicit axes from the generator, refine a saturated level into conditional child axes |
| `pipeline.py` | OpenRouter client, local Ollama embeddings, robust JSON recovery, cost accounting |
| `provenance.py` | which procedure produced which number (P1/P2/P2.5/P3) |

## Theory (no API keys, no network)
```bash
python3 verify_theory.py    # 8/8   core theorems
python3 verify_slices.py    # 7/7   conditional-dimension theorems
python3 calculus.py         #       calculus self-test, both objectives
```

## Simulation
`simulate.py`, `simulate_exam.py`, `simulate_slices.py`, `experiment_budget.py`,
`coverage_horizon.py` — then `make_figures.py`.

## Real generation
Needs `OPENROUTER_API_KEY` and a local Ollama with `nomic-embed-text`.
```bash
python3 real_run.py dalle naive 10000 4.0
python3 baselines.py dalle persona 2500 2.5     # + self_instruct, evol_instruct, high_temp
python3 metrics.py && python3 compare_arms.py && python3 head_to_head.py 1000
```

## Rendered artifacts
`OPENAI_API_KEY` for images, a Gemini key for Lyria audio.
```bash
python3 render_images.py render dalle_ihd 100 high
python3 render_images.py embed  dalle_ihd 0   high
python3 vision_compare.py                       # VISION_TAG=high|medium
python3 audio_domain.py prompts ihd 60 2.0 && python3 audio_domain.py audio ihd 50
python3 audio_compare.py
```

## Cross-modal steering (current framework, no rejection)
```bash
python3 lyria_steer.py 100          # CLAP-steered audio
python3 image_steer.py maxmin 60    # CLIP-steered, max-min objective
python3 image_steer.py coverage 60  # CLIP-steered, coverage objective
```

## Coverage half
See `coverage/` — same core, covering objective, its own benchmark and
head-to-head harnesses.
