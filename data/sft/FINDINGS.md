# Does coverage predict downstream SFT quality?

Status: in progress. This file records what is settled, so the negative results
are not lost if later runs overwrite the picture.

## The setup

Qwen2.5-0.5B base, full fine-tune, 3 epochs, lr 1e-5, three seeds per point.
Every corpus is answered by one responder (`openai/gpt-5.6-luna`, same system
prompt and settings), so a difference between arms is a difference in the
instructions rather than in who wrote the answers.

## Settled: the choice of evaluation decided the result

Three evaluations were run on the same checkpoints.

| eval | base model | best fine-tuned | does training help? |
|---|---|---|---|
| IFEval (prompt-level strict) | — | 0.0924 | no separation between arms |
| Super-NaturalInstructions (macro over 60 families) | **0.3092** | 0.3058 | **no — training hurts** |
| Held-out Dolly, ROUGE-L vs human answers | **0.1419** | 0.2038 | **yes, +44%** |

The base-model control is what made this legible, and it should have been the
first thing run. On SuperNI the untrained model beats every fine-tuned one, so
every comparison on that benchmark was measuring base capability plus noise. A
hyperparameter sweep confirmed no configuration rescues it:

| lr | epochs | n=300 | n=2398 | best vs base |
|---|---|---|---|---|
| 1e-5 | 2 | 0.3025 | 0.2933 | −0.0067 |
| 1e-5 | 4 | 0.3042 | 0.3042 | −0.0050 |
| 5e-5 | 2 | 0.2825 | 0.2933 | −0.0159 |
| 5e-5 | 4 | 0.2850 | 0.2925 | −0.0167 |
| 2e-4 | 2 | 0.2125 | 0.2742 | −0.0350 |
| 2e-4 | 4 | 0.2575 | 0.2725 | −0.0517 |

Training is destructive on that benchmark and more training destroys more. The
apparent "more data helps" column is the rate of degradation slowing, not a
gain: at lr 2e-4 the larger corpus is +0.062 over the smaller one and still
0.035 *below* doing nothing at all.

The eval that works starts the base model near the floor, because a base model
continues text rather than answering an instruction. Generation length shows the
mechanism directly: base answers at 85 tokens, 300 items pulls it to 73, and
2,398 items to 55, converging on the length of a human answer.

## Settled: coverage-greedy selection does not buy data efficiency here

Selecting 2,398 of Alpaca's 52k by coverage-greedy, against a random 2,398 from
the same pool, with Alpaca's own responses on both sides so no generation
confound enters.

Seed-matched differences (coverage-selected − random), Dolly ROUGE-L:

| items | seed 0 | mean | within-arm seed spread |
|---|---|---|---|
| 300 | +0.0074 | | 0.0057 |
| 600 | −0.0011 | **+0.0027** | 0.0056 |
| 1200 | +0.0018 | | |

The effect is half the noise. Interpolating items-to-target puts coverage-
selected at 929 items and random at 856 to reach 95% of the full-pool ceiling,
i.e. nominally *worse*. The honest reading is no detectable effect.

Two measurement traps were caught on the way and are worth recording:

- **Token axis flatters the selected arm.** Coverage-greedy picks items about 6%
  shorter (72 vs 76 tokens/item), so plotting against training tokens shifts its
  curve left for a reason unrelated to coverage. The axis is item count.
- **ROUGE-L partly measures answer length.** A length-capped variant, clipping
  generation and reference to the same budget, is reported alongside. Where the
  two columns move together the gain is not a length artifact.

## Open

Cross-corpus comparison at matched n: nine corpora that differ by construction
rather than by subset choice, including faithful re-runs of Self-Instruct,
Evol-Instruct and Persona-Hub on the same generator. Subset selection within one
pool is a small perturbation; this is the comparison with room to show an effect.
