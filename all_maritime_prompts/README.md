# Maritime Prompt Library

Prompt variants used across CASTOR experiments. This directory is the **library**
— experiments copy the prompts they need into their own `prompts/` directory,
where they are auto-discovered by glob and indexed by SLURM array task ID.

Copies are frozen once a run starts. Changing a prompt mid-experiment
invalidates comparisons against previous runs, so revise by adding a new
version rather than editing in place.

## `visual_answers/` — Classification

Prompts for the core task: classify a vessel as
`aground / capsized / on_fire / sunken` and extract vessel type, size, and cargo.

| Prompt | Size | Approach |
|---|---|---|
| `promptv1_1shot_noCoT.txt` | 22 lines | One worked example, no reasoning steps |
| `promptv2_CoT+1shot.txt` | 60 lines | Chain-of-thought plus one example |
| `promptv3.txt` | 41 lines | Intermediate revision |
| `promptv4.1.txt` | 109 lines | Staged: evidence catalog → visual grounding questions → JSON |
| `promptv5.txt` | ~800 chars | Single dense paragraph; description-first, no staging |

`promptv4.1.txt` is the most developed. It forces the model to catalog raw
observations before interpreting, then answer grounded questions citing that
evidence, then emit JSON — the structure aimed at suppressing hallucinated
detail. `promptv5.txt` is the opposite bet: one dense descriptive paragraph
with the discriminating questions inlined.

### `separated_fields_questions/`

The multi-turn variant — one narrow question per field instead of a single
prompt returning all fields:

`1_state` · `2_type` · `3_size` · `4_cargo` · `5_envConditions` ·
`6_limitations` · `7_rescuePlan`

Each is a single question (e.g. *"Is the vessel aground, capsized, sunken,
on_fire, or good? Answer with only one of the states."*). `prompt.txt` is the
shared preamble. Evaluated by Eval_CASTOR **P4**
(`pipelines/eval_separated.py`).

## `assertions_planning/` — Salvage Planning

Prompts that ask for a salvage plan as an ordered sequence of steps, each
naming the vessel, resource, or crew involved. All share a preamble
establishing the possible states, vessel types, size and draft bands, and
cargo/hazmat status.

They form an **ablation study** over whether supplying domain assertions
improves plan quality:

| Variant | Size | Role |
|---|---|---|
| `ABLATION_*` | 11–19 lines | Preamble + task only — **assertions removed** |
| `CONTROL_*` | 72–80 lines | Preamble + full assertion list |
| `IMPROVED_*` | 61–67 lines | Refined assertion wording |
| `neutral_assertions.txt` | 43 lines | Original base version |
| `visual_grounded_netural_assertions.txt` | 62 lines | Base + visual grounding |

Each variant comes in two flavors: plain (`*_neutral_assertion*`) and
visually grounded (`*_visual_grounded_*`), the latter tying assertions to what
is observable in the image.

Assertions are hedged (*"may be"*, *"may need"*) rather than prescriptive, so
they supply domain vocabulary without dictating the answer.

Consumed by Eval_CASTOR **P7** (assertion coverage) and **P8/P8+** (plan
coherence).

> **Filename typo:** several files read `netural` rather than `neutral`.
> Preserved as-is because run configs reference these paths — grep for both
> spellings.

## Currently In Use

Which library prompts are live in which experiment:

| Experiment | Prompts |
|---|---|
| `visual_classification/prompts/` | v1, v2, v3, v4.1, v5, `visual_grounded_assertions` |
| `ONLY/CASTOR/prompts/` | v1, v2, v4.1, v5 |
| `QWEN-Maritime/CASTOR/prompts/` | `visual_grounded_netural_assertions` + its CONTROL and ABLATION arms |

`DeGF/CASTOR/prompts/` is empty — DeGF runs currently take their prompt path
from config rather than a local prompt directory.
