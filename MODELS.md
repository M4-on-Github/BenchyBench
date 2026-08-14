# Model Setup

Every model the pipelines use, where it must live, and how it gets there.

**Most of this is automatic.** The container build scripts and
`judge_panel_submit.sh` download what they need on first run. This document
exists for when that fails — usually because compute nodes have no internet —
and so you can see the disk cost before starting.

All weights live under `/data/$USER/` (1.9 TB quota). Home is only 500 GB and
is not where models go.

---

## Quick start

```bash
cd ~/BenchyBench

# LLaVA-1.5-7B + the DeGF container
sbatch DeGF/CASTOR/build_container.sh

# The ONLY container (reuses the LLaVA weights above)
sbatch ONLY/CASTOR/build_container.sh

# Qwen3-VL-8B + its container
sbatch QWEN-Maritime/CASTOR/build_container.sh

# The judge container (vLLM)
sbatch Eval_CASTOR/containers/build_judge_container.sh

# Judge weights download themselves on first use:
cd Eval_CASTOR && bash containers/judge_panel_submit.sh --run <run_name>
```

Roughly **150 GB** and several hours end to end. Each step skips itself if the
destination already exists, so a failed run is safe to repeat.

---

## Inference models

| Model | HuggingFace repo | Lands at | Size | Used by |
|---|---|---|---|---|
| LLaVA-1.5-7B | `liuhaotian/llava-v1.5-7b` | `/data/$USER/llava-v1.5-7b` | ~14 GB | DeGF, ONLY |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `/data/$USER/qwen3vl-8b` | ~17 GB | QWEN-Maritime, P8+ |

Both are fetched by their repo's `CASTOR/build_container.sh` via
`snapshot_download`, which `CASTOR/submit.sh` chains before the array job.

**LLaVA is downloaded once, not twice.** DeGF and ONLY both target
`/data/$USER/llava-v1.5-7b` and both guard with a non-empty check, so whichever
container you build first fetches the weights and the second skips them. The
14 GB is paid once even though two repos use it.

Override the Qwen checkpoint with `QWEN_MODEL_HF_ID` if you need a variant.

## DeGF auxiliary models

DeGF generates a reference image and compares it to the original, so it needs
three more. These are **not** downloaded ahead of time — `diffusers` and
`transformers` pull them into the HF cache on first use.

| Model | Repo | Size | Purpose |
|---|---|---|---|
| Stable Diffusion v1-5 | `runwayml/stable-diffusion-v1-5` | ~4 GB | Generates the reference image |
| SD Image Variations | `lambdalabs/sd-image-variations-diffusers` @ `v2.0` | ~4 GB | Alternative reference mode |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | ~600 MB | Scores reference-vs-original similarity |

The revision pin on image-variations is deliberate: `v1.0` conditions
differently and produces visibly different variants.

**Baseline runs pay none of this.** DeGF's imports are inside the diffusion
branch, so `--no-diffusion` never loads them.

Cache location is set by the submit scripts:

```bash
export HF_HOME=/data/$USER/.cache/huggingface
```

## P5 judge panel

Three judges score every record. `judge_panel_submit.sh` checks each
destination and submits `download_job.sh` for whatever is missing, so normally
you run nothing here.

| Judge | HuggingFace repo | Directory | GPU mem |
|---|---|---|---|
| DeepSeek-R1-32B | `casperhansen/deepseek-r1-distill-qwen-32b-awq` | `deepseek-r1-distill-qwen-32b-awq` | 52 GB |
| GLM-4-32B | `mratsim/GLM-4-32B-0414.w4a16-gptq` | `glm-4-32b-0414-gptq` | 52 GB |
| Selene-Mini-8B | `AtlaAI/Selene-1-Mini-Llama-3.1-8B` | `selene-1-mini-llama-3.1-8b-awq` | 16 GB |

All under `/data/$USER/`. Roughly **45 GB** total.

Selene ships unquantized and is quantized locally on first use by
`pipelines/judge_panel/quantize_model.py` (AutoAWQ, `group_size=128`). That
takes tens of minutes once. The other two are already quantized upstream.

The judges **run sequentially, not in parallel** — two 32B models will not fit
on the allocated GPUs together, so wall time is the sum.

## P8 / P8+ plan coherence

P8 uses a **five**-judge panel, larger than P5's three:

```
deepseek_r1_32b · glm4_32b · llama_3_3_70b · phi4_14b · gemma4_31b
```

P8+ (the improved pipeline) uses three models, configured in
`pipelines/plan_coherence/improved/config.yaml`:

| Role | Directory | Notes |
|---|---|---|
| Plan generation | `qwen3vl-8b` | shared with QWEN-Maritime |
| Coherence judge | `llama-3.3-70b-instruct-w4a16` | ~35 GB, fits one RTX6000Ada |
| Assertion coverage | `selene-1-mini-llama-3.1-8b-awq` | shared with P5 |

Edit the `paths:` block in that config before the first run. `run_all.sh`
verifies the image set exists and fails early if the root is wrong.

## Ollama models (P2, P3, P6)

Three pipelines use a local Ollama server rather than cluster vLLM, so they can
be developed on a laptop without an allocation.

```bash
ollama pull gemma4:31b            # or set CASTOR_SALVAGE_MODEL
export OLLAMA_HOST=http://localhost:11434
```

| Pipeline | Default model | Purpose |
|---|---|---|
| P2 `extract_gemma.py` | `gemma4:31b-cloud` | Extracts structured fields from prose |
| P3 `judge_castor.py` | same | Semantic correct/wrong judge |
| P6 `normalize.py` | same, embeddings endpoint | Clusters salvage phrasings |

P6 can skip Ollama entirely with `--backend local`, which uses
`sentence-transformers/all-MiniLM-L6-v2` (~90 MB, downloaded on first use).
That is the path used on the cluster, where no Ollama service runs.

---

## When the automatic download fails

The usual cause is compute nodes having no internet. `download_job.sh` says so
in its own header. Run the download on the **head node** instead:

```bash
hf download casperhansen/deepseek-r1-distill-qwen-32b-awq \
    --local-dir /data/$USER/deepseek-r1-distill-qwen-32b-awq
```

Then re-run `judge_panel_submit.sh` — it detects the populated directory and
skips that download.

Every download step is guarded by a non-empty-directory check, so re-running
after a partial failure only fetches what is missing. A directory that is
present but *incomplete* is the one case that misleads it — delete such a
directory rather than re-running over it.

## Disk budget

| Group | Approx. |
|---|---|
| LLaVA-1.5-7B | 14 GB |
| Qwen3-VL-8B | 17 GB |
| DeGF auxiliary (SD ×2, CLIP) | 9 GB |
| P5 judges (×3) | 45 GB |
| P8+ Llama-3.3-70B | 35 GB |
| Apptainer containers (×4) | 24 GB |
| **Total** | **~145 GB** |

Comfortable inside the 1.9 TB `/data/$USER` quota. It will **not** fit in the
500 GB home directory alongside anything else, which is why every path points
at `/data/$USER`.

## Containers

Models are useless without the matching environment. Four Apptainer images:

| Image | Built by | Holds |
|---|---|---|
| `castor.sif` | `DeGF/CASTOR/build_container.sh` | torch 2.0.1, transformers 4.31.0, diffusers |
| `castor_ONLY.sif` | `ONLY/CASTOR/build_container.sh` | same pins as DeGF |
| `castor_qwen.sif` | `QWEN-Maritime/CASTOR/build_container.sh` | torch 2.3.1, transformers ≥ 4.51 |
| `castor_judge.sif` | `Eval_CASTOR/containers/build_judge_container.sh` | vLLM 0.8.5, pandas, scipy, sentence-transformers |

**The `transformers==4.31.0` pin on DeGF and ONLY is not negotiable.** Both
methods monkey-patch that version's generation functions; a different version
has a different internal contract and the patch either fails or silently
diverges. See [METHOD_NOTES.md](METHOD_NOTES.md).

Qwen needs a *newer* transformers, which is why it has its own container rather
than sharing one.
