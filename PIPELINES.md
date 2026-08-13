# Pipeline Directory

Every runnable pipeline across the four submodules and this repo, with where it
lives and how it is invoked.

**The submodules are where the work happens.** Each is an independent repo with
its own pipelines. `visual_classification/` is one integrated harness that wires
several of them together — it is not the entry point to the project.

Status legend:
- ⭐ **Primary** — actively used, the ones to reach for
- ✓ **Useful** — works, for specific questions
- ⚠️ **Experimental** — ablation, WIP, or superseded

> Status marks reflect the maintainer's current usage, not code quality.
> Everything listed here runs.

---

## Inference — produces `.jsonl` for the evaluation pipelines

### DeGF (`DeGF/`) — LLaVA-1.5-7B
ICLR 2025. Contrastive decoding against a Stable Diffusion reference image.

| Pipeline | Status | Entry point |
|---|---|---|
| Baseline decoding | ⭐ | `CASTOR/run_inference.py --no-diffusion` |
| DeGF decoding | ⭐ | `CASTOR/run_inference.py --use-diffusion` |

Batch submission: `CASTOR/submit.sh` → `CASTOR/submit_job.sh` (SLURM array).
Dataset prep: `CASTOR/prepare_dataset.py`.
Docs: `CASTOR/README.md`, `CASTOR/SPEC.md`, `CASTOR/docs/decisions/`.

### ONLY (`ONLY/`) — LLaVA-1.5-7B
ICCV 2025. Single transformer-layer suppression; no diffusion, so much cheaper
than DeGF.

| Pipeline | Status | Entry point |
|---|---|---|
| Baseline decoding | ✓ | `CASTOR/run_inference.py --no-only` |
| ONLY decoding | ⭐ | `CASTOR/run_inference.py --use-only` |

Batch submission: `CASTOR/submit.sh` → `CASTOR/submit_job.sh`.
Prompts: `CASTOR/prompts/` (4 variants).
Docs: repo-root `README.md` is the upstream paper README. **The CASTOR port
itself is undocumented** — see Known Gaps below.

### QWEN-Maritime (`QWEN-Maritime/`) — Qwen3-VL-8B
Baseline plus DeGF and ONLY ported to Qwen for cross-method comparison.
Requires 2 GPUs.

| Pipeline | Status | Entry point |
|---|---|---|
| Baseline / DeGF / ONLY | ⭐ | `CASTOR/run_inference.py --method {baseline\|degf\|only}` |
| Self-verification | ⭐ | `CASTOR/self_verify/run_self_verify.py` |
| DeGF hyperparameter ablation | ⚠️ | `CASTOR/degf_ablate/run_degf_ablate.py` |
| SD reference image test | ⚠️ | `stable_diffusion_test/` |

Each sub-pipeline has its own `submit.sh`.
Docs: repo-root `README.md` covers all four; plus `CASTOR/DECISIONS.md` and
`CASTOR/SPEC_self_verify.md`.

---

## Evaluation — consumes inference `.jsonl`

All in `Eval_CASTOR/`, run from that repo root. Cluster pipelines share one
Apptainer image (`castor_judge.sif`, vLLM 0.8.5) built from
`containers/container_judge.def`.

| # | Measures | Status | Entry point | Backend |
|---|---|---|---|---|
| P1 | Regex extraction accuracy | ✓ | `pipelines/eval_castor.py` | none |
| P2 | LLM-extracted field accuracy | ✓ | `pipelines/extract_gemma.py` → `eval_castor.py --pre-parsed` | Ollama |
| P3 | Semantic judge (binary) | ✓ | `pipelines/judge_castor.py` | Ollama |
| P4 | Separated-field format accuracy | ✓ | `pipelines/eval_separated.py` | none |
| P5 | LLM-as-judge panel (quality 1–3 + hallucinations) | ⭐ | `containers/judge_panel_submit.sh` | vLLM (cluster) |
| P6 | Salvage plan templating analysis | ✓ | `containers/submit_salvage.sh` | vLLM + embeddings |
| P7 | Assertion coverage | ✓ | `containers/submit_assertion_coverage.sh` | vLLM |
| P8 | Plan coherence (step sequencing) | ✓ | `containers/submit_coherence.sh` | vLLM |
| P8+ | Plan coherence, improved assertions | ⭐ | `pipelines/plan_coherence/improved/run_all.sh` | vLLM |

**P1** is the fast no-backend sanity check - start here.
**P5** is the judge panel `visual_classification/` calls.
**P8+** supersedes P8: self-contained under `plan_coherence/improved/` with its
own `config.yaml`, prompts, assertions, and a one-shot `run_all.sh`. Has its own
`README.md`. Not listed in `Eval_CASTOR/README.md`'s P1–P8 table.

Ground truth for all of the above:
`Eval_CASTOR/human_ground_truth_label/human_gt.csv`, joined on `image`.

---

## Integrated Harness (this repo)

| Pipeline | Status | Entry point |
|---|---|---|
| Full benchmark sweep | ⭐ | `visual_classification/submit.sh --run-name NAME` |

Runs all (model × method × prompt) combinations in one submission, then chains
health check → P1 regex → P5 judge panel → outcome analysis → HTML report via
SLURM `afterok` dependencies.

Use it to compare methods head-to-head on classification. For anything else —
planning, assertion coverage, salvage analysis, ablations — go to the submodule
directly. See [visual_classification/README.md](visual_classification/README.md).

---

## Choosing a Pipeline

| Question | Use |
|---|---|
| Does method X beat baseline on classification? | `visual_classification/` |
| Is my inference output sane? | P1 (`eval_castor.py`) — fast, no backend |
| Is the output right for the right reasons? | P5 judge panel |
| Are the generated salvage plans coherent? | P8+ (`plan_coherence/improved/`) |
| Does the model cover required domain concepts? | P7 assertion coverage |
| Are plans templated / boilerplate? | P6 salvage analysis |
| Which DeGF hyperparameters matter? | `QWEN-Maritime/CASTOR/degf_ablate/` |

---

## Known Gaps

Tracked here so they aren't rediscovered later.

1. **`ONLY/CASTOR/` has no documentation.** No README, SPEC, or decisions file —
   the only docs are the upstream paper README at the repo root, which does not
   describe the CASTOR port. The largest doc gap in the project.
2. **`plan_coherence/improved/` is missing from `Eval_CASTOR/README.md`.** It has
   its own README, but the top-level P1–P8 table does not mention it.
3. **Paths may predate BenchyBench.** The submodules were built standalone and
   later wrapped by this meta-repo. Some docs still describe the pre-wrap layout
   — e.g. `DeGF/CASTOR/README.md` says images "travel with the repo", but they
   now live once at `shipwreck_wiki_images/` in this repo. Path assumptions need
   a systematic sweep across all four submodules.
4. **`DeGF/CASTOR/prompts/` is empty** while ONLY and QWEN-Maritime have
   populated `prompts/` directories.
