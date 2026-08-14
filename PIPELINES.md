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
Docs: `CASTOR/README.md` covers the port; repo-root `README.md` is the upstream
paper README; `SPEC.md` holds design intent.

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

## Path Resolution

Every method repo locates the image set through `CASTOR/benchybench_paths.sh`,
a byte-identical copy in DeGF, ONLY and QWEN-Maritime. It resolves the
BenchyBench root by probing candidates in order and **erroring rather than
guessing**:

1. `$BENCHYBENCH_ROOT` — explicit; invalid is a hard error, not a fallback
2. `$SLURM_SUBMIT_DIR` and its parent — validated, never blindly trusted
3. The script's own location — parent (nested), then repo (standalone)

`$SLURM_SUBMIT_DIR` precedes the script location because SLURM copies a batch
script to a spool directory, so `$0` inside a running job may point at
`/var/spool/...` rather than the repo.

Each job log records the resolved root, so a run against the wrong tree leaves
evidence. To run a repo outside BenchyBench, set `BENCHYBENCH_ROOT` or pass
`--image-folder`.

## Tests

```bash
bash tests/run_all.sh
```

565 assertions across 22 suites. All run locally — no cluster, no GPU, no
model weights, no network.

| Suite | Covers |
|---|---|
| `test_paths.sh` | Path resolution, incl. a simulated SLURM spool execution and a drift guard on the three deployed library copies |
| `test_health_check.py` | State detection, health flags, gate-vs-warning logic |
| `test_regex_eval.py` | Combination identity resolution and its fallback chain |
| `test_judge_submit.py` | Judge record keying, plus a **Python 3.6 compatibility guard** |
| `test_aggregate_report.py` | Outcome tiering and sub-type classification |
| `test_prepare_dataset.py` | Dataset building across all three repo copies |
| `test_shared_metrics.py` | `normalize_state`, JSON extraction — the eval foundation |
| `test_run_config.py` | CLI-over-config precedence across all three repos |
| `test_diffusion_noise.py` | `add_diffusion_noise` — paper-method tensor math |
| `test_documentation.py` | Every first-party module has a docstring |
| `test_reference_selector.py` | The five DeGF-ablation reference modes |
| `test_judge_consensus.py` | P5 panel consensus and field majority votes |
| `test_run_discovery.py` | P1 run discovery and its two inference heuristics |
| `test_separated_discovery.py` | P4 directory discovery incl. the tolerated typo |
| `test_parse_steps.py` | P8 step parsing, incl. the empty-step defect |
| `test_judge_castor.py` | P3 prompt construction and verdict unpacking |
| `test_assertion_coverage.py` | P7 coverage vs contamination selection |
| `test_judge_response_parser.py` | P5 judge reply parsing and truncation handling |
| `test_coherence_stats.py` | P8 Fleiss' kappa over the five-judge panel |
| `test_salvage_stats.py` | P6 FDR correction and P2 resume |
| `QWEN-Maritime/.../self_verify/tests.py` | Self-verification claim/verdict parsing |
| `Eval_CASTOR/tests/` | Eval_CASTOR's own 154-test pytest suite |

The last two already existed but were never run alongside the others, so
regressions in them went unnoticed — four long-standing failures in
`Eval_CASTOR/tests/test_aggregate.py` were found and fixed this way.

The 3.6 guard exists because `judge_submit.py` is the one script that runs on
the node's bare `python3` rather than inside `castor.sif` (Python 3.10). A
`text=True` kwarg broke it in production once; the guard makes the constraint
enforceable rather than remembered.

**What these cannot cover:** apptainer bind behaviour, SLURM scheduling, and
anything needing model weights. A `--dry-run` on the cluster remains the check
for those.

### Structure

Every first-party pipeline module carries class structure for its core logic,
and each class documents the decisions that would otherwise have to be
reverse-engineered from the arithmetic. The only procedural files left are
`__init__.py` markers and test files, where classes would add nothing.

**`DeGF/degf_utils/`, `ONLY/only_utils/` and both `utils/` are excluded on
purpose.** They implement the published ICLR and ICCV methods and are kept
BYTE-IDENTICAL to the papers' code, so results stay directly diffable against
the upstream repositories and no local edit can move a published number. What
reading them taught is recorded in [METHOD_NOTES.md](METHOD_NOTES.md) instead
of by editing the files.

If they ever must change, capture a fixed-seed reference run first so output
can be diffed byte-for-byte afterwards. Local tests verify tensor properties;
they cannot tell you whether a changed number is still the published method.

## Known Gaps

Tracked here so they aren't rediscovered later.

1. **`DeGF/CASTOR/prompts/` is empty** while ONLY and QWEN-Maritime have
   populated `prompts/` directories. DeGF runs take their prompt path from
   config instead.
2. **`build_container.sh` still derives `REPO` from `$SLURM_SUBMIT_DIR`**
   unvalidated, in all three method repos. Low risk — those scripts only locate
   a `.def` file, so a wrong root fails loudly on a missing file rather than
   silently reading the wrong data.

Resolved: ONLY's CASTOR port is now documented; P8+ is indexed in
`Eval_CASTOR/README.md`; the pre-BenchyBench path assumptions were swept across
all four submodules. See `PATH_SWEEP_FINDINGS.md` for the audit.

---

## Pushing

**Submodules must be pushed before the parent.** The parent records a specific
commit per submodule, so if the parent is pushed first, anyone cloning gets
pointers to commits that do not exist on the submodule remotes:

```
fatal: remote error: upload-pack: not our ref <sha>
fatal: Fetched in submodule path 'DeGF', but it did not contain <sha>
```

The clone does not fail cleanly — it falls back to whatever the submodule's
default branch happens to be, so the tree looks populated while containing the
wrong code.

Correct order:

```bash
for d in DeGF ONLY QWEN-Maritime Eval_CASTOR; do
    git -C "$d" push
done
git push          # parent last
```

Verify before pushing the parent — every pointer must appear on a remote:

```bash
for d in DeGF ONLY QWEN-Maritime Eval_CASTOR; do
    ptr=$(git ls-tree HEAD "$d" | awk '{print $3}')
    echo "$d $(git -C "$d" branch -r --contains "$ptr" | head -1)"
done
```
