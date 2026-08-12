# visual_classification — Integrated Benchmark Harness

Runs every (model × method × prompt) combination over the CASTOR images in a
single submission and produces one unified report.

**This is one integration, not the project's entry point.** Most pipelines live
in the submodules and are run directly from there. See
[../PIPELINES.md](../PIPELINES.md) for the full directory.

## What It Does

Submits six inference array jobs — (LLaVA, Qwen) × (baseline, DeGF, ONLY) — by
wrapping the submodules' existing `run_inference.py` scripts in their Apptainer
containers. No model code is reimplemented here. It then chains the evaluation
phases through SLURM `afterok` dependencies:

```
inference (6 array jobs)
  └─ health check          gates on parse-failure rate and label bias
      └─ regex eval        Eval_CASTOR P1
          └─ judge panel   Eval_CASTOR P5 — DeepSeek-R1-32B, GLM-4-32B, Selene-Mini-8B
              └─ outcome   per-image tiers, method/model/prompt splits
                  └─ report  HTML + CSV + JSON
```

Outcome and report run only after all three judge models finish and their
consensus is aggregated.

## When to Use It

**Use it** to compare methods head-to-head on classification, or to get one
report covering all combinations.

**Don't** for anything else — planning coherence, assertion coverage, salvage
analysis, hyperparameter ablations. Those have their own pipelines in the
submodules; see [../PIPELINES.md](../PIPELINES.md).

## Running

Cluster-only. From the BenchyBench root on `pleiades`:

```bash
# Always dry-run first — prints the sbatch plan without submitting
bash visual_classification/submit.sh --run-name exp01 --dry-run

# Full run
bash visual_classification/submit.sh --run-name exp01

# Smoke test: 3 images, no judge (judge is the slow phase)
bash visual_classification/submit.sh --run-name smoke01 --limit 3 --skip-judge

# Re-run evaluation against existing inference output
bash visual_classification/submit.sh --run-name exp01 --eval-only
```

| Flag | Effect |
|---|---|
| `--run-name NAME` | Required. Labels the run and its output directory. |
| `--limit N` | Only N images per combination. |
| `--skip-judge` | Skip the judge panel; chain outcome directly after regex. |
| `--eval-only` | Skip inference; re-evaluate existing output. |
| `--dry-run` | Print the sbatch plan without submitting. |

Monitor with `squeue -u $USER` and
`tail -f /data/$USER/BenchyBench_results/visual_classification/<RUN>/logs/*.out`.

## Output

Everything lands under
`/data/$USER/BenchyBench_results/visual_classification/<RUN>/`:

| Path | Contents |
|---|---|
| `logs/` | All SLURM stdout/stderr for the run |
| `inference/` | `answers_*.jsonl` per combination (plus `sd_images/` for DeGF) |
| `health/health_report.json` | Health gate results |
| `eval/regex/` | `per_record.csv`, `summary.csv` |
| `eval/judge/` | Raw judge output and merged verdicts |
| `eval/outcome_analysis/` | `per_image.csv` and tier breakdowns |
| `report/` | `report.html`, `report.csv`, `report.json`, `run_meta.json` |

Results are **not** tracked in git — they stay under `/data/$USER/`.

## Files

| File | Role |
|---|---|
| `submit.sh` | Orchestrator — submits everything and wires dependencies |
| `slurm/` | Thin SLURM wrappers per phase |
| `health_check.py` | Validates inference output, gates the rest |
| `regex_eval.py` | Label extraction and per-combination metrics |
| `judge_submit.py` | Bridges output into the Eval_CASTOR P5 judge panel |
| `aggregate_report.py` | `--phase outcome` and `--phase report` |
| `prompts/` | Frozen prompt set, indexed by SLURM array task ID |

`prompts/` is auto-discovered by glob, one array task per file. **Treat it as
frozen once a run starts** — changing it invalidates comparisons against
previous runs.

## Further Reading

- [SPEC.md](SPEC.md) — full specification: record schemas, metrics, report sections
- [DECISIONS.md](DECISIONS.md) — design decisions and rationale
- [PAPER_NOTES.md](PAPER_NOTES.md) — which findings map to which paper claims
