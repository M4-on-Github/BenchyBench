# visual_classification — Design Decisions

Running log of ideation and design choices. Add entries as decisions are made.
Open items (O-prefix) are resolved before implementation begins.

---

## Resolved decisions

### D1 — Matrix scope
Run all 6 combos (LLaVA + Qwen) × (baseline + DeGF + ONLY) on all prompts.
**Why:** Full comparison is the point. Partial matrices make cross-method deltas ambiguous.

### D2 — Inference orchestration
Option A — one SLURM array job per (model × method), each task = one prompt.
6 sbatch calls per run, N tasks each. Maximum parallelism per combo.
**Why:** Prompts are independent; no reason to serialize them.

### D3 — Concurrency throttle
Default --max-concurrent 4. Mechanism: poll `squeue -u $USER -h | wc -l` before
each sbatch; sleep 60s if at limit. User can raise to 6 on a quiet cluster.
**Why:** At max-concurrent=4 worst case is 4 Qwen jobs = 8 GPUs. Reasonable on
a shared academic cluster.

### D4 — Output filenames
`answers_{model}_{method}_{jobid}.jsonl` — one file per (model × method × prompt).
jobid = SLURM task-level job ID, unique per task. prompt_stem lives as a field
in each record.
**Why:** Fine-grained, easy to retry individual combos, self-describing filenames.

### D5 — Output location
`/data/$USER/BenchyBench_results/visual_classification_{run_name}/`
**Why:** All user-writable cluster paths under /data/$USER. Consistent with other pipelines.

### D6 — Evaluation approach
Both regex (CPU, immediate) and LLM judge (GPU, chained).
Regex for sanity and speed; judge for reliable accuracy on free-text outputs.

### D7 — Report formats
CSV + HTML + JSON.
- CSV: per-record and per-combo summary for downstream analysis
- HTML: visual report with galleries and heatmaps; images embedded as base64
  thumbnails (150px) so report renders anywhere without source images
- JSON: machine-readable run summary for scripting
Everything is copied back locally after the run.

### D8 — Per-record output fields
Defined upfront before running so no re-runs are needed to add fields.
Full schema in SPEC.md §6. Key additions:
- `gpu_hours_used` in per-combo summary
- `inference_time_s` per record + mean/SD in summary
- DeGF-specific: `degf_first_pass_text`, `degf_first_pass_label`,
  `degf_first_pass_correct`, `degf_first_pass_question`, `degf_sd_image_path`
- `degf_first_pass_text` populated in regex eval phase by joining baseline run —
  no changes to inference code
- Health check flags: `repetition_detected`, `hedge_detected`, `refusal_detected`, `length_anomaly`
- Aggregate fields: `difficulty_score`, `consensus`, `failure_type`, `primary_failure_type`

### D9 — Failure mode taxonomy
13 named failure types split into known (need GT) and open (detectable without GT).
Multi-label per image; `primary_failure_type` is the dominant dimension.
Full taxonomy in SPEC.md §10.

### D10 — Health check gate
`health_check.py` exits non-zero if label_bias OR parse_failure_rate > 15% for
any combo. Blocks chained judge submission via --dependency=afterok semantics.
Also: SLURM --mail-type=FAIL on the health check job so user is notified;
`health_report.json` always written even on failure.

### D11 — SLURM chain
inference (all 6) → afterok → health_check → afterok → regex_eval
→ afterok → judge_submit (3 judge array jobs) → afterok → aggregate_report
(outcome) → afterok → aggregate_report (report)
Regex is CPU-only and finishes in seconds so no reason to break the chain.
Full diagram in SPEC.md §5.

### D12 — Pipeline location
`visual_classification/` directory in BenchyBench root. Not a submodule.
**Why:** This is the orchestration layer of BenchyBench, not a peer to DeGF/ONLY/etc.

### D13 — Pipeline name
`visual_classification`
**Why:** Descriptive and unambiguous. No need for a clever name when the repo is
already called BenchyBench.

### D14 — SD image storage
Store SD reference images as PNGs under `inference/sd_images/{jobid}_{image_id}.png`.
Path in per-record field `degf_sd_image_path`. No size cap.
**Why:** Visual inspection of SD images is essential for diagnosing degf_sd_flip
failures and the CLIP plausibility analysis.

### D15 — Outcome analysis tier framework
Three tiers: Tier 1 (universal success), Tier 2 (universal failure),
Tier 3 (contested). Each gets quantitative + qualitative assessment.
Tier 3 split into 5 sub-types (model_split, method_split, method_regression,
prompt_split, combo_split). Multi-label per image; primary = dominant dimension.
Full framework in SPEC.md §9.

### D16 — Metrics beyond basic accuracy
In addition to accuracy/F1/parse_rate, compute:
- Cohen's kappa (regex vs judge) — evaluation reliability proxy
- Fleiss' kappa (inter-judge) — judge panel agreement
- Per-class parse rate — systematic regex bias check
- Verbosity vs correctness correlation — uncertainty signal
- DeGF first-pass vs final accuracy — quantifies SD contribution
- CLIP cosine similarity (SD image vs original) — SD plausibility proxy
- Wrong-image-set Jaccard vs baseline — whether method changes failure distribution
- Prompt ranking stability (Kendall's tau across model × method)
- Per-image prompt variance
- Confusion asymmetry (McNemar on off-diagonal pairs)
Full list and rationale in SPEC.md §11 and PAPER_NOTES.md.

### D17 — Prompt source
`visual_classification/prompts/` is experiment-specific and curated manually.
User copies from `all_maritime_prompts/` intentionally per experiment.
**Why:** Different experiments may need different prompt subsets. Explicit copying
keeps the experiment self-contained.

### D18 — Path resolution
Use `BENCHYBENCH_ROOT` env var (with fallback to dirname of the script) to locate
DeGF/, ONLY/, QWEN-Maritime/ submodule repos at runtime.
**Why:** Already the established convention across all submodule submit scripts.

### D19 — Resume / partial re-run
Default: skip combos whose output file already exists.
`--force` flag overrides and re-runs everything.
**Why:** Safe by default; easy to retry individual failed combos by deleting their file.

### D20 — Regex extraction logic
Case-insensitive, standalone word/phrase match (not substring).
`on fire` and `on_fire` treated as equivalent.
First occurrence wins if multiple labels present.
Hedge flag set if more than one distinct class word found in raw_text.
**Why:** Prevents false positives from incidental mentions; first occurrence captures
the model's leading answer.

### D21 — Judge infrastructure
Reuse Eval_CASTOR's judge container (`castor_judge.sif`) and models.
No new container needed.
**Why:** Container is already built and tested on the cluster.

---

## Open — to revisit before implementation

### O1 — batch_submit.py: how to collect all 6 inference job IDs for the afterok chain
Options:
- A: parse sbatch stdout for job ID, store in list, pass as --dependency=afterok:id1:id2:...
- B: write job IDs to a file, health_check reads the file (more robust to long ID lists)
**Lean:** A — standard pattern, N=6 is short enough.

### O2 — aggregate_report.py: one script with --phase flag vs two separate scripts
**Lean:** one script, two phases — easier to maintain shared utilities.

### O3 — CLIP computation for SD plausibility
Where does it run? Options:
- During inference (adds dependency to DeGF inference job)
- As part of health check (extra model load but CPU is available)
- As part of outcome analysis phase (deferred, cleaner separation)
**Lean:** outcome analysis phase — CLIP is small, CPU-runnable, and logically belongs
with the analysis rather than inference.

### O4 — HTML report: single file vs index + linked pages
With 14 sections and galleries, report.html may be large (20-30MB with base64 images).
Options:
- A: single self-contained HTML (simple, always works)
- B: index.html + linked section pages (smaller per-page, but multiple files)
**Lean:** A — portability wins; 20-30MB is acceptable.
