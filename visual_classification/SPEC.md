# visual_classification — Pipeline Specification

Benchmark harness inside BenchyBench. Runs every (model × method × prompt) combo
over the CASTOR shipwreck image dataset, evaluates outputs against human GT with
regex and LLM judges, and produces a structured quantitative + qualitative report
covering successes, failures, and contested cases.

---

## 1. Scope

**Task:** 4-class shipwreck image classification (aground / capsized / on_fire / sunken).
**Dataset:** 110 images across 4 classes (aground=42, capsized=19, on_fire=16, sunken=33).
**GT source:** `Eval_CASTOR/human_ground_truth_label/human_gt.csv`

**Model × method matrix (6 combos):**

| | baseline | DeGF | ONLY |
|---|---|---|---|
| LLaVA-1.5-7B | ✓ | ✓ | ✓ |
| Qwen3-VL-8B | ✓ | ✓ | ✓ |

**Prompts:** all `.txt` files in `visual_classification/prompts/` — auto-discovered.
N prompts → 6N inference jobs, 6N output files.

---

## 2. Directory layout (pipeline source)

```
visual_classification/
  SPEC.md                       ← this file
  DECISIONS.md                  ← design decision log
  PAPER_NOTES.md                ← what to look for and report in the paper
  README.md                     ← usage (written after code is built)
  prompts/                      ← *.txt prompt variants; experiment-specific, curated manually
  batch_submit.py               ← orchestrator: submits all jobs, throttles concurrency
  health_check.py               ← CPU phase: validates inference outputs, gates judge
  regex_eval.py                 ← CPU phase: label extraction + accuracy
  judge_submit.py               ← submits LLM judge jobs (chained after regex)
  aggregate_report.py           ← outcome analysis + report generation
  slurm/
    infer_llava.sh              ← SLURM body for LLaVA inference array tasks
    infer_qwen.sh               ← SLURM body for Qwen inference array tasks
    judge_job.sh                ← SLURM body for LLM judge tasks
```

---

## 3. Output structure (cluster)

All outputs under `/data/$USER/BenchyBench_results/visual_classification/{run_name}/`

```
├── inference/
│   ├── answers_{model}_{method}_{prompt_stem}_{jobid}.jsonl    ← one per (model × method × prompt)
│   ├── questions_{model}_{method}_{prompt_stem}_{jobid}.jsonl  ← dataset passed to that job
│   ├── meta_{model}_{method}_{prompt_stem}_{jobid}.json        ← sidecar: method/prompt_stem/paths
│   ├── firstpass_{model}_degf_{prompt_stem}_{jobid}.jsonl      ← DeGF only: pre-SD baseline answers
│   └── sd_images/                                              ← DeGF only
│       └── {model}_degf_{prompt_stem}/                         ← one subfolder per degf run
│           └── {category}_{image_stem}.png
│
├── health/
│   └── health_report.json       ← per-combo flags + go/no-go verdict
│
├── eval/
│   ├── regex/
│   │   ├── per_record.csv       ← one row per image per combo
│   │   └── summary.csv          ← one row per (model × method × prompt)
│   ├── judge/
│   │   ├── raw/
│   │   │   └── judge_{model}_{method}_{prompt_stem}_{judge_model}_{jobid}.jsonl
│   │   ├── per_record.csv
│   │   └── summary.csv
│   └── outcome_analysis/
│       ├── per_image.csv                  ← difficulty_score, tier, sub_types, consensus
│       ├── tier1_successes.csv
│       ├── tier2_failures.csv
│       ├── tier3_method_split.csv
│       ├── tier3_method_regression.csv
│       ├── tier3_model_split.csv
│       ├── tier3_prompt_split.csv
│       └── tier3_combo_split.csv
│
└── report/
    ├── report.html              ← self-contained; images as base64 thumbnails (150px)
    ├── report.csv               ← one row, run-level aggregate
    ├── report.json              ← machine-readable run summary
    └── run_meta.json            ← flags, prompt sha256s, timestamps, model paths, node
```

Everything is copied back locally after the run. `report.html` is self-contained
(base64 thumbnails) so it renders anywhere without the source images.

---

## 4. Pipeline phases

### Phase 0 — Setup (`batch_submit.py`, login node)
1. Scans `visual_classification/prompts/*.txt` — discovers N prompt files
2. Creates full output directory tree
3. Loads `human_gt.csv` — validates all 110 image IDs have a GT label
4. For each of the 6 combos builds an sbatch command with `--array=0-{N-1}`
5. Submits one by one; before each sbatch polls `squeue -u $USER -h | wc -l`;
   sleeps 60s and retries if at `--max-concurrent` limit (default 4)
6. Writes `run_meta.json` immediately with submission metadata

**CLI:**
```
python visual_classification/batch_submit.py
  --run-name NAME             (required)
  --max-concurrent N          (default 4; raise to 6 on quiet cluster)
  --models llava qwen         (default: both)
  --methods baseline degf only (default: all three)
  --prompts-dir PATH          (default: visual_classification/prompts/)
  --limit N                   (smoke test: first N images only)
  --skip-judge                (inference + regex only, no GPU judge)
  --dry-run                   (print job plan, submit nothing)
  --force                     (re-run even if output files already exist)
```

Resume: by default skips combos whose output file already exists. Use `--force` to override.

### Phase 1 — Inference (GPU, SLURM array jobs)

**LLaVA × baseline and LLaVA × ONLY** (1 GPU, 40G, RTX6000ADA):

Each array task (= one prompt):
1. `prepare_dataset.py` pairs each of 110 images with the prompt text →
   `questions_{...}.jsonl`
2. `run_inference.py` (DeGF or ONLY repo) loads LLaVA-1.5-7B in 4-bit
3. Per image: forward pass → decode → record `raw_text`, `inference_time_s`,
   `output_length_chars`
4. Writes `answers_llava_{method}_{jobid}.jsonl`

**LLaVA × DeGF** (1 GPU, 40G):

Per image:
1. Forward pass with `sd_desc_prompt` + image → description text
2. Stable Diffusion generates reference image → save PNG to `sd_images/`
3. Forward pass with original image + classification prompt → token dist P_img
4. Forward pass with SD reference image + classification prompt → token dist P_ref
5. Contrastive decoding: amplify tokens where P_img >> P_ref, suppress otherwise → `raw_text`
6. Records `raw_text`, `degf_sd_image_path`, `degf_first_pass_question` (the sd_desc_prompt)
7. `degf_first_pass_text` / `degf_first_pass_label` populated during regex eval phase
   by joining the baseline run on (model, image_id, prompt_stem) — no inference code changes

**Qwen × all three methods** (2 GPU, 48G):

`QWEN-Maritime/CASTOR/run_inference.py` handles baseline / degf / only via
`--method` flag. Sampling: temperature=0.7, top_p=0.8, top_k=20.

### Phase 2 — Health check (`health_check.py`, CPU, ~1 min)

Chained `afterok` from all inference jobs. Reads all 6N JSONL files.

**Per record:**
1. Regex attempt — does `raw_text` contain a valid class word?
2. N-gram repetition scan — flag if any 5-gram repeats 3+ times
3. Hedge detection — flag if >1 distinct class word found
4. Refusal detection — keyword scan ("cannot", "unclear", "unable to determine")
5. Length outlier — flag if `output_length_chars` > combo mean + 3×SD

**Per combo:**
6. Label bias — flag if any predicted class > 40% of predictions
7. Self-inconsistency — per image, across prompts for same model+method: does label flip?

Writes `health/health_report.json`.
**Exits non-zero** (blocking judge chain) if any combo has label_bias OR
parse_failure_rate > 15%.

### Phase 3 — Regex eval (`regex_eval.py`, CPU)

Chained `afterok` from health check.

1. Loads all inference JSONLs + `human_gt.csv`
2. Label extraction: case-insensitive standalone word/phrase match; `on fire` and
   `on_fire` treated equivalent; multiple labels found → flag hedge, take first occurrence
3. Sets `regex_correct = (parsed_label == gt_label)`
4. Joins baseline records to populate `degf_first_pass_text` and
   `degf_first_pass_label` for DeGF rows (join key: model + image_id + prompt_stem)
5. Computes per-combo aggregates (see §6)
6. Writes `eval/regex/per_record.csv` and `eval/regex/summary.csv`

### Phase 4 — LLM judge (`judge_submit.py`, GPU)

Chained `afterok` from regex eval. Submits one array job per judge model
(3 judges: deepseek_r1_32b, glm4_32b, selene_mini_8b). Uses Eval_CASTOR judge
container (`castor_judge.sif`) — no new container needed.

Per record:
1. Structured prompt: "Model output: [raw_text]. GT label: [gt_label].
   What class did the model predict? Was it correct? Answer: CORRECT / INCORRECT / UNSURE."
2. Parses → `judge_verdict`, `judge_label`
3. Aggregates across 3 judges: majority vote verdict; flags records where judges disagree

Writes `eval/judge/per_record.csv` and `eval/judge/summary.csv`.

### Phase 5 — Outcome analysis (`aggregate_report.py --phase outcome`, CPU)

Chained `afterok` from judge jobs.

1. Merges regex + judge per_record CSVs
2. Per image across all 6N combos: computes `difficulty_score`, `consensus`
3. Assigns tier:
   - Tier 1: universal success (all combos correct)
   - Tier 2: universal failure (all combos wrong)
   - Tier 3: contested (mixed)
4. For tier 3 assigns sub-types (multi-label):
   - `model_split`: LLaVA and Qwen disagree
   - `method_split`: baseline wrong, DeGF or ONLY right
   - `method_regression`: baseline right, DeGF or ONLY wrong
   - `prompt_split`: same model+method, label flips across prompts
   - `combo_split`: multiple dimensions vary simultaneously
5. Primary sub-type = dimension explaining most variance for that image
6. Writes all tier CSVs and `eval/outcome_analysis/per_image.csv`

### Phase 6 — Report (`aggregate_report.py --phase report`, CPU)

Chained `afterok` from outcome analysis.

1. Reads all CSVs
2. Renders confusion matrices per (model × method) → base64 PNG
3. Renders prompt sensitivity heatmap → base64 PNG
4. Loads shipwreck images → 150px thumbnails → base64 for galleries
5. Assembles `report.html` with all 14 sections (see §8)
6. Writes `report.csv`, `report.json`, finalizes `run_meta.json`

---

## 5. SLURM chain

```
batch_submit.py
  ├── sbatch infer_llava.sh --array=0-{N-1}  (baseline)  ─┐
  ├── sbatch infer_llava.sh --array=0-{N-1}  (degf)      ─┤
  ├── sbatch infer_llava.sh --array=0-{N-1}  (only)      ─┤ all 6 jobs submitted
  ├── sbatch infer_qwen.sh  --array=0-{N-1}  (baseline)  ─┤ with throttle
  ├── sbatch infer_qwen.sh  --array=0-{N-1}  (degf)      ─┤
  └── sbatch infer_qwen.sh  --array=0-{N-1}  (only)      ─┘
        │
        └── afterok (all 6) → health_check.py
              └── afterok → regex_eval.py
                    └── afterok → judge_submit.py (submits 3 judge array jobs)
                          └── afterok (all 3) → aggregate_report.py --phase outcome
                                └── afterok → aggregate_report.py --phase report
```

---

## 6. Per-record schema

One row per image per (model × method × prompt). Written during inference;
eval and aggregate fields filled in later phases.

### Core fields

| Field | Type | Phase | Notes |
|---|---|---|---|
| `run_name` | str | setup | |
| `model` | str | inference | `llava` \| `qwen` |
| `method` | str | inference | `baseline` \| `degf` \| `only` |
| `prompt_stem` | str | inference | prompt filename without .txt |
| `image_id` | str | inference | e.g. `aground/00017` |
| `gt_label` | str | regex eval | from human_gt.csv |
| `raw_text` | str | inference | full model output |
| `output_length_chars` | int | inference | `len(raw_text)` |
| `inference_time_s` | float | inference | wall time for this image |
| `parsed_label` | str\|null | regex eval | extracted class or null |
| `parse_success` | bool | regex eval | |
| `regex_correct` | bool\|null | regex eval | null until eval phase |

### Health check flags

| Field | Type | Phase |
|---|---|---|
| `repetition_detected` | bool | health check |
| `hedge_detected` | bool | health check |
| `refusal_detected` | bool | health check |
| `length_anomaly` | bool | health check |

### Judge fields

| Field | Type | Notes |
|---|---|---|
| `judge_model` | str\|null | which judge produced this row |
| `judge_verdict` | str\|null | `CORRECT` \| `INCORRECT` \| `UNSURE` |
| `judge_label` | str\|null | class label the judge extracted |
| `judge_correct` | bool\|null | derived: judge_verdict == CORRECT |
| `judges_agree` | bool\|null | all 3 judges gave same verdict |

### DeGF-only fields (null for baseline / only)

| Field | Type | Notes |
|---|---|---|
| `degf_first_pass_text` | str\|null | raw_text from baseline run (joined in regex eval) |
| `degf_first_pass_label` | str\|null | parsed label from baseline run |
| `degf_first_pass_correct` | bool\|null | whether first pass was correct |
| `degf_first_pass_question` | str\|null | the sd_desc_prompt string (fixed per run) |
| `degf_sd_image_path` | str\|null | relative path to SD reference image PNG |

### Aggregate fields (populated in outcome analysis phase)

| Field | Type | Notes |
|---|---|---|
| `difficulty_score` | float | fraction of combos wrong for this image |
| `consensus` | str | `all_correct` \| `all_wrong` \| `contested` |
| `failure_type` | list[str] | multi-label; see §7 |
| `primary_failure_type` | str\|null | dominant dimension for tier 3 images |

---

## 7. Per-combo summary schema

One row per (model × method × prompt_stem).

| Field | Notes |
|---|---|
| `model`, `method`, `prompt_stem` | |
| `n_images`, `n_parsed`, `parse_rate` | |
| `regex_accuracy` | |
| `judge_accuracy` | majority-vote across 3 judges |
| `macro_f1`, `weighted_f1` | |
| `f1_aground`, `f1_capsized`, `f1_on_fire`, `f1_sunken` | |
| `parse_rate_aground` … `parse_rate_sunken` | per-class parse success rate |
| `delta_accuracy_vs_baseline` | signed diff; null for baseline rows |
| `delta_f1_vs_baseline` | |
| `n_regressions` | images where this method wrong but baseline right |
| `n_uplifts` | images where this method right but baseline wrong |
| `wrong_set_jaccard_vs_baseline` | Jaccard of wrong-image-sets; null for baseline |
| `regex_judge_kappa` | Cohen's kappa between regex and judge for this combo |
| `inter_judge_kappa` | Fleiss' kappa across 3 judges for this combo |
| `verbosity_correct_correlation` | Pearson r: output_length_chars vs regex_correct |
| `inference_time_mean_s` | |
| `inference_time_sd_s` | |
| `gpu_hours_used` | |

---

## 8. Run-level aggregate (report.csv, one row)

| Field | Notes |
|---|---|
| `run_name`, `timestamp` | |
| `n_prompts`, `n_images`, `n_combos` | |
| `best_combo_regex`, `best_combo_judge` | model+method+prompt with highest accuracy |
| `worst_combo_regex`, `worst_combo_judge` | |
| `most_improved_method` | biggest delta_accuracy_vs_baseline |
| `most_regressed_method` | most negative delta |
| `hardest_class` | lowest per-class F1 across all combos |
| `pct_tier1` | % images where all combos correct |
| `pct_tier2` | % images where all combos wrong |
| `pct_tier3` | % contested |
| `pct_method_regression` | % tier3 images flagged method_regression |
| `total_gpu_hours` | |

---

## 9. Outcome analysis — tier framework

### Tier 1 — Universal successes
**Quantitative:** count + %, per-class distribution, avg inference time, parse rate.
**Qualitative:** gallery of "anchor" images; output language analysis (short+confident vs long+hedging even when correct).

### Tier 2 — Universal failures
**Quantitative:** count + %, per-class rate, confusion pattern (what all combos predicted instead), GT quality flag if image is consistently misclassified by all models.
**Qualitative:** gallery with GT + all predictions; candidate images for GT label review.

### Tier 3 — Contested (mixed outcomes)

Sub-types (multi-label; primary = dominant dimension):

| Sub-type | Condition | Quantitative | Qualitative |
|---|---|---|---|
| `model_split` | LLaVA right, Qwen wrong or vice versa | count, which model wins per class | gallery: LLaVA vs Qwen output text side by side |
| `method_split` | baseline wrong, DeGF or ONLY right | count per method, class distribution | gallery: baseline vs method output; SD image shown |
| `method_regression` | baseline right, DeGF or ONLY wrong | count per method — most important | gallery: what SD image looked like when it broke the answer |
| `prompt_split` | same model+method, label flips across prompts | per-image prompt variance | sensitivity heatmap: rows=images sorted by variance |
| `combo_split` | multiple dimensions vary | count | gallery: full prediction matrix per image |

---

## 10. Failure mode taxonomy

| Type | Detection | Source |
|---|---|---|
| `parse_fail` | regex finds no label | per-record, health check |
| `repetition_loop` | n-gram repeat in raw_text | per-record, health check |
| `refusal` | no classification language | per-record, health check |
| `hedge` | multiple label words present | per-record, health check |
| `length_anomaly` | len > mean + 3×SD for combo | per-record, health check |
| `label_bias` | combo: one class > 40% of predictions | per-combo, health check |
| `hard_image` | difficulty_score ≥ 0.8 | per-image, aggregate |
| `regression` | method wrong, baseline right | per-record, aggregate |
| `uplift` | method right, baseline wrong | per-record, aggregate |
| `prompt_sensitive` | correctness flips across prompts | per-image, aggregate |
| `model_disagreement` | LLaVA and Qwen give different labels | per-image, aggregate |
| `degf_sd_flip` | degf_first_pass_label ≠ parsed_label | per-record, DeGF only |
| `self_inconsistency` | same model+method, different labels across prompts | per-image, health check |

---

## 11. Metrics computed

### Primary (in per-combo summary)
- Accuracy (regex + judge), macro F1, weighted F1, per-class F1
- Parse rate (overall + per class)
- Delta accuracy and F1 vs baseline
- Method regressions and uplifts (count)
- Inference time mean + SD
- GPU hours

### Reliability metrics
- **Cohen's kappa (regex vs judge):** how reliable is regex as a proxy?
  High kappa → can skip judge on future runs
- **Fleiss' kappa (inter-judge):** agreement across DeepSeek / GLM4 / Selene.
  Low kappa → evaluation itself is unstable; must report

### Signal quality metrics
- **Per-class parse rate:** is `on_fire` harder to extract than `aground`?
  Systematic parse bias distorts per-class accuracy
- **Verbosity vs correctness correlation:** Pearson r between output_length_chars
  and regex_correct. Negative correlation = model talks more when confused

### Method-specific metrics
- **DeGF first-pass vs final accuracy:** `degf_first_pass_correct` vs `regex_correct`
  per image. Aggregated: % of DeGF gains came from cases baseline was already wrong
  (true uplift) vs cases baseline was right (regression)
- **SD image plausibility (CLIP proxy):** CLIP cosine similarity between original
  image and SD reference image. High = plausible reference. Low = SD went off-topic
  (explains why DeGF hurts on specific images)
- **Wrong-image-set Jaccard:** Jaccard similarity between set of images baseline got
  wrong and set DeGF/ONLY got wrong. High = methods fail on same images (method not
  changing failure set). Low = method finds different hard cases

### Cross-cutting metrics
- **Prompt ranking stability:** rank prompts by accuracy per (model × method).
  Is ranking consistent across models/methods? Instability → prompt-model interaction effect
- **Per-image prompt variance:** std dev of correctness (0/1) across prompts per image.
  High = prompt-sensitive. Low = robust
- **Confusion asymmetry:** is aground→sunken more common than sunken→aground?
  McNemar-style test on off-diagonal confusion matrix pairs

---

## 12. HTML report sections

1. Run header — run_name, date, cluster, models, methods, N prompts, N images
2. Accuracy overview — table: rows=model×method, cols=prompt variants+avg; color-coded
3. Delta vs baseline — signed accuracy/F1 improvement; color: green=uplift, red=regression
4. Per-class breakdown — F1 per class per combo
5. Confusion matrices — one heatmap per (model×method) averaged across prompts
6. Regex vs judge agreement — Cohen's kappa per combo; divergence cases highlighted
7. Inter-judge agreement — Fleiss' kappa; flags unstable evaluation
8. Parse rate — per combo and per class; low rate = prompt wording problem
9. DeGF first-pass vs final — accuracy before vs after SD guidance; quantifies SD contribution
10. SD plausibility gallery — low-CLIP cases where SD went off-topic (DeGF-only)
11. Prompt sensitivity heatmap — rows=images sorted by variance, cols=prompts, cell=correct/wrong
12. Model disagreement — images where LLaVA and Qwen diverge
13. Outcome tier summary — Tier 1/2/3 counts + galleries for each sub-type
14. Timing + cost — mean±SD per combo; total GPU-hours
