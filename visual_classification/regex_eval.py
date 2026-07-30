#!/usr/bin/env python3
"""
visual_classification — Phase 2: regex-based label extraction and metrics

Reads all inference JSONLs from {output_root}/inference/, joins against the
human GT CSV, extracts predicted labels with normalize_state(), and writes:

  eval/regex/per_record.csv   — one row per (image × model × method × prompt)
  eval/regex/summary.csv      — one row per (model × method × prompt_stem)

Also populates degf_first_pass_* fields for DeGF records by joining baseline
runs on (model, image, prompt_stem) — no changes to inference code needed.

Usage:
  python regex_eval.py --output-root /data/$USER/.../run01 \
                       --benchybench-root /path/to/BenchyBench
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, StatisticsError


def _setup_eval_imports(benchybench_root):
    eval_path = str(benchybench_root / "Eval_CASTOR")
    if eval_path not in sys.path:
        sys.path.insert(0, eval_path)


def load_gt(benchybench_root):
    """Return {image: state} from human_gt.csv. Image key = 'class/filename.jpg'."""
    gt_path = benchybench_root / "Eval_CASTOR" / "human_ground_truth_label" / "human_gt.csv"
    gt = {}
    with gt_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("image") and row.get("state"):
                gt[row["image"]] = row["state"]
    return gt


def load_inference_records(output_root):
    """Load all answers_*.jsonl files. Attach source filename stem to each record."""
    infer_dir = output_root / "inference"
    records = []
    for path in sorted(infer_dir.glob("answers_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["_source_stem"] = path.stem  # e.g. "answers_llava_baseline_12345"
                    records.append(rec)
                except json.JSONDecodeError:
                    pass
    return records


def load_meta_map(output_root):
    """Return {answers_stem: meta_dict} from sidecar meta_*.json files."""
    infer_dir = output_root / "inference"
    meta_map = {}
    for path in sorted(infer_dir.glob("meta_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stem = Path(data.get("answers_file", "")).stem
            if stem:
                meta_map[stem] = data
        except (json.JSONDecodeError, KeyError):
            pass
    return meta_map


def extract_combo_fields(rec: dict, meta_map: dict):
    """Return (model_tag, method, prompt_stem) for a record."""
    source_stem = rec.get("_source_stem", "")
    meta = meta_map.get(source_stem, {})

    model_tag = meta.get("model") or rec.get("model_tag") or rec.get("model_id", "unknown")
    method = meta.get("method") or rec.get("method", "unknown")
    prompt_stem = meta.get("prompt_stem", "")
    if not prompt_stem:
        # Fallback: parse from run_name field (e.g. "llava_baseline_promptv3" → "promptv3")
        run_name = rec.get("run_name", "")
        parts = run_name.split("_")
        prompt_stem = parts[-1] if len(parts) >= 3 else run_name or "unknown"

    return model_tag, method, prompt_stem


def build_per_records(records, gt, normalize_state,
                      meta_map: dict):
    """Build one output row per record with all per-record fields."""
    rows = []
    for rec in records:
        image = rec.get("image", "")
        raw_text = rec.get("text", "")
        model_tag, method, prompt_stem = extract_combo_fields(rec, meta_map)

        gt_label = gt.get(image)
        parsed_label_raw = normalize_state(raw_text)
        parse_success = parsed_label_raw != "UNPARSEABLE"
        parsed_label = parsed_label_raw if parse_success else None
        regex_correct = (parsed_label == gt_label) if (parse_success and gt_label) else False

        timing = rec.get("timing", {}) or {}
        inference_time_s = timing.get("infer_s")
        total_time_s = timing.get("total_s")
        degf_desc_s = timing.get("desc_s")
        degf_sd_s = timing.get("sd_s")

        degf_fields = rec.get("degf", {}) or {}
        only_fields = rec.get("only", {}) or {}

        row = {
            # Identity
            "image": image,
            "model_tag": model_tag,
            "method": method,
            "prompt_stem": prompt_stem,
            "question_id": rec.get("question_id"),
            # Text
            "raw_text": raw_text,
            "raw_text_len": len(raw_text),
            "prompt": rec.get("prompt", ""),
            # Label extraction
            "parsed_label": parsed_label,
            "parse_success": parse_success,
            "gt_label": gt_label,
            "regex_correct": regex_correct,
            # Timing
            "inference_time_s": inference_time_s,
            "total_time_s": total_time_s,
            "degf_desc_s": degf_desc_s,
            "degf_sd_s": degf_sd_s,
            # DeGF-specific fields (from inference record)
            "degf_js_mean": degf_fields.get("js_mean"),
            "degf_contrastive_steps": degf_fields.get("contrastive_steps"),
            "degf_sd_image_path": degf_fields.get("sd_image_path") or rec.get("sd_image_path"),
            # DeGF first-pass — populated later by join against baseline
            "degf_first_pass_text": None,
            "degf_first_pass_label": None,
            "degf_first_pass_correct": None,
            "degf_sd_flip": None,
            # ONLY-specific fields
            "only_tvd_mean": only_fields.get("tvd_mean"),
            "only_contrastive_steps": only_fields.get("contrastive_steps"),
            # Health flags — populated below
            "repetition_detected": None,
            "hedge_detected": None,
            "refusal_detected": None,
            "length_anomaly": None,
            # Aggregate fields — populated by aggregate_report.py
            "difficulty_score": None,
            "consensus": None,
            "failure_type": None,
            "primary_failure_type": None,
            "tier": None,
            # Judge fields — merged by aggregate_report.py after judge runs
            "judge_verdict": None,
            "judge_score": None,
            "judge_state_correct": None,
        }
        rows.append(row)
    return rows


def populate_health_flags(rows):
    """Add health flags inline (reuses logic from health_check.py)."""
    import re

    CLASS_PATTERNS = {
        "aground":  re.compile(r"\b(aground|grounded|beached)\b", re.IGNORECASE),
        "capsized": re.compile(r"\b(capsized|capsizing|overturned)\b", re.IGNORECASE),
        "on_fire":  re.compile(r"\b(on fire|on_fire|fire|burning|ablaze|aflame)\b", re.IGNORECASE),
        "sunken":   re.compile(r"\b(sunken|sunk|submerged|underwater|sinking)\b", re.IGNORECASE),
    }
    REFUSAL_KEYWORDS = ["cannot", "can't", "unable to determine", "unclear",
                        "not sure", "impossible to tell", "no image", "no photo"]

    lengths = [r["raw_text_len"] for r in rows if r["raw_text_len"]]
    g_mean = mean(lengths) if lengths else 0
    g_std = stdev(lengths) if len(lengths) > 1 else 0

    def _repetition(text, ngram=5, threshold=3):
        words = text.lower().split()
        if len(words) < ngram * threshold:
            return False
        seen = {}
        for i in range(len(words) - ngram + 1):
            g = " ".join(words[i:i+ngram])
            seen[g] = seen.get(g, 0) + 1
            if seen[g] >= threshold:
                return True
        return False

    for row in rows:
        text = row["raw_text"]
        matched_classes = [cls for cls, pat in CLASS_PATTERNS.items() if pat.search(text)]
        row["repetition_detected"] = _repetition(text)
        row["hedge_detected"] = len(matched_classes) > 1
        row["refusal_detected"] = any(k in text.lower() for k in REFUSAL_KEYWORDS)
        row["length_anomaly"] = row["raw_text_len"] > g_mean + 3 * g_std if g_std else False

    return rows


def populate_degf_first_pass(rows):
    """
    For each DeGF record, join baseline records on (model_tag, image, prompt_stem)
    and populate degf_first_pass_* fields.

    "First pass" for DeGF = what the same model without SD guidance predicted
    on the same image with the same prompt. This is exactly the baseline record.
    """
    # Build lookup: (model_tag, image, prompt_stem) → baseline row
    baseline_lookup = {}
    for row in rows:
        if row["method"] == "baseline":
            key = (row["model_tag"], row["image"], row["prompt_stem"])
            baseline_lookup[key] = row

    for row in rows:
        if row["method"] != "degf":
            continue
        key = (row["model_tag"], row["image"], row["prompt_stem"])
        baseline = baseline_lookup.get(key)
        if baseline is None:
            continue
        row["degf_first_pass_text"] = baseline["raw_text"]
        row["degf_first_pass_label"] = baseline["parsed_label"]
        row["degf_first_pass_correct"] = baseline["regex_correct"]
        # SD flip = baseline and degf predicted different labels
        row["degf_sd_flip"] = (
            baseline["parsed_label"] != row["parsed_label"]
            if (baseline["parsed_label"] is not None and row["parsed_label"] is not None)
            else None
        )

    return rows


def build_combo_summary(rows):
    """One summary row per (model_tag, method, prompt_stem)."""
    from math import sqrt

    combos = defaultdict(list)
    for row in rows:
        key = (row["model_tag"], row["method"], row["prompt_stem"])
        combos[key].append(row)

    # Build baseline accuracy map for delta computation
    baseline_acc = {}
    for (model, method, prompt), recs in combos.items():
        if method == "baseline":
            parsed = [r for r in recs if r["parse_success"] and r["gt_label"]]
            if parsed:
                baseline_acc[(model, prompt)] = sum(1 for r in parsed if r["regex_correct"]) / len(parsed)

    VALID_STATES = ["aground", "capsized", "on_fire", "sunken"]

    summary_rows = []
    for (model, method, prompt), recs in sorted(combos.items()):
        n = len(recs)
        n_parsed = sum(1 for r in recs if r["parse_success"])
        n_with_gt = sum(1 for r in recs if r["gt_label"])
        n_correct = sum(1 for r in recs if r["regex_correct"])

        accuracy = n_correct / n_with_gt if n_with_gt else None
        parse_rate = n_parsed / n if n else None

        # Per-class metrics
        per_class_acc = {}
        per_class_f1_data = {}  # for macro F1
        for cls in VALID_STATES:
            cls_recs = [r for r in recs if r["gt_label"] == cls]
            if cls_recs:
                per_class_acc[cls] = sum(1 for r in cls_recs if r["regex_correct"]) / len(cls_recs)
                per_class_f1_data[cls] = {
                    "tp": sum(1 for r in recs if r["gt_label"] == cls and r["parsed_label"] == cls),
                    "fp": sum(1 for r in recs if r["gt_label"] != cls and r["parsed_label"] == cls),
                    "fn": sum(1 for r in recs if r["gt_label"] == cls and r["parsed_label"] != cls),
                }

        # Macro F1
        f1_scores = []
        for cls, d in per_class_f1_data.items():
            prec = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0
            rec_ = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0
            f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0
            f1_scores.append(f1)
        macro_f1 = mean(f1_scores) if f1_scores else None

        # Timing
        times = [r["inference_time_s"] for r in recs if r["inference_time_s"] is not None]
        time_mean = mean(times) if times else None
        time_sd = stdev(times) if len(times) > 1 else None
        total_gpu_h = sum(times) / 3600 if times else None  # assumes all tasks serial

        # Delta vs baseline
        base_acc = baseline_acc.get((model, prompt))
        delta_accuracy = (accuracy - base_acc) if (accuracy is not None and base_acc is not None) else None

        # Wrong-set Jaccard vs baseline (compare which images were wrong)
        wrong_here = {r["image"] for r in recs if not r["regex_correct"] and r["gt_label"]}
        baseline_recs = combos.get(("baseline", prompt), [])  # note: (model, baseline, prompt)
        # Recheck with proper key
        wrong_baseline = {
            r["image"] for k, k_recs in combos.items()
            if k[0] == model and k[1] == "baseline" and k[2] == prompt
            for r in k_recs if not r["regex_correct"] and r["gt_label"]
        }
        union_ = wrong_here | wrong_baseline
        jaccard = len(wrong_here & wrong_baseline) / len(union_) if union_ else None

        # Verbosity vs correctness correlation (Pearson r)
        pairs = [(r["raw_text_len"], int(r["regex_correct"])) for r in recs if r["gt_label"]]
        pearson_r = None
        if len(pairs) > 2:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            mx, my = mean(xs), mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            dx = sqrt(sum((x - mx) ** 2 for x in xs))
            dy = sqrt(sum((y - my) ** 2 for y in ys))
            if dx > 0 and dy > 0:
                pearson_r = round(num / (dx * dy), 4)

        summary_rows.append({
            "model_tag": model,
            "method": method,
            "prompt_stem": prompt,
            "n_records": n,
            "n_parsed": n_parsed,
            "n_with_gt": n_with_gt,
            "parse_rate": round(parse_rate, 4) if parse_rate is not None else None,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
            "acc_aground": round(per_class_acc.get("aground"), 4) if per_class_acc.get("aground") is not None else None,
            "acc_capsized": round(per_class_acc.get("capsized"), 4) if per_class_acc.get("capsized") is not None else None,
            "acc_on_fire": round(per_class_acc.get("on_fire"), 4) if per_class_acc.get("on_fire") is not None else None,
            "acc_sunken": round(per_class_acc.get("sunken"), 4) if per_class_acc.get("sunken") is not None else None,
            "delta_accuracy": round(delta_accuracy, 4) if delta_accuracy is not None else None,
            "wrong_set_jaccard_vs_baseline": round(jaccard, 4) if jaccard is not None else None,
            "inference_time_mean_s": round(time_mean, 2) if time_mean is not None else None,
            "inference_time_sd_s": round(time_sd, 2) if time_sd is not None else None,
            "gpu_hours_used": round(total_gpu_h, 4) if total_gpu_h is not None else None,
            "verbosity_correct_pearson_r": pearson_r,
            # Judge fields — null until aggregate_report.py merges judge output
            "judge_accuracy": None,
            "judge_macro_f1": None,
            "regex_judge_kappa": None,
            "fleiss_kappa": None,
        })

    return summary_rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchybench-root")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    benchybench_root = Path(args.benchybench_root) if args.benchybench_root else Path(__file__).resolve().parent.parent
    _setup_eval_imports(benchybench_root)

    from shared.metrics import normalize_state  # noqa: E402

    print(f"Output root     : {output_root}")
    print(f"Benchybench root: {benchybench_root}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    gt = load_gt(benchybench_root)
    print(f"GT records      : {len(gt)}")

    records = load_inference_records(output_root)
    print(f"Inference records: {len(records)}")
    if not records:
        sys.exit(f"ERROR: no inference records found in {output_root}/inference/")

    meta_map = load_meta_map(output_root)
    print(f"Meta sidecars   : {len(meta_map)}")

    # ── Per-record processing ─────────────────────────────────────────────────
    rows = build_per_records(records, gt, normalize_state, meta_map)
    rows = populate_health_flags(rows)
    rows = populate_degf_first_pass(rows)

    # ── Write per-record CSV ──────────────────────────────────────────────────
    per_record_path = output_root / "eval" / "regex" / "per_record.csv"
    write_csv(per_record_path, rows)
    print(f"per_record.csv  : {per_record_path}  ({len(rows)} rows)")

    # ── Per-combo summary ─────────────────────────────────────────────────────
    summary_rows = build_combo_summary(rows)
    summary_path = output_root / "eval" / "regex" / "summary.csv"
    write_csv(summary_path, summary_rows)
    print(f"summary.csv     : {summary_path}  ({len(summary_rows)} rows)")

    # Print quick accuracy table
    print("\nAccuracy summary:")
    print(f"  {'model':<20} {'method':<10} {'prompt':<30} {'acc':>6} {'macro_f1':>9} {'parse_rate':>11}")
    for row in summary_rows:
        acc = f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "   N/A"
        f1 = f"{row['macro_f1']:.3f}" if row["macro_f1"] is not None else "      N/A"
        pr = f"{row['parse_rate']:.3f}" if row["parse_rate"] is not None else "         N/A"
        print(f"  {row['model_tag']:<20} {row['method']:<10} {row['prompt_stem']:<30} {acc:>6} {f1:>9} {pr:>11}")


if __name__ == "__main__":
    main()
