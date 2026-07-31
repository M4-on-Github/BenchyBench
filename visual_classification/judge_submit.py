#!/usr/bin/env python3
"""
visual_classification — Phase 3: bridge to Eval_CASTOR judge panel

Reads per_record.csv from the regex phase, transforms records to the
{image, text} format the judge panel expects, writes them to
Eval_CASTOR/p5_to_judge/{run_name}.jsonl, and calls
Eval_CASTOR/containers/judge_panel_submit.sh --run {run_name}.

The judge output will land at:
  Eval_CASTOR/results/p5_judge/{run_name}/*_consensus.jsonl

aggregate_report.py --phase outcome reads and merges this back.

Usage:
  python judge_submit.py --output-root /data/$USER/.../run01 \
                         --benchybench-root /path/to/BenchyBench \
                         --run-name visual_classification_run01
  python judge_submit.py ... --dry-run
  python judge_submit.py ... --limit 5
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def load_per_record_csv(per_record_path):
    with per_record_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_judge_records(rows):
    """
    Build minimal records for the judge panel. Each record must have at least:
      image  — the GT join key (e.g. "aground/00017.jpg")
      text   — the raw model output

    We also carry model_tag, method, prompt_stem through for post-merge
    even though the judge ignores extra fields.
    """
    seen = set()
    records = []
    for row in rows:
        image = row.get("image", "")
        text = row.get("raw_text", "")
        model_tag = row.get("model_tag", "")
        method = row.get("method", "")
        prompt_stem = row.get("prompt_stem", "")
        question_id = row.get("question_id", "")

        # Judge key must be unique per (image × model × method × prompt_stem)
        key = (image, model_tag, method, prompt_stem)
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "image":       image,
            "text":        text,
            "model_tag":   model_tag,
            "method":      method,
            "prompt_stem": prompt_stem,
            "question_id": question_id,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchybench-root")
    parser.add_argument("--run-name", required=True,
                        help="Label for this judge run, e.g. 'visual_classification_run01'")
    parser.add_argument("--limit", type=int, help="Pass --limit N to judge (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print paths and command without writing or submitting")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    benchybench_root = (
        Path(args.benchybench_root)
        if args.benchybench_root
        else Path(__file__).resolve().parent.parent
    )
    eval_castor_root = benchybench_root / "Eval_CASTOR"
    judge_submit_script = eval_castor_root / "containers" / "judge_panel_submit.sh"

    per_record_path = output_root / "eval" / "regex" / "per_record.csv"
    p5_dir = eval_castor_root / "p5_to_judge"
    judge_input = p5_dir / f"{args.run_name}.jsonl"

    print(f"Run name        : {args.run_name}")
    print(f"Per-record CSV  : {per_record_path}")
    print(f"Judge input     : {judge_input}")
    print(f"Judge script    : {judge_submit_script}")
    print(f"Dry run         : {args.dry_run}")

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not per_record_path.exists():
        sys.exit(f"ERROR: per_record.csv not found: {per_record_path}\n"
                 f"       Run regex_eval.py first.")
    if not judge_submit_script.exists():
        sys.exit(f"ERROR: judge panel submit script not found: {judge_submit_script}")

    # ── Build judge records ───────────────────────────────────────────────────
    rows = load_per_record_csv(per_record_path)
    print(f"Records loaded  : {len(rows)}")

    judge_records = build_judge_records(rows)
    print(f"Judge records   : {len(judge_records)} (unique image×model×method×prompt)")

    # ── Clear stale judge output so re-runs start clean ──────────────────────
    judge_out_dir = eval_castor_root / "results" / "p5_judge" / args.run_name
    if not args.dry_run and judge_out_dir.exists():
        shutil.rmtree(judge_out_dir)
        print(f"Cleared stale judge output: {judge_out_dir}")

    # ── Write judge input ─────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[dry-run] would write {len(judge_records)} records to: {judge_input}")
    else:
        p5_dir.mkdir(parents=True, exist_ok=True)
        with judge_input.open("w", encoding="utf-8") as f:
            for rec in judge_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote judge input: {judge_input}")

    # ── Call judge_panel_submit.sh, capture output to parse agg job ID ───────
    import re as _re
    cmd = ["bash", str(judge_submit_script), "--run", args.run_name]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd += ["--dry-run"]

    print(f"\nCalling judge panel:")
    print("  " + " ".join(cmd))

    agg_job_id = None
    if args.dry_run:
        print("[dry-run] skipping actual submission")
    else:
        result = subprocess.run(cmd, cwd=str(eval_castor_root),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        print(result.stdout)
        if result.returncode != 0:
            sys.exit(f"ERROR: judge_panel_submit.sh exited {result.returncode}")
        # Parse the aggregation job ID from the last "job=NNNNN" in output
        matches = _re.findall(r'\bjob=(\d+)', result.stdout)
        if matches:
            agg_job_id = matches[-1]

    # ── Print merge info ──────────────────────────────────────────────────────
    expected_consensus = eval_castor_root / "results" / "p5_judge" / args.run_name
    print(f"\nJudge output will appear at: {expected_consensus}/")
    if agg_job_id:
        print(f"JUDGE_AGG_JOB_ID={agg_job_id}")


if __name__ == "__main__":
    main()
