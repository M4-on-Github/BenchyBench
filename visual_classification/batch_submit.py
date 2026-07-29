#!/usr/bin/env python3
"""
visual_classification — orchestration entry point

Submits 6 inference array jobs (LLaVAx3 + Qwenx3) and chains all eval
phases via SLURM --dependency=afterok. All outputs land under:

  /data/$USER/BenchyBench_results/visual_classification/{run_name}/

Usage:
  python batch_submit.py --run-name exp01
  python batch_submit.py --run-name smoke --limit 3 --skip-judge --dry-run

Options:
  --run-name NAME       Unique label for this run (required)
  --prompts-dir PATH    Path to prompts/ dir [default: visual_classification/prompts]
  --max-concurrent N    Max active jobs at once [default: 4]
  --limit N             Pass --limit N to each inference job (smoke tests)
  --skip-judge          Chain aggregate directly after regex; skip judge phase
  --force               Re-run even if output files already exist
  --dry-run             Print sbatch commands without submitting
  --output-root PATH    Override default output root
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _resolve_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _count_active_jobs(user: str) -> int:
    result = subprocess.run(
        ["squeue", "-u", user, "-h"],
        capture_output=True, text=True
    )
    return len([l for l in result.stdout.strip().splitlines() if l.strip()])


def sbatch(args: list[str], dry_run: bool) -> str:
    """Submit an sbatch job. Returns bare job ID string."""
    cmd = ["sbatch", "--parsable"] + args
    print("  " + " ".join(cmd))
    if dry_run:
        # Return a synthetic job ID so downstream chaining still prints correctly
        return f"DRY{abs(hash(tuple(args))) % 100000:05d}"
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split(";")[0]  # --parsable may include cluster name
    return job_id


def wait_for_slot(user: str, max_concurrent: int, dry_run: bool) -> None:
    while not dry_run and _count_active_jobs(user) >= max_concurrent:
        print(f"  [throttle] {_count_active_jobs(user)} active jobs >= {max_concurrent}, sleeping 60s …")
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--prompts-dir")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()

    benchybench_root = _resolve_root()
    vc_dir = benchybench_root / "visual_classification"
    slurm_dir = vc_dir / "slurm"

    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else vc_dir / "prompts"
    prompt_files = sorted(glob.glob(str(prompts_dir / "*.txt")))
    if not prompt_files:
        sys.exit(f"ERROR: no .txt files found in {prompts_dir}")
    n_prompts = len(prompt_files)

    user = os.environ.get("USER", os.environ.get("USERNAME", ""))
    if not user:
        sys.exit("ERROR: $USER not set")

    output_root = Path(args.output_root) if args.output_root else (
        Path(f"/data/{user}/BenchyBench_results/visual_classification/{args.run_name}")
    )
    logs_dir = output_root / "logs"

    print(f"Run name     : {args.run_name}")
    print(f"Benchybench  : {benchybench_root}")
    print(f"Prompts dir  : {prompts_dir}  ({n_prompts} prompt(s))")
    print(f"Output root  : {output_root}")
    print(f"Logs         : {logs_dir}")
    print(f"Max conc.    : {args.max_concurrent}")
    print(f"Dry run      : {args.dry_run}")
    print(f"Skip judge   : {args.skip_judge}")
    if args.limit:
        print(f"Limit        : {args.limit} images")
    print()

    # Create logs dir before any sbatch call so SLURM can open --output files
    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (output_root / "inference").mkdir(parents=True, exist_ok=True)

    combos = [
        ("llava", "baseline"),
        ("llava", "degf"),
        ("llava", "only"),
        ("qwen",  "baseline"),
        ("qwen",  "degf"),
        ("qwen",  "only"),
    ]

    # Phase 0: Submit inference array jobs
    print("-- Phase 0: inference --")
    inference_job_ids = []
    for model, method in combos:
        script = slurm_dir / (f"infer_{model}.sh")
        log_stem = f"infer_{model}_{method}"

        wait_for_slot(user, args.max_concurrent, args.dry_run)

        extra = []
        if args.limit:
            extra += ["--limit", str(args.limit)]
        if args.force:
            extra += ["--force"]

        job_id = sbatch([
            f"--array=0-{n_prompts - 1}",
            f"--output={logs_dir}/{log_stem}_%A_%a.out",
            f"--error={logs_dir}/{log_stem}_%A_%a.err",
            str(script),
            "--method", method,
            "--output-root", str(output_root),
            "--prompts-dir", str(prompts_dir),
        ] + extra, args.dry_run)

        print(f"  -> {model}x{method}: job {job_id}")
        inference_job_ids.append(job_id)

    dep_all_infer = "afterok:" + ":".join(inference_job_ids)

    # -- Phase 1: Health check -------------------------------------------------
    print("\n-- Phase 1: health_check -------------------------------------------")
    health_id = sbatch([
        f"--dependency={dep_all_infer}",
        f"--output={logs_dir}/health_%j.out",
        f"--error={logs_dir}/health_%j.err",
        str(slurm_dir / "health_job.sh"),
        "--output-root", str(output_root),
        "--benchybench-root", str(benchybench_root),
    ], args.dry_run)
    print(f"  -> health_check: job {health_id}")

    # -- Phase 2: Regex eval ---------------------------------------------------
    print("\n-- Phase 2: regex_eval ---------------------------------------------")
    regex_id = sbatch([
        f"--dependency=afterok:{health_id}",
        f"--output={logs_dir}/regex_%j.out",
        f"--error={logs_dir}/regex_%j.err",
        str(slurm_dir / "regex_job.sh"),
        "--output-root", str(output_root),
        "--benchybench-root", str(benchybench_root),
    ], args.dry_run)
    print(f"  -> regex_eval: job {regex_id}")

    # -- Phase 3: Judge or skip ------------------------------------------------
    prev_id = regex_id
    if not args.skip_judge:
        print("\n-- Phase 3: judge_submit -------------------------------------------")
        judge_id = sbatch([
            f"--dependency=afterok:{regex_id}",
            f"--output={logs_dir}/judge_%j.out",
            f"--error={logs_dir}/judge_%j.err",
            str(slurm_dir / "judge_job.sh"),
            "--output-root", str(output_root),
            "--benchybench-root", str(benchybench_root),
            "--run-name", args.run_name,
        ], args.dry_run)
        print(f"  -> judge_submit: job {judge_id}")
        prev_id = judge_id
    else:
        print("\n-- Phase 3: judge skipped (--skip-judge) ---------------------------")

    # -- Phase 4: Aggregate — outcome analysis ---------------------------------
    print("\n-- Phase 4a: aggregate_report --phase outcome ----------------------")
    outcome_id = sbatch([
        f"--dependency=afterok:{prev_id}",
        f"--output={logs_dir}/aggregate_outcome_%j.out",
        f"--error={logs_dir}/aggregate_outcome_%j.err",
        str(slurm_dir / "aggregate_job.sh"),
        "--phase", "outcome",
        "--output-root", str(output_root),
        "--benchybench-root", str(benchybench_root),
    ], args.dry_run)
    print(f"  -> aggregate outcome: job {outcome_id}")

    # -- Phase 4b: Aggregate — HTML report -------------------------------------
    print("\n-- Phase 4b: aggregate_report --phase report -----------------------")
    report_id = sbatch([
        f"--dependency=afterok:{outcome_id}",
        f"--output={logs_dir}/aggregate_report_%j.out",
        f"--error={logs_dir}/aggregate_report_%j.err",
        str(slurm_dir / "aggregate_job.sh"),
        "--phase", "report",
        "--output-root", str(output_root),
        "--benchybench-root", str(benchybench_root),
        "--run-name", args.run_name,
    ], args.dry_run)
    print(f"  -> aggregate report: job {report_id}")

    # -- Summary ---------------------------------------------------------------
    print()
    print("----------------------------------------------------------")
    print(f"  All jobs submitted for run: {args.run_name}")
    print(f"  Inference jobs : {', '.join(inference_job_ids)}")
    print(f"  Health check   : {health_id}")
    print(f"  Regex eval     : {regex_id}")
    if not args.skip_judge:
        print(f"  Judge          : {judge_id}")
    print(f"  Outcome        : {outcome_id}")
    print(f"  Report         : {report_id}")
    print(f"  Output root    : {output_root}")
    print(f"  Report will be : {output_root}/report/report.html")
    print("----------------------------------------------------------")

    # Write submission manifest
    manifest = {
        "run_name": args.run_name,
        "output_root": str(output_root),
        "prompts_dir": str(prompts_dir),
        "n_prompts": n_prompts,
        "combos": [f"{m}x{mt}" for m, mt in combos],
        "inference_job_ids": inference_job_ids,
        "health_job_id": health_id,
        "regex_job_id": regex_id,
        "outcome_job_id": outcome_id,
        "report_job_id": report_id,
        "skip_judge": args.skip_judge,
        "limit": args.limit,
        "dry_run": args.dry_run,
    }
    if not args.skip_judge:
        manifest["judge_job_id"] = judge_id
    if not args.dry_run:
        manifest_path = output_root / "logs" / "submission_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n  Manifest written to: {manifest_path}")
    else:
        print(f"\n  [dry-run] manifest (not written):")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
