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

PYTHON VERSION
--------------
Unlike the other visual_classification scripts, this one is NOT run inside
castor.sif — slurm/judge_job.sh invokes it with the node's bare `python3`,
which is Python 3.6. Therefore:

  * no dataclasses (3.7+)
  * no `subprocess.run(..., text=True)` (3.7+) — use universal_newlines
  * no walrus operator (3.8+)

Both mistakes have been made here before. Keep this module 3.6-clean.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def load_per_record_csv(per_record_path):
    """Read the regex phase's per_record.csv into a list of dicts."""
    with per_record_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_judge_records(rows):
    """Deduplicate per-record rows into the minimal shape the judge accepts.

    Compatibility facade over JudgeSubmission.build_records().
    """
    return JudgeSubmission.build_records(rows)


class JudgeSubmission:
    """Bridges one visual_classification run into the Eval_CASTOR judge panel.

    Groups what were previously free functions plus a tangle of derived paths.
    Every path is computed once from (output_root, benchybench_root, run_name)
    and exposed as a property, so the layout is stated in one place rather than
    rebuilt at each use site.

    Typical use:

        sub = JudgeSubmission(output_root, benchybench_root, run_name)
        sub.validate()
        records = sub.build_records(sub.load_rows())
        sub.write_input(records)
        agg_job_id = sub.submit()
    """

    #: Judge identity. A record is unique per image AND the combination that
    #: produced it — keying on image alone would collapse every method into one
    #: verdict and silently discard all but the last.
    RECORD_KEY_FIELDS = ("image", "model_tag", "method", "prompt_stem")

    def __init__(self, output_root, benchybench_root, run_name):
        self.output_root = Path(output_root)
        self.benchybench_root = Path(benchybench_root)
        self.run_name = run_name

    # ── Derived paths ────────────────────────────────────────────────────────

    @property
    def eval_castor_root(self):
        return self.benchybench_root / "Eval_CASTOR"

    @property
    def per_record_path(self):
        """Input: the regex phase's output."""
        return self.output_root / "eval" / "regex" / "per_record.csv"

    @property
    def judge_script(self):
        return self.eval_castor_root / "containers" / "judge_panel_submit.sh"

    @property
    def judge_input_path(self):
        """Staging file the judge panel reads."""
        return self.eval_castor_root / "p5_to_judge" / (self.run_name + ".jsonl")

    @property
    def judge_output_dir(self):
        """Where consensus files will appear once the panel finishes."""
        return self.eval_castor_root / "results" / "p5_judge" / self.run_name

    # ── Steps ────────────────────────────────────────────────────────────────

    def validate(self):
        """Exit with a diagnostic if a required input is missing."""
        if not self.per_record_path.exists():
            sys.exit("ERROR: per_record.csv not found: %s\n"
                     "       Run regex_eval.py first." % self.per_record_path)
        if not self.judge_script.exists():
            sys.exit("ERROR: judge panel submit script not found: %s"
                     % self.judge_script)

    def load_rows(self):
        return load_per_record_csv(self.per_record_path)

    @classmethod
    def build_records(cls, rows):
        """Reduce per-record rows to the judge's input shape, deduplicated.

        The judge needs only `image` and `text`; model_tag, method and
        prompt_stem are carried through so the verdict can be joined back onto
        the right row afterwards. The judge ignores the extra fields.

        Deduplication is on the full composite key, not `image`, because the
        same image appears once per combination and each deserves its own
        verdict.
        """
        seen = set()
        records = []
        for row in rows:
            key = tuple(row.get(f, "") for f in cls.RECORD_KEY_FIELDS)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "image":       row.get("image", ""),
                "text":        row.get("raw_text", ""),
                "model_tag":   row.get("model_tag", ""),
                "method":      row.get("method", ""),
                "prompt_stem": row.get("prompt_stem", ""),
                "question_id": row.get("question_id", ""),
            })
        return records

    def clear_stale_output(self):
        """Remove a previous run's consensus files.

        Without this a re-run merges old verdicts with new ones, and because
        the merge is keyed rather than positional the result looks plausible.
        """
        if self.judge_output_dir.exists():
            shutil.rmtree(str(self.judge_output_dir))
            print("Cleared stale judge output: %s" % self.judge_output_dir)

    def write_input(self, records):
        self.judge_input_path.parent.mkdir(parents=True, exist_ok=True)
        with self.judge_input_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print("Wrote judge input: %s" % self.judge_input_path)

    def build_command(self, limit=None, dry_run=False):
        cmd = ["bash", str(self.judge_script), "--run", self.run_name]
        if limit:
            cmd += ["--limit", str(limit)]
        if dry_run:
            cmd += ["--dry-run"]
        return cmd

    def submit(self, limit=None):
        """Run the judge panel submit script. Returns its aggregation job ID.

        Output is streamed before the return code is checked, so a failure is
        diagnosable from the log rather than silently swallowed.
        """
        cmd = self.build_command(limit=limit)
        # universal_newlines, not text= — see the PYTHON VERSION note above.
        result = subprocess.run(cmd, cwd=str(self.eval_castor_root),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        print(result.stdout)
        if result.returncode != 0:
            sys.exit("ERROR: judge_panel_submit.sh exited %d" % result.returncode)
        return self.parse_agg_job_id(result.stdout)

    @staticmethod
    def parse_agg_job_id(output):
        """Extract the aggregation job ID from the submit script's output.

        The LAST `job=NNNNN` is the aggregation job: the script prints one line
        per judge model first, then the aggregation job that depends on them.
        judge_job.sh needs this ID to chain the outcome and report phases.
        """
        matches = re.findall(r"\bjob=(\d+)", output)
        return matches[-1] if matches else None


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

    benchybench_root = (
        Path(args.benchybench_root)
        if args.benchybench_root
        else Path(__file__).resolve().parent.parent
    )
    sub = JudgeSubmission(args.output_root, benchybench_root, args.run_name)

    print("Run name        : %s" % sub.run_name)
    print("Per-record CSV  : %s" % sub.per_record_path)
    print("Judge input     : %s" % sub.judge_input_path)
    print("Judge script    : %s" % sub.judge_script)
    print("Dry run         : %s" % args.dry_run)

    sub.validate()

    rows = sub.load_rows()
    print("Records loaded  : %d" % len(rows))

    judge_records = sub.build_records(rows)
    print("Judge records   : %d (unique image×model×method×prompt)" % len(judge_records))

    if not args.dry_run:
        sub.clear_stale_output()

    if args.dry_run:
        print("\n[dry-run] would write %d records to: %s"
              % (len(judge_records), sub.judge_input_path))
    else:
        sub.write_input(judge_records)

    print("\nCalling judge panel:")
    print("  " + " ".join(sub.build_command(limit=args.limit, dry_run=args.dry_run)))

    agg_job_id = None
    if args.dry_run:
        print("[dry-run] skipping actual submission")
    else:
        agg_job_id = sub.submit(limit=args.limit)

    print("\nJudge output will appear at: %s/" % sub.judge_output_dir)
    if agg_job_id:
        # judge_job.sh greps for this exact token to chain outcome + report.
        print("JUDGE_AGG_JOB_ID=%s" % agg_job_id)


if __name__ == "__main__":
    main()
