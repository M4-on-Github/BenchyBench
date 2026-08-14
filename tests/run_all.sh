#!/bin/bash
# Run every BenchyBench test suite.
#
#     bash tests/run_all.sh
#
# All suites run locally: no cluster, no GPU, no model weights, no network.
# They cover path resolution and the visual_classification pipeline logic.
#
# What they CANNOT cover: apptainer bind behaviour, SLURM scheduling, and
# anything requiring model weights. A --dry-run on the cluster remains the
# check for those.

set -u
BB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BB_ROOT"

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

FAILED=0
run() {
    local name="$1"; shift
    printf "\n\033[1m── %s ──\033[0m\n" "$name"
    if "$@"; then
        return 0
    fi
    FAILED=$((FAILED + 1))
    printf "\033[31mFAILED: %s\033[0m\n" "$name"
}

run "path resolution (bash)"      bash    tests/test_paths.sh
run "health_check (python)"       $PYTHON tests/test_health_check.py
run "regex_eval (python)"         $PYTHON tests/test_regex_eval.py
run "judge_submit (python)"       $PYTHON tests/test_judge_submit.py
run "aggregate_report (python)"   $PYTHON tests/test_aggregate_report.py
run "prepare_dataset (python)"    $PYTHON tests/test_prepare_dataset.py
run "shared.metrics (python)"     $PYTHON tests/test_shared_metrics.py
run "run_config (python)"        $PYTHON tests/test_run_config.py
run "diffusion noise (python)"   $PYTHON tests/test_diffusion_noise.py
run "documentation (python)"     $PYTHON tests/test_documentation.py

printf "\n════════════════════════════════════\n"
if [[ "$FAILED" -eq 0 ]]; then
    printf "\033[32mall suites passed\033[0m\n"
else
    printf "\033[31m%d suite(s) failed\033[0m\n" "$FAILED"
fi
printf "════════════════════════════════════\n"
exit "$FAILED"
