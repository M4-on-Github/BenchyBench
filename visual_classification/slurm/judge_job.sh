#!/bin/bash
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH -J vc_judge
# --output and --error set by submit.sh

set -e
BENCHYBENCH_ROOT="${BENCHYBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VC_DIR="$BENCHYBENCH_ROOT/visual_classification"
SLURM_DIR="$VC_DIR/slurm"

# Parse args we need for chaining; pass everything through to judge_submit.py
OUTPUT_ROOT=""
RUN_NAME=""
LOGS_DIR=""
_args=("$@"); _i=0
while [[ $_i -lt ${#_args[@]} ]]; do
    case "${_args[$_i]}" in
        --output-root)  _i=$((_i+1)); OUTPUT_ROOT="${_args[$_i]}" ;;
        --output-root=*) OUTPUT_ROOT="${_args[$_i]#--output-root=}" ;;
        --run-name)     _i=$((_i+1)); RUN_NAME="${_args[$_i]}" ;;
        --run-name=*)   RUN_NAME="${_args[$_i]#--run-name=}" ;;
    esac
    _i=$((_i+1))
done
[[ -n "$OUTPUT_ROOT" ]] && LOGS_DIR="$OUTPUT_ROOT/logs"

echo "judge_submit starting $(date) on $(hostname)"

# Capture output so we can parse the agg job ID.
# Use set +e so a non-zero exit doesn't kill the script before we can echo the error.
set +e
JUDGE_OUT=$(python3 "$VC_DIR/judge_submit.py" "$@" 2>&1)
JUDGE_EXIT=$?
set -e
echo "$JUDGE_OUT"

if [[ $JUDGE_EXIT -ne 0 ]]; then
    echo "ERROR: judge_submit.py exited $JUDGE_EXIT — see output above"
    exit $JUDGE_EXIT
fi

# Extract the aggregation job ID printed by judge_submit.py
AGG_JOB_ID=$(echo "$JUDGE_OUT" | grep -oP 'JUDGE_AGG_JOB_ID=\K[0-9]+' || true)

echo "judge_submit done $(date)"

# ── Chain outcome + report after judge panel agg ─────────────────────────────
if [[ -n "$AGG_JOB_ID" && -n "$OUTPUT_ROOT" && -n "$RUN_NAME" ]]; then
    echo "Chaining aggregate phases after judge panel agg job $AGG_JOB_ID ..."

    OUTCOME_ID=$(sbatch \
        -p pleiades --cpus-per-task=4 --mem=16G --time=1:00:00 --parsable \
        --dependency="afterok:${AGG_JOB_ID}" \
        --output="${LOGS_DIR}/aggregate_outcome_%j.out" \
        --error="${LOGS_DIR}/aggregate_outcome_%j.err" \
        "$SLURM_DIR/aggregate_job.sh" \
        --phase outcome \
        --output-root "$OUTPUT_ROOT" \
        --benchybench-root "$BENCHYBENCH_ROOT")
    echo "  -> aggregate outcome: job $OUTCOME_ID"

    REPORT_ID=$(sbatch \
        -p pleiades --cpus-per-task=4 --mem=16G --time=1:00:00 --parsable \
        --dependency="afterok:${OUTCOME_ID}" \
        --output="${LOGS_DIR}/aggregate_report_%j.out" \
        --error="${LOGS_DIR}/aggregate_report_%j.err" \
        "$SLURM_DIR/aggregate_job.sh" \
        --phase report \
        --output-root "$OUTPUT_ROOT" \
        --benchybench-root "$BENCHYBENCH_ROOT" \
        --run-name "$RUN_NAME")
    echo "  -> aggregate report:  job $REPORT_ID"
else
    echo "WARNING: could not parse agg job ID — outcome/report not chained automatically."
    echo "         Run aggregate manually after judge panel completes."
fi
