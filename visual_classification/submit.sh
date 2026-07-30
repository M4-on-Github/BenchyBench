#!/bin/bash
# visual_classification — submission orchestrator
#
# Submits 6 inference array jobs (LLaVA × Qwen) × (baseline / degf / only)
# and chains all eval phases via --dependency=afterok.
# Mirrors the pattern from DeGF/CASTOR/submit.sh and QWEN-Maritime/CASTOR/submit.sh.
#
# Usage (from ~/BenchyBench/):
#   bash visual_classification/submit.sh --run-name exp01
#   bash visual_classification/submit.sh --run-name smoke01 --limit 3
#   bash visual_classification/submit.sh --run-name smoke01 --limit 3 --dry-run
#
# Options:
#   --run-name NAME     Required. Unique label for this run.
#   --limit N           Pass --limit N to each inference job (smoke tests)
#   --skip-judge        Skip LLM judge phase; chain aggregate directly after regex
#   --dry-run           Print sbatch commands without submitting
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/BenchyBench_results/visual_classification/<RUN>/logs/*.out

set -euo pipefail

BENCHYBENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BENCHYBENCH_ROOT   # propagated to sbatch jobs so $0 fallback is never used
VC_DIR="$BENCHYBENCH_ROOT/visual_classification"
SLURM_DIR="$VC_DIR/slurm"
DATA_DIR="/data/$USER"

# ── Parse args ────────────────────────────────────────────────────────────────
RUN_NAME=""
LIMIT=""
SKIP_JUDGE=false
DRY_RUN=false

_args=("$@"); _i=0
while [[ $_i -lt ${#_args[@]} ]]; do
    case "${_args[$_i]}" in
        --run-name)       _i=$((_i+1)); RUN_NAME="${_args[$_i]}" ;;
        --run-name=*)     RUN_NAME="${_args[$_i]#--run-name=}" ;;
        --limit)          _i=$((_i+1)); LIMIT="${_args[$_i]}" ;;
        --limit=*)        LIMIT="${_args[$_i]#--limit=}" ;;
        --skip-judge)     SKIP_JUDGE=true ;;
        --dry-run)        DRY_RUN=true ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "ERROR: unknown argument: ${_args[$_i]}" >&2; exit 1 ;;
    esac
    _i=$((_i+1))
done
unset _args _i

if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: --run-name is required" >&2; exit 1
fi

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS_DIR="$VC_DIR/prompts"
N=$(ls "$PROMPTS_DIR"/*.txt 2>/dev/null | wc -l)
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no .txt files in $PROMPTS_DIR" >&2; exit 1
fi
ARRAY_END=$(( N - 1 ))

OUTPUT_ROOT="$DATA_DIR/BenchyBench_results/visual_classification/$RUN_NAME"
LOGS_DIR="$OUTPUT_ROOT/logs"

echo "Run name     : $RUN_NAME"
echo "BenchyBench  : $BENCHYBENCH_ROOT"
echo "Prompts      : $PROMPTS_DIR  ($N prompt(s), array 0-$ARRAY_END)"
echo "Output root  : $OUTPUT_ROOT"
echo "Dry run      : $DRY_RUN"
echo "Skip judge   : $SKIP_JUDGE"
[[ -n "$LIMIT" ]] && echo "Limit        : $LIMIT images"
echo ""

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$LOGS_DIR" "$OUTPUT_ROOT/inference"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
_sbatch() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry] sbatch --parsable $*" >&2
        echo "DRY$(( RANDOM % 99999 ))"
    else
        sbatch --parsable "$@"
    fi
}

EXTRA_FLAGS=""
[[ -n "$LIMIT" ]] && EXTRA_FLAGS="--limit $LIMIT"

# ── Phase 0: inference (6 array jobs) ────────────────────────────────────────
echo "-- Phase 0: inference --"
INFER_IDS=()
for MODEL in llava qwen; do
    for METHOD in baseline degf only; do
        JOB_ID=$(_sbatch \
            --array="0-${ARRAY_END}" \
            --output="$LOGS_DIR/infer_${MODEL}_${METHOD}_%A_%a.out" \
            --error="$LOGS_DIR/infer_${MODEL}_${METHOD}_%A_%a.err" \
            "$SLURM_DIR/infer_${MODEL}.sh" \
            --method "$METHOD" \
            --output-root "$OUTPUT_ROOT" \
            --prompts-dir "$PROMPTS_DIR" \
            $EXTRA_FLAGS)
        echo "  -> ${MODEL}x${METHOD}: job $JOB_ID"
        INFER_IDS+=("$JOB_ID")
    done
done

# Build afterok:id1:id2:... dependency string
DEP_INFER="afterok"
for JID in "${INFER_IDS[@]}"; do
    DEP_INFER="${DEP_INFER}:${JID}"
done

# ── Phase 1: health check ─────────────────────────────────────────────────────
echo ""
echo "-- Phase 1: health_check --"
HEALTH_ID=$(_sbatch \
    --dependency="$DEP_INFER" \
    --output="$LOGS_DIR/health_%j.out" \
    --error="$LOGS_DIR/health_%j.err" \
    "$SLURM_DIR/health_job.sh" \
    --output-root "$OUTPUT_ROOT" \
    --benchybench-root "$BENCHYBENCH_ROOT")
echo "  -> health_check: job $HEALTH_ID"

# ── Phase 2: regex eval ───────────────────────────────────────────────────────
echo ""
echo "-- Phase 2: regex_eval --"
REGEX_ID=$(_sbatch \
    --dependency="afterok:${HEALTH_ID}" \
    --output="$LOGS_DIR/regex_%j.out" \
    --error="$LOGS_DIR/regex_%j.err" \
    "$SLURM_DIR/regex_job.sh" \
    --output-root "$OUTPUT_ROOT" \
    --benchybench-root "$BENCHYBENCH_ROOT")
echo "  -> regex_eval: job $REGEX_ID"

# ── Phase 3: judge ───────────────────────────────────────────────────────────
PREV_ID="$REGEX_ID"
if [[ "$SKIP_JUDGE" == false ]]; then
    echo ""
    echo "-- Phase 3: judge_submit --"
    JUDGE_ID=$(_sbatch \
        --dependency="afterok:${REGEX_ID}" \
        --output="$LOGS_DIR/judge_%j.out" \
        --error="$LOGS_DIR/judge_%j.err" \
        "$SLURM_DIR/judge_job.sh" \
        --output-root "$OUTPUT_ROOT" \
        --benchybench-root "$BENCHYBENCH_ROOT" \
        --run-name "$RUN_NAME")
    echo "  -> judge_submit: job $JUDGE_ID"
    PREV_ID="$JUDGE_ID"
else
    echo ""
    echo "-- Phase 3: judge skipped --"
fi

# ── Phase 4a: outcome analysis ────────────────────────────────────────────────
echo ""
echo "-- Phase 4a: aggregate --phase outcome --"
OUTCOME_ID=$(_sbatch \
    --dependency="afterok:${PREV_ID}" \
    --output="$LOGS_DIR/aggregate_outcome_%j.out" \
    --error="$LOGS_DIR/aggregate_outcome_%j.err" \
    "$SLURM_DIR/aggregate_job.sh" \
    --phase outcome \
    --output-root "$OUTPUT_ROOT" \
    --benchybench-root "$BENCHYBENCH_ROOT")
echo "  -> aggregate outcome: job $OUTCOME_ID"

# ── Phase 4b: HTML report ─────────────────────────────────────────────────────
echo ""
echo "-- Phase 4b: aggregate --phase report --"
REPORT_ID=$(_sbatch \
    --dependency="afterok:${OUTCOME_ID}" \
    --output="$LOGS_DIR/aggregate_report_%j.out" \
    --error="$LOGS_DIR/aggregate_report_%j.err" \
    "$SLURM_DIR/aggregate_job.sh" \
    --phase report \
    --output-root "$OUTPUT_ROOT" \
    --benchybench-root "$BENCHYBENCH_ROOT" \
    --run-name "$RUN_NAME")
echo "  -> aggregate report: job $REPORT_ID"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Run         : $RUN_NAME"
echo "  Inference   : ${INFER_IDS[*]}"
echo "  Health      : $HEALTH_ID"
echo "  Regex       : $REGEX_ID"
[[ "$SKIP_JUDGE" == false ]] && echo "  Judge       : $JUDGE_ID"
echo "  Outcome     : $OUTCOME_ID"
echo "  Report      : $REPORT_ID"
echo "  Output root : $OUTPUT_ROOT"
echo "  Monitor     : squeue -u $USER"
echo "=========================================="
