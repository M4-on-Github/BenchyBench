#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# visual_classification — Qwen3-VL-8B inference SLURM array job body
#
# One task per prompt file. Task ID = prompt index.
# Handles all three methods via --method {baseline|degf|only} flag
# (Qwen's run_inference.py accepts --method natively; no mode flag routing needed).
#
# Do NOT call directly — use batch_submit.py which sets --output/--error
# to $OUTPUT_ROOT/logs/ and passes --method, --output-root, --prompts-dir.
#
# Interactive debug (smoke):
#   srun -p pleiades --time=2:00:00 --cpus-per-task=4 --gpus-per-task=2 \
#        --mem=48G --constraint=RTX6000ADA --exclude=pleiades-1-3 --pty bash
#   cd ~/benchybench
#   SLURM_ARRAY_TASK_ID=0 bash visual_classification/slurm/infer_qwen.sh \
#       --method baseline --output-root /data/$USER/BenchyBench_results/visual_classification/smoke \
#       --prompts-dir visual_classification/prompts --limit 3
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -J vc_infer_qwen
#SBATCH --constraint=RTX6000ADA
#SBATCH --exclude=pleiades-1-3
# NOTE: --output and --error are set by batch_submit.py to $OUTPUT_ROOT/logs/

set -e

BENCHYBENCH_ROOT="${BENCHYBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VC_DIR="$BENCHYBENCH_ROOT/visual_classification"

JOB_START=$SECONDS

echo "=========================================="
echo " Job ID   : $SLURM_JOB_ID"
echo " Array    : task $SLURM_ARRAY_TASK_ID  (job $SLURM_ARRAY_JOB_ID)"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Args     : $@"
echo " User     : $USER"
echo " BenchRoot: $BENCHYBENCH_ROOT"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── Parse args ───────────────────────────────────────────────────────────────
METHOD=""
OUTPUT_ROOT=""
PROMPTS_DIR="$VC_DIR/prompts"
LIMIT=""
PASSTHROUGH=()
_args=("$@"); _i=0
while [[ $_i -lt ${#_args[@]} ]]; do
    case "${_args[$_i]}" in
        --method)       _i=$((_i+1)); METHOD="${_args[$_i]}" ;;
        --method=*)     METHOD="${_args[$_i]#--method=}" ;;
        --output-root)  _i=$((_i+1)); OUTPUT_ROOT="${_args[$_i]}" ;;
        --output-root=*) OUTPUT_ROOT="${_args[$_i]#--output-root=}" ;;
        --prompts-dir)  _i=$((_i+1)); PROMPTS_DIR="${_args[$_i]}" ;;
        --prompts-dir=*) PROMPTS_DIR="${_args[$_i]#--prompts-dir=}" ;;
        --limit)        _i=$((_i+1)); LIMIT="${_args[$_i]}" ;;
        --limit=*)      LIMIT="${_args[$_i]#--limit=}" ;;
        *)              PASSTHROUGH+=("${_args[$_i]}") ;;
    esac
    _i=$((_i+1))
done
unset _args _i

if [[ -z "$METHOD" ]]; then
    echo "ERROR: --method is required (baseline|degf|only)" >&2; exit 1
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "ERROR: --output-root is required" >&2; exit 1
fi

# ── Fixed repo and SIF (Qwen always uses its own repo) ───────────────────────
DATA_DIR="/data/$USER"
REPO="$BENCHYBENCH_ROOT/QWEN-Maritime"
SIF="$DATA_DIR/castor_qwen.sif"
DEGF_REPO="$BENCHYBENCH_ROOT/DeGF"

if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found" >&2; exit 1
fi
echo "[$(date)] Repo: $REPO  Container: $SIF  Method: $METHOD"

# ── Env and cache dirs ────────────────────────────────────────────────────────
export HF_HOME="$DATA_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$DATA_DIR/.cache/huggingface"
export TORCH_HOME="$DATA_DIR/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ── Image dir ─────────────────────────────────────────────────────────────────
IMAGE_DIR="$BENCHYBENCH_ROOT/shipwreck_wiki_images/sorted_images"

# ── Apptainer base ────────────────────────────────────────────────────────────
# Qwen: uses --home $HOME (not --env HOME=), binds DEGF_REPO for shared assets,
# and binds shipwreck_wiki_images via BENCHYBENCH_ROOT.
APPTAINER_BASE="apptainer exec --containall --nv \
    --pwd $REPO \
    --home $HOME \
    --env USER=$USER \
    --env HF_HOME=$HF_HOME \
    --env TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
    --env TORCH_HOME=$TORCH_HOME \
    --bind /tmp:/tmp \
    --bind $REPO:$REPO \
    --bind $DATA_DIR:$DATA_DIR \
    --bind $BENCHYBENCH_ROOT/shipwreck_wiki_images:$BENCHYBENCH_ROOT/shipwreck_wiki_images \
    --bind $DEGF_REPO:$DEGF_REPO \
    --bind $VC_DIR:$VC_DIR"
PYTHON=/opt/conda/bin/python3

# ── Resolve prompt file from array task ID ────────────────────────────────────
PROMPT_FILES=( "$PROMPTS_DIR"/*.txt )
PROMPT_FILE="${PROMPT_FILES[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "$PROMPT_FILE" || ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: no prompt file for task $SLURM_ARRAY_TASK_ID in $PROMPTS_DIR" >&2; exit 1
fi
PROMPT_STEM=$(basename "$PROMPT_FILE" .txt)

# ── Output paths ──────────────────────────────────────────────────────────────
INFER_DIR="$OUTPUT_ROOT/inference"
SD_DIR="$INFER_DIR/sd_images/qwen_degf_${PROMPT_STEM}"
mkdir -p "$INFER_DIR"
[[ "$METHOD" == "degf" ]] && mkdir -p "$SD_DIR"

QUESTIONS_FILE="$INFER_DIR/questions_qwen_${METHOD}_${PROMPT_STEM}.jsonl"
ANSWERS_FILE="$INFER_DIR/answers_qwen_${METHOD}_${PROMPT_STEM}.jsonl"
FIRSTPASS_FILE="$INFER_DIR/firstpass_qwen_${METHOD}_${PROMPT_STEM}.jsonl"

echo "=========================================="
echo " Method      : $METHOD"
echo " Prompt      : $PROMPT_STEM"
echo " Questions   : $QUESTIONS_FILE"
echo " Answers     : $ANSWERS_FILE"
echo "=========================================="

# ── Prepare dataset ───────────────────────────────────────────────────────────
$APPTAINER_BASE "$SIF" $PYTHON "$REPO/CASTOR/prepare_dataset.py" \
    --image-dir   "$IMAGE_DIR" \
    --output      "$QUESTIONS_FILE" \
    --prompt-file "$PROMPT_FILE"

# Apply --limit by truncating the questions file (run_inference.py has no --limit flag)
if [[ -n "$LIMIT" ]]; then
    tmp="${QUESTIONS_FILE}.tmp"
    head -n "$LIMIT" "$QUESTIONS_FILE" > "$tmp" && mv "$tmp" "$QUESTIONS_FILE"
    echo "[limit] Truncated questions file to $LIMIT entries"
fi

# ── Run inference ─────────────────────────────────────────────────────────────
DEGF_FLAGS=""
if [[ "$METHOD" == "degf" ]]; then
    DEGF_FLAGS="--sd-dir $SD_DIR --firstpass-file $FIRSTPASS_FILE"
fi

time $APPTAINER_BASE "$SIF" $PYTHON "$REPO/CASTOR/run_inference.py" \
    "${PASSTHROUGH[@]}" \
    --question-file "$QUESTIONS_FILE" \
    --answers-file  "$ANSWERS_FILE" \
    --run-name      "qwen_${METHOD}_${PROMPT_STEM}" \
    --method        "$METHOD" \
    $DEGF_FLAGS

# Write sidecar metadata for regex_eval.py to read prompt_stem and model/method
cat > "$INFER_DIR/meta_qwen_${METHOD}_${PROMPT_STEM}.json" <<EOF
{
  "model": "qwen",
  "method": "$METHOD",
  "prompt_stem": "$PROMPT_STEM",
  "prompt_file": "$PROMPT_FILE",
  "answers_file": "$ANSWERS_FILE",
  "questions_file": "$QUESTIONS_FILE",
  "job_id": "$SLURM_JOB_ID",
  "array_job_id": "$SLURM_ARRAY_JOB_ID",
  "array_task_id": "$SLURM_ARRAY_TASK_ID"
}
EOF

ELAPSED=$(( SECONDS - JOB_START ))
echo "=========================================="
echo " Finished    : $(date)"
echo " Wall time   : $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo " Answers     : $ANSWERS_FILE"
echo "=========================================="
