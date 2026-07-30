#!/bin/bash
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH -J vc_regex
# --output and --error set by submit.sh

set -e
BENCHYBENCH_ROOT="${BENCHYBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VC_DIR="$BENCHYBENCH_ROOT/visual_classification"
DATA_DIR="/data/$USER"
SIF="$DATA_DIR/castor.sif"

echo "regex_eval starting $(date) on $(hostname)"

apptainer exec --containall \
    --home $HOME \
    --env USER=$USER \
    --bind "$BENCHYBENCH_ROOT:$BENCHYBENCH_ROOT" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    /opt/conda/bin/python3 "$VC_DIR/regex_eval.py" "$@"

echo "regex_eval done $(date)"
