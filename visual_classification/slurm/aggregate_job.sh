#!/bin/bash
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -J vc_aggregate
# --output and --error set by submit.sh

set -e
BENCHYBENCH_ROOT="${BENCHYBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VC_DIR="$BENCHYBENCH_ROOT/visual_classification"
DATA_DIR="/data/$USER"
SIF="$DATA_DIR/castor.sif"

echo "aggregate_report starting $(date) on $(hostname)"

apptainer exec --containall \
    --env USER=$USER \
    --env HOME=$HOME \
    --bind "$BENCHYBENCH_ROOT:$BENCHYBENCH_ROOT" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    /opt/conda/bin/python3 "$VC_DIR/aggregate_report.py" "$@"

echo "aggregate_report done $(date)"
