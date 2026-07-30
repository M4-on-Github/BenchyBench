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

echo "judge_submit starting $(date) on $(hostname)"

# judge_submit.py only uses stdlib (json, csv, subprocess) and calls sbatch —
# run it on the host directly so sbatch is available (not inside apptainer).
python3 "$VC_DIR/judge_submit.py" "$@"

echo "judge_submit done $(date)"
