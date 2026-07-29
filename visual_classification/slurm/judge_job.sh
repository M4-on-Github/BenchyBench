#!/bin/bash
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH -J vc_judge
# --output and --error set by batch_submit.py

set -e
BENCHYBENCH_ROOT="${BENCHYBENCH_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VC_DIR="$BENCHYBENCH_ROOT/visual_classification"
echo "judge_submit starting $(date) on $(hostname)"
python3 "$VC_DIR/judge_submit.py" "$@"
echo "judge_submit done $(date)"
