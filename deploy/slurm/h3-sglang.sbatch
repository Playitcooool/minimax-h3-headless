#!/usr/bin/env bash
#SBATCH --job-name=h3-sglang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

variant=${1:-fl2va}
repo_dir=${H3_REPO_DIR:?Export H3_REPO_DIR before sbatch}

cd "${repo_dir}"
export H3_INFERENCE_HOST=0.0.0.0
export H3_PROFILE=${H3_PROFILE:-auto}

echo "node=$(hostname) variant=${variant} profile=${H3_PROFILE}"
exec deploy/start_sglang.sh "${variant}"
