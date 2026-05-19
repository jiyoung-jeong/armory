#!/usr/bin/env bash
#SBATCH --job-name=armory_collect
#SBATCH --output=logs/armory_collect_%j.out
#SBATCH --error=logs/armory_collect_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

set -euo pipefail

OUTPUT_DIR=${1:?Usage: sbatch/collect_results.sh <output_dir> <stamp>}
STAMP=${2:?Usage: sbatch/collect_results.sh <output_dir> <stamp>}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/utils.sh"

REPO_ROOT="$(find_repo_root)"
cd "${REPO_ROOT}"
setup_armory_env
mkdir -p logs

echo "======================================"
echo "Collector job: ${SLURM_JOB_ID:-manual}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Stamp: ${STAMP}"
echo "======================================"

uv run python scripts/sbatch/collect_results.py \
    --output-dir "${OUTPUT_DIR}" \
    --stamp "${STAMP}"
