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

RUN_DIR=${1:?Usage: collect_results.sh <run_dir>}

SCRIPT_DIR="${ARMORY_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/utils.sh"

cd "${SCRIPT_DIR}/../.."
setup_armory_env
mkdir -p logs

uv run python scripts/sbatch/collect_results.py --run-dir "${RUN_DIR}"
