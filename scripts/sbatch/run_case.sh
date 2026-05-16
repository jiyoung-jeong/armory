#!/usr/bin/env bash
#SBATCH --job-name=armory_sweep
#SBATCH --output=logs/armory_sweep_%j.out
#SBATCH --error=logs/armory_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

CASE_DIR=${1:?Usage: sbatch/run_case.sh <case_dir>}
SCRIPT_DIR="${ARMORY_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/utils.sh"

REPO_ROOT="$(find_repo_root)"
cd "${REPO_ROOT}"
setup_armory_env

mkdir -p logs "${CASE_DIR}/logs"

PORT="$(find_free_port 8000 9000 "${SLURM_JOB_ID:-7}")"
SERVER_NODELIST="${SLURM_JOB_NODELIST:-}"
SERVER_NODE="$(scontrol show hostnames "${SERVER_NODELIST}" | head -1)"
SERVER_CPUS="${ARMORY_SERVER_CPUS:-4}"
CLIENT_CPUS="${ARMORY_CLIENT_CPUS:-4}"
IFS=',' read -ra _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
SERVER_GPU_ID="${_GPU_IDS[0]:-0}"
CLIENT_GPU_ID="${_GPU_IDS[1]:-1}"

echo "======================================"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Case dir: ${CASE_DIR}"
echo "Server node: ${SERVER_NODE}"
echo "Port: ${PORT}"
echo "Server CPUs: ${SERVER_CPUS} / GPU: ${SERVER_GPU_ID}"
echo "Client CPUs: ${CLIENT_CPUS} / GPU: ${CLIENT_GPU_ID}"
echo "======================================"

SERVER_SRUN=(srun --het-group=0 --ntasks=1 --overlap --exact)
if [ -n "${SERVER_CPUS}" ]; then
    SERVER_SRUN+=(--cpus-per-task="${SERVER_CPUS}")
fi
CLIENT_SRUN=(srun --het-group=1 --ntasks=1 --overlap --exact)
if [ -n "${CLIENT_CPUS}" ]; then
    CLIENT_SRUN+=(--cpus-per-task="${CLIENT_CPUS}")
fi

python3 - "${CASE_DIR}" "${SERVER_NODE}" "${PORT}" <<'EOF'
import json
import pathlib
import sys

case_dir = pathlib.Path(sys.argv[1])
host = sys.argv[2]
port = int(sys.argv[3])

server_path = case_dir / "server_args.json"
client_path = case_dir / "client_args.json"
server_args = json.loads(server_path.read_text())
client_args = json.loads(client_path.read_text())
server_args["port"] = port
client_args["host"] = host
client_args["port"] = port
server_path.write_text(json.dumps(server_args, indent=2) + "\n")
client_path.write_text(json.dumps(client_args, indent=2) + "\n")
EOF

"${SERVER_SRUN[@]}" bash -lc "
    set -euo pipefail
    cd '${REPO_ROOT}'
    source '${SCRIPT_DIR}/utils.sh'
    setup_armory_env
    uv run python scripts/serve.py --json-path '${CASE_DIR}/server_args.json'
" >"${CASE_DIR}/logs/server.stdout.log" 2>"${CASE_DIR}/logs/server.stderr.log" &
SERVER_JOB_PID=$!
echo "Server launched with PID ${SERVER_JOB_PID}"
setup_server_monitor "${SERVER_JOB_PID}"

if ! wait_for_http_metadata "${SERVER_NODE}" "${PORT}" 900; then
    uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" --write-result --status failed --error "server metadata endpoint did not become ready"
    exit 1
fi

CLIENT_STATUS=ok
CLIENT_ERROR=""
if ! "${CLIENT_SRUN[@]}" bash -lc "
    set -euo pipefail
    cd '${REPO_ROOT}'
    source '${SCRIPT_DIR}/utils.sh'
    setup_armory_env
    uv run python scripts/run_libero.py --json-path '${CASE_DIR}/client_args.json'
" >"${CASE_DIR}/logs/client.stdout.log" 2>"${CASE_DIR}/logs/client.stderr.log"; then
    CLIENT_STATUS=failed
    CLIENT_ERROR="client exited nonzero"
fi

uv run python scripts/sbatch/collect_results.py \
    --case-dir "${CASE_DIR}" \
    --write-result \
    --status "${CLIENT_STATUS}" \
    --error "${CLIENT_ERROR}"

cleanup
trap - EXIT
wait "${SERVER_JOB_PID}" 2>/dev/null || true

if [ "${CLIENT_STATUS}" != "ok" ]; then
    exit 1
fi
