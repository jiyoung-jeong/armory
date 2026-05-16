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
IFS=',' read -ra _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
SERVER_GPU="${_GPU_IDS[0]:-0}"
CLIENT_GPU="${_GPU_IDS[1]:-${SERVER_GPU}}"

echo "======================================"
echo "Job ID:    ${SLURM_JOB_ID:-unknown}"
echo "Case dir:  ${CASE_DIR}"
echo "Port:      ${PORT}"
echo "Server GPU: ${SERVER_GPU}    Client GPU: ${CLIENT_GPU}"
echo "======================================"

python3 - "${CASE_DIR}" "${PORT}" <<'EOF'
import json, pathlib, sys
case_dir, port = pathlib.Path(sys.argv[1]), int(sys.argv[2])
for name, override in (
    ("server_args.json", {"port": port}),
    ("client_args.json", {"host": "127.0.0.1", "port": port}),
):
    p = case_dir / name
    data = json.loads(p.read_text())
    data.update(override)
    p.write_text(json.dumps(data, indent=2) + "\n")
EOF

CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
    uv run python scripts/serve.py --json-path "${CASE_DIR}/server_args.json" \
    >"${CASE_DIR}/logs/server.stdout.log" 2>"${CASE_DIR}/logs/server.stderr.log" &
SERVER_PID=$!
echo "Server PID ${SERVER_PID}"
setup_server_monitor "${SERVER_PID}"

if ! wait_for_http_metadata 127.0.0.1 "${PORT}" 900; then
    uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" --write-result --status failed --error "server metadata endpoint did not become ready"
    exit 1
fi

CLIENT_STATUS=ok
CLIENT_ERROR=""
if ! CUDA_VISIBLE_DEVICES="${CLIENT_GPU}" \
    uv run python scripts/run_libero.py --json-path "${CASE_DIR}/client_args.json" \
    >"${CASE_DIR}/logs/client.stdout.log" 2>"${CASE_DIR}/logs/client.stderr.log"; then
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
wait "${SERVER_PID}" 2>/dev/null || true

if [ "${CLIENT_STATUS}" != "ok" ]; then
    exit 1
fi
