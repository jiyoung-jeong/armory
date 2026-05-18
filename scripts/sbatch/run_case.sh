#!/usr/bin/env bash
#SBATCH --job-name=armory_sweep
#SBATCH --output=logs/armory_sweep_%j.out
#SBATCH --error=logs/armory_sweep_%j.err

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
SERVER_HOST="$(hostname)"

echo "======================================"
echo "Job ID:      ${SLURM_JOB_ID:-unknown}"
echo "Case dir:    ${CASE_DIR}"
echo "Server host: ${SERVER_HOST}"
echo "Port:        ${PORT}"
echo "======================================"

python3 - "${CASE_DIR}" "${PORT}" "${SERVER_HOST}" <<'EOF'
import json, pathlib, sys
case_dir, port, server_host = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for name, override in (
    ("server_args.json", {"port": port}),
    ("client_args.json", {"host": server_host, "port": port}),
):
    p = case_dir / name
    data = json.loads(p.read_text())
    data.update(override)
    p.write_text(json.dumps(data, indent=2) + "\n")
EOF

# Server: the batch body is already pinned to het-group 0's L40S node.
uv run python scripts/serve.py --json-path "${CASE_DIR}/server_args.json" \
    >"${CASE_DIR}/logs/server.stdout.log" 2>"${CASE_DIR}/logs/server.stderr.log" &
SERVER_PID=$!
echo "Server PID ${SERVER_PID}"
setup_server_monitor "${SERVER_PID}"

if ! wait_for_http_metadata "${SERVER_HOST}" "${PORT}" 900; then
    uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" --write-result --status failed --error "server metadata endpoint did not become ready"
    exit 1
fi

# Client: dispatch to het-group 1 (V100 node).
CLIENT_STATUS=ok
CLIENT_ERROR=""
if ! srun --het-group=1 \
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
