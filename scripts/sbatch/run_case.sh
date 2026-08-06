#!/usr/bin/env bash
#SBATCH --job-name=armory_sweep
#SBATCH --output=logs/armory_sweep_%j.out
#SBATCH --error=logs/armory_sweep_%j.err

set -euo pipefail

CASE_DIR=${1:?Usage: run_case.sh <case_dir>}
SCRIPT_DIR="${ARMORY_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/utils.sh"

cd "${SCRIPT_DIR}/../.."
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

uv run python - "${CASE_DIR}" "${PORT}" "${SERVER_HOST}" <<'EOF'
import json, pathlib, sys
case_dir, port, host = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for name, override in (("server_args.json", {"port": port}), ("client_args.json", {"host": host, "port": port})):
    path = case_dir / name
    path.write_text(json.dumps({**json.loads(path.read_text()), **override}, indent=2) + "\n")
EOF

uv run python -m scripts.serve --json-path "${CASE_DIR}/server_args.json" \
    >"${CASE_DIR}/logs/server.stdout.log" 2>"${CASE_DIR}/logs/server.stderr.log" &
SERVER_PID=$!
echo "Server PID ${SERVER_PID}"
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

if ! wait_for_server "${SERVER_PID}" "${SERVER_HOST}" "${PORT}" 900; then
    uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" \
        --write-result --status failed --error "server did not become ready"
    exit 1
fi

STATUS=ok
ERROR=""
if ! srun --het-group=1 \
        uv run python -m scripts.run --json-path "${CASE_DIR}/client_args.json" \
        >"${CASE_DIR}/logs/client.stdout.log" 2>"${CASE_DIR}/logs/client.stderr.log"; then
    STATUS=failed
    ERROR="client exited nonzero"
fi

kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
mkdir -p "${CASE_DIR}/output"
if [ -d "${CASE_DIR}/server" ]; then
    rm -rf "${CASE_DIR}/output/server"
    mv "${CASE_DIR}/server" "${CASE_DIR}/output/server"
fi

uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" \
    --write-result --status "${STATUS}" --error "${ERROR}"

[ "${STATUS}" = ok ]
