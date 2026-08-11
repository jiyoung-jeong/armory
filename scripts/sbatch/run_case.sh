#!/usr/bin/env bash
#SBATCH --job-name=armory_sweep
#SBATCH --output=logs/armory_sweep_%j.out
#SBATCH --error=logs/armory_sweep_%j.err

set -euo pipefail

CASE_DIR=${1:?Usage: run_case.sh <case_dir>}
SCRIPT_DIR="${ARMORY_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MAX_RETRIES=${ARMORY_MAX_RETRIES:-0}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}

if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ && "${RESTART_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "ARMORY_MAX_RETRIES and SLURM_RESTART_COUNT must be non-negative integers" >&2
    exit 2
fi

SERVER_PID=""
FAILURE_REASON="case wrapper exited unexpectedly"
RETRY_ALLOWED=1

cleanup_server() {
    if [ -n "${SERVER_PID}" ]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

handle_exit() {
    rc=$?
    trap - EXIT
    set +e
    cleanup_server

    if [ "${rc}" -ne 0 ] && [ "${RETRY_ALLOWED}" -eq 1 ] \
            && [ "${RESTART_COUNT}" -lt "${MAX_RETRIES}" ] \
            && [ -n "${SLURM_JOB_ID:-}" ]; then
        mkdir -p "${CASE_DIR}/logs"
        echo "$(date --iso-8601=seconds) attempt $((RESTART_COUNT + 1)) failed: ${FAILURE_REASON}; requeueing job ${SLURM_JOB_ID}" \
            | tee -a "${CASE_DIR}/logs/requeue.log"
        if scontrol requeue "${SLURM_JOB_ID}"; then
            exit 0
        fi
        echo "Failed to requeue Slurm job ${SLURM_JOB_ID}" \
            | tee -a "${CASE_DIR}/logs/requeue.log" >&2
    fi

    if [ "${rc}" -ne 0 ] && [ ! -s "${CASE_DIR}/result.json" ]; then
        uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" \
            --write-result --status failed \
            --error "${FAILURE_REASON} (exit code ${rc})" || true
    fi
    exit "${rc}"
}

handle_termination() {
    RETRY_ALLOWED=0
    FAILURE_REASON="case wrapper was terminated"
    exit 143
}

trap handle_exit EXIT
trap handle_termination INT TERM

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/utils.sh"

cd "${SCRIPT_DIR}/../.."
setup_armory_env
mkdir -p logs "${CASE_DIR}/logs"

if [ "${RESTART_COUNT}" -gt 0 ]; then
    previous_attempt=$((RESTART_COUNT - 1))
    for process in server client; do
        for stream in stdout stderr; do
            log="${CASE_DIR}/logs/${process}.${stream}.log"
            if [ -f "${log}" ]; then
                mv "${log}" \
                    "${CASE_DIR}/logs/${process}.attempt-${previous_attempt}.${stream}.log"
            fi
        done
    done
    rm -f "${CASE_DIR}/result.json"
    rm -rf "${CASE_DIR}/output" "${CASE_DIR}/server"
fi

if [ "${RESTART_COUNT}" -gt "${MAX_RETRIES}" ]; then
    FAILURE_REASON="retry limit exceeded (${RESTART_COUNT} restarts, maximum ${MAX_RETRIES})"
    exit 1
fi

PORT="$(find_free_port 8000 9000 "${SLURM_JOB_ID:-7}")"
SERVER_HOST="$(hostname)"

echo "======================================"
echo "Job ID:      ${SLURM_JOB_ID:-unknown}"
echo "Case dir:    ${CASE_DIR}"
echo "Server host: ${SERVER_HOST}"
echo "Port:        ${PORT}"
echo "Attempt:     $((RESTART_COUNT + 1)) / $((MAX_RETRIES + 1))"
echo "======================================"

uv run python - "${CASE_DIR}" "${PORT}" "${SERVER_HOST}" <<'EOF'
import json, pathlib, sys
case_dir, port, host = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
for name, override in (("server_args.json", {"port": port}), ("client_args.json", {"host": host, "port": port})):
    path = case_dir / name
    path.write_text(json.dumps({**json.loads(path.read_text()), **override}, indent=2) + "\n")
EOF

srun --het-group=0 uv run python -m scripts.serve --json-path "${CASE_DIR}/server_args.json" \
    >"${CASE_DIR}/logs/server.stdout.log" 2>"${CASE_DIR}/logs/server.stderr.log" &
SERVER_PID=$!
echo "Server PID ${SERVER_PID}"

if ! wait_for_server "${SERVER_PID}" "${SERVER_HOST}" "${PORT}" 900; then
    FAILURE_REASON="server did not become ready"
    uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" \
        --write-result --status failed --error "${FAILURE_REASON}"
    exit 1
fi

STATUS=ok
ERROR=""
if ! srun --het-group=1 \
        uv run python -m scripts.run --json-path "${CASE_DIR}/client_args.json" \
        >"${CASE_DIR}/logs/client.stdout.log" 2>"${CASE_DIR}/logs/client.stderr.log"; then
    STATUS=failed
    ERROR="client exited nonzero"
    FAILURE_REASON="${ERROR}"
fi

cleanup_server
SERVER_PID=""
mkdir -p "${CASE_DIR}/output"
if [ -d "${CASE_DIR}/server" ]; then
    rm -rf "${CASE_DIR}/output/server"
    mv "${CASE_DIR}/server" "${CASE_DIR}/output/server"
fi

uv run python scripts/sbatch/collect_results.py --case-dir "${CASE_DIR}" \
    --write-result --status "${STATUS}" --error "${ERROR}"

[ "${STATUS}" = ok ]
