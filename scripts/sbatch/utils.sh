#!/usr/bin/env bash
# Shared helpers for Slurm experiment scripts.

find_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/../.." && pwd
}

setup_armory_env() {
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
    export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
    export ABSL_FLAGS_VERBOSITY="${ABSL_FLAGS_VERBOSITY:-0}"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

    if [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
    fi
}

find_free_port() {
    local lo=${1:-8000}
    local hi=${2:-9000}
    local seed=${3:-$RANDOM}

    python3 - <<EOF
import random
import socket
import sys

random.seed($seed)
candidates = list(range($lo, $hi + 1))
random.shuffle(candidates)

for port in candidates:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("", port))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(port)
    sys.exit(0)

print("ERROR: no free port found in range [$lo, $hi]", file=sys.stderr)
sys.exit(1)
EOF
}

setup_server_monitor() {
    SERVER_JOB_PID=$1

    cleanup() {
        echo "Cleaning up..."
        if [ -n "${MONITOR_PID:-}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
            kill "${MONITOR_PID}" 2>/dev/null || true
        fi
        if [ -n "${SERVER_JOB_PID:-}" ] && kill -0 "${SERVER_JOB_PID}" 2>/dev/null; then
            echo "Stopping server process group ${SERVER_JOB_PID}"
            kill "${SERVER_JOB_PID}" 2>/dev/null || true
            sleep 5
            if kill -0 "${SERVER_JOB_PID}" 2>/dev/null; then
                echo "Force killing server process group ${SERVER_JOB_PID}"
                kill -9 "${SERVER_JOB_PID}" 2>/dev/null || true
            fi
        fi
        echo "Cleanup complete"
    }
    trap cleanup EXIT INT TERM

    (
        while sleep 5; do
            if ! kill -0 "${SERVER_JOB_PID}" 2>/dev/null; then
                echo "ERROR: server process ${SERVER_JOB_PID} exited unexpectedly"
                kill -TERM $$
                break
            fi
        done
    ) &
    MONITOR_PID=$!
}

wait_for_http_metadata() {
    local host=$1
    local port=$2
    local max_wait=${3:-600}
    local elapsed=0
    local url="http://${host}:${port}/metadata"

    echo "Waiting for ${url}..."
    while true; do
        if python3 - "$url" <<'EOF'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        if 200 <= response.status < 500:
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
EOF
        then
            echo "Server metadata endpoint is ready."
            return 0
        fi

        sleep 5
        elapsed=$((elapsed + 5))
        if [ "${elapsed}" -ge "${max_wait}" ]; then
            echo "ERROR: server did not become ready within ${max_wait}s"
            return 1
        fi
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "Still waiting for server metadata (${elapsed}s)..."
        fi
    done
}
