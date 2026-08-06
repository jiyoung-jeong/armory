#!/usr/bin/env bash

setup_armory_env() {
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
    export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
    export ABSL_FLAGS_VERBOSITY="${ABSL_FLAGS_VERBOSITY:-0}"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

    if [ -f "${HOME}/.bashrc" ]; then
        set +u
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
        set -u
    fi
}

find_free_port() {
    uv run python - "$1" "$2" "$3" <<'EOF'
import random, socket, sys

lo, hi, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
random.seed(seed)
candidates = list(range(lo, hi + 1))
random.shuffle(candidates)
for port in candidates:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    sys.exit(0)
print(f"ERROR: no free port found in range [{lo}, {hi}]", file=sys.stderr)
sys.exit(1)
EOF
}

wait_for_server() {
    local pid=$1 host=$2 port=$3 max_wait=${4:-600} elapsed=0
    echo "Waiting for http://${host}:${port}/metadata..."
    while true; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: server process ${pid} exited before becoming ready"
            return 1
        fi
        if curl -sf -m 5 "http://${host}:${port}/metadata" >/dev/null; then
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
