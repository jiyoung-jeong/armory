#!/usr/bin/env bash
# Profile a single-robot run-libero run with py-spy, producing a speedscope file.
#
# Usage:
#   scripts/profile_libero.sh [extra run-libero args...]
#
# Environment overrides:
#   CLIENT_CONFIG  experiment config (default: configs/client/libero/profile_1robot.json)
#   SERVER_CONFIG  serve.py config used only if no server is already running
#                  (default: configs/server/mock_max_batch.json — mock policy, no GPU)
#   HOST / PORT    where to look for (or start) the policy server (default: 127.0.0.1:8080)
#   RUN_DIR        output dir for episode data + logs (default: data/profile_runs/<timestamp>)
#   RATE           py-spy sampling rate in Hz (default: 200)
#   NATIVE=1       include native (C/mujoco) frames in the profile
#   NO_PYSPY=1     run the client without py-spy (baseline timing / smoke test)
#
# The profile lands in $RUN_DIR/profile.speedscope.json — open it at
# https://www.speedscope.app or with `npx speedscope <file>`.
#
# Note: on macOS py-spy must run as root; this script prepends sudo there.
set -euo pipefail
cd "$(dirname "$0")/.."

CLIENT_CONFIG=${CLIENT_CONFIG:-configs/client/libero/profile_1robot.json}
SERVER_CONFIG=${SERVER_CONFIG:-configs/server/mock_max_batch.json}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
RUN_DIR=${RUN_DIR:-data/profile_runs/$(date +%Y%m%d_%H%M%S)}
RATE=${RATE:-200}
PROFILE_OUT="$RUN_DIR/profile.speedscope.json"
mkdir -p "$RUN_DIR"

PYTHON=$(uv run python -c 'import sys; print(sys.executable)')

# macOS: robosuite's import chain loads the glfw Python package even though
# offscreen rendering goes through CGL; point it at the Homebrew dylib if the
# environment doesn't already resolve it.
if [[ "$(uname)" == "Darwin" && -z "${PYGLFW_LIBRARY:-}" && -f /opt/homebrew/lib/libglfw.3.dylib ]]; then
    export PYGLFW_LIBRARY=/opt/homebrew/lib/libglfw.3.dylib
fi

# Reuse an already-running server (e.g. a real policy server on the cluster);
# otherwise start a mock-policy server and tear it down on exit.
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if curl -sf "http://$HOST:$PORT/metadata" >/dev/null 2>&1; then
    echo "Using existing server at $HOST:$PORT"
else
    echo "No server at $HOST:$PORT — starting mock server ($SERVER_CONFIG)"
    "$PYTHON" scripts/serve.py --json-path "$SERVER_CONFIG" >"$RUN_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        curl -sf "http://$HOST:$PORT/metadata" >/dev/null 2>&1 && break
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server died during startup; see $RUN_DIR/server.log" >&2
            exit 1
        fi
        sleep 1
    done
    curl -sf "http://$HOST:$PORT/metadata" >/dev/null 2>&1 || {
        echo "Server did not become ready; see $RUN_DIR/server.log" >&2
        exit 1
    }
fi

CLIENT_CMD=(
    uv run run-libero
    --experiment-config "$CLIENT_CONFIG"
    --host "$HOST" --port "$PORT"
    --output-dir "$RUN_DIR/client"
    --overwrite
    --debug
    "$@"
)

if [[ "${NO_PYSPY:-0}" == "1" ]]; then
    "${CLIENT_CMD[@]}"
else
    PYSPY_ARGS=(record --format speedscope --output "$PROFILE_OUT" --rate "$RATE" --idle --subprocesses)
    [[ "${NATIVE:-0}" == "1" ]] && PYSPY_ARGS+=(--native)
    SUDO=()
    if [[ "$(uname)" == "Darwin" && "$EUID" -ne 0 ]]; then
        echo "macOS: running py-spy under sudo (required to read process memory)"
        SUDO=(sudo)
        # sudo resets the environment; re-inject what the sim needs.
        [[ -n "${PYGLFW_LIBRARY:-}" ]] && SUDO+=(env "PYGLFW_LIBRARY=$PYGLFW_LIBRARY")
    fi
    "${SUDO[@]}" uvx py-spy "${PYSPY_ARGS[@]}" -- "${CLIENT_CMD[@]}"
    echo
    echo "Profile written to $PROFILE_OUT"
    echo "Open it at https://www.speedscope.app or with: npx speedscope $PROFILE_OUT"
fi
