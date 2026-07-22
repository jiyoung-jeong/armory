"""Run scripts/run.py (single-robot client) on Modal, optionally against a server.

    # mock agent + mock env, no server (one CPU container)
    uv run modal run scripts/modal/run.py

    # single robot vs a real policy in LIBERO sim (server GPU + libero client GPU)
    uv run modal run scripts/modal/run.py --json-path client.json --server sim

    # ...vs a lightweight mock policy server (for scheduling tests)
    uv run modal run scripts/modal/run.py --json-path client.json --server mock

The client image is picked from the run.py Args `env` (1=LIBERO -> GPU, 2=MOCK ->
CPU). With --server, a server is spawned in its own container and bridged to the
client by a modal.forward TCP tunnel. Outputs land on the
`armory-experiment-artifacts` volume and download to --output-dir.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

import modal
from scripts.modal.images import (
    CHECKPOINT_VOLUME_PATH,
    REMOTE_ROOT,
    cpu_mock_image,
    gpu_libero_client_image,
    gpu_server_image,
)
from scripts.modal.utils import ARTIFACTS_VOLUME_NAME

REGION = "us-east"
PORT = 8080
TIMEOUT = 2 * 60 * 60
CLIENT_OUT = REMOTE_ROOT / "run_out"
REMOTE_ARTIFACTS = pathlib.Path("/artifacts")

app = modal.App("armory-run")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)


# --------------------------------------------------------------------------
# On-container bodies (thin wrappers below fix the image per case)
# --------------------------------------------------------------------------
def _serve(server_json: str, stamp: str, urls, shutdown) -> None:  # noqa: ANN001
    """Start scripts/serve.py, forward its port, hold until the client signals done."""
    args_path = pathlib.Path("/tmp/server_args.json")
    args_path.write_text(server_json)
    # Run as a module (not `python scripts/serve.py`) so cwd (/app) leads sys.path
    # instead of scripts/, letting the top-level src `utils`/`logging_config`
    # resolve rather than the shadowing scripts/utils.py.
    proc = subprocess.Popen(
        [sys.executable, "-m", "scripts.serve", "--json-path", str(args_path)],
        cwd=str(REMOTE_ROOT),
    )
    try:
        with modal.forward(PORT, unencrypted=True) as tunnel:
            urls[stamp] = tunnel.tcp_socket
            while stamp not in shutdown:
                if proc.poll() is not None:
                    urls[stamp] = ("", 0)  # poison so the entrypoint stops waiting
                    return
                time.sleep(2)
    finally:
        proc.terminate()


def _run_client(client_json: str, stamp: str, shutdown) -> None:  # noqa: ANN001
    """Run scripts/run.py to completion, then ship its output dir to the volume."""
    args_path = pathlib.Path("/tmp/run_args.json")
    args_path.write_text(client_json)
    try:
        subprocess.run(
            [sys.executable, "-m", "scripts.run", "--json-path", str(args_path)],
            cwd=str(REMOTE_ROOT),
            check=True,
        )
    finally:
        if shutdown is not None:
            shutdown[stamp] = True  # release the server
        if CLIENT_OUT.exists():
            shutil.copytree(CLIENT_OUT, REMOTE_ARTIFACTS / stamp, dirs_exist_ok=True)
            artifacts_volume.commit()


@app.function(
    image=gpu_server_image,
    gpu="L40S",
    region=REGION,
    volumes={CHECKPOINT_VOLUME_PATH: checkpoint_volume},
    timeout=TIMEOUT,
)
def serve_sim(server_json: str, stamp: str, urls, shutdown) -> None:  # noqa: ANN001
    _serve(server_json, stamp, urls, shutdown)


@app.function(image=cpu_mock_image, region=REGION, timeout=TIMEOUT)
def serve_mock(server_json: str, stamp: str, urls, shutdown) -> None:  # noqa: ANN001
    _serve(server_json, stamp, urls, shutdown)


@app.function(
    image=gpu_libero_client_image,
    gpu="T4",
    region=REGION,
    volumes={str(REMOTE_ARTIFACTS): artifacts_volume},
    timeout=TIMEOUT,
)
def client_libero(client_json: str, stamp: str, shutdown=None) -> None:  # noqa: ANN001
    _run_client(client_json, stamp, shutdown)


@app.function(
    image=cpu_mock_image,
    region=REGION,
    volumes={str(REMOTE_ARTIFACTS): artifacts_volume},
    timeout=TIMEOUT,
)
def client_mock(client_json: str, stamp: str, shutdown=None) -> None:  # noqa: ANN001
    _run_client(client_json, stamp, shutdown)


# --------------------------------------------------------------------------
# Local orchestration
# --------------------------------------------------------------------------
def _server_config() -> dict:
    """Minimal scripts/serve.py Args (see serve.py). Omitting `policy` -> Default;
    a mock policy is selected by the caller by adding a `policy` block."""
    return {"model": "pi05", "env": "libero", "max_batch_size": 1, "port": PORT}


def _await_server(urls, stamp: str) -> tuple[str, int]:  # noqa: ANN001
    """Block until the server publishes its tunnel and /metadata answers."""
    deadline = time.time() + 15 * 60
    while stamp not in urls:
        if time.time() > deadline:
            raise RuntimeError("server never published its tunnel")
        time.sleep(2)
    host, port = urls[stamp]
    if not host:
        raise RuntimeError("server failed before forwarding its port")
    ready = time.time() + 10 * 60
    while True:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/metadata", timeout=5):
                return host, port
        except Exception as exc:  # noqa: BLE001
            if time.time() > ready:
                raise RuntimeError(f"server /metadata never came up: {exc!r}") from exc
            time.sleep(5)


@app.local_entrypoint()
def main(json_path: str = "", server: str = "none", output_dir: str = "modal_run_out") -> None:
    if server not in {"none", "mock", "sim"}:
        raise SystemExit("--server must be 'none', 'mock', or 'sim'.")

    client_cfg = json.loads(pathlib.Path(json_path).read_text()) if json_path else {}
    env = int(client_cfg.get("env", 2))  # 1=LIBERO, 2=MOCK
    client_cfg["overwrite"] = True
    client_cfg["output_dir"] = str(CLIENT_OUT)
    client_fn = client_libero if env == 1 else client_mock

    stamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")

    if server == "none":
        # No server to talk to: only a mock agent can run standalone.
        client_cfg["agent"] = "mock"
        client_fn.remote(json.dumps(client_cfg), stamp)
    else:
        client_cfg["agent"] = "policy"
        server_cfg = _server_config()
        if server == "mock":
            server_cfg["policy"] = {"action_horizon": 50, "action_dim": 14, "gpu": "l40s"}
        server_fn = serve_sim if server == "sim" else serve_mock
        with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
            handle = server_fn.spawn(json.dumps(server_cfg), stamp, urls, shutdown)
            print(f"[{stamp}] server spawned; waiting for tunnel + /metadata")
            client_cfg["host"], client_cfg["port"] = _await_server(urls, stamp)
            print(f"[{stamp}] server ready at {client_cfg['host']}:{client_cfg['port']}")
            try:
                client_fn.remote(json.dumps(client_cfg), stamp, shutdown)
            finally:
                handle.cancel()

    dest = pathlib.Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{stamp}] downloading outputs -> {dest / stamp}")
    subprocess.run(
        ["modal", "volume", "get", ARTIFACTS_VOLUME_NAME, stamp, str(dest), "--force"],
        check=True,
    )
