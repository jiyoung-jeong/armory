"""Run a single robot (scripts/run.py) on Modal, optionally against a server.

    # mock agent + mock env, no server (one CPU container)
    uv run modal run scripts/modal/run.py

    # single robot vs a real policy in LIBERO sim (server GPU + libero client GPU)
    uv run modal run scripts/modal/run.py --json-path client.json --server sim

    # ...vs a lightweight mock policy server (for scheduling tests)
    uv run modal run scripts/modal/run.py --json-path client.json --server mock

The server/client workers and the tunnel handshake are shared with setups.py.
The client image is picked from the run.py Args `env` ("libero" -> GPU,
"mock" -> CPU). Outputs land on the artifacts volume and download to --output-dir.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess

import modal
from scripts.modal.images import REMOTE_ROOT
from scripts.modal.setups import (
    CpuMockClient,
    CpuMockServer,
    GpuServer,
    LiberoClient,
    _await_server,
    app,
)
from scripts.modal.utils import ARTIFACTS_VOLUME_NAME

from evaluation.types import EnvironmentType

PORT = 8080


@app.local_entrypoint()
def main(json_path: str = "", server: str = "none", output_dir: str = "outputs") -> None:
    if server not in {"none", "mock", "sim"}:
        raise SystemExit("--server must be 'none', 'mock', or 'sim'.")

    stamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = str(REMOTE_ROOT / stamp)

    client_cfg = json.loads(pathlib.Path(json_path).read_text()) if json_path else {}
    # ``--json-path`` describes ExperimentConfig directly; scripts.run.Args
    # nests it under ``experiment_config``. Also accept an already-wrapped
    # Args payload for callers that use scripts.run's native schema.
    if "experiment_config" not in client_cfg:
        client_cfg = {"experiment_config": client_cfg}
    client_cfg["overwrite"] = True
    client_cfg["output_dir"] = str(REMOTE_ROOT / stamp)
    env = EnvironmentType(client_cfg.get("env", EnvironmentType.MOCK))
    client = {EnvironmentType.LIBERO: LiberoClient, EnvironmentType.MOCK: CpuMockClient}[env]()

    with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
        if server == "none":
            # No server to talk to: only a mock agent can run standalone.
            client_cfg["agent"] = "mock"
            client.run.remote(
                module="scripts.run",
                run_dir=run_dir,
                args_json=json.dumps(client_cfg),
                run_id=stamp,
                stream_logs=True,
                shutdown=shutdown,
            )
        else:
            client_cfg["agent"] = "policy"
            server_cfg = {"model": "pi05", "env": "libero", "max_batch_size": 1, "port": PORT}
            if server == "mock":
                server_cfg["policy"] = {"action_horizon": 50, "action_dim": 14, "gpu": "l40s"}
            server_worker = GpuServer() if server == "sim" else CpuMockServer()
            handle = server_worker.serve.spawn(
                run_dir=run_dir,
                args_json=json.dumps(server_cfg),
                port=PORT,
                run_id=stamp,
                stream_logs=True,
                urls=urls,
                shutdown=shutdown,
            )
            print(f"[{stamp}] server spawned; waiting for tunnel + /metadata")
            host, port = _await_server(urls, stamp)
            if not host:
                raise RuntimeError("server failed before forwarding its port")
            client_cfg["host"], client_cfg["port"] = host, port
            print(f"[{stamp}] server ready at {host}:{port}")
            try:
                client.run.remote(
                    module="scripts.run",
                    run_dir=run_dir,
                    args_json=json.dumps(client_cfg),
                    run_id=stamp,
                    stream_logs=True,
                    shutdown=shutdown,
                )
            finally:
                handle.cancel()

    dest = pathlib.Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{stamp}] downloading outputs -> {dest / stamp}")
    subprocess.run(
        ["modal", "volume", "get", ARTIFACTS_VOLUME_NAME, stamp, str(dest), "--force"],
        check=True,
    )
