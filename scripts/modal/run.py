"""Run one case (one fleet, one scheduler) on Modal.

# mock policy + mock envs, all CPU.
uv run modal run scripts/modal/run.py --mode mock

# real policy on an L40S driving LIBERO sim on a T4.
uv run modal run scripts/modal/run.py --mode gpu --client-config configs/client/libero/short.json

# LIBERO sim with no server at all, to debug the environment/agent loop.
uv run modal run scripts/modal/run.py --mode runtime
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess

from scripts.modal.app import MODES, app, launch
from scripts.modal.images import REMOTE_ROOT
from scripts.modal.utils import ARTIFACTS_VOLUME_NAME

DEFAULT_SERVER_CONFIG = {"model": "pi05", "env": "libero", "max_batch_size": 1, "port": 8080}


@app.local_entrypoint()
def main(
    mode: str = "mock",
    client_config: str = "",
    server_config: str = "",
    output_dir: str = "runs",
    stream_logs: bool = True,
) -> None:
    if mode not in MODES:
        raise SystemExit(f"--mode must be one of {', '.join(MODES)}.")

    stamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    server = (
        json.loads(pathlib.Path(server_config).read_text())
        if server_config
        else dict(DEFAULT_SERVER_CONFIG)
    )
    # --client-config holds an ExperimentConfig; scripts.run.Args nests it under
    # `experiment_config`. Accept an already-wrapped Args payload too.
    client = json.loads(pathlib.Path(client_config).read_text()) if client_config else {}
    if "experiment_config" not in client:
        client = {"experiment_config": client}

    result = launch(
        mode=mode,
        run_id=stamp,
        run_dir=str(REMOTE_ROOT / stamp),
        server_config=server,
        client_config=client,
        stream_logs=stream_logs,
    )
    print(f"[{stamp}] {result.get('status', '?')}: {result.get('error') or 'done'}")

    dest = pathlib.Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{stamp}] downloading outputs -> {dest / stamp}")
    subprocess.run(
        ["modal", "volume", "get", ARTIFACTS_VOLUME_NAME, stamp, str(dest), "--force"],
        check=True,
    )
