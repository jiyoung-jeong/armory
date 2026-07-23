"""Shared Modal workers for running a policy server + client on Modal.

A run is a *policy server* talking to a *client*, always on separate containers
bridged by a ``modal.forward`` tunnel and a pair of ephemeral ``modal.Dict``s
(one to publish the server's address, one for the client to signal completion).
Either side is mock (CPU, no weights/rendering) or real (GPU): a real policy
server needs a GPU, a LIBERO client needs a GPU for EGL rendering. Hence four
image-pinned workers: ``GpuServer``/``CpuMockServer`` and ``LiberoClient``/
``CpuMockClient``.

The sweep entrypoint (``sweep_experiments.py``) drives these per ``Case`` via
``CaseRunner``; ``run.py`` drives the same workers directly for a single robot.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import TYPE_CHECKING, Any

import modal
from scripts.modal.images import (
    CHECKPOINT_VOLUME_PATH,
    REMOTE_ROOT,
    cpu_mock_image,
    gpu_libero_client_image,
    gpu_server_image,
)
from scripts.modal.utils import ARTIFACTS_VOLUME_NAME, summarize

if TYPE_CHECKING:
    # Heavy, GPU/Linux-only imports; the annotations below never resolve them at
    # runtime (thanks to `from __future__ import annotations`), so importing
    # setups.py stays light enough to launch run.py from a macOS dev venv.
    import serve
    from scripts import run_all

    from evaluation.types import EnvironmentType, ExperimentConfig

APP_NAME = "armory-experiments"
REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")
CHECKPOINT_VOLUME_NAME = "openpi-checkpoints"

REGION = "us-east"
SERVER_GPU = "L40S"
LIBERO_GPU = "T4"
TIMEOUT_S = 2 * 60 * 60
# Safety net on the client subprocess; the container `timeout` is the real cap.
CLIENT_TIMEOUT_S = 90 * 60

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


@dataclasses.dataclass(frozen=True)
class Case:
    server_args: serve.Args
    client_args: run_all.Args  # embeds experiment_config + scheduler_config
    experiment_name: str
    stream_logs: bool
    stamp: str
    server_variant: str = ""

    def __post_init__(self) -> None:
        # serve.Args has no output_dir; only the client needs it for metrics.
        self.client_args.output_dir = self.run_dir / "outputs"
        self.client_args.overwrite = True

    @property
    def experiment_config(self) -> ExperimentConfig:
        return self.client_args.experiment_config

    @property
    def num_robots(self) -> int:
        return len(self.experiment_config.robots)

    @property
    def client_env(self) -> EnvironmentType:
        return self.experiment_config.env

    @property
    def run_id(self) -> str:
        parts = [
            f"scheduler={self.server_args.scheduler.scheduling_algorithm}",
            f"experiment={self.experiment_name}",
            f"num_robots={self.num_robots}",
            f"seed={self.experiment_config.seed}",
            f"max_batch_size={self.server_args.max_batch_size}",
            f"alpha={self.server_args.scheduler.alpha}",
        ]
        if self.server_variant:
            parts.append(f"server_variant={self.server_variant}")
        return "__".join(parts)

    # Run path is separate from artifact path because Modal Volumes don't love
    # lots of small writes; we write hot to the container disk and copy at the end.
    @property
    def run_dir(self) -> pathlib.Path:
        return REMOTE_ROOT / self.stamp / self.run_id

    def to_payload(self) -> dict[str, Any]:
        """Flatten to primitives + JSON strings for CaseRunner.

        Case holds pydantic ``serve.Args``/``run_all.Args``; Modal would have to
        import ``serve`` on the orchestrator container to unpickle them, and that
        bare module isn't importable there. So Case stays local and only this
        plain dict (str/int/bool) crosses the Modal boundary.
        """
        import serve

        from evaluation.types import EnvironmentType

        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "server_args_json": self.server_args.model_dump_json(),
            "client_args_json": self.client_args.model_dump_json(),
            "num_robots": self.num_robots,
            "mock_policy": isinstance(self.server_args.policy, serve.Mock),
            "mock_env": self.client_env == EnvironmentType.MOCK,
            "port": self.server_args.port,
            "stream_logs": self.stream_logs,
        }


# --------------------------------------------------------------------------
# On-container helpers
# --------------------------------------------------------------------------
def _popen_logged(
    cmd: list[str], *, log_path: pathlib.Path, tag: str, stream_logs: bool
) -> subprocess.Popen:
    """Run ``cmd`` in the repo root, always writing its output to ``log_path``.

    With ``stream_logs``, also tag and tee each line to container stdout (which
    Modal surfaces); off by default so high-volume logs don't flood the terminal.
    """
    if stream_logs:
        # '#' as the sed delimiter because tags (e.g. "server/scheduler=...") contain '/'.
        shell_cmd = (
            f"{shlex.join(cmd)} 2>&1 | sed -u 's#^#[{tag}] #' | tee {shlex.quote(str(log_path))}"
        )
        return subprocess.Popen(shell_cmd, shell=True, cwd=str(REMOTE_ROOT))
    return subprocess.Popen(
        cmd, cwd=str(REMOTE_ROOT), stdout=log_path.open("w"), stderr=subprocess.STDOUT
    )


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _write_command_manifest(run_dir: pathlib.Path, commands: dict[str, list[str]]) -> None:
    """Record the subprocess commands so a run can be reproduced by hand."""
    manifest = {
        "cwd": str(REMOTE_ROOT),
        "commands": {
            name: {"argv": argv, "shell": shlex.join(argv)} for name, argv in commands.items()
        },
    }
    (run_dir / "commands.json").write_text(json.dumps(manifest, indent=2))
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REMOTE_ROOT))}", ""]
    for name, argv in commands.items():
        lines += [f"# {name}", shlex.join(argv), ""]
    (run_dir / "commands.sh").write_text("\n".join(lines) + "\n")


def _ship(run_dir: pathlib.Path) -> str:
    """Copy a run dir onto the artifacts volume; return the remote path."""
    artifact_dir = REMOTE_ARTIFACTS_ROOT / run_dir.relative_to(REMOTE_ROOT)
    shutil.copytree(run_dir, artifact_dir, dirs_exist_ok=True)
    artifacts_volume.commit()
    return str(artifact_dir)


def _server_cmd(run_dir: pathlib.Path) -> list[str]:
    # `-m scripts.serve` (not the file path) so /app leads sys.path and the src
    # `utils`/`logging_config` win over the shadowing scripts/utils.py.
    return [sys.executable, "-m", "scripts.serve", "--json-path", str(run_dir / "server_args.json")]


def _client_cmd(run_dir: pathlib.Path, module: str) -> list[str]:
    return [sys.executable, "-m", module, "--json-path", str(run_dir / "client_args.json")]


def _prepare(run_dir: pathlib.Path, *, args_name: str, args_json: str) -> pathlib.Path:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / args_name).write_text(args_json)
    return log_dir


def _await_server(urls: modal.Dict, run_id: str) -> tuple[str, int]:
    """Block until the server publishes its tunnel and /metadata answers.

    Returns ``("", 0)`` if the server posted the poison address after failing.
    """
    deadline = time.time() + 15 * 60
    while run_id not in urls:
        if time.time() > deadline:
            raise RuntimeError(f"server never published tunnel for {run_id}")
        time.sleep(2)
    host, port = urls[run_id]
    if not host:
        return "", 0
    ready = time.time() + 10 * 60
    while True:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/metadata", timeout=5):
                return host, port
        except Exception as exc:  # noqa: BLE001
            if time.time() > ready:
                raise RuntimeError(f"server /metadata never came up: {exc!r}") from exc
            time.sleep(5)


def _serve(
    *,
    run_dir: str,
    args_json: str,
    port: int,
    run_id: str,
    stream_logs: bool,
    urls: modal.Dict,
    shutdown: modal.Dict,
) -> dict[str, Any]:
    """Start the policy server, forward its port, hold until the client is done."""
    run_dir = pathlib.Path(run_dir)
    log_dir = _prepare(run_dir, args_name="server_args.json", args_json=args_json)
    cmd = _server_cmd(run_dir)
    _write_command_manifest(run_dir, {"server": cmd})
    status, error, proc = "ok", None, None
    try:
        proc = _popen_logged(
            cmd, log_path=log_dir / "server.log", tag=f"server/{run_id}", stream_logs=stream_logs
        )
        with modal.forward(port, unencrypted=True) as tunnel:
            urls[run_id] = tunnel.tcp_socket
            print(f"[server/{run_id}] tunnel up at {tunnel.tcp_socket}", flush=True)
            while run_id not in shutdown:
                if proc.poll() is not None:
                    status, error = "failed", f"server exited early (code={proc.returncode})"
                    break
                time.sleep(2)
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", repr(exc)
        urls[run_id] = ("", 0)  # poison so the orchestrator doesn't hang
    finally:
        _terminate(proc)
    return {
        "run_id": run_id,
        "status": status,
        "error": error,
        "artifact_remote_path": _ship(run_dir),
    }


def _run(
    *,
    module: str,
    run_dir: str,
    args_json: str,
    run_id: str,
    stream_logs: bool,
    shutdown: modal.Dict,
) -> dict[str, Any]:
    """Run the client to completion, summarize its metrics, ship the run dir."""
    run_dir = pathlib.Path(run_dir)
    log_dir = _prepare(run_dir, args_name="client_args.json", args_json=args_json)
    cmd = _client_cmd(run_dir, module)
    _write_command_manifest(run_dir, {"client": cmd})
    result: dict[str, Any] = {"run_id": run_id}
    try:
        proc = _popen_logged(
            cmd, log_path=log_dir / "client.log", tag=f"client/{run_id}", stream_logs=stream_logs
        )
        rc = proc.wait(timeout=CLIENT_TIMEOUT_S)
        if rc != 0:
            result.update(status="failed", error=f"client exited with code {rc}")
        else:
            result.update(summarize(run_dir / "outputs"), status="ok")
    except Exception as exc:  # noqa: BLE001
        result.update(status="failed", error=repr(exc))
    finally:
        shutdown[run_id] = True  # always release the server
    result["artifact_remote_path"] = _ship(run_dir)
    return result


# --------------------------------------------------------------------------
# Modal workers (one per image; resources fixed per class)
# --------------------------------------------------------------------------
@app.cls(
    image=gpu_server_image,
    timeout=TIMEOUT_S,
    cpu=4,
    memory=16384,
    gpu=SERVER_GPU,
    region=REGION,
    max_containers=5,
    volumes={
        str(REMOTE_ARTIFACTS_ROOT): artifacts_volume,
        CHECKPOINT_VOLUME_PATH: checkpoint_volume,
    },
)
class GpuServer:
    """Real PI05/GR00T policy server on a GPU."""

    @modal.method()
    def serve(self, **kwargs) -> dict[str, Any]:
        return _serve(**kwargs)


@app.cls(
    image=cpu_mock_image,
    timeout=TIMEOUT_S,
    cpu=2,
    memory=8192,
    region=REGION,
    max_containers=10,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CpuMockServer:
    """Mock policy server (no weights, no GPU)."""

    @modal.method()
    def serve(self, **kwargs) -> dict[str, Any]:
        return _serve(**kwargs)


# Client cpu is set per call (1 per robot process) via `.with_options`; the base
# here is the single-robot default used by run.py.
@app.cls(
    image=gpu_libero_client_image,
    timeout=TIMEOUT_S,
    cpu=1,
    memory=16384,
    gpu=LIBERO_GPU,
    region=REGION,
    max_containers=5,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class LiberoClient:
    """LIBERO sim client with hardware EGL rendering on a T4."""

    @modal.method()
    def run(self, **kwargs) -> dict[str, Any]:
        return _run(**kwargs)


@app.cls(
    image=cpu_mock_image,
    timeout=TIMEOUT_S,
    cpu=1,
    memory=8192,
    region=REGION,
    max_containers=10,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CpuMockClient:
    """Mock-env client (no rendering, no GPU)."""

    @modal.method()
    def run(self, **kwargs) -> dict[str, Any]:
        return _run(**kwargs)


# --------------------------------------------------------------------------
# Orchestrator (Case-based; used by the sweep)
# --------------------------------------------------------------------------
@app.cls(
    image=cpu_mock_image,  # cheap: only spawns the server/client and hands off URLs
    timeout=TIMEOUT_S,
    max_containers=10,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CaseRunner:
    """Run one case: server and client on separate containers, bridged by a tunnel.

    Takes a plain ``Case.to_payload()`` dict (never a ``Case``) so nothing here
    needs ``serve``/``run_all`` importable to deserialize the argument.
    """

    @modal.method()
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload["run_id"]
        run_dir = payload["run_dir"]
        server = CpuMockServer() if payload["mock_policy"] else GpuServer()
        client_cls = CpuMockClient if payload["mock_env"] else LiberoClient
        client = client_cls.with_options(cpu=payload["num_robots"])()  # 1 cpu per robot process
        with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
            handle = server.serve.spawn(
                run_dir=run_dir,
                args_json=payload["server_args_json"],
                port=payload["port"],
                run_id=run_id,
                stream_logs=payload["stream_logs"],
                urls=urls,
                shutdown=shutdown,
            )
            print(f"[orch/{run_id}] server spawned; waiting for tunnel", flush=True)
            host, port = _await_server(urls, run_id)
            if not host:
                return {"run_id": run_id, "status": "failed", "error": "server failed before fwd"}
            client_args = json.loads(payload["client_args_json"])
            client_args["host"], client_args["port"] = host, port
            print(f"[orch/{run_id}] server ready at {host}:{port}; launching client", flush=True)
            try:
                return client.run.remote(
                    module="scripts.run_all",
                    run_dir=run_dir,
                    args_json=json.dumps(client_args),
                    run_id=run_id,
                    stream_logs=payload["stream_logs"],
                    shutdown=shutdown,
                )
            finally:
                # The client sets shutdown[run_id] before returning; cancel is a
                # belt-and-braces net for the failure paths.
                try:
                    handle.cancel()
                except Exception:  # noqa: BLE001
                    pass
