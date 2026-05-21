"""Shared Modal classes and setups for the experiment sweep scripts.

An experiment is a *policy server* talking to one or more *clients*. Combinations:

    server policy    client env    image(s)                       containers
    --------------   -----------   ----------------------------   ----------
    mock             mock          cpu_mock_image (colocated)     1
    mock             libero        cpu_mock_image + libero GPU    2 (split)
    default/ckpt     mock          gpu_server + cpu_mock_image    2 (split)
    default/ckpt     libero        gpu_server + libero GPU        2 (split)

A *case* is a ``serve.Args`` plus a ``run_libero.Args``. The container pickles each
into the run dir and execs the matching script. The sweep entrypoint inspects the
server policy and client env to pick a setup; nothing else here needs that knowledge.

The colocated mock setup runs server + client as two subprocesses in one container.
Split setups put them on separate containers, bridged by ``modal.forward`` and a
pair of ephemeral ``modal.Dict``s (one to publish the server's address, one for the
client to signal completion).
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
from typing import Any

import modal

# Modal loads this module as /root/_setups.py for class services, but the repo's
# scripts/ tree is mounted at /app/scripts/. Add both: the parent (for local
# runs) and /app/scripts/modal (for remote, where _images.py + _utils.py live).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, "/app/scripts")
sys.path.insert(0, "/app/scripts/modal")
import run_libero  # noqa: E402
import serve  # noqa: E402
from _images import (  # noqa: E402
    CHECKPOINT_VOLUME_PATH,
    REMOTE_ROOT,
    cpu_mock_image,
    gpu_libero_client_image,
    gpu_server_image,
)
from _utils import ARTIFACTS_VOLUME_NAME, summarize  # noqa: E402

APP_NAME = "armory-experiments"

REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")

CHECKPOINT_VOLUME_NAME = "openpi-checkpoints"

REGION = "us-east"
SERVER_GPU = "L40S"
LIBERO_CLIENT_GPU = "L40S"

# Safety net on the client subprocess; the container `timeout` is the real cap.
CLIENT_TIMEOUT_S = 90 * 60

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


@dataclasses.dataclass(frozen=True)
class Case:
    server_args: serve.Args
    client_args: run_libero.Args
    experiment_config: dict[str, Any]
    experiment_name: str
    stream_logs: bool
    stamp: str
    server_variant: str = ""

    def __post_init__(self) -> None:
        # serve.Args has no output_dir; only the client needs it for metrics.
        self.client_args.output_dir = self.run_dir / "outputs"
        self.client_args.experiment_config = str(self.experiment_config_path)

    @property
    def settings(self) -> run_libero.ExperimentSettings:
        return run_libero.ExperimentSettings.from_config(self.experiment_config)

    @property
    def num_robots(self) -> int:
        return self.settings.num_robots

    @property
    def client_env(self) -> str:
        return self.settings.env

    @property
    def run_id(self) -> str:
        parts = [
            f"scheduler={self.server_args.scheduling_algorithm}",
            f"experiment={self.experiment_name}",
            f"num_robots={self.num_robots}",
            f"seed={self.client_args.seed}",
            f"max_batch_size={self.server_args.max_batch_size}",
            f"alpha={self.server_args.alpha}",
        ]
        if self.server_variant:
            parts.append(f"server_variant={self.server_variant}")
        return "__".join(parts)

    # Run path is separate from artifact path because Modal Volumes don't love
    # lots of small writes; we write hot to the container disk and copy at the end.
    @property
    def run_dir(self) -> pathlib.Path:
        return REMOTE_ROOT / self.stamp / self.run_id

    @property
    def experiment_config_path(self) -> pathlib.Path:
        return self.run_dir / "experiment_config.json"

    @property
    def artifact_dir(self) -> pathlib.Path:
        return REMOTE_ARTIFACTS_ROOT / self.stamp / self.run_id


# --------------------------------------------------------------------------
# On-container helpers
# --------------------------------------------------------------------------
def _popen_logged(
    cmd: list[str],
    *,
    cwd: str,
    log_path: pathlib.Path,
    tag: str,
    stream_logs: bool,
) -> subprocess.Popen:
    """Run ``cmd`` and always write its output to ``log_path``.

    When ``stream_logs`` is true, also tag and tee each line to container stdout.
    Modal surfaces container stdout in ``modal run`` and ``modal app logs``, so
    keeping this off by default prevents high-volume server/client logs from
    flooding the local terminal while preserving on-disk logs in artifacts.
    """
    if stream_logs:
        # Use '#' as the sed delimiter because tags (e.g. "server/scheduler=...") contain '/'.
        shell_cmd = (
            f"{shlex.join(cmd)} 2>&1 | sed -u 's#^#[{tag}] #' | tee {shlex.quote(str(log_path))}"
        )
        return subprocess.Popen(shell_cmd, shell=True, cwd=cwd)

    log_file = log_path.open("w")
    return subprocess.Popen(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT)


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _write_experiment_config(case: Case) -> None:
    case.experiment_config_path.write_text(json.dumps(case.experiment_config, indent=2) + "\n")


def _ship(case: Case) -> str:
    """Copy the case's run dir onto the artifacts volume; return the remote path."""
    shutil.copytree(case.run_dir, case.artifact_dir)
    artifacts_volume.commit()
    return str(case.artifact_dir)


def _write_command_manifest(run_dir: pathlib.Path, commands: dict[str, list[str]]) -> None:
    """Write a human-readable + runnable record of the subprocess commands.

    ``commands.json`` keeps the raw argv; ``commands.sh`` is a runnable script so a
    case can be reproduced by hand from the run dir.
    """
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


# --------------------------------------------------------------------------
# On-container run bodies
# --------------------------------------------------------------------------
def _run_server(case: Case, *, urls: modal.Dict, shutdown: modal.Dict) -> dict[str, Any]:
    """Start the policy server, forward its port, hold until the client is done."""
    case.run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = case.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    case.server_args.to_json(case.run_dir / "server_args.json")
    _write_experiment_config(case)
    server_cmd = [
        sys.executable,
        "scripts/serve.py",
        "--json-path",
        str(case.run_dir / "server_args.json"),
    ]
    _write_command_manifest(case.run_dir, {"server": server_cmd})
    status, error = "ok", None
    proc: subprocess.Popen | None = None
    try:
        proc = _popen_logged(
            server_cmd,
            cwd=str(REMOTE_ROOT),
            log_path=log_dir / "server.log",
            tag=f"server/{case.run_id}",
            stream_logs=case.stream_logs,
        )
        with modal.forward(case.server_args.port, unencrypted=True) as tunnel:
            urls[case.run_id] = tunnel.tcp_socket
            print(f"[server/{case.run_id}] tunnel up at {tunnel.tcp_socket}", flush=True)
            last_beat = 0.0
            while case.run_id not in shutdown:
                if proc.poll() is not None:
                    status, error = "failed", f"server exited early (code={proc.returncode})"
                    break
                now = time.time()
                if now - last_beat > 30:
                    print(f"[server/{case.run_id}] alive, waiting for client", flush=True)
                    last_beat = now
                time.sleep(2)
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", repr(exc)
        urls[case.run_id] = ("", 0)  # poison so the orchestrator doesn't hang
    finally:
        _terminate(proc)
    _ship(case)
    return {"run_id": case.run_id, "status": status, "error": error}


def _run_client(case: Case, *, shutdown: modal.Dict) -> dict[str, Any]:
    """Run the client to completion, summarize, ship the run dir."""
    case.run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = case.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_experiment_config(case)
    case.client_args.to_json(case.run_dir / "client_args.json")
    client_cmd = [
        sys.executable,
        "scripts/run_libero.py",
        "--json-path",
        str(case.run_dir / "client_args.json"),
    ]
    _write_command_manifest(case.run_dir, {"client": client_cmd})

    result: dict[str, Any] = {"run_id": case.run_id}
    try:
        proc = _popen_logged(
            client_cmd,
            cwd=str(REMOTE_ROOT),
            log_path=log_dir / "client.log",
            tag=f"client/{case.run_id}",
            stream_logs=case.stream_logs,
        )
        rc = proc.wait(timeout=CLIENT_TIMEOUT_S)
        if rc != 0:
            result.update(status="failed", error=f"client exited with code {rc}")
        else:
            result.update(summarize(pathlib.Path(case.client_args.output_dir)))
            result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result.update(status="failed", error=repr(exc))
    finally:
        shutdown[case.run_id] = True  # always release the server
    result["artifact_remote_path"] = _ship(case)
    return result


# --------------------------------------------------------------------------
# Modal classes (one per image; resources fixed per class)
# --------------------------------------------------------------------------
@app.cls(
    image=gpu_server_image,
    timeout=2 * 60 * 60,
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
    """Real PI05/GR00T policy server on a GPU; no sim code."""

    @modal.method()
    def serve(self, case: Case, *, urls: modal.Dict, shutdown: modal.Dict) -> dict[str, Any]:  # noqa: ANN001
        return _run_server(case, urls=urls, shutdown=shutdown)


@app.cls(
    image=cpu_mock_image,
    timeout=2 * 60 * 60,
    cpu=2,
    memory=8192,
    region=REGION,
    max_containers=10,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CpuMockServer:
    """Mock policy server (no weights, no GPU); used when policy.type == 'mock'."""

    @modal.method()
    def serve(self, case: Case, *, urls: modal.Dict, shutdown: modal.Dict) -> dict[str, Any]:  # noqa: ANN001
        return _run_server(case, urls=urls, shutdown=shutdown)


@app.cls(
    image=gpu_libero_client_image,
    timeout=2 * 60 * 60,
    cpu=16,
    memory=16384,
    gpu=LIBERO_CLIENT_GPU,
    region=REGION,
    max_containers=5,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class GpuLiberoClient:
    """LIBERO sim client with hardware EGL rendering on a small GPU."""

    @modal.method()
    def run(self, case: Case, *, shutdown: modal.Dict) -> dict[str, Any]:  # noqa: ANN001
        return _run_client(case, shutdown=shutdown)


@app.cls(
    image=cpu_mock_image,
    timeout=2 * 60 * 60,
    cpu=4,
    memory=8192,
    region=REGION,
    max_containers=10,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CpuMockClient:
    """Mock-env client (no rendering, no GPU); used when client.env == 'mock'."""

    @modal.method()
    def run(self, case: Case, *, shutdown: modal.Dict) -> dict[str, Any]:  # noqa: ANN001
        return _run_client(case, shutdown=shutdown)


# --------------------------------------------------------------------------
# Setups
# --------------------------------------------------------------------------
@app.cls(
    image=cpu_mock_image,
    cpu=16,
    timeout=2 * 60 * 60,
    memory=16384,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class MockSetup:
    """Server + client colocated in one CPU container (mock policy + mock env only)."""

    @modal.method()
    def run(self, case: Case) -> dict[str, Any]:
        case.run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = case.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        case.server_args.to_json(case.run_dir / "server_args.json")
        _write_experiment_config(case)
        case.client_args.to_json(case.run_dir / "client_args.json")
        server_cmd = [
            sys.executable,
            "scripts/serve.py",
            "--json-path",
            str(case.run_dir / "server_args.json"),
        ]
        client_cmd = [
            sys.executable,
            "scripts/run_libero.py",
            "--json-path",
            str(case.run_dir / "client_args.json"),
        ]
        _write_command_manifest(case.run_dir, {"server": server_cmd, "client": client_cmd})
        result: dict[str, Any] = {"run_id": case.run_id}
        server_proc = _popen_logged(
            server_cmd,
            cwd=str(REMOTE_ROOT),
            log_path=log_dir / "server.log",
            tag=f"server/{case.run_id}",
            stream_logs=case.stream_logs,
        )
        try:
            client_proc = _popen_logged(
                client_cmd,
                cwd=str(REMOTE_ROOT),
                log_path=log_dir / "client.log",
                tag=f"client/{case.run_id}",
                stream_logs=case.stream_logs,
            )
            rc = client_proc.wait(timeout=CLIENT_TIMEOUT_S)
            if rc != 0:
                result.update(status="failed", error=f"client exited with code {rc}")
            else:
                result.update(summarize(pathlib.Path(case.client_args.output_dir)))
                result["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            result.update(status="failed", error=repr(exc))
        finally:
            _terminate(server_proc)
        result["artifact_remote_path"] = _ship(case)
        return result


@app.cls(
    image=cpu_mock_image,  # orchestrator is lightweight; only spawns class instances
    timeout=2 * 60 * 60,
    max_containers=10,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class SplitSetup:
    """Server and client on separate containers, bridged by a forwarded tunnel.

    The server class is picked from ``case.server_args.policy`` (mock -> CPU image,
    real -> GPU image); the client class is picked from ``case.client_env``
    (mock -> CPU image, libero -> GPU image). The orchestrator itself runs on the
    cheap CPU mock image since it only does spawn + URL handoff + wait.
    """

    @modal.method()
    def run(self, case: Case) -> dict[str, Any]:
        if isinstance(case.server_args.policy, serve.Mock):
            server = CpuMockServer()
            server_kind = "cpu-mock"
        else:
            server = GpuServer()
            server_kind = f"gpu-{SERVER_GPU.lower()}"
        if case.client_env == "mock":
            client = CpuMockClient()
            client_kind = "cpu-mock"
        else:
            client = GpuLiberoClient()
            client_kind = f"gpu-{LIBERO_CLIENT_GPU.lower()}"
        print(
            f"[orch/{case.run_id}] server={server_kind} client={client_kind}",
            flush=True,
        )

        with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
            server_handle = server.serve.spawn(case, urls=urls, shutdown=shutdown)
            print(
                f"[orch/{case.run_id}] server spawned (id={server_handle.object_id}); "
                f"waiting for tunnel",
                flush=True,
            )
            # Block until the server publishes its forwarded address, then point the
            # client at it. ``urls`` may carry ("", 0) as a poison value if the server
            # failed before forwarding — surface that as a clean failure.
            deadline = time.time() + 15 * 60
            while case.run_id not in urls:
                if time.time() > deadline:
                    raise RuntimeError(f"server never published tunnel for {case.run_id}")
                time.sleep(2)
            host, port = urls[case.run_id]
            if not host:
                return {
                    "run_id": case.run_id,
                    "status": "failed",
                    "error": "server failed before forwarding",
                }
            case.client_args.host = host
            case.client_args.port = port
            # Wait for the server to finish loading weights so the client's per-worker
            # barrier (60s in run_libero.py) isn't racing model load on cold start.
            import urllib.request

            metadata_url = f"http://{host}:{port}/metadata"
            ready_deadline = time.time() + 10 * 60
            while True:
                try:
                    with urllib.request.urlopen(metadata_url, timeout=5):
                        break
                except Exception as exc:  # noqa: BLE001
                    if time.time() > ready_deadline:
                        return {
                            "run_id": case.run_id,
                            "status": "failed",
                            "error": f"server /metadata never came up: {exc!r}",
                        }
                    print(
                        f"[orch/{case.run_id}] /metadata not ready ({exc.__class__.__name__})",
                        flush=True,
                    )
                    time.sleep(5)
            print(
                f"[orch/{case.run_id}] server ready at {host}:{port}; launching client", flush=True
            )
            try:
                result = client.run.remote(case, shutdown=shutdown)
                print(
                    f"[orch/{case.run_id}] client returned status={result.get('status')}",
                    flush=True,
                )
                return result
            finally:
                # The client sets shutdown[run_id] before returning, so the server
                # should already be winding down; cancel is a belt-and-braces net
                # for the failure paths (client crashed before signaling, etc.).
                try:
                    server_handle.cancel()
                except Exception:  # noqa: BLE001
                    pass


def select_setup(case: Case) -> modal.Cls:
    """Mock policy + mock env -> colocated CPU; everything else -> split."""
    if isinstance(case.server_args.policy, serve.Mock) and case.client_env == "mock":
        return MOCK
    return SPLIT


MOCK = MockSetup()
SPLIT = SplitSetup()
