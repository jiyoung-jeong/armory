"""Shared Modal classes and predefined run setups for the sweep scripts.

An experiment is always a *policy server* talking to one or more *clients*. There
are only a few combinations we run:

    setup       server                client                containers
    --------    ------------------    -----------------     ----------------
    MOCK        mock policy (CPU)     mock client (CPU)     1 (colocated)
    REAL_CPU    real policy (GPU)     LIBERO, OSMesa CPU    2 (server on GPU)
    REAL_GPU    real policy (GPU)     LIBERO, EGL GPU       2 (both on GPU)

A *case* is just a ``serve.Args`` plus a ``run_libero.Args`` — the real argument
dataclasses of ``scripts/serve.py`` and ``scripts/run_libero.py``. The container
pickles each into the run dir and execs ``scripts/_run_entry.py``, which calls the
script's ``main(args)``. There is no argv reconstruction and no per-experiment
glue here: the experiment scripts build the two dataclasses, the setup runs them.

The MOCK setup runs server + client as two subprocesses in one container. The
REAL_* setups put them on separate containers, bridged by a ``modal.forward``
tunnel and a pair of ephemeral ``modal.Dict``s (one to publish the server's
address, one for the client to signal it's done).
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import modal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# Modal may import this module as /root/_setups.py for class services while the
# repo's scripts directory is mounted separately in the image at /app/scripts.
sys.path.insert(0, "/app/scripts")
sys.path.insert(0, "/app/scripts/experiments")
import run_libero  # noqa: E402
import serve  # noqa: E402
from _images import (  # noqa: E402
    CHECKPOINT_VOLUME_PATH,
    REMOTE_ROOT,
    cpu_libero_client_image,
    cpu_mock_image,
    gpu_libero_client_image,
    gpu_server_image,
)
from _utils import ARTIFACTS_VOLUME_NAME, summarize  # noqa: E402

APP_NAME = "armory-experiments"

REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")
# Bulky binaries we never want in the downloaded artifact tree.

CHECKPOINT_VOLUME_NAME = "openpi-checkpoints"

REGION = "us-east"
SERVER_GPU = "L40S"
LIBERO_CLIENT_GPU = "A10G"

# Safety net on the client subprocess; the container `timeout` is the real cap.
CLIENT_TIMEOUT_S = 90 * 60

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


@dataclasses.dataclass(frozen=True)
class Case:
    server_args: serve.Args
    client_args: run_libero.Args
    stamp: str

    def __post_init__(self) -> None:
        self.server_args.output_dir = self.run_dir / "outputs"
        self.client_args.output_dir = self.run_dir / "outputs"

    @property
    def run_id(self) -> str:
        parts = [
            f"scheduler={self.server_args.scheduling_algorithm}",
            f"num_robots={self.client_args.num_robots}",
            f"seed={self.client_args.seed}",
            f"max_batch_size={self.server_args.max_batch_size}",
            f"alpha={self.server_args.alpha}",
        ]
        return "__".join(parts)

    # NOTE: run path is separate from artifact path because modal Volumes might not be good for lots of writes
    @property
    def run_dir(self) -> pathlib.Path:
        return REMOTE_ROOT / self.stamp / self.run_id

    @property
    def artifact_dir(self) -> str:
        return REMOTE_ARTIFACTS_ROOT / self.stamp / self.run_id


# --------------------------------------------------------------------------
# On-container helpers
# --------------------------------------------------------------------------
def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


# TODO: can try writing to volume to see if it doesn't hurt
def _ship(case: Case) -> str:
    """Copy the case's run dir onto the artifacts volume; return the remote path."""
    shutil.copytree(case.run_dir, case.artifact_dir)
    artifacts_volume.commit()
    return str(case.artifact_dir)


# --------------------------------------------------------------------------
# On-container run bodies
# --------------------------------------------------------------------------
def _run_server(
    case: Case, stamp: str, *, urls: modal.Dict, shutdown: modal.Dict
) -> dict[str, Any]:
    """Start the policy server, forward its port, hold until the client is done."""
    case.run_dir.mkdir(parents=True, exist_ok=True)
    case.server_args.to_json(case.run_dir / "server_args.json")
    status, error = "ok", None
    proc: subprocess.Popen | None = None
    try:
        with (
            open(case.run_dir / "server.log", "w") as stdout_file,
            open(case.run_dir / "server.log", "w") as stderr_file,
        ):
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/serve.py",
                    "--json-path",
                    str(case.run_dir / "server_args.json"),
                ],
                cwd=str(REMOTE_ROOT),
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
            )
        with modal.forward(case.server_args.port, unencrypted=True) as tunnel:
            urls[case.run_id] = tunnel.tcp_socket
            print(f"[{case.run_id}] server tunnel up at {tunnel.tcp_socket}", flush=True)
            while case.run_id not in shutdown:
                if proc.poll() is not None:
                    status, error = "failed", f"server exited early (code={proc.returncode})"
                    break
                time.sleep(2)
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", repr(exc)
        urls[case.run_id] = ("", 0)  # poison so the orchestrator doesn't hang
    finally:
        _terminate(proc)
    _ship(case)
    return {"run_id": case.run_id, "status": status, "error": error}


def _run_client(case: Case, stamp: str, *, shutdown: modal.Dict) -> dict[str, Any]:
    """Run the LIBERO client to completion, summarize, ship the run dir."""
    case.run_dir.mkdir(parents=True, exist_ok=True)
    case.server_args.to_json(case.run_dir / "server_args.json")

    result: dict[str, Any] = {"run_id": case.run_id}
    try:
        with (
            open(case.run_dir / "client.log", "w") as stdout_file,
            open(case.run_dir / "client.log", "w") as stderr_file,
        ):
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/run_libero.py",
                    "--json-path",
                    str(case.run_dir / "client_args.json"),
                ],
                cwd=str(REMOTE_ROOT),
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
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
# Modal classes (one per image; resources retuned per-setup via with_options)
# --------------------------------------------------------------------------
# NOTE: we use modal classes instead of functions so we can specify resources using with_options
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
    image=cpu_libero_client_image,
    timeout=2 * 60 * 60,
    cpu=16,
    memory=16384,
    region=REGION,
    max_containers=5,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CpuLiberoClient:
    """LIBERO sim client with OSMesa software rendering; no GPU."""

    @modal.method()
    def run(self, case: Case, *, shutdown: modal.Dict) -> dict[str, Any]:  # noqa: ANN001
        return _run_client(case, shutdown=shutdown)


# --------------------------------------------------------------------------
# Setups
# --------------------------------------------------------------------------
@app.cls(
    image=cpu_mock_image,
    timeout=2 * 60 * 60,
    memory=16384,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class MockSetup:
    """One container per case, server + client colocated."""

    @modal.method()
    def run(
        self,
        case: Case,
    ) -> Iterator[dict[str, Any]]:
        """Run server + client as two subprocesses in a single container."""
        case.run_dir.mkdir(parents=True, exist_ok=True)
        case.server_args.to_json(case.run_dir / "server_args.json")
        case.client_args.to_json(case.run_dir / "client_args.json")
        result: dict[str, Any] = {"run_id": case.run_id}
        with (
            open(case.run_dir / "server.log", "w") as stdout_file,
            open(case.run_dir / "server.log", "w") as stderr_file,
        ):
            server_proc = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/serve.py",
                    "--json-path",
                    str(case.run_dir / "server_args.json"),
                ],
                cwd=str(REMOTE_ROOT),
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
            )
        try:
            with (
                open(case.run_dir / "client.log", "w") as stdout_file,
                open(case.run_dir / "client.log", "w") as stderr_file,
            ):
                client_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/run_libero.py",
                        "--json-path",
                        str(case.run_dir / "client_args.json"),
                    ],
                    cwd=str(REMOTE_ROOT),
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
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
    image=cpu_mock_image,  # reusing this because it's lightweight
    timeout=2 * 60 * 60,
    max_containers=10,  # TODO: change to 5 if we also need gpu for clients
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class SplitSetup:
    """Server and client on separate containers, bridged by a forwarded tunnel.

    Every server is spawned up front so they all queue in Modal's scheduler; each
    client launches as soon as its server publishes a tunnel address. Concurrency
    is capped by ``max_concurrent`` because a split case holds two containers (and,
    for REAL_GPU, two GPUs) at once.
    """

    @modal.method()
    def run(
        self,
        case: Case,
    ) -> Iterator[dict[str, Any]]:
        # TODO: automatically determine server and client resources
        server = GpuServer()
        # TODO: figure out if CPU can work, otherwise just use GPU
        client = CpuLiberoClient()

        with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
            server_handle = server.serve.spawn(case, urls=urls, shutdown=shutdown)
            try:
                return client.run.remote(case, shutdown=shutdown)
            finally:
                _terminate(server_handle)


# Predefined setups: an experiment script picks one of these.
MOCK = MockSetup()
LIBERO = SplitSetup()
