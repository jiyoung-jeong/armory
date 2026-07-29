from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any

import modal
from scripts.modal.images import (
    CHECKPOINT_VOLUME_PATH,
    REMOTE_ROOT,
    cpu_mock_image,
    gpu_libero_client_image,
    gpu_server_image,
)
from scripts.modal.utils import ARTIFACTS_VOLUME_NAME, summarize

APP_NAME = "armory-experiments"
REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")
STAGING_ROOT = pathlib.Path("/tmp/armory-modal")  # noqa: S108
CHECKPOINT_VOLUME_NAME = "openpi-checkpoints"

SERVER_GPU = "L40S"
LIBERO_GPU = "T4"
TIMEOUT_S = 2 * 60 * 60
# Safety net on the client subprocess; the container `timeout` is the real cap.
CLIENT_TIMEOUT_S = 90 * 60

# One CPU per robot process, so a fleet does not contend for the single-robot
# default. Modal caps a single function at 16 CPUs; larger fleets share those.
MAX_CLIENT_CPUS = 16
# A 20-robot LIBERO run used ~44 GiB while holding the 16 GiB single-robot
# reservation, so ask for 3 GiB per allocated CPU (48 GiB at the cap).
MIN_LIBERO_MEMORY_MIB = 16 * 1024
LIBERO_MEMORY_PER_CPU_MIB = 3 * 1024

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


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


def _ship(run_dir: pathlib.Path, staging: pathlib.Path) -> str:
    """Fold the staged args/log into the run dir, copy it to the artifacts volume."""
    run_dir.mkdir(parents=True, exist_ok=True)  # the client may have wiped it
    for path in staging.iterdir():
        shutil.copy(path, run_dir / path.name)
    artifact_dir = REMOTE_ARTIFACTS_ROOT / run_dir.relative_to(REMOTE_ROOT)
    shutil.copytree(run_dir, artifact_dir, dirs_exist_ok=True)
    artifacts_volume.commit()
    return str(artifact_dir)


def _prepare(
    run_dir: str, *, name: str, module: str, args_json: str
) -> tuple[pathlib.Path, pathlib.Path, list[str]]:
    """Write a subprocess's args + command manifest; return run dir, staging dir, argv.

    Args, manifest and log are staged outside ``run_dir`` and copied back by
    ``_ship``: the client's own ``--overwrite`` rmtree's its output dir at
    startup, which is the same directory, and would take them with it.

    ``-m scripts.<module>`` (not the file path) so /app leads sys.path and the
    src ``utils``/``logging_config`` win over the shadowing scripts/utils.py.
    """
    staging = STAGING_ROOT / name
    staging.mkdir(parents=True, exist_ok=True)
    args_path = staging / f"{name}_args.json"
    args_path.write_text(args_json)
    argv = [sys.executable, "-m", module, "--json-path", str(args_path)]
    manifest = {"cwd": str(REMOTE_ROOT), "argv": argv, "shell": shlex.join(argv)}
    (staging / f"{name}_command.json").write_text(json.dumps(manifest, indent=2))
    return pathlib.Path(run_dir), staging, argv


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
    directory, staging, argv = _prepare(
        run_dir, name="server", module="scripts.serve", args_json=args_json
    )
    status, error, proc = "ok", None, None
    try:
        proc = _popen_logged(
            argv, log_path=staging / "server.log", tag=f"server/{run_id}", stream_logs=stream_logs
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
        "artifact_remote_path": _ship(directory, staging),
    }


def _run(
    *,
    run_dir: str,
    args_json: str,
    run_id: str,
    stream_logs: bool,
    shutdown: modal.Dict,
) -> dict[str, Any]:
    """Run the client to completion, summarize its metrics, ship the run dir."""
    directory, staging, argv = _prepare(
        run_dir, name="client", module="scripts.run", args_json=args_json
    )
    result: dict[str, Any] = {"run_id": run_id}
    try:
        proc = _popen_logged(
            argv, log_path=staging / "client.log", tag=f"client/{run_id}", stream_logs=stream_logs
        )
        rc = proc.wait(timeout=CLIENT_TIMEOUT_S)
        if rc != 0:
            result.update(status="failed", error=f"client exited with code {rc}")
        else:
            result.update(summarize(directory), status="ok")
    except Exception as exc:  # noqa: BLE001
        result.update(status="failed", error=repr(exc))
    finally:
        shutdown[run_id] = True  # always release the server
    result["artifact_remote_path"] = _ship(directory, staging)
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
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CpuMockServer:
    """Mock policy server (no weights, no GPU)."""

    @modal.method()
    def serve(self, **kwargs) -> dict[str, Any]:
        return _serve(**kwargs)


# Client cpu/memory are set per call from the fleet size via `.with_options`
# (see `_client_worker`); the values here are the single-robot defaults.
@app.cls(
    image=gpu_libero_client_image,
    timeout=TIMEOUT_S,
    cpu=1,
    memory=MIN_LIBERO_MEMORY_MIB,
    gpu=LIBERO_GPU,
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
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CpuMockClient:
    """Mock-env client (no rendering, no GPU)."""

    @modal.method()
    def run(self, **kwargs) -> dict[str, Any]:
        return _run(**kwargs)


# --------------------------------------------------------------------------
# The three modes
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Mode:
    server: type | None  # None => no server; the agent returns null actions
    client: type
    agent: str  # scripts.run AgentType
    env_kind: str  # evaluation.envs.config discriminator


MODES: dict[str, Mode] = {
    "gpu": Mode(server=GpuServer, client=LiberoClient, agent="policy", env_kind="libero"),
    "mock": Mode(server=CpuMockServer, client=CpuMockClient, agent="policy", env_kind="mock"),
    "runtime": Mode(server=None, client=LiberoClient, agent="mock", env_kind="libero"),
}

# Only the mock server reads these; the horizon and dim shape what the client
# receives, so a server config may override them (batch latency comes from the
# `gpu` profile regardless of the CPU the mock actually runs on).
MOCK_POLICY = {"action_horizon": 50, "action_dim": 14, "gpu": "l40s"}


def _client_worker(mode: Mode, num_robots: int) -> Any:
    cpus = min(num_robots, MAX_CLIENT_CPUS)
    if mode.client is LiberoClient:
        memory = max(MIN_LIBERO_MEMORY_MIB, cpus * LIBERO_MEMORY_PER_CPU_MIB)
        return mode.client.with_options(cpu=cpus, memory=memory)()
    return mode.client.with_options(cpu=cpus)()


def _apply_mode(mode: Mode, server_config: dict, client_config: dict, run_dir: str) -> None:
    """Stamp the mode's consequences onto both configs, in place.

    The mode -- not the config files -- decides which agent runs, which
    environment backend it drives, and whether the policy loads weights. That
    keeps "env: mock but a GPU server" from being expressible at all.
    """
    client_config["agent"] = mode.agent
    client_config["output_dir"] = run_dir
    client_config["overwrite"] = True
    client_config.setdefault("experiment_config", {}).setdefault("environment", {})["kind"] = (
        mode.env_kind
    )
    if mode.server is CpuMockServer:
        overrides = {k: v for k, v in server_config.get("policy", {}).items() if k in MOCK_POLICY}
        model = {"model": server_config.get("model", "pi05")}
        server_config["policy"] = {**MOCK_POLICY, **model, **overrides}


def _num_robots(client_config: dict) -> int:
    return len(client_config.get("experiment_config", {}).get("robots", [{}]))


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


def launch(
    *,
    mode: str,
    run_id: str,
    run_dir: str,
    server_config: dict[str, Any],
    client_config: dict[str, Any],
    stream_logs: bool = False,
) -> dict[str, Any]:
    """Run one case end to end; return the client's result row.

    Everything crossing this boundary is plain JSON: ``launch`` runs both
    locally (``run.py``) and on a container (``CaseRunner``), and a pydantic
    ``serve.Args`` can not be unpickled on the orchestrator container -- the
    bare ``serve`` module isn't importable there.
    """
    spec = MODES[mode]
    server_config, client_config = copy.deepcopy(server_config), copy.deepcopy(client_config)
    _apply_mode(spec, server_config, client_config, run_dir)
    client = _client_worker(spec, _num_robots(client_config))

    with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
        if spec.server is None:
            return client.run.remote(
                run_dir=run_dir,
                args_json=json.dumps(client_config),
                run_id=run_id,
                stream_logs=stream_logs,
                shutdown=shutdown,
            )

        handle = spec.server().serve.spawn(
            run_dir=run_dir,
            args_json=json.dumps(server_config),
            port=server_config["port"],
            run_id=run_id,
            stream_logs=stream_logs,
            urls=urls,
            shutdown=shutdown,
        )
        print(f"[{run_id}] server spawned; waiting for tunnel + /metadata", flush=True)
        try:
            host, port = _await_server(urls, run_id)
            if not host:
                return {
                    "run_id": run_id,
                    "status": "failed",
                    "error": "server failed before forwarding",
                }
            client_config["host"], client_config["port"] = host, port
            print(f"[{run_id}] server ready at {host}:{port}; launching client", flush=True)
            return client.run.remote(
                run_dir=run_dir,
                args_json=json.dumps(client_config),
                run_id=run_id,
                stream_logs=stream_logs,
                shutdown=shutdown,
            )
        finally:
            # The client sets shutdown[run_id] before returning, so this is only
            # load-bearing on the failure paths -- including a server that never
            # answers /metadata, which would otherwise hold a GPU until its
            # container timeout.
            try:
                handle.cancel()
            except Exception:  # noqa: BLE001
                pass


@app.cls(
    image=cpu_mock_image,  # cheap: only spawns the server/client and hands off URLs
    timeout=TIMEOUT_S,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CaseRunner:
    """Run one sweep case on its own containers, so cases fan out via `.map`."""

    @modal.method()
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return launch(**payload)
