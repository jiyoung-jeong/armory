from __future__ import annotations

import copy
import dataclasses
import json
import os
import pathlib
import shlex
import shutil
import signal
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

from armory.serving.protocol import SchedulerConfig
from evaluation.server_control_client import ServerControlClient

APP_NAME = "armory-experiments"
REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")
STAGING_ROOT = pathlib.Path("/tmp/armory-modal")  # noqa: S108
CHECKPOINT_VOLUME_NAME = "openpi-checkpoints"

SERVER_GPU = "L40S"
LIBERO_GPU = "T4"
REGION = "us-east"
TIMEOUT_S = 2 * 60 * 60
POOL_TIMEOUT_S = 12 * 60 * 60
# Safety net on the client subprocess; the container `timeout` is the real cap.
CLIENT_TIMEOUT_S = 90 * 60
# The CaseRunner is already executing while its child L40S call is queued, so
# this application-level deadline (unlike Modal's function timeout) includes
# GPU scheduling delay. Keep policy startup as a separate, shorter phase.
SERVER_START_TIMEOUT_S = 60 * 60
SERVER_READY_TIMEOUT_S = 10 * 60

# One CPU per robot process plus a small orchestration/rendering buffer. Modal
# caps a single function at 16 CPUs; larger fleets share those.
CLIENT_CPU_BUFFER = 2
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
        return subprocess.Popen(
            ["bash", "-o", "pipefail", "-c", shell_cmd],
            cwd=str(REMOTE_ROOT),
            start_new_session=True,
        )
    return subprocess.Popen(
        cmd,
        cwd=str(REMOTE_ROOT),
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    # scripts.run owns one process per robot. Signal the subprocess's isolated
    # process group so a timed-out client cannot leave robot connections alive
    # when the next pooled case calls /reset.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


def _ship(run_dir: pathlib.Path, staging: pathlib.Path) -> str:
    """Fold the staged args/log into the run dir, copy it to the artifacts volume."""
    run_dir.mkdir(parents=True, exist_ok=True)  # the client may have wiped it
    for path in staging.iterdir():
        shutil.copy(path, run_dir / path.name)
    artifact_dir = REMOTE_ARTIFACTS_ROOT / run_dir.relative_to(REMOTE_ROOT)
    artifacts_volume.reload()
    shutil.copytree(run_dir, artifact_dir, dirs_exist_ok=True)
    artifacts_volume.commit()
    return str(artifact_dir)


def _finalize_artifact(artifact_dir: pathlib.Path) -> dict[str, Any]:
    """Regenerate plots after all producers have shipped, then summarize."""
    from evaluation.metrics import generate_all_plots  # noqa: PLC0415

    generate_all_plots(artifact_dir)
    summary = summarize(artifact_dir)
    artifacts_volume.commit()
    return summary


def _prepare(
    run_dir: str, *, name: str, module: str, args_json: str
) -> tuple[pathlib.Path, pathlib.Path, list[str]]:
    """Write a subprocess's args + command manifest; return run dir, staging dir, argv.

    Args, manifest and log are staged outside ``run_dir`` and copied back by
    ``_ship``: the client's own ``--overwrite`` rmtree's its output dir at
    startup, which is the same directory, and would take them with it.

    ``-m scripts.<module>`` (not the file path) so package imports resolve from
    the repository root consistently.
    """
    staging = STAGING_ROOT / name
    shutil.rmtree(staging, ignore_errors=True)
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
    artifact_dir = pathlib.Path(_ship(directory, staging))
    return {
        "run_id": run_id,
        "status": status,
        "error": error,
        "artifact_remote_path": str(artifact_dir),
        **_finalize_artifact(artifact_dir),
    }


def _run(
    *,
    run_dir: str,
    args_json: str,
    run_id: str,
    stream_logs: bool,
    shutdown: modal.Dict | None,
) -> dict[str, Any]:
    """Run the client to completion, summarize its metrics, ship the run dir."""
    directory, staging, argv = _prepare(
        run_dir, name="client", module="scripts.run", args_json=args_json
    )
    result: dict[str, Any] = {"run_id": run_id}
    proc: subprocess.Popen | None = None
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
        # A timed-out client must not survive into the next pooled case with
        # the same robot IDs and global scheduler state.
        _terminate(proc)
    try:
        result["artifact_remote_path"] = _ship(directory, staging)
    finally:
        if shutdown is not None:
            # Release a single-case server only after the client artifacts are
            # committed, so its later commit can merge server telemetry into them.
            shutdown[run_id] = True
    return result


# --------------------------------------------------------------------------
# Modal workers (one per image; resources fixed per class)
# --------------------------------------------------------------------------
@app.cls(
    image=gpu_server_image,
    region=REGION,
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
    region=REGION,
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
    region=REGION,
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
    region=REGION,
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
    cpus = min(num_robots + CLIENT_CPU_BUFFER, MAX_CLIENT_CPUS)
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
    server_config.setdefault("server", {})["output_dir"] = run_dir
    client_config.setdefault("experiment_config", {}).setdefault("environment", {})["kind"] = (
        mode.env_kind
    )
    if mode.server is CpuMockServer:
        overrides = {k: v for k, v in server_config.get("policy", {}).items() if k in MOCK_POLICY}
        model = {"model": server_config.get("model", "pi05")}
        server_config["policy"] = {**MOCK_POLICY, **model, **overrides}


def _num_robots(client_config: dict) -> int:
    return len(client_config.get("experiment_config", {}).get("robots", [{}]))


def _await_server(
    urls: modal.Dict,
    run_id: str,
    *,
    start_timeout_s: float = SERVER_START_TIMEOUT_S,
    ready_timeout_s: float = SERVER_READY_TIMEOUT_S,
) -> tuple[str, int]:
    """Block until the server publishes its tunnel and /metadata answers.

    Returns ``("", 0)`` if the server posted the poison address after failing.
    """
    deadline = time.time() + start_timeout_s
    while run_id not in urls:
        if time.time() > deadline:
            raise RuntimeError(f"server never published tunnel for {run_id}")
        time.sleep(2)
    host, port = urls[run_id]
    if not host:
        return "", 0
    ready = time.time() + ready_timeout_s
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
    server_start_timeout_s: float = SERVER_START_TIMEOUT_S,
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

    with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
        if spec.server is None:
            client = _client_worker(spec, _num_robots(client_config))
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
        server_finished = False
        try:
            host, port = _await_server(
                urls,
                run_id,
                start_timeout_s=server_start_timeout_s,
            )
            if not host:
                return {
                    "run_id": run_id,
                    "status": "failed",
                    "error": "server failed before forwarding",
                }
            client_config["host"], client_config["port"] = host, port
            print(f"[{run_id}] server ready at {host}:{port}; launching client", flush=True)
            client = _client_worker(spec, _num_robots(client_config))
            result = client.run.remote(
                run_dir=run_dir,
                args_json=json.dumps(client_config),
                run_id=run_id,
                stream_logs=stream_logs,
                shutdown=shutdown,
            )
            server_result = handle.get(timeout=SERVER_READY_TIMEOUT_S)
            server_finished = True
            result.update(
                {
                    key: value
                    for key, value in server_result.items()
                    if key not in {"run_id", "status", "error"}
                }
            )
            if server_result.get("status") != "ok" and result.get("status") == "ok":
                result.update(
                    status="failed",
                    error=server_result.get("error") or "server failed during shutdown",
                )
            return result
        finally:
            if not server_finished:
                # Failure paths include a server that never answers /metadata;
                # cancel it so the GPU is not held until the container timeout.
                try:
                    handle.cancel()
                except Exception:  # noqa: BLE001
                    pass


def _run_case(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep one failed allocation/startup from aborting an entire mapped sweep."""
    try:
        return launch(**payload)
    except Exception as exc:  # noqa: BLE001
        run_id = str(payload.get("run_id", ""))
        print(f"[{run_id}] case failed before producing a result: {exc!r}", flush=True)
        return {
            "run_id": run_id,
            "status": "failed",
            "error": repr(exc),
        }


@app.cls(
    image=cpu_mock_image,  # cheap: only spawns the server/client and hands off URLs
    region=REGION,
    timeout=TIMEOUT_S,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class CaseRunner:
    """Run one sweep case on its own containers, so cases fan out via `.map`."""

    @modal.method()
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _run_case(payload)


def server_reuse_key(server_config: dict[str, Any]) -> str:
    """Canonical key for settings that require a policy-server restart.

    Scheduler configuration is deliberately absent: the client applies it at
    the acknowledged ``/reset`` boundary between cases. This is the same
    reset-before-run lifecycle used by the main-branch interactive runner.
    """
    config = copy.deepcopy(server_config)
    config.pop("log_dir", None)
    server = config.get("server") or {}
    server.pop("output_dir", None)
    server.pop("scheduler", None)
    config["server"] = server
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _wait_for_pooled_server(
    host: str,
    port: int,
    proc: subprocess.Popen,
    *,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"pooled server exited during startup (code={proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/metadata", timeout=5):
                return
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() >= deadline:
                raise RuntimeError(f"pooled server /metadata never came up: {exc!r}") from exc
            time.sleep(5)


class _PooledServerSession:
    """One persistent serve.py process and tunnel inside an L40S lane."""

    def __init__(
        self,
        *,
        argv: list[str],
        staging: pathlib.Path,
        port: int,
        pool_id: str,
        stream_logs: bool,
        startup_timeout_s: float,
    ) -> None:
        self.argv = argv
        self.staging = staging
        self.port = port
        self.pool_id = pool_id
        self.stream_logs = stream_logs
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None
        self._tunnel_context: Any = None
        self.host = ""
        self.forwarded_port = 0
        self.restarts = 0

    def ensure_started(self) -> tuple[str, int]:
        if self.proc is not None and self.proc.poll() is None and self.host:
            return self.host, self.forwarded_port
        self.close()
        self.restarts += 1
        log_path = self.staging / f"server_{self.restarts:02d}.log"
        self.proc = _popen_logged(
            self.argv,
            log_path=log_path,
            tag=f"server-pool/{self.pool_id}",
            stream_logs=self.stream_logs,
        )
        try:
            self._tunnel_context = modal.forward(self.port, unencrypted=True)
            tunnel = self._tunnel_context.__enter__()
            self.host, self.forwarded_port = tunnel.tcp_socket
            _wait_for_pooled_server(
                self.host,
                self.forwarded_port,
                self.proc,
                timeout_s=self.startup_timeout_s,
            )
        except Exception:
            self.close()
            raise
        print(
            f"[server-pool/{self.pool_id}] ready at {self.host}:{self.forwarded_port}",
            flush=True,
        )
        return self.host, self.forwarded_port

    def close(self) -> None:
        _terminate(self.proc)
        self.proc = None
        if self._tunnel_context is not None:
            try:
                self._tunnel_context.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._tunnel_context = None
        self.host = ""
        self.forwarded_port = 0


SERVER_METRIC_FILES = (
    "batches.jsonl",
    "events.jsonl",
    "scheduler_decisions.jsonl",
)


def _server_metric_offsets(metrics_dir: pathlib.Path) -> dict[str, int]:
    return {
        name: (metrics_dir / name).stat().st_size if (metrics_dir / name).exists() else 0
        for name in SERVER_METRIC_FILES
    }


def _copy_pooled_server_metrics(
    payload: dict[str, Any],
    *,
    metrics_dir: pathlib.Path,
    offsets: dict[str, int],
) -> pathlib.Path:
    """Copy this case's fenced slice of the pool-wide telemetry into its artifact."""
    artifacts_volume.reload()
    run_dir = pathlib.Path(payload["run_dir"])
    artifact_dir = REMOTE_ARTIFACTS_ROOT / run_dir.relative_to(REMOTE_ROOT)
    destination = artifact_dir / "server"
    destination.mkdir(parents=True, exist_ok=True)

    for name in SERVER_METRIC_FILES:
        source = metrics_dir / name
        if not source.exists():
            continue
        size = source.stat().st_size
        start = offsets.get(name, 0)
        # A recovered server may have reopened its pool log with ``w``.
        if start > size:
            start = 0
        with source.open("rb") as stream:
            stream.seek(start)
            (destination / name).write_bytes(stream.read())

    metadata = metrics_dir / "metadata.json"
    if metadata.exists():
        shutil.copy2(metadata, destination / metadata.name)
    return artifact_dir


def _run_pooled_case(
    payload: dict[str, Any],
    *,
    host: str,
    port: int,
    pool_id: str,
    reused: bool,
    server_metrics_dir: pathlib.Path,
) -> dict[str, Any]:
    run_id = str(payload["run_id"])
    offsets = _server_metric_offsets(server_metrics_dir)
    try:
        spec = MODES["gpu"]
        client_config = copy.deepcopy(payload["client_config"])
        _apply_mode(
            spec, copy.deepcopy(payload["server_config"]), client_config, payload["run_dir"]
        )
        client_config["host"], client_config["port"] = host, port
        print(f"[{run_id}] using server pool {pool_id}; launching client", flush=True)
        client = _client_worker(spec, _num_robots(client_config))
        result = client.run.remote(
            run_dir=payload["run_dir"],
            args_json=json.dumps(client_config),
            run_id=run_id,
            stream_logs=bool(payload.get("stream_logs", False)),
            shutdown=None,
        )
    except Exception as exc:  # noqa: BLE001
        result = {"run_id": run_id, "status": "failed", "error": repr(exc)}

    fenced = False
    try:
        # Fence the final in-flight batch before slicing the shared pool logs.
        # The next client performs its own reset, so this extra boundary only
        # finalizes telemetry and guarantees a clean server state after failures.
        scheduler = SchedulerConfig.model_validate(payload["client_config"]["scheduler_config"])
        ServerControlClient(host=host, port=port).reset_server(
            scheduler,
            active_session_timeout_s=60.0,
        )
        fenced = True
    except Exception as exc:  # noqa: BLE001
        previous = result.get("error")
        detail = f"post-run server reset failed: {exc!r}"
        result.update(status="failed", error=f"{previous}; {detail}" if previous else detail)

    if fenced:
        try:
            artifact_dir = _copy_pooled_server_metrics(
                payload,
                metrics_dir=server_metrics_dir,
                offsets=offsets,
            )
            result.update(_finalize_artifact(artifact_dir))
        except Exception as exc:  # noqa: BLE001
            detail = f"server telemetry finalization failed: {exc!r}"
            previous = result.get("error")
            result.update(
                status="failed",
                error=f"{previous}; {detail}" if previous else detail,
                server_metrics_error=repr(exc),
            )

    result.update(server_pool_id=pool_id, server_reused=reused)
    return result


def _load_pool_progress(path: pathlib.Path, run_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Recover successful cases if Modal replays a preempted shard input."""
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text())
        return {
            row["run_id"]: row
            for row in rows
            if isinstance(row, dict) and row.get("run_id") in run_ids and row.get("status") == "ok"
        }
    except Exception as exc:  # noqa: BLE001
        print(f"Ignoring unreadable server-pool checkpoint {path}: {exc!r}", flush=True)
        return {}


def _save_pool_progress(
    path: pathlib.Path,
    cases: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [results[case["run_id"]] for case in cases if case["run_id"] in results]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2))
    temporary.replace(path)
    artifacts_volume.commit()


def _run_pooled_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a cold-compatible sequence of cases on one persistent L40S."""
    cases: list[dict[str, Any]] = payload["cases"]
    if not cases:
        return []
    pool_id = str(payload["pool_id"])
    expected_key = server_reuse_key(cases[0]["server_config"])
    if payload.get("server_reuse_key") != expected_key:
        raise ValueError(f"server pool shard {pool_id} has an invalid reuse key")
    if any(server_reuse_key(case["server_config"]) != expected_key for case in cases[1:]):
        raise ValueError(f"server pool shard {pool_id} mixes restart-required configurations")

    server_config = copy.deepcopy(cases[0]["server_config"])
    pool_run_dir = str(payload["pool_run_dir"])
    server_config["log_dir"] = str(pathlib.Path(pool_run_dir) / "internal_logs")
    server_config.setdefault("server", {})["output_dir"] = pool_run_dir
    directory, staging, argv = _prepare(
        pool_run_dir,
        name=f"server_pool_{pool_id}",
        module="scripts.serve",
        args_json=json.dumps(server_config),
    )
    (staging / "pool_manifest.json").write_text(
        json.dumps(
            {
                "pool_id": pool_id,
                "run_ids": [case["run_id"] for case in cases],
                "server_reuse_key": expected_key,
            },
            indent=2,
        )
    )
    artifact_dir = REMOTE_ARTIFACTS_ROOT / directory.relative_to(REMOTE_ROOT)
    progress_path = artifact_dir / "pool_results.json"
    run_ids = {str(case["run_id"]) for case in cases}
    try:
        artifacts_volume.reload()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not refresh server-pool checkpoints: {exc!r}", flush=True)
    results_by_id = _load_pool_progress(progress_path, run_ids)
    if results_by_id:
        print(
            f"[server-pool/{pool_id}] resuming after {len(results_by_id)} completed case(s)",
            flush=True,
        )

    session = _PooledServerSession(
        argv=argv,
        staging=staging,
        port=int(server_config["port"]),
        pool_id=pool_id,
        stream_logs=bool(payload.get("stream_logs", False)),
        startup_timeout_s=float(payload.get("server_start_timeout_s", SERVER_START_TIMEOUT_S)),
    )
    try:
        for index, case in enumerate(cases):
            run_id = str(case["run_id"])
            if run_id in results_by_id:
                continue

            startup_error: Exception | None = None
            for attempt in range(2):
                restart_count = session.restarts
                try:
                    host, port = session.ensure_started()
                    startup_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    startup_error = exc
                    print(
                        f"[server-pool/{pool_id}] server start attempt {attempt + 1}/2 "
                        f"failed: {exc!r}",
                        flush=True,
                    )
                    session.close()

            if startup_error is not None:
                error = f"server failed to start after 2 attempts: {startup_error!r}"
                for remaining in cases[index:]:
                    remaining_id = str(remaining["run_id"])
                    if remaining_id in results_by_id:
                        continue
                    results_by_id[remaining_id] = {
                        "run_id": remaining_id,
                        "status": "failed",
                        "error": error,
                        "server_pool_id": pool_id,
                        "server_reused": False,
                    }
                try:
                    _save_pool_progress(progress_path, cases, results_by_id)
                except Exception as exc:  # noqa: BLE001
                    print(f"Could not save server-pool progress: {exc!r}", flush=True)
                break

            result = _run_pooled_case(
                case,
                host=host,
                port=port,
                pool_id=pool_id,
                reused=index > 0 and session.restarts == restart_count,
                server_metrics_dir=directory / "server",
            )
            results_by_id[run_id] = result
            if result.get("status") != "ok":
                # A failed/terminated client may leave control state in an
                # uncertain phase; recover with a clean process next case.
                session.close()
            try:
                _save_pool_progress(progress_path, cases, results_by_id)
            except Exception as exc:  # noqa: BLE001
                print(f"Could not save server-pool progress: {exc!r}", flush=True)
    finally:
        session.close()
        try:
            artifact_path = _ship(directory, staging)
        except Exception as exc:  # noqa: BLE001
            artifact_path = ""
            for result in results_by_id.values():
                result["server_artifact_error"] = repr(exc)

    for result in results_by_id.values():
        result["server_artifact_remote_path"] = artifact_path
    try:
        _save_pool_progress(progress_path, cases, results_by_id)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not finalize server-pool progress: {exc!r}", flush=True)
    return [results_by_id[str(case["run_id"])] for case in cases]


@app.cls(
    image=gpu_server_image,
    region=REGION,
    timeout=POOL_TIMEOUT_S,
    startup_timeout=SERVER_START_TIMEOUT_S,
    cpu=4,
    memory=16384,
    gpu=SERVER_GPU,
    volumes={
        str(REMOTE_ARTIFACTS_ROOT): artifacts_volume,
        CHECKPOINT_VOLUME_PATH: checkpoint_volume,
    },
)
class PooledGpuServer:
    """Run one cold-compatible case shard on a persistent policy server."""

    @modal.method()
    def run(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return _run_pooled_shard(payload)
        except Exception as exc:  # noqa: BLE001
            pool_id = str(payload.get("pool_id", ""))
            print(f"[server-pool/{pool_id}] shard failed: {exc!r}", flush=True)
            return [
                {
                    "run_id": str(case.get("run_id", "")),
                    "status": "failed",
                    "error": repr(exc),
                    "server_pool_id": pool_id,
                    "server_reused": False,
                }
                for case in payload.get("cases", [])
            ]
