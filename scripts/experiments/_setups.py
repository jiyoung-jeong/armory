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
import json
import pathlib
import pickle
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import modal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _images import (  # noqa: E402
    CHECKPOINT_VOLUME_PATH,
    REMOTE_ROOT,
    cpu_libero_client_image,
    cpu_mock_image,
    gpu_libero_client_image,
    gpu_server_image,
)

APP_NAME = "armory-experiments"

REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")
# Bulky binaries we never want in the downloaded artifact tree.
ARTIFACT_SKIP_SUFFIXES = {".mp4", ".parquet", ".npz"}

ARTIFACTS_VOLUME_NAME = "armory-experiment-artifacts"
CHECKPOINT_VOLUME_NAME = "openpi-checkpoints"

REGION = "us-east"
SERVER_GPU = "L40S"
LIBERO_CLIENT_GPU = "A10G"

# Safety net on the client subprocess; the container `timeout` is the real cap.
CLIENT_TIMEOUT_S = 90 * 60

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


@dataclasses.dataclass
class Case:
    """One sweep case: the two real argument dataclasses plus where it lives.

    ``run_dir`` is the per-case directory the container ships to the artifacts
    volume; the experiment script points ``server_args.log_dir``,
    ``client_args.log_dir`` and ``client_args.output_dir`` underneath it.
    ``experiment_config``, if set, is written to ``run_dir/experiment_config.json``
    on the container and wired into ``client_args.experiment_config``.
    """

    run_id: str
    run_dir: pathlib.Path
    server_args: Any  # scripts/serve.py Args
    client_args: Any  # scripts/run_libero.py Args
    experiment_config: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# On-container helpers
# --------------------------------------------------------------------------
def _spawn(entry: str, args: Any, *, args_path: pathlib.Path) -> subprocess.Popen:
    """Pickle an Args dataclass into the run dir and exec it via _run_entry.py.

    stdout/stderr are inherited, so the subprocess logs straight into the Modal
    container log; serve.py / run_libero.py write their own log files via their
    ``log_dir`` arg.
    """
    args_path.parent.mkdir(parents=True, exist_ok=True)
    args_path.write_bytes(pickle.dumps(args))
    return subprocess.Popen(
        [sys.executable, "scripts/_run_entry.py", entry, str(args_path)],
        cwd=str(REMOTE_ROOT),
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


def _materialize_experiment_config(case: Case) -> None:
    if case.experiment_config is None:
        return
    path = case.run_dir / "experiment_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case.experiment_config, indent=2))
    case.client_args.experiment_config = str(path)


def _ship(case: Case, stamp: str) -> str:
    """Copy the case's run dir onto the artifacts volume; return the remote path."""
    import shutil  # noqa: PLC0415

    dest = REMOTE_ARTIFACTS_ROOT / stamp / case.run_id
    for src in case.run_dir.rglob("*"):
        if src.is_dir() or src.suffix in ARTIFACT_SKIP_SUFFIXES:
            continue
        dst = dest / src.relative_to(case.run_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    artifacts_volume.commit()
    return str(dest)


def summarize(output_dir: pathlib.Path) -> dict[str, Any]:
    """Compute every available metric for a finished run; skip what isn't present.

    Shared by all three sweeps — the starvation and fairness experiments just read
    different columns out of the union.
    """
    import csv  # noqa: PLC0415
    import statistics  # noqa: PLC0415

    def _f(value: Any, default: float = 0.0) -> float:
        try:
            return default if value is None else float(value)
        except (TypeError, ValueError):
            return default

    out: dict[str, Any] = {}

    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        with summary_path.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            out["success_rate"] = sum(_f(r.get("success")) for r in rows) / len(rows)
            observed = sum(_f(r.get("observed_steps")) for r in rows)
            starved = sum(_f(r.get("starvation_steps")) for r in rows)
            out["starvation_rate"] = starved / observed if observed else 0.0
            pf_observed = sum(_f(r.get("post_first_observed_steps")) for r in rows)
            pf_starved = sum(_f(r.get("post_first_starvation_steps")) for r in rows)
            out["post_first_starvation_rate"] = pf_starved / pf_observed if pf_observed else 0.0

    results_path = output_dir / "results.csv"
    if results_path.exists():
        by_robot: dict[str, dict[str, float]] = {}
        with results_path.open() as f:
            for row in csv.DictReader(f):
                robot = str(row.get("robot_idx", "unknown"))
                stats = by_robot.setdefault(robot, {"starved": 0.0, "observed": 0.0})
                stats["starved"] += _f(row.get("starvation_steps"))
                stats["observed"] += _f(row.get("observed_steps"))
        rates = sorted(s["starved"] / s["observed"] for s in by_robot.values() if s["observed"] > 0)
        if rates:
            out["robot_starvation_rate_max"] = max(rates)
            out["robot_starvation_rate_std"] = statistics.pstdev(rates) if len(rates) > 1 else 0.0
            tail = max(1, int(len(rates) * 0.1))
            out["robot_starvation_rate_cvar90"] = sum(rates[-tail:]) / tail

    runtime_path = output_dir / "runtime_metadata.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        out["max_steps"] = runtime.get("max_steps", "")
        out["num_trials_per_task"] = runtime.get("num_trials_per_task", "")
    server_path = output_dir / "server_metadata.json"
    if server_path.exists():
        server = json.loads(server_path.read_text())
        out["max_batch_size"] = server.get("max_batch_size", "")
        out["action_horizon"] = server.get("action_horizon", "")

    try:
        from sims.libero.metrics import compute_server_timing_health  # noqa: PLC0415

        health = compute_server_timing_health(output_dir)
        if health:
            out.update(health)
    except Exception:  # noqa: BLE001
        pass
    try:
        from sims.libero.metrics import compute_fairness_metrics  # noqa: PLC0415

        fairness = compute_fairness_metrics(output_dir)
        if fairness is not None:
            out["alpha_observed"] = fairness.get("alpha")
            out["jain_freshness"] = fairness.get("jain_freshness")
            out["jain_starvation"] = fairness.get("jain_starvation")
            rates = fairness.get("starvation_rate") or []
            if rates:
                out["mean_starvation"] = float(sum(rates) / len(rates))
                out["max_starvation"] = float(max(rates))
                out["min_starvation"] = float(min(rates))
    except Exception:  # noqa: BLE001
        pass
    try:
        from sims.libero.metrics import compute_starvation_variance_series  # noqa: PLC0415

        series = compute_starvation_variance_series(output_dir)
        if series is not None:
            out["starvation_variance"] = series["final_starvation_variance"]
    except Exception:  # noqa: BLE001
        pass

    return out


# --------------------------------------------------------------------------
# On-container run bodies
# --------------------------------------------------------------------------
def _run_server(
    case: Case, stamp: str, *, urls: modal.Dict, shutdown: modal.Dict
) -> dict[str, Any]:
    """Start the policy server, forward its port, hold until the client is done."""
    case.run_dir.mkdir(parents=True, exist_ok=True)
    status, error = "ok", None
    proc: subprocess.Popen | None = None
    try:
        proc = _spawn("serve", case.server_args, args_path=case.run_dir / "server_args.pkl")
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
    _ship(case, stamp)
    return {"run_id": case.run_id, "status": status, "error": error}


def _run_client(case: Case, stamp: str, *, shutdown: modal.Dict) -> dict[str, Any]:
    """Run the LIBERO client to completion, summarize, ship the run dir."""
    case.run_dir.mkdir(parents=True, exist_ok=True)
    _materialize_experiment_config(case)
    result: dict[str, Any] = {"run_id": case.run_id}
    try:
        proc = _spawn("run_libero", case.client_args, args_path=case.run_dir / "client_args.pkl")
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
    result["artifact_remote_path"] = _ship(case, stamp)
    return result


def _run_colocated(case: Case, stamp: str) -> dict[str, Any]:
    """Run server + client as two subprocesses in a single container."""
    case.run_dir.mkdir(parents=True, exist_ok=True)
    _materialize_experiment_config(case)
    result: dict[str, Any] = {"run_id": case.run_id}
    server_proc = _spawn("serve", case.server_args, args_path=case.run_dir / "server_args.pkl")
    try:
        client_proc = _spawn(
            "run_libero", case.client_args, args_path=case.run_dir / "client_args.pkl"
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
    result["artifact_remote_path"] = _ship(case, stamp)
    return result


# --------------------------------------------------------------------------
# Modal classes (one per image; resources retuned per-setup via with_options)
# --------------------------------------------------------------------------
@app.cls(
    image=cpu_mock_image,
    timeout=2 * 60 * 60,
    cpu=4,
    memory=8192,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
class MockRunner:
    """Colocated mock policy server + mock client on one CPU container."""

    @modal.method()
    def run(self, case: Case, stamp: str) -> dict[str, Any]:
        return _run_colocated(case, stamp)


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
    def serve(
        self, case: Case, stamp: str, *, urls: modal.Dict, shutdown: modal.Dict
    ) -> dict[str, Any]:
        return _run_server(case, stamp, urls=urls, shutdown=shutdown)


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
    def run(self, case: Case, stamp: str, *, shutdown: modal.Dict) -> dict[str, Any]:
        return _run_client(case, stamp, shutdown=shutdown)


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
    def run(self, case: Case, stamp: str, *, shutdown: modal.Dict) -> dict[str, Any]:
        return _run_client(case, stamp, shutdown=shutdown)


# --------------------------------------------------------------------------
# Setups
# --------------------------------------------------------------------------
class MockSetup:
    """One container per case, server + client colocated. Parallelized via .map()."""

    def run(
        self,
        cases: list[Case],
        *,
        stamp: str,
        cpu: int = 4,
        memory: int = 8192,
        max_containers: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        cases = list(cases)
        for case in cases:
            case.client_args.host = "127.0.0.1"
            case.client_args.port = case.server_args.port
        opts: dict[str, Any] = {"cpu": cpu, "memory": memory}
        if max_containers is not None:
            opts["max_containers"] = max_containers
        runner = MockRunner.with_options(**opts)()
        yield from runner.run.map(cases, kwargs={"stamp": stamp}, order_outputs=False)


class SplitSetup:
    """Server and client on separate containers, bridged by a forwarded tunnel.

    Every server is spawned up front so they all queue in Modal's scheduler; each
    client launches as soon as its server publishes a tunnel address. Concurrency
    is capped by ``max_concurrent`` because a split case holds two containers (and,
    for REAL_GPU, two GPUs) at once.
    """

    def __init__(self, *, client_cls: Any):
        self.client_cls = client_cls

    def run(
        self,
        cases: list[Case],
        *,
        stamp: str,
        max_concurrent: int = 5,
        server_cpu: int = 4,
        server_memory: int = 16384,
        client_cpu: int = 16,
        client_memory: int = 16384,
        client_gpu: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

        cases = list(cases)
        if not cases:
            return

        server = GpuServer.with_options(
            cpu=server_cpu, memory=server_memory, max_containers=max_concurrent
        )()
        client_opts: dict[str, Any] = {
            "cpu": client_cpu,
            "memory": client_memory,
            "max_containers": max_concurrent,
        }
        if client_gpu is not None:
            client_opts["gpu"] = client_gpu
        client = self.client_cls.with_options(**client_opts)()

        with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
            handles = {
                case.run_id: server.serve.spawn(case, stamp, urls=urls, shutdown=shutdown)
                for case in cases
            }
            print(
                f"Spawned {len(handles)} server(s); launching clients as tunnels open",
                flush=True,
            )

            def _run_one(case: Case) -> dict[str, Any]:
                handle = handles[case.run_id]
                while case.run_id not in urls:
                    try:
                        done = handle.get(timeout=0)
                    except TimeoutError:
                        time.sleep(5)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        return {
                            "run_id": case.run_id,
                            "status": "failed",
                            "error": f"server raised before publishing a URL: {exc!r}",
                        }
                    return {
                        "run_id": case.run_id,
                        "status": "failed",
                        "error": f"server finished without publishing a URL: {done!r}",
                    }

                host, port = urls[case.run_id]
                if not host:
                    shutdown[case.run_id] = True
                    return {
                        "run_id": case.run_id,
                        "status": "failed",
                        "error": "server failed before opening a tunnel",
                    }
                case.client_args.host = host
                case.client_args.port = port
                try:
                    return client.run.remote(case, stamp, shutdown=shutdown)
                except Exception as exc:  # noqa: BLE001
                    shutdown[case.run_id] = True
                    return {"run_id": case.run_id, "status": "failed", "error": repr(exc)}

            with ThreadPoolExecutor(max_workers=len(cases)) as ex:
                futures = [ex.submit(_run_one, case) for case in cases]
                for fut in as_completed(futures):
                    yield fut.result()

            # Drain server handles to surface any server-side exceptions.
            for run_id, handle in handles.items():
                try:
                    handle.get(timeout=300)
                except Exception as exc:  # noqa: BLE001
                    print(f"server {run_id} cleanup failed: {exc!r}", flush=True)


# Predefined setups: an experiment script picks one of these.
MOCK = MockSetup()
REAL_CPU = SplitSetup(client_cls=CpuLiberoClient)
REAL_GPU = SplitSetup(client_cls=GpuLiberoClient)
