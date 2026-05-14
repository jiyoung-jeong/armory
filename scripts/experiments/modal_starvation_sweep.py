"""Run server/client scheduler sweeps on Modal L40S GPUs (real PI05 + LIBERO).

The mock-CPU version of this script lives at ``modal_starvation_sweep_mock.py``;
this file is the GPU sibling and uses real weights + the LIBERO simulator.

Example:
    modal run scripts/experiments/modal_starvation_sweep.py \
        --schedulers fixed-max-batch,greedy-deadline,lookahead-actions \
        --num-robots 1,2,4,6,8,10 \
        --seeds 7 \
        --output-dir experiments/sweeps/l40s
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
from typing import Any

import modal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _images import (  # noqa: E402
    CHECKPOINT_VOLUME_PATH,
    REMOTE_ROOT,
    gpu_libero_client_image,
    gpu_server_image,
)

APP_NAME = "armory-scheduler-sweep-l40s"
ARTIFACTS_VOLUME_NAME = "armory-scheduler-sweep-l40s-artifacts"
CHECKPOINT_VOLUME_NAME = "openpi-checkpoints"
GPU = "L40S"
REGION = "us-east"

REMOTE_OUTPUT_ROOT = pathlib.Path("/tmp/armory_sweep")
REMOTE_ARTIFACTS_ROOT = pathlib.Path("/artifacts")
ARTIFACT_SKIP_SUFFIXES = {".mp4", ".parquet", ".npz"}

# Each case gets its own (server_port, client_port) pair off this base so two
# accidentally-colocated cases can't bind the same port.
BASE_PORT = 8000
# Cold-start budget for the server subprocess: model download, JIT compile,
# warmup, and per-batch-size profiling. Generous because the first time a
# checkpoint is pulled it can take many minutes.
SERVER_READY_TIMEOUT_S = 30 * 60
# Modal 1.x can't vary cpu per spawn, so we bake a single count into the client
# function. The sweep asserts max(num_robots) + 2 <= CLIENT_CPU. Bump this if
# you're sweeping more than 10 robots; smaller cases overpay slightly.
CLIENT_CPU = 16
# LIBERO/mujoco does GPU rendering, so the client runs on a small GPU. Without
# one the per-step time inflates well above the policy's control period.
CLIENT_GPU = "A10G"
# Modal's account-wide GPU cap is shared across server + client. Each case
# consumes one L40S + one A10G simultaneously, so concurrent cases must be
# capped at GPU_CAP // 2 — otherwise a sweep can deadlock with all GPUs held
# by servers waiting for clients that can never be scheduled.
GPU_CAP = 10
MAX_CONCURRENT_CASES = GPU_CAP // 2

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


@dataclasses.dataclass(frozen=True)
class SweepCase:
    scheduler: str
    experiment_config: str  # path relative to repo root
    num_robots: int
    seed: int

    @property
    def run_id(self) -> str:
        config_name = pathlib.Path(self.experiment_config).stem
        return f"scheduler={self.scheduler}__config={config_name}__robots={self.num_robots}__seed={self.seed}"


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _copy_run_dir(src_root: pathlib.Path, dest_root: pathlib.Path) -> None:
    """Copy run_dir into the mounted artifacts volume, skipping bulky binaries."""
    for src in src_root.rglob("*"):
        if src.is_dir() or src.suffix in ARTIFACT_SKIP_SUFFIXES:
            continue
        dst = dest_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_server_cmd(srv_cfg: dict[str, Any], *, port: int, scheduler: str) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/serve.py",
        "--port",
        str(port),
        "--env",
        srv_cfg.get("env", "LIBERO"),
        "--model",
        srv_cfg.get("model", "PI05"),
        "--max-batch-size",
        str(srv_cfg.get("max_batch_size", 1)),
        "--scheduling-algorithm",
        scheduler,
        "policy:default",
    ]
    if "num_steps" in srv_cfg:
        cmd += ["--num-steps", str(srv_cfg["num_steps"])]
    return cmd


def _expand_experiment_config(exp_cfg: dict[str, Any], num_robots: int) -> dict[str, Any]:
    """Return a copy of exp_cfg with robot profiles set for num_robots robots.

    If the config already defines more than one robot explicitly, those profiles are used
    as-is and num_robots is ignored (the config is authoritative).
    Otherwise robot_0's profile is replicated to fill num_robots robots.
    """
    if len(exp_cfg["robots"]) > 1:
        actual = len(exp_cfg["robots"])
        return {**exp_cfg, "experiment": {**exp_cfg["experiment"], "num_robots": actual}}
    robot_template = exp_cfg["robots"]["robot_0"]
    return {
        **exp_cfg,
        "experiment": {**exp_cfg["experiment"], "num_robots": num_robots},
        "robots": {f"robot_{i}": dict(robot_template) for i in range(num_robots)},
    }


def _config_num_robots(cfg_path: str, fallback: int) -> int:
    """Read robot count from a local config file when robots are pre-defined, else fallback."""
    try:
        n = len(json.loads(pathlib.Path(cfg_path).read_text()).get("robots", {}))
        if n > 1:
            return n
    except Exception:
        pass
    return fallback


def _build_client_cmd(
    *,
    host: str,
    port: int,
    seed: int,
    output_dir: pathlib.Path,
    experiment_config_path: pathlib.Path,
    max_steps: int,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_libero.py",
        "--host",
        host,
        "--port",
        str(port),
        "--env",
        "libero",
        "--overwrite",
        "--progress-type",
        "logging",
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--experiment-config",
        str(experiment_config_path),
    ]


def _summarize_run(output_dir: pathlib.Path, case: SweepCase) -> dict[str, Any]:
    summary_path = output_dir / "summary.csv"
    results_path = output_dir / "results.csv"
    runtime_path = output_dir / "runtime_metadata.json"
    server_path = output_dir / "server_metadata.json"

    total_success = 0.0
    overall_starvation_rate = 0.0
    post_first_starvation_rate = 0.0
    if summary_path.exists():
        with summary_path.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            total_success = sum(_safe_float(r.get("success")) for r in rows) / len(rows)
            starvation_steps = sum(_safe_float(r.get("starvation_steps")) for r in rows)
            observed_steps = sum(_safe_float(r.get("observed_steps")) for r in rows)
            post_first_starvation_steps = sum(
                _safe_float(r.get("post_first_starvation_steps")) for r in rows
            )
            post_first_observed_steps = sum(
                _safe_float(r.get("post_first_observed_steps")) for r in rows
            )
            overall_starvation_rate = starvation_steps / observed_steps if observed_steps else 0.0
            post_first_starvation_rate = (
                post_first_starvation_steps / post_first_observed_steps
                if post_first_observed_steps
                else 0.0
            )

    robot_rates: list[float] = []
    if results_path.exists():
        by_robot: dict[str, dict[str, float]] = {}
        with results_path.open() as f:
            for row in csv.DictReader(f):
                robot = str(row.get("robot_idx", "unknown"))
                stats = by_robot.setdefault(robot, {"starvation_steps": 0.0, "observed_steps": 0.0})
                stats["starvation_steps"] += _safe_float(row.get("starvation_steps"))
                stats["observed_steps"] += _safe_float(row.get("observed_steps"))
        robot_rates = [
            stats["starvation_steps"] / stats["observed_steps"]
            for stats in by_robot.values()
            if stats["observed_steps"] > 0
        ]

    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    server = json.loads(server_path.read_text()) if server_path.exists() else {}
    sorted_rates = sorted(robot_rates)
    tail_count = max(1, int(len(sorted_rates) * 0.1)) if sorted_rates else 0

    summary = {
        "run_id": case.run_id,
        "scheduler": case.scheduler,
        "experiment_config": case.experiment_config,
        "num_robots": case.num_robots,
        "seed": case.seed,
        "success_rate": total_success,
        "starvation_rate": overall_starvation_rate,
        "post_first_starvation_rate": post_first_starvation_rate,
        "robot_starvation_rate_max": max(robot_rates) if robot_rates else 0.0,
        "robot_starvation_rate_std": _safe_float(__import__("statistics").pstdev(robot_rates))
        if len(robot_rates) > 1
        else 0.0,
        "robot_starvation_rate_cvar90": sum(sorted_rates[-tail_count:]) / tail_count
        if tail_count
        else 0.0,
        "max_batch_size": server.get("max_batch_size", ""),
        "action_horizon": server.get("action_horizon", ""),
        "max_steps": runtime.get("max_steps", ""),
        "num_trials_per_task": runtime.get("num_trials_per_task", ""),
    }

    try:
        from sims.libero.metrics import compute_server_timing_health  # noqa: PLC0415

        health = compute_server_timing_health(output_dir)
        if health is not None:
            summary.update(health)
    except Exception:  # noqa: BLE001
        pass

    return summary


def _spawn_subprocess_with_streaming(
    cmd: list[str],
    *,
    log_path: pathlib.Path,
    prefix: str,
) -> subprocess.Popen:
    """Popen a subprocess, tee-streaming its combined stdout to print() + a file.

    Modal captures the Python ``print`` stream for its log UI, so subprocess
    output has to flow through it to show up there. We also persist a copy on
    the artifacts volume.
    """
    import threading  # noqa: PLC0415

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=REMOTE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _tee() -> None:
        with log_path.open("w") as log_file:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(f"[{prefix}] {line}", end="", flush=True)
                log_file.write(line)
                log_file.flush()

    threading.Thread(target=_tee, daemon=True).start()
    return proc


def _wait_for_server_ready(proc: subprocess.Popen, port: int, timeout_s: int) -> None:
    """Poll the server's /metadata endpoint until it answers or the process dies."""
    import time as _time

    import requests  # noqa: PLC0415

    start = _time.time()
    deadline = start + timeout_s
    last_print = 0.0
    while _time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming ready")
        try:
            requests.get(f"http://127.0.0.1:{port}/metadata", timeout=2).raise_for_status()
            print(
                f"[ready-probe] /metadata responded after {_time.time() - start:.1f}s", flush=True
            )
            return
        except Exception:  # noqa: BLE001
            now = _time.time()
            if now - last_print > 30:
                print(
                    f"[ready-probe] still waiting for /metadata on :{port} "
                    f"(elapsed {now - start:.0f}s / budget {timeout_s}s)",
                    flush=True,
                )
                last_print = now
            _time.sleep(2)
    raise TimeoutError(f"server failed to respond on port {port} within {timeout_s}s")


def _failure_summary(case: SweepCase, error: str) -> dict[str, Any]:
    return {
        "run_id": case.run_id,
        "scheduler": case.scheduler,
        "experiment_config": case.experiment_config,
        "num_robots": case.num_robots,
        "seed": case.seed,
        "status": "failed",
        "error": error,
    }


@app.function(
    image=gpu_server_image,
    timeout=2 * 60 * 60,
    cpu=4,
    memory=16384,
    gpu=GPU,
    region=REGION,
    max_containers=MAX_CONCURRENT_CASES,
    volumes={
        str(REMOTE_ARTIFACTS_ROOT): artifacts_volume,
        CHECKPOINT_VOLUME_PATH: checkpoint_volume,
    },
)
def run_server(
    case: SweepCase,
    *,
    server_config: str,
    port: int,
    stamp: str,
    max_batch_size_override: int | None,
    urls: modal.Dict,
    shutdown: modal.Dict,
) -> dict[str, Any]:
    """L40S container: start the policy server, forward its port, wait for client."""
    import time as _time

    print(f"[server {case.run_id}] container started, port={port}", flush=True)

    srv_cfg: dict[str, Any] = json.loads((REMOTE_ROOT / server_config).read_text())
    if max_batch_size_override is not None:
        srv_cfg["max_batch_size"] = max_batch_size_override

    run_dir = REMOTE_OUTPUT_ROOT / case.run_id
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "case.json").write_text(json.dumps(dataclasses.asdict(case), indent=2))
    (run_dir / "server_config.json").write_text(json.dumps(srv_cfg, indent=2))

    server_cmd = _build_server_cmd(srv_cfg, port=port, scheduler=case.scheduler)
    (run_dir / "server_command.json").write_text(
        json.dumps({"argv": server_cmd, "shell": shlex.join(server_cmd)}, indent=2)
    )
    print(f"[server {case.run_id}] launching: {shlex.join(server_cmd)}", flush=True)

    status = "ok"
    error: str | None = None
    proc: subprocess.Popen | None = None
    try:
        proc = _spawn_subprocess_with_streaming(
            server_cmd, log_path=log_dir / "server.log", prefix=f"srv:{case.run_id}"
        )
        try:
            _wait_for_server_ready(proc, port, SERVER_READY_TIMEOUT_S)
            with modal.forward(port, unencrypted=True) as tunnel:
                host, fport = tunnel.tcp_socket
                print(
                    f"[server {case.run_id}] tunnel up -> {host}:{fport}; "
                    f"publishing URL and waiting for client",
                    flush=True,
                )
                urls[case.run_id] = (host, fport)
                # Block until the client signals it's done — or until the server
                # dies under us.
                while case.run_id not in shutdown:
                    if proc.poll() is not None:
                        error = f"server exited unexpectedly (code={proc.returncode})"
                        status = "failed"
                        print(f"[server {case.run_id}] {error}", flush=True)
                        break
                    _time.sleep(2)
                else:
                    print(f"[server {case.run_id}] shutdown signal received", flush=True)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        status = "failed"
        print(f"[server {case.run_id}] FAILED: {error}", flush=True)
        # Poison the URL slot so the client orchestrator doesn't hang.
        urls[case.run_id] = ("", 0)

    dest = REMOTE_ARTIFACTS_ROOT / stamp / case.run_id
    _copy_run_dir(run_dir, dest)
    artifacts_volume.commit()
    print(f"[server {case.run_id}] exiting (status={status})", flush=True)

    return {"run_id": case.run_id, "status": status, "error": error}


@app.function(
    image=gpu_libero_client_image,
    timeout=2 * 60 * 60,
    cpu=CLIENT_CPU,
    memory=16384,
    gpu=CLIENT_GPU,
    region=REGION,
    max_containers=MAX_CONCURRENT_CASES,
    volumes={str(REMOTE_ARTIFACTS_ROOT): artifacts_volume},
)
def run_client(
    case: SweepCase,
    *,
    server_host: str,
    server_port: int,
    stamp: str,
    max_steps_override: int | None,
    shutdown: modal.Dict,
) -> dict[str, Any]:
    """CPU container: connect to the server tunnel, run the LIBERO client."""
    print(
        f"[client {case.run_id}] container started, connecting to {server_host}:{server_port}",
        flush=True,
    )
    try:
        exp_cfg: dict[str, Any] = json.loads((REMOTE_ROOT / case.experiment_config).read_text())
        if max_steps_override is not None:
            exp_cfg["experiment"]["max_steps"] = max_steps_override
        exp_cfg = _expand_experiment_config(exp_cfg, case.num_robots)
        max_steps = int(exp_cfg["experiment"]["max_steps"])

        run_dir = REMOTE_OUTPUT_ROOT / case.run_id
        output_dir = run_dir / "output"
        log_dir = run_dir / "logs"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        saved_exp_config = run_dir / "experiment_config.json"
        saved_exp_config.write_text(json.dumps(exp_cfg, indent=2))

        client_cmd = _build_client_cmd(
            host=server_host,
            port=server_port,
            seed=case.seed,
            output_dir=output_dir,
            experiment_config_path=saved_exp_config,
            max_steps=max_steps,
        )
        (run_dir / "client_command.json").write_text(
            json.dumps({"argv": client_cmd, "shell": shlex.join(client_cmd)}, indent=2)
        )
        print(f"[client {case.run_id}] launching: {shlex.join(client_cmd)}", flush=True)

        try:
            client_proc = _spawn_subprocess_with_streaming(
                client_cmd,
                log_path=log_dir / "client.log",
                prefix=f"cli:{case.run_id}",
            )
            rc = client_proc.wait(timeout=60 * 75)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, client_cmd)
            summary = _summarize_run(output_dir, case)
            summary["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            summary = _failure_summary(case, repr(exc))

        dest = REMOTE_ARTIFACTS_ROOT / stamp / case.run_id
        _copy_run_dir(run_dir, dest)
        artifacts_volume.commit()
        summary["artifact_remote_path"] = str(dest)
        print(f"[client {case.run_id}] exiting (status={summary.get('status')})", flush=True)
        return summary
    finally:
        # Always release the server, even if the client crashed.
        shutdown[case.run_id] = True


def _write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


@app.local_entrypoint()
def main(
    # schedulers: str = "max-batch,greedy-deadline,lookahead-actions",
    schedulers: str = "greedy-deadline",
    experiment_configs: str = "configs/experiments/mock/short.json",
    # num_robots: str = "1,2,3,4,5",
    num_robots: str = "10",
    server_config: str = "configs/server/l40s_libero_pi05.json",
    seeds: str = "7",
    output_dir: str = "experiments/sweeps/l40s",
    max_batch_size: int | None = None,
    max_steps: int | None = 150,
) -> None:
    """Run the Cartesian product of schedulers, experiment_configs, num_robots, and seeds."""
    import time as _time  # noqa: PLC0415
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    artifacts_dir = out / "artifacts"
    cases = [
        SweepCase(
            scheduler=scheduler,
            experiment_config=cfg,
            num_robots=_config_num_robots(cfg, n),
            seed=seed,
        )
        for scheduler in _parse_csv(schedulers)
        for cfg in _parse_csv(experiment_configs)
        for n in _parse_csv(num_robots, cast=int)
        for seed in _parse_csv(seeds, cast=int)
    ]
    if not cases:
        raise ValueError("sweep is empty (check schedulers/experiment_configs/num_robots/seeds)")

    # Client CPU is fixed at module load (Modal 1.x can't vary cpu= per spawn).
    # Bail out early if the sweep needs more cpus than CLIENT_CPU was sized for.
    max_robots = max(case.num_robots for case in cases)
    required_cpu = max_robots + 2
    if required_cpu > CLIENT_CPU:
        raise ValueError(
            f"Sweep needs at least {required_cpu} client cpus "
            f"(max num_robots={max_robots} + 2) but CLIENT_CPU={CLIENT_CPU}. "
            "Bump CLIENT_CPU at the top of this script."
        )
    print(f"Sweeping {len(cases)} cases | client cpu={CLIENT_CPU} (required >= {required_cpu})")

    rows: list[dict[str, Any]] = []
    with modal.Dict.ephemeral() as urls, modal.Dict.ephemeral() as shutdown:
        # Stagger ports so two accidentally-colocated cases can't collide.
        case_ports = {case.run_id: BASE_PORT + i for i, case in enumerate(cases)}
        case_start_time: dict[str, float] = {}

        server_handles = {}
        for case in cases:
            server_handles[case.run_id] = run_server.spawn(
                case,
                server_config=server_config,
                port=case_ports[case.run_id],
                stamp=stamp,
                max_batch_size_override=max_batch_size,
                urls=urls,
                shutdown=shutdown,
            )
            case_start_time[case.run_id] = _time.time()
        print(
            f"Spawned {len(server_handles)} server functions; orchestrating clients as URLs land",
            flush=True,
        )

        def _orchestrate(case: SweepCase) -> dict[str, Any]:
            handle = server_handles[case.run_id]
            # No fixed deadline: with a 10-GPU concurrency cap a queued server
            # may not start for tens of minutes. Instead, poll the server
            # function handle so we can detect "function completed without
            # publishing URL" and bail.
            poll_interval = 5
            print(f"[orch {case.run_id}] waiting for server tunnel URL", flush=True)
            last_print = _time.time()
            while case.run_id not in urls:
                try:
                    server_result = handle.get(timeout=0)
                    # Server returned without publishing — treat as failure.
                    return _failure_summary(
                        case,
                        f"server completed without URL: {server_result!r}",
                    )
                except TimeoutError:
                    # Modal's poll_function raises builtin TimeoutError when the
                    # call hasn't completed yet (queued or in flight) — keep waiting.
                    pass
                except Exception as exc:  # noqa: BLE001
                    return _failure_summary(case, f"server function raised: {exc!r}")
                now = _time.time()
                if now - last_print > 60:
                    print(
                        f"[orch {case.run_id}] still waiting for URL "
                        f"(elapsed {now - case_start_time[case.run_id]:.0f}s)",
                        flush=True,
                    )
                    last_print = now
                _time.sleep(poll_interval)

            server_host, server_port = urls[case.run_id]
            if not server_host:
                shutdown[case.run_id] = True
                return _failure_summary(case, "server failed to start (poison URL)")

            print(
                f"[orch {case.run_id}] server up at {server_host}:{server_port}; spawning client",
                flush=True,
            )
            try:
                return run_client.remote(
                    case,
                    server_host=server_host,
                    server_port=server_port,
                    stamp=stamp,
                    max_steps_override=max_steps,
                    shutdown=shutdown,
                )
            except Exception as exc:  # noqa: BLE001
                shutdown[case.run_id] = True
                return _failure_summary(case, f"client crashed: {exc!r}")

        with ThreadPoolExecutor(max_workers=len(cases)) as ex:
            futures = {ex.submit(_orchestrate, case): case for case in cases}
            for fut in as_completed(futures):
                result = fut.result()
                rows.append(result)
                msg = (
                    f"{result['status']}: {result['run_id']} "
                    f"starvation={_safe_float(result.get('starvation_rate')):.3f}"
                )
                if result.get("status") != "ok":
                    msg += f" error={result.get('error')!r}"
                print(msg, flush=True)

        # Drain server handles. Shutdown signals were already posted; this just
        # surfaces server-side exceptions if any.
        for run_id, handle in server_handles.items():
            try:
                server_result = handle.get(timeout=300)
                if server_result.get("status") != "ok":
                    print(
                        f"  server {run_id}: status={server_result.get('status')} "
                        f"error={server_result.get('error')!r}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"  server {run_id} cleanup failed: {exc!r}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts from volume '{ARTIFACTS_VOLUME_NAME}/{stamp}' -> {artifacts_dir}")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            ARTIFACTS_VOLUME_NAME,
            stamp,
            str(artifacts_dir),
            "--force",
        ],
        check=True,
    )
    for row in rows:
        row["artifact_path"] = str(artifacts_dir / stamp / row["run_id"])

    sweep_csv = out / f"sweep_results_{stamp}.csv"
    latest_csv = out / "sweep_results.csv"
    _write_rows(sweep_csv, rows)
    _write_rows(latest_csv, rows)
    print(f"Wrote {latest_csv}")
    print(f"Wrote {sweep_csv}")

    suspicious = [r for r in rows if r.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for r in suspicious:
            print(f"  {r['run_id']}: {r.get('timing_flags', '')}")

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from plot_starvation_sweep import DEFAULT_METRICS, plot_results  # noqa: PLC0415

    timing_metrics = [
        "step_interval_p95_ms",
        "inference_p99_ms",
        "inbound_p95_ms",
        "outbound_p95_ms",
    ]
    # Stamp parallels artifacts/<stamp>/ so a re-run never overwrites a prior
    # plot set; the CSV next to it (sweep_results_<stamp>.csv) is the inputs.
    plots_dir = out / "plots" / stamp
    plot_results(latest_csv, plots_dir, metrics=list(DEFAULT_METRICS) + timing_metrics)
