"""Run server/client scheduler sweeps on Modal L40S GPUs (real PI05 + LIBERO).

Uses the REAL_GPU setup: a real policy server on an L40S and a LIBERO sim client
on a small GPU, one pair of containers per case. The mock-CPU sibling is
``modal_starvation_sweep_mock.py``.

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
import subprocess
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # _setups
sys.path.insert(0, str(_HERE.parent))  # serve, run_libero

import run_libero  # noqa: E402
import serve  # noqa: E402
from _setups import ARTIFACTS_VOLUME_NAME, REAL_GPU, Case, app  # noqa: E402

# Modal's account-wide GPU cap is shared across server + client. Each case
# consumes one L40S + one A10G simultaneously, so concurrent cases must be capped
# at GPU_CAP // 2 — otherwise a sweep can deadlock with all GPUs held by servers
# waiting for clients that can never be scheduled.
GPU_CAP = 10
MAX_CONCURRENT_CASES = GPU_CAP // 2


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _expand_experiment_config(exp_cfg: dict[str, Any], num_robots: int) -> dict[str, Any]:
    """Return a copy of exp_cfg with robot profiles set for num_robots robots.

    If the config already defines more than one robot explicitly, those profiles are
    used as-is and num_robots is ignored (the config is authoritative). Otherwise
    robot_0's profile is replicated to fill num_robots robots.
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


def _build_case(
    case: SweepCase,
    *,
    server_config: str,
    output_dir: pathlib.Path,
    max_batch_size: int | None,
    max_steps: int,
) -> Case:
    """Resolve a SweepCase into the serve.Args + run_libero.Args the setup runs."""
    srv_cfg = json.loads(pathlib.Path(server_config).read_text())
    exp_cfg = _expand_experiment_config(
        json.loads(pathlib.Path(case.experiment_config).read_text()), case.num_robots
    )
    run_dir = output_dir / "runs" / case.run_id

    server_kwargs: dict[str, Any] = dict(
        env=serve.EnvMode[srv_cfg.get("env", "LIBERO")],
        model=serve.ModelFamily[srv_cfg.get("model", "PI05")],
        max_batch_size=srv_cfg.get("max_batch_size", 1),
        scheduling_algorithm=case.scheduler,
        policy=serve.Default(),
        log_dir=str(run_dir / "server_logs"),
    )
    if max_batch_size is not None:
        server_kwargs["max_batch_size"] = max_batch_size
    if "num_steps" in srv_cfg:
        server_kwargs["num_steps"] = srv_cfg["num_steps"]

    client_args = run_libero.Args(
        env="libero",
        overwrite=True,
        progress_type="logging",
        max_steps=max_steps,
        seed=case.seed,
        output_dir=run_dir / "output",
        log_dir=run_dir / "client_logs",
    )
    return Case(
        run_id=case.run_id,
        run_dir=run_dir,
        server_args=serve.Args(**server_kwargs),
        client_args=client_args,
        experiment_config=exp_cfg,
    )


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
    schedulers: str = "greedy-deadline",
    experiment_configs: str = "configs/client/mock/short.json",
    num_robots: str = "10",
    server_config: str = "configs/server/l40s_libero_pi05.json",
    seeds: str = "7",
    output_dir: str = "experiments/sweeps/l40s",
    max_batch_size: int | None = None,
    max_steps: int = 150,
) -> None:
    """Run the Cartesian product of schedulers, experiment_configs, num_robots, and seeds."""
    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017

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

    built = [
        _build_case(
            case,
            server_config=server_config,
            output_dir=out,
            max_batch_size=max_batch_size,
            max_steps=max_steps,
        )
        for case in cases
    ]
    case_meta = {
        case.run_id: {
            "run_id": case.run_id,
            "scheduler": case.scheduler,
            "experiment_config": case.experiment_config,
            "num_robots": case.num_robots,
            "seed": case.seed,
        }
        for case in cases
    }

    # The LIBERO client needs roughly one cpu per robot plus headroom; with
    # with_options() we size the client container to the sweep.
    client_cpu = max(case.num_robots for case in cases) + 2
    print(f"Sweeping {len(cases)} cases | client cpu={client_cpu}")

    rows: list[dict[str, Any]] = []
    for result in REAL_GPU.run(
        built,
        stamp=stamp,
        max_concurrent=MAX_CONCURRENT_CASES,
        client_cpu=client_cpu,
    ):
        row = {**case_meta[result["run_id"]], **result}
        rows.append(row)
        msg = f"{row['status']}: {row['run_id']} starvation={_safe_float(row.get('starvation_rate')):.3f}"
        if row.get("status") != "ok":
            msg += f" error={row.get('error')!r}"
        print(msg, flush=True)

    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts from volume '{ARTIFACTS_VOLUME_NAME}/{stamp}' -> {artifacts_dir}")
    subprocess.run(
        ["modal", "volume", "get", ARTIFACTS_VOLUME_NAME, stamp, str(artifacts_dir), "--force"],
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
    # Stamp parallels artifacts/<stamp>/ so a re-run never overwrites a prior plot
    # set; the CSV next to it (sweep_results_<stamp>.csv) is the inputs.
    plots_dir = out / "plots" / stamp
    plot_results(latest_csv, plots_dir, metrics=list(DEFAULT_METRICS) + timing_metrics)
