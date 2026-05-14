"""Run server/client scheduler sweeps on Modal (mock CPU policy + mock client).

Uses the MOCK setup: a mock policy server and a mock client run as two
subprocesses inside a single CPU container, one container per case. The GPU
sibling is ``modal_starvation_sweep.py``.

Example:
    modal run scripts/experiments/modal_starvation_sweep_mock.py \
        --schedulers fixed-max-batch,greedy-deadline,round-robin \
        --experiment-configs configs/client/mock/short.json \
        --num-robots 1,2,3,4,5,6,7,8,9,10 \
        --server-config configs/server/mock.json \
        --seeds 7,42 \
        --output-dir experiments/sweeps/big_mock
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
from _setups import ARTIFACTS_VOLUME_NAME, MOCK, Case, app  # noqa: E402


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
    max_steps: int | None,
) -> Case:
    """Resolve a SweepCase into the serve.Args + run_libero.Args the setup runs."""
    srv_cfg = json.loads(pathlib.Path(server_config).read_text())
    exp_cfg = _expand_experiment_config(
        json.loads(pathlib.Path(case.experiment_config).read_text()), case.num_robots
    )
    if max_steps is not None:
        exp_cfg["experiment"]["max_steps"] = max_steps
    run_dir = output_dir / "runs" / case.run_id

    if srv_cfg.get("policy_type", "default") == "mock":
        policy: Any = serve.Mock(**srv_cfg.get("policy", {}))
    else:
        policy = serve.Default()
    server_args = serve.Args(
        env=serve.EnvMode[srv_cfg.get("env", "LIBERO")],
        max_batch_size=max_batch_size
        if max_batch_size is not None
        else srv_cfg.get("max_batch_size", 1),
        scheduling_algorithm=case.scheduler,
        policy=policy,
        log_dir=str(run_dir / "server_logs"),
    )

    client_kwargs: dict[str, Any] = dict(
        env="mock",
        overwrite=True,
        progress_type="logging",
        seed=case.seed,
        output_dir=run_dir / "output",
        log_dir=run_dir / "client_logs",
    )
    if max_steps is not None:
        client_kwargs["max_steps"] = max_steps
    return Case(
        run_id=case.run_id,
        run_dir=run_dir,
        server_args=server_args,
        client_args=run_libero.Args(**client_kwargs),
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
    schedulers: str = "fixed-max-batch,greedy-deadline,round-robin",
    experiment_configs: str = "configs/client/mock/short.json",
    num_robots: str = "2,4,6",
    server_config: str = "configs/server/mock.json",
    seeds: str = "7",
    output_dir: str = "experiments/sweeps/mock",
    max_batch_size: int | None = None,
    max_steps: int | None = None,
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

    rows: list[dict[str, Any]] = []
    for result in MOCK.run(built, stamp=stamp):
        row = {**case_meta[result["run_id"]], **result}
        rows.append(row)
        print(
            f"{row['status']}: {row['run_id']} "
            f"starvation={_safe_float(row.get('starvation_rate')):.3f}",
            flush=True,
        )

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
