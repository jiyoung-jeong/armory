"""Unified Modal sweep entrypoint for scheduler experiments.

Examples:
    modal run scripts/experiments/modal_sweep.py \
        --server-config configs/server/mock.json \
        --client-config configs/client/mock/short.json \
        --schedulers fixed-max-batch,greedy-deadline,round-robin \
        --num-robots 2,4,6 \
        --seeds 7 \
        --output-dir experiments/sweeps/mock

    modal run scripts/experiments/modal_sweep.py \
        --server-config configs/server/gpu.json \
        --client-config configs/client/libero/short.json \
        --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action \
        --num-robots 2,4,6 \
        --seeds 7,42 \
        --max-batch-size 1,2,4 \
        --alpha 0.0,0.25,0.5,0.75,1.0 \
        --output-dir experiments/sweeps/gpu
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import pathlib
import re
import sys
from typing import Any

# TODO: kind of ugly, is this necessary?
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # _setups, plot modules
sys.path.insert(0, str(_HERE.parent))  # serve, run_libero

import run_libero  # noqa: E402
import serve  # noqa: E402
from _setups import MOCK, REAL_CPU, REAL_GPU, Case, app  # noqa: E402

# Modal's account-wide GPU cap is shared across server + client. Each real-gpu
# case consumes one L40S + one A10G simultaneously.
GPU_CAP = 10
DEFAULT_REAL_GPU_MAX_CONCURRENT = GPU_CAP // 2


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", value).strip("-")

@dataclasses.dataclass(frozen=True)
class SweepCase:
    server_args: serve.Args
    client_args: run_libero.Args

    @property
    def run_id(self) -> str:
        # TODO: make sure this uniquely identifies
        cfg_name = f"{self.config_index:02d}-{_safe_id(pathlib.Path(self.experiment_config).stem)}"
        parts = [
            f"scheduler={self.args.scheduling_algorithm}",
            f"num_robots={self.args.num_robots}",
            f"seed={self.args.seed}",
        ]
        return "__".join(parts)

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

def _make_cases(
    *,
    server_config: pathlib.Path,
    client_config: pathlib.Path,
    schedulers: list[str],
    max_batch_size: list[int],
    alpha: list[float],
    num_robots: list[int],
    seeds: list[int],
) -> tuple[list[SweepCase], dict[str, dict[str, Any]]]:
    cases: list[SweepCase] = []

    for seed in seeds:
        for scheduler in schedulers:
            for num_robots in num_robots:
                server_args = serve.Args.from_json(server_config)
                client_args = run_libero.Args.from_json(client_config)

                server_args.seed = seed
                client_args.seed = seed
                
                server_args.scheduling_algorithm = scheduler

                client_args.num_robots = num_robots

                if len(max_batch_size) > 1:

                cases.append(SweepCase(
                    server_args=server_args,
                    client_args=client_args,
                ))
            

    return cases

def _run_cases(
    *,
    setup: str,
    cases: list[SweepCase],
    max_concurrent: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # TODO: handle throttling behavior
    print(f"Sweeping {len(cases)} cases | setup={setup} max_concurrent={max_concurrent}")
    results = runner.run(
        built,
        max_concurrent=max_concurrent,
    )

    for result in results:
        row = {**meta[result["run_id"]], **result}
        rows.append(row)
        if row.get("mean_starvation") not in (None, ""):
            msg = (
                f"{row['status']}: {row['run_id']} "
                f"mean_starvation={float(row.get('mean_starvation')):.4f} "
                f"max_starvation={float(row.get('max_starvation')):.4f}"
            )
        else:
            msg = f"{row['status']}: {row['run_id']} starvation={float(row.get('starvation_rate')):.3f}"
        if row.get("status") != "ok":
            msg += f" error={row.get('error')!r}"
        print(msg, flush=True)
    return rows

def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]



@app.local_entrypoint()
def main(
    server_config: pathlib.Path = pathlib.Path("configs/server/mock.json"),
    client_config: pathlib.Path = pathlib.Path("configs/client/mock/short.json"),
    output_dir: pathlib.Path = pathlib.Path("experiments/sweeps/modal"),
    schedulers: str = "",
    num_robots: str = "2,4,6",
    seeds: str = "7",
    max_batch_size: list[int] = [],
    alpha: list[float] = [],
    max_concurrent: int | None = None,
) -> None:
    """Run scheduler sweeps on Modal.

    For ``experiment=alpha-fairness``, every path in ``experiment_configs`` is an
    explicit scenario config. No robot profiles are generated from shorthand.
    """

    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    scheduler_list = parse_list_args(schedulers)
    seed_list = parse_list_args(seeds, cast=int)
    robot_count_list = parse_list_args(num_robots, cast=int)

    cases = _make_cases(
        server_config=server_config,
        client_config=client_config,
        schedulers=scheduler_list,
        num_robots=robot_count_list,
        seeds=seed_list,
        max_batch_size=max_batch_size,
        alpha=alpha,
    )
    if not cases:
        raise ValueError("sweep is empty")

    print(
        f"Submitting {len(cases)} cases "
        f"(server_config={server_config} client_config={client_config} "
        f"schedulers={scheduler_list} seeds={seed_list})"
    )

    rows = _run_cases(
        cases=cases,
        stamp=stamp,
        max_concurrent=max_concurrent,
    )

    # TODO: fix import
    _download_artifacts(stamp=stamp, out=out, rows=rows)
    _write_rows(out / f"sweep_results_{stamp}.csv", rows)
    print(f"Wrote {sweep_csv}")

    suspicious = [r for r in rows if r.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for row in suspicious:
            print(f"  {row['run_id']}: {row.get('timing_flags', '')}")

    # TODO: one function
    _plot(experiment, latest_csv, out, stamp)
