"""Unified Modal sweep entrypoint for scheduler experiments.

Examples:
    uv run modal run scripts/experiments/modal_sweep.py \
        --server-config configs/server/mock.json \
        --client-config configs/client/mock/short.json \
        --schedulers max-batch,dynamic-action \
        --num-robots 2,4,6 \
        --seeds 7 \
        --output-dir experiments/sweeps/mock

    uv run modal run scripts/experiments/modal_sweep.py \
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

import dataclasses
import datetime as dt
import pathlib
import sys
from typing import Any

import modal

# TODO: kind of ugly, is this necessary?
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # _setups, plot modules
sys.path.insert(0, str(_HERE.parent))  # serve, run_libero

import run_libero  # noqa: E402
import serve  # noqa: E402
from _setups import LIBERO, MOCK, Case, app  # noqa: E402
from _utils import download_artifacts, write_rows  # noqa: E402


def _make_cases(
    *,
    server_args: serve.Args,
    client_args: run_libero.Args,
    schedulers: list[str],
    num_robots_list: list[int],
    seeds: list[int],
    max_batch_size: list[int],
    alpha: list[float],
    stamp: str,
) -> list[Case]:
    cases: list[Case] = []

    for seed in seeds:
        for scheduler in schedulers:
            for num_robots in num_robots_list:
                server_args_copy = dataclasses.replace(
                    server_args, seed=seed, scheduling_algorithm=scheduler
                )
                client_args_copy = dataclasses.replace(
                    client_args, seed=seed, num_robots=num_robots
                )

                # NOTE: only supports sweeping one of these
                if len(max_batch_size) > 0:
                    for max_batch_size in max_batch_size:
                        server_args_copy = dataclasses.replace(
                            server_args_copy, max_batch_size=max_batch_size
                        )
                        cases.append(
                            Case(
                                server_args=server_args_copy,
                                client_args=client_args_copy,
                                stamp=stamp,
                            )
                        )
                elif len(alpha) > 0:
                    for alpha in alpha:
                        server_args_copy = dataclasses.replace(server_args_copy, alpha=alpha)
                        cases.append(
                            Case(
                                server_args=server_args_copy,
                                client_args=client_args_copy,
                                stamp=stamp,
                            )
                        )
                else:
                    cases.append(
                        Case(
                            server_args=server_args_copy,
                            client_args=client_args_copy,
                            stamp=stamp,
                        )
                    )

    return cases


def get_worker(gpu: str) -> modal.Cls:
    """``mock`` colocates server + client on CPU; any GPU name uses the split setup."""
    return MOCK if gpu.lower() == "mock" else LIBERO


def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


@app.local_entrypoint()
def main(
    server_config: str = "configs/server/mock.json",
    client_config: str = "configs/client/mock/short.json",
    output_dir: str = "experiments/sweeps/modal",
    schedulers: str = "max-batch,dynamic-action",
    num_robots: str = "2,4,6,8,10",
    seeds: str = "7",
    max_batch_size: str = "",
    alpha: str = "",
    gpu: str = "mock",
) -> None:
    """Run scheduler sweeps on Modal.

    ``gpu`` is the single knob that picks mock vs. real: ``mock`` runs the mock
    policy + mock client colocated on CPU; any Modal GPU name (``l40s``,
    ``h100``, ...) runs the real ``default`` policy server on that GPU with a
    LIBERO client. It overrides whatever ``policy``/``env`` the config files set.
    """

    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    scheduler_list = parse_list_args(schedulers)
    seed_list = parse_list_args(seeds, cast=int)
    num_robots_list = parse_list_args(num_robots, cast=int)

    server_args = serve.Args.from_json(server_config)
    client_args = run_libero.Args.from_json(client_config)

    # `gpu` is the source of truth for mock vs. real; reconcile the configs to it.
    if gpu.lower() == "mock":
        policy = server_args.policy if isinstance(server_args.policy, serve.Mock) else serve.Mock()
        server_args = dataclasses.replace(server_args, policy=policy)
        client_args = dataclasses.replace(client_args, env="mock", progress_type="logging")
    else:
        server_args = dataclasses.replace(server_args, policy=serve.Default())
        client_args = dataclasses.replace(client_args, env="libero", progress_type="logging")

    cases = _make_cases(
        server_args=server_args,
        client_args=client_args,
        schedulers=scheduler_list,
        num_robots_list=num_robots_list,
        seeds=seed_list,
        max_batch_size=max_batch_size,
        alpha=alpha,
        stamp=stamp,
    )
    cases = [dataclasses.replace(case, gpu=gpu) for case in cases]

    worker = get_worker(gpu)
    rows: list[dict[str, Any]] = []
    for row in worker.run.map(cases, order_outputs=False):
        rows.append(row)
        sr = row.get("starvation_rate")
        sr_str = f"{sr:.3f}" if isinstance(sr, (int, float)) else "n/a"
        print(f"{row.get('status', '?')}: {row['run_id']} starvation={sr_str}")

    download_artifacts(stamp=stamp, out=out, rows=rows)
    write_rows(out / f"sweep_results_{stamp}.csv", rows)

    suspicious = [r for r in rows if r.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for row in suspicious:
            print(f"  {row['run_id']}: {row.get('timing_flags', '')}")

    # TODO: one function
    # plot(experiment, latest_csv, out, stamp)
