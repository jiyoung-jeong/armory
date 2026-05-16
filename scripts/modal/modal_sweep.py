"""Modal scheduler sweep entrypoint.

Mirrors ``scripts/sbatch/launch_sweep.py``: same sweep axes (``--schedulers``,
``--num-robots``, ``--seeds``, ``--max-batch-size`` or ``--alpha``), same case
layout (one dir per case under ``<output_dir>/<stamp>/``), same downstream CSV
and plot pipeline. The setup (mock-colocated vs. split GPU+CPU) is picked from
the server policy and client env in the configs — no GPU flag.

Examples:
    # Mock policy + mock env: server + client colocated on one CPU container.
    uv run modal run scripts/modal/modal_sweep.py \\
        --server-config configs/server/mock.json \\
        --client-config configs/client/mock/short.json \\
        --server-policy config \\
        --schedulers max-batch,dynamic-action \\
        --num-robots 2,4 \\
        --seeds 7 \\
        --output-dir experiments/sweeps/modal_mock

    # Real default policy on GPU + mock-env client on CPU (split, two containers).
    uv run modal run scripts/modal/modal_sweep.py \\
        --server-config configs/server/gpu.json \\
        --client-config configs/client/mock/half_fast_half_slow/10.json \\
        --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action \\
        --num-robots 10 \\
        --seeds 7,42 \\
        --alpha 0.0,0.25,0.5,0.75,1.0 \\
        --output-dir experiments/sweeps/mock
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # _setups, _utils
sys.path.insert(0, str(_HERE.parent))  # serve, run_libero
sys.path.insert(0, str(_HERE.parent / "visualization"))  # plot_sweep

import run_libero  # noqa: E402
import serve  # noqa: E402
from _setups import Case, app, select_setup  # noqa: E402
from _utils import download_artifacts, write_rows  # noqa: E402


def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _make_cases(
    *,
    server_args: serve.Args,
    client_args: run_libero.Args,
    schedulers: list[str],
    num_robots_list: list[int],
    seeds: list[int],
    max_batch_sizes: list[int],
    alphas: list[float],
    stamp: str,
) -> list[Case]:
    if max_batch_sizes and alphas:
        raise SystemExit("Sweep only one of --max-batch-size or --alpha at a time.")
    if not max_batch_sizes:
        max_batch_sizes = [server_args.max_batch_size]
    if not alphas:
        alphas = [server_args.alpha]

    cases: list[Case] = []
    for seed in seeds:
        for scheduler in schedulers:
            for num_robots in num_robots_list:
                for max_batch_size in max_batch_sizes:
                    for alpha in alphas:
                        server = dataclasses.replace(
                            server_args,
                            seed=seed,
                            scheduling_algorithm=scheduler,
                            max_batch_size=max_batch_size,
                            alpha=alpha,
                        )
                        client = dataclasses.replace(
                            client_args,
                            seed=seed,
                            num_robots=num_robots,
                            overwrite=True,
                        )
                        cases.append(Case(server_args=server, client_args=client, stamp=stamp))
    return cases


def _case_row(case: Case) -> dict[str, Any]:
    return {
        "stamp": case.stamp,
        "run_id": case.run_id,
        "scheduler": case.server_args.scheduling_algorithm,
        "num_robots": case.client_args.num_robots,
        "seed": case.client_args.seed,
        "max_batch_size": case.server_args.max_batch_size,
        "alpha": case.server_args.alpha,
    }


@app.local_entrypoint()
def main(
    server_config: str = "configs/server/mock.json",
    client_config: str = "",
    output_dir: str = "experiments/sweeps/modal",
    schedulers: str = "fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action",
    num_robots: str = "2,4,6,8,10",
    seeds: str = "7",
    max_batch_size: str = "",
    alpha: str = "",
    server_policy: str = "default",
) -> None:
    """Submit a scheduler sweep on Modal.

    ``server_policy='default'`` rewrites the server config's ``policy`` to a real
    ``Default()`` checkpoint (matches sbatch's behavior). Pass ``--server-policy
    config`` to preserve the policy as written in the JSON (useful for mock runs).

    The setup is auto-picked: mock policy + mock env runs colocated on one CPU
    container; everything else runs split (server + client in their own containers,
    bridged by a Modal-forwarded TCP tunnel).
    """
    if not client_config:
        raise SystemExit("--client-config is required.")
    if server_policy not in {"default", "config"}:
        raise SystemExit("--server-policy must be 'default' or 'config'.")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    run_root = pathlib.Path(output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    server_args = serve.Args.from_json(server_config)
    client_args = run_libero.Args.from_json(client_config)
    if server_policy == "default":
        server_args = dataclasses.replace(server_args, policy=serve.Default())

    cases = _make_cases(
        server_args=server_args,
        client_args=client_args,
        schedulers=parse_list_args(schedulers),
        num_robots_list=parse_list_args(num_robots, cast=int),
        seeds=parse_list_args(seeds, cast=int),
        max_batch_sizes=parse_list_args(max_batch_size, cast=int),
        alphas=parse_list_args(alpha, cast=float),
        stamp=stamp,
    )
    if not cases:
        raise SystemExit("No cases produced; check sweep arguments.")

    # Setup is determined by policy + env, neither of which varies per case.
    worker = select_setup(cases[0])
    print(f"Running {len(cases)} case(s) on setup={type(worker).__name__}")

    case_rows = [_case_row(case) for case in cases]
    write_rows(run_root / f"cases_{stamp}.csv", case_rows)

    rows: list[dict[str, Any]] = []
    for row in worker.run.map(cases, order_outputs=False):
        rows.append(row)
        sr = row.get("starvation_rate")
        sr_str = f"{sr:.3f}" if isinstance(sr, (int, float)) else "n/a"
        print(f"{row.get('status', '?')}: {row['run_id']} starvation={sr_str}")

    download_artifacts(stamp=stamp, out=run_root, rows=rows)
    results_csv = run_root / f"sweep_results_{stamp}.csv"
    write_rows(results_csv, rows)

    suspicious = [r for r in rows if r.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for row in suspicious:
            print(f"  {row['run_id']}: {row.get('timing_flags', '')}")

    try:
        from plot_sweep import plot_results  # noqa: PLC0415

        plot_results(results_csv, run_root / "plots")
    except Exception as exc:  # noqa: BLE001
        print(f"Plot generation skipped: {exc!r}")
