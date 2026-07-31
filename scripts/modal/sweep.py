"""Modal scheduler sweep: one Modal case per (scheduler x seed x fleet x ...) combo.

Mirrors ``scripts/sbatch/launch_sweep.py``. Scheduler and server axes are swept
by flags; client fleet shapes come from one ``--client-config`` file or every
JSON/JSONC file under a config directory. ``--mode`` picks where it all runs
(see ``app.py``); only ``gpu`` and ``mock`` make sense here, since a sweep with
no server has no scheduler to sweep.

Examples:
    # Everything on CPU with the timing-faithful mock policy. Start here.
    uv run modal run scripts/modal/sweep.py \\
        --mode mock \\
        --server-config configs/server/mock.json \\
        --client-config configs/client/mock/short.json \\
        --schedulers max-batch,dynamic-action \\
        --seeds 7 \\
        --output-dir experiments/sweeps/modal_mock

    # Real policy on an L40S driving LIBERO fleets on T4s.
    uv run modal run scripts/modal/sweep.py \\
        --mode gpu \\
        --server-config configs/server/libero.json \\
        --client-config configs/modal_sweep/half_fast_half_slow \\
        --schedulers max-batch,greedy-deadline,round-robin,lookahead-actions \\
        --seeds 7,42 \\
        --alpha 0.0,0.5,1.0 \\
        --output-dir experiments/sweeps/libero

    # Paper-style LIBERO sweep. The multiplier axis applies only to the
    # shortest horizon of lookahead-actions; baselines still run once each.
    uv run modal run scripts/modal/sweep.py \
        --mode gpu \
        --server-config configs/server/libero.json \
        --client-config configs/modal_sweep \
        --schedulers round-robin,max-batch,lookahead-actions \
        --seeds 1,2,3 \
        --max-batch-size 3 \
        --action-horizon-multiplier 1,3,5 \
        --server-pool-size 6 \
        --server-start-timeout-minutes 60 \
        --stamp libero_5min_paper \
        --output-dir experiments/sweeps/libero_5min
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import sys
from collections.abc import Iterable
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # repo root
# Order matters: src first so bare `utils` -> src/utils.py (not the shadowing
# scripts/utils.py), then repo root for `scripts.*`, then scripts/ for `import
# serve` and visualization/ for plot_sweep.
sys.path[:0] = [
    str(_ROOT / "src"),
    str(_ROOT),
    str(_HERE.parent),
    str(_HERE.parent / "visualization"),
]

import serve  # noqa: E402
from scripts.modal.app import (  # noqa: E402
    SERVER_START_TIMEOUT_S,
    CaseRunner,
    PooledGpuServer,
    app,
    server_reuse_key,
)
from scripts.modal.images import REMOTE_ROOT  # noqa: E402
from scripts.modal.utils import download_artifacts, write_rows  # noqa: E402

from evaluation.types import ExperimentConfig  # noqa: E402

ALPHA_SWEEP_SCHEDULERS = {"dynamic-action", "lookahead-actions"}
SERVER_CONFIG_SWEEP_SCHEDULERS = {"lookahead-actions"}
ACTION_HORIZON_SWEEP_SCHEDULERS = {"lookahead-actions"}
MAX_SERVER_POOL_SIZE = 6


@dataclasses.dataclass(frozen=True)
class Case:
    """One point in the sweep. Stays local: only ``payload`` crosses to Modal."""

    server: serve.Args  # embeds the SchedulerConfig the client also reconfigures with
    experiment: ExperimentConfig
    experiment_name: str
    stamp: str
    mode: str
    stream_logs: bool
    server_variant: str = ""
    action_horizon_multiplier: float | None = None
    server_start_timeout_s: float = SERVER_START_TIMEOUT_S

    @property
    def run_id(self) -> str:
        parts = [
            f"scheduler={self.server.scheduler.scheduling_algorithm}",
            f"experiment={self.experiment_name}",
            f"num_robots={len(self.experiment.robots)}",
            f"seed={self.experiment.seed}",
            f"max_batch_size={self.server.max_batch_size}",
            f"alpha={self.server.scheduler.alpha}",
        ]
        if self.server_variant:
            parts.append(f"server_variant={self.server_variant}")
        if self.action_horizon_multiplier is not None:
            parts.append(f"ahm={self.action_horizon_multiplier}")
        return "__".join(parts)

    def payload(self) -> dict[str, Any]:
        """Flatten to the plain JSON ``app.launch`` takes.

        Modal would have to import ``serve`` on the orchestrator container to
        unpickle ``serve.Args``, and that bare module isn't importable there --
        hence dicts, not models.

        Run path is separate from the artifact path because Modal Volumes don't
        love many small writes; runs write hot to container disk and copy at the end.
        """
        return {
            "mode": self.mode,
            "run_id": self.run_id,
            "run_dir": str(REMOTE_ROOT / self.stamp / self.run_id),
            "server_config": self.server.model_dump(mode="json"),
            "client_config": {
                "experiment_config": self.experiment.model_dump(mode="json"),
                "scheduler_config": self.server.scheduler.model_dump(mode="json"),
            },
            "stream_logs": self.stream_logs,
            "server_start_timeout_s": self.server_start_timeout_s,
        }

    def row(self) -> dict[str, Any]:
        multipliers = self.server.scheduler.action_horizon_multipliers
        scheduler_variant = self.server.scheduler.scheduling_algorithm
        if self.action_horizon_multiplier is not None:
            scheduler_variant += f" (ahm={self.action_horizon_multiplier})"
        return {
            "stamp": self.stamp,
            "run_id": self.run_id,
            "scheduler": self.server.scheduler.scheduling_algorithm,
            "scheduler_variant": scheduler_variant,
            "server_variant": self.server_variant,
            "experiment": self.experiment_name,
            "num_robots": len(self.experiment.robots),
            "seed": self.experiment.seed,
            "max_batch_size": self.server.max_batch_size,
            "alpha": self.server.scheduler.alpha,
            "action_horizon_multiplier": (
                self.action_horizon_multiplier if self.action_horizon_multiplier is not None else ""
            ),
            "action_horizon_multipliers": dict(multipliers) if multipliers else "",
        }


def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _client_config_paths(path: str) -> list[pathlib.Path]:
    candidate = pathlib.Path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        paths = sorted([*candidate.rglob("*.json"), *candidate.rglob("*.jsonc")])
        if paths:
            return paths
    raise SystemExit(f"--client-config must be a JSON/JSONC file or directory: {path}")


def _server_config_paths(value: str) -> list[pathlib.Path]:
    paths = [pathlib.Path(item) for item in parse_list_args(value)]
    missing = [str(path) for path in paths if not path.is_file()]
    if not paths or missing:
        raise SystemExit(f"--server-config file(s) not found: {', '.join(missing) or value}")
    return paths


def _experiment_name(path: pathlib.Path, *, root: pathlib.Path | None = None) -> str:
    rel = path.relative_to(root) if root is not None else pathlib.Path(path.name)
    return str(rel.with_suffix("")).replace("/", "_")


def _override_shortest_horizon(base: dict[int, float], multiplier: float) -> dict[int, float]:
    """Return ``base`` with only its numerically shortest horizon overridden."""
    if not base:
        raise SystemExit(
            "--action-horizon-multiplier requires a non-empty "
            "scheduler.action_horizon_multipliers mapping in --server-config."
        )
    result = dict(base)
    shortest_horizon = min(result)
    result[shortest_horizon] = multiplier
    return result


def _validate_horizon_multipliers(
    scheduler: str,
    multipliers: dict[int, float],
    experiment: ExperimentConfig,
    *,
    experiment_name: str,
) -> None:
    """Fail locally instead of letting lookahead crash remotely on a missing key."""
    if scheduler not in ACTION_HORIZON_SWEEP_SCHEDULERS:
        return
    required = {robot.execution_horizon.max for robot in experiment.robots}
    missing = sorted(required - multipliers.keys())
    if missing:
        raise SystemExit(
            f"lookahead-actions is missing action-horizon multiplier(s) for {missing} "
            f"required by client config {experiment_name!r}."
        )


def _make_cases(
    *,
    server_variants: list[tuple[str, serve.Args]],
    experiment_configs: list[tuple[str, ExperimentConfig]],
    schedulers: list[str],
    seeds: list[int],
    max_batch_sizes: list[int],
    alphas: list[float],
    action_horizon_multipliers: list[float] | None = None,
    mode: str,
    stream_logs: bool,
    stamp: str,
    server_start_timeout_s: float = SERVER_START_TIMEOUT_S,
) -> list[Case]:
    if not max_batch_sizes:
        max_batch_sizes = [server_variants[0][1].max_batch_size]
    if not alphas:
        alphas = [server_variants[0][1].scheduler.alpha]
    action_horizon_multipliers = action_horizon_multipliers or []

    cases: list[Case] = []
    for seed in seeds:
        for scheduler in schedulers:
            # Only config-sensitive schedulers are re-run per server config;
            # ordinary baselines run once, against the first one.
            variants = (
                server_variants
                if scheduler in SERVER_CONFIG_SWEEP_SCHEDULERS
                else [("", server_variants[0][1])]
            )
            scheduler_alphas = (
                alphas if scheduler in ALPHA_SWEEP_SCHEDULERS else [variants[0][1].scheduler.alpha]
            )
            scheduler_horizon_multipliers: list[float | None] = (
                action_horizon_multipliers
                if action_horizon_multipliers and scheduler in ACTION_HORIZON_SWEEP_SCHEDULERS
                else [None]
            )
            for variant, server_args in variants:
                for experiment_name, experiment_config in experiment_configs:
                    for max_batch_size in max_batch_sizes:
                        for alpha in scheduler_alphas:
                            for horizon_multiplier in scheduler_horizon_multipliers:
                                multipliers = dict(server_args.scheduler.action_horizon_multipliers)
                                if horizon_multiplier is not None:
                                    multipliers = _override_shortest_horizon(
                                        multipliers, horizon_multiplier
                                    )
                                _validate_horizon_multipliers(
                                    scheduler,
                                    multipliers,
                                    experiment_config,
                                    experiment_name=experiment_name,
                                )

                                # One SchedulerConfig drives both boot (server, for the
                                # boot-only alpha) and runtime reconfigure (client).
                                sched = server_args.scheduler.model_copy(
                                    update={
                                        "scheduling_algorithm": scheduler,
                                        "alpha": alpha,
                                        "action_horizon_multipliers": multipliers,
                                    }
                                )
                                cases.append(
                                    Case(
                                        server=server_args.model_copy(
                                            update={
                                                "scheduler": sched,
                                                "max_batch_size": max_batch_size,
                                                "seed": seed,
                                            }
                                        ),
                                        experiment=experiment_config.model_copy(
                                            update={"seed": seed}
                                        ),
                                        experiment_name=experiment_name,
                                        stamp=stamp,
                                        mode=mode,
                                        stream_logs=stream_logs,
                                        server_variant=variant,
                                        action_horizon_multiplier=horizon_multiplier,
                                        server_start_timeout_s=server_start_timeout_s,
                                    )
                                )
    return cases


def _run_case_payloads(
    payloads: list[dict[str, Any]], *, max_concurrent_cases: int
) -> Iterable[dict[str, Any]]:
    """Map cases while bounding how many can spawn child server/client workers.

    Inputs beyond ``max_concurrent_cases`` remain queued at the CaseRunner
    boundary. They therefore have not requested an L40S/T4 and have not started
    the application-level server-allocation deadline in ``app.launch``.
    """
    if max_concurrent_cases < 1:
        raise SystemExit("--max-concurrent-cases must be positive.")
    runner = CaseRunner.with_options(max_containers=max_concurrent_cases)()
    return runner.run.map(payloads, order_outputs=False)


def _split_evenly(items: list[Case], count: int) -> list[list[Case]]:
    size, remainder = divmod(len(items), count)
    chunks: list[list[Case]] = []
    start = 0
    for index in range(count):
        stop = start + size + (index < remainder)
        chunks.append(items[start:stop])
        start = stop
    return chunks


def _make_server_pool_shards(cases: list[Case], *, max_workers: int) -> list[dict[str, Any]]:
    """Split cases into balanced, cold-compatible persistent-server lanes."""
    if not 1 <= max_workers <= MAX_SERVER_POOL_SIZE:
        raise SystemExit(f"--server-pool-size must be between 1 and {MAX_SERVER_POOL_SIZE}.")

    groups: dict[str, list[Case]] = {}
    for case in cases:
        key = server_reuse_key(case.server.model_dump(mode="json"))
        groups.setdefault(key, []).append(case)

    # Every cold configuration needs at least one server start. Distribute any
    # remaining GPU capacity to the group with the largest current shard.
    shard_counts = {key: 1 for key in groups}
    extra = max(0, min(max_workers, len(cases)) - len(groups))
    while extra:
        candidates = [key for key, group in groups.items() if shard_counts[key] < len(group)]
        if not candidates:
            break
        key = max(candidates, key=lambda item: len(groups[item]) / shard_counts[item])
        shard_counts[key] += 1
        extra -= 1

    shards: list[dict[str, Any]] = []
    for key, group in groups.items():
        for chunk in _split_evenly(group, shard_counts[key]):
            pool_id = f"pool-{len(shards) + 1:02d}"
            shards.append(
                {
                    "pool_id": pool_id,
                    "pool_run_dir": str(REMOTE_ROOT / group[0].stamp / "_server_pools" / pool_id),
                    "server_reuse_key": key,
                    "server_start_timeout_s": chunk[0].server_start_timeout_s,
                    "stream_logs": chunk[0].stream_logs,
                    "cases": [case.payload() for case in chunk],
                }
            )
    return shards


def _run_server_pool_shards(
    shards: list[dict[str, Any]], *, max_workers: int
) -> Iterable[list[dict[str, Any]]]:
    worker = PooledGpuServer.with_options(max_containers=max_workers)()
    return worker.run.map(shards, order_outputs=False)


@app.local_entrypoint()
def main(
    mode: str = "mock",
    server_config: str = "configs/server/mock.json",
    client_config: str = "",
    output_dir: str = "experiments/sweeps/modal",
    stamp: str = "",
    schedulers: str = "max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action",
    seeds: str = "7",
    max_batch_size: str = "",
    alpha: str = "",
    action_horizon_multiplier: str = "",
    server_pool_size: int = MAX_SERVER_POOL_SIZE,
    max_concurrent_cases: int = 100,
    server_start_timeout_minutes: float = SERVER_START_TIMEOUT_S / 60,
    stream_logs: bool = False,
) -> None:
    """Submit a scheduler sweep on Modal.

    ``--server-config`` accepts one path or a comma-separated list. With several,
    config-sensitive schedulers such as ``lookahead-actions`` run once per config
    while ordinary baselines run only against the first.
    """
    if mode not in {"gpu", "mock"}:
        raise SystemExit("--mode must be 'gpu' or 'mock'; a sweep needs a server.")
    if not client_config:
        raise SystemExit("--client-config is required.")
    if max_concurrent_cases < 1:
        raise SystemExit("--max-concurrent-cases must be positive.")
    if not 1 <= server_pool_size <= MAX_SERVER_POOL_SIZE:
        raise SystemExit(f"--server-pool-size must be between 1 and {MAX_SERVER_POOL_SIZE}.")
    if server_start_timeout_minutes <= 0:
        raise SystemExit("--server-start-timeout-minutes must be positive.")
    if stamp and (stamp in {".", ".."} or pathlib.PurePath(stamp).name != stamp):
        raise SystemExit("--stamp must be a single path component.")

    stamp = stamp or dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    run_root = pathlib.Path(output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    server_variants = [
        (path.stem.removeprefix("lookahead_actions_short_horizon_"), serve.Args.from_json(path))
        for path in _server_config_paths(server_config)
    ]
    config_root = pathlib.Path(client_config) if pathlib.Path(client_config).is_dir() else None
    experiment_configs = [
        (_experiment_name(path, root=config_root), ExperimentConfig.from_json(path))
        for path in _client_config_paths(client_config)
    ]

    cases = _make_cases(
        server_variants=server_variants,
        experiment_configs=experiment_configs,
        schedulers=parse_list_args(schedulers),
        seeds=parse_list_args(seeds, cast=int),
        max_batch_sizes=parse_list_args(max_batch_size, cast=int),
        alphas=parse_list_args(alpha, cast=float),
        action_horizon_multipliers=parse_list_args(action_horizon_multiplier, cast=float),
        mode=mode,
        stream_logs=stream_logs,
        stamp=stamp,
        server_start_timeout_s=server_start_timeout_minutes * 60,
    )
    if not cases:
        raise SystemExit("No cases produced; check sweep arguments.")

    rows_by_run_id = {case.run_id: case.row() for case in cases}
    write_rows(run_root / f"cases_{stamp}.csv", list(rows_by_run_id.values()))

    rows: list[dict[str, Any]] = []
    if mode == "gpu":
        shards = _make_server_pool_shards(cases, max_workers=server_pool_size)
        print(
            f"Running {len(cases)} GPU case(s) in {len(shards)} persistent-server "
            f"shard(s), using up to {server_pool_size} L40S workers"
        )
        shard_results = _run_server_pool_shards(shards, max_workers=server_pool_size)
        results = (result for shard in shard_results for result in shard)
    else:
        print(f"Running {len(cases)} mock case(s), at most {max_concurrent_cases} concurrently")
        results = _run_case_payloads(
            [case.payload() for case in cases],
            max_concurrent_cases=max_concurrent_cases,
        )
    for result in results:
        row = {**rows_by_run_id.get(result.get("run_id", ""), {}), **result}
        rows.append(row)
        starvation = row.get("starvation_rate")
        rate = f"{starvation:.3f}" if isinstance(starvation, (int, float)) else "n/a"
        print(f"{row.get('status', '?')}: {row['run_id']} starvation={rate}")

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

        plot_results(results_csv, run_root / "plots", line="scheduler_variant")
    except Exception as exc:  # noqa: BLE001
        print(f"Plot generation skipped: {exc!r}")
