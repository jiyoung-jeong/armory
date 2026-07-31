"""Run a generated server-config x client-config x seed sweep on Modal.

``scripts/gen_configs.py`` owns the server and fleet axes. Seed stays here
because each replicate must update both sides of the product. The compatibility
``--action-horizon-multiplier`` axis expands lookahead cases by weighting robots
on each fleet's shortest horizon. GPU sweeps reuse persistent L40S policy
servers; mock sweeps fan out through bounded CPU case runners.

Examples:
    # Timing-faithful CPU smoke sweep.
    uv run modal run scripts/modal/sweep.py \
        --mode mock \
        --server-config configs/gen/mock/server \
        --client-config configs/gen/mock/client \
        --seeds 7 \
        --output-dir experiments/sweeps/modal_mock

    # Real policy on persistent L40S servers driving LIBERO fleets on T4s.
    uv run modal run scripts/modal/sweep.py \
        --mode gpu \
        --server-config configs/gen/libero/server \
        --client-config configs/client/sim_sweep \
        --seeds 1,2,3 \
        --action-horizon-multiplier 1,3,5 \
        --server-pool-size 5 \
        --stamp libero_5min_paper \
        --output-dir experiments/sweeps/libero_5min
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import pathlib
import sys
from collections.abc import Iterable
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
# Make the source tree and `scripts.*` importable, then add scripts/ for the
# intentionally bare `serve` module used by the local CLI config loader.
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT), str(_HERE.parent)]

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

MAX_SERVER_POOL_SIZE = 6


@dataclasses.dataclass(frozen=True)
class Case:
    """One point in the sweep; only its plain-JSON payload crosses to Modal."""

    server: serve.Args
    experiment: ExperimentConfig
    server_name: str
    experiment_name: str
    seed: int
    action_horizon_multiplier: float | None = None

    @property
    def run_id(self) -> str:
        config = self.server.server
        parts = [
            f"scheduler={config.scheduler.scheduling_algorithm}",
            f"experiment={self.experiment_name}",
            f"num_robots={len(self.experiment.robots)}",
            f"seed={self.seed}",
            f"max_batch_size={config.max_batch_size}",
            f"alpha={config.scheduler.alpha}",
            f"server_variant={self.server_name}",
        ]
        if self.action_horizon_multiplier is not None:
            parts.append(f"ahm={self.action_horizon_multiplier}")
        return "__".join(parts)

    def payload(
        self,
        *,
        mode: str,
        stamp: str,
        stream_logs: bool,
        server_start_timeout_s: float,
    ) -> dict[str, Any]:
        """Flatten the pydantic configs into the primitives ``app.launch`` accepts."""
        return {
            "mode": mode,
            "run_id": self.run_id,
            "run_dir": str(REMOTE_ROOT / stamp / self.run_id),
            "server_config": self.server.model_dump(mode="json"),
            "client_config": {
                "experiment_config": self.experiment.model_dump(mode="json"),
                "scheduler_config": self.server.server.scheduler.model_dump(mode="json"),
            },
            "stream_logs": stream_logs,
            "server_start_timeout_s": server_start_timeout_s,
        }

    def row(self, stamp: str) -> dict[str, Any]:
        """Return config metadata used by the result CSV and plotting scripts."""
        server = self.server.server
        scheduler = server.scheduler.scheduling_algorithm
        scheduler_variant = scheduler
        method_variant = self.server_name
        if self.action_horizon_multiplier is not None:
            scheduler_variant += f" (ahm={self.action_horizon_multiplier})"
            method_variant += f" (ahm={self.action_horizon_multiplier})"
        return {
            "stamp": stamp,
            "run_id": self.run_id,
            "scheduler": scheduler,
            "scheduler_variant": scheduler_variant,
            "method_variant": method_variant,
            "server_variant": self.server_name,
            "experiment": self.experiment_name,
            "num_robots": len(self.experiment.robots),
            "seed": self.seed,
            "max_batch_size": server.max_batch_size,
            "alpha": server.scheduler.alpha,
            "action_horizon_multiplier": (
                self.action_horizon_multiplier if self.action_horizon_multiplier is not None else ""
            ),
            "weights": ",".join(f"{robot.weight:g}" for robot in self.experiment.robots),
        }


def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _config_paths(path: str, *, flag: str) -> list[tuple[str, pathlib.Path]]:
    """Resolve a config file or tree into stable ``(name, path)`` pairs."""
    root = pathlib.Path(path)
    if root.is_file():
        return [(root.stem, root)]
    if root.is_dir():
        found = sorted(root.rglob("*.json"))
        if found:
            return [
                (str(candidate.relative_to(root).with_suffix("")).replace("/", "_"), candidate)
                for candidate in found
            ]
    raise SystemExit(f"{flag} must be a JSON file or a directory containing them: {path}")


def _weight_shortest_horizon(experiment: ExperimentConfig, multiplier: float) -> ExperimentConfig:
    """Scale weights for robots on the fleet's shortest execution horizon."""
    if multiplier <= 0:
        raise SystemExit("--action-horizon-multiplier values must be positive.")
    shortest = min(robot.execution_horizon.max for robot in experiment.robots)
    robots = [
        robot.model_copy(
            update={"weight": robot.weight * multiplier}
            if robot.execution_horizon.max == shortest
            else {}
        )
        for robot in experiment.robots
    ]
    return experiment.model_copy(update={"robots": robots})


def _make_cases(
    *,
    servers: list[tuple[str, serve.Args]],
    experiments: list[tuple[str, ExperimentConfig]],
    seeds: list[int],
    action_horizon_multipliers: list[float],
) -> list[Case]:
    """Expand the config product, varying client weights only for lookahead."""
    cases: list[Case] = []
    for (server_name, server), (experiment_name, experiment), seed in itertools.product(
        servers, experiments, seeds
    ):
        scheduler = server.server.scheduler.scheduling_algorithm
        multipliers: list[float | None] = (
            action_horizon_multipliers
            if scheduler == "lookahead-actions" and action_horizon_multipliers
            else [None]
        )
        for multiplier in multipliers:
            seeded_experiment = experiment.model_copy(update={"seed": seed})
            if multiplier is not None:
                seeded_experiment = _weight_shortest_horizon(seeded_experiment, multiplier)
            cases.append(
                Case(
                    server=server.model_copy(update={"seed": seed}),
                    experiment=seeded_experiment,
                    server_name=server_name,
                    experiment_name=experiment_name,
                    seed=seed,
                    action_horizon_multiplier=multiplier,
                )
            )
    return cases


def _run_case_payloads(
    payloads: list[dict[str, Any]], *, max_concurrent_cases: int
) -> Iterable[dict[str, Any]]:
    """Map mock cases while leaving excess inputs queued at the runner boundary."""
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


def _make_server_pool_shards(
    cases: list[Case],
    *,
    max_workers: int,
    stamp: str,
    stream_logs: bool,
    server_start_timeout_s: float,
) -> list[dict[str, Any]]:
    """Split cases into balanced lanes whose cold server settings are compatible."""
    if not 1 <= max_workers <= MAX_SERVER_POOL_SIZE:
        raise SystemExit(f"--server-pool-size must be between 1 and {MAX_SERVER_POOL_SIZE}.")

    groups: dict[str, list[Case]] = {}
    for case in cases:
        key = server_reuse_key(case.server.model_dump(mode="json"))
        groups.setdefault(key, []).append(case)

    # Every cold configuration needs a lane. Allocate remaining capacity to the
    # group with the largest current shard; workers start independently, with no
    # all-ready barrier.
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
                    "pool_run_dir": str(REMOTE_ROOT / stamp / "_server_pools" / pool_id),
                    "server_reuse_key": key,
                    "server_start_timeout_s": server_start_timeout_s,
                    "stream_logs": stream_logs,
                    "cases": [
                        case.payload(
                            mode="gpu",
                            stamp=stamp,
                            stream_logs=stream_logs,
                            server_start_timeout_s=server_start_timeout_s,
                        )
                        for case in chunk
                    ],
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
    server_config: str = "",
    client_config: str = "",
    output_dir: str = "experiments/sweeps/modal",
    stamp: str = "",
    seeds: str = "7",
    action_horizon_multiplier: str = "",
    server_pool_size: int = MAX_SERVER_POOL_SIZE,
    max_concurrent_cases: int = 100,
    server_start_timeout_minutes: float = SERVER_START_TIMEOUT_S / 60,
    stream_logs: bool = False,
) -> None:
    """Submit a generated-config scheduler sweep on Modal."""
    if mode not in {"gpu", "mock"}:
        raise SystemExit("--mode must be 'gpu' or 'mock'; a sweep needs a server.")
    if not server_config or not client_config:
        raise SystemExit("--server-config and --client-config are both required.")
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

    servers = [
        (name, serve.Args.from_json(path))
        for name, path in _config_paths(server_config, flag="--server-config")
    ]
    experiments = [
        (name, ExperimentConfig.from_json(path))
        for name, path in _config_paths(client_config, flag="--client-config")
    ]
    seed_values = parse_list_args(seeds, cast=int)
    if not seed_values:
        raise SystemExit("--seeds must list at least one seed.")

    cases = _make_cases(
        servers=servers,
        experiments=experiments,
        seeds=seed_values,
        action_horizon_multipliers=parse_list_args(action_horizon_multiplier, cast=float),
    )
    rows_by_run_id = {case.run_id: case.row(stamp) for case in cases}
    if len(rows_by_run_id) != len(cases):
        raise SystemExit("Config names produced duplicate run IDs; rename the colliding files.")
    write_rows(run_root / f"cases_{stamp}.csv", list(rows_by_run_id.values()))

    timeout_s = server_start_timeout_minutes * 60
    if mode == "gpu":
        shards = _make_server_pool_shards(
            cases,
            max_workers=server_pool_size,
            stamp=stamp,
            stream_logs=stream_logs,
            server_start_timeout_s=timeout_s,
        )
        print(
            f"Running {len(cases)} GPU case(s) in {len(shards)} persistent-server "
            f"shard(s), using up to {server_pool_size} L40S workers"
        )
        shard_results = _run_server_pool_shards(shards, max_workers=server_pool_size)
        results: Iterable[dict[str, Any]] = (result for shard in shard_results for result in shard)
    else:
        print(f"Running {len(cases)} mock case(s), at most {max_concurrent_cases} concurrently")
        payloads = [
            case.payload(
                mode=mode,
                stamp=stamp,
                stream_logs=stream_logs,
                server_start_timeout_s=timeout_s,
            )
            for case in cases
        ]
        results = _run_case_payloads(payloads, max_concurrent_cases=max_concurrent_cases)

    rows: list[dict[str, Any]] = []
    for result in results:
        row = {**rows_by_run_id.get(result.get("run_id", ""), {}), **result}
        rows.append(row)
        starvation = row.get("starvation_rate")
        rate = f"{starvation:.3f}" if isinstance(starvation, (int, float)) else "n/a"
        print(f"{row.get('status', '?')}: {row['run_id']} starvation={rate}")

    download_artifacts(stamp=stamp, out=run_root, rows=rows)
    results_csv = run_root / f"sweep_results_{stamp}.csv"
    write_rows(results_csv, rows)

    suspicious = [row for row in rows if row.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for row in suspicious:
            print(f"  {row['run_id']}: {row.get('timing_flags', '')}")

    if mode == "gpu":
        print(
            f"\nPlot with:\n  uv run python scripts/visualization/plot_libero_sweep.py {run_root}"
        )
    else:
        print(
            "\nPlot with:\n  uv run python scripts/visualization/plot_sweep.py "
            f"--results {results_csv} --line method_variant"
        )
