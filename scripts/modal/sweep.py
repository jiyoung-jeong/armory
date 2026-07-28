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
        --client-config configs/client/libero/half_fast_half_slow \\
        --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions \\
        --seeds 7,42 \\
        --alpha 0.0,0.5,1.0 \\
        --output-dir experiments/sweeps/libero
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import sys
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
from scripts.modal.app import CaseRunner, app  # noqa: E402
from scripts.modal.images import REMOTE_ROOT  # noqa: E402
from scripts.modal.utils import download_artifacts, write_rows  # noqa: E402

from evaluation.types import ExperimentConfig  # noqa: E402

ALPHA_SWEEP_SCHEDULERS = {"dynamic-action", "lookahead-actions"}
SERVER_CONFIG_SWEEP_SCHEDULERS = {"lookahead-actions"}


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
        }

    def row(self) -> dict[str, Any]:
        multipliers = self.server.scheduler.action_horizon_multipliers
        return {
            "stamp": self.stamp,
            "run_id": self.run_id,
            "scheduler": self.server.scheduler.scheduling_algorithm,
            "server_variant": self.server_variant,
            "experiment": self.experiment_name,
            "num_robots": len(self.experiment.robots),
            "seed": self.experiment.seed,
            "max_batch_size": self.server.max_batch_size,
            "alpha": self.server.scheduler.alpha,
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


def _make_cases(
    *,
    server_variants: list[tuple[str, serve.Args]],
    experiment_configs: list[tuple[str, ExperimentConfig]],
    schedulers: list[str],
    seeds: list[int],
    max_batch_sizes: list[int],
    alphas: list[float],
    mode: str,
    stream_logs: bool,
    stamp: str,
) -> list[Case]:
    if not max_batch_sizes:
        max_batch_sizes = [server_variants[0][1].max_batch_size]
    if not alphas:
        alphas = [server_variants[0][1].scheduler.alpha]

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
            for variant, server_args in variants:
                for experiment_name, experiment_config in experiment_configs:
                    for max_batch_size in max_batch_sizes:
                        for alpha in scheduler_alphas:
                            # One SchedulerConfig drives both boot (server, for the
                            # boot-only alpha) and runtime reconfigure (client);
                            # action_horizon_multipliers come from the server variant.
                            sched = server_args.scheduler.model_copy(
                                update={"scheduling_algorithm": scheduler, "alpha": alpha}
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
                                    experiment=experiment_config.model_copy(update={"seed": seed}),
                                    experiment_name=experiment_name,
                                    stamp=stamp,
                                    mode=mode,
                                    stream_logs=stream_logs,
                                    server_variant=variant,
                                )
                            )
    return cases


@app.local_entrypoint()
def main(
    mode: str = "mock",
    server_config: str = "configs/server/mock.json",
    client_config: str = "",
    output_dir: str = "experiments/sweeps/modal",
    schedulers: str = "fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action",
    seeds: str = "7",
    max_batch_size: str = "",
    alpha: str = "",
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

    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
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
        mode=mode,
        stream_logs=stream_logs,
        stamp=stamp,
    )
    if not cases:
        raise SystemExit("No cases produced; check sweep arguments.")

    rows_by_run_id = {case.run_id: case.row() for case in cases}
    write_rows(run_root / f"cases_{stamp}.csv", list(rows_by_run_id.values()))

    rows: list[dict[str, Any]] = []
    print(f"Running {len(cases)} case(s) in {mode} mode")
    for result in CaseRunner().run.map([case.payload() for case in cases], order_outputs=False):
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

        plot_results(results_csv, run_root / "plots")
    except Exception as exc:  # noqa: BLE001
        print(f"Plot generation skipped: {exc!r}")
