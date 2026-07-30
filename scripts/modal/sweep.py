"""Modal scheduler sweep: one Modal case per (server config x client config x seed).

The sweep is a plain product. It does not know what a scheduler is, which knobs
one reads, or how a fleet is shaped -- those decisions were made when the config
trees were generated (``scripts/gen_configs.py``). Both ``--server-config`` and
``--client-config`` take a single file or a directory of them.

Seed is the one axis left here, because it lands on both sides of the product.

``--mode`` picks where it runs (see ``app.py``); only ``gpu`` and ``mock`` make
sense, since a sweep with no server has no scheduler to sweep.

Examples:
    # Everything on CPU with the timing-faithful mock policy. Start here.
    uv run modal run scripts/modal/sweep.py \\
        --mode mock \\
        --server-config configs/gen/mock/server \\
        --client-config configs/gen/mock/client \\
        --seeds 7 \\
        --output-dir experiments/sweeps/modal_mock

    # Real policy on an L40S driving LIBERO fleets on T4s.
    uv run modal run scripts/modal/sweep.py \\
        --mode gpu \\
        --server-config configs/gen/libero/server \\
        --client-config configs/gen/libero/client \\
        --seeds 7,42 \\
        --output-dir experiments/sweeps/libero
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import pathlib
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # repo root
# Order matters: src first so bare `utils` -> src/utils.py (not the shadowing
# scripts/utils.py), then repo root for `scripts.*`, then scripts/ for `import serve`.
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT), str(_HERE.parent)]

import serve  # noqa: E402
from scripts.modal.app import CaseRunner, app  # noqa: E402
from scripts.modal.images import REMOTE_ROOT  # noqa: E402
from scripts.modal.utils import download_artifacts, write_rows  # noqa: E402

from evaluation.types import ExperimentConfig  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Case:
    """One point in the sweep. Stays local: only ``payload`` crosses to Modal."""

    server: serve.Args  # embeds the SchedulerConfig the client also reconfigures with
    experiment: ExperimentConfig
    server_name: str
    experiment_name: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"server={self.server_name}__client={self.experiment_name}__seed={self.seed}"

    def payload(self, *, mode: str, stamp: str, stream_logs: bool) -> dict[str, Any]:
        """Flatten to the plain JSON ``app.launch`` takes.

        Modal would have to import ``serve`` on the orchestrator container to
        unpickle ``serve.Args``, and that bare module isn't importable there --
        hence dicts, not models.

        Run path is separate from the artifact path because Modal Volumes don't
        love many small writes; runs write hot to container disk and copy at the end.
        """
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
        }

    def row(self, stamp: str) -> dict[str, Any]:
        """Case metadata for the results CSV, read straight off the configs.

        The plotting scripts group by these columns, so they have to be present
        whether or not this particular sweep varied them.
        """
        server = self.server.server
        return {
            "stamp": stamp,
            "run_id": self.run_id,
            "scheduler": server.scheduler.scheduling_algorithm,
            "server_variant": self.server_name,
            "experiment": self.experiment_name,
            "num_robots": len(self.experiment.robots),
            "seed": self.seed,
            "max_batch_size": server.max_batch_size,
            "alpha": server.scheduler.alpha,
            "weights": ",".join(f"{robot.weight:g}" for robot in self.experiment.robots),
        }


def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _config_paths(path: str, *, flag: str) -> list[tuple[str, pathlib.Path]]:
    """Resolve a file or a directory of configs to ``(name, path)`` pairs.

    A directory name is its path relative to that directory, flattened, so
    ``one_fast/4_robots.json`` reads as ``one_fast_4_robots`` in the results.
    """
    root = pathlib.Path(path)
    if root.is_file():
        return [(root.stem, root)]
    if root.is_dir():
        found = sorted(root.rglob("*.json"))
        if found:
            return [(str(p.relative_to(root).with_suffix("")).replace("/", "_"), p) for p in found]
    raise SystemExit(f"{flag} must be a JSON file or a directory containing them: {path}")


@app.local_entrypoint()
def main(
    mode: str = "mock",
    server_config: str = "",
    client_config: str = "",
    output_dir: str = "experiments/sweeps/modal",
    seeds: str = "7",
    stream_logs: bool = False,
) -> None:
    """Submit a scheduler sweep on Modal.

    Generate the config trees first with ``scripts/gen_configs.py``.
    """
    if mode not in {"gpu", "mock"}:
        raise SystemExit("--mode must be 'gpu' or 'mock'; a sweep needs a server.")
    if not server_config or not client_config:
        raise SystemExit("--server-config and --client-config are both required.")

    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
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

    cases = [
        Case(
            server=server.model_copy(update={"seed": seed}),
            experiment=experiment.model_copy(update={"seed": seed}),
            server_name=server_name,
            experiment_name=experiment_name,
            seed=seed,
        )
        for (server_name, server), (experiment_name, experiment), seed in itertools.product(
            servers, experiments, seed_values
        )
    ]

    rows_by_run_id = {case.run_id: case.row(stamp) for case in cases}

    rows: list[dict[str, Any]] = []
    print(f"Running {len(cases)} case(s) in {mode} mode")
    payloads = [case.payload(mode=mode, stamp=stamp, stream_logs=stream_logs) for case in cases]
    for result in CaseRunner().run.map(payloads, order_outputs=False):
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

    print(
        f"\nPlot with:\n  uv run python scripts/visualization/plot_sweep.py --results {results_csv}"
    )
