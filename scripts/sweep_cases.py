from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from scripts import serve

from evaluation.types import ExperimentConfig

UNWEIGHTED_SCHEDULERS = {"deficit-round-robin", "max-batch", "round-robin"}


@dataclasses.dataclass(frozen=True)
class Case:
    server: serve.Args
    experiment: ExperimentConfig
    server_name: str
    experiment_name: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"server={self.server_name}__client={self.experiment_name}__seed={self.seed}"

    def client_config(self) -> dict[str, Any]:
        return {
            "experiment_config": self.experiment.model_dump(mode="json"),
            "scheduler_config": self.server.server.scheduler.model_dump(mode="json"),
        }

    def row(self, stamp: str) -> dict[str, Any]:
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
            "weights": ",".join(f"{robot.weight:g}" for robot in self.experiment.robots),
        }


def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def config_paths(path: str, *, flag: str) -> list[tuple[str, pathlib.Path]]:
    root = pathlib.Path(path)
    if root.is_file():
        return [(root.stem, root)]
    if root.is_dir():
        found = sorted(root.rglob("*.json"))
        if found:
            return [(str(p.relative_to(root).with_suffix("")).replace("/", "_"), p) for p in found]
    raise SystemExit(f"{flag} must be a JSON file or a directory containing them: {path}")


def _experiments_for_server(
    server: serve.Args, experiments: list[tuple[str, ExperimentConfig]]
) -> list[tuple[str, ExperimentConfig]]:
    if server.server.scheduler.scheduling_algorithm not in UNWEIGHTED_SCHEDULERS:
        return experiments

    # Collapse configs that differ only by a weight this scheduler ignores,
    # preferring the unit-weight representative when one exists.
    representatives: dict[tuple[str, str], tuple[str, ExperimentConfig]] = {}
    for name, experiment in experiments:
        family, separator, weight_label = name.rpartition("_w")
        try:
            float(weight_label)
        except ValueError:
            family = name
        if not separator:
            family = name
        unweighted = experiment.model_copy(
            update={
                "robots": [robot.model_copy(update={"weight": 1.0}) for robot in experiment.robots]
            }
        )
        key = family, unweighted.model_dump_json()
        if key not in representatives or all(robot.weight == 1.0 for robot in experiment.robots):
            representatives[key] = (name, experiment)
    return list(representatives.values())


def build_cases(server_config: str, client_config: str, seeds: list[int]) -> list[Case]:
    if not seeds:
        raise SystemExit("--seeds must list at least one seed.")
    servers = [
        (name, serve.Args.from_json(path))
        for name, path in config_paths(server_config, flag="--server-config")
    ]
    experiments = [
        (name, ExperimentConfig.from_json(path))
        for name, path in config_paths(client_config, flag="--client-config")
    ]
    return [
        Case(
            server=server.model_copy(update={"seed": seed}),
            experiment=experiment.model_copy(update={"seed": seed}),
            server_name=server_name,
            experiment_name=experiment_name,
            seed=seed,
        )
        for server_name, server in servers
        for experiment_name, experiment in _experiments_for_server(server, experiments)
        for seed in seeds
    ]
