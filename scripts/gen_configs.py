"""Generate the server and client config trees that a sweep consumes.

Sweeps take the product of a server config dir and a client config dir, so every
axis that *shapes* a config is decided here: scheduler and batch size on the
server side, fleet shape and robot weights on the client side.

Seed is deliberately *not* an axis here: it lands on both the server and the
client, so it stays a sweep-level replication flag.

Examples:
    # LIBERO: 3 schedulers x 3 fleet shapes x 5 sizes
    uv run python scripts/gen_configs.py --output-dir configs/gen/libero \\
        --env libero --schedulers max-batch greedy-deadline lookahead-actions

    # Weight the shortest-horizon robots for weighted scheduler comparisons.
    uv run python scripts/gen_configs.py --output-dir configs/gen/weighted \\
        --env mock --schedulers weighted-edf weighted-deficit-round-robin lookahead-actions \\
        --short-horizon-weights 1 3 5 --fleet-sizes 4 8 --shapes one_fast
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import pathlib
import sys
from typing import Literal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tyro  # noqa: E402
from scripts import serve  # noqa: E402

from armory.serving.config import EngineConfig, ServerConfig  # noqa: E402
from armory.serving.protocol import SchedulerConfig  # noqa: E402
from evaluation.envs.config import LiberoConfig, MockConfig  # noqa: E402
from evaluation.types import ExecutionHorizon, ExperimentConfig, Robot  # noqa: E402

# A shape maps fleet size -> each robot's max execution horizon. "Fast" robots
# carry the shorter horizon: they run out of actions sooner, so they need to be
# scheduled more often.
FLEET_SHAPES = {
    "hom": lambda n, fast, slow: [fast] * n,
    "half_fast_half_slow": lambda n, fast, slow: [fast] * (n // 2) + [slow] * (n - n // 2),
    "one_fast": lambda n, fast, slow: [fast] + [slow] * (n - 1),
}


@dataclasses.dataclass
class Args:
    output_dir: pathlib.Path
    """Root to write under; server configs land in server/, clients in client/<shape>/."""

    env: Literal["libero", "mock"] = "libero"
    """Client-side environment kind. Selects the *simulator*, not the policy."""
    server_env: str = "libero"
    """serve.Args EnvMode. Stays a real env even for mock sweeps: EnvMode has no
    mock member, because the mock policy is swapped in by the runner's --mode."""
    model: str = "pi05"

    schedulers: tuple[str, ...] = ("max-batch", "greedy-deadline", "round-robin")
    max_batch_sizes: tuple[int, ...] = (5,)
    num_steps: int = 10
    port: int = 8080

    fleet_sizes: tuple[int, ...] = (2, 4, 6, 8, 10)
    shapes: tuple[str, ...] = ("hom", "half_fast_half_slow", "one_fast")
    fast_horizon: int = 6
    slow_horizon: int = 10
    short_horizon_weights: tuple[float, ...] = (1.0,)
    """Weights assigned to robots with the fleet's shortest execution horizon."""
    min_horizon: int = 1
    control_hz: int = 20
    time_limit: float = 120.0
    max_steps_per_episode: int = 300
    task_suite_name: str = "libero_10"


def _num(value: float) -> str:
    return f"{value:g}"


def _server_configs(args: Args) -> list[tuple[str, serve.Args]]:
    """One entry per *distinct* server config, named after the knobs that matter.

    Variants differing only in a knob the chosen scheduler ignores collapse to a
    single file instead of becoming duplicate sweep cases.
    """
    label_batch = len(args.max_batch_sizes) > 1
    seen: set[tuple] = set()
    configs: list[tuple[str, serve.Args]] = []

    for scheduler, batch in itertools.product(args.schedulers, args.max_batch_sizes):
        config = SchedulerConfig(scheduling_algorithm=scheduler)
        key = (batch, config.model_dump_json())
        if key in seen:
            continue
        seen.add(key)

        name = scheduler
        if label_batch:
            name += f"_b{batch}"

        configs.append(
            (
                name,
                serve.Args(
                    env=args.server_env,
                    model=args.model,
                    port=args.port,
                    server=ServerConfig(
                        max_batch_size=batch,
                        scheduler=config,
                        engine=EngineConfig(num_steps=args.num_steps),
                    ),
                ),
            )
        )
    return configs


def _client_configs(args: Args) -> list[tuple[pathlib.Path, ExperimentConfig]]:
    environment = (
        LiberoConfig(
            task_suite_name=args.task_suite_name, max_steps_per_episode=args.max_steps_per_episode
        )
        if args.env == "libero"
        else MockConfig(max_steps_per_episode=args.max_steps_per_episode)
    )

    short_weights = tuple(dict.fromkeys(args.short_horizon_weights))
    label_weight = len(short_weights) > 1 or short_weights != (1.0,)

    configs: list[tuple[pathlib.Path, ExperimentConfig]] = []
    for shape, size in itertools.product(args.shapes, args.fleet_sizes):
        horizons = FLEET_SHAPES[shape](size, args.fast_horizon, args.slow_horizon)
        shortest = min(horizons)
        # Uniformly scaling every robot does not change a weighted scheduler,
        # so homogeneous fleets need only the unit-weight representative.
        active_weights = (1.0,) if len(set(horizons)) == 1 else short_weights
        for short_weight in active_weights:
            robots = [
                Robot(
                    execution_horizon=ExecutionHorizon(min=args.min_horizon, max=horizon),
                    control_hz=args.control_hz,
                    weight=short_weight if horizon == shortest else 1.0,
                )
                for horizon in horizons
            ]
            name = f"{size}_robots"
            if label_weight and len(set(horizons)) > 1:
                name += f"_w{_num(short_weight)}"
            configs.append(
                (
                    pathlib.Path(shape) / f"{name}.json",
                    ExperimentConfig(
                        environment=environment, robots=robots, time_limit=args.time_limit
                    ),
                )
            )
    return configs


def main(args: Args) -> None:
    unknown = set(args.shapes) - set(FLEET_SHAPES)
    if unknown:
        raise SystemExit(f"Unknown shape(s) {sorted(unknown)}; known: {sorted(FLEET_SHAPES)}")
    if not args.short_horizon_weights:
        raise SystemExit("--short-horizon-weights must list at least one weight.")
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in args.short_horizon_weights):
        raise SystemExit("--short-horizon-weights must be positive and finite.")

    server_dir = args.output_dir / "server"
    client_dir = args.output_dir / "client"
    server_dir.mkdir(parents=True, exist_ok=True)

    for name, server in _server_configs(args):
        path = server_dir / f"{name}.json"
        # json_path is a CLI-only field on JsonArgs; writing it back as null
        # would just be noise in a generated file.
        path.write_text(json.dumps(server.model_dump(mode="json", exclude={"json_path"}), indent=4))
        print(f"wrote {path}")

    for relative, experiment in _client_configs(args):
        path = client_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        experiment.to_json(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
