"""Generate the server and client config trees that a sweep consumes.

Sweeps take the product of a server config dir and a client config dir, so every
axis that *shapes* a config is decided here: schedulers and their knobs on the
server side, fleet size and horizon mix on the client side. That split is what
lets a sweeper stay a plain product -- it never has to know that ``alpha`` only
reaches ``dynamic-action``, because this script already collapsed the variants
that a scheduler would have ignored.

Seed is deliberately *not* an axis here: it lands on both the server and the
client, so it stays a sweep-level replication flag.

Examples:
    # LIBERO: 3 schedulers x 3 fleet shapes x 5 sizes
    uv run python scripts/gen_configs.py --output-dir configs/gen/libero \\
        --env libero --schedulers max-batch greedy-deadline lookahead-actions

    # Alpha fairness study on the mock stack
    uv run python scripts/gen_configs.py --output-dir configs/gen/alpha \\
        --env mock --schedulers dynamic-action --alphas 0.0 0.5 1.0 \\
        --fleet-sizes 4 8 --shapes one_fast
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib
import sys
from typing import Literal

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Keep the source tree importable, then add the repository and scripts roots so
# ``scripts.*`` and the bare ``serve`` module both resolve.
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT), str(_HERE)]

import serve  # noqa: E402
import tyro  # noqa: E402

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

SCHEDULER_AXES = {
    "dynamic-action": ("alpha",),
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
    alphas: tuple[float, ...] = (1.0,)
    """Only reaches dynamic-action; variants collapse for every other scheduler."""
    num_steps: int = 10
    port: int = 8080

    fleet_sizes: tuple[int, ...] = (2, 4, 6, 8, 10)
    shapes: tuple[str, ...] = ("hom", "half_fast_half_slow", "one_fast")
    fast_horizon: int = 6
    slow_horizon: int = 10
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

    for scheduler, batch, alpha in itertools.product(
        args.schedulers, args.max_batch_sizes, args.alphas
    ):
        axes = SCHEDULER_AXES.get(scheduler, ())
        values = {"alpha": alpha}
        config = SchedulerConfig(
            scheduling_algorithm=scheduler,
            **{axis: values[axis] for axis in axes},
        )
        key = (batch, config.model_dump_json())
        if key in seen:
            continue
        seen.add(key)

        name = scheduler
        if "alpha" in axes:
            name += f"_alpha{_num(alpha)}"
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

    configs: list[tuple[pathlib.Path, ExperimentConfig]] = []
    for shape, size in itertools.product(args.shapes, args.fleet_sizes):
        horizons = FLEET_SHAPES[shape](size, args.fast_horizon, args.slow_horizon)
        robots = [
            Robot(
                execution_horizon=ExecutionHorizon(min=args.min_horizon, max=horizon),
                control_hz=args.control_hz,
            )
            for horizon in horizons
        ]
        configs.append(
            (
                pathlib.Path(shape) / f"{size}_robots.json",
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
