import dataclasses
import datetime
import enum
import logging
import multiprocessing as mp
import pathlib
import socket
import sys
from typing import Literal

import tyro

from armory.serving.server import PolicyServer
from armory.utils import logging_config
from openpi_adapter.serve_factory import EnvMode

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import resolve_policy  # noqa: E402


class ModelFamily(str, enum.Enum):
    PI05 = "pi05"
    GROOT_N17 = "gr00t-n1.7"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a specific checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default checkpoint for the given --env."""


@dataclasses.dataclass
class Mock:
    """Use a lightweight mock policy that does not load weights or use a GPU."""

    action_horizon: int = 50
    action_dim: int = 14
    profile: str = "l40s_pi05"


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.LIBERO

    # options are PI05, GROOT_N17
    model: ModelFamily = ModelFamily.PI05

    default_prompt: str | None = None

    port: int = 8080

    policy: Checkpoint | Default | Mock = dataclasses.field(default_factory=Default)

    max_batch_size: int = 1

    num_steps: int = 10

    log_dir: str = "logs/server"

    scheduling_algorithm: str = "greedy-deadline"

    alpha: float = 1.0

    min_ex: int = 0

    lookahead_horizon_ms: int = 500
    lookahead_timestep_ms: int = 50
    lookahead_control_hz: int = 20

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def build_scheduler_kwargs(args: Args, *, action_horizon_steps: int) -> dict | None:
    if args.scheduling_algorithm == "dynamic-action":
        return {
            "alpha": args.alpha,
        }
    if args.scheduling_algorithm == "lookahead":
        return {
            "horizon_ms": args.lookahead_horizon_ms,
            "timestep_ms": args.lookahead_timestep_ms,
            "action_horizon_steps": action_horizon_steps,
            "control_hz": args.lookahead_control_hz,
        }
    return None


def main(args: Args) -> None:
    log_path = (
        pathlib.Path(args.log_dir)
        / f"serve_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_queue, log_listener = logging_config.setup_logging(
        log_path=log_path, level=getattr(logging, args.log_level)
    )

    policy_config = args.policy.config if isinstance(args.policy, Checkpoint) else None
    policy_dir = args.policy.dir if isinstance(args.policy, Checkpoint) else None
    mock = args.policy if isinstance(args.policy, Mock) else None

    resolved = resolve_policy(
        model=args.model.value,
        env=args.env,
        policy_config=policy_config,
        policy_dir=policy_dir,
        max_batch_size=args.max_batch_size,
        num_steps=args.num_steps,
        default_prompt=args.default_prompt,
        scheduling_algorithm=args.scheduling_algorithm,
        mock=mock,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    scheduler_kwargs = build_scheduler_kwargs(
        args, action_horizon_steps=resolved.metadata.action_horizon
    )
    resolved.metadata.scheduler_kwargs = scheduler_kwargs
    resolved.metadata.min_ex = args.min_ex

    server = PolicyServer(
        metadata=resolved.metadata,
        policy_factory=resolved.factory,
        scheduler_kwargs=scheduler_kwargs,
        log_queue=log_queue,
    )
    try:
        server.serve_forever(host="0.0.0.0", port=args.port)
    finally:
        log_listener.stop()


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main(tyro.cli(Args))
