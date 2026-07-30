import dataclasses
import datetime
import logging
import multiprocessing as mp
import pathlib
import socket
from dataclasses import field
from typing import Literal

from scripts.utils import JsonArgs

from armory.backends.registry import resolve_policy
from armory.backends.types import EnvMode, ModelFamily
from armory.serving.protocol import SchedulerConfig
from armory.serving.server import PolicyServer
from armory.utils.logging_config import setup_logging
from utils import seed_everything  # noqa: E402


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
    model: str = "pi05"
    gpu: str = "l40s"


class Args(JsonArgs):
    env: EnvMode = EnvMode.LIBERO
    model: ModelFamily = ModelFamily.PI05
    policy: Checkpoint | Default | Mock = dataclasses.field(default_factory=Default)
    num_steps: int = 10

    max_batch_size: int = 1
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    port: int = 8080
    seed: int = 7
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: str = "logs/server"


def main(args: Args) -> None:
    seed_everything(args.seed)
    log_path = (
        pathlib.Path(args.log_dir)
        / f"serve_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_queue, log_listener = setup_logging(
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
        scheduling_algorithm=args.scheduler.scheduling_algorithm,
        mock=mock,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # TODO: don't need to turn into kwargs anymore, just pass the pydantic BaseModel
    # Can delete to_scheduler_kwargs() function definition after too.
    scheduler_kwargs = args.scheduler.to_scheduler_kwargs()
    resolved.metadata.scheduler_kwargs = scheduler_kwargs

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
    # TODO manual: check this thoroughly, decide, and document
    mp.set_start_method("fork", force=True)
    main(Args.from_cli())
