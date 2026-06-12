import dataclasses
import datetime
import enum
import logging
import multiprocessing as mp
import pathlib
import socket
import sys
from dataclasses import field
from typing import Literal

from armory.serving.server import PolicyServer
from armory.utils import logging_config
from armory_client.schemas import SchedulerConfig
from openpi_adapter.serve_factory import EnvMode

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import JsonArgs, resolve_policy  # noqa: E402

from sims.libero.seeding import seed_everything  # noqa: E402


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
    model: str = "pi05"
    gpu: str = "l40s"


@dataclasses.dataclass
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
        scheduling_algorithm=args.scheduler.scheduling_algorithm,
        mock=mock,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

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
    # TODO: check this thoroughly, decide, and document
    mp.set_start_method("fork", force=True)
    main(Args.from_cli())
