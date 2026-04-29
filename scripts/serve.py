import dataclasses
import datetime
import logging
import pathlib
import socket
import sys
from typing import Literal

import tyro

from armory_client.schemas import ServerMetadata
from armory.serving.server import PolicyServer
from armory.utils import logging_config
from openpi_adapter.serve_factory import EnvMode
from openpi_adapter.serve_factory import create_policy
from openpi_adapter.serve_factory import get_model_dims

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import DEFAULT_CHECKPOINT  # noqa: E402


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve script."""

    env: EnvMode = EnvMode.ALOHA_SIM

    default_prompt: str | None = None

    port: int = 8080

    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    max_batch_size: int = 1

    num_steps: int = 10

    log_dir: str = "logs/server"

    scheduling_algorithm: str = "greedy-deadline"

    useful_action_weight: float = 0.0 # rewards action
    useful_tardiness_weight: float = 0.0 # penalizes length of unusable chunk
    useful_slack_weight: float = 0.0 # rewards slack
    useful_deficit_weight: float = 4.0 # rewards underserved

    lookahead_horizon_ms: int = 500
    lookahead_timestep_ms: int = 50
    lookahead_control_hz: int = 20

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class _PolicyFactory:
    """Picklable callable required by spawn multiprocessing."""

    def __init__(self, args: Args, config_name: str, checkpoint_dir: str):
        self._args = args
        self._config_name = config_name
        self._checkpoint_dir = checkpoint_dir

    def __call__(self):
        return create_policy(
            self._config_name,
            self._checkpoint_dir,
            default_prompt=self._args.default_prompt,
            sample_kwargs={"num_steps": self._args.num_steps},
            env_mode=self._args.env,
        )


def build_scheduler_kwargs(args: Args, *, action_horizon_steps: int) -> dict | None:
    if args.scheduling_algorithm in {"useful-action", "marginal-utility", "slack-aware-deficit"}:
        return {
            "useful_action_weight": args.useful_action_weight,
            "tardiness_weight": args.useful_tardiness_weight,
            "slack_weight": args.useful_slack_weight,
            "deficit_weight": args.useful_deficit_weight,
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
    log_queue, log_listener = logging_config.setup_logging(log_path=log_path, level=getattr(logging, args.log_level))

    match args.policy:
        case Checkpoint():
            config_name = args.policy.config
            checkpoint_dir = args.policy.dir
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                config_name = checkpoint["config"]
                checkpoint_dir = checkpoint["dir"]
            else:
                raise ValueError(f"Unsupported environment mode: {args.env}")

    action_horizon, action_dim = get_model_dims(config_name)

    server_metadata = ServerMetadata(
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_steps=args.num_steps,
        max_batch_size=args.max_batch_size,
        env=args.env.value,
        scheduling_algorithm=args.scheduling_algorithm,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    scheduler_kwargs = build_scheduler_kwargs(args, action_horizon_steps=action_horizon)
    policy_factory = _PolicyFactory(args, config_name, checkpoint_dir)

    server = PolicyServer(
        metadata=server_metadata,
        policy_factory=policy_factory,
        scheduler_kwargs=scheduler_kwargs,
        log_queue=log_queue,
    )
    try:
        server.serve_forever(host="0.0.0.0", port=args.port)
    finally:
        log_listener.stop()


if __name__ == "__main__":
    main(tyro.cli(Args))
