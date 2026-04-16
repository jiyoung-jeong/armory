import dataclasses
import datetime
import logging
import pathlib
import socket
import sys
from typing import Literal

from openpi_client.schemas import ServerMetadata
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import EnvMode
from openpi.serving.server import PolicyServer
from openpi.shared import logging_config
from openpi.training import config as _config

# Import shared utilities
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import DEFAULT_CHECKPOINT
from utils import create_default_policy


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8080

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Batch size to use for inference.
    max_batch_size: int = 1

    # Number of steps to use for sampling.
    num_steps: int = 10

    # Log directory to save the logs to.
    log_dir: str = "logs/server"

    # Scheduling algorithm for batching requests. # TODO: maybe should use enum?
    scheduling_algorithm: Literal["greedy", "lookahead", "round_robin", "random", "receding_horizon_ilp"] = "greedy"

    # Lookahead rollout horizon in milliseconds.
    lookahead_horizon_ms: int = 500

    # Lookahead time discretization in milliseconds.
    lookahead_timestep_ms: int = 50

    # Control frequency used to convert action chunks to wall-clock duration.
    lookahead_control_hz: int = 20

    # ILP discretization step in milliseconds.
    ilp_timestep_ms: int = 10

    # ILP planning horizon in discretized timesteps.
    ilp_horizon_steps: int = 100

    # Fraction of the horizon executed before swapping to the next receding plan.
    ilp_execution_fraction: float = 0.75

    # Per-solve time budget for the ILP backend.
    ilp_solve_timeout_ms: int = 1000

    # Optional action chunk length override for ILP. If unset, uses model action_horizon.
    ilp_action_horizon_steps: int | None = None

    # Secondary ILP tie-break: prioritize earlier scheduling density.
    ilp_pack_early_weight: float = 0.2

    # Secondary ILP tie-break: prioritize robots with older observations.
    ilp_obs_staleness_weight: float = 0

    # Logging level (DEBUG, INFO, WARNING, ERROR).
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# FIXME: may not be needed
class _PolicyFactory:
    """Module-level picklable callable required by spawn multiprocessing."""

    def __init__(self, args: Args):
        self._args = args

    def __call__(self) -> _policy.Policy:
        policy = create_policy(self._args)
        if "env" not in policy._metadata:  # noqa: SLF001
            policy._metadata["env"] = self._args.env.value  # noqa: SLF001
        return policy


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs={"num_steps": args.num_steps},
                use_triton_optimized=(args.env == EnvMode.LIBERO_REALTIME),
                batch_size=args.max_batch_size,
            )
        case Default():
            print(type(args.policy))
            return create_default_policy(
                args.env,
                batch_size=args.max_batch_size,
                default_prompt=args.default_prompt,
                sample_kwargs={"num_steps": args.num_steps},
            )


def build_scheduler_kwargs(args: Args, *, action_horizon_steps: int) -> dict[str, object] | None:
    if args.scheduling_algorithm == "lookahead":
        return {
            "horizon_ms": args.lookahead_horizon_ms,
            "timestep_ms": args.lookahead_timestep_ms,
            "action_horizon_steps": action_horizon_steps,
            "control_hz": args.lookahead_control_hz,
        }

    if args.scheduling_algorithm == "receding_horizon_ilp":
        ilp_action_horizon = args.ilp_action_horizon_steps or action_horizon_steps or 10
        if ilp_action_horizon < 1:
            ilp_action_horizon = 10
        return {
            "tick_ms": args.ilp_timestep_ms,
            "horizon_steps": args.ilp_horizon_steps,
            "execution_fraction": args.ilp_execution_fraction,
            "solve_timeout_ms": args.ilp_solve_timeout_ms,
            "action_horizon_steps": ilp_action_horizon,
            "pack_early_weight": args.ilp_pack_early_weight,
            "obs_staleness_weight": args.ilp_obs_staleness_weight,
        }

    return None


def main(args: Args) -> None:
    log_path = (
        pathlib.Path(args.log_dir)
        / f"serve_policy_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_queue, log_listener = logging_config.setup_logging(log_path=log_path, level=getattr(logging, args.log_level))

    # Create policy factory to avoid CUDA context fork issues
    policy_factory = _PolicyFactory(args)

    # Build metadata without loading the model to avoid CUDA initialization
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            checkpoint_dir = args.policy.dir
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                train_config = _config.get_config(checkpoint["config"])
                checkpoint_dir = checkpoint["dir"]
            else:
                raise ValueError(f"Unsupported environment mode: {args.env}")

    # Build server metadata
    server_metadata = ServerMetadata(
        config_name=train_config.name,
        checkpoint_dir=checkpoint_dir,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
        num_steps=args.num_steps,
        max_batch_size=args.max_batch_size,
        env=args.env.value,
        scheduling_algorithm=args.scheduling_algorithm,
    )

    # TODO: this looks sus to me
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    scheduler_kwargs = build_scheduler_kwargs(args, action_horizon_steps=server_metadata.action_horizon)

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
