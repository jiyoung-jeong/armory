import dataclasses
import datetime
import logging
import multiprocessing as mp
import pathlib
import socket
import sys
import time
from typing import Any, Literal

import numpy as np
import tyro

from armory_client.messages import InferRequest, InferType
from armory_client.schemas import ServerMetadata
from armory.serving.server import PolicyServer
from armory.utils import logging_config
from openpi_adapter.serve_factory import EnvMode

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


profiles = {
    "l40s_pi05": {
        1: 0.070,
        2: 0.135,
        3: 0.195,
        4: 0.250,
    }
}

@dataclasses.dataclass
class Mock:
    """Use a lightweight mock policy that does not load weights or use a GPU."""

    action_horizon: int = 50
    action_dim: int = 14
    profile: str = "l40s_pi05"


@dataclasses.dataclass
class Args:
    """Arguments for the serve script."""

    env: EnvMode = EnvMode.ALOHA_SIM

    default_prompt: str | None = None

    port: int = 8080

    policy: Checkpoint | Default | Mock = dataclasses.field(default_factory=Default)

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
        from openpi_adapter.serve_factory import create_policy

        return create_policy(
            self._config_name,
            self._checkpoint_dir,
            default_prompt=self._args.default_prompt,
            sample_kwargs={"num_steps": self._args.num_steps},
            env_mode=self._args.env,
        )


class _MockPolicy:
    """Stub policy implementing the armory engine interface without weights/GPU."""

    def __init__(self, *, env: str, action_horizon: int, action_dim: int, inference_latency: dict[int, float]):
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._inference_latency = inference_latency
        self.metadata = {"env": env}

    def make_infer_request(self) -> InferRequest:
        now = time.time()
        return InferRequest(
            robot_id="__warmup__",
            observation={},
            observation_step=0,
            action_start_step=0,
            request_timestamp=now,
            deadline=now + 60.0,
            execution_horizon=0,
            infer_type=InferType.SYNC,
            params=None,
            noise=None,
        )

    def warmup(self, max_batch_size: int) -> None:
        del max_batch_size

    def infer_batch(self, requests: list[InferRequest]) -> list[dict[str, Any]]:
        inference_latency = self._inference_latency[len(requests)]
        now = time.time()
        while time.time() - now < inference_latency:
            time.sleep(0.001)
        actions = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)
        return [
            {"actions": actions, "noise": None, "rtc_prev_actions": actions}
            for _ in requests
        ]


class _MockPolicyFactory:
    """Picklable factory for the mock policy."""

    def __init__(self, *, env: str, action_horizon: int, action_dim: int, profile: str):
        self._env = env
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._profile = profile
        self._inference_latency = profiles[profile]

    def __call__(self) -> _MockPolicy:
        return _MockPolicy(
            env=self._env,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            inference_latency=self._inference_latency,
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
        case Mock():
            config_name = "mock"
            checkpoint_dir = ""
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                config_name = checkpoint["config"]
                checkpoint_dir = checkpoint["dir"]
            else:
                raise ValueError(f"Unsupported environment mode: {args.env}")

    if isinstance(args.policy, Mock):
        action_horizon = args.policy.action_horizon
        action_dim = args.policy.action_dim
    else:
        from openpi_adapter.serve_factory import get_model_dims

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
    if isinstance(args.policy, Mock):
        policy_factory: Any = _MockPolicyFactory(
            env=args.env.value,
            action_horizon=action_horizon,
            action_dim=action_dim,
            profile=args.policy.profile,
        )
    else:
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
    mp.set_start_method("fork", force=True)
    main(tyro.cli(Args))
