"""Run a single robot against a policy server and save its episodes."""

import datetime
import logging
import pathlib
import shutil
from enum import Enum

from pydantic import Field, model_validator

import logging_config
from armory.serving.protocol import SchedulerConfig
from armory_client.action_chunkers import ActionChunkBrokerType, BrokerConfig
from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.client import BidirectionalWebsocket
from evaluation.agents import base as _agent
from evaluation.agents.mock_agent import MockAgent
from evaluation.agents.policy_agent import PolicyAgent
from evaluation.cli import JsonArgs
from evaluation.envs import base as _environment
from evaluation.envs.mock import MockEnvironment
from evaluation.run_robot import run_robot
from evaluation.save import SaveMeta
from evaluation.server_control_client import ServerControlClient
from evaluation.types import EnvironmentType, ExecutionHorizon
from utils import seed_everything

logger = logging.getLogger(__name__)


class AgentType(Enum):
    POLICY = "policy"  # queries the remote policy server
    MOCK = "mock"  # returns null actions, no server (offline loop test)


class Args(JsonArgs):
    # environment
    env: EnvironmentType = EnvironmentType.MOCK
    task_suite_name: str = "libero_10"
    task_id: int = 0
    max_steps: int = Field(gt=0, default=100)

    # agent
    agent: AgentType = AgentType.POLICY
    action_chunk_broker_type: ActionChunkBrokerType = ActionChunkBrokerType.NAIVE_ASYNC
    execution_horizon: ExecutionHorizon = ExecutionHorizon()
    control_hz: int = Field(gt=0, default=20)

    # rollout
    num_episodes: int = Field(gt=0, default=1)
    time_limit: float = Field(default=0.0, ge=0.0)

    scheduler_config: SchedulerConfig = SchedulerConfig()

    robot_idx: int = 0
    seed: int = Field(default=7, ge=0)
    host: str = "0.0.0.0"
    port: int = 8080
    output_dir: pathlib.Path = pathlib.Path("data/libero/multi_robot_videos")
    overwrite: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "Args":
        if not self.overwrite and self.output_dir.exists():
            raise ValueError(f"Output path {self.output_dir} already exists")
        return self


def create_environment(args: Args) -> _environment.Environment:
    if args.env == EnvironmentType.MOCK:
        return MockEnvironment(max_episode_steps=args.max_steps)
    if args.env == EnvironmentType.LIBERO:
        # Imported lazily: LIBERO/robosuite are heavy and Linux/GL-only.
        from evaluation.envs.libero import LiberoSimEnvironment

        return LiberoSimEnvironment(
            task_id=args.task_id,
            task_suite_name=args.task_suite_name,
            max_episode_steps=args.max_steps,
            seed=args.seed,
        )
    raise ValueError(f"Unsupported env: {args.env}")


def create_agent(
    args: Args,
) -> tuple[_agent.Agent, BidirectionalWebsocket | None, ActionChunkBroker | None]:
    """Build the agent plus the resources the caller must later close/snapshot."""
    if args.agent == AgentType.MOCK:
        return MockAgent(), None, None

    ws_client = BidirectionalWebsocket(
        robot_id=f"robot_{args.robot_idx}",
        host=args.host,
        port=args.port,
        control_hz=args.control_hz,
    )
    ws_client.connect()

    broker = args.action_chunk_broker_type.create(
        BrokerConfig(
            ws_client=ws_client,
            control_hz=args.control_hz,
            min_execution_horizon=args.execution_horizon.min,
            max_execution_horizon=args.execution_horizon.max,
        )
    )
    return PolicyAgent(broker), ws_client, broker


def main(args: Args) -> None:
    seed_everything(args.seed)

    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        args.output_dir
        / f"run_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging_config.setup_logging(log_path=log_path, level=logging.INFO)

    if args.agent == AgentType.POLICY:
        control_client = ServerControlClient(host=args.host, port=args.port)
        control_client.reconfigure_server(args.scheduler_config)
        control_client.reset_server()

    environment = create_environment(args)
    agent, ws_client, broker = create_agent(args)

    meta = SaveMeta(
        out_dir=args.output_dir,
        robot_idx=args.robot_idx,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        task_language=environment.task_language,
        control_hz=args.control_hz,
        # Zero-image mock frames aren't worth encoding.
        save_video=args.env != EnvironmentType.MOCK,
    )

    try:
        run_robot(
            environment=environment,
            agent=agent,
            meta=meta,
            broker=broker,
            num_episodes=args.num_episodes,
            time_limit=args.time_limit,
        )
    finally:
        environment.close()
        # close() stops the broker's background receive thread and the websocket;
        # closing only ws_client would leave that daemon thread to be killed
        # mid-I/O at interpreter exit (SIGABRT).
        if broker is not None:
            broker.close()
        elif ws_client is not None:
            ws_client.close()


if __name__ == "__main__":
    main(Args.from_cli())
