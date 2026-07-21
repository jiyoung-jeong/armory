import datetime
import logging
import pathlib
import shutil
import time

from pydantic import Field, model_validator

from armory.serving.protocol import SchedulerConfig
from armory_client.action_chunkers import ActionChunkBrokerType, BrokerConfig
from armory_client.client import BidirectionalWebsocket
from evaluation.cli import JsonArgs
from evaluation.runtime.agents import policy_agent as _policy_agent
from evaluation.runtime.runtime import Runtime
from evaluation.server_control_client import ServerControlClient
from evaluation.sims.libero import logging_config
from evaluation.sims.libero.env import LiberoSimEnvironment
from evaluation.sims.libero.mock_env import MockEnvironment
from evaluation.types import EnvironmentType, ExecutionHorizon, NetworkLatency
from utils import seed_everything

logger = logging.getLogger(__name__)


class Args(JsonArgs):
    # environment
    env: EnvironmentType = EnvironmentType.MOCK
    max_steps: int = Field(gt=0, default=100)
    action_chunk_broker_type: ActionChunkBrokerType = ActionChunkBrokerType.NAIVE_ASYNC
    time_limit: float = Field(default=0.0, ge=0.0)

    # robot
    execution_horizon: ExecutionHorizon = ExecutionHorizon()
    observation_latency: NetworkLatency = NetworkLatency()
    action_latency: NetworkLatency = NetworkLatency()
    control_hz: int = Field(gt=0, default=20)

    scheduler_config: SchedulerConfig = SchedulerConfig()

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


def create_mock_environment():
    return MockEnvironment()


def create_libero_environment(task_id: int, seed: int):
    from libero.libero.benchmark import Benchmark, Task, get_benchmark_dict

    from evaluation.sims.libero.utils import _get_libero_env

    benchmark_dict: dict[str, type[Benchmark]] = get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()

    task: Task = task_suite.get_task(task_id)
    raw_env, _ = _get_libero_env(task, seed=seed)

    return LiberoSimEnvironment(
        env=raw_env,
        task_description=task.language,
        initial_state=task_suite.get_task_init_states(task_id),
    )


def create_agent(
    host: str,
    port: str,
    control_hz: float,
    execution_horizon: ExecutionHorizon,
    action_chunk_broker_type: ActionChunkBrokerType,
):
    ws_client = BidirectionalWebsocket(
        robot_id="robot",
        host=host,
        port=port,
        control_hz=control_hz,
    )
    ws_client.connect()

    config = BrokerConfig(
        ws_client=ws_client,
        control_hz=control_hz,
        min_execution_horizon=execution_horizon.min,
        max_execution_horizon=execution_horizon.max,
    )
    broker = action_chunk_broker_type.create(config)
    return _policy_agent.PolicyAgent(broker=broker)


def run_robot(create_agent, create_environment, control_hz: float, time_limit: float) -> None:
    # NOTE: we pass factory methods instead of directly creating objects so this function can be directly used with multiprocessing
    env = create_environment()
    agent = create_agent()

    runtime = Runtime(
        environment=env,
        agent=agent,
        control_hz=control_hz,  # NOTE: maybe don't need to pass this
        deadline=time.monotonic() + time_limit,
    )
    runtime.run()
    runtime.close()


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

    control_client = ServerControlClient(host=args.host, port=args.port)
    control_client.reconfigure_server(args.scheduler_config)
    control_client.reset_server()

    run_robot()


if __name__ == "__main__":
    main(Args.from_cli())


# to decide:
# how to save?
# I don't want to import env stuff if I don't need, but I also want to share imports when I can
# does mock agent need a websocket
