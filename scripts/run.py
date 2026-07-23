import datetime
import logging
import multiprocessing
import pathlib
import shutil
import sys
import time
from enum import Enum

from pydantic import model_validator
from scripts.utils import JsonArgs

import logging_config
from armory.serving.protocol import SchedulerConfig
from armory_client.action_chunkers import BrokerConfig
from armory_client.client import BidirectionalWebsocket
from evaluation.agents.base import Agent
from evaluation.agents.mock_agent import MockAgent
from evaluation.agents.policy_agent import PolicyAgent
from evaluation.envs import base as _environment
from evaluation.envs.config import LiberoConfig, MockConfig
from evaluation.envs.mock import MockEnvironment
from evaluation.metrics import calculate_metrics, generate_all_plots
from evaluation.runtime import Runtime
from evaluation.save import SaveMeta, save_episode
from evaluation.server_control_client import ServerControlClient
from evaluation.types import ExperimentConfig
from utils import assert_egl_rendering, seed_everything

logger = logging.getLogger(__name__)


class AgentType(Enum):
    POLICY = "policy"  # queries the remote policy server
    MOCK = "mock"  # returns null actions, no server (offline loop test)


class Args(JsonArgs):
    experiment_config: ExperimentConfig = ExperimentConfig()
    scheduler_config: SchedulerConfig = SchedulerConfig()
    agent: AgentType = AgentType.POLICY

    host: str = "0.0.0.0"
    port: int = 8080
    output_dir: pathlib.Path = pathlib.Path("output/run")
    overwrite: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "Args":
        if not self.overwrite and self.output_dir.exists():
            raise ValueError(f"Output path {self.output_dir} already exists")
        return self


def create_environment(
    config: ExperimentConfig, robot_idx: int, libero_spec: object | None = None
) -> _environment.Environment:
    """Build robot ``robot_idx``'s environment from its pre-planned spec."""
    if isinstance(config.environment, MockConfig):
        return MockEnvironment(max_episode_steps=config.environment.max_steps_per_episode)
    if isinstance(config.environment, LiberoConfig):
        # Imported lazily: LIBERO/robosuite are heavy and Linux/GL-only.
        from evaluation.envs.libero import LiberoRobotSpec, LiberoSimEnvironment

        # not using EGL will slow down step times
        assert_egl_rendering()
        if not isinstance(libero_spec, LiberoRobotSpec):
            raise ValueError("A LIBERO robot spec is required for a LIBERO environment")

        return LiberoSimEnvironment(
            spec=libero_spec,
            max_episode_steps=config.environment.max_steps_per_episode,
            seed=config.seed + robot_idx,
        )
    raise ValueError(f"Unsupported environment: {config.environment}")


def create_agent(args: Args, robot_idx: int) -> Agent:
    """Build robot ``robot_idx``'s agent plus the resources the worker must later
    close/snapshot. A MOCK agent needs no server, so it opens no websocket."""
    if args.agent == AgentType.MOCK:
        return MockAgent()

    robot = args.experiment_config.robots[robot_idx]
    ws_client = BidirectionalWebsocket(
        robot_id=f"robot_{robot_idx}",
        host=args.host,
        port=args.port,
        control_hz=robot.control_hz,
    )
    ws_client.connect()

    broker = robot.action_chunk_broker_type.create(
        BrokerConfig(
            ws_client=ws_client,
            control_hz=robot.control_hz,
            min_execution_horizon=robot.execution_horizon.min,
            max_execution_horizon=robot.execution_horizon.max,
        )
    )
    return PolicyAgent(broker)


def run_robot(args: Args, robot_idx: int, libero_spec: object | None = None) -> None:
    """One robot's whole life: seed, build env/agent, roll out, tear down.

    Called inline for a single-robot fleet and inside a worker process for a
    multi-robot fleet; in the latter case the per-process isolation keeps heavy
    env state and the websocket receive thread separate per robot.
    """
    config = args.experiment_config
    seed_everything(config.seed + robot_idx)

    environment = create_environment(config, robot_idx, libero_spec)
    agent = create_agent(args, robot_idx)

    meta = SaveMeta(
        out_dir=args.output_dir,
        robot_idx=robot_idx,
        task_suite_name=libero_spec.task_suite_name if libero_spec is not None else "mock",
        task_id=libero_spec.task_id if libero_spec is not None else 0,
        task_language=environment.task_language,
        control_hz=config.robots[robot_idx].control_hz,
        # Zero-image mock frames aren't worth encoding.
        save_video=not isinstance(config.environment, MockConfig),
    )

    runtime = Runtime(environment, agent, control_hz=meta.control_hz)
    deadline = time.monotonic() + args.experiment_config.time_limit
    try:
        episode = 0
        while time.monotonic() < deadline:
            rollout = runtime.run_episode(deadline)

            save_episode(rollout, meta)
            episode += 1

        logger.info("robot %d: ran %d episode(s)", meta.robot_idx, episode)

    finally:
        runtime.close()


def run_fleet(args: Args) -> None:
    num_robots = len(args.experiment_config.robots)
    libero_specs: list[object | None]
    if isinstance(args.experiment_config.environment, LiberoConfig):
        from evaluation.envs.libero import plan_robot_specs

        libero_specs = plan_robot_specs(
            args.experiment_config.environment,
            num_robots=num_robots,
            experiment_seed=args.experiment_config.seed,
        )
    else:
        libero_specs = [None] * num_robots
    if num_robots == 1:
        # NOTE: runs inline for easy debugging
        logger.info("Running 1 robot inline")
        run_robot(args, 0, libero_specs[0])
        return

    logger.info("Launching %d robot process(es)", num_robots)
    processes = [
        multiprocessing.Process(
            target=run_robot,
            args=(args, robot_idx, libero_specs[robot_idx]),
            name=f"robot_{robot_idx}",
        )
        for robot_idx in range(num_robots)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    failed = [p.name for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Robot worker(s) exited non-zero: {failed}")


def main(args: Args) -> None:
    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        args.output_dir
        / f"run_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging_config.setup_logging(log_path=log_path, level=logging.INFO)

    control_client: ServerControlClient | None = None
    if args.agent == AgentType.POLICY:
        control_client = ServerControlClient(host=args.host, port=args.port)
        control_client.reconfigure_server(args.scheduler_config)
        control_client.reset_server()

    args.to_json(args.output_dir / "experiment_args.json")

    run_fleet(args)

    calculate_metrics(args.output_dir)
    generate_all_plots(args.output_dir)


def cli() -> None:
    if sys.platform == "linux":
        # forkserver: workers fork from a server process that has already
        # imported the heavy libraries, so their read-only pages are shared
        # copy-on-write across robots instead of duplicated per process. Safe
        # with sim envs because GL contexts are created per-worker after the fork.
        multiprocessing.set_start_method("forkserver")
    else:
        # macOS: forked processes can crash inside Apple frameworks; keep spawn.
        multiprocessing.set_start_method("spawn")

    main(Args.from_cli())


if __name__ == "__main__":
    cli()
