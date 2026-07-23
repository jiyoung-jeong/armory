"""Run many robots against a policy server, one OS process per robot.

Multi-robot analogue of ``scripts/run.py``: each robot builds its own
environment / agent / websocket *inside its own process* and drives a single
rollout via ``run_robot``. The fleet is described by ``ExperimentConfig``
(``configs/`` in ``evaluation``); robot ``i`` takes its per-robot settings from
``experiment_config.robots[i]``.
"""

import datetime
import json
import logging
import multiprocessing
import pathlib
import shutil
import sys

from pydantic import model_validator

import logging_config
from armory.serving.protocol import SchedulerConfig
from armory_client.action_chunkers import BrokerConfig
from armory_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from armory_client.client import BidirectionalWebsocket
from evaluation.agents import base as _agent
from evaluation.agents.mock_agent import MockAgent
from evaluation.agents.policy_agent import PolicyAgent
from evaluation.cli import JsonArgs
from evaluation.envs import base as _environment
from evaluation.envs.mock import MockEnvironment
from evaluation.metrics import calculate_metrics, generate_all_plots
from evaluation.run_robot import run_robot
from evaluation.save import SaveMeta
from evaluation.server_control_client import ServerControlClient
from evaluation.types import EnvironmentType, ExperimentConfig
from utils import seed_everything

logger = logging.getLogger(__name__)


class Args(JsonArgs):
    experiment_config: ExperimentConfig = ExperimentConfig()
    scheduler_config: SchedulerConfig = SchedulerConfig()

    host: str = "0.0.0.0"
    port: int = 8080
    output_dir: pathlib.Path = pathlib.Path("data/libero/multi_robot_videos")
    overwrite: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "Args":
        if not self.overwrite and self.output_dir.exists():
            raise ValueError(f"Output path {self.output_dir} already exists")
        return self


def create_environment(config: ExperimentConfig, robot_idx: int) -> _environment.Environment:
    """Build robot ``robot_idx``'s environment. Each robot runs its own task
    (``task_id = robot_idx``) and gets an offset seed so the fleet isn't
    perfectly correlated."""
    if config.env == EnvironmentType.MOCK:
        return MockEnvironment(max_episode_steps=config.max_steps)
    if config.env == EnvironmentType.LIBERO:
        # Imported lazily: LIBERO/robosuite are heavy and Linux/GL-only.
        from evaluation.envs.libero import LiberoSimEnvironment

        return LiberoSimEnvironment(
            task_id=robot_idx,
            task_suite_name=config.task_suite_name,
            max_episode_steps=config.max_steps,
            seed=config.seed + robot_idx,
        )
    raise ValueError(f"Unsupported env: {config.env}")


def create_agent(
    args: Args, robot_idx: int
) -> tuple[_agent.Agent, BidirectionalWebsocket | None, ActionChunkBroker | None]:
    """Build robot ``robot_idx``'s agent plus the resources the worker must
    later close/snapshot. A MOCK env needs no server, so it runs a MockAgent."""
    config = args.experiment_config
    if config.env == EnvironmentType.MOCK:
        return MockAgent(), None, None

    robot = config.robots[robot_idx]
    ws_client = BidirectionalWebsocket(
        robot_id=f"robot_{robot_idx}",
        host=args.host,
        port=args.port,
        control_hz=robot.control_hz,
    )
    ws_client.connect()

    broker = config.action_chunk_broker_type.create(
        BrokerConfig(
            ws_client=ws_client,
            control_hz=robot.control_hz,
            min_execution_horizon=robot.execution_horizon.min,
            max_execution_horizon=robot.execution_horizon.max,
        )
    )
    return PolicyAgent(broker), ws_client, broker


def _run_robot_worker(args: Args, robot_idx: int) -> None:
    """One robot's whole life: seed, build env/agent, roll out, tear down.

    Runs in its own process so heavy env state and the websocket receive thread
    stay isolated per robot.
    """
    config = args.experiment_config
    seed_everything(config.seed + robot_idx)

    environment = create_environment(config, robot_idx)
    agent, ws_client, broker = create_agent(args, robot_idx)

    meta = SaveMeta(
        out_dir=args.output_dir,
        robot_idx=robot_idx,
        task_suite_name=config.task_suite_name,
        task_id=robot_idx,
        task_language=environment.task_language,
        control_hz=config.robots[robot_idx].control_hz,
        # Zero-image mock frames aren't worth encoding.
        save_video=config.env != EnvironmentType.MOCK,
    )

    try:
        run_robot(
            environment=environment,
            agent=agent,
            meta=meta,
            broker=broker,
            num_episodes=config.num_trials_per_task,
            time_limit=config.wall_clock_time_limit_s,
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


def main(args: Args) -> None:
    config = args.experiment_config

    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        args.output_dir
        / f"run_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging_config.setup_logging(log_path=log_path, level=logging.INFO)

    control_client: ServerControlClient | None = None
    if config.env != EnvironmentType.MOCK:
        control_client = ServerControlClient(host=args.host, port=args.port)
        control_client.reconfigure_server(args.scheduler_config)
        control_client.reset_server()

    args.to_json(args.output_dir / "experiment_args.json")

    num_robots = len(config.robots)
    logger.info("Launching %d robot process(es)", num_robots)
    processes = [
        multiprocessing.Process(
            target=_run_robot_worker, args=(args, robot_idx), name=f"robot_{robot_idx}"
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

    if control_client is not None:
        history = control_client.fetch_server_metrics()
        (args.output_dir / "server_metrics_history.json").write_text(json.dumps(history, indent=2))

    calculate_metrics(args.output_dir)
    generate_all_plots(args.output_dir)


def cli() -> None:
    """Console-script entrypoint. Sets the multiprocessing start method before
    any worker process is created, then runs ``main``."""
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
