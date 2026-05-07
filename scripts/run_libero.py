import datetime
import json
import logging
import multiprocessing
import pathlib
import queue
import shutil
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
)  # Any used for shared globals

import numpy as np
import requests
import tyro

from armory_client.action_chunkers import ActionChunkBrokerType, BrokerConfig
from armory_client.client import BidirectionalWebsocket
from armory_client.network_emulation import (
    NetworkEmulationManager,
    RobotNetworkHook,
    WorkerNetworkContext,
    experiment_requires_network_emulation,
    load_experiment_config,
)
from armory_client.runtime import runtime as _runtime
from armory_client.runtime import subscriber as _subscriber
from armory_client.runtime.agents import policy_agent as _policy_agent
from armory_client.schemas import RuntimeMetadata, ServerMetadata
from sims.libero import logging_config
from sims.libero.episodes import Episode, create_episodes, create_mock_episodes
from sims.libero.metrics import calculate_metrics, generate_all_plots
from sims.libero.mock_env import MockEnvironment
from sims.libero.progress_manager import get_progress_manager
from sims.libero.seeding import seed_everything
from sims.libero.subscribers.progress_subscriber import ProgressSubscriber
from sims.libero.subscribers.saver import Saver
from sims.libero.subscribers.task_metrics_publisher import TaskMetricsPublisher

logger = logging.getLogger(__name__)


@dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8080
    resize_size: int = 224
    action_chunk_broker_type: ActionChunkBrokerType = ActionChunkBrokerType.NAIVE_ASYNC
    execution_horizon: list[int] = field(default_factory=list)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env: Literal["libero", "mock"] = "libero"
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 10  # Number of rollouts per task
    max_steps: int = 600  # Maximum number of control steps per episode

    #################################################################################################################
    # Multi-robot / threading parameters
    #################################################################################################################
    num_robots: int = 5  # Number of always-running sims (robots)
    control_hz: int = (
        20  # Target control frequency for each sim #NOTE: int because this is the fps of the video
    )

    #################################################################################################################
    # Network emulation parameters
    #################################################################################################################
    experiment_config: str | None = None
    toxiproxy_server_bin: str | None = "../toxiproxy-server-linux-amd64"

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)
    output_dir: pathlib.Path = pathlib.Path("data/libero/multi_robot_videos")
    overwrite: bool = False
    progress_type: Literal["verbose", "concise", "logging", None] = "verbose"
    log_dir: pathlib.Path | None = None
    debug: bool = False  # Run in single process with immediate progress output

    def execution_horizon_for_robot(self, robot_idx: int) -> int:
        if not self.execution_horizon:
            return 10
        return int(self.execution_horizon[robot_idx])

    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.port}"


def _apply_experiment_config(args: Args, experiment_config: dict[str, object]) -> None:
    """Apply experiment config settings onto runtime args."""
    experiment = experiment_config["experiment"]
    robots = experiment_config["robots"]
    if not isinstance(experiment, dict) or not isinstance(robots, dict):
        raise ValueError("Experiment config is malformed")

    args.action_chunk_broker_type = ActionChunkBrokerType.from_string(
        str(experiment["action_chunk_broker_type"])
    )
    args.num_robots = int(experiment["num_robots"])
    args.num_trials_per_task = int(experiment["trials_per_robot"])
    args.execution_horizon = [
        int(robots[f"robot_{idx}"]["execution_horizon"]) for idx in range(args.num_robots)
    ]


# Shared worker state: set via pool initializer so these are inherited by spawned
# processes rather than pickled as task arguments (multiprocessing.Queue and Barrier
# cannot be pickled after spawning).
_episode_queue: Any | None = None
_progress_queue: Any | None = None
_start_barrier: Any | None = None
_network_worker_contexts: dict[str, WorkerNetworkContext] | None = None


def _init_worker_shared(
    episode_queue,
    progress_queue,
    start_barrier,
    network_worker_contexts: dict[str, WorkerNetworkContext] | None = None,
) -> None:
    global _episode_queue, _progress_queue, _start_barrier, _network_worker_contexts
    _episode_queue = episode_queue
    _progress_queue = progress_queue
    _start_barrier = start_barrier
    _network_worker_contexts = network_worker_contexts


@dataclass
class _WorkerArgs:
    args: Args
    server_metadata: ServerMetadata
    robot_idx: int


class _StartupSyncSubscriber(_subscriber.Subscriber):
    """One-shot startup synchronization right before first episode steps."""

    def __init__(self) -> None:
        self._done = False

    def on_episode_start(self) -> None:
        if self._done:
            return
        if _start_barrier is not None:
            _start_barrier.wait()
        self._done = True
        # Notify the progress manager that this worker has crossed the start barrier.
        # The manager sets its start_time on the first such message it receives.
        if _progress_queue is not None:
            try:
                _progress_queue.put_nowait({"type": "run_start"})
            except Exception:
                pass

    def on_step(self, observation, action) -> None:
        return

    def on_episode_end(self) -> None:
        return


def _robot_worker(worker_args: _WorkerArgs) -> None:
    """Worker process: initialize, then pull episodes from the shared queue until empty."""
    args = worker_args.args
    robot_idx = worker_args.robot_idx
    robot_id = f"robot_{robot_idx}"

    # Stagger startup to avoid flooding the server with simultaneous warmup.
    time.sleep(robot_idx * 0.5)

    ws_host = args.host
    ws_port = args.port
    pre_send_hook = None
    network_hook = None
    if _network_worker_contexts is not None:
        context = _network_worker_contexts.get(robot_id)
        if context is None:
            raise RuntimeError(f"Missing network context for worker robot_id={robot_id}")
        if bool(context.get("emulate_network", True)):
            ws_host = str(context["proxy_host"])
            ws_port = int(context["proxy_port"])
            network_hook = RobotNetworkHook(context)
            pre_send_hook = network_hook.before_send

    ws_client = BidirectionalWebsocket(
        robot_id=robot_id,
        host=ws_host,
        port=ws_port,
        control_hz=float(args.control_hz),
        pre_send_hook=pre_send_hook,
    )
    config = BrokerConfig(
        ws_client=ws_client,
        control_hz=args.control_hz,
        execution_horizon=args.execution_horizon_for_robot(robot_idx),
    )
    broker = args.action_chunk_broker_type.create(config)
    agent = _policy_agent.PolicyAgent(broker=broker)

    if args.env == "libero":
        from libero.libero import benchmark

        from sims.libero import utils as libero_utils
        from sims.libero.env import LiberoSimEnvironment

        benchmark_dict: dict[str, type[benchmark.Benchmark]] = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()

    # Single instance reused across episodes so _done persists across iterations.
    startup_sync = _StartupSyncSubscriber()

    try:
        while True:
            try:
                episode = _episode_queue.get_nowait()
            except queue.Empty:
                break

            if args.env == "libero":
                raw_env, _ = libero_utils._get_libero_env(
                    task_suite.get_task(episode.task_id),
                    seed=args.seed + robot_idx,
                )
                env = LiberoSimEnvironment(
                    env=raw_env,
                    task_description=episode.task.language,
                    initial_states=np.array([episode.initial_state]),
                    resize_size=args.resize_size,
                    max_episode_steps=args.max_steps,
                    control_hz=args.control_hz,
                )
            elif args.env == "mock":
                env = MockEnvironment(
                    max_episode_steps=args.max_steps,
                    control_hz=args.control_hz,
                    task_id=episode.task_id,
                    episode_idx=episode.idx,
                )
            else:
                raise ValueError(f"Invalid environment: {args.env}")

            subscribers: list[_subscriber.Subscriber] = [
                startup_sync,
                Saver(
                    out_dir=args.output_dir,
                    environment=env,
                    action_chunk_broker=broker,
                    task_suite_name=episode.task_suite_name,
                    task_id=episode.task_id,
                    task=episode.task,
                    robot_idx=robot_idx,
                    save_video=args.env != "mock",
                ),
                TaskMetricsPublisher(
                    ws_client=ws_client,
                    environment=env,
                    task_suite_name=episode.task_suite_name,
                    task_id=episode.task_id,
                    task=episode.task,
                ),
            ]
            if _progress_queue is not None:
                subscribers.append(
                    ProgressSubscriber(
                        queue=_progress_queue,
                        robot_idx=robot_idx,
                        episode=episode,
                        environment=env,
                        update_frequency=10,
                    )
                )

            runtime = _runtime.Runtime(
                environment=env,
                agent=agent,
                subscribers=subscribers,
                max_hz=args.control_hz,
                num_episodes=1,
                max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
            )
            runtime.run()
            runtime.close()
    finally:
        if network_hook is not None:
            network_hook.close()


def run_robots(
    args: Args,
    episodes: list[Episode],
    server_metadata: ServerMetadata,
    network_worker_contexts: dict[str, WorkerNetworkContext] | None = None,
) -> None:
    if args.debug:
        # Debug mode: single process for pdb compatibility, no progress manager.
        ep_queue: queue.Queue = queue.Queue()
        for ep in episodes:
            ep_queue.put(ep)
        _init_worker_shared(ep_queue, None, None, network_worker_contexts)
        _robot_worker(_WorkerArgs(args=args, server_metadata=server_metadata, robot_idx=0))
    else:
        total_episodes = len(episodes)
        active_workers = min(args.num_robots, total_episodes)
        start_barrier = multiprocessing.Barrier(active_workers, timeout=60)
        logging.info("Using one-time startup barrier across %d worker(s)", active_workers)

        mp_episode_queue: multiprocessing.Queue = multiprocessing.Queue()
        for ep in episodes:
            mp_episode_queue.put(ep)

        with get_progress_manager(
            args.progress_type,
            total_episodes=total_episodes,
            max_steps=args.max_steps,
        ) as progress_manager:
            worker_args = [
                _WorkerArgs(args=args, server_metadata=server_metadata, robot_idx=i)
                for i in range(active_workers)
            ]
            with multiprocessing.Pool(
                processes=active_workers,
                initializer=_init_worker_shared,
                initargs=(
                    mp_episode_queue,
                    progress_manager.queue,
                    start_barrier,
                    network_worker_contexts,
                ),
            ) as pool:
                try:
                    # use imap_unordered so that exceptions surface immediately
                    for _ in pool.imap_unordered(_robot_worker, worker_args):
                        pass
                except Exception as e:
                    logging.error(f"Error in robot worker: {e}")
                    raise
                finally:
                    pool.close()
                    pool.join()


def fetch_server_metadata(args: Args, timeout_s: float = 300.0) -> ServerMetadata:
    """Fetch server metadata, retrying until timeout_s seconds have elapsed."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            resp = requests.get(f"{args.http_base}/metadata", timeout=5.0)
            resp.raise_for_status()
            return ServerMetadata(**resp.json())
        except Exception as e:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Server at {args.http_base} did not respond within {timeout_s:.0f}s"
                ) from e
            logging.info("Waiting for server to be ready (%s); retrying in 5s...", e)
            time.sleep(5.0)


def reset_server(args: Args) -> None:
    try:
        requests.post(f"{args.http_base}/reset", timeout=5.0)
        logging.info("Reset server metrics")
    except Exception as e:
        logging.warning(f"Could not reset server metrics: {e}")


def _normalize_metrics_times(history: dict) -> dict:
    """Subtract start_time from all absolute timestamps for readability."""
    t0 = history.get("start_time", 0.0)
    if t0 == 0.0 or t0 == float("inf"):
        return history

    def shift(v: float) -> float:
        return round(v - t0, 6) if v and v > 0 else v

    history = dict(history)
    history["start_time"] = 0.0
    history["end_time"] = shift(history.get("end_time", 0.0))

    normalized_batches = []
    for b in history.get("batches", []):
        if isinstance(b, dict):
            b = dict(b)
            b["inference_start_time"] = shift(b.get("inference_start_time", 0.0))
            b["inference_end_time"] = shift(b.get("inference_end_time", 0.0))
        else:
            # NamedTuple serialized as list: [batch_id, robot_ids, request_ids, inference_start_time, inference_end_time, ...]
            b = list(b)
            b[3] = shift(b[3])
            b[4] = shift(b[4])
        normalized_batches.append(b)
    history["batches"] = normalized_batches

    normalized_robots = {}
    for robot_id, robot in history.get("robots", {}).items():
        robot = dict(robot)
        normalized_episodes = []
        for ep in robot.get("episodes", []):
            ep = dict(ep)
            ep["requests"] = [
                {
                    **r,
                    "request_timestamp": shift(r["request_timestamp"]),
                    "server_arrival_time": shift(r["server_arrival_time"]),
                }
                for r in ep.get("requests", [])
            ]
            normalized_responses = []
            for resp in ep.get("responses", []):
                resp = dict(resp)
                req = dict(resp.get("request", {}))
                req["request_timestamp"] = shift(req.get("request_timestamp", 0.0))
                req["server_arrival_time"] = shift(req.get("server_arrival_time", 0.0))
                resp["request"] = req
                resp["inference_start_time"] = shift(resp.get("inference_start_time", 0.0))
                resp["inference_end_time"] = shift(resp.get("inference_end_time", 0.0))
                if resp.get("server_send_time", 0.0) > 0:
                    resp["server_send_time"] = shift(resp["server_send_time"])
                if resp.get("receive_time", 0.0) > 0:
                    resp["receive_time"] = shift(resp["receive_time"])
                normalized_responses.append(resp)
            ep["responses"] = normalized_responses
            ep["step_timestamps"] = [shift(ts) for ts in ep.get("step_timestamps", [])]
            normalized_episodes.append(ep)
        robot["episodes"] = normalized_episodes
        normalized_robots[robot_id] = robot
    history["robots"] = normalized_robots

    normalized_decisions = []
    for d in history.get("scheduler_decisions", []):
        d = dict(d)
        d["started_at"] = shift(d.get("started_at", 0.0))
        d["next_server_available"] = shift(d.get("next_server_available", 0.0))
        d["deadlines"] = {k: shift(v) for k, v in d.get("deadlines", {}).items()}
        notes = d.get("notes")
        if isinstance(notes, dict) and "next_server_available" in notes:
            notes = dict(notes)
            notes["next_server_available"] = shift(notes["next_server_available"])
            d["notes"] = notes
        normalized_decisions.append(d)
    history["scheduler_decisions"] = normalized_decisions

    return history


def save_server_metrics_history(args: Args) -> None:
    try:
        history = requests.get(f"{args.http_base}/save-metrics", timeout=10.0).json()
        history = _normalize_metrics_times(history)
        hist_path = args.output_dir / "server_metrics_history.json"
        hist_path.write_text(json.dumps(history, indent=2))
        logging.info(f"Saved server metrics history to {hist_path}")
    except Exception as e:
        logging.warning(f"Could not fetch server metrics history: {e}", exc_info=True)


def validate_args(args: Args) -> None:
    assert args.overwrite or not args.output_dir.exists(), (
        f"Output path {args.output_dir} already exists"
    )
    assert not args.execution_horizon or len(args.execution_horizon) == args.num_robots, (
        f"execution_horizon must either be empty or have exactly {args.num_robots} values (one per robot), but got {len(args.execution_horizon)} values"
    )
    assert args.num_robots > 0, "num_robots must be positive"
    assert args.num_trials_per_task > 0, "num_trials_per_task must be positive"
    assert args.max_steps > 0, "max_steps must be positive"
    assert args.seed >= 0, "seed must be non-negative"
    assert args.resize_size > 0, "resize_size must be positive"


def main(args: Args) -> None:
    experiment_config = None
    if args.experiment_config is not None:
        experiment_config = load_experiment_config(args.experiment_config)
        _apply_experiment_config(args, experiment_config)
        logging.info(
            "Loaded experiment config from %s: mode=%s num_robots=%d trials_per_robot=%d",
            args.experiment_config,
            args.action_chunk_broker_type.value,
            args.num_robots,
            args.num_trials_per_task,
        )

    validate_args(args)

    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_dir is not None:
        log_file_name = f"libero_multi_robot_runtime_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}.log"
        log_file_path = args.log_dir / log_file_name
        args.log_dir.mkdir(parents=True, exist_ok=True)
        logging_config.setup_logging(
            log_path=log_file_path, level=logging.DEBUG if args.debug else logging.INFO
        )
    else:
        logging_config.setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    seed_everything(args.seed)
    if args.env == "libero":
        episodes = create_episodes(args.task_suite_name, args.num_trials_per_task)
    else:
        episodes = create_mock_episodes(args.num_trials_per_task * args.num_robots)

    server_metadata = fetch_server_metadata(args)
    active_workers = 1 if args.debug else min(args.num_robots, len(episodes))

    network_manager = None
    network_worker_contexts: dict[str, WorkerNetworkContext] | None = None
    if experiment_config is not None:
        if experiment_requires_network_emulation(experiment_config, worker_count=active_workers):
            if not args.toxiproxy_server_bin:
                raise ValueError(
                    "--toxiproxy-server-bin is required when experiment config enables network emulation"
                )
            network_output_dir = args.output_dir / "network_emulation"
            network_manager = NetworkEmulationManager(
                experiment_config,
                toxiproxy_server_bin=str(args.toxiproxy_server_bin),
                upstream_host=args.host,
                upstream_port=args.port,
                worker_count=active_workers,
                output_dir=network_output_dir,
            )
            try:
                network_worker_contexts = network_manager.start()
            except Exception:
                network_manager.close()
                raise
            logging.info(
                "Network emulation enabled for %d worker(s)",
                sum(
                    1
                    for context in network_worker_contexts.values()
                    if bool(context.get("emulate_network", True))
                ),
            )
        else:
            logging.info(
                "Network emulation disabled: all active robots have zero uplink/downlink medians and sigmas"
            )

    runtime_metadata = RuntimeMetadata(
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        max_steps=args.max_steps,
        num_robots=args.num_robots,
        control_hz=args.control_hz,
        broker_type=args.action_chunk_broker_type.value,
        seed=args.seed,
        resize_size=args.resize_size,
        episodes=[str(ep) for ep in episodes],
        execution_horizon=args.execution_horizon,
    )

    runtime_metadata.to_json(args.output_dir / "runtime_metadata.json")
    logging.info(f"Saved runtime metadata to {args.output_dir / 'runtime_metadata.json'}")

    server_metadata.to_json(args.output_dir / "server_metadata.json")
    logging.info(f"Saved server metadata to {args.output_dir / 'server_metadata.json'}")

    reset_server(args)
    try:
        run_robots(
            args,
            episodes,
            server_metadata,
            network_worker_contexts=network_worker_contexts,
        )
    finally:
        if network_manager is not None:
            network_manager.close()

    save_server_metrics_history(args)

    calculate_metrics(args.output_dir)
    generate_all_plots(args.output_dir)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # allows multiple processes with envs
    main(tyro.cli(Args))
