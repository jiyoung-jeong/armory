import datetime
import json
import logging
import multiprocessing
import pathlib
import queue
import shutil
import sys
import time
from dataclasses import dataclass
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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import JsonArgs  # noqa: E402

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
RESIZE_SIZE = 224


@dataclass(frozen=True)
class ExecutionHorizon:
    min: int
    max: int


@dataclass(frozen=True)
class ExperimentSettings:
    env: Literal["libero", "mock"]
    task_suite_name: str
    num_trials_per_task: int
    max_steps: int
    num_robots: int
    control_hz: int
    action_chunk_broker_type: ActionChunkBrokerType
    execution_horizons: list[ExecutionHorizon]
    # New "trial" mode: when wall_clock_time_limit_s > 0, the seed picks
    # ``subset_size`` tasks from the suite (0 = all tasks), each robot is
    # pinned to one of those tasks, and runs episodes back-to-back until
    # its per-robot wall-clock budget is exhausted. ``max_steps`` still
    # caps each individual episode.
    subset_size: int = 0
    wall_clock_time_limit_s: float = 0.0

    @property
    def use_trial_mode(self) -> bool:
        return self.wall_clock_time_limit_s > 0.0

    @classmethod
    def from_config(cls, experiment_config: dict[str, object]) -> "ExperimentSettings":
        experiment = experiment_config["experiment"]
        robots = experiment_config["robots"]
        if not isinstance(experiment, dict) or not isinstance(robots, dict):
            raise ValueError("Experiment config is malformed")

        num_robots = int(experiment["num_robots"])
        execution_horizons = []
        for idx in range(num_robots):
            robot_cfg = robots[f"robot_{idx}"]
            execution_horizons.append(
                ExecutionHorizon(
                    min=int(robot_cfg["min_execution_horizon"]),
                    max=int(robot_cfg["max_execution_horizon"]),
                )
            )

        env = str(experiment["env"])
        if env not in ("libero", "mock"):
            raise ValueError(f"Invalid env in experiment config: {env}")

        return cls(
            env=env,
            task_suite_name=str(experiment["task_suite_name"]),
            num_trials_per_task=int(experiment.get("trials_per_robot", 1)),
            max_steps=int(experiment["max_steps"]),
            num_robots=num_robots,
            control_hz=int(experiment["control_hz"]),
            action_chunk_broker_type=ActionChunkBrokerType.from_string(
                str(experiment["action_chunk_broker_type"])
            ),
            execution_horizons=execution_horizons,
            subset_size=int(experiment.get("subset_size", 0)),
            wall_clock_time_limit_s=float(experiment.get("wall_clock_time_limit_s", 0.0)),
        )

    def execution_horizon_for_robot(self, robot_idx: int) -> ExecutionHorizon:
        return self.execution_horizons[robot_idx]

    def max_execution_horizons(self) -> list[int]:
        return [h.max for h in self.execution_horizons]


@dataclass
class Args(JsonArgs):
    json_path: pathlib.Path | None = None
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8080

    #################################################################################################################
    # Network emulation parameters
    #################################################################################################################
    experiment_config: str = ""
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

    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _serialize(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "experiment_config": self.experiment_config,
            "toxiproxy_server_bin": self.toxiproxy_server_bin,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "overwrite": self.overwrite,
            "progress_type": self.progress_type,
            "log_dir": str(self.log_dir) if self.log_dir is not None else None,
            "debug": self.debug,
        }

    @classmethod
    def _deserialize(cls, data: dict) -> "Args":
        kwargs = dict(data)
        if "output_dir" in kwargs:
            kwargs["output_dir"] = pathlib.Path(kwargs["output_dir"])
        if "log_dir" in kwargs and kwargs["log_dir"] is not None:
            kwargs["log_dir"] = pathlib.Path(kwargs["log_dir"])
        return cls(**kwargs)


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
    settings: ExperimentSettings
    server_metadata: ServerMetadata
    robot_idx: int
    # In trial mode, the task this robot is pinned to. ``None`` outside trial mode.
    assigned_task_id: int | None = None


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
    """Worker process. Behaviour depends on ``settings.use_trial_mode``:

    - **Trial mode** (wall_clock_time_limit_s > 0): the robot is pinned to a
      single ``assigned_task_id`` for the entire wall-clock budget. The LIBERO
      ``raw_env`` is created once and reused across episodes (skips the
      expensive BDDL load) — only the per-episode ``LiberoSimEnvironment``
      wrapper is rebuilt.
    - **Legacy mode**: pull episodes from the shared queue until empty.
    """
    args = worker_args.args
    settings = worker_args.settings
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
        control_hz=float(settings.control_hz),
        pre_send_hook=pre_send_hook,
    )
    execution_horizon = settings.execution_horizon_for_robot(robot_idx)
    config = BrokerConfig(
        ws_client=ws_client,
        control_hz=settings.control_hz,
        min_execution_horizon=execution_horizon.min,
        max_execution_horizon=execution_horizon.max,
    )
    broker = settings.action_chunk_broker_type.create(config)
    agent = _policy_agent.PolicyAgent(broker=broker)

    LiberoSimEnvironment = None  # noqa: N806
    libero_utils = None
    task_suite = None
    if settings.env == "libero":
        from libero.libero import benchmark

        from sims.libero import utils as libero_utils  # noqa: F811
        from sims.libero.env import LiberoSimEnvironment  # noqa: F811

        benchmark_dict: dict[str, type[benchmark.Benchmark]] = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[settings.task_suite_name]()

    # Single instance reused across episodes so _done persists across iterations.
    startup_sync = _StartupSyncSubscriber()

    def _build_subscribers(episode: Episode, env: Any) -> list[_subscriber.Subscriber]:
        subs: list[_subscriber.Subscriber] = [
            startup_sync,
            Saver(
                out_dir=args.output_dir,
                environment=env,
                action_chunk_broker=broker,
                task_suite_name=episode.task_suite_name,
                task_id=episode.task_id,
                task=episode.task,
                robot_idx=robot_idx,
                save_video=settings.env != "mock",
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
            subs.append(
                ProgressSubscriber(
                    queue=_progress_queue,
                    robot_idx=robot_idx,
                    episode=episode,
                    environment=env,
                    update_frequency=10,
                )
            )
        return subs

    def _run_one(env: Any, subscribers: list[_subscriber.Subscriber]) -> None:
        runtime = _runtime.Runtime(
            environment=env,
            agent=agent,
            subscribers=subscribers,
            max_hz=settings.control_hz,
            num_episodes=1,
            max_episode_steps=env._max_episode_steps,  # type: ignore[attr-defined]
        )
        runtime.run()
        runtime.close()

    try:
        if settings.use_trial_mode:
            _trial_loop(
                args=args,
                settings=settings,
                robot_idx=robot_idx,
                worker_args=worker_args,
                task_suite=task_suite,
                libero_utils=libero_utils,
                LiberoSimEnvironment=LiberoSimEnvironment,
                build_subscribers=_build_subscribers,
                run_one=_run_one,
            )
        else:
            while True:
                try:
                    episode = _episode_queue.get_nowait()
                except queue.Empty:
                    break

                if settings.env == "libero":
                    raw_env, _ = libero_utils._get_libero_env(
                        task_suite.get_task(episode.task_id),
                        seed=args.seed + robot_idx,
                    )
                    env = LiberoSimEnvironment(
                        env=raw_env,
                        task_description=episode.task.language,
                        initial_states=np.array([episode.initial_state]),
                        resize_size=RESIZE_SIZE,
                        max_episode_steps=settings.max_steps,
                        control_hz=settings.control_hz,
                    )
                elif settings.env == "mock":
                    env = MockEnvironment(
                        max_episode_steps=settings.max_steps,
                        control_hz=settings.control_hz,
                        task_id=episode.task_id,
                        episode_idx=episode.idx,
                    )
                else:
                    raise ValueError(f"Invalid environment: {settings.env}")

                _run_one(env, _build_subscribers(episode, env))
    finally:
        if network_hook is not None:
            network_hook.close()


def _trial_loop(
    *,
    args: "Args",
    settings: ExperimentSettings,
    robot_idx: int,
    worker_args: _WorkerArgs,
    task_suite: Any,
    libero_utils: Any,
    LiberoSimEnvironment: Any,  # noqa: N803
    build_subscribers,
    run_one,
) -> None:
    """Trial-mode loop: pinned task, env reuse, wall-clock budget."""
    task_id = worker_args.assigned_task_id
    if task_id is None:
        raise RuntimeError(
            f"robot {robot_idx}: missing assigned_task_id in trial mode"
        )

    raw_env = None
    initial_states: np.ndarray
    if settings.env == "libero":
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        raw_env, _ = libero_utils._get_libero_env(task, seed=args.seed + robot_idx)

        # No-op close so the shared raw_env survives across iterations.
        class _ReusableLiberoEnv(LiberoSimEnvironment):  # type: ignore[misc, valid-type]
            def close(self) -> None:  # noqa: D401
                return None

    else:
        from sims.libero.episodes import _MockTask

        task = _MockTask(language=f"mock task {task_id}")
        initial_states = np.zeros((1, 1), dtype=np.float32)

    try:
        start_t = time.monotonic()
        ep_idx = 0
        while time.monotonic() - start_t < settings.wall_clock_time_limit_s:
            state = initial_states[ep_idx % len(initial_states)]
            episode = Episode(
                idx=ep_idx + 1,
                task_suite_name=settings.task_suite_name,
                task_id=task_id,
                task=task,
                initial_state=state,
            )
            if settings.env == "libero":
                env = _ReusableLiberoEnv(
                    env=raw_env,
                    task_description=task.language,
                    initial_states=np.array([state]),
                    resize_size=RESIZE_SIZE,
                    max_episode_steps=settings.max_steps,
                    control_hz=settings.control_hz,
                )
            else:
                env = MockEnvironment(
                    max_episode_steps=settings.max_steps,
                    control_hz=settings.control_hz,
                    task_id=task_id,
                    episode_idx=episode.idx,
                )
            run_one(env, build_subscribers(episode, env))
            ep_idx += 1
        logging.info(
            "robot_%d: completed %d episode(s) on task_id=%d within %.1fs budget",
            robot_idx,
            ep_idx,
            task_id,
            settings.wall_clock_time_limit_s,
        )
    finally:
        if raw_env is not None:
            try:
                raw_env.close()
            except Exception:  # noqa: BLE001
                pass


def run_robots(
    args: Args,
    settings: ExperimentSettings,
    episodes: list[Episode],
    server_metadata: ServerMetadata,
    network_worker_contexts: dict[str, WorkerNetworkContext] | None = None,
    robot_task_assignment: list[int] | None = None,
) -> None:
    trial_mode = settings.use_trial_mode and robot_task_assignment is not None

    if args.debug:
        # Debug mode: single process for pdb compatibility, no progress manager.
        ep_queue: queue.Queue = queue.Queue()
        for ep in episodes:
            ep_queue.put(ep)
        _init_worker_shared(ep_queue, None, None, network_worker_contexts)
        _robot_worker(
            _WorkerArgs(
                args=args,
                settings=settings,
                server_metadata=server_metadata,
                robot_idx=0,
                assigned_task_id=(
                    robot_task_assignment[0] if trial_mode else None
                ),
            )
        )
    else:
        if trial_mode:
            active_workers = settings.num_robots
            # In trial mode the unit of progress is "one robot finished its
            # wall-clock budget" rather than "one episode in the queue".
            total_episodes = active_workers
        else:
            total_episodes = len(episodes)
            active_workers = min(settings.num_robots, total_episodes)
        start_barrier = multiprocessing.Barrier(active_workers, timeout=60)
        logging.info("Using one-time startup barrier across %d worker(s)", active_workers)

        mp_episode_queue: multiprocessing.Queue = multiprocessing.Queue()
        for ep in episodes:
            mp_episode_queue.put(ep)

        with get_progress_manager(
            args.progress_type,
            total_episodes=total_episodes,
            max_steps=settings.max_steps,
        ) as progress_manager:
            worker_args = [
                _WorkerArgs(
                    args=args,
                    settings=settings,
                    server_metadata=server_metadata,
                    robot_idx=i,
                    assigned_task_id=(robot_task_assignment[i] if trial_mode else None),
                )
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
        if isinstance(notes, dict):
            notes = dict(notes)
            if "next_server_available" in notes:
                notes["next_server_available"] = shift(notes["next_server_available"])
            phases = notes.get("phases")
            if isinstance(phases, list):
                notes["phases"] = [
                    {**ph, "start": shift(ph.get("start", 0.0)), "end": shift(ph.get("end", 0.0))}
                    for ph in phases
                    if isinstance(ph, dict)
                ]
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


def validate_args(args: Args, settings: ExperimentSettings) -> None:
    assert args.overwrite or not args.output_dir.exists(), (
        f"Output path {args.output_dir} already exists"
    )
    assert args.experiment_config, "experiment_config is required"
    assert settings.num_robots > 0, "num_robots must be positive"
    if not settings.use_trial_mode:
        assert settings.num_trials_per_task > 0, "num_trials_per_task must be positive"
    assert settings.max_steps > 0, "max_steps must be positive"
    assert settings.control_hz > 0, "control_hz must be positive"
    if settings.use_trial_mode:
        assert settings.wall_clock_time_limit_s > 0.0, (
            "wall_clock_time_limit_s must be positive in trial mode"
        )
    assert len(settings.execution_horizons) == settings.num_robots
    for idx, horizon in enumerate(settings.execution_horizons):
        assert horizon.min >= 0, f"robot_{idx}.min_execution_horizon must be non-negative"
        assert horizon.max > 0, f"robot_{idx}.max_execution_horizon must be positive"
        assert horizon.min <= horizon.max, (
            f"robot_{idx}.min_execution_horizon must be <= max_execution_horizon"
        )
    assert args.seed >= 0, "seed must be non-negative"


def main(args: Args) -> None:
    if args.json_path is not None:
        args = Args.from_json(args.json_path)
    experiment_config = load_experiment_config(args.experiment_config)
    settings = ExperimentSettings.from_config(experiment_config)
    if settings.use_trial_mode:
        logging.info(
            "Loaded experiment config from %s: env=%s mode=%s num_robots=%d "
            "subset_size=%d wall_clock_time_limit_s=%.1f",
            args.experiment_config,
            settings.env,
            settings.action_chunk_broker_type.value,
            settings.num_robots,
            settings.subset_size,
            settings.wall_clock_time_limit_s,
        )
    else:
        logging.info(
            "Loaded experiment config from %s: env=%s mode=%s num_robots=%d trials_per_robot=%d",
            args.experiment_config,
            settings.env,
            settings.action_chunk_broker_type.value,
            settings.num_robots,
            settings.num_trials_per_task,
        )

    validate_args(args, settings)

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

    robot_task_assignment: list[int] | None = None
    subset_task_ids: list[int] = []
    if settings.use_trial_mode:
        from sims.libero.episodes import (
            _MockTask,
            assign_robots_to_tasks,
            pick_subset_task_ids,
        )

        if settings.env == "libero":
            subset_task_ids = pick_subset_task_ids(
                settings.task_suite_name, settings.subset_size, args.seed
            )
        else:
            # For mock env, "task ids" are synthetic. Pick the first
            # `subset_size or 1` ids deterministically.
            n = max(1, settings.subset_size or 1)
            subset_task_ids = list(range(n))

        robot_task_assignment = assign_robots_to_tasks(
            settings.num_robots, subset_task_ids
        )
        # Trial-mode workers generate episodes inline based on their assigned
        # task. The list below is only used downstream for runtime_metadata.
        episodes = [
            Episode(
                idx=robot_idx + 1,
                task_suite_name=settings.task_suite_name,
                task_id=task_id,
                task=_MockTask(language=f"trial mode placeholder task_id={task_id}"),
                initial_state=np.zeros(1, dtype=np.float32),
            )
            for robot_idx, task_id in enumerate(robot_task_assignment)
        ]
        logging.info(
            "Trial mode: seed=%d subset_task_ids=%s assignment=%s budget=%.1fs",
            args.seed,
            subset_task_ids,
            robot_task_assignment,
            settings.wall_clock_time_limit_s,
        )
    else:
        if settings.env == "libero":
            episodes = create_episodes(settings.task_suite_name, settings.num_trials_per_task)
        else:
            episodes = create_mock_episodes(settings.num_trials_per_task * settings.num_robots)

    server_metadata = fetch_server_metadata(args)
    if settings.use_trial_mode:
        active_workers = 1 if args.debug else settings.num_robots
    else:
        active_workers = 1 if args.debug else min(settings.num_robots, len(episodes))

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
        task_suite_name=settings.task_suite_name,
        num_trials_per_task=settings.num_trials_per_task,
        max_steps=settings.max_steps,
        num_robots=settings.num_robots,
        control_hz=settings.control_hz,
        broker_type=settings.action_chunk_broker_type.value,
        seed=args.seed,
        resize_size=RESIZE_SIZE,
        episodes=[str(ep) for ep in episodes],
        max_execution_horizon=settings.max_execution_horizons(),
    )

    runtime_metadata.to_json(args.output_dir / "runtime_metadata.json")
    logging.info(f"Saved runtime metadata to {args.output_dir / 'runtime_metadata.json'}")

    server_metadata.to_json(args.output_dir / "server_metadata.json")
    logging.info(f"Saved server metadata to {args.output_dir / 'server_metadata.json'}")

    reset_server(args)
    try:
        run_robots(
            args,
            settings,
            episodes,
            server_metadata,
            network_worker_contexts=network_worker_contexts,
            robot_task_assignment=robot_task_assignment,
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
