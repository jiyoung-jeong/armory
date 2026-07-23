import pathlib
import statistics

import modal
import numpy as np
from scripts.modal.images import gpu_libero_client_image

from armory_client.action_chunkers import ActionChunkBrokerType, BrokerConfig
from armory_client.client import BidirectionalWebsocket
from armory_client.schemas import Action, Observation
from evaluation.agents.policy_agent import PolicyAgent
from evaluation.runtime import Runtime

app = modal.App("libero-speed")

REMOTE_OUT_DIR = pathlib.Path("/tmp/libero_speed_out")

CONTROL_HZ = 20
MIN_EXECUTION_HORIZON = 1
MAX_EXECUTION_HORIZON = 10


class DummyAgent:
    def __init__(self):
        pass

    def reset(self):
        pass

    def get_action(self, observation: Observation):
        return Action(0, np.zeros(7), 0, 0)


@app.function(image=gpu_libero_client_image, gpu="T4", timeout=600)
def run(server_url: str | None) -> bytes | None:
    import cProfile
    import csv
    import pathlib as _pathlib
    import pstats

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from evaluation.sims.libero.env import LiberoSimEnvironment
    from evaluation.sims.libero.subscribers.saver import Saver

    task_id = 0
    benchmark_dict: dict[str, type[benchmark.Benchmark]] = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Experiment: render natively at 224 (what the policy actually consumes)
    # instead of the usual 256 (see evaluation.sims.libero.utils.LIBERO_ENV_RESOLUTION),
    # to see how much of the render/readback cost is wasted upscaled pixels.
    task_bddl_file = (
        _pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    raw_env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file, camera_heights=224, camera_widths=224
    )
    raw_env.seed(7)

    import os
    import subprocess as _subprocess

    print(f"[env] NVIDIA_DRIVER_CAPABILITIES={os.environ.get('NVIDIA_DRIVER_CAPABILITIES')!r}")
    print(
        "[env] egl_vendor.d="
        + _subprocess.run(
            ["ls", "-la", "/usr/share/glvnd/egl_vendor.d/"],
            capture_output=True,
            text=True,
        ).stdout.replace("\n", " | ")
    )
    find_result = _subprocess.run(
        ["find", "/", "-iname", "*nvidia*egl*", "-o", "-iname", "*EGL_nvidia*"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(f"[env] nvidia egl libs found: {find_result.stdout!r} err={find_result.stderr[:500]!r}")

    # Diagnostic: which GL implementation did EGL actually bind to? If this
    # says "llvmpipe"/"softpipe"/"SWR" instead of an NVIDIA string, the
    # NVIDIA EGL device isn't the one being used despite MUJOCO_GL=egl.
    try:
        from OpenGL import GL as _gl

        print(
            f"[gl] vendor={_gl.glGetString(_gl.GL_VENDOR)} "
            f"renderer={_gl.glGetString(_gl.GL_RENDERER)} "
            f"version={_gl.glGetString(_gl.GL_VERSION)}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[gl] could not query GL strings: {e!r}")
    env = LiberoSimEnvironment(
        env=raw_env,
        task_description=task.language,
        initial_states=np.array([initial_states[0]]),
        max_episode_steps=60,
        control_hz=CONTROL_HZ,
    )

    subscribers = []
    if server_url is not None:
        ws_client = BidirectionalWebsocket(
            robot_id="robot_0",
            host=server_url,
            control_hz=CONTROL_HZ,
        )
        ws_client.connect()
        broker = ActionChunkBrokerType.NAIVE_ASYNC.create(
            BrokerConfig(
                ws_client=ws_client,
                control_hz=CONTROL_HZ,
                min_execution_horizon=MIN_EXECUTION_HORIZON,
                max_execution_horizon=MAX_EXECUTION_HORIZON,
            )
        )
        agent = PolicyAgent(broker=broker)
        saver = Saver(
            out_dir=REMOTE_OUT_DIR,
            environment=env,
            action_chunk_broker=broker,
            task_suite_name="libero_10",
            task_id=task_id,
            task=task,
            robot_idx=0,
        )
        subscribers.append(saver)

    else:
        agent = DummyAgent()
        broker = None

    runtime = Runtime(
        environment=env,
        agent=agent,
        subscribers=subscribers,
        # Unthrottled: max_hz>0 would pace steps at a fixed rate, which floors
        # the measured per-step time at 1/max_hz and hides the true sim cost.
        max_hz=0,
        num_episodes=1,
        max_episode_steps=60,  # type: ignore[attr-defined]
    )
    profiler = cProfile.Profile()
    profiler.enable()
    runtime.run()
    profiler.disable()
    runtime.close()
    # ws_client.close()

    stats = pstats.Stats(profiler)
    print("\n=== cProfile: top 25 by cumulative time ===")
    stats.sort_stats("cumulative").print_stats(25)
    print("\n=== cProfile: top 25 by self (internal) time ===")
    stats.sort_stats("tottime").print_stats(25)

    episode_dir = None
    if server_url is not None:
        episode_dir = next(REMOTE_OUT_DIR.glob("0/0_*"))
        with open(episode_dir / "timestamps.csv") as f:
            timestamps = [float(row["timestamp"]) for row in csv.DictReader(f)]
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
        if deltas:
            print(
                f"[speed] steps={len(deltas)} "
                f"mean={statistics.mean(deltas) * 1000:.2f}ms "
                f"median={statistics.median(deltas) * 1000:.2f}ms "
                f"min={min(deltas) * 1000:.2f}ms "
                f"max={max(deltas) * 1000:.2f}ms "
                f"hz={1 / statistics.mean(deltas):.2f}"
            )

    # Experiment: how much of the render/readback cost is just "2 cameras"?
    # Raw loop bypassing LiberoSimEnvironment/Runtime/Saver entirely, rendering
    # only the agentview camera, to isolate the per-camera render+readback cost.
    import time as _time

    single_cam_env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=224,
        camera_widths=224,
        camera_names=["agentview"],
    )
    single_cam_env.seed(7)
    single_cam_env.reset()
    single_cam_env.set_init_state(initial_states[0])
    dummy_action = [0.0] * 6 + [-1.0]
    for _ in range(10):
        single_cam_env.step(dummy_action)  # settle + reuse already-JIT-warmed controller

    # Diagnostic: poll nvidia-smi in the background while stepping, to see
    # whether the GPU is actually doing any work during "GPU-accelerated" render.
    import subprocess
    import threading

    gpu_util_samples: list[int] = []
    stop_poll = threading.Event()

    def _poll_gpu() -> None:
        while not stop_poll.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=1,
                )
                gpu_util_samples.append(int(out.decode().strip()))
            except Exception:  # noqa: BLE001
                pass
            _time.sleep(0.02)

    poll_thread = threading.Thread(target=_poll_gpu, daemon=True)
    poll_thread.start()

    single_cam_deltas = []
    last = _time.perf_counter()
    for _ in range(60):
        single_cam_env.step(dummy_action)
        now = _time.perf_counter()
        single_cam_deltas.append(now - last)
        last = now

    stop_poll.set()
    poll_thread.join()
    single_cam_env.close()

    if gpu_util_samples:
        print(
            f"[gpu-util] samples={len(gpu_util_samples)} "
            f"max={max(gpu_util_samples)}% mean={statistics.mean(gpu_util_samples):.1f}%"
        )
    else:
        print("[gpu-util] no samples collected (nvidia-smi unavailable?)")
    print(
        f"[single-camera] steps={len(single_cam_deltas)} "
        f"mean={statistics.mean(single_cam_deltas) * 1000:.2f}ms "
        f"median={statistics.median(single_cam_deltas) * 1000:.2f}ms "
        f"min={min(single_cam_deltas) * 1000:.2f}ms "
        f"max={max(single_cam_deltas) * 1000:.2f}ms"
    )

    if episode_dir is None:
        return None
    return (episode_dir / "out.mp4").read_bytes()


@app.local_entrypoint()
def main(server_url: str | None = None):
    video_bytes = run.remote(server_url)
    if video_bytes is None:
        print("No server_url given; skipping video save.")
        return
    out_path = pathlib.Path("data/out.mp4")
    out_path.write_bytes(video_bytes)
    print(f"Saved video to {out_path}")
