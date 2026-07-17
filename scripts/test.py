import pathlib
import statistics

import modal
import numpy as np
from scripts.modal._images import gpu_libero_client_image

from armory_client.action_chunkers.action_chunk_broker import ActionChunkBrokerBase
from armory_client.schemas import Action
from evaluation.runtime.runtime import Runtime

app = modal.App("libero-speed")

REMOTE_OUT_DIR = pathlib.Path("/tmp/libero_speed_out")


class DummyAgent:
    def __init__(self):
        pass

    def reset(self):
        pass

    def get_action(self, observation):
        return Action(
            step=observation.step,
            action=np.zeros(7),
            action_chunk_index=None,
            index_in_chunk=None,
        )


@app.function(image=gpu_libero_client_image, gpu="L40S", timeout=600)
def run() -> bytes:
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
    env = LiberoSimEnvironment(
        env=raw_env,
        task_description=task.language,
        initial_states=np.array([initial_states[0]]),
        max_episode_steps=60,
        control_hz=20,
    )

    saver = Saver(
        out_dir=REMOTE_OUT_DIR,
        environment=env,
        action_chunk_broker=ActionChunkBrokerBase(),
        task_suite_name="libero_10",
        task_id=task_id,
        task=task,
        robot_idx=0,
    )

    runtime = Runtime(
        environment=env,
        agent=DummyAgent(),
        subscribers=[saver],
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

    stats = pstats.Stats(profiler)
    print("\n=== cProfile: top 25 by cumulative time ===")
    stats.sort_stats("cumulative").print_stats(25)
    print("\n=== cProfile: top 25 by self (internal) time ===")
    stats.sort_stats("tottime").print_stats(25)

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

    single_cam_deltas = []
    last = _time.perf_counter()
    for _ in range(60):
        single_cam_env.step(dummy_action)
        now = _time.perf_counter()
        single_cam_deltas.append(now - last)
        last = now
    single_cam_env.close()
    print(
        f"[single-camera] steps={len(single_cam_deltas)} "
        f"mean={statistics.mean(single_cam_deltas) * 1000:.2f}ms "
        f"median={statistics.median(single_cam_deltas) * 1000:.2f}ms "
        f"min={min(single_cam_deltas) * 1000:.2f}ms "
        f"max={max(single_cam_deltas) * 1000:.2f}ms"
    )

    return (episode_dir / "out.mp4").read_bytes()


@app.local_entrypoint()
def main():
    video_bytes = run.remote()
    out_path = pathlib.Path("scripts/out.mp4")
    out_path.write_bytes(video_bytes)
    print(f"Saved video to {out_path}")
