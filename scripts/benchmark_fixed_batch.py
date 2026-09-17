"""Capture actual LIBERO observations, then time materialized SYNC infer_batch calls.

Run capture and run as separate processes so EGL is gone before model timing.
The fixed/dynamic phases exclude serving, scheduling, networking and rendering.
An optional off/on/off control adds four concurrent LIBERO environments.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import time
from pathlib import Path

import numpy as np
from scripts.local_batch_sweep import ROOT, gpu_processes


def configure(gpu):
    if gpu_processes(gpu):
        raise RuntimeError(f"GPU {gpu} is occupied")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENPI_DATA_HOME=str(ROOT / ".cache/openpi-private/cache"),
        MUJOCO_EGL_DEVICE_ID=str(gpu),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        LIBERO_CONFIG_PATH=str(ROOT / ".cache/libero"),
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )


def capture(output, tasks=(5, 2, 6, 9)):
    from evaluation.envs.libero import LiberoRobotSpec, LiberoSimEnvironment

    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, task in enumerate(tasks):
        env = LiberoSimEnvironment(
            LiberoRobotSpec("libero_10", task), max_episode_steps=500, seed=7 + index
        )
        try:
            env.reset()
            obs = dataclasses.asdict(env.get_observation())
            path = output / f"robot_{index}.npz"
            np.savez_compressed(path, **obs)
            records.append(
                dict(
                    robot=index,
                    task=task,
                    seed=7 + index,
                    initial_state_index=0,
                    prompt=obs["prompt"],
                    observation_step=obs["step"],
                    arrays={
                        k: dict(shape=list(v.shape), dtype=str(v.dtype))
                        for k, v in obs.items()
                        if isinstance(v, np.ndarray)
                    },
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        finally:
            env.close()
        print(f"Captured task {task}", flush=True)
    (output / "metadata.json").write_text(json.dumps(records, indent=2))


def render_worker(index, ready, stop, results):
    """A controlled background load: real cameras and CPU simulation at 20 Hz.

    Uses null actions, so this is a resource-contention control, not a policy
    success experiment. Inference inputs in the parent stay exactly fixed.
    """
    from armory_client.schemas import Action
    from evaluation.envs.libero import LiberoRobotSpec, LiberoSimEnvironment

    env = LiberoSimEnvironment(
        LiberoRobotSpec("libero_10", [5, 2, 6, 9][index]), max_episode_steps=500, seed=7 + index
    )
    count = 0
    try:
        env.reset()
        ready.put(index)
        begin = time.monotonic()
        next_step = begin
        while not stop.is_set():
            if env.is_episode_complete():
                env.reset()
            obs = env.get_observation()
            env.apply_action(
                Action(
                    step=obs.step,
                    action=np.array([0.0] * 6 + [-1.0]),
                    action_chunk_index=None,
                    index_in_chunk=None,
                )
            )
            count += 1
            next_step = max(next_step + 0.05, time.monotonic())
            stop.wait(max(0, next_step - time.monotonic()))
        results.put(dict(robot=index, steps=count, elapsed_s=time.monotonic() - begin))
    finally:
        env.close()


def renderer_control(args, measure, manifest):
    context = mp.get_context("spawn")
    ready, results = context.Queue(), context.Queue()
    stop = context.Event()
    workers = [
        context.Process(target=render_worker, args=(i, ready, stop, results)) for i in range(4)
    ]
    # Off/on/off blocks use the same compiled model, input and RNG.
    for size in [2, 4]:
        for index in range(args.samples):
            measure(size, "control_before", 0, index)
    try:
        for worker in workers:
            worker.start()
        assert {ready.get(timeout=90) for _ in workers} == set(range(4))
        stop.wait(2)  # All four environments reach steady pacing before timing.
        own = {os.getpid(), *(worker.pid for worker in workers)}
        if set(gpu_processes(args.gpu)) - own:
            raise RuntimeError("Another GPU compute workload appeared")
        for size in [2, 4]:
            for index in range(args.samples):
                if any(not worker.is_alive() for worker in workers):
                    raise RuntimeError("Background renderer exited during measurement")
                measure(size, "four_renderers", 0, index)
            print(f"Renderer control size={size} complete", flush=True)
    finally:
        stop.set()
        for worker in workers:
            if worker.pid is None:
                continue
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
    render_stats = []
    for _ in workers:
        try:
            render_stats.append(results.get(timeout=2))
        except queue.Empty:
            break
    assert len(render_stats) == 4 and all(worker.exitcode == 0 for worker in workers)
    manifest["renderer_control"] = dict(
        workers=render_stats,
        description="Off/on/off; four real LIBERO environments using null actions at 20 Hz, both cameras and observation preprocessing; no WebSocket",
        same_gpu=True,
        model_inputs_unchanged=True,
    )
    for size in [4, 2]:
        for index in range(args.samples):
            measure(size, "control_after", 0, index)
    if set(gpu_processes(args.gpu)) - {os.getpid()}:
        raise RuntimeError("GPU compute processes remain after renderer control")


def run(args):
    import jax

    from armory.backends.types import EnvMode, warmup_request
    from openpi_adapter.serve_factory import create_policy

    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads((args.inputs / "metadata.json").read_text())
    requests = []
    for record in records:
        path = args.inputs / f"robot_{record['robot']}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        with np.load(path, allow_pickle=False) as saved:
            obs = {
                k: (saved[k].item() if saved[k].ndim == 0 else saved[k].copy()) for k in saved.files
            }
        requests.append(
            dataclasses.replace(
                warmup_request(obs),
                min_execution_horizon=1,
                max_execution_horizon=10,
                control_hz=20,
            )
        )
    policy = create_policy(
        "pi05_libero",
        ROOT / ".cache/openpi-private/cache/openpi-assets/checkpoints/pi05_libero",
        sample_kwargs={"num_steps": 10},
        env_mode=EnvMode.LIBERO,
    )
    # Match server's compilation scope, then warm each actual fixed input shape.
    policy.warmup(4)
    rng = jax.random.key(7)
    expected = {}
    for size in range(1, 5):
        for _ in range(5):
            policy._policy._rng = rng
            result = policy.infer_batch(requests[:size])
        expected[size] = [r["actions"].copy() for r in result]
    manifest = dict(
        status="running",
        gpu=args.gpu,
        config="pi05_libero",
        num_steps=10,
        infer_type="SYNC",
        input_dir=str(args.inputs.resolve()),
        inputs=records,
        repeats=args.repeats,
        samples_per_size_per_repeat=args.samples,
        fixed_rng_seed=7,
        rng_reset_outside_timer=True,
        timer="perf_counter_ns",
        timing_scope="Full infer_batch, including transforms, dispatch, NumPy materialization and output transforms",
        renderer_active_in_fixed_and_dynamic=False,
        renderer_control_requested=args.with_renderer_control,
        jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        started_at=time.time(),
        output_shapes={size: list(rows[0].shape) for size, rows in expected.items()},
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.output / "calls.jsonl").open("w") as log:
        counter = 0
        previous = None

        def measure(size, phase, repeat, index):
            nonlocal counter, previous
            # Same RNG key means the same per-shape noise, using the normal sampling path.
            policy._policy._rng = rng
            start_epoch = time.time()
            start = time.perf_counter_ns()
            outputs = policy.infer_batch(requests[:size])
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            # Adapter materializes actions with np.asarray before returning; this
            # check is deliberately outside the timed call.
            assert len(outputs) == size
            for actual, reference in zip(outputs, expected[size], strict=True):
                assert isinstance(actual["actions"], np.ndarray)
                assert np.isfinite(actual["actions"]).all()
                np.testing.assert_allclose(actual["actions"], reference, rtol=1e-5, atol=1e-5)
            log.write(
                json.dumps(
                    dict(
                        index=counter,
                        phase=phase,
                        repeat=repeat,
                        block_index=index,
                        batch_size=size,
                        previous_batch=previous,
                        start_epoch=start_epoch,
                        duration_ms=elapsed_ms,
                    )
                )
                + "\n"
            )
            log.flush()
            counter += 1
            previous = size

        for repeat in range(args.repeats):
            order = [1, 2, 3, 4]
            order = order[repeat % 4 :] + order[: repeat % 4]
            for size in order:
                # Exclude shape transitions from the fixed-shape block.
                for _ in range(3):
                    policy._policy._rng = rng
                    policy.infer_batch(requests[:size])
                for index in range(args.samples):
                    measure(size, "fixed", repeat + 1, index)
                print(f"Fixed repeat={repeat + 1} size={size} complete", flush=True)
                own = {os.getpid()}
                if set(gpu_processes(args.gpu)) - own:
                    raise RuntimeError("Another GPU compute workload appeared")
        # Includes shape switching after every shape was already compiled.
        for cycle in range(args.samples):
            for size in [1, 2, 3, 2]:
                measure(size, "dynamic", 0, cycle)
        if args.with_renderer_control:
            renderer_control(args, measure, manifest)
    manifest.update(
        status="complete",
        finished_at=time.time(),
        calls=counter,
        gpu_memory_stats=jax.devices()[0].memory_stats(),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(status="complete", calls=counter)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--with-renderer-control", action="store_true")
    parser.add_argument("--tasks", type=int, nargs="+", default=[5, 2, 6, 9])
    args = parser.parse_args()
    if args.mode == "run" and args.inputs is None:
        parser.error("run requires --inputs")
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    configure(args.gpu)
    capture(args.output, args.tasks) if args.mode == "capture" else run(args)


if __name__ == "__main__":
    main()
