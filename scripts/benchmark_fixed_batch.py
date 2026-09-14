"""Capture actual LIBERO observations, then time materialized SYNC infer_batch calls.

Run capture and run as separate processes so EGL is gone before model timing.
No serving, scheduler, network or renderer is included in measured calls.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
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


def capture(output):
    from evaluation.envs.libero import LiberoRobotSpec, LiberoSimEnvironment

    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, task in enumerate([5, 2, 6, 9]):
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
        renderer_active=False,
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
    args = parser.parse_args()
    if args.mode == "run" and args.inputs is None:
        parser.error("run requires --inputs")
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    configure(args.gpu)
    capture(args.output) if args.mode == "capture" else run(args)


if __name__ == "__main__":
    main()
