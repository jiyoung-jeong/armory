"""Fixed SYNC inference cost, with separate host-stage timing and read-only GPU telemetry.

Capture inputs in a separate process with benchmark_fixed_batch first. This run
has no simulator, server, request queue, Nsight, or device clock/power changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.local_batch_sweep import ROOT, gpu_processes

GPU_FIELDS = [
    "index",
    "uuid",
    "name",
    "driver_version",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "clocks_event_reasons.active",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
]
NUMERIC_FIELDS = {
    "memory.used": "memory_used_mib",
    "memory.total": "memory_total_mib",
    "utilization.gpu": "gpu_util_percent",
    "utilization.memory": "memory_util_percent",
    "power.draw": "power_w",
    "power.limit": "power_limit_w",
    "temperature.gpu": "temperature_c",
    "clocks.sm": "sm_clock_mhz",
    "clocks.mem": "memory_clock_mhz",
}


def gpu_sample(gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=" + ",".join(GPU_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    raw = dict(zip(GPU_FIELDS, [x.strip() for x in result.stdout.strip().split(",")], strict=True))
    sample = {k: raw[k] for k in ["index", "uuid", "name", "driver_version"]}
    for key, label in NUMERIC_FIELDS.items():
        sample[label] = float(raw[key])
    sample.update(
        clock_event_reasons=raw["clocks_event_reasons.active"],
        sw_thermal=raw["clocks_event_reasons.sw_thermal_slowdown"] == "Active",
        hw_thermal=raw["clocks_event_reasons.hw_thermal_slowdown"] == "Active",
    )
    return sample


class Telemetry:
    def __init__(self, gpu, output):
        self.gpu, self.output = gpu, output
        self.context = {"phase": "startup"}
        self.stop = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)

    def _run(self):
        try:
            with self.output.open("w") as log:
                while not self.stop.is_set():
                    context = self.context
                    start = time.monotonic()
                    sample = gpu_sample(self.gpu)
                    others = set(gpu_processes(self.gpu)) - {os.getpid()}
                    if others:
                        raise RuntimeError(
                            f"Other GPU compute processes appeared: {sorted(others)}"
                        )
                    sample.update(
                        epoch=time.time(),
                        monotonic=(start + time.monotonic()) / 2,
                        query_ms=(time.monotonic() - start) * 1000,
                        **context,
                        context_changed=context is not self.context,
                    )
                    log.write(json.dumps(sample) + "\n")
                    log.flush()
                    self.stop.wait(max(0, 1.0 - (time.monotonic() - start)))
        except Exception as exc:
            self.failure = repr(exc)


class StageTimings:
    """Host wall ranges only; no new synchronization or GPU profiling calls."""

    def __init__(self):
        self.values = {}

    @contextmanager
    def range(self, name):
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.values[name.rsplit(".", 1)[-1] + "_ms"] = (time.perf_counter_ns() - start) / 1e6


def load_inputs(path):
    import numpy as np

    from armory.backends.types import warmup_request

    records = json.loads((path / "metadata.json").read_text())
    requests = []
    for record in records:
        file = path / f"robot_{record['robot']}.npz"
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"]
        with np.load(file, allow_pickle=False) as saved:
            observation = {
                k: saved[k].item() if saved[k].ndim == 0 else saved[k].copy() for k in saved.files
            }
        requests.append(
            dataclasses.replace(
                warmup_request(observation),
                min_execution_horizon=1,
                max_execution_horizon=10,
                control_hz=20,
            )
        )
    return records, requests


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="starting",
        started_at=time.time(),
        gpu=args.gpu,
        command=list(__import__("sys").argv),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        submodules=subprocess.check_output(["git", "submodule", "status"], text=True).strip(),
        config="pi05_libero",
        infer_type="SYNC",
        num_steps=10,
        batch_sizes=args.batches,
        repeats=args.repeats,
        samples_per_batch_per_repeat=args.samples,
        warmup_per_shape=args.warmup,
        warmup_before_each_block=args.block_warmup,
        component_samples_per_block=args.components,
        rng_seed=7,
        rng_reset_outside_timer=True,
        inputs=str(args.inputs.resolve()),
        timing_scope="Full infer_batch through NumPy materialization and output transforms",
        component_scope="Separate calls; host wall ranges, no extra synchronization; not pure GPU times",
        gpu_sampling_period_s=1.0,
        gpu_memory_scope="Sampled total framebuffer usage, all shapes prewarmed",
        renderer=False,
        server=False,
        nsight=False,
        order=[
            args.batches[i % len(args.batches) :] + args.batches[: i % len(args.batches)]
            for i in range(args.repeats)
        ],
    )
    manifest_path = args.output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    save()
    monitor = Telemetry(args.gpu, args.output / "gpu_telemetry.jsonl")
    monitor.thread.start()
    try:
        import jax
        import numpy as np
        from flax import nnx

        from armory.backends.types import EnvMode
        from openpi_adapter.serve_factory import create_policy

        records, requests = load_inputs(args.inputs)
        if len(requests) < max(args.batches):
            raise ValueError("Need a distinct captured input for every batch member")
        manifest.update(
            inputs_metadata=records, gpu_info=gpu_sample(args.gpu), jax_version=jax.__version__
        )
        save()
        monitor.context = {"phase": "model_load"}
        checkpoint = ROOT / ".cache/openpi-private/cache/openpi-assets/checkpoints/pi05_libero"
        policy = create_policy(
            "pi05_libero", checkpoint, sample_kwargs={"num_steps": 10}, env_mode=EnvMode.LIBERO
        )
        manifest["checkpoint"] = str(checkpoint)
        assert not policy._is_pytorch_model and not policy._is_triton_optimized
        manifest["devices"] = [str(d) for d in jax.devices()]
        dtypes = Counter()
        for leaf in jax.tree.leaves(nnx.state(policy._model)):
            if hasattr(leaf, "dtype"):
                dtypes[str(leaf.dtype)] += int(leaf.size)
        manifest["model_state_elements_by_dtype"] = dict(dtypes)
        rng = jax.random.key(7)
        expected = {}
        with (args.output / "warmup.jsonl").open("w") as warmup_log:
            for size in args.batches:
                monitor.context = dict(phase="shape_warmup", batch_size=size)
                for i in range(args.warmup):
                    monitor.check()
                    policy._policy._rng = rng
                    begin = time.perf_counter_ns()
                    outputs = policy.infer_batch(requests[:size])
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                    warmup_log.write(
                        json.dumps(dict(batch_size=size, index=i, duration_ms=elapsed)) + "\n"
                    )
                    warmup_log.flush()
                expected[size] = [x["actions"].copy() for x in outputs]
                print(
                    json.dumps(dict(event="shape_warmed", batch_size=size, last_call_ms=elapsed)),
                    flush=True,
                )
        manifest.update(
            status="running",
            warmed_at=time.time(),
            output_shapes={b: [list(x.shape) for x in out] for b, out in expected.items()},
        )
        save()
        counter = 0
        stages = StageTimings()
        with (
            (args.output / "calls.jsonl").open("w") as log,
            (args.output / "blocks.jsonl").open("w") as blocks,
        ):

            def measure(size, phase, repeat, index):
                nonlocal counter
                monitor.check()
                policy._policy._rng = rng
                stages.values = {}
                epoch = time.time()
                mono = time.monotonic()
                begin = time.perf_counter_ns()
                outputs = policy.infer_batch(requests[:size])
                elapsed = (time.perf_counter_ns() - begin) / 1e6
                # Materialized CPU arrays are checked outside the timer.
                assert len(outputs) == size
                for output, reference in zip(outputs, expected[size], strict=True):
                    assert isinstance(output["actions"], np.ndarray)
                    assert (
                        output["actions"].shape == (10, 7) and np.isfinite(output["actions"]).all()
                    )
                    np.testing.assert_allclose(output["actions"], reference, rtol=1e-5, atol=1e-5)
                row = dict(
                    index=counter,
                    phase=phase,
                    repeat=repeat,
                    block_index=index,
                    batch_size=size,
                    start_epoch=epoch,
                    start_monotonic=mono,
                    duration_ms=elapsed,
                )
                if phase == "components":
                    assert set(stages.values) == {
                        "prepare_inputs_ms",
                        "sample_dispatch_ms",
                        "materialize_ms",
                        "output_transform_ms",
                    }
                    row.update(stages.values)
                    row["outside_ranges_ms"] = elapsed - sum(stages.values.values())
                    assert row["outside_ranges_ms"] >= 0
                log.write(json.dumps(row) + "\n")
                log.flush()
                counter += 1

            for repeat, order in enumerate(manifest["order"], 1):
                for size in order:
                    monitor.context = dict(phase="block_warmup", repeat=repeat, batch_size=size)
                    for _ in range(args.block_warmup):
                        monitor.check()
                        policy._policy._rng = rng
                        policy.infer_batch(requests[:size])
                    monitor.context = dict(phase="fixed", repeat=repeat, batch_size=size)
                    begin = time.monotonic()
                    epoch = time.time()
                    for index in range(args.samples):
                        measure(size, "fixed", repeat, index)
                    end = time.monotonic()
                    blocks.write(
                        json.dumps(
                            dict(
                                phase="fixed",
                                repeat=repeat,
                                batch_size=size,
                                samples=args.samples,
                                start_monotonic=begin,
                                end_monotonic=end,
                                start_epoch=epoch,
                                wall_seconds=end - begin,
                                jax_memory_stats=jax.devices()[0].memory_stats(),
                            )
                        )
                        + "\n"
                    )
                    blocks.flush()
                    print(
                        json.dumps(
                            dict(
                                event="fixed_block_complete",
                                repeat=repeat,
                                batch_size=size,
                                samples=args.samples,
                                wall_seconds=end - begin,
                            )
                        ),
                        flush=True,
                    )
                    monitor.context = dict(phase="components", repeat=repeat, batch_size=size)
                    with patch("openpi_adapter.policy_adapter.nvtx_range", stages.range):
                        for index in range(args.components):
                            measure(size, "components", repeat, index)
        monitor.check()
        manifest.update(
            status="complete",
            finished_at=time.time(),
            calls=counter,
            validated_outputs=True,
            jax_memory_stats=jax.devices()[0].memory_stats(),
        )
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), finished_at=time.time())
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=25)
        if monitor.failure or monitor.thread.is_alive():
            manifest.update(
                status="failed", telemetry_error=monitor.failure or "thread did not stop"
            )
        save()
    if manifest["status"] != "complete":
        raise RuntimeError(manifest)
    print(json.dumps(dict(event="experiment_complete", calls=counter)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--block-warmup", type=int, default=10)
    parser.add_argument("--components", type=int, default=20)
    args = parser.parse_args()
    if (
        min(
            args.repeats,
            args.samples,
            args.warmup,
            args.block_warmup,
            args.components,
            *args.batches,
        )
        < 1
    ):
        parser.error("counts and batch sizes must be positive")
    if len(set(args.batches)) != len(args.batches):
        parser.error("batch sizes must be unique")
    if gpu_processes(args.gpu):
        raise RuntimeError(f"GPU {args.gpu} is occupied")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENPI_DATA_HOME=str(ROOT / ".cache/openpi-private/cache"),
        ARMORY_NVTX="0",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MPLBACKEND="Agg",
    )
    run(args)


if __name__ == "__main__":
    main()
