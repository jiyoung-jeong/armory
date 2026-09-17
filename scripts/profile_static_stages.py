"""Profile the unchanged monolithic JAX policy with stage names in HLO metadata.

Named scopes label existing methods; they do not split JIT or add GPU barriers.
Before compilation, compare unannotated/annotated StableHLO without debug info.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import inspect
import json
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_static_inference import Telemetry, load_inputs
from scripts.local_batch_sweep import ROOT, gpu_processes

STAGES = {
    "embed_prefix": "armory_stage_vlm_embed",
    "prefill": "armory_stage_vlm_prefill",
    "flow_matching": "armory_stage_action",
}


def scoped_method(method, scope):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        import jax

        with jax.named_scope(scope):
            return method(*args, **kwargs)

    return wrapped


def jit_parts(wrapper):
    values = inspect.getclosurevars(wrapper).nonlocals
    return values["jitted_fn"], values["state"]


class ExternalGuard:
    def __init__(self, path):
        self.path = path
        self.failure = None
        self.context = {}
        deadline = time.monotonic() + 40
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("External GPU guard did not start")
            time.sleep(0.25)
        self.check()

    def check(self):
        state = json.loads(self.path.read_text())
        self.failure = state["error"]
        if time.time() - state["checked_at"] > 20:
            self.failure = "External GPU guard heartbeat is stale"
        if self.failure:
            raise RuntimeError(self.failure)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="starting",
        pid=os.getpid(),
        external_gpu_observer=args.external_observer,
        capture_until_exit=args.capture_until_exit,
        started_at=time.time(),
        gpu=args.gpu,
        graph_mode=args.graph_mode,
        batch_sizes=args.batches,
        warmup=args.warmup,
        samples=args.samples,
        control_samples=args.controls,
        input_directory=str(args.inputs.resolve()),
        rng_seed=7,
        config="pi05_libero",
        num_steps=10,
        infer_type="SYNC",
        jit="single original module_jit; only named_scope metadata added",
        stages=STAGES,
        gpu_barriers_added_inside_inference=False,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        submodules=subprocess.check_output(["git", "submodule", "status"], text=True).strip(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
    )

    def save():
        temporary = args.output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(args.output / "manifest.json")

    save()
    if args.external_observer:
        monitor = ExternalGuard(args.output / "gpu_guard.json")
    else:
        monitor = Telemetry(args.gpu, args.output / "gpu_telemetry.jsonl")
        monitor.thread.start()
    profiler_started = False
    try:
        import jax
        import numpy as np
        from openpi.models.pi0 import Pi0
        from openpi.shared import array_typing as at
        from openpi.shared import nnx_utils

        from armory.backends.types import EnvMode
        from armory.utils.profiling import nvtx_range
        from openpi_adapter.policy_adapter import _rename_keys
        from openpi_adapter.serve_factory import create_policy

        records, requests = load_inputs(args.inputs)
        assert max(args.batches) <= len(requests)
        manifest["inputs_metadata"] = records
        policy = create_policy(
            "pi05_libero",
            ROOT / ".cache/openpi-private/cache/openpi-assets/checkpoints/pi05_libero",
            sample_kwargs={"num_steps": 10},
            env_mode=EnvMode.LIBERO,
        )
        rng = jax.random.key(7)
        lower_inputs = {}
        reference_ir = {}
        original_fn, state = jit_parts(policy._sample_actions)
        for size in args.batches:
            monitor.check()
            obs = policy.create_batch_obs([_rename_keys(r.observation) for r in requests[:size]])
            noise = policy._model.sample_noise(rng, batch_size=size)
            lower_inputs[size] = (obs, noise)
            with at.disable_typechecking():
                lowered = original_fn.lower(state, rng, obs, num_steps=10, noise=noise)
            reference_ir[size] = lowered.compiler_ir("stablehlo").operation.get_asm(
                enable_debug_info=False
            )
        expected = {}
        manifest["stablehlo_checks"] = {}
        with ExitStack() as stack:
            for method, scope in STAGES.items():
                stack.enter_context(
                    patch.object(Pi0, method, scoped_method(getattr(Pi0, method), scope))
                )
            policy._policy._sample_actions = nnx_utils.module_jit(
                policy._model.sample_actions, static_argnames=["use_rtc"]
            )
            tagged_fn, tagged_state = jit_parts(policy._sample_actions)
            for size in args.batches:
                monitor.context = dict(phase="warmup", batch_size=size)
                obs, noise = lower_inputs[size]
                with at.disable_typechecking():
                    lowered = tagged_fn.lower(tagged_state, rng, obs, num_steps=10, noise=noise)
                tagged_ir = lowered.compiler_ir("stablehlo").operation.get_asm(
                    enable_debug_info=False
                )
                identical = tagged_ir == reference_ir[size]
                (args.output / f"b{size}.original.stablehlo").write_text(reference_ir[size])
                (args.output / f"b{size}.annotated.stablehlo").write_text(tagged_ir)
                manifest["stablehlo_checks"][size] = dict(
                    identical=identical,
                    original_sha256=hashlib.sha256(reference_ir[size].encode()).hexdigest(),
                    annotated_sha256=hashlib.sha256(tagged_ir.encode()).hexdigest(),
                )
                save()
                if not identical:
                    raise RuntimeError(f"Named scopes changed non-debug StableHLO for B{size}")
                for _ in range(args.warmup):
                    monitor.check()
                    policy._policy._rng = rng
                    outputs = policy.infer_batch(requests[:size])
                expected[size] = [o["actions"].copy() for o in outputs]
                with at.disable_typechecking():
                    hlo_text = lowered.compile().as_text()
                (args.output / f"b{size}.optimized_hlo.txt").write_text(hlo_text)
                print(
                    json.dumps(
                        dict(event="shape_warmed", batch_size=size, stablehlo_identical=identical)
                    ),
                    flush=True,
                )
            manifest.update(status="running", warmed_at=time.time())
            save()
            libcudart = ctypes.CDLL(
                str(
                    ROOT
                    / ".venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
                )
            )
            for name in ["cudaProfilerStart", "cudaProfilerStop"]:
                getattr(libcudart, name).argtypes = []
                getattr(libcudart, name).restype = ctypes.c_int
            with (args.output / "calls.jsonl").open("w") as log:

                def call(size, phase, index):
                    monitor.check()
                    monitor.context = dict(phase=phase, batch_size=size)
                    policy._policy._rng = rng
                    label = f"armory.stage_call phase={phase} b={size} i={index}"
                    start_epoch = time.time()
                    with nvtx_range(label):
                        begin = time.perf_counter_ns()
                        outputs = policy.infer_batch(requests[:size])
                        duration = (time.perf_counter_ns() - begin) / 1e6
                    assert len(outputs) == size
                    for output, reference in zip(outputs, expected[size], strict=True):
                        assert isinstance(output["actions"], np.ndarray)
                        assert (
                            output["actions"].shape == (10, 7)
                            and np.isfinite(output["actions"]).all()
                        )
                        np.testing.assert_allclose(
                            output["actions"], reference, atol=1e-5, rtol=1e-5
                        )
                    log.write(
                        json.dumps(
                            dict(
                                phase=phase,
                                batch_size=size,
                                index=index,
                                duration_ms=duration,
                                start_epoch=start_epoch,
                                label=label,
                            )
                        )
                        + "\n"
                    )
                    log.flush()

                for size in args.batches:
                    for i in range(args.controls):
                        call(size, "before", i)
                assert libcudart.cudaProfilerStart() == 0
                profiler_started = True
                for size in args.batches:
                    for i in range(3):
                        call(size, "transition", i)
                    for i in range(args.samples):
                        call(size, "profile", i)
                    print(
                        json.dumps(
                            dict(
                                event="profile_block_complete", batch_size=size, calls=args.samples
                            )
                        ),
                        flush=True,
                    )
                if not args.capture_until_exit:
                    assert libcudart.cudaProfilerStop() == 0
                    profiler_started = False
                for size in reversed(args.batches):
                    for i in range(args.controls):
                        call(size, "after_traced" if args.capture_until_exit else "after", i)
        monitor.check()
        manifest.update(status="complete", finished_at=time.time(), outputs_validated=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), finished_at=time.time())
        raise
    finally:
        if profiler_started:
            libcudart.cudaProfilerStop()
        alive = False
        if not args.external_observer:
            monitor.stop.set()
            monitor.thread.join(timeout=25)
            alive = monitor.thread.is_alive()
        if monitor.failure or alive:
            manifest.update(
                status="failed", telemetry_error=monitor.failure or "thread did not stop"
            )
        save()
    if manifest["status"] != "complete":
        raise RuntimeError(manifest)
    print(json.dumps(dict(event="experiment_complete")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--external-observer", action="store_true")
    parser.add_argument(
        "--capture-until-exit",
        action="store_true",
        help="Use with nsys --capture-range-end=none; trailing calls remain traced.",
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--graph-mode", choices=["default", "disabled"], default="default")
    args = parser.parse_args()
    if min(*args.batches, args.samples, args.controls, args.warmup) < 1:
        parser.error("Counts must be positive")
    if gpu_processes(args.gpu):
        raise RuntimeError(f"GPU {args.gpu} is occupied")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENPI_DATA_HOME=str(ROOT / ".cache/openpi-private/cache"),
        ARMORY_NVTX="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        JAX_TRACEBACK_IN_LOCATIONS_LIMIT="-1",
    )
    if args.graph_mode == "disabled":
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
        ).strip()
    run(args)


if __name__ == "__main__":
    main()
