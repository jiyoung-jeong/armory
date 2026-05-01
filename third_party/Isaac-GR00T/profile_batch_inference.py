"""
Profile GR00T inference time across batch sizes 1-5 using the LIBERO checkpoint.

Usage:
    python scripts/deployment/profile_batch_inference.py \
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10

    # Custom batch sizes / repetitions:
    python scripts/deployment/profile_batch_inference.py \
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
        --batch-sizes 1 2 4 8 \
        --n-warmup 3 \
        --n-reps 10
"""

import argparse
import time

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


# ── LIBERO modality dimensions ──────────────────────────────────────────────
# Matches the "libero_sim" embodiment config and LiberoEnv observation space:
#   video: ["image", "wrist_image"]  – (B, T=1, H=256, W=256, 3)
#   state: x/y/z/roll/pitch/yaw – each (B, T=1, 1); gripper – (B, T=1, 2)
#   language: ["annotation.human.action.task_description"]  – [[str]] * B

VIDEO_KEYS = ["image", "wrist_image"]
# Maps state key -> feature dimension D (gripper has 2 joint positions)
STATE_KEY_DIMS = {"x": 1, "y": 1, "z": 1, "roll": 1, "pitch": 1, "yaw": 1, "gripper": 2}
LANGUAGE_KEY = "annotation.human.action.task_description"
T_VIDEO = 1   # temporal horizon for video
T_STATE = 1   # temporal horizon for state
H, W = 256, 256
TASK_TEXT = "pick up the cube"


def make_observation(batch_size: int) -> dict:
    return {
        "video": {
            key: np.random.randint(0, 256, (batch_size, T_VIDEO, H, W, 3), dtype=np.uint8)
            for key in VIDEO_KEYS
        },
        "state": {
            key: np.random.randn(batch_size, T_STATE, dim).astype(np.float32)
            for key, dim in STATE_KEY_DIMS.items()
        },
        "language": {
            LANGUAGE_KEY: [[TASK_TEXT]] * batch_size
        },
    }


def run_profile(policy: Gr00tPolicy, batch_sizes: list[int], n_warmup: int, n_reps: int):
    results = {}

    for bs in batch_sizes:
        obs = make_observation(bs)
        print(f"\n[batch_size={bs}] Warming up ({n_warmup} runs)...", flush=True)

        # Warmup
        for _ in range(n_warmup):
            policy.get_action(obs)
            torch.cuda.synchronize()

        # Timed runs
        print(f"[batch_size={bs}] Timing {n_reps} runs...", flush=True)
        times = []
        for _ in range(n_reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.get_action(obs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times = np.array(times)
        results[bs] = times
        print(
            f"  mean={times.mean()*1000:.1f}ms  std={times.std()*1000:.1f}ms  "
            f"min={times.min()*1000:.1f}ms  max={times.max()*1000:.1f}ms  "
            f"p50={np.percentile(times,50)*1000:.1f}ms  p90={np.percentile(times,90)*1000:.1f}ms"
        )

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Batch':>6}  {'Mean (ms)':>10}  {'Std (ms)':>9}  {'Min (ms)':>9}  {'P90 (ms)':>9}")
    print("-" * 65)
    for bs, times in results.items():
        print(
            f"{bs:>6}  {times.mean()*1000:>10.1f}  {times.std()*1000:>9.1f}  "
            f"{times.min()*1000:>9.1f}  {np.percentile(times,90)*1000:>9.1f}"
        )
    print("=" * 65)
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile GR00T batch inference latency")
    parser.add_argument("--model-path", required=True, help="Path to the LIBERO checkpoint")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Batch sizes to profile (default: 1 2 3 4 5)"
    )
    parser.add_argument("--n-warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--n-reps", type=int, default=10, help="Timed iterations per batch size (default: 10)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profiling")

    print(f"Loading model from: {args.model_path}")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=args.model_path,
        device="cuda",
    )
    print("Model loaded.\n")

    run_profile(policy, args.batch_sizes, args.n_warmup, args.n_reps)


if __name__ == "__main__":
    main()
