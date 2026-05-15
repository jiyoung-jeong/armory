"""Submit Slurm scheduler sweeps using the current serve.py/run_libero.py JSON flow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS_DIR = _HERE.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR / "modal"))

from _utils import write_rows  # noqa: E402


EXAMPLES = """examples:
  # 1. Dry-run a tiny sweep. This writes case dirs + jobs CSV but submits nothing.
  uv run python scripts/sbatch/launch_sweep.py \\
      --server-config configs/server/mock.json \\
      --client-config /path/to/client_short.json \\
      --output-dir experiments/sweeps/slurm \\
      --schedulers greedy-deadline,dynamic-action \\
      --num-robots 2,4 \\
      --seeds 7 \\
      --max-batch-size 1 \\
      --dry-run

  # 2. Submit a real LIBERO GPU scheduler sweep.
  #    The checked-in mock server config is used as a base, but policy is
  #    rewritten to {"type": "default"} unless --server-policy config is set.
  uv run python scripts/sbatch/launch_sweep.py \\
      --server-config configs/server/mock.json \\
      --client-config /path/to/client_short.json \\
      --output-dir experiments/sweeps/slurm_libero \\
      --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action \\
      --num-robots 2,4,6,8,10 \\
      --seeds 7,42 \\
      --max-batch-size 1,2,4 \\
      --partition overcap \\
      --server-gpu l40s \\
      --client-gpu a40

  # 3. Submit an alpha sweep for dynamic-action plus baselines.
  uv run python scripts/sbatch/launch_sweep.py \\
      --server-config configs/server/mock.json \\
      --client-config /path/to/client_short.json \\
      --output-dir experiments/sweeps/slurm_alpha \\
      --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action \\
      --num-robots 6,8,10 \\
      --seeds 7,42 \\
      --alpha 0.0,0.25,0.5,0.75,1.0

  # 4. Preserve the server config policy exactly, useful for mock/smoke tests.
  uv run python scripts/sbatch/launch_sweep.py \\
      --server-config configs/server/mock.json \\
      --client-config /path/to/mock_client.json \\
      --server-policy config \\
      --schedulers greedy-deadline \\
      --num-robots 1 \\
      --seeds 7 \\
      --dry-run

  # 5. Collect results after jobs finish.
  uv run python scripts/sbatch/collect_results.py \\
      --output-dir experiments/sweeps/slurm_libero \\
      --stamp 20260515_130000
"""


class Case:
    def __init__(
        self,
        *,
        server_args: dict[str, Any],
        client_args: dict[str, Any],
        stamp: str,
        scheduler: str,
        num_robots: int,
        seed: int,
        max_batch_size: int,
        alpha: float,
    ) -> None:
        self.server_args = server_args
        self.client_args = client_args
        self.stamp = stamp
        self.scheduler = scheduler
        self.num_robots = num_robots
        self.seed = seed
        self.max_batch_size = max_batch_size
        self.alpha = alpha

    @property
    def run_id(self) -> str:
        return "__".join(
            [
                f"scheduler={self.scheduler}",
                f"num_robots={self.num_robots}",
                f"seed={self.seed}",
                f"max_batch_size={self.max_batch_size}",
                f"alpha={self.alpha}",
            ]
        )


def parse_list_args(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _resolve_path(path: str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(_resolve_path(path).read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_command_manifest(case_dir: pathlib.Path) -> None:
    server_cmd = [
        "uv",
        "run",
        "python",
        "scripts/serve.py",
        "--json-path",
        str(case_dir / "server_args.json"),
    ]
    client_cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_libero.py",
        "--json-path",
        str(case_dir / "client_args.json"),
    ]
    manifest = {
        "cwd": str(REPO_ROOT),
        "commands": {
            "server": {"argv": server_cmd, "shell": shlex.join(server_cmd)},
            "client": {"argv": client_cmd, "shell": shlex.join(client_cmd)},
        },
    }
    (case_dir / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}", ""]
    for name, command in manifest["commands"].items():
        lines += [f"# {name}", command["shell"], ""]
    command_sh = case_dir / "commands.sh"
    command_sh.write_text("\n".join(lines))
    command_sh.chmod(0o755)


def _make_cases(
    *,
    server_args: dict[str, Any],
    client_args: dict[str, Any],
    schedulers: list[str],
    num_robots_list: list[int],
    seeds: list[int],
    max_batch_sizes: list[int],
    alphas: list[float],
    stamp: str,
) -> list[Case]:
    if max_batch_sizes and alphas:
        raise SystemExit("Sweep only one of --max-batch-size or --alpha at a time.")
    if not max_batch_sizes:
        max_batch_sizes = [int(server_args.get("max_batch_size", 1))]
    if not alphas:
        alphas = [float(server_args.get("alpha", 1.0))]

    cases: list[Case] = []
    for seed in seeds:
        for scheduler in schedulers:
            for num_robots in num_robots_list:
                for max_batch_size in max_batch_sizes:
                    for alpha in alphas:
                        server = {
                            **server_args,
                            "seed": seed,
                            "scheduling_algorithm": scheduler,
                            "max_batch_size": max_batch_size,
                            "alpha": alpha,
                        }
                        client = {
                            **client_args,
                            "seed": seed,
                            "num_robots": num_robots,
                            "env": "libero",
                            "progress_type": "logging",
                            "overwrite": True,
                        }
                        cases.append(
                            Case(
                                server_args=server,
                                client_args=client,
                                stamp=stamp,
                                scheduler=scheduler,
                                num_robots=num_robots,
                                seed=seed,
                                max_batch_size=max_batch_size,
                                alpha=alpha,
                            )
                        )
    return cases


def _materialize_case(case: Case, *, run_root: pathlib.Path) -> pathlib.Path:
    case_dir = run_root / case.run_id
    output_dir = case_dir / "outputs"
    log_dir = case_dir / "logs"
    case_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    server_args = {**case.server_args, "log_dir": str(log_dir / "server")}
    client_args = {
        **case.client_args,
        "output_dir": str(output_dir),
        "log_dir": str(log_dir / "client"),
    }
    _write_json(case_dir / "server_args.json", server_args)
    _write_json(case_dir / "client_args.json", client_args)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "stamp": case.stamp,
                "run_id": case.run_id,
                "scheduler": case.scheduler,
                "num_robots": case.num_robots,
                "seed": case.seed,
                "max_batch_size": case.max_batch_size,
                "alpha": case.alpha,
                "case_dir": str(case_dir),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
        + "\n"
    )
    _write_command_manifest(case_dir)
    return case_dir


def _submit_case(case_dir: pathlib.Path, args: argparse.Namespace) -> str:
    cmd = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        f"--time={args.time}",
        f"--cpus-per-task={args.server_cpus}",
        f"--gpus-per-node={args.server_gpu}:1",
        "--mem=32G",
        ":",
        f"--partition={args.partition}",
        f"--time={args.time}",
        f"--cpus-per-task={args.client_cpus}",
        f"--gpus-per-node={args.client_gpu}:1",
        "--mem=64G",
    ]
    if args.exclude:
        cmd.insert(2, f"--exclude={args.exclude}")
    for opt in args.sbatch_option:
        cmd.append(opt)
    cmd += [str(SCRIPTS_DIR / "sbatch" / "run_case.sh"), str(case_dir)]
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--server-config", default="configs/server/mock.json")
    parser.add_argument("--client-config", required=True)
    parser.add_argument(
        "--server-policy",
        choices=["default", "config"],
        default="default",
        help="Use the default real policy unless set to 'config' to preserve the server config policy.",
    )
    parser.add_argument("--output-dir", default="experiments/sweeps/slurm")
    parser.add_argument("--schedulers", default="fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action")
    parser.add_argument("--num-robots", default="2,4,6,8,10")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--max-batch-size", default="")
    parser.add_argument("--alpha", default="")
    parser.add_argument("--partition", default="overcap")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--server-gpu", default="l40s")
    parser.add_argument("--client-gpu", default="a40")
    parser.add_argument("--server-cpus", type=int, default=4)
    parser.add_argument("--client-cpus", type=int, default=20)
    parser.add_argument("--exclude", default="")
    parser.add_argument("--sbatch-option", action="append", default=[])
    parser.add_argument("--stamp", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    run_root = pathlib.Path(args.output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    server_args = _read_json(args.server_config)
    client_args = _read_json(args.client_config)
    if args.server_policy == "default":
        server_args["policy"] = {"type": "default"}
    cases = _make_cases(
        server_args=server_args,
        client_args=client_args,
        schedulers=parse_list_args(args.schedulers),
        num_robots_list=parse_list_args(args.num_robots, cast=int),
        seeds=parse_list_args(args.seeds, cast=int),
        max_batch_sizes=parse_list_args(args.max_batch_size, cast=int),
        alphas=parse_list_args(args.alpha, cast=float),
        stamp=stamp,
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_dir = _materialize_case(case, run_root=run_root)
        row = {
            "stamp": stamp,
            "run_id": case.run_id,
            "scheduler": case.scheduler,
            "num_robots": case.num_robots,
            "seed": case.seed,
            "max_batch_size": case.max_batch_size,
            "alpha": case.alpha,
            "case_dir": str(case_dir),
            "status": "dry_run" if args.dry_run else "submitted",
            "job_id": "",
        }
        if not args.dry_run:
            row["job_id"] = _submit_case(case_dir, args)
            print(f"Submitted {row['job_id']}: {case.run_id}")
        else:
            print(f"Prepared {case.run_id}: {case_dir}")
        rows.append(row)

    jobs_csv = run_root / f"jobs_{stamp}.csv"
    write_rows(jobs_csv, rows)
    print(f"Prepared {len(rows)} case(s) under {run_root}")
    if args.dry_run:
        print("Dry run only; no Slurm jobs submitted.")


if __name__ == "__main__":
    main()
