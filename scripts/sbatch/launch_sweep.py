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
  # 1. Dry-run a tiny sweep. Writes case dirs + jobs CSV but submits nothing.
  uv run python scripts/sbatch/launch_sweep.py \
      --server-config configs/server/mock.json \
      --client-config configs/client/libero/short.json \
      --output-dir experiments/sweeps/slurm \
      --schedulers greedy-deadline,dynamic-action \
      --num-robots 2,4 \
      --seeds 7 \
      --max-batch-size 1 \
      --dry-run

  # 2. Submit a real LIBERO scheduler sweep.
  #    The checked-in mock server config is used as a base, but policy is
  #    rewritten to {"type": "default"} unless --server-policy config is set.
  uv run python scripts/sbatch/launch_sweep.py \
      --server-config configs/server/mock.json \
      --client-config configs/client/libero/short.json \
      --output-dir experiments/sweeps/slurm_libero \
      --schedulers max-batch,dynamic-action \
      --num-robots 2,4,6,8,10 \
      --seeds 7,42 \
      --max-batch-size 1,2,4 \
      --account gts-dxu345-rl2 \
      --partition gpu-l40s

  # 3. Submit an alpha sweep for dynamic-action plus baselines.
  uv run python scripts/sbatch/launch_sweep.py \
      --account gts-<pi-uid> \
      --server-config configs/server/mock.json \
      --client-config configs/client/libero/short.json \
      --output-dir experiments/sweeps/slurm_alpha \
      --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action \
      --num-robots 6,8,10 \
      --seeds 7,42 \
      --alpha 0.0,0.25,0.5,0.75,1.0 \
      --submit-collector

  # 4. Preserve the server config policy exactly, useful for mock/smoke tests.
  uv run python scripts/sbatch/launch_sweep.py \
      --server-config configs/server/mock.json \
      --client-config configs/client/mock/short.json \
      --server-policy config \
      --schedulers greedy-deadline \
      --num-robots 1 \
      --seeds 7 \
      --dry-run

  # 5. Collect results after jobs finish.
  uv run python scripts/sbatch/collect_results.py \
      --output-dir experiments/sweeps/slurm_libero \
      --stamp 20260515_130000

  # 6. Submit only the collector for an existing run, after specific jobs finish.
  sbatch --parsable \
      --account=gts-<pi-uid> \
      --dependency=afterany:12345:12346:12347 \
      scripts/sbatch/collect_results.sh \
      experiments/sweeps/slurm_libero \
      20260515_130000

phoenix notes:
  - Use pace-quota to find valid --account values.
  - Default --partition gpu-l40s + 2 GPUs per case (server on GPU 0, client on GPU 1).
  - QOS is optional in the launcher; Phoenix defaults to inferno.
  - Pass --qos embers explicitly for preemptible backfill.
  - See scripts/sbatch/PHOENIX_NOTES.md for cluster-specific details.
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
    return repo_candidate if repo_candidate.exists() else candidate


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(_resolve_path(path).read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _run_sbatch(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed\ncommand: {shlex.join(cmd)}\n"
            f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


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
    log_dir.mkdir(parents=True, exist_ok=True)

    _write_json(case_dir / "server_args.json", {**case.server_args, "log_dir": str(log_dir / "server")})
    _write_json(
        case_dir / "client_args.json",
        {**case.client_args, "output_dir": str(output_dir), "log_dir": str(log_dir / "client")},
    )
    _write_json(
        case_dir / "case.json",
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
    )
    return case_dir


def _cpus_for(num_robots: int, cpus_per_robot: int) -> int:
    return max(8, num_robots * cpus_per_robot)


def _submit_case(case_dir: pathlib.Path, *, num_robots: int, args: argparse.Namespace) -> str:
    cmd = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        f"--time={args.time}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={_cpus_for(num_robots, args.cpus_per_robot)}",
        "--gres=gpu:2",
        f"--mem={args.mem}",
        f"--export=ALL,ARMORY_SCRIPTS_DIR={SCRIPTS_DIR / 'sbatch'}",
    ]
    if args.account:
        cmd.append(f"--account={args.account}")
    if args.qos:
        cmd.append(f"--qos={args.qos}")
    cmd += [str(SCRIPTS_DIR / "sbatch" / "run_case.sh"), str(case_dir)]
    return _run_sbatch(cmd)


def _job_id_for_dependency(job_id: str) -> str:
    # `sbatch --parsable` may return "jobid" or "jobid;cluster".
    return job_id.split(";", 1)[0]


def _submit_collector(*, run_root: pathlib.Path, stamp: str, job_ids: list[str], args: argparse.Namespace) -> str:
    dependency_ids = [_job_id_for_dependency(jid) for jid in job_ids if jid]
    if not dependency_ids:
        raise RuntimeError("Cannot submit collector without case job ids.")

    cmd = ["sbatch", "--parsable", f"--dependency=afterany:{':'.join(dependency_ids)}"]
    if args.account:
        cmd.append(f"--account={args.account}")
    if args.qos:
        cmd.append(f"--qos={args.qos}")
    cmd += [str(SCRIPTS_DIR / "sbatch" / "collect_results.sh"), str(pathlib.Path(args.output_dir)), stamp]
    collector_job_id = _run_sbatch(cmd)

    metadata = {
        "stamp": stamp,
        "run_root": str(run_root),
        "collector_job_id": collector_job_id,
        "dependency": "afterany:" + ":".join(dependency_ids),
        "case_job_ids": job_ids,
    }
    collector_path = run_root / f"collector_{stamp}.json"
    collector_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {collector_path}")
    return collector_job_id


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
    parser.add_argument("--account", default="")
    parser.add_argument("--partition", default="gpu-l40s")
    parser.add_argument("--qos", default="")
    parser.add_argument("--time", default="1:00:00")
    parser.add_argument("--mem", default="96G")
    parser.add_argument(
        "--cpus-per-robot",
        type=int,
        default=2,
        help="Total CPU request per case = max(8, num_robots * this).",
    )
    parser.add_argument("--submit-collector", action="store_true")
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
            row["job_id"] = _submit_case(case_dir, num_robots=case.num_robots, args=args)
            print(f"Submitted {row['job_id']}: {case.run_id}")
        else:
            print(f"Prepared {case.run_id}: {case_dir}")
        rows.append(row)

    write_rows(run_root / f"jobs_{stamp}.csv", rows)

    if args.submit_collector:
        if args.dry_run:
            print("Skipping collector submission because --dry-run is set.")
        else:
            job_ids = [str(row["job_id"]) for row in rows if row.get("job_id")]
            collector_job_id = _submit_collector(run_root=run_root, stamp=stamp, job_ids=job_ids, args=args)
            print(f"Submitted collector {collector_job_id} after {len(job_ids)} case job(s).")

    print(f"Prepared {len(rows)} case(s) under {run_root}")
    if args.dry_run:
        print("Dry run only; no Slurm jobs submitted.")


if __name__ == "__main__":
    main()
