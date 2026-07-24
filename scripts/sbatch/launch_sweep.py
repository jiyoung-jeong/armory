"""Submit Slurm scheduler sweeps using the current serve.py/run-libero JSON flow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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

SERVER_CONFIG_SWEEP_SCHEDULERS = {"lookahead-actions"}

EXAMPLES = """examples:
  # 1. Dry-run a tiny sweep. Writes case dirs + jobs CSV but submits nothing.
  uv run python scripts/sbatch/launch_sweep.py \
      --server-config configs/server/mock.json \
      --client-config configs/client/libero/short.json \
      --output-dir experiments/sweeps/slurm \
      --schedulers greedy-deadline,dynamic-action \
      --seeds 7 \
      --max-batch-size 1 \
      --dry-run

  # 2. Submit a real LIBERO scheduler sweep.
  #    The checked-in mock server config is used as a base, but policy is
  #    rewritten to {"type": "default"} unless --server-policy config is set.
  #    Each case is submitted as a heterogeneous job: het-group 0 is the
  #    server on a single L40S (embers QoS); het-group 1 is the client on
  #    enough V100s to satisfy the CPU:GPU<12 rule.
  uv run python scripts/sbatch/launch_sweep.py \
      --server-config configs/server/mock.json \
      --client-config configs/client/libero/half_fast_half_slow \
      --output-dir experiments/sweeps/slurm_libero \
      --schedulers max-batch,dynamic-action \
      --seeds 7,42 \
      --max-batch-size 1,2,4 \
      --account gts-dxu345-rl2

  # 3. Submit an alpha sweep for dynamic-action plus baselines.
  uv run python scripts/sbatch/launch_sweep.py \
      --account gts-dxu345-rl2 \
      --server-config configs/server/mock.json \
      --client-config configs/client/libero/half_fast_half_slow \
      --output-dir experiments/sweeps/slurm_alpha \
      --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action \
      --seeds 7,42 \
      --alpha 0.0,0.25,0.5,0.75,1.0 \
      --submit-collector

  # 4. Preserve the server config policy exactly, useful for mock/smoke tests.
  uv run python scripts/sbatch/launch_sweep.py \
      --server-config configs/server/mock.json \
      --client-config configs/client/mock/short.json \
      --server-policy config \
      --schedulers greedy-deadline \
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

  # 7. Requeue any cases whose result.json is missing or not status=ok.
  #    Replays each case's saved submit_cmd.json; old logs + result.json are
  #    archived to *.previous_<stamp> in the case dir.
  uv run python scripts/sbatch/launch_sweep.py \
      --requeue experiments/sweeps/slurm_alpha/20260516_014758

  # 7b. Preview which cases would be requeued without submitting.
  uv run python scripts/sbatch/launch_sweep.py \
      --requeue experiments/sweeps/slurm_alpha/20260516_014758 \
      --dry-run

phoenix notes:
  - Use pace-quota to find valid --account values.
  - Each case is a heterogeneous job: 1 L40S (embers) for the server,
    1+ V100s (default QoS) for the client. V100 count is derived from the
    client's CPU request to keep CPU:GPU < 12.
  - See scripts/sbatch/PHOENIX_NOTES.md for cluster-specific details.
"""


class Case:
    def __init__(
        self,
        *,
        server_args: dict[str, Any],
        client_args: dict[str, Any],
        experiment_config: dict[str, Any],
        experiment_name: str,
        stamp: str,
        scheduler: str,
        seed: int,
        max_batch_size: int,
        alpha: float,
        server_variant: str = "",
        action_horizon_multiplier: float = 0.0,
    ) -> None:
        self.server_args = server_args
        self.client_args = client_args
        self.experiment_config = experiment_config
        self.experiment_name = experiment_name
        self.stamp = stamp
        self.scheduler = scheduler
        self.seed = seed
        self.max_batch_size = max_batch_size
        self.alpha = alpha
        self.server_variant = server_variant
        self.action_horizon_multiplier = action_horizon_multiplier

    @property
    def num_robots(self) -> int:
        return int(self.experiment_config["experiment"]["num_robots"])

    @property
    def run_id(self) -> str:
        parts = [
            f"scheduler={self.scheduler}",
            f"experiment={self.experiment_name}",
            f"num_robots={self.num_robots}",
            f"seed={self.seed}",
            f"max_batch_size={self.max_batch_size}",
            f"alpha={self.alpha}",
        ]
        if self.server_variant:
            parts.append(f"server={self.server_variant}")
        if self.action_horizon_multiplier:
            parts.append(f"ahm={self.action_horizon_multiplier}")
        return "__".join(parts)


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


def _server_config_paths(value: str) -> list[pathlib.Path]:
    paths = [_resolve_path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise SystemExit("--server-config is required.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"--server-config file(s) not found: {', '.join(missing)}")
    return paths


def _server_variant_name(path: pathlib.Path) -> str:
    return path.stem.removeprefix("lookahead_actions_short_horizon_")


def _client_config_paths(path: str) -> list[pathlib.Path]:
    candidate = _resolve_path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        paths = sorted([*candidate.rglob("*.json"), *candidate.rglob("*.jsonc")])
        if paths:
            return paths
    raise SystemExit(f"--client-config must be a JSON/JSONC file or directory: {path}")


def _experiment_name(path: pathlib.Path, *, root: pathlib.Path | None = None) -> str:
    rel = path.relative_to(root) if root is not None else pathlib.Path(path.name)
    return str(rel.with_suffix("")).replace("/", "_")


def _read_experiment_config(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    experiment = data["experiment"]
    robots = data["robots"]
    for key in (
        "env",
        "task_suite_name",
        "action_chunk_broker_type",
        "num_robots",
        "max_steps",
        "control_hz",
    ):
        if key not in experiment:
            raise ValueError(f"{path}: missing experiment.{key}")
    # Either legacy-mode (trials_per_robot) or trial-mode
    # (wall_clock_time_limit_s) must specify how a run terminates.
    if "trials_per_robot" not in experiment and not experiment.get("wall_clock_time_limit_s"):
        raise ValueError(
            f"{path}: experiment must set either 'trials_per_robot' or 'wall_clock_time_limit_s'."
        )
    for idx in range(int(experiment["num_robots"])):
        robot = robots[f"robot_{idx}"]
        for key in ("min_execution_horizon", "max_execution_horizon"):
            if key not in robot:
                raise ValueError(f"{path}: missing robots.robot_{idx}.{key}")
    return data


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
    server_variants: list[tuple[str, dict[str, Any]]],
    client_args: dict[str, Any],
    experiment_configs: list[tuple[str, dict[str, Any]]],
    schedulers: list[str],
    seeds: list[int],
    max_batch_sizes: list[int],
    alphas: list[float],
    action_horizon_multipliers: list[float] | None = None,
    stamp: str,
) -> list[Case]:
    if max_batch_sizes and alphas:
        raise SystemExit("Sweep only one of --max-batch-size or --alpha at a time.")
    if not server_variants:
        raise SystemExit("At least one server config is required.")
    if not max_batch_sizes:
        max_batch_sizes = [int(server_variants[0][1].get("max_batch_size", 1))]
    if not alphas:
        alphas = [float(server_variants[0][1].get("alpha", 1.0))]
    action_horizon_multipliers = action_horizon_multipliers or []

    cases: list[Case] = []
    for seed in seeds:
        for scheduler in schedulers:
            active_server_variants = (
                server_variants
                if scheduler in SERVER_CONFIG_SWEEP_SCHEDULERS
                else server_variants[:1]
            )
            # The shortest-horizon multiplier override only affects the
            # config-sensitive lookahead schedulers; other schedulers ignore
            # action_horizon_multipliers, so sweeping them would just create
            # duplicate baseline cases.
            scheduler_ahms: list[float | None] = (
                action_horizon_multipliers
                if action_horizon_multipliers and scheduler in SERVER_CONFIG_SWEEP_SCHEDULERS
                else [None]
            )
            for server_variant, server_args in active_server_variants:
                for experiment_name, experiment_config in experiment_configs:
                    for max_batch_size in max_batch_sizes:
                        for alpha in alphas:
                            for ahm in scheduler_ahms:
                                # scheduling_algorithm and
                                # action_horizon_multipliers used to live on the
                                # server config, but they are now hot-swappable
                                # via POST /reconfigure (issued by run_libero on
                                # startup). Keeping them server-side too would
                                # force the interactive sweep driver to restart
                                # the server for every case that varies them.
                                # ``max_batch_size`` and ``alpha`` stay
                                # server-startup-only.
                                server = {
                                    **server_args,
                                    "seed": seed,
                                    "max_batch_size": max_batch_size,
                                    "alpha": alpha,
                                }
                                server.pop("scheduling_algorithm", None)
                                server.pop("action_horizon_multipliers", None)
                                if ahm is not None:
                                    case_multipliers = _override_shortest_horizon(
                                        server_args.get("action_horizon_multipliers"),
                                        ahm,
                                    )
                                else:
                                    case_multipliers = dict(
                                        server_args.get("action_horizon_multipliers") or {}
                                    )
                                client = {
                                    **client_args,
                                    "seed": seed,
                                    "progress_type": "logging",
                                    "overwrite": True,
                                    "scheduling_algorithm": scheduler,
                                    "action_horizon_multipliers": case_multipliers,
                                }
                                cases.append(
                                    Case(
                                        server_args=server,
                                        client_args=client,
                                        experiment_config=experiment_config,
                                        experiment_name=experiment_name,
                                        stamp=stamp,
                                        scheduler=scheduler,
                                        seed=seed,
                                        max_batch_size=max_batch_size,
                                        alpha=alpha,
                                        server_variant=(
                                            server_variant if len(server_variants) > 1 else ""
                                        ),
                                        action_horizon_multiplier=(ahm or 0.0),
                                    )
                                )
    return cases


def _override_shortest_horizon(base: dict[str, Any] | None, multiplier: float) -> dict[str, Any]:
    """Override only the shortest-horizon key of ``base`` with ``multiplier``.

    Keys are horizon lengths (as strings); longer-horizon weights are left at
    their base values. With no base dict the multiplier can't be placed, so the
    result is empty.
    """
    result = dict(base or {})
    if not result:
        return result
    shortest_key = min(result, key=lambda k: float(k))
    result[shortest_key] = multiplier
    return result


def _materialize_case(case: Case, *, run_root: pathlib.Path) -> pathlib.Path:
    case_dir = run_root / case.run_id
    output_dir = case_dir
    log_dir = case_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "server_args.json", {**case.server_args, "log_dir": str(log_dir / "server")}
    )
    experiment_config_path = case_dir / "experiment_config.json"
    _write_json(experiment_config_path, case.experiment_config)
    _write_json(
        case_dir / "client_args.json",
        {
            **case.client_args,
            "experiment_config": str(experiment_config_path),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir / "client"),
        },
    )
    _write_json(
        case_dir / "case.json",
        {
            "stamp": case.stamp,
            "run_id": case.run_id,
            "scheduler": case.scheduler,
            "experiment": case.experiment_name,
            "num_robots": case.num_robots,
            "seed": case.seed,
            "max_batch_size": case.max_batch_size,
            "alpha": case.alpha,
            "server_variant": case.server_variant,
            "action_horizon_multiplier": case.action_horizon_multiplier,
            "action_horizon_multipliers": case.client_args.get("action_horizon_multipliers", {}),
            "case_dir": str(case_dir),
            "output_dir": str(output_dir),
        },
    )
    return case_dir


def _cpus_for(num_robots: int, cpus_per_robot: int) -> int:
    return max(8, num_robots * cpus_per_robot)


def _v100s_for(client_cpus: int) -> int:
    # Phoenix enforces CpusPerTres=gpu:12 on V100 nodes — keep ratio < 12.
    return max(1, math.ceil(client_cpus / 11))


def _submit_case(case_dir: pathlib.Path, *, num_robots: int, args: argparse.Namespace) -> str:
    client_cpus = _cpus_for(num_robots, args.cpus_per_robot)
    n_v100 = _v100s_for(client_cpus)

    cmd = [
        "sbatch",
        "--parsable",
        f"--export=ALL,ARMORY_SCRIPTS_DIR={SCRIPTS_DIR / 'sbatch'}",
    ]
    if args.cluster == "pace":
        # Het-group 0: server on a single L40S (embers QoS is required here).
        cmd += [
            "--constraint=gpu-l40s",
            "--gres=gpu:1",
            "--ntasks=1",
            "--cpus-per-task=4",
            f"--mem={args.server_mem}",
            "--qos=embers",
            f"--time={args.time}",
        ]
    else:  # ice
        cmd += [
            "--gres=gpu:L40S:1",
            "--ntasks=1",
            "--cpus-per-task=4",
            f"--mem={args.server_mem}",
            f"--time={args.time}",
        ]
    if args.account:
        cmd.append(f"--account={args.account}")
    cmd.append(":")
    if args.cluster == "pace":
        # Het-group 1: client on V100(s). GPU count is driven by the CPU:GPU<12 rule.
        cmd += [
            "--constraint=V100",
            "--nodes=1",
            f"--gres=gpu:{n_v100}",
            "--ntasks=1",
            f"--cpus-per-task={client_cpus}",
            f"--mem={args.client_mem}",
            f"--time={args.time}",
        ]
    else:  # ice
        if num_robots > 20:
            client_gpu_type = "L40S"
            client_cpus = num_robots
            n_client_gpus = math.ceil(client_cpus / 8)
        else:
            client_gpu_type = "V100"
            n_client_gpus = n_v100
        cmd += [
            "--nodes=1",
            f"--gres=gpu:{client_gpu_type}:{n_client_gpus}",
            "--ntasks=1",
            f"--cpus-per-task={client_cpus}",
            f"--mem={args.client_mem}",
            f"--time={args.time}",
        ]
    if args.account:
        cmd.append(f"--account={args.account}")
    cmd += [str(SCRIPTS_DIR / "sbatch" / "run_case.sh"), str(case_dir)]
    _save_submit_cmd(case_dir, cmd)
    return _run_sbatch(cmd)


def _save_submit_cmd(case_dir: pathlib.Path, cmd: list[str]) -> None:
    payload = {"argv": cmd, "shell": shlex.join(cmd), "cwd": str(REPO_ROOT)}
    (case_dir / "submit_cmd.json").write_text(json.dumps(payload, indent=2) + "\n")


def _load_submit_cmd(case_dir: pathlib.Path) -> list[str] | None:
    path = case_dir / "submit_cmd.json"
    if not path.exists():
        return None
    try:
        argv = json.loads(path.read_text()).get("argv") or []
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[skip] {case_dir.name}: cannot read submit_cmd.json ({exc})")
        return None
    return [str(x) for x in argv] if argv else None


def _case_succeeded(case_dir: pathlib.Path) -> bool:
    result = case_dir / "result.json"
    if not result.exists():
        return False
    try:
        return json.loads(result.read_text()).get("status") == "ok"
    except (OSError, json.JSONDecodeError):
        return False


def _archive_previous_attempt(case_dir: pathlib.Path, stamp: str) -> None:
    for name in ("result.json", "logs"):
        p = case_dir / name
        if p.exists():
            p.rename(case_dir / f"{name}.previous_{stamp}")


def _requeue_run(run_root: pathlib.Path, *, dry_run: bool) -> list[dict[str, Any]]:
    case_dirs = sorted({p.parent for p in run_root.glob("**/case.json")})
    if not case_dirs:
        raise SystemExit(f"No case.json files found under {run_root}")
    failed = [d for d in case_dirs if not _case_succeeded(d)]
    print(f"{len(failed)} of {len(case_dirs)} case(s) under {run_root} are not status=ok.")
    if not failed:
        return []

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    rows: list[dict[str, Any]] = []
    for case_dir in failed:
        argv = _load_submit_cmd(case_dir)
        rel = case_dir.relative_to(run_root)
        if argv is None:
            print(
                f"[skip] {rel}: no submit_cmd.json (was this case submitted before the requeue feature existed?)"
            )
            continue
        if dry_run:
            print(f"[dry] would requeue {rel}: {shlex.join(argv)}")
            rows.append({"case_dir": str(case_dir), "job_id": "", "status": "dry_run"})
            continue
        _archive_previous_attempt(case_dir, stamp)
        job_id = _run_sbatch(argv)
        print(f"Requeued {job_id}: {rel}")
        rows.append({"case_dir": str(case_dir), "job_id": job_id, "status": "requeued"})

    if rows:
        write_rows(run_root / f"requeue_{stamp}.csv", rows)
    return rows


def _job_id_for_dependency(job_id: str) -> str:
    # `sbatch --parsable` may return "jobid" or "jobid;cluster".
    return job_id.split(";", 1)[0]


def _submit_collector(
    *, run_root: pathlib.Path, stamp: str, job_ids: list[str], args: argparse.Namespace
) -> str:
    dependency_ids = [_job_id_for_dependency(jid) for jid in job_ids if jid]
    if not dependency_ids:
        raise RuntimeError("Cannot submit collector without case job ids.")

    cmd = ["sbatch", "--parsable", f"--dependency=afterany:{':'.join(dependency_ids)}"]
    if args.account:
        cmd.append(f"--account={args.account}")
    cmd += [
        str(SCRIPTS_DIR / "sbatch" / "collect_results.sh"),
        str(pathlib.Path(args.output_dir)),
        stamp,
    ]
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
    parser.add_argument(
        "--server-config",
        default="configs/server/mock.json",
        help="Server config JSON file, or comma-separated JSON files for config-sensitive schedulers.",
    )
    parser.add_argument(
        "--client-config",
        default="",
        help="Experiment config JSON/JSONC file, or a directory whose JSON/JSONC files become cases.",
    )
    parser.add_argument(
        "--requeue",
        default="",
        help="Path to a previous stamped run dir (e.g. experiments/sweeps/slurm_alpha/20260516_014758). "
        "Resubmits cases whose result.json is missing or not status=ok using the saved submit_cmd.json.",
    )
    parser.add_argument(
        "--server-policy",
        choices=["default", "config"],
        default="default",
        help="Use the default real policy unless set to 'config' to preserve the server config policy.",
    )
    parser.add_argument("--output-dir", default="experiments/sweeps/slurm")
    parser.add_argument(
        "--schedulers",
        default="fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action",
    )
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--max-batch-size", default="")
    parser.add_argument("--alpha", default="")
    parser.add_argument("--account", default="")
    parser.add_argument(
        "--cluster",
        choices=["pace", "ice"],
        default="pace",
        help="Cluster preset: pace (Phoenix; L40S+V100 constraints, embers QoS) or ice (gpu:TYPE:N gres).",
    )
    parser.add_argument("--time", default="1:00:00")
    parser.add_argument("--server-mem", default="32G", help="Memory for the L40S server component.")
    parser.add_argument(
        "--client-mem", default="128G", help="Memory for the V100 client component."
    )
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
    if args.requeue:
        run_root = pathlib.Path(args.requeue)
        if not run_root.is_dir():
            raise SystemExit(f"--requeue path is not a directory: {run_root}")
        _requeue_run(run_root, dry_run=args.dry_run)
        return

    if not args.client_config:
        raise SystemExit("--client-config is required (unless --requeue is set).")

    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    run_root = pathlib.Path(args.output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    server_paths = _server_config_paths(args.server_config)
    server_variants = [
        (_server_variant_name(path), json.loads(path.read_text())) for path in server_paths
    ]
    client_paths = _client_config_paths(args.client_config)
    resolved_client_config = _resolve_path(args.client_config)
    config_root = resolved_client_config if resolved_client_config.is_dir() else None
    experiment_configs = [
        (_experiment_name(path, root=config_root), _read_experiment_config(path))
        for path in client_paths
    ]
    client_args = {
        "experiment_config": "",
        "progress_type": "logging",
        "overwrite": True,
    }
    if args.server_policy == "default":
        server_variants = [
            (variant, {**server_args, "policy": {"type": "default"}})
            for variant, server_args in server_variants
        ]
    cases = _make_cases(
        server_variants=server_variants,
        client_args=client_args,
        experiment_configs=experiment_configs,
        schedulers=parse_list_args(args.schedulers),
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
            "experiment": case.experiment_name,
            "num_robots": case.num_robots,
            "seed": case.seed,
            "max_batch_size": case.max_batch_size,
            "alpha": case.alpha,
            "server_variant": case.server_variant,
            "action_horizon_multiplier": case.action_horizon_multiplier,
            "action_horizon_multipliers": case.client_args.get("action_horizon_multipliers", {}),
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
            collector_job_id = _submit_collector(
                run_root=run_root, stamp=stamp, job_ids=job_ids, args=args
            )
            print(f"Submitted collector {collector_job_id} after {len(job_ids)} case job(s).")

    print(f"Prepared {len(rows)} case(s) under {run_root}")
    if args.dry_run:
        print("Dry run only; no Slurm jobs submitted.")


if __name__ == "__main__":
    main()
