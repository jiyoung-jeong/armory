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

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import run  # noqa: E402
from scripts.modal.utils import write_rows  # noqa: E402
from scripts.sweep_cases import Case, build_cases, parse_list_args  # noqa: E402


def materialize(case: Case, case_dir: pathlib.Path, stamp: str) -> None:
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
    output_dir = case_dir / "output"
    server = case.server.model_copy(
        update={
            "log_dir": str(case_dir / "logs"),
            "server": case.server.server.model_copy(update={"output_dir": case_dir}),
        }
    )
    client = run.Args(
        experiment_config=case.experiment,
        scheduler_config=server.server.scheduler,
        output_dir=output_dir,
        overwrite=True,
    )
    for name, model in (("server_args.json", server), ("client_args.json", client)):
        (case_dir / name).write_text(
            json.dumps(model.model_dump(mode="json", exclude={"json_path"}), indent=2) + "\n"
        )
    (case_dir / "case.json").write_text(
        json.dumps({**case.row(stamp), "case_dir": str(case_dir)}, indent=2) + "\n"
    )


def submit_cmd(case_dir: pathlib.Path, num_robots: int, args: argparse.Namespace) -> list[str]:
    client_cpus = max(8, num_robots * args.cpus_per_robot)
    account = [f"--account={args.account}"] if args.account else []
    qos = [f"--qos={args.qos}"] if args.qos else []
    time = f"--time={args.time}"

    cmd = ["sbatch", "--parsable", f"--export=ALL,ARMORY_SCRIPTS_DIR={HERE}"]
    if args.cluster == "pace":
        cmd += [
            "--constraint=gpu-l40s",
            "--gres=gpu:1",
            "--ntasks=1",
            "--cpus-per-task=4",
            f"--mem={args.server_mem}",
            "--qos=embers",
            time,
            *account,
            ":",
            "--constraint=V100",
            "--nodes=1",
            f"--gres=gpu:{max(1, math.ceil(client_cpus / 11))}",
            "--ntasks=1",
            f"--cpus-per-task={client_cpus}",
            f"--mem={args.client_mem}",
            time,
            *account,
        ]
    elif args.cluster == "ice":
        if num_robots > 20:
            client_gpus = f"L40S:{math.ceil(num_robots / 8)}"
            client_cpus = num_robots
        else:
            client_gpus = f"V100:{max(1, math.ceil(client_cpus / 11))}"
        cmd += [
            "--gres=gpu:L40S:1",
            "--ntasks=1",
            "--cpus-per-task=4",
            f"--mem={args.server_mem}",
            time,
            *account,
            ":",
            "--nodes=1",
            f"--gres=gpu:{client_gpus}",
            "--ntasks=1",
            f"--cpus-per-task={client_cpus}",
            f"--mem={args.client_mem}",
            time,
            *account,
        ]
    else:
        cmd += [
            "--partition=overcap",
            "--gres=gpu:l40s:1",
            "--ntasks=1",
            "--cpus-per-task=4",
            f"--mem={args.server_mem}",
            time,
            *account,
            *qos,
            ":",
            "--partition=overcap",
            "--nodes=1",
            "--gres=gpu:a40:1",
            "--ntasks=1",
            f"--cpus-per-task={client_cpus}",
            f"--mem={args.client_mem}",
            time,
            *account,
            *qos,
        ]
    return cmd + [str(HERE / "run_case.sh"), str(case_dir)]


def run_sbatch(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed\ncommand: {shlex.join(cmd)}\n"
            f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def case_succeeded(case_dir: pathlib.Path) -> bool:
    result = case_dir / "result.json"
    return result.exists() and json.loads(result.read_text()).get("status") == "ok"


def requeue(run_root: pathlib.Path, *, dry_run: bool) -> None:
    case_dirs = sorted({p.parent for p in run_root.glob("**/case.json")})
    if not case_dirs:
        raise SystemExit(f"No case.json files found under {run_root}")
    failed = [d for d in case_dirs if not case_succeeded(d)]
    print(f"{len(failed)} of {len(case_dirs)} case(s) under {run_root} are not status=ok.")

    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, Any]] = []
    for case_dir in failed:
        rel = case_dir.relative_to(run_root)
        if not (case_dir / "submit_cmd.json").exists():
            print(f"[skip] {rel}: no submit_cmd.json")
            continue
        argv = [str(x) for x in json.loads((case_dir / "submit_cmd.json").read_text())["argv"]]
        if dry_run:
            print(f"[dry] would requeue {rel}: {shlex.join(argv)}")
            rows.append({"case_dir": str(case_dir), "job_id": "", "status": "dry_run"})
            continue
        for name in ("result.json", "logs"):
            if (case_dir / name).exists():
                (case_dir / name).rename(case_dir / f"{name}.previous_{stamp}")
        job_id = run_sbatch(argv)
        print(f"Requeued {job_id}: {rel}")
        rows.append({"case_dir": str(case_dir), "job_id": job_id, "status": "requeued"})

    if rows:
        write_rows(run_root / f"requeue_{stamp}.csv", rows)


def submit_collector(run_root: pathlib.Path, job_ids: list[str], args: argparse.Namespace) -> str:
    dependency_ids = [job_id.split(";", 1)[0] for job_id in job_ids if job_id]
    cmd = ["sbatch", "--parsable", f"--dependency=afterany:{':'.join(dependency_ids)}"]
    if args.account:
        cmd.append(f"--account={args.account}")
    cmd += [str(HERE / "collect_results.sh"), str(run_root)]
    collector_job_id = run_sbatch(cmd)
    (run_root / "collector.json").write_text(
        json.dumps(
            {
                "collector_job_id": collector_job_id,
                "dependency": "afterany:" + ":".join(dependency_ids),
                "case_job_ids": job_ids,
            },
            indent=2,
        )
        + "\n"
    )
    return collector_job_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-config", default="")
    parser.add_argument("--client-config", default="")
    parser.add_argument("--requeue", default="")
    parser.add_argument("--output-dir", default="experiments/sweeps/slurm")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--cluster", choices=["skynet", "pace", "ice"], default="skynet")
    parser.add_argument("--account", default="")
    parser.add_argument("--qos", default="")
    parser.add_argument("--time", default="1:00:00")
    parser.add_argument("--server-mem", default="32G")
    parser.add_argument("--client-mem", default="128G")
    parser.add_argument("--cpus-per-robot", type=int, default=2)
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
        requeue(run_root, dry_run=args.dry_run)
        return

    if not args.server_config or not args.client_config:
        raise SystemExit("--server-config and --client-config are required (unless --requeue).")

    stamp = args.stamp or dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    run_root = pathlib.Path(args.output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    cases = build_cases(
        args.server_config, args.client_config, parse_list_args(args.seeds, cast=int)
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_dir = run_root / case.run_id
        materialize(case, case_dir, stamp)
        cmd = submit_cmd(case_dir, len(case.experiment.robots), args)
        (case_dir / "submit_cmd.json").write_text(
            json.dumps({"argv": cmd, "shell": shlex.join(cmd), "cwd": str(REPO_ROOT)}, indent=2)
            + "\n"
        )
        row = {
            **json.loads((case_dir / "case.json").read_text()),
            "status": "dry_run" if args.dry_run else "submitted",
            "job_id": "",
        }
        if args.dry_run:
            print(f"Prepared {case.run_id}: {case_dir}")
        else:
            row["job_id"] = run_sbatch(cmd)
            print(f"Submitted {row['job_id']}: {case.run_id}")
        rows.append(row)

    write_rows(run_root / f"jobs_{stamp}.csv", rows)

    if args.submit_collector and not args.dry_run:
        job_ids = [str(row["job_id"]) for row in rows if row["job_id"]]
        print(f"Submitted collector {submit_collector(run_root, job_ids, args)}")

    print(f"Prepared {len(rows)} case(s) under {run_root}")
    if args.dry_run:
        print("Dry run only; no Slurm jobs submitted.")


if __name__ == "__main__":
    main()
