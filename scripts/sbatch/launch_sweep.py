from __future__ import annotations

import argparse
import csv
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

from scripts import run, serve  # noqa: E402
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

    cmd = [
        "sbatch",
        "--parsable",
        f"--export=ALL,ARMORY_SCRIPTS_DIR={HERE},ARMORY_MAX_RETRIES={args.max_retries}",
    ]
    if args.max_retries:
        cmd += ["--requeue", "--open-mode=append"]
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
            *([f"--nodelist={args.server_nodelist}"] if args.server_nodelist else []),
            *([f"--exclude={args.server_exclude}"] if args.server_exclude else []),
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


def load_case_ids(path: pathlib.Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"--requeue-cases path is not a file: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "run_id" not in reader.fieldnames:
            raise SystemExit(f"--requeue-cases CSV must contain a run_id column: {path}")
        case_ids = [row["run_id"].strip() for row in reader if row["run_id"].strip()]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for run_id in case_ids:
        if run_id in seen:
            duplicates.add(run_id)
        seen.add(run_id)
    if duplicates:
        raise SystemExit(f"Duplicate run_id values in {path}: {', '.join(sorted(duplicates))}")
    if not case_ids:
        raise SystemExit(f"No run_id values found in {path}")
    return case_ids


def materialize_rerun(source_dir: pathlib.Path, case_dir: pathlib.Path, stamp: str) -> list[str]:
    (case_dir / "logs").mkdir(parents=True)
    server = serve.Args.from_json(source_dir / "server_args.json")
    server = server.model_copy(
        update={
            "log_dir": str(case_dir / "logs"),
            "server": server.server.model_copy(update={"output_dir": case_dir}),
        }
    )
    client = run.Args.from_json(source_dir / "client_args.json").model_copy(
        update={"output_dir": case_dir / "output", "overwrite": True}
    )
    for name, model in (("server_args.json", server), ("client_args.json", client)):
        (case_dir / name).write_text(
            json.dumps(model.model_dump(mode="json", exclude={"json_path"}), indent=2) + "\n"
        )

    case = json.loads((source_dir / "case.json").read_text())
    case.update({"stamp": stamp, "case_dir": str(case_dir)})
    (case_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n")

    argv = [str(x) for x in json.loads((source_dir / "submit_cmd.json").read_text())["argv"]]
    argv[-1] = str(case_dir)
    (case_dir / "submit_cmd.json").write_text(
        json.dumps({"argv": argv, "shell": shlex.join(argv), "cwd": str(REPO_ROOT)}, indent=2)
        + "\n"
    )
    return argv


def requeue(
    source_root: pathlib.Path,
    rerun_root: pathlib.Path,
    *,
    case_ids: list[str] | None,
    dry_run: bool,
) -> list[str]:
    case_dirs = sorted({p.parent for p in source_root.glob("**/case.json")})
    if not case_dirs:
        raise SystemExit(f"No case.json files found under {source_root}")
    by_run_id: dict[str, pathlib.Path] = {}
    for case_dir in case_dirs:
        case = json.loads((case_dir / "case.json").read_text())
        run_id = str(case.get("run_id") or case_dir.name)
        if run_id in by_run_id:
            raise SystemExit(f"Duplicate run_id under {source_root}: {run_id}")
        by_run_id[run_id] = case_dir

    if case_ids is None:
        selected = [d for d in case_dirs if not case_succeeded(d)]
        print(f"{len(selected)} of {len(case_dirs)} case(s) under {source_root} are not status=ok.")
    else:
        missing = [run_id for run_id in case_ids if run_id not in by_run_id]
        if missing:
            raise SystemExit(f"run_id values not found under {source_root}: {', '.join(missing)}")
        selected = [by_run_id[run_id] for run_id in case_ids]
        print(f"Selected {len(selected)} of {len(case_dirs)} case(s) from the requeue CSV.")

    if not selected:
        print("No cases selected; nothing to rerun.")
        return []
    if rerun_root.exists():
        raise SystemExit(f"Refusing to overwrite existing rerun directory: {rerun_root}")
    rerun_root.mkdir(parents=True)

    stamp = rerun_root.name
    rows: list[dict[str, Any]] = []
    for source_dir in selected:
        run_id = str(json.loads((source_dir / "case.json").read_text())["run_id"])
        case_dir = rerun_root / run_id
        argv = materialize_rerun(source_dir, case_dir, stamp)
        row = {
            **json.loads((case_dir / "case.json").read_text()),
            "status": "dry_run" if dry_run else "submitted",
            "job_id": "",
        }
        if dry_run:
            print(f"Prepared {run_id}: {case_dir}")
        else:
            row["job_id"] = run_sbatch(argv)
            print(f"Submitted {row['job_id']}: {run_id}")
        rows.append(row)

    write_rows(rerun_root / f"jobs_{stamp}.csv", rows)
    print(f"Prepared {len(rows)} rerun case(s) under {rerun_root}")
    if dry_run:
        print("Dry run only; no Slurm jobs submitted.")
    return [str(row["job_id"]) for row in rows if row["job_id"]]


def submit_collector(run_root: pathlib.Path, job_ids: list[str], args: argparse.Namespace) -> str:
    dependency_ids = [job_id.split(";", 1)[0] for job_id in job_ids if job_id]
    cmd = [
        "sbatch",
        "--parsable",
        f"--export=ALL,ARMORY_SCRIPTS_DIR={HERE}",
        f"--dependency=afterany:{':'.join(dependency_ids)}",
    ]
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
    parser.add_argument(
        "--requeue-cases",
        type=pathlib.Path,
        help="CSV with a run_id column selecting exactly which cases to requeue.",
    )
    parser.add_argument("--output-dir", default="experiments/sweeps/slurm")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--cluster", choices=["skynet", "pace", "ice"], default="skynet")
    parser.add_argument("--account", default="")
    parser.add_argument("--qos", default="")
    parser.add_argument("--time", default="1:00:00")
    parser.add_argument("--server-mem", default="32G")
    parser.add_argument("--server-nodelist", default="")
    parser.add_argument("--server-exclude", default="bishop")
    parser.add_argument("--client-mem", default="128G")
    parser.add_argument("--cpus-per-robot", type=int, default=2)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Automatically requeue each failed case at most this many times.",
    )
    parser.add_argument("--submit-collector", action="store_true")
    parser.add_argument("--stamp", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be non-negative.")
    if args.requeue:
        source_root = pathlib.Path(args.requeue)
        if not source_root.is_dir():
            raise SystemExit(f"--requeue path is not a directory: {source_root}")
        case_ids = load_case_ids(args.requeue_cases) if args.requeue_cases else None
        stamp = args.stamp or dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
        rerun_root = pathlib.Path(args.output_dir) / stamp
        job_ids = requeue(
            source_root,
            rerun_root,
            case_ids=case_ids,
            dry_run=args.dry_run,
        )
        if args.submit_collector and not args.dry_run and job_ids:
            print(f"Submitted collector {submit_collector(rerun_root, job_ids, args)}")
        return

    if args.requeue_cases:
        raise SystemExit("--requeue-cases requires --requeue")

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
