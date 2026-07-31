"""Collect Slurm sweep case outputs into Modal-compatible result CSVs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "visualization"))

from scripts.modal.utils import summarize, write_rows  # noqa: E402


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def collect_case(
    case_dir: pathlib.Path, *, status: str | None = None, error: str = ""
) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    client_args = _load_json(case_dir / "client_args.json")
    server_args = _load_json(case_dir / "server_args.json")
    output_dir = pathlib.Path(client_args.get("output_dir", case_dir))
    experiment = client_args.get("experiment_config") or {}
    scheduler = client_args.get("scheduler_config") or {}
    server = server_args.get("server") or {}
    server_scheduler = server.get("scheduler") or {}
    results_path = output_dir / "results.csv"

    row: dict[str, Any] = {
        "run_id": case.get("run_id", case_dir.name),
        "status": status or ("ok" if results_path.is_file() else "missing"),
        "error": error,
        "artifact_path": str(output_dir),
        "case_dir": str(case_dir),
        "scheduler": case.get(
            "scheduler",
            scheduler.get("scheduling_algorithm", server_scheduler.get("scheduling_algorithm", "")),
        ),
        "num_robots": case.get("num_robots", len(experiment.get("robots") or [])),
        "seed": case.get("seed", experiment.get("seed", "")),
        "max_batch_size": case.get("max_batch_size", server.get("max_batch_size", "")),
        "alpha": case.get("alpha", scheduler.get("alpha", server_scheduler.get("alpha", ""))),
    }
    if output_dir.exists():
        row.update(summarize(output_dir))
    if row["status"] == "ok" and not results_path.is_file():
        row["status"] = "missing"
        row["error"] = row["error"] or "client results.csv not found"
    return row


def write_case_result(case_dir: pathlib.Path, *, status: str, error: str) -> pathlib.Path:
    row = collect_case(case_dir, status=status, error=error)
    result_path = case_dir / "result.json"
    result_path.write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(f"Wrote {result_path}")
    return result_path


def collect_run(
    output_dir: pathlib.Path, *, stamp: str | None = None, plots: bool = True
) -> pathlib.Path:
    root = output_dir / stamp if stamp else output_dir
    case_dirs = sorted(path.parent for path in root.glob("**/case.json"))
    rows: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        result_path = case_dir / "result.json"
        if result_path.exists():
            rows.append(_load_json(result_path))
        else:
            rows.append(collect_case(case_dir))

    if stamp is None:
        stamp = root.name
    out = root / f"sweep_results_{stamp}.csv"
    write_rows(out, rows)
    if plots:
        from plot_sweep import plot_results  # noqa: PLC0415

        plot_results(out, root / "plots")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=pathlib.Path)
    parser.add_argument("--write-result", action="store_true")
    parser.add_argument("--status", default="ok")
    parser.add_argument("--error", default="")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--no-plots", action="store_true", help="Only write the sweep results CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.case_dir:
        if args.write_result:
            write_case_result(args.case_dir, status=args.status, error=args.error)
        else:
            print(
                json.dumps(
                    collect_case(args.case_dir, status=args.status, error=args.error), indent=2
                )
            )
        return
    if args.output_dir is None:
        raise SystemExit("Provide --case-dir or --output-dir.")
    collect_run(args.output_dir, stamp=args.stamp, plots=not args.no_plots)


if __name__ == "__main__":
    main()
