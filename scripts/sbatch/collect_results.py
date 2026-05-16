"""Collect Slurm sweep case outputs into Modal-compatible result CSVs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "modal"))

from _utils import summarize, write_rows  # noqa: E402


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def collect_case(case_dir: pathlib.Path, *, status: str | None = None, error: str = "") -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    client_args = _load_json(case_dir / "client_args.json")
    server_args = _load_json(case_dir / "server_args.json")
    output_dir = pathlib.Path(client_args.get("output_dir", case_dir / "outputs"))

    row: dict[str, Any] = {
        "run_id": case.get("run_id", case_dir.name),
        "status": status or ("ok" if output_dir.exists() else "missing"),
        "error": error,
        "artifact_path": str(output_dir),
        "case_dir": str(case_dir),
        "scheduler": case.get("scheduler", server_args.get("scheduling_algorithm", "")),
        "num_robots": case.get("num_robots", client_args.get("num_robots", "")),
        "seed": case.get("seed", client_args.get("seed", "")),
        "max_batch_size": case.get("max_batch_size", server_args.get("max_batch_size", "")),
        "alpha": case.get("alpha", server_args.get("alpha", "")),
    }
    if output_dir.exists():
        row.update(summarize(output_dir))
    if row["status"] == "ok" and not output_dir.exists():
        row["status"] = "missing"
        row["error"] = row["error"] or "client output directory not found"
    return row


def write_case_result(case_dir: pathlib.Path, *, status: str, error: str) -> pathlib.Path:
    row = collect_case(case_dir, status=status, error=error)
    result_path = case_dir / "result.json"
    result_path.write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(f"Wrote {result_path}")
    return result_path


def collect_run(output_dir: pathlib.Path, *, stamp: str | None = None) -> pathlib.Path:
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
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=pathlib.Path)
    parser.add_argument("--write-result", action="store_true")
    parser.add_argument("--status", default="ok")
    parser.add_argument("--error", default="")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--stamp", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.case_dir:
        if args.write_result:
            write_case_result(args.case_dir, status=args.status, error=args.error)
        else:
            print(json.dumps(collect_case(args.case_dir, status=args.status, error=args.error), indent=2))
        return
    if args.output_dir is None:
        raise SystemExit("Provide --case-dir or --output-dir.")
    collect_run(args.output_dir, stamp=args.stamp)


if __name__ == "__main__":
    main()
