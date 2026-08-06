from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.modal.utils import summarize, write_rows  # noqa: E402


def collect_case(case_dir: pathlib.Path, *, status: str = "", error: str = "") -> dict[str, Any]:
    case = json.loads((case_dir / "case.json").read_text())
    output_dir = case_dir / "output"
    row = {
        **case,
        "status": status or ("ok" if output_dir.exists() else "missing"),
        "error": error,
        "artifact_path": str(output_dir),
    }
    if output_dir.exists():
        row.update(summarize(output_dir))
    return row


def write_case_result(case_dir: pathlib.Path, *, status: str, error: str) -> None:
    row = collect_case(case_dir, status=status, error=error)
    (case_dir / "result.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(f"Wrote {case_dir / 'result.json'}")


def collect_run(run_root: pathlib.Path, *, plots: bool = True) -> pathlib.Path:
    case_dirs = sorted(path.parent for path in run_root.glob("**/case.json"))
    rows = [
        json.loads((d / "result.json").read_text())
        if (d / "result.json").exists()
        else collect_case(d)
        for d in case_dirs
    ]
    out = run_root / f"sweep_results_{run_root.name}.csv"
    write_rows(out, rows)
    if plots:
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "visualization"))
        from plot_sweep import plot_results  # noqa: PLC0415

        plot_results(out, run_root / "plots")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=pathlib.Path)
    parser.add_argument("--write-result", action="store_true")
    parser.add_argument("--status", default="ok")
    parser.add_argument("--error", default="")
    parser.add_argument("--run-dir", type=pathlib.Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.case_dir:
        if args.write_result:
            write_case_result(args.case_dir, status=args.status, error=args.error)
        else:
            print(json.dumps(collect_case(args.case_dir, status=args.status), indent=2))
        return
    if args.run_dir is None:
        raise SystemExit("Provide --case-dir or --run-dir.")
    collect_run(args.run_dir, plots=not args.no_plots)


if __name__ == "__main__":
    main()
