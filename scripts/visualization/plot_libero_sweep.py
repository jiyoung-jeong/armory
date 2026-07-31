"""Summarize a collected Modal LIBERO sweep and render its paper artifacts.

This is the one-command entry point for the historical summary-line plots,
per-slice bar plots, and LaTeX tables.  Run it after the artifact download has
finished so the collector sees a stable case set.

Example:
    uv run python scripts/visualization/plot_libero_sweep.py \
        experiments/sweeps/libero_5min/libero_5min_paper_new \
        --num-robots 2,4,6,8,10 --max-batch-size 3
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artifact_root", type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Summary destination (default: <artifact-root>/_summary).",
    )
    parser.add_argument("--num-robots", default=None, help="Comma-separated robot counts.")
    parser.add_argument("--max-batch-size", default=None, help="Comma-separated batch sizes.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Stop before plotting if any discovered case is failed or incomplete.",
    )
    parser.add_argument("--skip-bars", action="store_true")
    parser.add_argument("--skip-latex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    if not artifact_root.is_dir():
        raise SystemExit(f"Not a directory: {artifact_root}")
    output_dir = (args.output_dir or (artifact_root / "_summary")).resolve()

    scripts_dir = pathlib.Path(__file__).resolve().parent
    summarize = [
        sys.executable,
        str(scripts_dir / "summarize_sweep_runs.py"),
        str(artifact_root),
        "--output-dir",
        str(output_dir),
    ]
    if args.require_complete:
        summarize.append("--require-complete")
    _run(summarize)

    lines = [sys.executable, str(scripts_dir / "plot_summary_lines.py"), str(output_dir)]
    if args.num_robots:
        lines.extend(("--num-robots", args.num_robots))
    if args.max_batch_size:
        lines.extend(("--max-batch-size", args.max_batch_size))
    _run(lines)

    if not args.skip_bars:
        _run([sys.executable, str(scripts_dir / "plot_summary_bars.py"), str(output_dir)])

    if not args.skip_latex:
        latex = [
            sys.executable,
            str(scripts_dir / "generate_latex_tables.py"),
            str(output_dir),
        ]
        if args.num_robots:
            latex.extend(("--num-robots", args.num_robots))
        _run(latex)

    print(f"Paper artifacts are under {output_dir}")


if __name__ == "__main__":
    main()
