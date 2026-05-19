"""Plot all available summary views for a collected scheduler sweep."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from plot_alpha_fairness import plot_results as plot_alpha_fairness_results
from plot_starvation_sweep import DEFAULT_METRICS, plot_action_fate_sweep
from plot_starvation_sweep import plot_results as plot_starvation_sweep_results


def _available_metrics(results_csv: pathlib.Path, requested: list[str] | None) -> list[str]:
    columns = set(pd.read_csv(results_csv, nrows=0).columns)
    metrics = requested or list(DEFAULT_METRICS)
    return [metric for metric in metrics if metric in columns]


def plot_results(
    results_csv: pathlib.Path,
    plots_dir: pathlib.Path | None = None,
    *,
    x: str = "num_robots",
    line: str = "scheduler",
    metrics: list[str] | None = None,
) -> list[pathlib.Path]:
    """Generate every plot supported by the available sweep result columns."""
    plots_dir = plots_dir or (results_csv.parent / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    written: list[pathlib.Path] = []
    available_metrics = _available_metrics(results_csv, metrics)
    if available_metrics:
        written.extend(
            plot_starvation_sweep_results(
                results_csv,
                plots_dir / "metrics",
                x=x,
                line=line,
                metrics=available_metrics,
            )
        )
    else:
        print("Skipping metric plots: no requested metric columns are present.")

    written.extend(plot_action_fate_sweep(results_csv, plots_dir / "action_fate", x=x, line=line))
    written.extend(plot_alpha_fairness_results(results_csv, plots_dir / "alpha_fairness"))

    if written:
        print("Wrote sweep plot outputs:")
        for path in written:
            print(path)
    else:
        print("No sweep plots were written.")
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--x", default="num_robots")
    parser.add_argument("--line", default="scheduler")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    plot_results(args.results, args.output_dir, x=args.x, line=args.line, metrics=metrics)


if __name__ == "__main__":
    main()
