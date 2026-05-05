"""Plot scheduler sweep metrics from scripts/modal_sweep.py output.

Example:
    uv run python scripts/plot_sweep.py \
        --results experiments/sweeps/mock/sweep_results.csv \
        --output-dir experiments/sweeps/mock/plots
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METRICS = [
    "starvation_rate",
    "post_first_starvation_rate",
    "robot_starvation_rate_max",
    "service_jain_fairness",
    "success_rate",
]


METRIC_LABELS = {
    "starvation_rate": "Starvation rate",
    "post_first_starvation_rate": "Starvation rate excl. startup",
    "robot_starvation_rate_max": "Worst robot starvation rate",
    "robot_starvation_rate_std": "Robot starvation std. dev.",
    "robot_starvation_rate_cvar90": "Tail robot starvation rate",
    "service_jain_fairness": "Service Jain fairness",
    "success_rate": "Success rate",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--x", default="num_robots")
    parser.add_argument("--line", default="scheduler")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation used when multiple seeds/runs share the same x and line values.",
    )
    return parser.parse_args()


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    x_col: str,
    line_col: str,
    aggregate: str,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    grouped = (
        df.groupby([line_col, x_col], as_index=False)[metric]
        .agg(aggregate)
        .sort_values([line_col, x_col])
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for line_value, group in grouped.groupby(line_col):
        ax.plot(
            group[x_col],
            group[metric],
            marker="o",
            linewidth=2.0,
            label=str(line_value),
        )

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(_metric_label(metric))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title=line_col.replace("_", " ").title())
    fig.tight_layout()

    output_path = output_dir / f"{metric}_by_{x_col}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or (args.results.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise SystemExit("No successful rows found in results CSV")

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    missing = [metric for metric in metrics if metric not in df.columns]
    if missing:
        raise SystemExit(f"Missing metric column(s): {', '.join(missing)}")
    for column in [args.x, args.line, *metrics]:
        if column not in df.columns:
            raise SystemExit(f"Missing required column: {column}")

    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    numeric_x = pd.to_numeric(df[args.x], errors="coerce")
    if numeric_x.notna().all():
        df[args.x] = numeric_x

    written = [
        _plot_metric(
            df,
            metric=metric,
            x_col=args.x,
            line_col=args.line,
            aggregate=args.aggregate,
            output_dir=output_dir,
        )
        for metric in metrics
    ]
    print("Wrote plots:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
