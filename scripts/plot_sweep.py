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
    "success_rate",
]


METRIC_LABELS = {
    "starvation_rate": "Starvation rate",
    "post_first_starvation_rate": "Starvation rate excl. startup",
    "robot_starvation_rate_max": "Worst robot starvation rate",
    "robot_starvation_rate_std": "Robot starvation std. dev.",
    "robot_starvation_rate_cvar90": "Tail robot starvation rate",
    "success_rate": "Success rate",
    "step_interval_p95_ms": "Step interval p95 (ms)",
    "inference_p99_ms": "Inference latency p99 (ms)",
    "inbound_p95_ms": "Client→server transport p95 (ms)",
    "outbound_p95_ms": "Server→client transport p95 (ms)",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--x", default="num_robots")
    parser.add_argument("--line", default="scheduler")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    return parser.parse_args()


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion p estimated from n observations."""
    if n == 0:
        return (p, p)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5 / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    x_col: str,
    line_col: str,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    agg = df.groupby([line_col, x_col])[metric].agg(["mean", "count"]).reset_index()
    agg.columns = [line_col, x_col, "value", "n"]
    agg = agg.sort_values([line_col, x_col])

    is_proportion = agg["value"].between(0.0, 1.0).all()
    if is_proportion:
        ci = agg.apply(
            lambda r: pd.Series(_wilson_ci(r["value"], int(r["n"])), index=["lo", "hi"]), axis=1
        )
        agg = pd.concat([agg, ci], axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for line_value, group in agg.groupby(line_col):
        xs = group[x_col].to_numpy()
        ys = group["value"].to_numpy()
        (line,) = ax.plot(xs, ys, marker="o", linewidth=2.0, label=str(line_value))
        if is_proportion and (group["n"] > 1).any():
            ax.fill_between(
                xs,
                group["lo"].to_numpy(),
                group["hi"].to_numpy(),
                alpha=0.15,
                color=line.get_color(),
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


def plot_results(
    results: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    *,
    x: str = "num_robots",
    line: str = "scheduler",
    metrics: list[str] | None = None,
) -> None:
    output_dir = output_dir or (results.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        metrics = list(DEFAULT_METRICS)

    df = pd.read_csv(results)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise SystemExit("No successful rows found in results CSV")

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise SystemExit(f"Missing metric column(s): {', '.join(missing)}")
    for column in [x, line, *metrics]:
        if column not in df.columns:
            raise SystemExit(f"Missing required column: {column}")

    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    numeric_x = pd.to_numeric(df[x], errors="coerce")
    if numeric_x.notna().all():
        df[x] = numeric_x

    written = [
        _plot_metric(df, metric=metric, x_col=x, line_col=line, output_dir=output_dir)
        for metric in metrics
    ]
    print("Wrote plots:")
    for path in written:
        print(path)


def main() -> None:
    args = _parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    plot_results(
        args.results,
        args.output_dir,
        x=args.x,
        line=args.line,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
