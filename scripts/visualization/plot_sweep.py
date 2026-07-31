"""Plot all available summary views for a collected scheduler sweep.

Example:
    uv run python scripts/visualization/plot_sweep.py \
        --results experiments/sweeps/mock/sweep_results.csv

    uv run python scripts/visualization/plot_sweep.py \
        --action-chunks runs/20260730_221114/0/0_mock_0_failure/action_chunks.parquet
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from action_fate import plot_action_fate, plot_action_fate_sweep

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

STARVATION_METRICS = {
    "starvation_rate",
    "post_first_starvation_rate",
    "robot_starvation_rate_max",
    "robot_starvation_rate_std",
    "robot_starvation_rate_cvar90",
}


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _metric_higher_is_better(metric: str) -> bool:
    return metric in {"success_rate"}


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (p, p)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5 / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _aggregate_metric(
    df: pd.DataFrame, group_cols: list[str], metric: str, *, reduce: str
) -> pd.DataFrame:
    reducer = "min" if reduce == "min" else "mean"
    agg = df.groupby(group_cols, dropna=False)[metric].agg([reducer, "count"]).reset_index()
    return agg.rename(columns={reducer: "value", "count": "n"})


def _swept_line_values(df: pd.DataFrame, line_col: str) -> set[object]:
    if "max_batch_size" not in df.columns:
        return set()
    param = pd.to_numeric(df["max_batch_size"], errors="coerce")
    param_df = df.assign(_max_batch_size=param).dropna(subset=["_max_batch_size"])
    counts = param_df.groupby(line_col)["_max_batch_size"].nunique()
    return set(counts[counts > 1].index)


def _plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    x_col: str,
    line_col: str,
    output_dir: pathlib.Path,
    reduce: str = "mean",
) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    swept_lines = _swept_line_values(df, line_col)
    is_proportion = reduce == "mean" and df[metric].dropna().between(0.0, 1.0).all()

    def _fill_ci(agg: pd.DataFrame, color) -> None:
        if not (is_proportion and (agg["n"] > 1).any()):
            return
        ci = agg.apply(
            lambda r: pd.Series(_wilson_ci(r["value"], int(r["n"])), index=["lo", "hi"]),
            axis=1,
        )
        ax.fill_between(
            agg[x_col].to_numpy(),
            ci["lo"].to_numpy(),
            ci["hi"].to_numpy(),
            alpha=0.15,
            color=color,
        )

    for line_value in sorted(df[line_col].dropna().unique(), key=str):
        line_df = df[df[line_col] == line_value].copy()

        if line_value not in swept_lines:
            agg = _aggregate_metric(line_df, [x_col], metric, reduce=reduce).sort_values(x_col)
            (line,) = ax.plot(
                agg[x_col].to_numpy(),
                agg["value"].to_numpy(),
                marker="o",
                linewidth=2.0,
                label=str(line_value),
            )
            _fill_ci(agg, line.get_color())
            continue

        line_df["_max_batch_size"] = pd.to_numeric(line_df["max_batch_size"], errors="coerce")
        line_df = line_df.dropna(subset=["_max_batch_size"])
        param_agg = _aggregate_metric(
            line_df, ["_max_batch_size", x_col], metric, reduce=reduce
        ).sort_values(["_max_batch_size", x_col])
        if param_agg.empty:
            continue

        color = None
        for _, group in param_agg.groupby("_max_batch_size"):
            (param_line,) = ax.plot(
                group[x_col].to_numpy(),
                group["value"].to_numpy(),
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                alpha=0.32,
                color=color,
                label="_nolegend_",
            )
            if color is None:
                color = param_line.get_color()

        if _metric_higher_is_better(metric):
            best_idx = param_agg.groupby(x_col)["value"].idxmax()
        else:
            best_idx = param_agg.groupby(x_col)["value"].idxmin()
        best = param_agg.loc[best_idx].sort_values(x_col)
        ax.plot(
            best[x_col].to_numpy(),
            best["value"].to_numpy(),
            marker="o",
            linewidth=2.8,
            color=color,
            label=f"{line_value} (best max_batch_size)",
        )
        _fill_ci(best, color)

    title_suffix = " (best seed)" if reduce == "min" else ""
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(_metric_label(metric) + title_suffix)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title=line_col.replace("_", " ").title())
    fig.tight_layout()

    filename = (
        f"{metric}_by_{x_col}_min_seed.png" if reduce == "min" else f"{metric}_by_{x_col}.png"
    )
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_metric_results(
    results: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    x: str,
    line: str,
    metrics: list[str],
) -> list[pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise SystemExit("No successful rows found in results CSV")
    for column in [x, line]:
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
    written += [
        _plot_metric(df, metric=metric, x_col=x, line_col=line, output_dir=output_dir, reduce="min")
        for metric in metrics
        if metric in STARVATION_METRICS
    ]
    return written


def plot_results(
    results_csv: pathlib.Path,
    plots_dir: pathlib.Path | None = None,
    *,
    x: str = "num_robots",
    line: str = "scheduler",
    metrics: list[str] | None = None,
) -> list[pathlib.Path]:
    plots_dir = plots_dir or (results_csv.parent / "plots")

    columns = set(pd.read_csv(results_csv, nrows=0).columns)
    available = [m for m in (metrics or DEFAULT_METRICS) if m in columns]
    written: list[pathlib.Path] = []
    if available:
        written.extend(
            plot_metric_results(
                results_csv, plots_dir / "metrics", x=x, line=line, metrics=available
            )
        )
    else:
        print("Skipping metric plots: no requested metric columns are present.")

    written.extend(plot_action_fate_sweep(results_csv, plots_dir / "action_fate", x=x, line=line))
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--x", default="num_robots")
    parser.add_argument("--line", default="scheduler")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--action-chunks",
        type=pathlib.Path,
        nargs="*",
        default=None,
        help="action_chunks.parquet file(s) or directories for a single-run action-fate plot",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.results is None and not args.action_chunks:
        raise SystemExit("Provide --results, --action-chunks, or both")

    written: list[pathlib.Path] = []
    if args.results is not None:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        written.extend(
            plot_results(args.results, args.output_dir, x=args.x, line=args.line, metrics=metrics)
        )

    if args.action_chunks:
        output_dir = args.output_dir
        if output_dir is None:
            first = args.action_chunks[0]
            output_dir = (first if first.is_dir() else first.parent) / "plots"
        written.extend(plot_action_fate(args.action_chunks, output_dir))

    if written:
        print("Wrote sweep plot outputs:")
        for path in written:
            print(path)
    else:
        print("No sweep plots were written.")


if __name__ == "__main__":
    main()
