"""Plot scheduler sweep metrics from scripts/modal_sweep.py output.

Example:
    uv run python scripts/plot_sweep.py \
        --results experiments/sweeps/mock/sweep_results.csv \
        --output-dir experiments/sweeps/mock/plots

    uv run python scripts/experiments/plot_starvation_sweep.py \
        --action-chunks experiments/10/9/0_mock_0_failure/action_chunks.parquet
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import deque

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

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
        help=(
            "Optional action_chunks.parquet file(s) or directory/directories to summarize "
            "as an action-fate stacked bar plot."
        ),
    )
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


STARVATION_METRICS = {
    "starvation_rate",
    "post_first_starvation_rate",
    "robot_starvation_rate_max",
    "robot_starvation_rate_std",
    "robot_starvation_rate_cvar90",
}


def _metric_higher_is_better(metric: str) -> bool:
    return metric in {"success_rate"}


ACTION_FATE_LABELS = {
    "executed": "Executed",
    "lost_before_arrival": "Lost before arrival",
    "overwritten": "Overwritten",
    "cutoff_by_max_steps": "Cut off by max steps",
}

ACTION_FATE_COLORS = {
    "executed": "#4C78A8",
    "lost_before_arrival": "#F58518",
    "overwritten": "#E45756",
    "cutoff_by_max_steps": "#72B7B2",
}
ACTION_FATE_HATCHES = ["", "///", "\\\\\\", "xx", "...", "++", "oo", "**"]


def _discover_action_chunk_files(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in paths:
        if path.is_file():
            if path.name != "action_chunks.parquet":
                raise SystemExit(f"Expected action_chunks.parquet file, got: {path}")
            files.append(path)
        elif path.is_dir():
            files.extend(path.glob("**/action_chunks.parquet"))
        else:
            raise SystemExit(f"Action chunk path does not exist: {path}")
    return sorted(set(files))


def _load_steps_taken(chunk_file: pathlib.Path) -> int:
    metadata_file = chunk_file.parent / "metadata.json"
    if not metadata_file.exists():
        raise SystemExit(f"Missing metadata.json next to {chunk_file}")
    metadata = json.loads(metadata_file.read_text())
    try:
        return int(metadata["steps_taken"])
    except KeyError as exc:
        raise SystemExit(f"metadata.json missing steps_taken: {metadata_file}") from exc


def _action_fate_counts_for_episode(chunk_file: pathlib.Path) -> dict[str, int]:
    df = pd.read_parquet(chunk_file)
    needed = {"action_index_start", "execution_horizon", "execution_start_step"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{chunk_file} missing required column(s): {', '.join(sorted(missing))}")

    steps_taken = _load_steps_taken(chunk_file)
    sort_columns = [
        column
        for column in ["execution_start_step", "response_timestamp", "chunk_id"]
        if column in df.columns
    ]
    chunks = df.sort_values(sort_columns, kind="stable")

    queue: deque[int] = deque()
    next_action_step = 0
    counts = {
        "executed": 0,
        "lost_before_arrival": 0,
        "overwritten": 0,
        "cutoff_by_max_steps": 0,
    }

    arrivals_by_step: dict[int, list[tuple[int, int]]] = {}
    total_produced = 0
    for row in chunks.itertuples(index=False):
        start = int(row.action_index_start)
        horizon = int(row.execution_horizon)
        execution_start_step = int(row.execution_start_step)
        total_produced += horizon
        arrivals_by_step.setdefault(execution_start_step, []).append((start, horizon))

    def receive_chunk(start: int, horizon: int) -> None:
        while queue and queue[-1] >= start:
            queue.pop()
            counts["overwritten"] += 1

        for action_step in range(start, start + horizon):
            if action_step < next_action_step:
                counts["lost_before_arrival"] += 1
            else:
                queue.append(action_step)

    for step in range(steps_taken):
        for start, horizon in arrivals_by_step.pop(step, []):
            receive_chunk(start, horizon)
        if queue:
            queue.popleft()
            next_action_step += 1
            counts["executed"] += 1

    for arrivals in arrivals_by_step.values():
        for _, horizon in arrivals:
            counts["cutoff_by_max_steps"] += horizon
    counts["cutoff_by_max_steps"] += len(queue)

    accounted = sum(counts.values())
    if accounted != total_produced:
        raise RuntimeError(
            f"Action fate accounting mismatch for {chunk_file}: "
            f"accounted={accounted}, produced={total_produced}"
        )
    counts["produced"] = total_produced
    return counts


ACTION_FATE_KEYS = ["executed", "lost_before_arrival", "overwritten", "cutoff_by_max_steps"]


def _plot_action_fate_bar(counts: dict[str, int], output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bottom = 0
    total = counts["produced"]
    for key in ACTION_FATE_KEYS:
        value = counts[key]
        ax.bar(
            ["Produced actions"],
            [value],
            bottom=[bottom],
            color=ACTION_FATE_COLORS[key],
            label=ACTION_FATE_LABELS[key],
            width=0.48,
        )
        if value > 0 and total > 0:
            ax.text(
                0,
                bottom + value / 2,
                f"{value:,}\n{value / total:.1%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value / total > 0.08 else "black",
            )
        bottom += value

    ax.set_ylabel("Actions")
    ax.set_title("Produced Action Fate")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()

    output_path = output_dir / "action_fate_stacked_bar.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_action_fate(
    action_chunk_paths: list[pathlib.Path], output_dir: pathlib.Path
) -> list[pathlib.Path]:
    files = _discover_action_chunk_files(action_chunk_paths)
    if not files:
        raise SystemExit("No action_chunks.parquet files found")

    total_counts = {
        "executed": 0,
        "lost_before_arrival": 0,
        "overwritten": 0,
        "cutoff_by_max_steps": 0,
        "produced": 0,
    }
    for chunk_file in files:
        counts = _action_fate_counts_for_episode(chunk_file)
        for key, value in counts.items():
            total_counts[key] += value

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "action_fate_counts.csv"
    pd.DataFrame([total_counts]).to_csv(csv_path, index=False)
    return [_plot_action_fate_bar(total_counts, output_dir), csv_path]


def _counts_from_action_chunk_files(files: list[pathlib.Path]) -> dict[str, int]:
    counts = {key: 0 for key in [*ACTION_FATE_KEYS, "produced"]}
    for chunk_file in files:
        episode_counts = _action_fate_counts_for_episode(chunk_file)
        for key, value in episode_counts.items():
            counts[key] += value
    return counts


def _artifact_action_chunk_files(artifact_path: pathlib.Path) -> list[pathlib.Path]:
    output_path = artifact_path / "output"
    search_root = output_path if output_path.exists() else artifact_path
    return sorted(search_root.glob("**/action_chunks.parquet"))


def _load_action_fate_sweep(results: pathlib.Path, *, x_col: str, line_col: str) -> pd.DataFrame:
    df = pd.read_csv(results)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if df.empty or "artifact_path" not in df.columns:
        return pd.DataFrame()

    rows = []
    extra_cols = [
        col
        for col in ["max_batch_size", "starvation_rate", "post_first_starvation_rate"]
        if col in df.columns
    ]
    for _, row in df.iterrows():
        artifact_path = pathlib.Path(str(row["artifact_path"]))
        chunk_files = _artifact_action_chunk_files(artifact_path)
        if not chunk_files:
            continue

        counts = _counts_from_action_chunk_files(chunk_files)
        rows.append(
            {
                x_col: row[x_col],
                line_col: row[line_col],
                "run_id": row.get("run_id", ""),
                **{col: row[col] for col in extra_cols},
                **counts,
            }
        )

    if not rows:
        return pd.DataFrame()

    fate = pd.DataFrame(rows)
    numeric_x = pd.to_numeric(fate[x_col], errors="coerce")
    if numeric_x.notna().all():
        fate[x_col] = numeric_x
    if "max_batch_size" in fate.columns:
        fate["max_batch_size"] = pd.to_numeric(fate["max_batch_size"], errors="coerce")
    if "starvation_rate" in fate.columns:
        fate["starvation_rate"] = pd.to_numeric(fate["starvation_rate"], errors="coerce")
    return fate


def _aggregate_action_fate_sweep(fate: pd.DataFrame, *, x_col: str, line_col: str) -> pd.DataFrame:
    agg = (
        fate.groupby([x_col, line_col], dropna=False)[[*ACTION_FATE_KEYS, "produced"]]
        .sum()
        .reset_index()
        .sort_values([x_col, line_col], key=lambda s: s.map(str) if s.dtype == object else s)
    )
    return agg


def _best_starvation_param_rows(fate: pd.DataFrame, *, x_col: str, line_col: str) -> pd.DataFrame:
    if "max_batch_size" not in fate.columns or "starvation_rate" not in fate.columns:
        return pd.DataFrame()

    param_rows = fate.dropna(subset=["max_batch_size", "starvation_rate"]).copy()
    if param_rows.empty:
        return pd.DataFrame()

    best_rows = []
    for line_value, line_df in fate.groupby(line_col, dropna=False):
        line_params = param_rows[param_rows[line_col] == line_value]
        if line_params["max_batch_size"].nunique() <= 1:
            best_rows.append(line_df)
            continue

        param_metric = (
            line_params.groupby([x_col, "max_batch_size"], dropna=False)["starvation_rate"]
            .mean()
            .reset_index()
            .sort_values([x_col, "starvation_rate", "max_batch_size"])
        )
        best_param = param_metric.loc[param_metric.groupby(x_col)["starvation_rate"].idxmin()]
        keep = line_df.merge(
            best_param[[x_col, "max_batch_size"]],
            on=[x_col, "max_batch_size"],
            how="inner",
        )
        best_rows.append(keep)

    if not best_rows:
        return pd.DataFrame()
    return pd.concat(best_rows, ignore_index=True)


def _plot_action_fate_sweep(
    agg: pd.DataFrame,
    *,
    x_col: str,
    line_col: str,
    output_dir: pathlib.Path,
    normalize: bool,
    best_starvation: bool = False,
) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    x_values = sorted(agg[x_col].dropna().unique())
    line_values = sorted(agg[line_col].dropna().unique(), key=str)
    hatches = {
        line_value: ACTION_FATE_HATCHES[idx % len(ACTION_FATE_HATCHES)]
        for idx, line_value in enumerate(line_values)
    }
    width = min(0.8 / max(len(line_values), 1), 0.28)
    x_positions = {value: i for i, value in enumerate(x_values)}

    for line_idx, line_value in enumerate(line_values):
        group = agg[agg[line_col] == line_value].sort_values(x_col)
        offset = (line_idx - (len(line_values) - 1) / 2) * width
        bottoms = [0.0] * len(group)
        xs = [x_positions[value] + offset for value in group[x_col]]
        produced = group["produced"].replace(0, pd.NA)

        for key in ACTION_FATE_KEYS:
            values = group[key] / produced if normalize else group[key]
            values = values.fillna(0.0).to_numpy(dtype=float)
            ax.bar(
                xs,
                values,
                bottom=bottoms,
                width=width,
                color=ACTION_FATE_COLORS[key],
                edgecolor="#222222",
                linewidth=0.35,
                hatch=hatches[line_value],
                label=ACTION_FATE_LABELS[key] if line_idx == 0 else "_nolegend_",
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Share of produced actions" if normalize else "Actions")
    title = "Action Fate by Scheduler"
    if best_starvation:
        title += " (best starvation max_batch_size)"
    if normalize:
        title += " (fraction)"
    ax.set_title(title)
    if normalize:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, axis="y", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    fate_legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(fate_legend)
    line_handles = [
        Patch(
            facecolor="white",
            edgecolor="#222222",
            hatch=hatches[value],
            label=str(value),
        )
        for value in line_values
    ]
    ax.legend(
        handles=line_handles,
        title=line_col.replace("_", " ").title(),
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )
    fig.tight_layout()

    suffix = "fraction" if normalize else "counts"
    best_suffix = "_best_starvation_param" if best_starvation else ""
    output_path = output_dir / f"action_fate_{suffix}{best_suffix}_by_{x_col}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_action_fate_sweep(
    results: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    x: str,
    line: str,
) -> list[pathlib.Path]:
    fate = _load_action_fate_sweep(results, x_col=x, line_col=line)
    if fate.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / "action_fate_runs.csv"
    fate.to_csv(run_csv, index=False)

    agg = _aggregate_action_fate_sweep(fate, x_col=x, line_col=line)
    agg_csv = output_dir / "action_fate_by_sweep_group.csv"
    agg.to_csv(agg_csv, index=False)

    written = [
        _plot_action_fate_sweep(agg, x_col=x, line_col=line, output_dir=output_dir, normalize=True),
        _plot_action_fate_sweep(
            agg, x_col=x, line_col=line, output_dir=output_dir, normalize=False
        ),
        run_csv,
        agg_csv,
    ]

    best_fate = _best_starvation_param_rows(fate, x_col=x, line_col=line)
    if not best_fate.empty:
        best_run_csv = output_dir / "action_fate_runs_best_starvation_param.csv"
        best_fate.to_csv(best_run_csv, index=False)

        best_agg = _aggregate_action_fate_sweep(best_fate, x_col=x, line_col=line)
        best_agg_csv = output_dir / "action_fate_by_sweep_group_best_starvation_param.csv"
        best_agg.to_csv(best_agg_csv, index=False)
        written.extend(
            [
                _plot_action_fate_sweep(
                    best_agg,
                    x_col=x,
                    line_col=line,
                    output_dir=output_dir,
                    normalize=True,
                    best_starvation=True,
                ),
                _plot_action_fate_sweep(
                    best_agg,
                    x_col=x,
                    line_col=line,
                    output_dir=output_dir,
                    normalize=False,
                    best_starvation=True,
                ),
                best_run_csv,
                best_agg_csv,
            ]
        )

    return written


def _aggregate_metric(
    df: pd.DataFrame, group_cols: list[str], metric: str, *, reduce: str
) -> pd.DataFrame:
    if reduce == "min":
        agg = df.groupby(group_cols, dropna=False)[metric].agg(["min", "count"]).reset_index()
    else:
        agg = df.groupby(group_cols, dropna=False)[metric].agg(["mean", "count"]).reset_index()
    return agg.rename(columns={"min": "value", "mean": "value", "count": "n"})


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

    for line_value in sorted(df[line_col].dropna().unique(), key=str):
        line_df = df[df[line_col] == line_value].copy()

        if line_value not in swept_lines:
            agg = _aggregate_metric(line_df, [x_col], metric, reduce=reduce).sort_values(x_col)
            xs = agg[x_col].to_numpy()
            ys = agg["value"].to_numpy()
            (line,) = ax.plot(xs, ys, marker="o", linewidth=2.0, label=str(line_value))
            if is_proportion and (agg["n"] > 1).any():
                ci = agg.apply(
                    lambda r: pd.Series(_wilson_ci(r["value"], int(r["n"])), index=["lo", "hi"]),
                    axis=1,
                )
                agg = pd.concat([agg, ci], axis=1)
                ax.fill_between(
                    xs,
                    agg["lo"].to_numpy(),
                    agg["hi"].to_numpy(),
                    alpha=0.15,
                    color=line.get_color(),
                )
            continue

        line_df["_max_batch_size"] = pd.to_numeric(line_df["max_batch_size"], errors="coerce")
        line_df = line_df.dropna(subset=["_max_batch_size"])
        param_agg = _aggregate_metric(
            line_df, ["_max_batch_size", x_col], metric, reduce=reduce
        ).sort_values(["_max_batch_size", x_col])
        if param_agg.empty:
            continue

        color = None
        for batch_size, group in param_agg.groupby("_max_batch_size"):
            xs = group[x_col].to_numpy()
            ys = group["value"].to_numpy()
            (param_line,) = ax.plot(
                xs,
                ys,
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

        if is_proportion and (best["n"] > 1).any():
            ci = best.apply(
                lambda r: pd.Series(_wilson_ci(r["value"], int(r["n"])), index=["lo", "hi"]),
                axis=1,
            )
            best = pd.concat([best, ci], axis=1)
            ax.fill_between(
                best[x_col].to_numpy(),
                best["lo"].to_numpy(),
                best["hi"].to_numpy(),
                alpha=0.15,
                color=color,
            )

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


def plot_results(
    results: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    *,
    x: str = "num_robots",
    line: str = "scheduler",
    metrics: list[str] | None = None,
) -> list[pathlib.Path]:
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
    written += [
        _plot_metric(df, metric=metric, x_col=x, line_col=line, output_dir=output_dir, reduce="min")
        for metric in metrics
        if metric in STARVATION_METRICS
    ]
    print("Wrote plots:")
    for path in written:
        print(path)
    return written


def main() -> None:
    args = _parse_args()
    if args.results is None and not args.action_chunks:
        raise SystemExit("Provide --results, --action-chunks, or both")

    written: list[pathlib.Path] = []
    if args.results is not None:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        output_dir = args.output_dir or (args.results.parent / "plots")
        plot_results(
            args.results,
            output_dir,
            x=args.x,
            line=args.line,
            metrics=metrics,
        )
        written.extend(plot_action_fate_sweep(args.results, output_dir, x=args.x, line=args.line))

    if args.action_chunks:
        output_dir = args.output_dir
        if output_dir is None:
            first = args.action_chunks[0]
            output_dir = (first if first.is_dir() else first.parent) / "plots"
        written.extend(plot_action_fate(args.action_chunks, output_dir))

    if written:
        print("Wrote action fate outputs:")
        for path in written:
            print(path)


if __name__ == "__main__":
    main()
