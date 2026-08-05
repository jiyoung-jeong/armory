"""Reconstruct the fate of every produced action from saved action_chunks.parquet files."""

from __future__ import annotations

import json
import pathlib
from collections import deque

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ACTION_FATE_KEYS = ["executed", "lost_before_arrival", "overwritten", "cutoff_by_max_steps"]

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
ACTION_FATE_CASE_METADATA_COLS = [
    "scheduler",
    "server_variant",
    "experiment",
    "num_robots",
    "seed",
    "max_batch_size",
    "starvation_rate",
    "post_first_starvation_rate",
]
ACTION_FATE_CASE_SORT_COLS = [
    "scheduler",
    "max_batch_size",
    "server_variant",
    "experiment",
    "seed",
    "run_id",
]
ACTION_FATE_CASE_LABELS = {
    "experiment": "exp",
    "max_batch_size": "B",
    "num_robots": "robots",
    "seed": "seed",
    "server_variant": "variant",
}


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
    needed = {"action_index_start", "max_execution_horizon", "execution_start_step"}
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
    counts = {key: 0 for key in ACTION_FATE_KEYS}

    arrivals_by_step: dict[int, list[tuple[int, int]]] = {}
    total_produced = 0
    for row in chunks.itertuples(index=False):
        start = int(row.action_index_start)
        horizon = int(row.max_execution_horizon)
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


def _counts_from_action_chunk_files(files: list[pathlib.Path]) -> dict[str, int]:
    counts = {key: 0 for key in [*ACTION_FATE_KEYS, "produced"]}
    for chunk_file in files:
        episode_counts = _action_fate_counts_for_episode(chunk_file)
        for key, value in episode_counts.items():
            counts[key] += value
    return counts


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

    total_counts = _counts_from_action_chunk_files(files)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "action_fate_counts.csv"
    pd.DataFrame([total_counts]).to_csv(csv_path, index=False)
    return [_plot_action_fate_bar(total_counts, output_dir), csv_path]


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
    metadata_cols = []
    for col in [x_col, line_col, "run_id", *ACTION_FATE_CASE_METADATA_COLS]:
        if col in df.columns and col not in metadata_cols:
            metadata_cols.append(col)
    for _, row in df.iterrows():
        chunk_files = _artifact_action_chunk_files(pathlib.Path(str(row["artifact_path"])))
        if not chunk_files:
            continue
        rows.append(
            {
                **{col: row[col] for col in metadata_cols},
                **_counts_from_action_chunk_files(chunk_files),
            }
        )

    if not rows:
        return pd.DataFrame()

    fate = pd.DataFrame(rows)
    numeric_x = pd.to_numeric(fate[x_col], errors="coerce")
    if numeric_x.notna().all():
        fate[x_col] = numeric_x
    for col in ["max_batch_size", "seed", "starvation_rate"]:
        if col in fate.columns:
            fate[col] = pd.to_numeric(fate[col], errors="coerce")
    return fate


def _aggregate_action_fate_sweep(fate: pd.DataFrame, *, x_col: str, line_col: str) -> pd.DataFrame:
    return (
        fate.groupby([x_col, line_col], dropna=False)[[*ACTION_FATE_KEYS, "produced"]]
        .sum()
        .reset_index()
        .sort_values([x_col, line_col], key=lambda s: s.map(str) if s.dtype == object else s)
    )


def _sort_action_fate_cases(fate: pd.DataFrame, *, x_col: str, line_col: str) -> pd.DataFrame:
    order_cols = []
    for col in [x_col, line_col, *ACTION_FATE_CASE_SORT_COLS]:
        if col in fate.columns and col not in order_cols:
            order_cols.append(col)

    sorted_fate = fate.copy()
    sort_keys = []
    for col in order_cols:
        key_col = f"__sort_{col}"
        numeric = pd.to_numeric(sorted_fate[col], errors="coerce")
        sorted_fate[key_col] = numeric if numeric.notna().all() else sorted_fate[col].map(str)
        sort_keys.append(key_col)

    if sort_keys:
        sorted_fate = sorted_fate.sort_values(sort_keys, kind="stable").drop(columns=sort_keys)
    return sorted_fate.reset_index(drop=True)


def _format_case_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _action_fate_case_label_columns(fate: pd.DataFrame, *, x_col: str, line_col: str) -> list[str]:
    label_cols = []
    for col in [
        x_col,
        line_col,
        "scheduler",
        "max_batch_size",
        "server_variant",
        "experiment",
        "seed",
    ]:
        if col not in fate.columns or col in label_cols:
            continue
        if col == line_col or fate[col].nunique(dropna=False) > 1:
            label_cols.append(col)

    if not label_cols and "run_id" in fate.columns:
        label_cols.append("run_id")

    labels = fate.apply(lambda row: _format_action_fate_case_label(row, label_cols), axis=1)
    if labels.duplicated().any() and "run_id" in fate.columns and "run_id" not in label_cols:
        label_cols.append("run_id")
    return label_cols


def _format_action_fate_case_label(row: pd.Series, label_cols: list[str]) -> str:
    parts = []
    for col in label_cols:
        value = _format_case_value(row[col])
        if col in {"run_id", "scheduler"}:
            parts.append(value)
        else:
            parts.append(f"{ACTION_FATE_CASE_LABELS.get(col, col)}={value}")
    return "\n".join(parts)


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
    fate: pd.DataFrame,
    *,
    x_col: str,
    line_col: str,
    output_dir: pathlib.Path,
    normalize: bool,
    best_starvation: bool = False,
) -> pathlib.Path:
    fate = _sort_action_fate_cases(fate, x_col=x_col, line_col=line_col)
    label_cols = _action_fate_case_label_columns(fate, x_col=x_col, line_col=line_col)
    case_labels = [_format_action_fate_case_label(row, label_cols) for _, row in fate.iterrows()]
    bar_count = len(fate)
    fig_width = min(max(9.6, 0.34 * bar_count + 3.4), 30.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))

    line_values = sorted(fate[line_col].dropna().unique(), key=str)
    hatches = {
        line_value: ACTION_FATE_HATCHES[idx % len(ACTION_FATE_HATCHES)]
        for idx, line_value in enumerate(line_values)
    }
    xs = list(range(bar_count))
    bottoms = [0.0] * bar_count
    produced = fate["produced"].replace(0, pd.NA)

    for key in ACTION_FATE_KEYS:
        values = fate[key] / produced if normalize else fate[key]
        values = values.fillna(0.0).to_numpy(dtype=float)
        for idx, value in enumerate(values):
            line_value = fate.iloc[idx][line_col]
            ax.bar(
                xs[idx],
                value,
                bottom=bottoms[idx],
                width=0.74,
                color=ACTION_FATE_COLORS[key],
                edgecolor="#222222",
                linewidth=0.35,
                hatch=hatches.get(line_value, ""),
                label=ACTION_FATE_LABELS[key] if idx == 0 else "_nolegend_",
            )
            bottoms[idx] += value

    if x_col in fate.columns:
        previous_value = None
        for idx, value in enumerate(fate[x_col]):
            if idx > 0 and value != previous_value:
                ax.axvline(idx - 0.5, color="#777777", linewidth=0.5, alpha=0.5)
            previous_value = value

    ax.set_xticks(xs)
    ax.set_xticklabels(case_labels, rotation=90, ha="center", fontsize=7)
    label_bits = []
    for col in [x_col, *label_cols]:
        if col not in label_bits:
            label_bits.append(col)
    ax.set_xlabel("Cases grouped by " + ", ".join(col.replace("_", " ") for col in label_bits))
    ax.set_ylabel("Share of produced actions" if normalize else "Actions")
    title = "Action Fate by Run"
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
    output_path = output_dir / f"action_fate_{suffix}{best_suffix}_by_case.png"
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
        _plot_action_fate_sweep(
            fate, x_col=x, line_col=line, output_dir=output_dir, normalize=True
        ),
        _plot_action_fate_sweep(
            fate, x_col=x, line_col=line, output_dir=output_dir, normalize=False
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
                    best_fate,
                    x_col=x,
                    line_col=line,
                    output_dir=output_dir,
                    normalize=True,
                    best_starvation=True,
                ),
                _plot_action_fate_sweep(
                    best_fate,
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
