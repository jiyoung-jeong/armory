"""Metrics and plotting utilities for LIBERO experiments."""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from rich.console import Console
from rich.table import Table

from armory_client.schemas import ActionChunk, RuntimeMetadata, pathlib
from sims.libero.subscribers.saver import Result

logger = logging.getLogger(__name__)

# =============================================================================
# Data Loading
# =============================================================================


def _episode_step_timestamps(ep: dict) -> list[float]:
    ts = ep.get("step_timestamps") or []
    if ts:
        return [float(t) for t in ts]

    requests = ep.get("requests") or []
    return [float(req.get("request_timestamp")) for req in requests if req.get("request_timestamp")]


# =============================================================================
# Data Loading
# =============================================================================


def load_episodes(output_path: pathlib.Path) -> pd.DataFrame:
    """Load all metadata.json files into a DataFrame."""
    metadata_files = list(output_path.glob("**/metadata.json"))
    if not metadata_files:
        return pd.DataFrame()

    results: list[Result] = [Result.from_json(f) for f in metadata_files]
    return pd.DataFrame([asdict(result) for result in results])


def load_actions_left(
    output_path: pathlib.Path,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load actions_left.npy files grouped by robot_idx, with per-step timestamps.

    Returns:
        ``{robot_idx_str: [(timestamps, episode_array), ...]}`` sorted by episode
        order. ``timestamps`` is a 1-D float array, one wall-clock value per step
        from ``timestamps.csv`` (matching ``len(episode_array)``). When the
        timestamps file is missing or shorter than the array, falls back to a
        synthetic series starting at 0 with 1-step spacing — callers that align
        episodes onto a wall-clock grid will produce a degenerate but
        non-crashing result in that case.
    """
    files = sorted(output_path.glob("**/actions_left.npy"))
    by_robot: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for f in files:
        # path: <out_dir>/<robot_idx>/<ep_idx>_<suite>_<task>_<result>/actions_left.npy
        parts = f.parts
        robot_idx = parts[-3]  # e.g. "0"
        ep_prefix = parts[-2]  # e.g. "0_libero_10_0_success"
        ep_idx = int(ep_prefix.split("_")[0])
        arr = np.load(f)
        ts_file = f.parent / "timestamps.csv"
        if ts_file.exists():
            ts = pd.read_csv(ts_file)["timestamp"].to_numpy(dtype=float)
            if len(ts) < len(arr):
                # Pad with linear extrapolation at the trailing cadence so we
                # never index past the array.
                if len(ts) >= 2:
                    dt = float(np.median(np.diff(ts)))
                else:
                    dt = 0.0
                pad = ts[-1] + dt * np.arange(1, len(arr) - len(ts) + 1)
                ts = np.concatenate([ts, pad])
            elif len(ts) > len(arr):
                ts = ts[: len(arr)]
        else:
            ts = np.arange(len(arr), dtype=float)
        by_robot.setdefault(robot_idx, []).append((ep_idx, ts, arr))

    return {
        robot: [(ts, arr) for _, ts, arr in sorted(eps, key=lambda x: x[0])]
        for robot, eps in sorted(by_robot.items(), key=lambda kv: int(kv[0]))
    }


def _load_control_hz(output_path: pathlib.Path, fallback: float = 20.0) -> float:
    """Load control frequency from runtime metadata, falling back when absent."""
    runtime_metadata_path = output_path / "runtime_metadata.json"
    if runtime_metadata_path.exists():
        control_hz = RuntimeMetadata.from_json(runtime_metadata_path).control_hz
        if control_hz is not None and float(control_hz) > 0:
            return float(control_hz)
    return float(fallback)


def _build_actions_left_matrix(
    output_path: pathlib.Path,
    control_hz: float | None = None,
) -> tuple[list[str], np.ndarray, list[list[int]], float, float]:
    """Align per-episode actions_left traces onto a shared wall-clock grid.

    Returns ``(robots, matrix, episode_boundaries, control_hz, t0)`` where
    ``t0`` is the earliest perf_counter timestamp used as the column-0 origin.
    """
    by_robot = load_actions_left(output_path)
    if not by_robot:
        return (
            [],
            np.empty((0, 0), dtype=float),
            [],
            float(control_hz or _load_control_hz(output_path)),
            0.0,
        )

    robots = sorted(by_robot.keys(), key=int, reverse=True)

    # Canvas rate: caller-supplied wins; otherwise derive from data so the
    # heatmap respects heterogeneous control rates. Using the max observed
    # rate keeps the ratio of cells-per-step correct (e.g. with a 30 Hz
    # robot and a 10 Hz robot, the slow robot's bars should be 3x wider
    # than the fast robot's). Falling back to runtime_metadata.control_hz
    # when there's no per-step timing.
    if control_hz is not None:
        resolved_control_hz = float(control_hz)
    else:
        per_robot_rates = []
        for eps in by_robot.values():
            for ts, _ in eps:
                if len(ts) >= 2:
                    median_gap = float(np.median(np.diff(ts)))
                    if median_gap > 0:
                        per_robot_rates.append(1.0 / median_gap)
        resolved_control_hz = (
            max(per_robot_rates) if per_robot_rates else _load_control_hz(output_path)
        )

    # Global t0: earliest first-step timestamp across all robots.
    t0 = min(ts[0] for eps in by_robot.values() for ts, _ in eps if len(ts) > 0)

    # Place each step at its actual wall-clock column instead of stacking
    # consecutive steps in consecutive columns. With heterogeneous control
    # rates (e.g. WS-14 at 30 Hz, others at 10 Hz), the old behavior made
    # faster robots' rows extend past the trial's wall-clock end on the time
    # axis. Now every robot terminates at the column matching its real last
    # timestamp, regardless of how many steps it took.
    max_col = 0
    placements: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for robot in robots:
        per_episode: list[tuple[np.ndarray, np.ndarray]] = []
        for ts, arr in by_robot[robot]:
            cols = np.round((ts - t0) * resolved_control_hz).astype(int)
            np.clip(cols, 0, None, out=cols)
            per_episode.append((cols, arr))
            if cols.size:
                max_col = max(max_col, int(cols.max()))
        placements.append(per_episode)

    matrix = np.full((len(robots), max_col + 1), np.nan, dtype=float)
    episode_boundaries: list[list[int]] = []
    for i, per_episode in enumerate(placements):
        boundaries = []
        for cols, arr in per_episode:
            if not cols.size:
                continue
            # Forward-fill each step's queue depth until the next step at
            # this robot. Without this, robots whose actual rate is below
            # the canvas rate (e.g. 10 Hz on a 20 Hz canvas) leave NaN gaps
            # at every other column, which imshow renders as transparent
            # bars. Forward-fill is the right semantics: the queue depth
            # observed at step t is the best estimate of depth at any
            # wall-clock moment between t and the next step.
            for j in range(len(cols)):
                start = int(cols[j])
                end = int(cols[j + 1]) if j + 1 < len(cols) else start + 1
                if end <= start:
                    end = start + 1
                matrix[i, start:end] = arr[j]
            boundaries.append(int(cols[0]))
        episode_boundaries.append(boundaries)

    return robots, matrix, episode_boundaries, resolved_control_hz, t0


def load_action_chunks(output_path: pathlib.Path) -> pd.DataFrame:
    """Load all action_chunks.parquet files with task metadata."""
    action_chunk_files = list(output_path.glob("**/action_chunks.parquet"))

    rows = []
    for action_chunk_file in action_chunk_files:
        episode_dir = action_chunk_file.parent
        metadata_file = episode_dir / "metadata.json"

        if not metadata_file.exists():
            print(f"Warning: metadata.json not found in {episode_dir}, skipping")
            continue

        result = Result.from_json(metadata_file)
        chunks = ActionChunk.from_parquet(action_chunk_file)

        for chunk in chunks:
            rows.append(
                {
                    "task_suite_name": result.task_suite_name,
                    "task_id": result.task_id,
                    "task_language": result.task_language,
                    "latency": chunk.latency,
                    "max_execution_horizon": chunk.max_execution_horizon,
                }
            )

    return pd.DataFrame(rows)


def _jains_index(values: list[float]) -> float:
    """Jain's fairness index: (Σx)^2 / (n · Σx^2). 1.0 if all values equal, 1/n at worst."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 1.0
    sq = float(np.sum(arr * arr))
    if sq <= 0:
        return 1.0
    return float(np.sum(arr) ** 2 / (arr.size * sq))


def compute_fairness_metrics(output_path: pathlib.Path) -> dict | None:
    """Per-robot starvation rate and Jain's index on the freshness rate (1 - starvation_rate).

    Starvation rate is the fraction of control steps a robot had no fresh action to execute,
    so 1 - starvation_rate is the per-robot quality of service. Jain's on freshness measures
    whether all robots received fresh actions at equal rates.

    Returns None when no per-episode metadata is available.
    """
    df = load_episodes(output_path)
    if df.empty:
        return None
    psd = load_planner_starvation_metrics(output_path)
    if psd.empty:
        return None
    df = df.merge(
        psd,
        on=["robot_idx", "episode_idx", "task_suite_name", "task_id"],
        how="left",
    )
    if "starvation_steps" not in df.columns or "observed_steps" not in df.columns:
        return None

    agg = (
        df.groupby("robot_idx")
        .agg(
            starvation_steps=("starvation_steps", "sum"),
            observed_steps=("observed_steps", "sum"),
        )
        .reset_index()
        .sort_values("robot_idx")
    )
    if agg.empty:
        return None

    robot_idx = [int(i) for i in agg["robot_idx"].tolist()]
    starvation_rate = [
        float(s) / float(o) if o > 0 else 0.0
        for s, o in zip(agg["starvation_steps"], agg["observed_steps"])
    ]
    freshness_rate = [1.0 - r for r in starvation_rate]

    alpha = None
    server_path = output_path / "server_metadata.json"
    if server_path.exists():
        try:
            with open(server_path) as f:
                kwargs = (json.load(f) or {}).get("scheduler_kwargs") or {}
            alpha = kwargs.get("alpha")
        except (json.JSONDecodeError, OSError):
            alpha = None

    return {
        "alpha": alpha,
        "robot_idx": robot_idx,
        "starvation_rate": starvation_rate,
        "freshness_rate": freshness_rate,
        "jain_freshness": _jains_index(freshness_rate),
        "jain_starvation": _jains_index(starvation_rate),
    }


def load_experiment_duration(output_path: pathlib.Path) -> float | None:
    """Compute total experiment wall-clock duration from timestamps.csv files.

    Returns the span from the earliest first-step timestamp to the latest
    last-step timestamp across all episodes, or None if no timestamps exist.
    """
    ts_files = list(output_path.glob("**/timestamps.csv"))
    if not ts_files:
        return None

    t_min = float("inf")
    t_max = float("-inf")
    for f in ts_files:
        df = pd.read_csv(f, usecols=["timestamp"])
        if df.empty:
            continue
        t_min = min(t_min, float(df["timestamp"].iloc[0]))
        t_max = max(t_max, float(df["timestamp"].iloc[-1]))

    if t_min == float("inf"):
        return None
    return t_max - t_min


def _server_batch_fields(
    batch,
) -> tuple[object, list[str], list[int], float | None, float | None, int]:
    """Parse old 5-field and new 6-field server batch records."""
    if isinstance(batch, dict):
        robot_ids = batch.get("robot_ids") or []
        request_ids = batch.get("request_ids") or []
        batch_size = int(batch.get("batch_size") or len(robot_ids))
        return (
            batch.get("batch_id"),
            robot_ids,
            request_ids,
            batch.get("inference_start_time"),
            batch.get("inference_end_time"),
            batch_size,
        )

    batch_id, robot_ids, request_ids, start, end = batch[:5]
    batch_size = int(batch[5]) if len(batch) >= 6 and batch[5] is not None else len(robot_ids)
    return batch_id, robot_ids, request_ids, start, end, batch_size


def _load_scheduler_decisions(output_path: pathlib.Path) -> list[dict]:
    """Return raw scheduler-decision records from server_metrics_history.json.

    Each record carries `started_at` (wall clock), `duration` (s), `candidates`,
    `scheduled`, `batch_id`, `notes`, etc. Older histories that pre-date the
    schema return an empty list, since old fields are normalized by the
    SchedulerDecision loader on the server side.
    """
    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        return []
    data = json.loads(history_path.read_text())
    return list(data.get("scheduler_decisions") or [])


def _server_clock_t0(output_path: pathlib.Path) -> float | None:
    """Return server-side t0 (time.time()) used as origin for server timeline plots."""
    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        return None
    data = json.loads(history_path.read_text())
    t0 = data.get("start_time")
    return float(t0) if t0 is not None and t0 != float("inf") else None


def _server_to_perf_offset(output_path: pathlib.Path) -> float | None:
    """Return ``time.time() - time.perf_counter()`` offset, derived from saved data.

    server_metrics_history.json holds time.time() values; timestamps.csv holds
    time.perf_counter() values. Within the libero sim driver these clocks live
    in the same process so the offset is approximately constant. We match the
    earliest first-request timestamp on the server side with the earliest first
    perf_counter from timestamps.csv to estimate it.
    """
    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        return None
    data = json.loads(history_path.read_text())
    earliest_request_time: float | None = None
    for robot in data.get("robots", {}).values():
        for ep in robot.get("episodes", []):
            for req in ep.get("requests", []):
                ts = req.get("request_timestamp")
                if ts is None:
                    continue
                ts = float(ts)
                if earliest_request_time is None or ts < earliest_request_time:
                    earliest_request_time = ts
    if earliest_request_time is None:
        return None

    earliest_perf: float | None = None
    for ts_file in output_path.glob("**/timestamps.csv"):
        df = pd.read_csv(ts_file, usecols=["timestamp"], nrows=1)
        if df.empty:
            continue
        first = float(df["timestamp"].iloc[0])
        if earliest_perf is None or first < earliest_perf:
            earliest_perf = first
    if earliest_perf is None:
        return None
    return earliest_request_time - earliest_perf


def load_planner_starvation_metrics(output_path: pathlib.Path) -> pd.DataFrame:
    """Load per-episode no-action metrics from saved cost histories.

    Uses obs cost: A NaN in cost_history means the runtime executed a null action for that
    control step.
    """
    runtime_metadata_path = output_path / "runtime_metadata.json"
    assert runtime_metadata_path.exists()
    control_hz = RuntimeMetadata.from_json(runtime_metadata_path).control_hz

    rows = []
    for cost_history_file in sorted(output_path.glob("**/cost_history.npy")):
        episode_dir = cost_history_file.parent
        metadata_file = episode_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"Warning: metadata.json not found in {episode_dir}, skipping")
            continue

        result = Result.from_json(metadata_file)
        costs = np.load(cost_history_file)
        nan_mask = np.isnan(costs)
        starvation_steps = int(nan_mask.sum())
        total_steps = int(costs.shape[0])
        assert total_steps > 0, "cost_history should contain at least one step"
        assert control_hz is not None and control_hz > 0

        # Starvation excluding leading NaNs (before the robot's first action).
        non_nan_idx = np.flatnonzero(~nan_mask)
        if non_nan_idx.size > 0:
            first = int(non_nan_idx[0])
            post_first_observed = total_steps - first
            post_first_starvation = int(nan_mask[first:].sum())
        else:
            post_first_observed = 0
            post_first_starvation = 0

        row = {
            "robot_idx": result.robot_idx,
            "episode_idx": result.episode_idx,
            "task_suite_name": result.task_suite_name,
            "task_id": result.task_id,
            "starvation_steps": starvation_steps,
            "observed_steps": total_steps,
            "planner_starvation_seconds": starvation_steps / control_hz,
            "post_first_starvation_steps": post_first_starvation,
            "post_first_observed_steps": post_first_observed,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Plot Primitives
# =============================================================================


def plot_histogram(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    xlabel: str,
    color: str = "steelblue",
    show_stats: bool = True,
) -> None:
    """Plot histogram with optional percentile markers."""
    ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="black")

    if show_stats:
        stats = {
            "mean": np.mean(data),
            "median": np.median(data),
            "p90": np.percentile(data, 90),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99),
        }
        ax.axvline(
            stats["mean"],
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {stats['mean']:.3f}",
        )
        ax.axvline(
            stats["median"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {stats['median']:.3f}",
        )
        ax.axvline(
            stats["p90"],
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"P90: {stats['p90']:.3f}",
        )
        ax.axvline(
            stats["p95"],
            color="purple",
            linestyle="-.",
            linewidth=2,
            label=f"P95: {stats['p95']:.3f}",
        )
        ax.axvline(
            stats["p99"],
            color="brown",
            linestyle="-",
            linewidth=1,
            label=f"P99: {stats['p99']:.3f}",
        )
        ax.legend(fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def plot_bar_chart(
    ax: plt.Axes,
    labels: list[str],
    values: np.ndarray,
    ylabel: str = "Value",
    title: str = "",
    counts: np.ndarray | None = None,
    overall_line: tuple[float, str] | None = None,
    color_fn: Callable[[float], str] | None = None,
) -> None:
    """Plot bar chart with optional annotations.

    Args:
        ax: Matplotlib axes
        labels: Bar labels
        values: Bar values
        ylabel: Y-axis label
        title: Plot title
        counts: Optional counts to show on bars as (n=X)
        overall_line: Optional (value, label) for horizontal line
        color_fn: Optional function(value) -> color for conditional coloring
    """
    bars = ax.bar(range(len(values)), values, color="steelblue", edgecolor="black", alpha=0.7)

    if color_fn:
        for bar, val in zip(bars, values):
            bar.set_color(color_fn(val))

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if overall_line:
        value, label = overall_line
        ax.axhline(y=value, color="red", linestyle="--", linewidth=2, label=label)
        ax.legend()

    if counts is not None:
        for bar, val, count in zip(bars, values, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.02,
                f"{val:.1%}\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def plot_grouped_violin(
    ax: plt.Axes,
    groups: dict[str, dict[str, np.ndarray]],
    ylabel: str = "Value",
    title: str = "",
    group_colors: dict[str, str] | None = None,
) -> None:
    """Plot violin plots comparing multiple groups per category.

    Args:
        ax: Matplotlib axes
        groups: {category: {group_name: values}}
                e.g. {"Task 0": {"success": [...], "failure": [...]}}
        ylabel: Y-axis label
        title: Plot title
        group_colors: Optional {group_name: color} mapping
    """
    if group_colors is None:
        group_colors = {"success": "lightgreen", "failure": "lightcoral"}

    # Collect all group names for consistent ordering
    all_group_names = set()
    for category_groups in groups.values():
        all_group_names.update(category_groups.keys())
    group_names = sorted(all_group_names)

    positions = []
    labels = []
    all_data = []
    all_colors = []

    for i, (category, category_groups) in enumerate(groups.items()):
        base_pos = i * (len(group_names) + 1)
        for j, group_name in enumerate(group_names):
            if group_name in category_groups and len(category_groups[group_name]) > 0:
                all_data.append(category_groups[group_name])
                all_colors.append(group_colors.get(group_name, "lightblue"))
                positions.append(base_pos + j)
                labels.append(f"{category}\n({group_name})")

    if not all_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    # Plot violins
    parts = ax.violinplot(
        all_data, positions=positions, widths=0.8, showmeans=True, showmedians=True
    )

    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(all_colors[i])
        pc.set_alpha(0.7)

    # Style the lines
    for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if partname in parts:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(1)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_elements = [
        Patch(facecolor=group_colors.get(name, "lightblue"), edgecolor="black", label=name)
        for name in group_names
        if any(name in g for g in groups.values())
    ]
    if legend_elements:
        ax.legend(handles=legend_elements)


# =============================================================================
# Layout Helper
# =============================================================================


def plot_task_breakdown(
    df: pd.DataFrame,
    column: str,
    plot_fn: Callable[[plt.Axes, np.ndarray, str], None],
    title: str,
    filename: pathlib.Path,
    title_pad: float | None = None,
) -> None:
    """Create grid: 'All Tasks' in first cell, then one cell per task.

    Args:
        df: DataFrame with 'task_suite_name' and 'task_id' columns
        column: Which column to extract values from
        plot_fn: Function(ax, data, subtitle) that plots on a single axes
        title: Overall figure title
        filename: Where to save
        title_pad: Optional extra padding (points) between suptitle and subplots
    """
    if df.empty:
        logger.warning(f"No data for {column}")
        return

    # Create task labels and group
    df = df.copy()
    if "task_language" in df.columns:
        df["task_label"] = "Task " + df["task_id"].astype(str) + "\n" + df["task_language"].str[:30]
    else:
        df["task_label"] = df["task_suite_name"] + " - Task " + df["task_id"].astype(str)
    grouped = df.groupby(["task_id", "task_label"], sort=True)

    n_tasks = len(grouped)
    n_plots = 1 + n_tasks  # overall + per-task

    # Grid dimensions
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    # Normalize axes to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0 if title_pad else 0.98)
    if title_pad is not None:
        fig.subplots_adjust(top=0.88)

    # Plot overall
    plot_fn(axes.flat[0], df[column].values, "All Tasks Combined")

    # Plot per task
    for idx, ((task_id, task_label), group) in enumerate(grouped, start=1):
        plot_fn(axes.flat[idx], group[column].values, task_label)

    # Hide unused subplots
    for idx in range(n_plots, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    logger.info(f"Saved {filename}")


# =============================================================================
# Plot Generators
# =============================================================================


def generate_latency_plot(output_path: pathlib.Path) -> None:
    """Latency distribution: overall + per-task (in milliseconds)."""
    df = load_action_chunks(output_path)
    if not df.empty:
        df = df.copy()
        df["latency_ms"] = df["latency"] * 1000
    plot_task_breakdown(
        df,
        column="latency_ms",
        plot_fn=lambda ax, data, title: plot_histogram(ax, data, title, "Latency (ms)"),
        title="Action Chunk Latency Distribution",
        filename=output_path / "plots" / "action_chunk_latency.png",
        title_pad=20,
    )


def generate_success_rate_plot(output_path: pathlib.Path) -> None:
    """Success rate bar chart by task."""
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No episode data for success rate plot")
        return

    # Aggregate by task
    summary = (
        df.groupby(["task_suite_name", "task_id", "task_language"])["success"]
        .agg(["mean", "count"])
        .reset_index()
    )
    summary["task_label"] = (
        "Task " + summary["task_id"].astype(str) + "\n" + summary["task_language"].str[:30]
    )

    overall_rate = df["success"].mean()

    def success_color(rate: float) -> str:
        if rate >= 0.8:
            return "green"
        elif rate >= 0.5:
            return "orange"
        return "red"

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(top=0.88)
    plot_bar_chart(
        ax,
        labels=summary["task_label"].tolist(),
        values=summary["mean"].values,
        ylabel="Success Rate",
        title="Success Rate by Task",
        counts=summary["count"].values,
        overall_line=(overall_rate, f"Overall: {overall_rate:.2%}"),
        color_fn=success_color,
    )
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "success_rate.png", dpi=150)
    plt.close(fig)

    logger.info(f"Saved {plots_dir / 'success_rate.png'}")


def generate_steps_plot(output_path: pathlib.Path) -> None:
    """Steps analysis: successful episodes histogram + per-task violin plot."""
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No episode data for steps plot")
        return

    fig = plt.figure(figsize=(16, 10), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.3)
    fig.suptitle("Steps Taken Analysis", fontsize=16, fontweight="bold")

    # Overall distribution for successful episodes only
    success_steps = df[df["success"]]["steps_taken"].values

    ax_success = fig.add_subplot(gs[0])
    if len(success_steps) > 0:
        plot_histogram(
            ax_success,
            success_steps,
            "Successful Episodes",
            "Steps",
            color="green",
            show_stats=False,
        )
    else:
        ax_success.text(
            0.5,
            0.5,
            "No successful episodes",
            ha="center",
            va="center",
            transform=ax_success.transAxes,
        )
        ax_success.set_title("Successful Episodes")

    # Per-task violin plot (success only)
    df["task_label"] = "Task " + df["task_id"].astype(str) + "\n" + df["task_language"].str[:30]
    groups = {}
    for (task_id, task_label), group in df.groupby(["task_id", "task_label"], sort=True):
        groups[task_label] = {
            "success": group[group["success"]]["steps_taken"].values,
        }

    ax_violin = fig.add_subplot(gs[1])
    plot_grouped_violin(
        ax_violin,
        groups,
        ylabel="Steps",
        title="Steps by Task (Successful Episodes)",
        group_colors={"success": "lightgreen"},
    )
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "steps_taken.png", dpi=150)
    plt.close(fig)

    logger.info(f"Saved {plots_dir / 'steps_taken.png'}")


def generate_actions_left_heatmap(
    output_path: pathlib.Path, control_hz: float | None = None
) -> None:
    """Heatmap of actions_left[step, robot] using ground-truth queue lengths.

    Episodes are positioned on the time axis using their start timestamps, so
    inter-episode gaps appear as NaN columns (mirroring the schemas.py approach
    of using request_timestamp to place each step at its real wall-clock position).
    Episode boundaries are marked with vertical lines.
    """
    robots, matrix, episode_boundaries, control_hz, t0_perf = _build_actions_left_matrix(
        output_path, control_hz
    )
    if matrix.size == 0:
        logger.warning("No actions_left.npy data found")
        return

    n_robots = len(robots)
    max_len = matrix.shape[1]

    fig_width = min(
        400, max(12, max_len / max(control_hz, 1.0))
    )  # cap at 400 inches (~60k px at 150 dpi)
    fig, ax = plt.subplots(figsize=(fig_width, max(4, n_robots * 0.6)))

    vmax = max(1, int(np.nanmax(matrix))) if not np.all(np.isnan(matrix)) else 1
    # Black for 0 (starvation), then RdYlGn for 1..vmax
    rdylgn = matplotlib.colormaps["RdYlGn"].resampled(vmax)
    cmap_colors = [(0.0, 0.0, 0.0, 1.0)] + [rdylgn(i) for i in range(vmax)]
    cmap = mcolors.ListedColormap(cmap_colors)

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=vmax,
    )

    # Episode boundary markers (thin white lines)
    for i, bounds in enumerate(episode_boundaries):
        for b in bounds[1:]:  # skip first episode
            ax.plot(
                [b - 0.5, b - 0.5],
                [i - 0.4, i + 0.4],
                color="white",
                linewidth=0.8,
                alpha=0.7,
            )

    # Overlay scheduler decisions: green tick on the row of each scheduled robot
    # for batched decisions, faint cyan tick along the top for skipped decisions.
    decisions_overlaid = 0
    decisions = _load_scheduler_decisions(output_path)
    offset = _server_to_perf_offset(output_path) if decisions else None
    if decisions and offset is not None:
        robot_to_row = {rid: i for i, rid in enumerate(robots)}
        for d in decisions:
            started_at = d.get("started_at")
            if started_at is None:
                continue
            col = (float(started_at) - offset - t0_perf) * control_hz
            if col < -0.5 or col > max_len - 0.5:
                continue
            scheduled = d.get("scheduled") or []
            if scheduled and d.get("batch_id") is not None:
                for rid in scheduled:
                    row = robot_to_row.get(str(rid))
                    if row is None:
                        continue
                    ax.plot(
                        [col, col],
                        [row - 0.42, row + 0.42],
                        color="lime",
                        linewidth=0.7,
                        alpha=0.8,
                    )
                decisions_overlaid += 1
            else:
                # Skipped decisions get a faint top-edge tick — quick visual of
                # how often the scheduler woke up without dispatching.
                ax.plot(
                    [col, col],
                    [n_robots - 0.5, n_robots - 0.4],
                    color="deepskyblue",
                    linewidth=0.5,
                    alpha=0.6,
                )

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Actions left in queue", fontweight="bold")

    ax.set_yticks(range(n_robots))
    ax.set_yticklabels([f"robot_{r}" for r in robots], fontsize=8)
    tick_interval = max(1, int(round(control_hz)))  # one tick per second
    x_ticks = np.arange(0, max_len, tick_interval)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t // tick_interval}s" for t in x_ticks], fontsize=6)
    decision_legend = (
        " | green ticks = scheduler decision dispatch" if decisions_overlaid > 0 else ""
    )
    ax.set_xlabel(
        f"Wall-clock time in seconds (white lines = episode boundaries{decision_legend})",
        fontweight="bold",
    )
    ax.set_ylabel("Robot", fontweight="bold")
    ax.set_title("Actions Left Per Robot Over Time", fontsize=14, fontweight="bold")

    fig.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "actions_left_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {plots_dir / 'actions_left_heatmap.png'}")


def generate_per_robot_success_rate_plot(output_path: pathlib.Path) -> None:
    """Success rate bar chart broken down by robot."""
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No episode data for per-robot success rate plot")
        return

    robot_summary = df.groupby("robot_idx")["success"].agg(["mean", "count"]).reset_index()
    robot_summary = robot_summary.sort_values("robot_idx")

    overall_rate = df["success"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        robot_summary["robot_idx"].astype(str),
        robot_summary["mean"],
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.axhline(
        y=overall_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall: {overall_rate:.2%}",
    )

    for bar, rate, count in zip(bars, robot_summary["mean"], robot_summary["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{rate:.0%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Robot Index", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Per-Robot Success Rate", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "per_robot_success_rate.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved {plots_dir / 'per_robot_success_rate.png'}")


def _load_robot_starvation_rates(output_path: pathlib.Path) -> pd.DataFrame:
    """Aggregate per-robot starvation rates from saved episode metrics."""
    starvation_df = load_planner_starvation_metrics(output_path)
    if starvation_df.empty:
        return pd.DataFrame()

    robot_starvation = (
        starvation_df.groupby("robot_idx")[["starvation_steps", "observed_steps"]]
        .sum()
        .reset_index()
        .sort_values("robot_idx")
    )
    robot_starvation["starvation_rate"] = (
        robot_starvation["starvation_steps"] / robot_starvation["observed_steps"]
    )
    return robot_starvation


def generate_starvation_plot(output_path: pathlib.Path) -> None:
    """Per-robot starvation rate bar chart."""
    robot_starvation = _load_robot_starvation_rates(output_path)
    if robot_starvation.empty:
        logger.warning("No starvation data found")
        return

    rates = robot_starvation["starvation_rate"].values
    robot_labels = robot_starvation["robot_idx"].astype(str).tolist()
    n_robots = len(robot_labels)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_robots), 5))
    bars = ax.bar(robot_labels, rates, color="tomato", edgecolor="black", alpha=0.8)
    overall_rate = (
        robot_starvation["starvation_steps"].sum() / robot_starvation["observed_steps"].sum()
    )
    ax.axhline(
        overall_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall: {overall_rate:.2%}",
    )
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("Robot Index", fontsize=12)
    ax.set_ylabel("Starvation Rate", fontsize=12)
    ax.set_title("Per-Robot Starvation Rate", fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(rates) * 1.3 + 0.05))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "starvation_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {plots_dir / 'starvation_rate.png'}")


def generate_starvation_tail_metrics_plot(output_path: pathlib.Path) -> None:
    """Bar chart of tail starvation metrics across robots."""
    robot_starvation = _load_robot_starvation_rates(output_path)
    if robot_starvation.empty:
        logger.warning("No starvation data found for tail metrics plot")
        return

    rates = robot_starvation["starvation_rate"].to_numpy(dtype=float)
    max_starvation = float(np.max(rates))
    var90 = float(np.percentile(rates, 90))
    cvar90 = float(np.mean(rates[rates >= var90]))

    metric_names = ["Max Starvation", "CVaR_90 Starvation"]
    metric_values = [max_starvation, cvar90]
    colors = ["firebrick", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        metric_names,
        metric_values,
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Starvation Rate", fontsize=12)
    ax.set_title(
        "Starvation Tail Metrics Across Robots",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, min(1.0, max(metric_values) * 1.25 + 0.05))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plots_dir / "starvation_tail_metrics.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"Saved {plots_dir / 'starvation_tail_metrics.png'}")


def compute_starvation_variance_series(
    output_path: pathlib.Path, control_hz: float | None = None
) -> dict | None:
    """Cross-robot variance of cumulative starvation rate over wall-clock time.

    Starvation here matches the ``actions_left <= 0`` definition used by the
    starvation_variance_over_time plot (queue depth at the control step),
    aligned onto a shared wall-clock canvas via ``_build_actions_left_matrix``.

    Returns ``None`` when no per-robot actions_left data is available.
    """
    robots, matrix, _, control_hz, _ = _build_actions_left_matrix(output_path, control_hz)
    if matrix.size == 0:
        return None

    valid_mask = ~np.isnan(matrix)
    starved_mask = valid_mask & (matrix <= 0)
    cumulative_observed = np.cumsum(valid_mask, axis=1)
    cumulative_starved = np.cumsum(starved_mask, axis=1)
    cumulative_rates = np.divide(
        cumulative_starved,
        cumulative_observed,
        out=np.full(matrix.shape, np.nan, dtype=float),
        where=cumulative_observed > 0,
    )
    starvation_variance = np.nanvar(cumulative_rates, axis=0)
    time_seconds = np.arange(matrix.shape[1], dtype=float) / max(control_hz, 1.0)
    final_variance = float(starvation_variance[-1]) if starvation_variance.size else float("nan")
    return {
        "robots": robots,
        "cumulative_rates": cumulative_rates,
        "cumulative_starved": cumulative_starved,
        "cumulative_observed": cumulative_observed,
        "starvation_variance": starvation_variance,
        "time_seconds": time_seconds,
        "control_hz": float(control_hz),
        "final_starvation_variance": final_variance,
    }


def generate_starvation_variance_plot(
    output_path: pathlib.Path, control_hz: float | None = None
) -> None:
    """Plot cumulative starvation rate per robot and its cross-robot variance."""
    series = compute_starvation_variance_series(output_path, control_hz)
    if series is None:
        logger.warning("No actions_left.npy data found for starvation variance plot")
        return
    robots = series["robots"]
    cumulative_rates = series["cumulative_rates"]
    starvation_variance = series["starvation_variance"]
    time_seconds = series["time_seconds"]

    fig, (ax_rates, ax_var) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )
    fig.suptitle(
        "Starvation Fairness Across Robots Over Time",
        fontsize=14,
        fontweight="bold",
    )

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(robots), 2)))
    plot_order = np.argsort([int(robot) for robot in robots])
    for color_idx, row_idx in enumerate(plot_order):
        robot = robots[row_idx]
        ax_rates.plot(
            time_seconds,
            cumulative_rates[row_idx],
            linewidth=1.5,
            color=colors[color_idx % len(colors)],
            label=f"robot_{robot}",
        )

    ax_rates.set_ylabel("Cumulative starvation rate", fontsize=12)
    ax_rates.set_ylim(0, 1)
    ax_rates.grid(True, alpha=0.3)
    ax_rates.legend(
        loc="upper right",
        ncol=min(max(1, len(robots)), 5),
        fontsize=8,
        frameon=False,
    )

    ax_var.plot(
        time_seconds,
        starvation_variance,
        color="darkred",
        linewidth=2,
    )
    ax_var.fill_between(
        time_seconds,
        starvation_variance,
        color="tomato",
        alpha=0.2,
    )
    ax_var.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax_var.set_ylabel("Variance", fontsize=12)
    ax_var.set_ylim(bottom=0)
    ax_var.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plots_dir / "starvation_variance_over_time.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"Saved {plots_dir / 'starvation_variance_over_time.png'}")


def generate_per_robot_starvation_rate_gif(
    output_path: pathlib.Path,
    control_hz: float | None = None,
    fps: int = 15,
    max_frames: int = 150,
) -> None:
    """Animate per-robot cumulative starvation rate as stacked line + bar chart.

    Top panel: per-robot lines revealed over wall-clock time (same data as the
    top panel of starvation_variance_over_time.png).
    Bottom panel: per-robot bar chart whose heights track each robot's
    cumulative starvation rate at the current frame, with an overall
    weighted-average reference line that updates per frame.
    """
    series = compute_starvation_variance_series(output_path, control_hz)
    if series is None:
        logger.warning("No actions_left.npy data found for per-robot starvation GIF")
        return
    import matplotlib.animation as animation  # noqa: PLC0415

    robots = series["robots"]
    cumulative_rates = series["cumulative_rates"]
    cumulative_starved = series["cumulative_starved"]
    cumulative_observed = series["cumulative_observed"]
    time_seconds = series["time_seconds"]
    n_cols = cumulative_rates.shape[1]
    if n_cols < 2:
        logger.warning("Not enough timesteps for per-robot starvation GIF")
        return

    # Downsample frame indices: include t=0 and the final frame, evenly spaced.
    n_frames = min(max_frames, n_cols)
    frame_indices = np.unique(np.linspace(0, n_cols - 1, n_frames).astype(int))

    plot_order = np.argsort([int(robot) for robot in robots])
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(robots), 2)))
    ordered_robots = [robots[i] for i in plot_order]
    bar_labels = [str(r) for r in ordered_robots]

    fig, (ax_line, ax_bar) = plt.subplots(
        2, 1, figsize=(max(8, 1.2 * len(robots)), 8), gridspec_kw={"height_ratios": [2, 1.5]}
    )

    lines = []
    for color_idx, row_idx in enumerate(plot_order):
        robot = robots[row_idx]
        (line,) = ax_line.plot(
            [],
            [],
            linewidth=1.5,
            color=colors[color_idx % len(colors)],
            label=f"robot_{robot}",
        )
        lines.append((row_idx, line))
    ax_line.set_xlim(0, time_seconds[-1])
    ax_line.set_ylim(0, 1)
    ax_line.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax_line.set_ylabel("Cumulative starvation rate", fontsize=12)
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(
        loc="upper right",
        ncol=min(max(1, len(robots)), 5),
        fontsize=8,
        frameon=False,
    )
    title = ax_line.set_title("", fontsize=13, fontweight="bold")
    time_marker = ax_line.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)

    bar_colors = [colors[i % len(colors)] for i in range(len(plot_order))]
    bars = ax_bar.bar(
        bar_labels,
        np.zeros(len(plot_order)),
        color=bar_colors,
        edgecolor="black",
        alpha=0.85,
    )
    bar_texts = [
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            0.0,
            "0.0%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        for bar in bars
    ]
    overall_line = ax_bar.axhline(
        0.0, color="red", linestyle="--", linewidth=2, label="Overall: 0.0%"
    )
    overall_legend = ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.set_xlabel("Robot index", fontsize=12)
    ax_bar.set_ylabel("Cumulative starvation rate", fontsize=12)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.grid(axis="y", alpha=0.3)

    def update(frame_col: int):
        upto = frame_col + 1
        for row_idx, line in lines:
            line.set_data(time_seconds[:upto], cumulative_rates[row_idx, :upto])
        t_now = time_seconds[frame_col]
        time_marker.set_xdata([t_now, t_now])
        title.set_text(f"Per-robot cumulative starvation rate — t = {t_now:.1f}s")

        col_rates = cumulative_rates[:, frame_col]
        ordered_rates = col_rates[plot_order]
        for bar, text, rate in zip(bars, bar_texts, ordered_rates):
            height = float(rate) if np.isfinite(rate) else 0.0
            bar.set_height(height)
            text.set_y(height + 0.01)
            text.set_text("--" if not np.isfinite(rate) else f"{height * 100:.1f}%")

        total_starved = float(np.nansum(cumulative_starved[:, frame_col]))
        total_observed = float(np.nansum(cumulative_observed[:, frame_col]))
        overall = total_starved / total_observed if total_observed > 0 else 0.0
        overall_line.set_ydata([overall, overall])
        overall_legend.get_texts()[0].set_text(f"Overall: {overall * 100:.1f}%")
        return [ln for _, ln in lines] + list(bars) + bar_texts + [overall_line, title, time_marker]

    anim = animation.FuncAnimation(
        fig, update, frames=frame_indices.tolist(), interval=1000 / max(fps, 1), blit=False
    )
    fig.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "per_robot_starvation_rate.gif"
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    logger.info(f"Saved {out}")


def generate_jains_starvation_over_time_plot(
    output_path: pathlib.Path, control_hz: float | None = None
) -> None:
    """Plot Jain's index on cumulative per-robot starvation rate over time.

    1.0 = all robots have equal starvation rate at this point in the run; lower means
    one or more robots are disproportionately starved. Only robots that have observed
    at least one step are included at each timestep.
    """
    robots, matrix, _, control_hz, _ = _build_actions_left_matrix(output_path, control_hz)
    if matrix.size == 0:
        logger.warning("No actions_left.npy data found for Jain's-over-time plot")
        return

    valid_mask = ~np.isnan(matrix)
    starved_mask = valid_mask & (matrix <= 0)
    cumulative_observed = np.cumsum(valid_mask, axis=1)
    cumulative_starved = np.cumsum(starved_mask, axis=1)
    cumulative_rates = np.divide(
        cumulative_starved,
        cumulative_observed,
        out=np.full(matrix.shape, np.nan, dtype=float),
        where=cumulative_observed > 0,
    )

    n_steps = matrix.shape[1]
    jains = np.full(n_steps, np.nan, dtype=float)
    active_count = (cumulative_observed > 0).sum(axis=0)
    for t in range(n_steps):
        active = cumulative_observed[:, t] > 0
        if active.sum() < 2:
            continue
        rates = cumulative_rates[active, t]
        s = float(rates.sum())
        sq = float((rates * rates).sum())
        jains[t] = (s * s) / (active.sum() * sq) if sq > 0 else 1.0

    time_seconds = np.arange(n_steps, dtype=float) / max(control_hz, 1.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_seconds, jains, color="navy", linewidth=1.6)
    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Jain's index on cumulative starvation rate", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Starvation Fairness (Jain's) Over Time",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    final_jain = float(jains[~np.isnan(jains)][-1]) if np.any(~np.isnan(jains)) else float("nan")
    n_active = int(active_count[-1]) if active_count.size else 0
    ax.axhline(final_jain, color="navy", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(
        time_seconds[-1],
        final_jain,
        f" final = {final_jain:.4f}  (n={n_active})",
        va="center",
        ha="right",
        fontsize=9,
        color="navy",
    )

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "jains_starvation_over_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


def generate_staleness_plot(output_path: pathlib.Path) -> None:
    """Per-robot actions_left distribution (staleness), excluding starvation steps (NaN).

    Shows violin bodies with mean, median, and p5 markers (lower = more stale).
    """
    by_robot = load_actions_left(output_path)
    if not by_robot:
        logger.warning("No actions_left data found")
        return

    robots = sorted(by_robot.keys(), key=int)
    robot_actions: dict[str, np.ndarray] = {}
    for robot in robots:
        vals = np.concatenate([arr for _, arr in by_robot[robot]])
        robot_actions[robot] = vals[~np.isnan(vals)]

    valid_robots = [r for r in robots if len(robot_actions.get(r, [])) > 0]
    if not valid_robots:
        logger.warning("No non-starvation actions_left data found")
        return

    data = [robot_actions[r] for r in valid_robots]
    positions = list(range(len(valid_robots)))
    n_robots = len(valid_robots)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_robots), 5))

    parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=False, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    for partname in ["cbars", "cmins", "cmaxes"]:
        if partname in parts:
            parts[partname].set_edgecolor("black")
            parts[partname].set_linewidth(0.8)

    # Overlay mean, median, p5
    stat_styles = [
        ("mean", np.mean, "red", "D", "Mean"),
        ("median", np.median, "white", "o", "Median"),
        ("p5", lambda x: np.percentile(x, 5), "orange", "s", "P5"),
    ]
    for _, fn, color, marker, label in stat_styles:
        vals_stat = [fn(d) for d in data]
        ax.scatter(
            positions,
            vals_stat,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            marker=marker,
            s=60,
            zorder=3,
            label=label,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"robot_{r}" for r in valid_robots], fontsize=9)
    ax.set_xlabel("Robot", fontsize=12)
    ax.set_ylabel("Actions left in queue", fontsize=12)
    ax.set_title(
        "Staleness Distribution (excl. starvation steps)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "staleness_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {plots_dir / 'staleness_distribution.png'}")


def generate_batch_size_plot(output_path: pathlib.Path) -> None:
    """Distribution of action chunk execution horizons (batch sizes)."""

    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        logger.warning("No server_metrics_history.json; skipping batch size plot")
        return

    with open(history_path) as f:
        data = json.load(f)

    # FIXME: should use JSONDataclass loading
    batch_sizes = [_server_batch_fields(batch)[5] for batch in data["batches"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(batch_sizes, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_title("Batch Sizes chosen by Server", fontsize=14, fontweight="bold")
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "batch_size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {plots_dir / 'batch_size_distribution.png'}")


def compute_server_timing_health(
    output_path: pathlib.Path,
    *,
    step_interval_p95_threshold_ms: float = 100.0,
    inbound_p95_threshold_ms: float = 50.0,
    inference_p99_threshold_ms: float = 500.0,
    outbound_p95_threshold_ms: float = 50.0,
) -> dict | None:
    """Return timing percentile stats and flag suspicious values.

    Returns None when server_metrics_history.json is absent.
    """
    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        return None
    data = json.loads(history_path.read_text())

    step_intervals: list[float] = []
    inbound: list[float] = []
    outbound: list[float] = []

    for robot in data.get("robots", {}).values():
        for ep in robot.get("episodes", []):
            ts = _episode_step_timestamps(ep)
            if len(ts) >= 2:
                step_intervals.extend((np.diff(np.asarray(ts, dtype=float)) * 1000.0).tolist())
            for req in ep.get("requests", []):
                ra = req.get("server_arrival_time")
                send_ts = req.get("request_timestamp")
                if ra and send_ts:
                    inbound.append((float(ra) - float(send_ts)) * 1000.0)
            for resp in ep.get("responses", []):
                rcv = resp.get("receive_time", 0.0) or 0.0
                snd = resp.get("server_send_time", 0.0) or 0.0
                if rcv > 0 and snd > 0:
                    outbound.append((float(rcv) - float(snd)) * 1000.0)

    infer: list[float] = []
    for b in data.get("batches", []):
        _, _, _, start, end, _ = _server_batch_fields(b)
        if start and end and end >= start:
            infer.append((float(end) - float(start)) * 1000.0)

    def _pct(arr: list[float], p: float) -> float:
        return float(np.percentile(arr, p)) if arr else 0.0

    si_p95 = _pct(step_intervals, 95)
    ib_p95 = _pct(inbound, 95)
    inf_p99 = _pct(infer, 99)
    ob_p95 = _pct(outbound, 95)

    flags: list[str] = []
    if si_p95 > step_interval_p95_threshold_ms:
        flags.append(f"step_interval_p95={si_p95:.1f}ms>{step_interval_p95_threshold_ms:.0f}ms")
    if ib_p95 > inbound_p95_threshold_ms:
        flags.append(f"inbound_p95={ib_p95:.1f}ms>{inbound_p95_threshold_ms:.0f}ms")
    if inf_p99 > inference_p99_threshold_ms:
        flags.append(f"inference_p99={inf_p99:.1f}ms>{inference_p99_threshold_ms:.0f}ms")
    if ob_p95 > outbound_p95_threshold_ms:
        flags.append(f"outbound_p95={ob_p95:.1f}ms>{outbound_p95_threshold_ms:.0f}ms")

    return {
        "timing_suspicious": bool(flags),
        "timing_flags": "; ".join(flags),
        "step_interval_p50_ms": _pct(step_intervals, 50),
        "step_interval_p95_ms": si_p95,
        "step_interval_p99_ms": _pct(step_intervals, 99),
        "inbound_p50_ms": _pct(inbound, 50),
        "inbound_p95_ms": ib_p95,
        "inference_p50_ms": _pct(infer, 50),
        "inference_p95_ms": _pct(infer, 95),
        "inference_p99_ms": inf_p99,
        "outbound_p50_ms": _pct(outbound, 50),
        "outbound_p95_ms": ob_p95,
    }


def generate_server_timings_plot(output_path: pathlib.Path) -> None:
    """Plot distributions of step interval, client->server, inference, server->client."""
    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        logger.warning("No server_metrics_history.json; skipping server timings plot")
        return
    data = json.loads(history_path.read_text())

    step_intervals: list[float] = []
    inbound: list[float] = []
    outbound: list[float] = []
    for robot in data.get("robots", {}).values():
        for ep in robot.get("episodes", []):
            ts = _episode_step_timestamps(ep)
            if len(ts) >= 2:
                step_intervals.extend((np.diff(np.asarray(ts, dtype=float)) * 1000.0).tolist())
            for req in ep.get("requests", []):
                ra = req.get("server_arrival_time")
                send_ts = req.get("request_timestamp")
                if ra and send_ts:
                    inbound.append((ra - send_ts) * 1000.0)
            for resp in ep.get("responses", []):
                rcv = resp.get("receive_time", 0.0) or 0.0
                snd = resp.get("server_send_time", 0.0) or 0.0
                if rcv > 0 and snd > 0:
                    outbound.append((rcv - snd) * 1000.0)

    infer: list[float] = []
    for b in data.get("batches", []):
        _, _, _, start, end, _ = _server_batch_fields(b)
        if start and end and end >= start:
            infer.append((end - start) * 1000.0)

    timings = {
        "step interval (client-local ms)": np.asarray(step_intervals),
        "client->server transport delay (ms)": np.asarray(inbound),
        "inference time (ms)": np.asarray(infer),
        "server->client delay (ms)": np.asarray(outbound),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    assert len(timings) == axes.size
    for ax, (label, arr) in zip(axes.flat, timings.items()):
        if arr.size == 0:
            ax.set_title(f"{label} (no data)")
            ax.set_axis_off()
            continue
        hi = float(np.percentile(arr, 99.5))
        ax.hist(
            np.clip(arr, None, hi),
            bins=100,
            color="steelblue",
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.set_title(
            f"n={arr.size}  p50={np.percentile(arr, 50):.1f}  p95={np.percentile(arr, 95):.1f}  "
            f"p99={np.percentile(arr, 99):.1f}  max={arr.max():.1f}",
            fontsize=9,
        )
        for p, c in [(50, "green"), (95, "orange"), (99, "red")]:
            ax.axvline(np.percentile(arr, p), color=c, linestyle="--", linewidth=1)
    fig.tight_layout()

    out = output_path / "plots" / "server_timings.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved server timings plot to %s", out)


def generate_server_timings_over_time_plot(output_path: pathlib.Path) -> None:
    """Plot server timings over wall-clock time to check temporal alignment of spikes."""
    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        logger.warning("No server_metrics_history.json; skipping server timings over time plot")
        return
    data = json.loads(history_path.read_text())

    step_t: list[float] = []
    step_v: list[float] = []
    step_robot: list[str] = []
    inbound_t: list[float] = []
    inbound_v: list[float] = []
    inbound_robot: list[str] = []
    outbound_t: list[float] = []
    outbound_v: list[float] = []
    outbound_robot: list[str] = []

    for robot_id, robot in data.get("robots", {}).items():
        for ep in robot.get("episodes", []):
            ts = _episode_step_timestamps(ep)
            for i in range(1, len(ts)):
                step_t.append(float(ts[i]))
                step_v.append((float(ts[i]) - float(ts[i - 1])) * 1000.0)
                step_robot.append(robot_id)
            for req in ep.get("requests", []):
                ra = req.get("server_arrival_time")
                send_ts = req.get("request_timestamp")
                if ra and send_ts:
                    inbound_t.append(float(send_ts))
                    inbound_v.append((float(ra) - float(send_ts)) * 1000.0)
                    inbound_robot.append(robot_id)
            for resp in ep.get("responses", []):
                rcv = resp.get("receive_time", 0.0) or 0.0
                snd = resp.get("server_send_time", 0.0) or 0.0
                if rcv > 0 and snd > 0:
                    outbound_t.append(float(snd))
                    outbound_v.append((float(rcv) - float(snd)) * 1000.0)
                    outbound_robot.append(robot_id)

    infer_t: list[float] = []
    infer_v: list[float] = []
    infer_bs: list[int] = []
    for b in data.get("batches", []):
        _, _, _, start, end, batch_size = _server_batch_fields(b)
        if start and end and end >= start:
            infer_t.append(float(start))
            infer_v.append((float(end) - float(start)) * 1000.0)
            infer_bs.append(batch_size)

    all_times = step_t + inbound_t + outbound_t + infer_t
    if not all_times:
        logger.warning("No server timing samples; skipping over-time plot")
        return
    t0 = min(all_times)

    def _rel(ts: list[float]) -> np.ndarray:
        return np.asarray(ts, dtype=float) - t0

    series = [
        (
            "step interval (client-local ms)",
            _rel(step_t),
            np.asarray(step_v),
            step_robot,
            None,
        ),
        (
            "client->server transport delay (ms)",
            _rel(inbound_t),
            np.asarray(inbound_v),
            inbound_robot,
            None,
        ),
        (
            "inference time (ms)",
            _rel(infer_t),
            np.asarray(infer_v),
            None,
            np.asarray(infer_bs),
        ),
        (
            "server->client delay (ms)",
            _rel(outbound_t),
            np.asarray(outbound_v),
            outbound_robot,
            None,
        ),
    ]

    robot_ids = sorted({*step_robot, *inbound_robot, *outbound_robot})
    cmap = matplotlib.colormaps["tab20" if len(robot_ids) > 10 else "tab10"]
    robot_color = {r: cmap(i % cmap.N) for i, r in enumerate(robot_ids)}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    assert len(series) == axes.size
    for ax, (label, t, v, robots, bs) in zip(axes, series):
        if t.size == 0:
            ax.set_title(f"{label} (no data)")
            ax.set_ylabel(label)
            continue
        if bs is not None:
            sc = ax.scatter(t, v, c=bs, cmap="viridis", s=6, alpha=0.7)
            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label("batch size", fontsize=8)
        else:
            colors = [robot_color[r] for r in robots]  # type: ignore[index]
            ax.scatter(t, v, c=colors, s=4, alpha=0.5, linewidths=0)
        ax.set_ylabel(label, fontsize=9)
        hi = float(np.percentile(v, 99.5))
        ax.set_ylim(0, max(hi * 1.1, 1.0))
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"n={v.size}  p50={np.percentile(v, 50):.1f}  "
            f"p95={np.percentile(v, 95):.1f}  p99={np.percentile(v, 99):.1f}  "
            f"max={v.max():.1f} (y clipped at p99.5)",
            fontsize=9,
        )

    axes[-1].set_xlabel("time since first sample (s)")
    if robot_ids:
        handles = [Patch(facecolor=robot_color[r], label=r) for r in robot_ids]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(robot_ids), 10),
            fontsize=8,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = output_path / "plots" / "server_timings_over_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved server timings over time plot to %s", out)


def generate_server_batch_gantt_plot(output_path: pathlib.Path) -> None:
    """Plot server inference batches as robot-lane Gantt bars over wall-clock time.

    A `Decisions` lane below the robot lanes overlays scheduler decisions:
    decision-start ticks, decision-duration whiskers, and a green/red marker
    indicating whether the decision produced a batch (linked to the batch_id).
    """
    history_path = output_path / "server_metrics_history.json"
    if not history_path.exists():
        logger.warning("No server_metrics_history.json; skipping server batch Gantt plot")
        return
    data = json.loads(history_path.read_text())

    batch_rows = []
    for batch in data.get("batches", []):
        batch_id, robot_ids, _, start, end, batch_size = _server_batch_fields(batch)
        if start is None or end is None:
            continue
        if not robot_ids:
            continue
        start = float(start)
        end = float(end)
        if end < start:
            continue
        batch_rows.append(
            {
                "batch_id": batch_id,
                "robot_ids": list(robot_ids),
                "start": start,
                "duration": max(end - start, 0.0),
                "batch_size": int(batch_size),
            }
        )

    if not batch_rows:
        logger.warning("No valid server batches; skipping server batch Gantt plot")
        return

    batch_rows.sort(key=lambda row: (row["start"], str(row["batch_id"])))
    t0 = float(data.get("start_time") or min(row["start"] for row in batch_rows))

    robot_ids = sorted({str(rid) for row in batch_rows for rid in row["robot_ids"]})
    robot_y = {rid: i for i, rid in enumerate(robot_ids)}
    cmap = matplotlib.colormaps["tab20" if len(robot_ids) > 10 else "tab10"]
    robot_color = {rid: cmap(i % cmap.N) for i, rid in enumerate(robot_ids)}

    decisions = _load_scheduler_decisions(output_path)
    decisions = [
        d for d in decisions if d.get("started_at") is not None and float(d["started_at"]) >= t0
    ]
    decisions.sort(key=lambda d: float(d["started_at"]))
    has_decisions = bool(decisions)
    decision_y = len(robot_ids)  # extra lane below all robot lanes

    fig_height = max(3.0, (len(robot_ids) + (1.2 if has_decisions else 0)) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for row in batch_rows:
        start_t = row["start"] - t0
        duration = row["duration"]
        for rid_raw in row["robot_ids"]:
            rid = str(rid_raw)
            ax.barh(
                robot_y[rid],
                duration,
                left=start_t,
                height=0.72,
                color=robot_color[rid],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
            )

    PHASE_COLORS = {
        "setup": "#bdbdbd",
        "gc": "#7b3294",
        "search_init": "#fdae61",
        "search_step": "#4292c6",
        "search": "#4292c6",
        "postprocess": "#9ecae1",
        "return_overhead": "#d62728",
        "greedy": "#fc8d59",
        "dispatch": "#41ab5d",
    }
    if has_decisions:
        # If notes include per-phase records, draw colored phase segments on the
        # Decisions lane. Otherwise fall back to a flat whisker per decision.
        # Dedupe by started_at — multi-batch calls produce one decision per
        # batch but share a single phases list.
        seen_phase_calls: set[float] = set()
        flat_scheduled_t, flat_scheduled_dur = [], []
        flat_skipped_t, flat_skipped_dur = [], []
        phase_kinds_seen: set[str] = set()
        for d in decisions:
            phases = (
                ((d.get("notes") or {}).get("phases")) if isinstance(d.get("notes"), dict) else None
            )
            t = float(d["started_at"]) - t0
            dur = float(d.get("duration") or 0.0)
            if phases:
                started_at = float(d["started_at"])
                if started_at in seen_phase_calls:
                    continue
                seen_phase_calls.add(started_at)
                for ph in phases:
                    name = ph.get("name", "")
                    p_start = float(ph.get("start", 0.0)) - t0
                    p_end = float(ph.get("end", 0.0)) - t0
                    if p_end <= p_start:
                        continue
                    phase_kinds_seen.add(name)
                    # search_step gets a thicker edge so consecutive tiles
                    # render as visibly distinct segments.
                    edge_lw = 0.6 if name == "search_step" else 0.2
                    ax.barh(
                        decision_y,
                        p_end - p_start,
                        left=p_start,
                        height=0.5,
                        color=PHASE_COLORS.get(name, "#888888"),
                        edgecolor="black",
                        linewidth=edge_lw,
                        alpha=0.9,
                    )
            elif d.get("batch_id") is not None:
                flat_scheduled_t.append(t)
                flat_scheduled_dur.append(dur)
            else:
                flat_skipped_t.append(t)
                flat_skipped_dur.append(dur)

        for t, dur in zip(flat_scheduled_t, flat_scheduled_dur):
            ax.plot(
                [t, t + dur],
                [decision_y, decision_y],
                color="seagreen",
                linewidth=2.2,
                alpha=0.85,
                solid_capstyle="butt",
            )
        for t, dur in zip(flat_skipped_t, flat_skipped_dur):
            ax.plot(
                [t, t + dur],
                [decision_y, decision_y],
                color="indianred",
                linewidth=2.2,
                alpha=0.55,
                solid_capstyle="butt",
            )
        # Keep variable names from changing for the marker code below.
        scheduled_t, _ = flat_scheduled_t, flat_scheduled_dur
        skipped_t, _ = flat_skipped_t, flat_skipped_dur
        if scheduled_t:
            ax.scatter(
                scheduled_t,
                [decision_y] * len(scheduled_t),
                marker="|",
                color="darkgreen",
                s=80,
                linewidths=1.4,
                zorder=3,
                label="decision → batch",
            )
        if skipped_t:
            ax.scatter(
                skipped_t,
                [decision_y] * len(skipped_t),
                marker="|",
                color="firebrick",
                s=80,
                linewidths=1.0,
                alpha=0.7,
                zorder=3,
                label="decision → skip",
            )

        # Connect each dispatched decision to the corresponding batch with a faint vertical line.
        batch_start_by_id = {row["batch_id"]: row["start"] - t0 for row in batch_rows}
        for d in decisions:
            bid = d.get("batch_id")
            if bid is None:
                continue
            batch_t = batch_start_by_id.get(bid)
            if batch_t is None:
                continue
            t = float(d["started_at"]) - t0
            ax.plot(
                [t, batch_t],
                [decision_y, decision_y - 0.5],
                color="dimgray",
                linewidth=0.4,
                alpha=0.35,
                zorder=1,
            )

    yticks = list(range(len(robot_ids)))
    ylabels = list(robot_ids)
    if has_decisions:
        yticks.append(decision_y)
        ylabels.append("Decisions")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()

    title = "GPU Gantt"
    if has_decisions:
        scheduler_names = sorted(
            {d.get("scheduler") or d.get("scheduler_name") or "" for d in decisions}
        )
        scheduler_names = [n for n in scheduler_names if n]
        if scheduler_names:
            title = f"GPU Gantt ({', '.join(scheduler_names)})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time since server start (s)", fontsize=12)
    ax.set_ylabel("Robot", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    handles = [Patch(facecolor=robot_color[rid], label=rid) for rid in robot_ids]
    if len(handles) <= 20:
        legend = ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
        if has_decisions:
            ax.add_artist(legend)
            ax.legend(loc="lower right", fontsize=8, frameon=False)
    elif has_decisions:
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    if has_decisions and phase_kinds_seen:
        phase_handles = [
            Patch(facecolor=PHASE_COLORS.get(name, "#888888"), label=name)
            for name in [
                "setup",
                "gc",
                "search_init",
                "search_step",
                "search",
                "postprocess",
                "return_overhead",
                "greedy",
                "dispatch",
            ]
            if name in phase_kinds_seen
        ]
        phase_legend = ax.legend(
            handles=phase_handles,
            loc="upper left",
            fontsize=8,
            frameon=False,
            title="Scheduler phases",
            title_fontsize=8,
        )
        ax.add_artist(phase_legend)

    fig.tight_layout()

    out = output_path / "plots" / "server_batch_gantt.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved server batch Gantt plot to %s", out)


def generate_all_plots(output_path: pathlib.Path) -> None:
    """Generate all plots; one failure doesn't kill the rest."""
    logger.info("Generating plots...")
    plotters = [
        generate_latency_plot,
        generate_success_rate_plot,
        generate_steps_plot,
        generate_per_robot_success_rate_plot,
        generate_actions_left_heatmap,
        generate_starvation_plot,
        generate_starvation_tail_metrics_plot,
        generate_starvation_variance_plot,
        # generate_per_robot_starvation_rate_gif,  # slow (~5-10s per run); run manually if needed
        generate_jains_starvation_over_time_plot,
        generate_staleness_plot,
        generate_batch_size_plot,
        generate_server_timings_plot,
        generate_server_timings_over_time_plot,
        generate_server_batch_gantt_plot,
    ]
    for plotter in plotters:
        try:
            plotter(output_path)
        except Exception:
            logger.exception("Plot %s failed; continuing", plotter.__name__)
    logger.info("Done!")


# =============================================================================
# Metrics Calculation (console output + CSV)
# =============================================================================


def calculate_metrics(output_path: pathlib.Path) -> None:
    """Aggregate results and display summary table."""
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No results found")
        return

    planner_starvation_df = load_planner_starvation_metrics(output_path)
    if not planner_starvation_df.empty:
        df = df.merge(
            planner_starvation_df,
            on=["robot_idx", "episode_idx", "task_suite_name", "task_id"],
            how="left",
        )

    df.to_csv(output_path / "results.csv", index=False)

    aggregation_spec: dict[str, str] = {"success": "mean"}
    assert "starvation_steps" in df.columns
    assert "observed_steps" in df.columns
    assert "planner_starvation_seconds" in df.columns
    aggregation_spec["starvation_steps"] = "sum"
    aggregation_spec["observed_steps"] = "sum"
    aggregation_spec["planner_starvation_seconds"] = "sum"
    aggregation_spec["post_first_starvation_steps"] = "sum"
    aggregation_spec["post_first_observed_steps"] = "sum"

    summary = df.groupby(["task_suite_name", "task_id"]).agg(aggregation_spec)
    summary["planner_starvation_rate"] = summary["starvation_steps"] / summary["observed_steps"]
    summary["post_first_starvation_rate"] = (
        summary["post_first_starvation_steps"] / summary["post_first_observed_steps"]
    )
    summary.reset_index().to_csv(output_path / "summary.csv", index=False)

    # Display with rich
    console = Console()
    table = Table(title="Task Success Summary")
    table.add_column("Task Suite", style="cyan")
    table.add_column("Task ID", style="magenta")
    table.add_column("Success Rate", style="green")
    table.add_column("Total Starvation Steps", style="yellow")
    table.add_column("Starvation Rate", style="yellow")

    for _, row in summary.reset_index().iterrows():
        table.add_row(
            str(row["task_suite_name"]),
            str(row["task_id"]),
            f"{row['success']:.2%}",
            str(int(row["starvation_steps"])),
            f"{row['planner_starvation_rate']:.2%}",
        )

    console.print(table)

    # Per-robot success summary
    robot_agg_spec: dict[str, str] = {
        "success": "mean",
        "episode_idx": "count",
        "starvation_steps": "sum",
        "observed_steps": "sum",
    }
    robot_summary = df.groupby("robot_idx").agg(robot_agg_spec).reset_index()
    robot_summary.rename(columns={"episode_idx": "count"}, inplace=True)
    robot_summary["planner_starvation_rate"] = (
        robot_summary["starvation_steps"] / robot_summary["observed_steps"]
    )

    robot_table = Table(title="Per-Robot Success Summary")
    robot_table.add_column("Robot", style="cyan")
    robot_table.add_column("Success Rate", style="green")
    robot_table.add_column("Episodes", style="magenta")
    robot_table.add_column("Total Starvation Steps", style="yellow")
    robot_table.add_column("Starvation Rate", style="yellow")
    for _, row in robot_summary.sort_values("robot_idx").iterrows():
        robot_table.add_row(
            str(int(row["robot_idx"])),
            f"{row['success']:.2%}",
            str(int(row["count"])),
            str(int(row["starvation_steps"])),
            f"{row['planner_starvation_rate']:.2%}",
        )
    console.print(robot_table)

    total_starvation_steps = int(df["starvation_steps"].sum())
    total_observed_steps = int(df["observed_steps"].sum())
    overall_starvation_rate = total_starvation_steps / total_observed_steps
    total_post_first_starvation_steps = int(df["post_first_starvation_steps"].sum())
    total_post_first_observed_steps = int(df["post_first_observed_steps"].sum())
    overall_post_first_starvation_rate = (
        total_post_first_starvation_steps / total_post_first_observed_steps
        if total_post_first_observed_steps > 0
        else 0.0
    )
    console.print(f"\n[bold green]Total success rate: {summary['success'].mean():.2%}[/bold green]")
    console.print(
        f"[bold yellow]Total starvation steps: {total_starvation_steps} control steps[/bold yellow]"
    )
    console.print(
        f"[bold yellow]Planner starvation rate: {overall_starvation_rate:.2%}[/bold yellow]"
    )
    console.print(
        f"[bold yellow]Planner starvation rate (excl. pre-first-action): "
        f"{overall_post_first_starvation_rate:.2%}[/bold yellow]"
    )
    console.print(
        f"[bold yellow]Planner starvation time: {df['planner_starvation_seconds'].sum():.2f}s[/bold yellow]"
    )

    total_successes = int(df["success"].sum())
    experiment_duration = load_experiment_duration(output_path)
    if experiment_duration is not None:
        successes_per_second = total_successes / experiment_duration
        console.print(
            f"[bold cyan]Total experiment time: {experiment_duration:.1f}s ({experiment_duration / 60:.1f}min)[/bold cyan]"
        )
        console.print(
            f"[bold cyan]Throughput: {successes_per_second:.3f} successes/second[/bold cyan]"
        )

    fairness = compute_fairness_metrics(output_path)
    if fairness is not None:
        fair_table = Table(title="Per-Robot Outcome Fairness")
        fair_table.add_column("Robot", style="cyan")
        fair_table.add_column("Starvation Rate", style="yellow")
        fair_table.add_column("Freshness Rate (1-starv)", style="green")
        for idx, sr, fr in zip(
            fairness["robot_idx"],
            fairness["starvation_rate"],
            fairness["freshness_rate"],
        ):
            fair_table.add_row(str(idx), f"{sr:.3f}", f"{fr:.3f}")
        console.print(fair_table)
        alpha_str = f" (alpha={fairness['alpha']})" if fairness["alpha"] is not None else ""
        console.print(
            f"[bold magenta]Jain's index on freshness rate{alpha_str}: "
            f"{fairness['jain_freshness']:.4f}[/bold magenta]"
        )
        console.print(
            f"[bold magenta]Jain's index on starvation rate: "
            f"{fairness['jain_starvation']:.4f}[/bold magenta]"
        )
