import logging
import pathlib

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics.loading import (
    actions_left_matrix,
    completed_episodes,
    load_action_chunks,
    load_episodes,
    load_scheduler_decisions,
    robot_starvation_rates,
    starvation_variance_series,
    steps_by_robot,
)

logger = logging.getLogger(__name__)


def save_fig(fig: plt.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def robot_colors(robots: list[str]) -> dict[str, tuple]:
    cmap = matplotlib.colormaps["tab20" if len(robots) > 10 else "tab10"]
    return {robot: cmap(i % cmap.N) for i, robot in enumerate(robots)}


def percentile_histogram(
    ax: plt.Axes, data: np.ndarray, xlabel: str, title: str = "", bins: int = 100
) -> None:
    arr = np.asarray(data, dtype=float)
    ax.hist(
        np.clip(arr, None, np.percentile(arr, 99.5)),
        bins=bins,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    stats = (
        f"n={arr.size}  p50={np.percentile(arr, 50):.1f}  p95={np.percentile(arr, 95):.1f}  "
        f"p99={np.percentile(arr, 99):.1f}  max={arr.max():.1f}"
    )
    ax.set_title(f"{title}\n{stats}" if title else stats, fontsize=9)
    for percentile, color in [(50, "green"), (95, "orange"), (99, "red")]:
        ax.axvline(np.percentile(arr, percentile), color=color, linestyle="--", linewidth=1)


def plot_bar_chart(
    ax: plt.Axes,
    labels: list[str],
    values: np.ndarray,
    ylabel: str,
    title: str,
    xlabel: str = "Task",
    counts: np.ndarray | None = None,
    overall_line: tuple[float, str] | None = None,
    color_fn=None,
) -> None:
    colors = [color_fn(value) for value in values] if color_fn else "steelblue"
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    rotate = max((len(label) for label in labels), default=0) > 4
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
    ax.grid(axis="y", alpha=0.3)

    if overall_line:
        value, label = overall_line
        ax.axhline(y=value, color="red", linestyle="--", linewidth=2, label=label)
        ax.legend()

    for i, (bar, value) in enumerate(zip(bars, values)):
        annotation = f"{value:.1%}"
        if counts is not None:
            annotation += f"\n(n={counts[i]})"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            annotation,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _task_groups(df):
    df = df.copy()
    df["task_label"] = "Task " + df["task_id"].astype(str) + "\n" + df["task_language"].str[:30]
    return df, df.groupby(["task_id", "task_label"], sort=True)


def generate_latency_plot(output_path: pathlib.Path) -> None:
    df = load_action_chunks(output_path)
    if df.empty:
        logger.warning("No action chunks for latency plot")
        return
    df["latency_ms"] = df["latency"] * 1000
    df, grouped = _task_groups(df)
    panels = [("All Tasks Combined", df)]
    panels += [(task_label, group) for (_, task_label), group in grouped]

    n_cols = min(3, len(panels))
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    fig.suptitle("Action Chunk Latency Distribution", fontsize=16, fontweight="bold")
    for ax, (title, group) in zip(axes.flat, panels):
        percentile_histogram(ax, group["latency_ms"].to_numpy(), "Latency (ms)", title=title)
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "action_chunk_latency.png")


def generate_client_step_intervals_plot(output_path: pathlib.Path) -> None:
    intervals_by_robot = {
        robot: [
            np.diff(steps["timestamp"].to_numpy(dtype=float)) * 1000.0
            for steps in episodes
            if len(steps) >= 2
        ]
        for robot, episodes in steps_by_robot(output_path).items()
    }
    intervals_by_robot = {
        robot: np.concatenate(episodes)
        for robot, episodes in intervals_by_robot.items()
        if episodes
    }
    if not intervals_by_robot:
        logger.warning("No client timestamp intervals; skipping client step interval plot")
        return

    intervals = np.concatenate(list(intervals_by_robot.values()))
    fig, (ax_time, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))
    colors = robot_colors(list(intervals_by_robot))

    for robot, samples in intervals_by_robot.items():
        ax_time.scatter(
            np.arange(1, len(samples) + 1),
            samples,
            s=7,
            alpha=0.65,
            color=colors[robot],
            label=f"robot {robot}",
            linewidths=0,
        )
        ax_hist.hist(
            np.clip(samples, None, np.percentile(samples, 99.5)),
            bins=80,
            histtype="step",
            linewidth=1.2,
            color=colors[robot],
            label=f"robot {robot}",
        )
    ax_time.set_xlabel("client step index (episodes concatenated)")
    ax_time.set_ylabel("client step interval (ms)")
    ax_time.grid(True, alpha=0.3)

    ax_hist.set_xlabel("client step interval (ms)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(
        f"n={len(intervals)}  p50={np.percentile(intervals, 50):.1f}  "
        f"p95={np.percentile(intervals, 95):.1f}  p99={np.percentile(intervals, 99):.1f}  "
        f"max={intervals.max():.1f}",
        fontsize=9,
    )
    for percentile, color in [(50, "green"), (95, "orange"), (99, "red")]:
        ax_hist.axvline(
            np.percentile(intervals, percentile), color=color, linestyle="--", linewidth=1
        )
    if len(intervals_by_robot) > 1:
        ax_time.legend(fontsize=7)
        ax_hist.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "client_step_intervals.png")


def _success_color(rate: float) -> str:
    if rate >= 0.8:
        return "green"
    if rate >= 0.5:
        return "orange"
    return "red"


def generate_success_rate_plot(output_path: pathlib.Path) -> None:
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No episode data for success rate plot")
        return
    df = completed_episodes(df)
    if df.empty:
        logger.warning("Every episode was truncated by the deadline; no success rate to plot")
        return

    df, grouped = _task_groups(df)
    summary = grouped["success"].agg(["mean", "count"]).reset_index()
    overall_rate = df["success"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_bar_chart(
        ax,
        labels=summary["task_label"].tolist(),
        values=summary["mean"].to_numpy(),
        ylabel="Success Rate",
        title="Success Rate by Task",
        counts=summary["count"].to_numpy(),
        overall_line=(overall_rate, f"Overall: {overall_rate:.2%}"),
        color_fn=_success_color,
    )
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "success_rate.png")


def generate_per_robot_success_rate_plot(output_path: pathlib.Path) -> None:
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No episode data for per-robot success rate plot")
        return
    df = completed_episodes(df)
    if df.empty:
        logger.warning("Every episode was truncated by the deadline; no success rate to plot")
        return

    summary = df.groupby("robot_idx")["success"].agg(["mean", "count"]).reset_index()
    summary = summary.sort_values("robot_idx")
    overall_rate = df["success"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_bar_chart(
        ax,
        labels=summary["robot_idx"].astype(str).tolist(),
        values=summary["mean"].to_numpy(),
        ylabel="Success Rate",
        title="Per-Robot Success Rate",
        xlabel="Robot Index",
        counts=summary["count"].to_numpy(),
        overall_line=(overall_rate, f"Overall: {overall_rate:.2%}"),
    )
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "per_robot_success_rate.png")


def generate_starvation_plot(output_path: pathlib.Path) -> None:
    rates = robot_starvation_rates(output_path)
    if rates.empty:
        logger.warning("No starvation data found")
        return

    overall = rates["starvation_steps"].sum() / rates["observed_steps"].sum()
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(rates)), 5))
    plot_bar_chart(
        ax,
        labels=rates["robot_idx"].astype(str).tolist(),
        values=rates["starvation_rate"].to_numpy(),
        ylabel="Starvation Rate",
        title="Per-Robot Starvation Rate",
        xlabel="Robot Index",
        overall_line=(overall, f"Overall: {overall:.2%}"),
    )
    ax.set_ylim(0, min(1.0, rates["starvation_rate"].max() * 1.3 + 0.05))
    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "starvation_rate.png")


def generate_steps_plot(output_path: pathlib.Path) -> None:
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No episode data for steps plot")
        return

    fig, (ax_hist, ax_violin) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Steps Taken Analysis", fontsize=16, fontweight="bold")

    success_steps = df[df["success"]]["steps_taken"].to_numpy()
    if len(success_steps) > 0:
        ax_hist.hist(success_steps, bins=30, color="green", alpha=0.7, edgecolor="black")
        ax_hist.set_xlabel("Steps")
        ax_hist.set_ylabel("Count")
    else:
        ax_hist.text(
            0.5,
            0.5,
            "No successful episodes",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
    ax_hist.set_title("Successful Episodes")

    df, grouped = _task_groups(df)
    groups = [
        (task_label, group[group["success"]]["steps_taken"].to_numpy())
        for (_, task_label), group in grouped
    ]
    groups = [(label, steps) for label, steps in groups if len(steps) > 0]
    if groups:
        parts = ax_violin.violinplot(
            [steps for _, steps in groups],
            positions=range(len(groups)),
            widths=0.7,
            showmeans=True,
            showmedians=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor("lightgreen")
            body.set_alpha(0.7)
        ax_violin.set_xticks(range(len(groups)))
        ax_violin.set_xticklabels(
            [label for label, _ in groups], rotation=45, ha="right", fontsize=8
        )
        ax_violin.set_ylabel("Steps")
        ax_violin.grid(axis="y", alpha=0.3)
    else:
        ax_violin.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_violin.transAxes)
    ax_violin.set_title("Steps by Task (Successful Episodes)")

    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "steps_taken.png")


def generate_staleness_plot(output_path: pathlib.Path) -> None:
    robot_actions = {
        robot: np.concatenate(
            [steps["actions_left"].dropna().to_numpy(dtype=float) for steps in episodes]
        )
        for robot, episodes in steps_by_robot(output_path).items()
    }
    robot_actions = {robot: values for robot, values in robot_actions.items() if len(values)}
    if not robot_actions:
        logger.warning("No actions_left data found")
        return

    robots = list(robot_actions)
    data = [robot_actions[robot] for robot in robots]
    positions = list(range(len(robots)))

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(robots)), 5))
    parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("steelblue")
        body.set_alpha(0.6)

    stat_styles = [
        (np.mean, "red", "D", "Mean"),
        (np.median, "white", "o", "Median"),
        (lambda x: np.percentile(x, 5), "orange", "s", "P5"),
    ]
    for fn, color, marker, label in stat_styles:
        ax.scatter(
            positions,
            [fn(values) for values in data],
            color=color,
            edgecolors="black",
            linewidths=0.8,
            marker=marker,
            s=60,
            zorder=3,
            label=label,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"robot_{robot}" for robot in robots], fontsize=9)
    ax.set_xlabel("Robot", fontsize=12)
    ax.set_ylabel("Actions left in queue", fontsize=12)
    ax.set_title("Staleness Distribution (excl. starvation steps)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "staleness_distribution.png")


def generate_actions_left_heatmap(
    output_path: pathlib.Path, control_hz: float | None = None
) -> None:
    robots, matrix, episode_boundaries, control_hz, t0 = actions_left_matrix(
        output_path, control_hz
    )
    if matrix.size == 0:
        logger.warning("No actions_left data found")
        return

    n_robots = len(robots)
    max_len = matrix.shape[1]

    fig_width = min(400, max(12, max_len / max(control_hz, 1.0)))
    fig, ax = plt.subplots(figsize=(fig_width, max(4, n_robots * 0.6)))

    vmax = max(1, int(np.nanmax(matrix))) if not np.all(np.isnan(matrix)) else 1
    # Black for 0 (starvation), then RdYlGn for 1..vmax.
    rdylgn = matplotlib.colormaps["RdYlGn"].resampled(vmax)
    cmap = mcolors.ListedColormap([(0.0, 0.0, 0.0, 1.0)] + [rdylgn(i) for i in range(vmax)])

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=vmax,
    )

    for row, bounds in enumerate(episode_boundaries):
        for bound in bounds[1:]:
            ax.plot(
                [bound - 0.5, bound - 0.5],
                [row - 0.4, row + 0.4],
                color="white",
                linewidth=0.8,
                alpha=0.7,
            )

    decisions_overlaid = 0
    robot_to_row = {robot: row for row, robot in enumerate(robots)}
    for decision in load_scheduler_decisions(output_path):
        col = (decision["started_at"] - t0) * control_hz
        if col < -0.5 or col > max_len - 0.5:
            continue
        for rid in decision["scheduled"]:
            row = robot_to_row.get(rid)
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

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Actions left in queue", fontweight="bold")

    ax.set_yticks(range(n_robots))
    ax.set_yticklabels([f"robot_{robot}" for robot in robots], fontsize=8)
    tick_interval = max(1, int(round(control_hz)))
    x_ticks = np.arange(0, max_len, tick_interval)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick // tick_interval}s" for tick in x_ticks], fontsize=6)
    decision_legend = " | green ticks = scheduler dispatch" if decisions_overlaid else ""
    ax.set_xlabel(
        f"Wall-clock time in seconds (white lines = episode boundaries{decision_legend})",
        fontweight="bold",
    )
    ax.set_ylabel("Robot", fontweight="bold")
    ax.set_title("Actions Left Per Robot Over Time", fontsize=14, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "actions_left_heatmap.png")


def generate_starvation_variance_plot(
    output_path: pathlib.Path, control_hz: float | None = None
) -> None:
    series = starvation_variance_series(output_path, control_hz)
    if series is None:
        logger.warning("No step data found for starvation variance plot")
        return
    robots = series["robots"]
    time_seconds = series["time_seconds"]

    fig, (ax_rates, ax_var) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )
    fig.suptitle("Starvation Fairness Across Robots Over Time", fontsize=14, fontweight="bold")

    colors = robot_colors(robots)
    for row, robot in enumerate(robots):
        ax_rates.plot(
            time_seconds,
            series["cumulative_rates"][row],
            linewidth=1.5,
            color=colors[robot],
            label=f"robot_{robot}",
        )
    ax_rates.set_ylabel("Cumulative starvation rate", fontsize=12)
    ax_rates.set_ylim(0, 1)
    ax_rates.grid(True, alpha=0.3)
    ax_rates.legend(loc="upper right", ncol=min(len(robots), 5), fontsize=8, frameon=False)

    ax_var.plot(time_seconds, series["starvation_variance"], color="darkred", linewidth=2)
    ax_var.fill_between(time_seconds, series["starvation_variance"], color="tomato", alpha=0.2)
    ax_var.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax_var.set_ylabel("Variance", fontsize=12)
    ax_var.set_ylim(bottom=0)
    ax_var.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, output_path / "plots" / "starvation_variance_over_time.png")
