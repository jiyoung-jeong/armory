import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from evaluation.metrics.loading import (
    iter_steps,
    load_scheduler_decisions,
    load_server_batches,
    load_server_events,
    request_timings,
)
from evaluation.metrics.plots import percentile_histogram, robot_colors, save_fig

logger = logging.getLogger(__name__)


def _timing_series(
    output_path: pathlib.Path,
) -> list[tuple[str, np.ndarray, np.ndarray, list[str] | None, np.ndarray | None]]:
    step_t: list[float] = []
    step_v: list[float] = []
    step_robot: list[str] = []
    for episode_dir, steps in iter_steps(output_path):
        ts = steps["timestamp"].to_numpy(dtype=float)
        if len(ts) >= 2:
            step_t.extend(ts[1:])
            step_v.extend(np.diff(ts) * 1000.0)
            step_robot.extend([episode_dir.parent.name] * (len(ts) - 1))

    events = load_server_events(output_path)
    requests = events[events["kind"] == "request"] if not events.empty else pd.DataFrame()
    acks = events[events["kind"] == "ack"] if not events.empty else pd.DataFrame()
    batches = load_server_batches(output_path)
    infer = batches[batches["batch_size"] > 0] if not batches.empty else pd.DataFrame()

    def _series(df: pd.DataFrame, t_col: str, value) -> tuple[np.ndarray, np.ndarray]:
        if df.empty:
            return np.array([]), np.array([])
        return df[t_col].to_numpy(dtype=float), value(df).to_numpy(dtype=float) * 1000.0

    request_t, request_v = _series(
        requests, "request_timestamp", lambda df: df["arrival_time"] - df["request_timestamp"]
    )
    ack_t, ack_v = _series(
        acks, "server_send_time", lambda df: df["receive_time"] - df["server_send_time"]
    )
    infer_t, infer_v = _series(infer, "inference_start_time", lambda df: df["inference_duration"])

    return [
        (
            "step interval (client-local ms)",
            np.asarray(step_t),
            np.asarray(step_v),
            step_robot,
            None,
        ),
        (
            "client->server transport delay (ms)",
            request_t,
            request_v,
            requests["robot_id"].tolist() if not requests.empty else [],
            None,
        ),
        (
            "inference time (ms)",
            infer_t,
            infer_v,
            None,
            infer["batch_size"].to_numpy() if not infer.empty else np.array([]),
        ),
        (
            "server->client delay (ms)",
            ack_t,
            ack_v,
            acks["robot_id"].tolist() if not acks.empty else [],
            None,
        ),
    ]


def compute_server_timing_health(
    output_path: pathlib.Path,
    *,
    step_interval_p95_threshold_ms: float = 100.0,
    inbound_p95_threshold_ms: float = 50.0,
    inference_p99_threshold_ms: float = 500.0,
    outbound_p95_threshold_ms: float = 50.0,
) -> dict | None:
    series = _timing_series(output_path)
    step_intervals, inbound, infer, outbound = (values for _, _, values, _, _ in series)
    if not inbound.size and not infer.size and not outbound.size:
        return None

    def _pct(arr: np.ndarray, p: float) -> float:
        return float(np.percentile(arr, p)) if arr.size else 0.0

    si_p95 = _pct(step_intervals, 95)
    ib_p95 = _pct(inbound, 95)
    inf_p99 = _pct(infer, 99)
    ob_p95 = _pct(outbound, 95)

    flags = []
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
    series = _timing_series(output_path)
    if all(values.size == 0 for _, _, values, _, _ in series):
        logger.warning("No server timing samples; skipping server timings plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (label, _, values, _, _) in zip(axes.flat, series):
        if values.size == 0:
            ax.set_title(f"{label} (no data)")
            ax.set_axis_off()
            continue
        percentile_histogram(ax, values, label)
    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "server_timings.png")


def generate_server_timings_over_time_plot(output_path: pathlib.Path) -> None:
    series = _timing_series(output_path)
    all_times = np.concatenate([t for _, t, _, _, _ in series])
    if not all_times.size:
        logger.warning("No server timing samples; skipping over-time plot")
        return
    t0 = all_times.min()

    robot_ids = sorted({robot for _, _, _, robots, _ in series if robots for robot in robots})
    colors = robot_colors(robot_ids)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, (label, t, values, robots, batch_sizes) in zip(axes, series):
        if values.size == 0:
            ax.set_title(f"{label} (no data)")
            ax.set_ylabel(label)
            continue
        if batch_sizes is not None:
            sc = ax.scatter(t - t0, values, c=batch_sizes, cmap="viridis", s=6, alpha=0.7)
            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label("batch size", fontsize=8)
        else:
            ax.scatter(
                t - t0, values, c=[colors[robot] for robot in robots], s=4, alpha=0.5, linewidths=0
            )
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, max(float(np.percentile(values, 99.5)) * 1.1, 1.0))
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"n={values.size}  p50={np.percentile(values, 50):.1f}  "
            f"p95={np.percentile(values, 95):.1f}  p99={np.percentile(values, 99):.1f}  "
            f"max={values.max():.1f} (y clipped at p99.5)",
            fontsize=9,
        )

    axes[-1].set_xlabel("time since first sample (s)")
    if robot_ids:
        fig.legend(
            handles=[Patch(facecolor=colors[robot], label=robot) for robot in robot_ids],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(robot_ids), 10),
            fontsize=8,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, output_path / "plots" / "server_timings_over_time.png")


TIMING_PHASES = [
    ("send_ms", "client->server send (ms)"),
    ("queue_ms", "server queue wait (ms)"),
    ("inference_ms", "GPU inference (ms)"),
    ("receive_ms", "server->client receive (ms)"),
]


def generate_request_timing_plot(output_path: pathlib.Path) -> None:
    df = request_timings(output_path)
    if df.empty:
        logger.warning("No per-request server timings; skipping request timing plot")
        return

    robots = sorted(df["robot_id"].unique(), key=int)
    colors = robot_colors(robots)

    fig, axes = plt.subplots(2, 2, figsize=(7 * min(2, len(robots)) + 4, 9))
    fig.suptitle("Per-Request Timing Distributions", fontsize=16, fontweight="bold")
    for ax, (column, label) in zip(axes.flat, TIMING_PHASES):
        samples = [
            (robot, df.loc[df["robot_id"] == robot, column].dropna().to_numpy()) for robot in robots
        ]
        samples = [(robot, values) for robot, values in samples if values.size > 1]
        if not samples:
            ax.set_title(f"{label} (no data)")
            ax.set_axis_off()
            continue

        combined = np.concatenate([values for _, values in samples])
        clip = max(float(np.percentile(combined, 99.5)), 1.0)
        parts = ax.violinplot(
            [np.clip(values, None, clip) for _, values in samples],
            positions=range(len(samples)),
            widths=0.7,
            showmedians=True,
        )
        for body, (robot, _) in zip(parts["bodies"], samples):
            body.set_facecolor(colors[robot])
            body.set_alpha(0.7)

        ax.set_xticks(range(len(samples)))
        ax.set_xticklabels([f"robot_{robot}" for robot, _ in samples], fontsize=8, rotation=45)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, clip * 1.05)
        ax.set_title(
            f"n={combined.size}  p50={np.percentile(combined, 50):.1f}  "
            f"p95={np.percentile(combined, 95):.1f}  p99={np.percentile(combined, 99):.1f}  "
            f"max={combined.max():.1f} (y clipped at p99.5)",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, output_path / "plots" / "request_timings.png")


def generate_batch_size_plot(output_path: pathlib.Path) -> None:
    batches = load_server_batches(output_path)
    if batches.empty:
        logger.warning("No server batches; skipping batch size plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(batches["batch_size"], bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_title("Batch Sizes chosen by Server", fontsize=14, fontweight="bold")
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "batch_size_distribution.png")


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


def generate_server_batch_gantt_plot(output_path: pathlib.Path) -> None:
    batches = load_server_batches(output_path)
    batches = batches[batches["robot_ids"].map(len) > 0] if not batches.empty else batches
    if batches.empty:
        logger.warning("No server batches; skipping server batch Gantt plot")
        return

    decisions = load_scheduler_decisions(output_path)
    t0 = min(
        float(batches["inference_start_time"].min()),
        min((d["started_at"] for d in decisions), default=float("inf")),
    )

    robot_ids = sorted({rid for rids in batches["robot_ids"] for rid in rids})
    robot_y = {rid: i for i, rid in enumerate(robot_ids)}
    colors = robot_colors(robot_ids)
    decision_y = len(robot_ids)

    fig_height = max(3.0, (len(robot_ids) + (1.2 if decisions else 0)) * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for batch in batches.itertuples(index=False):
        for rid in batch.robot_ids:
            ax.barh(
                robot_y[rid],
                batch.inference_duration,
                left=batch.inference_start_time - t0,
                height=0.72,
                color=colors[rid],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
            )

    phase_kinds_seen: set[str] = set()
    seen_phase_calls: set[float] = set()
    batch_start_by_id = dict(
        zip(batches["batch_id"], batches["inference_start_time"].astype(float) - t0)
    )
    for decision in decisions:
        t = decision["started_at"] - t0
        phases = (decision.get("notes") or {}).get("phases")
        if phases and decision["started_at"] not in seen_phase_calls:
            # Multi-batch calls produce one decision per batch but share a
            # single phases list; draw it once.
            seen_phase_calls.add(decision["started_at"])
            for phase in phases:
                start, end = float(phase.get("start", 0.0)) - t0, float(phase.get("end", 0.0)) - t0
                if end <= start:
                    continue
                phase_kinds_seen.add(phase.get("name", ""))
                ax.barh(
                    decision_y,
                    end - start,
                    left=start,
                    height=0.5,
                    color=PHASE_COLORS.get(phase.get("name", ""), "#888888"),
                    edgecolor="black",
                    linewidth=0.6 if phase.get("name") == "search_step" else 0.2,
                    alpha=0.9,
                )
        elif not phases:
            ax.plot(
                [t, t + float(decision.get("duration") or 0.0)],
                [decision_y, decision_y],
                color="seagreen",
                linewidth=2.2,
                alpha=0.85,
                solid_capstyle="butt",
            )

        batch_t = batch_start_by_id.get(decision.get("batch_id"))
        if batch_t is not None:
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
    if decisions:
        yticks.append(decision_y)
        ylabels.append("Decisions")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()

    scheduler_names = sorted({d.get("scheduler_name") or "" for d in decisions} - {""})
    title = f"GPU Gantt ({', '.join(scheduler_names)})" if scheduler_names else "GPU Gantt"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time since server start (s)", fontsize=12)
    ax.set_ylabel("Robot", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    if len(robot_ids) <= 20:
        robot_legend = ax.legend(
            handles=[Patch(facecolor=colors[rid], label=rid) for rid in robot_ids],
            loc="upper right",
            fontsize=8,
            frameon=False,
        )
        ax.add_artist(robot_legend)
    if phase_kinds_seen:
        ax.legend(
            handles=[
                Patch(facecolor=color, label=name)
                for name, color in PHASE_COLORS.items()
                if name in phase_kinds_seen
            ],
            loc="upper left",
            fontsize=8,
            frameon=False,
            title="Scheduler phases",
            title_fontsize=8,
        )

    fig.tight_layout()
    save_fig(fig, output_path / "plots" / "server_batch_gantt.png")
