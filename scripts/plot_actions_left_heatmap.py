"""Regenerate the actions-left heatmap(s) with a white->blue color scheme.

Same data pipeline as ``sims.libero.metrics.generate_actions_left_heatmap``
(per-episode ``actions_left.npy`` aligned onto a shared wall-clock grid via
``_build_actions_left_matrix``), with these differences:

* Colormap is a continuous white->blue ramp (#2166AC): white = empty queue
  (0 actions left / starvation), blue = full queue.
* Robot 11 is pinned to the top row and drawn with a white->red ramp (#D6604D)
  with its own (shorter-horizon) vmax.
* Remaining robots are sorted below robot 11 by whichever first executed an
  action earliest (first control step with a non-empty queue), and rows are
  renumbered 0..N as bare integer labels.
* The time axis is capped to ``--num-steps`` control steps (default 150).
* No episode-boundary annotations.

Pass one trial directory for a single heatmap, or two to render them side by
side for comparison (titled "Earliest Deadline First" and "Lookahead" by
default). When comparing, both panels share a common color scale.

The plot is written to the current working directory.

Usage:
    uv run python scripts/plot_actions_left_heatmap.py <trial_dir>

    # side-by-side comparison (first dir = left panel):
    uv run python scripts/plot_actions_left_heatmap.py <edf_dir> <lookahead_dir>

    uv run python scripts/plot_actions_left_heatmap.py <dir> \
        --num-steps 300 --control-hz 20 -o my_heatmap.png

    uv run python scripts/plot_actions_left_heatmap.py     /coc/flash7/rbansal66/vvla/data/new_real/1f9s/max_batch/trial_20260519_021326     /coc/flash7/rbansal66/vvla/data/new_real/1f9s/lookahead_w5/trial_20260524_215847
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from evaluation.sims.libero.metrics import (
    _build_actions_left_matrix,
    _load_scheduler_decisions,
    _server_to_perf_offset,
)

PINNED_ROBOT = "11"  # special short-horizon robot: top row, red ramp.
COLORS = [
    "#247BA0",  # blue
    "#F95738",  # red-orange
    "#F7B801",  # yellow
]


def _white_to(color: str) -> mcolors.Colormap:
    """White (value 0) -> ``color`` (value vmax) ramp, transparent for NaN."""
    cmap = mcolors.LinearSegmentedColormap.from_list("white_to", ["#FFFFFF", color]).copy()
    cmap.set_bad(alpha=0.0)
    return cmap


@dataclass
class HeatmapData:
    """Row-ordered, column-capped actions-left matrix ready to render."""

    ordered: np.ndarray  # [n_robots, max_len], origin="lower" (row 0 = bottom)
    labels: list[str]  # y-tick labels, in array-row order
    arr_row_of_robot: dict[str, int]  # robot id -> array row
    pinned_row: int | None  # array row of robot 11, if present
    control_hz: float
    t0_perf: float
    max_len: int


def build_heatmap_data(
    output_path: pathlib.Path,
    num_steps: int = 150,
    control_hz: float | None = None,
) -> HeatmapData | None:
    robots, matrix, _eb, control_hz, t0_perf = _build_actions_left_matrix(output_path, control_hz)
    if matrix.size == 0:
        return None

    matrix = matrix[:, :num_steps]
    max_len = matrix.shape[1]

    # Sort key: the first control step at which the robot has a non-empty queue
    # (queue > 0), i.e. the earliest step it actually had an action to execute.
    # A finite 0 means "observed but starved", which is NOT an executed action.
    has_action = matrix > 0
    first_action = np.array(
        [int(np.argmax(h)) if h.any() else np.inf for h in has_action], dtype=float
    )
    row_of_robot = {rid: i for i, rid in enumerate(robots)}

    others = [r for r in robots if r != PINNED_ROBOT]
    others_sorted = sorted(others, key=lambda r: first_action[row_of_robot[r]])

    # Top-to-bottom display order: robot 11 first, then others (earliest first).
    top_to_bottom = ([PINNED_ROBOT] if PINNED_ROBOT in robots else []) + others_sorted
    labels_ttb = (["Fast"] if PINNED_ROBOT in robots else []) + ["Slow" for _ in others_sorted]

    # imshow uses origin="lower" (array row 0 = bottom), so reverse.
    array_ids = top_to_bottom[::-1]
    labels = labels_ttb[::-1]
    ordered = np.vstack([matrix[row_of_robot[rid]] for rid in array_ids])
    arr_row_of_robot = {rid: i for i, rid in enumerate(array_ids)}
    pinned_row = arr_row_of_robot.get(PINNED_ROBOT)

    return HeatmapData(
        ordered=ordered,
        labels=labels,
        arr_row_of_robot=arr_row_of_robot,
        pinned_row=pinned_row,
        control_hz=control_hz,
        t0_perf=t0_perf,
        max_len=max_len,
    )


def draw_heatmap(
    ax: plt.Axes,
    data: HeatmapData,
    output_path: pathlib.Path,
    vmax_blue: int,
    vmax_red: int,
    title: str,
    show_ylabel: bool = True,
) -> tuple[matplotlib.image.AxesImage, matplotlib.image.AxesImage | None]:
    main = data.ordered.copy()
    if data.pinned_row is not None:
        main[data.pinned_row, :] = np.nan
    im_blue = ax.imshow(
        main,
        aspect="auto",
        cmap=_white_to(COLORS[0]),
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=vmax_blue,
    )

    im_red = None
    if data.pinned_row is not None:
        red_mat = np.full_like(data.ordered, np.nan)
        red_mat[data.pinned_row] = data.ordered[data.pinned_row]
        im_red = ax.imshow(
            red_mat,
            aspect="auto",
            cmap=_white_to(COLORS[1]),
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=vmax_red,
        )

    # Scheduler-decision overlay (black ticks for dispatched batches).
    decisions = _load_scheduler_decisions(output_path)
    offset = _server_to_perf_offset(output_path) if decisions else None
    if decisions and offset is not None:
        for d in decisions:
            started_at = d.get("started_at")
            if started_at is None:
                continue
            col = (float(started_at) - offset - data.t0_perf) * data.control_hz
            if col < -0.5 or col > data.max_len - 0.5:
                continue
            scheduled = d.get("scheduled") or []
            if scheduled and d.get("batch_id") is not None:
                for rid in scheduled:
                    row = data.arr_row_of_robot.get(str(rid))
                    if row is None:
                        continue
                    ax.plot(
                        [col, col],
                        [row - 0.42, row + 0.42],
                        color="black",
                        linewidth=0.7,
                        alpha=0.85,
                    )

    ax.set_yticks([])
    tick_interval = max(1, int(round(data.control_hz)))  # one tick per second
    x_ticks = np.arange(0, data.max_len, tick_interval)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t // tick_interval}" for t in x_ticks], fontsize=15)
    ax.set_xlabel("Time (seconds)", fontweight="bold", fontsize=18)
    if show_ylabel:
        ax.set_ylabel("Robots", fontweight="bold", fontsize=18)
    ax.set_title(title, fontsize=22, fontweight="bold")
    return im_blue, im_red


def generate(
    trial_dirs: list[pathlib.Path],
    titles: list[str],
    save_to: pathlib.Path,
    num_steps: int = 150,
    control_hz: float | None = None,
) -> None:
    datasets = [
        (d, t, build_heatmap_data(d, num_steps, control_hz)) for d, t in zip(trial_dirs, titles)
    ]
    datasets = [(d, t, data) for d, t, data in datasets if data is not None]
    if not datasets:
        print("No actions_left.npy data found in any trial directory")
        return

    # Shared color scale across panels so the comparison is fair.
    def _vmax(use_pinned: bool) -> int:
        vals = []
        for _, _, data in datasets:
            m = data.ordered
            if data.pinned_row is None:
                sel = m
            elif use_pinned:
                sel = m[data.pinned_row]
            else:
                sel = np.delete(m, data.pinned_row, axis=0)
            if sel.size and not np.all(np.isnan(sel)):
                vals.append(int(np.nanmax(sel)))
        return max(1, max(vals)) if vals else 1

    vmax_blue = _vmax(use_pinned=False)
    vmax_red = _vmax(use_pinned=True)

    n = len(datasets)
    max_robots = max(data.ordered.shape[0] for _, _, data in datasets)
    fig, axes = plt.subplots(
        1, n, figsize=(8 * n + 2, max(4, max_robots * 0.6)), squeeze=False, layout="constrained"
    )
    axes = axes[0]

    im_blue = im_red = None
    for i, (d, t, data) in enumerate(datasets):
        im_blue, im_red = draw_heatmap(
            axes[i], data, d, vmax_blue, vmax_red, t, show_ylabel=(i == 0)
        )

    cb_blue = fig.colorbar(im_blue, ax=list(axes), pad=0.01, fraction=0.03)
    cb_blue.set_label("Remaining Actions in Queue", fontweight="bold", fontsize=18)
    cb_blue.ax.tick_params(labelsize=15)
    if im_red is not None:
        cb_red = fig.colorbar(im_red, ax=list(axes), pad=0.01, fraction=0.03)
        cb_red.ax.tick_params(labelsize=15)

    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_to.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trial_dirs",
        type=pathlib.Path,
        nargs="+",
        help="One trial dir, or two to render side by side (first = left panel).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output PNG path (default chosen from the trial name(s)).",
    )
    parser.add_argument(
        "--titles",
        type=str,
        default=None,
        help="Comma-separated panel titles. Default: 'Earliest Deadline First,Lookahead'.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=60,
        help="Cap the time axis to this many control steps (default: 150).",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Canvas control rate. Defaults to the max observed per-robot rate.",
    )
    args = parser.parse_args()

    dirs = args.trial_dirs
    if args.titles is not None:
        titles = [t.strip() for t in args.titles.split(",")]
    elif len(dirs) == 2:
        titles = ["Earliest Deadline First", "Lookahead"]
    else:
        titles = ["Actions Left Per Robot Over Time"] * len(dirs)

    if args.output is not None:
        save_to = args.output
    elif len(dirs) == 1:
        save_to = pathlib.Path.cwd() / f"{dirs[0].name}_actions_left_heatmap.png"
    else:
        save_to = pathlib.Path.cwd() / "comparison_actions_left_heatmap.png"

    generate(dirs, titles, save_to, args.num_steps, args.control_hz)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
