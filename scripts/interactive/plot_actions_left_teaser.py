"""Paper-teaser figure: actions-left heatmaps for two runs side-by-side.

Builds the per-robot ``actions_left[step, robot]`` matrix for two runs via the
canonical ``_build_actions_left_matrix`` helper, then renders them as a single
horizontal-stack figure with minimal axis chrome and one shared colorbar.

Run:
    uv run python scripts/interactive/plot_actions_left_teaser.py \\
        --run-a data/libero/multi_robot_videos \\
        --run-b data/libero/multi_robot_videos_3 \\
        --label-a "Reactive" --label-b "Lookahead" \\
        --out data/libero/actions_left_teaser.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sims.libero.metrics import _build_actions_left_matrix  # noqa: E402


def _identify_fast_rows(run_dir: pathlib.Path, mat_robots: list[str]) -> list[int]:
    """Return matrix-row indices that correspond to 'fast' robots — those whose
    max_execution_horizon equals the per-run minimum. Returns empty if all
    horizons are equal (homogeneous run, no fast/slow split)."""
    meta_path = run_dir / "experiment_args.json"
    if not meta_path.exists():
        return []
    try:
        ec = json.loads(meta_path.read_text())["experiment_config"]
    except (json.JSONDecodeError, KeyError):
        return []
    horizons = [h["max"] for h in ec.get("execution_horizons", [])]
    if not horizons or len(set(horizons)) <= 1:
        return []
    min_h = min(horizons)
    fast_robot_ids = {str(i) for i, h in enumerate(horizons) if h == min_h}
    return [r for r, rid in enumerate(mat_robots) if rid in fast_robot_ids]


def _build_listed_cmap(
    anchor_hex: str, light_hex: str, dark_hex: str, vmax: int
) -> mcolors.ListedColormap:
    grad = mcolors.LinearSegmentedColormap.from_list(
        f"grad_{anchor_hex}",
        [light_hex, anchor_hex, dark_hex],
    ).resampled(vmax)
    cmap = mcolors.ListedColormap([(1.0, 1.0, 1.0, 0.0)] + [grad(i) for i in range(vmax)])
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def _strip_chrome(ax) -> None:
    """Hide all spines and tick marks/labels — keep only the axis labels
    set via set_xlabel / set_ylabel."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-a", type=pathlib.Path, required=True, help="Run directory (left panel).")
    p.add_argument("--run-b", type=pathlib.Path, required=True, help="Run directory (right panel).")
    p.add_argument(
        "--label-a", type=str, default="", help="Short title above the left panel (default: empty)."
    )
    p.add_argument(
        "--label-b",
        type=str,
        default="",
        help="Short title above the right panel (default: empty).",
    )
    p.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("data/libero/actions_left_teaser.png")
    )
    p.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Override canvas rate (default: derive from data).",
    )
    p.add_argument(
        "--max-robots",
        type=int,
        default=None,
        help="Show only robots with index 0..max_robots-1. Default: show all.",
    )
    p.add_argument(
        "--height", type=float, default=3.2, help="Figure height in inches (default 3.2)."
    )
    p.add_argument(
        "--width-scale",
        type=float,
        default=0.07,
        help="Width in inches per second of run (default 0.07).",
    )
    p.add_argument(
        "--min-width", type=float, default=10.0, help="Lower bound on figure width (default 10)."
    )
    p.add_argument("--transparent", action="store_true", help="Save with a transparent background.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    robots_a, mat_a, _, hz_a, _ = _build_actions_left_matrix(args.run_a, args.control_hz)
    robots_b, mat_b, _, hz_b, _ = _build_actions_left_matrix(args.run_b, args.control_hz)
    if mat_a.size == 0:
        raise SystemExit(f"No actions_left.npy data under {args.run_a}")
    if mat_b.size == 0:
        raise SystemExit(f"No actions_left.npy data under {args.run_b}")

    # Trim trailing columns where coverage drops below half the robots —
    # this drops isolated tail samples (e.g. one robot's final value=0 past
    # a long NaN gap) that would otherwise render as a thin black bar at
    # the right edge.
    def _trim(mat: np.ndarray) -> np.ndarray:
        if mat.size == 0:
            return mat
        threshold = max(1, mat.shape[0] // 2)
        coverage = np.sum(~np.isnan(mat), axis=0)
        valid = coverage >= threshold
        if not valid.any():
            return mat
        last = int(np.flatnonzero(valid)[-1])
        return mat[:, : last + 1]

    mat_a = _trim(mat_a)
    mat_b = _trim(mat_b)

    # Optionally keep only the first N robots (by robot_idx 0..N-1).
    if args.max_robots is not None and args.max_robots > 0:
        keep = {str(i) for i in range(args.max_robots)}
        keep_a = [i for i, rid in enumerate(robots_a) if rid in keep]
        keep_b = [i for i, rid in enumerate(robots_b) if rid in keep]
        mat_a, robots_a = mat_a[keep_a, :], [robots_a[i] for i in keep_a]
        mat_b, robots_b = mat_b[keep_b, :], [robots_b[i] for i in keep_b]

    # Shared color scale across both panels for direct visual comparison.
    vmax_a = int(np.nanmax(mat_a)) if not np.all(np.isnan(mat_a)) else 1
    vmax_b = int(np.nanmax(mat_b)) if not np.all(np.isnan(mat_b)) else 1
    vmax = max(1, max(vmax_a, vmax_b))
    # Two gradients sharing the same scale: blue for slow robots, amber for
    # the fast robot (identified per-run via runtime_metadata.json). 0
    # (starvation) renders as white in both.
    # Match the throughput plot: slow tier gray (#777B7F), fast tier blue (#37A3D2).
    cmap_slow = _build_listed_cmap("#777B7F", "#D8DADC", "#3A3D40", vmax)
    cmap_fast = _build_listed_cmap("#37A3D2", "#D9EEF7", "#0E3B4F", vmax)

    fast_rows_a = _identify_fast_rows(args.run_a, robots_a)
    fast_rows_b = _identify_fast_rows(args.run_b, robots_b)

    # Panel widths proportional to run duration so a second of wall-clock
    # occupies the same horizontal distance on both panels.
    sec_a = mat_a.shape[1] / hz_a
    sec_b = mat_b.shape[1] / hz_b
    fig_width = max(args.min_width, args.width_scale * (sec_a + sec_b) + 1.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_width, args.height),
        gridspec_kw={"width_ratios": [sec_a, sec_b]},
        sharey=True,
    )
    fig.set_facecolor("white")

    im_slow = None
    im_fast = None
    for idx, (ax, mat, robots, fast_rows, label, hz_, secs) in enumerate(
        [
            (axes[0], mat_a, robots_a, fast_rows_a, args.label_a, hz_a, sec_a),
            (axes[1], mat_b, robots_b, fast_rows_b, args.label_b, hz_b, sec_b),
        ]
    ):
        # Split into slow (default) and fast (anchored) views by masking.
        mat_slow = mat.copy()
        for r in fast_rows:
            mat_slow[r, :] = np.nan
        im_slow = ax.imshow(
            mat_slow,
            aspect="auto",
            cmap=cmap_slow,
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=vmax,
        )
        if fast_rows:
            mat_fast = np.full_like(mat, np.nan)
            for r in fast_rows:
                mat_fast[r, :] = mat[r, :]
            im_fast = ax.imshow(
                mat_fast,
                aspect="auto",
                cmap=cmap_fast,
                interpolation="nearest",
                origin="lower",
                vmin=0,
                vmax=vmax,
            )

        _strip_chrome(ax)

    # Two colorbars on the right (slow / blue, fast / red). Both share the
    # same 0..vmax scale; ticks every 5. Compact labels with a shared
    # "Actions Left" header above so the side labels don't collide.
    cbar_ticks = sorted({0, vmax, *range(0, vmax + 1, 5)})

    def _minimal_cbar(cb, show_ticklabels: bool):
        cb.set_ticks(cbar_ticks)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=10, length=0)
        if show_ticklabels:
            cb.set_ticklabels([str(int(t)) for t in cbar_ticks])
        else:
            cb.set_ticklabels([])

    if im_fast is not None:
        # Two manually positioned colorbars on the right with a real gap.
        fig.subplots_adjust(right=0.86)
        cax_slow = fig.add_axes([0.88, 0.18, 0.012, 0.70])
        cax_fast = fig.add_axes([0.93, 0.18, 0.012, 0.70])
        cbar_slow = fig.colorbar(im_slow, cax=cax_slow)
        cbar_fast = fig.colorbar(im_fast, cax=cax_fast)
        _minimal_cbar(cbar_slow, show_ticklabels=True)
        _minimal_cbar(cbar_fast, show_ticklabels=True)
    else:
        cbar_slow = fig.colorbar(im_slow, ax=axes, fraction=0.018, pad=0.02)
        _minimal_cbar(cbar_slow, show_ticklabels=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.transparent:
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.set_facecolor("none")
        fig.savefig(args.out, dpi=200, transparent=True, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), transparent=True, bbox_inches="tight")
    else:
        fig.savefig(args.out, dpi=200, facecolor="white", bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"Wrote {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
