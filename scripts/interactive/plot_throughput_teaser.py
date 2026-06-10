"""Paper-teaser bar chart for a two-scheduler throughput comparison.

Static data: EDF=83, LA@5=99. Renders a clean two-bar figure with tick marks
preserved but tick labels stripped (user adds annotations in post). The
y-axis is clipped above zero so the bars don't visually exaggerate the
difference.

Run:
    uv run python scripts/interactive/plot_throughput_teaser.py
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt

BAR_COLORS = {
    "RR":   "#F8961E",
    "EDF":  "#F8961E",
    "LA@5": "#90BE6D",
}
DEFAULT_BAR_COLOR = "#F8961E"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("data/libero/throughput_teaser.png"))
    p.add_argument("--ymin", type=float, default=75.0,
                   help="Y-axis lower bound (default 75).")
    p.add_argument("--ymax", type=float, default=105.0,
                   help="Y-axis upper bound (default 105).")
    p.add_argument("--bar-width", type=float, default=0.5)
    p.add_argument("--figsize", type=float, nargs=2, default=(3.2, 3.6))
    p.add_argument("--transparent", action="store_true",
                   help="Save with a transparent background.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labels = ["RR", "EDF", "LA@5"]
    values = [83.0, 83.5, 99.0]

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.set_facecolor("white")

    xs = list(range(len(labels)))
    colors = [BAR_COLORS.get(l, DEFAULT_BAR_COLOR) for l in labels]
    ax.bar(xs, values, width=args.bar_width, color=colors, edgecolor="none")

    ax.set_xticks(xs)
    ax.set_xticklabels([])  # x labels added in post
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(args.ymin, args.ymax)

    # Keep tick marks visible on both axes; remove top/right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_linewidth(1.0)
    ax.tick_params(axis="both", length=4, width=1.0, direction="out")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        fig.savefig(args.out, dpi=200, transparent=True, bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), transparent=True, bbox_inches="tight")
    else:
        fig.savefig(args.out, dpi=200, facecolor="white", bbox_inches="tight")
        fig.savefig(args.out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"Wrote {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
