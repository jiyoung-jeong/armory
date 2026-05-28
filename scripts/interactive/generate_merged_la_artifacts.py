"""Generate one combined system-throughput+starvation table and the overview /
tier-breakdown plots from a *merged* view of two summary dirs:

  * lookahead-actions scheduler columns come from ``--la-dir`` (e.g. mbs=5)
  * all other scheduler columns (max-batch, round-robin, ...) come from
    ``--base-dir`` (e.g. mbs=3)

Both directories must contain a ``results_max_batch_size=<N>.csv`` produced by
``summarize_sweep_runs.py``. The merge is column-level: rows (num_robots) are
taken from the base CSV; lookahead columns are reindexed onto that row set.

Outputs (under ``--out-dir``, default ``_summary_merged_la_only``):
    latex/system_combined_mbs<base_mbs>.tex
    plots/overview_lines__mbs<base_mbs>.{png,pdf}
    plots/tier_breakdown__1f9s__mbs<base_mbs>.{png,pdf}
    plots/tier_breakdown__5f5s__mbs<base_mbs>.{png,pdf}

Run:
    uv run python scripts/interactive/generate_merged_la_artifacts.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_latex_tables import (
    _build_system_combined_table,
    _parse_columns as _parse_table_cols,
)
from plot_summary_lines import (
    TIERED_SCENARIOS,
    _ordered_schedulers,
    _parse_columns as _parse_plot_cols,
    _plot_overview_stack,
    _plot_tier_breakdown,
    _plot_tier_breakdown_combined,
)

LA_COL_RE = re.compile(r"__lookahead-actions@ahm=")

DEFAULT_LA_DIR = pathlib.Path(
    "experiments/sweeps/interactive/_summary_new_5min"
)
DEFAULT_BASE_DIR = pathlib.Path(
    "experiments/sweeps/interactive/_summary_new_5min_b3"
)
DEFAULT_OUT_DIR = pathlib.Path(
    "experiments/sweeps/interactive/_summary_merged_la_only"
)


def merge_summaries(la_csv: pathlib.Path, base_csv: pathlib.Path) -> pd.DataFrame:
    """Use base_csv as the row backbone; replace its lookahead columns with
    the corresponding ones from la_csv (reindexed onto base row set)."""
    la_df = pd.read_csv(la_csv, index_col="num_robots")
    base_df = pd.read_csv(base_csv, index_col="num_robots")

    non_la_base_cols = [c for c in base_df.columns if not LA_COL_RE.search(c)]
    la_cols = [c for c in la_df.columns if LA_COL_RE.search(c)]

    la_block = la_df[la_cols].reindex(base_df.index)
    merged = pd.concat([base_df[non_la_base_cols], la_block], axis=1).copy()
    merged.index.name = base_df.index.name
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--la-dir", type=pathlib.Path, default=DEFAULT_LA_DIR,
                   help="Summary dir whose lookahead-actions columns we'll keep.")
    p.add_argument("--base-dir", type=pathlib.Path, default=DEFAULT_BASE_DIR,
                   help="Summary dir we'll take everything *except* lookahead from.")
    p.add_argument("--la-mbs", type=int, default=5,
                   help="max_batch_size value to pick the lookahead CSV (default 5).")
    p.add_argument("--base-mbs", type=int, default=3,
                   help="max_batch_size value to pick the base CSV (default 3); also "
                        "stamped into output filenames and the table caption.")
    p.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--num-robots", type=str, default=None,
                   help="Comma-separated list of num_robots to include.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    la_csv = (args.la_dir / f"results_max_batch_size={args.la_mbs}.csv").resolve()
    base_csv = (args.base_dir / f"results_max_batch_size={args.base_mbs}.csv").resolve()
    for path in (la_csv, base_csv):
        if not path.is_file():
            raise SystemExit(f"Missing CSV: {path}")

    nr_filter: set[int] | None = None
    if args.num_robots:
        nr_filter = {int(x.strip()) for x in args.num_robots.split(",") if x.strip()}

    merged = merge_summaries(la_csv, base_csv)

    out_dir = args.out_dir.resolve()
    latex_dir = out_dir / "latex"
    plots_dir = out_dir / "plots"
    latex_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_csv(
        out_dir / f"merged_results_max_batch_size={args.base_mbs}.csv",
        index=False,
    )

    # --- LaTeX combined table ---
    table_cols = _parse_table_cols(merged)
    combined = _build_system_combined_table(
        merged, table_cols, args.base_mbs,
        starv_decimals=1,
        starv_scale=100.0,
        thr_decimals=2,
        thr_scale=60.0,
        thr_unit_label="successes/min",
        num_robots_filter=nr_filter,
    )
    if not combined:
        raise SystemExit("Combined table came back empty — check that schedulers/scenarios exist in both CSVs.")
    table_path = latex_dir / f"system_combined_mbs{args.base_mbs}.tex"
    table_path.write_text(combined)

    # --- Plots ---
    plot_cols = _parse_plot_cols(merged)
    schedulers = _ordered_schedulers(sorted({sch for _, sch in plot_cols}))

    overview_path = plots_dir / f"overview_lines__mbs{args.base_mbs}.png"
    _plot_overview_stack(
        overview_path, merged, plot_cols, schedulers, nr_filter=nr_filter,
    )

    tier_paths: list[pathlib.Path] = []
    tiered_present = [sc for sc in TIERED_SCENARIOS if any(sc == k[0] for k in plot_cols)]
    for scenario in tiered_present:
        path = plots_dir / f"tier_breakdown__{scenario}__mbs{args.base_mbs}.png"
        _plot_tier_breakdown(
            path, merged, plot_cols, schedulers, scenario, nr_filter=nr_filter,
        )
        tier_paths.append(path)
    if len(tiered_present) >= 2:
        combined_path = plots_dir / f"tier_breakdown_combined__mbs{args.base_mbs}.png"
        _plot_tier_breakdown_combined(
            combined_path, merged, plot_cols, schedulers, tiered_present, nr_filter=nr_filter,
        )
        tier_paths.append(combined_path)

    print(f"Lookahead source : {la_csv}")
    print(f"Base source      : {base_csv}")
    print(f"Wrote table      : {table_path}")
    print(f"Wrote overview   : {overview_path} (+ .pdf)")
    for p in tier_paths:
        print(f"Wrote tier       : {p} (+ .pdf)")


if __name__ == "__main__":
    main()
