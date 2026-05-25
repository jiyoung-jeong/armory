"""Real-world throughput bar plots, matching the format of ``plot_summary_bars.py``.

Reads the three human-graded CSVs under ``/coc/flash7/rbansal66/vvla/data/new_real/``::

    hom.csv         all 10 robots have max_exec_horizon == 10 (one tier)
    1f9s.csv        1 fast (max_ex=10), 9 slow (max_ex=20)
    5f5s.csv        5 fast, 5 slow

Each CSV stacks several mini-tables, two side-by-side: a left scheduler
(round_robin or max_batch) and a right scheduler (lookahead_w1 or
lookahead_w5). Each mini-table is one trial of ~10 robots; the per-robot value
recorded is "Legos Completed in 60 s" (i.e. legos per minute throughput).

Robots labelled ``robot N (fast)`` belong to the fast tier; others are slow.
For ``hom`` (homogeneous workload), all robots are treated as the fast tier —
no (fast) markers appear in the CSV.

Output (PNG bar charts), into ``<output_dir>/``::

    throughput__hom__real.png       single bar per scheduler  (fast only)
    throughput__1f9s__real.png      paired fast / slow bars per scheduler
    throughput__5f5s__real.png      same as 1f9s

Starvation rate columns exist in the CSVs but are empty (human grading was
throughput-only), so no starvation plots are emitted from this script.

Run:
    uv run python scripts/interactive/plot_real_throughput.py
    uv run python scripts/interactive/plot_real_throughput.py --data-dir <DIR> --output-dir <OUT>
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Scheduler labels in the CSVs -> canonical labels used in the sim plots.
SCHEDULER_RENAME = {
    "round_robin": "round-robin",
    "max_batch": "max-batch",
    "lookahead_w1": "lookahead-actions@ahm=1",
    "lookahead_w5": "lookahead-actions@ahm=5",
}
KNOWN_RAW_SCHEDULERS = set(SCHEDULER_RENAME)

# Same ordering as the sim plots so the side-by-side comparison is direct.
PREFERRED_SCHEDULER_ORDER = [
    "max-batch",
    "round-robin",
    "lookahead-actions@ahm=1",
    "lookahead-actions@ahm=3",
    "lookahead-actions@ahm=5",
]

DEFAULT_DATA_DIR = pathlib.Path("/coc/flash7/rbansal66/vvla/data/new_real")
SCENARIOS = ("hom", "1f9s", "5f5s")
TWO_TIER_SCENARIOS = ("1f9s", "5f5s")
HOM_SCENARIO = "hom"

ROBOT_RE = re.compile(r"^robot\s+(\d+)(?:\s+\((fast)\))?\s*$", re.IGNORECASE)

# --- parsing -------------------------------------------------------------------


@dataclass
class Sample:
    scenario: str
    scheduler: str          # canonical (max-batch / round-robin / lookahead-actions@ahm=W)
    trial_idx: int          # 0-based within (scenario, scheduler)
    robot_id: int
    is_fast: bool
    legos_per_minute: float


def _parse_csv(scenario: str, path: pathlib.Path) -> list[Sample]:
    """Walk the CSV, track which scheduler each side is currently under, and
    pull ``robot N -> legos_60s`` from cols (1,3) for the left side and (7,9)
    for the right side.

    ``trial_idx`` increments per (scenario, scheduler) every time a new header
    row appears for that scheduler on either side.
    """
    samples: list[Sample] = []
    trial_counter: dict[str, int] = defaultdict(int)  # scheduler -> trial count seen
    left_sched: str | None = None
    right_sched: str | None = None
    left_trial: int | None = None
    right_trial: int | None = None

    with path.open() as f:
        rows = list(csv.reader(f))

    for r in rows:
        # pad row to expected width
        if len(r) < 12:
            r = r + [""] * (12 - len(r))

        # New scheduler header on the LEFT side?
        if r[1] in KNOWN_RAW_SCHEDULERS:
            left_sched = SCHEDULER_RENAME[r[1]]
            left_trial = trial_counter[left_sched]
            trial_counter[left_sched] += 1
        # New scheduler header on the RIGHT side?
        if r[7] in KNOWN_RAW_SCHEDULERS:
            right_sched = SCHEDULER_RENAME[r[7]]
            right_trial = trial_counter[right_sched]
            trial_counter[right_sched] += 1

        # Robot row on the LEFT side?
        m = ROBOT_RE.match(r[1])
        if m and left_sched is not None:
            legos = _parse_lego_count(r[3])
            if legos is not None:
                samples.append(Sample(
                    scenario=scenario,
                    scheduler=left_sched,
                    trial_idx=left_trial or 0,
                    robot_id=int(m.group(1)),
                    is_fast=(scenario == HOM_SCENARIO) or (m.group(2) is not None),
                    legos_per_minute=legos,
                ))
        # Robot row on the RIGHT side?
        m = ROBOT_RE.match(r[7])
        if m and right_sched is not None:
            legos = _parse_lego_count(r[9])
            if legos is not None:
                samples.append(Sample(
                    scenario=scenario,
                    scheduler=right_sched,
                    trial_idx=right_trial or 0,
                    robot_id=int(m.group(1)),
                    is_fast=(scenario == HOM_SCENARIO) or (m.group(2) is not None),
                    legos_per_minute=legos,
                ))
    return samples


def _parse_lego_count(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# --- aggregation ---------------------------------------------------------------


def _aggregate_tier(
    samples: list[Sample], scenario: str, scheduler: str, tier: str
) -> tuple[float, int] | tuple[None, int]:
    """tier = 'fast' or 'slow'. Returns (mean_legos_per_minute, n)."""
    want_fast = tier == "fast"
    vals = [
        s.legos_per_minute
        for s in samples
        if s.scenario == scenario and s.scheduler == scheduler and s.is_fast == want_fast
    ]
    if not vals:
        return (None, 0)
    return (sum(vals) / len(vals), len(vals))


def _ordered_schedulers(observed: list[str]) -> list[str]:
    head = [s for s in PREFERRED_SCHEDULER_ORDER if s in observed]
    rest = sorted(s for s in observed if s not in PREFERRED_SCHEDULER_ORDER)
    return head + rest


# --- plotting (mirrors plot_summary_bars.py style) -----------------------------


def _annotate(ax, x: float, y: float, val: float | None) -> None:
    if val is None:
        return
    ax.text(x, y, f"{val:.2f}", ha="center", va="bottom", fontsize=8)


def _plot_single_bar(
    out_path: pathlib.Path,
    schedulers: list[str],
    values: list[float | None],
    title: str,
    ylabel: str,
    color: str = "#1f77b4",
) -> None:
    keep = [(s, v) for s, v in zip(schedulers, values) if v is not None]
    if not keep:
        return
    schedulers_k, values_k = zip(*keep)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(schedulers_k) + 3), 4.5))
    xs = np.arange(len(schedulers_k))
    bars = ax.bar(xs, values_k, color=color, edgecolor="black", linewidth=0.6)
    for x, b, v in zip(xs, bars, values_k):
        _annotate(ax, x, b.get_height(), v)
    ax.set_xticks(xs)
    ax.set_xticklabels(schedulers_k, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_grouped_bars(
    out_path: pathlib.Path,
    schedulers: list[str],
    series: list[tuple[str, list[float | None], str]],
    title: str,
    ylabel: str,
) -> None:
    keep_idx = [
        i for i in range(len(schedulers))
        if any(s[1][i] is not None for s in series)
    ]
    if not keep_idx:
        return
    schedulers_k = [schedulers[i] for i in keep_idx]
    series_k = [(lab, [vs[i] for i in keep_idx], col) for lab, vs, col in series]

    n_series = len(series_k)
    width = 0.8 / n_series
    xs = np.arange(len(schedulers_k))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(schedulers_k) + 3), 4.8))
    for s_idx, (label, vals, color) in enumerate(series_k):
        offset = (s_idx - (n_series - 1) / 2) * width
        plot_vals = [0.0 if v is None else v for v in vals]
        bars = ax.bar(
            xs + offset, plot_vals, width=width,
            label=label, color=color, edgecolor="black", linewidth=0.6,
        )
        for x, b, raw in zip(xs + offset, bars, vals):
            _annotate(ax, x, b.get_height(), raw)
    ax.set_xticks(xs)
    ax.set_xticklabels(schedulers_k, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- driver --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR,
                   help=f"Directory with hom.csv, 1f9s.csv, 5f5s.csv (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--output-dir", type=pathlib.Path, default=None,
                   help="Output directory for PNGs (default: <data-dir>/plots).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir.resolve()
    output_dir: pathlib.Path = (args.output_dir or (data_dir / "plots")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[Sample] = []
    for scenario in SCENARIOS:
        path = data_dir / f"{scenario}.csv"
        if not path.is_file():
            print(f"WARN: {path} missing; skipping")
            continue
        all_samples.extend(_parse_csv(scenario, path))

    if not all_samples:
        raise SystemExit(f"No samples parsed from {data_dir}")

    observed_schedulers = sorted({s.scheduler for s in all_samples})
    schedulers = _ordered_schedulers(observed_schedulers)

    # Summary to stdout so the user can sanity-check the counts.
    print(f"Parsed {len(all_samples)} samples ({len(observed_schedulers)} schedulers).")
    by_scenario_scheduler: dict[tuple[str, str], int] = defaultdict(int)
    for s in all_samples:
        by_scenario_scheduler[(s.scenario, s.scheduler)] += 1
    for k in sorted(by_scenario_scheduler):
        print(f"  {k[0]:>4s}  {k[1]:<28s}  n_robot_samples={by_scenario_scheduler[k]}")

    # Plot 1: hom throughput (fast tier only).
    fast_vals = [_aggregate_tier(all_samples, HOM_SCENARIO, s, "fast")[0] for s in schedulers]
    _plot_single_bar(
        output_dir / "throughput__hom__real.png",
        schedulers, fast_vals,
        title="hom throughput (real)",
        ylabel="Throughput (legos / minute)",
        color="#1f77b4",
    )

    # Plots 2-3: 1f9s / 5f5s grouped fast vs slow.
    for scenario in TWO_TIER_SCENARIOS:
        fast = [_aggregate_tier(all_samples, scenario, s, "fast")[0] for s in schedulers]
        slow = [_aggregate_tier(all_samples, scenario, s, "slow")[0] for s in schedulers]
        _plot_grouped_bars(
            output_dir / f"throughput__{scenario}__real.png",
            schedulers,
            series=[
                ("fast tier", fast, "#1f77b4"),
                ("slow tier", slow, "#ff7f0e"),
            ],
            title=f"{scenario} throughput (real)",
            ylabel="Throughput (legos / minute)",
        )

    print(f"Wrote plots under {output_dir}")


if __name__ == "__main__":
    main()
