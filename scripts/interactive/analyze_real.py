"""End-to-end real-world analysis: throughput + starvation plots and LaTeX tables.

For each scenario (``hom``, ``1f9s``, ``5f5s``) this script:

  1. Parses the human-graded ``<scenario>.csv`` for per-(trial, robot) legos/min
     throughput (same parser as ``plot_real_throughput.py``). The fast/slow
     tier assignment per robot_id is derived from the ``(fast)`` markers in
     the CSV (``hom`` has no markers — all robots treated as fast tier).
  2. Walks the trial directories
     ``<data-dir>/<scenario>/<raw_scheduler>/trial_*/results.csv``
     for per-(trial, robot) ``starvation_steps / observed_steps`` — the same
     definition used by the sim's per-trial ``results.csv``.
  3. Aggregates both metrics per (scenario, scheduler, tier) and emits:
        - Throughput PNGs   throughput__<scenario>__real.png
        - Starvation PNGs   starvation__<scenario>__real.png
        - LaTeX table       <data-dir>/tables.tex  (one tblr per scenario)

LaTeX table format (one per scenario, written to ``<data-dir>/latex/real_<scenario>.tex``):
    rows    = schedulers (MB, RR, LA $w{=}1$, [LA $w{=}5$])
    columns = thr_fast | thr_slow | thr_system | starv_avg | starv_fast | starv_slow
    cells   = mean across (trial × robot in tier)
    best per column bolded (max for throughput, min for starvation)

Run:
    uv run python scripts/interactive/analyze_real.py
    uv run python scripts/interactive/analyze_real.py --data-dir <DIR> --output-dir <OUT>
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# -------- shared constants ---------------------------------------------------

SCHEDULER_RENAME = {
    "round_robin": "round-robin",
    "max_batch": "max-batch",
    "lookahead_w1": "lookahead-actions@ahm=1",
    "lookahead_w5": "lookahead-actions@ahm=5",
}
KNOWN_RAW_SCHEDULERS = set(SCHEDULER_RENAME)
RAW_NAME_BY_CANONICAL = {v: k for k, v in SCHEDULER_RENAME.items()}

PREFERRED_SCHEDULER_ORDER = [
    "max-batch",
    "round-robin",
    "lookahead-actions@ahm=1",
    "lookahead-actions@ahm=3",
    "lookahead-actions@ahm=5",
]
SCHEDULER_DISPLAY = {
    "max-batch": "MB",
    "round-robin": "RR",
}
LA_AHM_RE = re.compile(r"^lookahead-actions@ahm=(\d+(?:\.\d+)?)$")

DEFAULT_DATA_DIR = pathlib.Path("/coc/flash7/rbansal66/vvla/data/new_real")
SCENARIOS = ("hom", "1f9s", "5f5s")
TWO_TIER_SCENARIOS = ("1f9s", "5f5s")
HOM_SCENARIO = "hom"

ROBOT_RE = re.compile(r"^robot\s+(\d+)(?:\s+\((fast)\))?\s*$", re.IGNORECASE)

# -------- CSV parsing (throughput + tier assignment) -------------------------


@dataclass
class ThrSample:
    scenario: str
    scheduler: str
    trial_idx: int
    robot_id: int
    is_fast: bool
    legos_per_minute: float


@dataclass
class StarvSample:
    scenario: str
    scheduler: str
    trial_dir: str
    robot_id: int
    is_fast: bool
    starv_rate: float


def _parse_lego_count(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_throughput_csv(scenario: str, path: pathlib.Path) -> tuple[list[ThrSample], dict[int, bool]]:
    """Returns (samples, fast_robot_map). fast_robot_map[robot_id] = True if fast."""
    samples: list[ThrSample] = []
    fast_map: dict[int, bool] = {}
    trial_counter: dict[str, int] = defaultdict(int)
    left_sched: str | None = None
    right_sched: str | None = None
    left_trial: int | None = None
    right_trial: int | None = None

    with path.open() as f:
        rows = list(csv.reader(f))

    for r in rows:
        if len(r) < 12:
            r = r + [""] * (12 - len(r))

        if r[1] in KNOWN_RAW_SCHEDULERS:
            left_sched = SCHEDULER_RENAME[r[1]]
            left_trial = trial_counter[left_sched]
            trial_counter[left_sched] += 1
        if r[7] in KNOWN_RAW_SCHEDULERS:
            right_sched = SCHEDULER_RENAME[r[7]]
            right_trial = trial_counter[right_sched]
            trial_counter[right_sched] += 1

        m = ROBOT_RE.match(r[1])
        if m and left_sched is not None:
            rid = int(m.group(1))
            is_fast = (scenario == HOM_SCENARIO) or (m.group(2) is not None)
            fast_map.setdefault(rid, is_fast)
            legos = _parse_lego_count(r[3])
            if legos is not None:
                samples.append(ThrSample(scenario, left_sched, left_trial or 0, rid, is_fast, legos))
        m = ROBOT_RE.match(r[7])
        if m and right_sched is not None:
            rid = int(m.group(1))
            is_fast = (scenario == HOM_SCENARIO) or (m.group(2) is not None)
            fast_map.setdefault(rid, is_fast)
            legos = _parse_lego_count(r[9])
            if legos is not None:
                samples.append(ThrSample(scenario, right_sched, right_trial or 0, rid, is_fast, legos))
    return samples, fast_map


# -------- trial-dir walk (starvation) ----------------------------------------


def _walk_starvation(
    data_dir: pathlib.Path, scenario: str, fast_map: dict[int, bool]
) -> list[StarvSample]:
    """Walk ``<data_dir>/<scenario>/<raw_scheduler>/trial_*/results.csv`` and
    return per-(trial, robot) starvation rates."""
    out: list[StarvSample] = []
    scen_dir = data_dir / scenario
    if not scen_dir.is_dir():
        return out
    for raw_sched in sorted(scen_dir.iterdir()):
        if not raw_sched.is_dir() or raw_sched.name not in KNOWN_RAW_SCHEDULERS:
            continue
        scheduler = SCHEDULER_RENAME[raw_sched.name]
        for trial_dir in sorted(raw_sched.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue
            csv_path = trial_dir / "results.csv"
            if not csv_path.is_file():
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rid = int(row["robot_idx"])
                        observed = int(row["observed_steps"])
                        starv = int(row["starvation_steps"])
                    except (KeyError, ValueError):
                        continue
                    if observed <= 0:
                        continue
                    is_fast = fast_map.get(rid, scenario == HOM_SCENARIO)
                    out.append(StarvSample(
                        scenario=scenario,
                        scheduler=scheduler,
                        trial_dir=trial_dir.name,
                        robot_id=rid,
                        is_fast=is_fast,
                        starv_rate=starv / observed,
                    ))
    return out


# -------- aggregation --------------------------------------------------------


def _ordered_schedulers(observed: list[str]) -> list[str]:
    head = [s for s in PREFERRED_SCHEDULER_ORDER if s in observed]
    rest = sorted(s for s in observed if s not in PREFERRED_SCHEDULER_ORDER)
    return head + rest


def _scheduler_display(name: str) -> str:
    if name in SCHEDULER_DISPLAY:
        return SCHEDULER_DISPLAY[name]
    m = LA_AHM_RE.match(name)
    if m:
        w = m.group(1)
        if "." in w and w.endswith(".0"):
            w = w.split(".")[0]
        return f"LA $w{{=}}{w}$"
    return name


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _agg(samples: list, key_attr: str, want_fast: bool | None) -> float | None:
    """Mean of samples[key_attr] filtered by is_fast (None = all)."""
    vals = []
    for s in samples:
        if want_fast is not None and s.is_fast != want_fast:
            continue
        vals.append(getattr(s, key_attr))
    return _mean(vals)


def _per_trial_total_throughput(samples: list[ThrSample]) -> float | None:
    """Sum legos/min across all robots within each trial, then average across
    trials. This is the correct cluster-level system throughput — naive
    fast_mean + slow_mean weights tiers equally regardless of population."""
    by_trial: dict[int, list[float]] = defaultdict(list)
    for s in samples:
        by_trial[s.trial_idx].append(s.legos_per_minute)
    per_trial_totals = [sum(vals) for vals in by_trial.values() if vals]
    return _mean(per_trial_totals)


# -------- plots --------------------------------------------------------------

# Paper-ready scheduler display labels.
SCHED_PAPER_LABEL = {
    "max-batch": "MB",
    "round-robin": "RR",
}
LA_AHM_PAPER_RE = re.compile(r"^lookahead-actions@ahm=(\d+(?:\.\d+)?)$")

# Paper-ready scenario display labels.
SCENARIO_PAPER_LABEL = {
    "hom": "10f",
    "1f9s": "1f9s",
    "5f5s": "5f5s",
}


def _scheduler_paper_label(name: str) -> str:
    if name in SCHED_PAPER_LABEL:
        return SCHED_PAPER_LABEL[name]
    m = LA_AHM_PAPER_RE.match(name)
    if m:
        w = m.group(1)
        if w.endswith(".0"):
            w = w.split(".")[0]
        return f"LA@{w}"
    return name


def _scenario_paper_label(name: str) -> str:
    return SCENARIO_PAPER_LABEL.get(name, name)


def _annotate_top(ax, x: float, y: float, val: float | None, fmt: str = ".2f") -> None:
    if val is None:
        return
    ax.text(x, y, format(val, fmt), ha="center", va="bottom", fontsize=8)


def _annotate_inside(ax, x: float, y: float, val: float | None, fmt: str, fontsize: int) -> None:
    """Place the value text inside the bar (white, near the top edge)."""
    if val is None or y <= 0:
        return
    ax.text(
        x, y * 0.96,
        format(val, fmt),
        ha="center", va="top",
        fontsize=fontsize, color="white", fontweight="bold",
    )


def _strip_chrome(ax) -> None:
    """Paper-ready: no spines, no grid, white background, bigger ticks."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="major", length=8, width=1.5, labelsize=14)


def _plot_single_bar(
    out_path: pathlib.Path,
    schedulers: list[str],
    values: list[float | None],
    title: str,
    ylabel: str,
    color: str,
    fmt: str = ".2f",
    paper_style: bool = False,
) -> None:
    keep = [(s, v) for s, v in zip(schedulers, values) if v is not None]
    if not keep:
        return
    schedulers_k, values_k = zip(*keep)
    display_labels = [_scheduler_paper_label(s) if paper_style else s for s in schedulers_k]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(schedulers_k) + 3), 4.5))
    fig.set_facecolor("white")
    xs = np.arange(len(schedulers_k))
    edge = "none" if paper_style else "black"
    lw = 0 if paper_style else 0.6
    bars = ax.bar(xs, values_k, color=color, edgecolor=edge, linewidth=lw)

    if paper_style:
        for x, b, v in zip(xs, bars, values_k):
            _annotate_inside(ax, x, b.get_height(), v, fmt=fmt, fontsize=14)
    else:
        for x, b, v in zip(xs, bars, values_k):
            _annotate_top(ax, x, b.get_height(), v, fmt=fmt)

    ax.set_xticks(xs)
    if paper_style:
        ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        _strip_chrome(ax)
    else:
        ax.set_xticklabels(display_labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def _plot_grouped_bars(
    out_path: pathlib.Path,
    schedulers: list[str],
    series: list[tuple[str, list[float | None], str]],
    title: str,
    ylabel: str,
    fmt: str = ".2f",
    paper_style: bool = False,
) -> None:
    keep_idx = [
        i for i in range(len(schedulers))
        if any(s[1][i] is not None for s in series)
    ]
    if not keep_idx:
        return
    schedulers_k = [schedulers[i] for i in keep_idx]
    series_k = [(lab, [vs[i] for i in keep_idx], col) for lab, vs, col in series]
    display_labels = [_scheduler_paper_label(s) if paper_style else s for s in schedulers_k]

    n_series = len(series_k)
    width = 0.8 / n_series
    xs = np.arange(len(schedulers_k))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(schedulers_k) + 3), 4.8))
    fig.set_facecolor("white")
    edge = "none" if paper_style else "black"
    lw = 0 if paper_style else 0.6
    for s_idx, (label, vals, color) in enumerate(series_k):
        offset = (s_idx - (n_series - 1) / 2) * width
        plot_vals = [0.0 if v is None else v for v in vals]
        bars = ax.bar(
            xs + offset, plot_vals, width=width,
            label=label, color=color, edgecolor=edge, linewidth=lw,
        )
        if paper_style:
            for x, b, raw in zip(xs + offset, bars, vals):
                _annotate_inside(ax, x, b.get_height(), raw, fmt=fmt, fontsize=12)
        else:
            for x, b, raw in zip(xs + offset, bars, vals):
                _annotate_top(ax, x, b.get_height(), raw, fmt=fmt)
    ax.set_xticks(xs)
    if paper_style:
        ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        _strip_chrome(ax)
        # No legend (tier color encoding stands on its own).
    else:
        ax.set_xticklabels(display_labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


# -------- LaTeX table generation --------------------------------------------


METRIC_ORDER = [
    ("thr_fast",  "thr fast",  "max", ".2f"),
    ("thr_slow",  "thr slow",  "max", ".2f"),
    ("thr_total", "thr total", "max", ".2f"),
    ("starv_avg",  "starv avg",  "min", ".2f"),
    ("starv_fast", "starv fast", "min", ".2f"),
    ("starv_slow", "starv slow", "min", ".2f"),
]


def _fmt(v: float | None, spec: str) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return format(v, spec)


def _build_real_table(
    scenario: str,
    schedulers: list[str],
    cells: dict[tuple[str, str], float | None],
) -> str:
    """Rows = schedulers, columns = metrics. Bold best per column."""
    # Only keep metrics with at least one non-None value across schedulers.
    metrics = [m for m in METRIC_ORDER if any(cells.get((sch, m[0])) is not None for sch in schedulers)]
    if not metrics:
        return ""
    n_metric_cols = len(metrics)
    colspec = f"Q[c,wd=1.3cm] *{{{n_metric_cols}}}{{X[c]}}"

    # Headers
    header_cells = ["Scheduler"] + [m[1] for m in metrics]
    header_line = " & ".join(header_cells) + " \\\\"

    # Per-column best.
    best_per_metric: dict[str, set[str]] = {}
    for (name, _label, direction, _spec) in metrics:
        vals = [(sch, cells.get((sch, name))) for sch in schedulers]
        ranked = [(sch, v) for sch, v in vals if v is not None]
        if not ranked:
            best_per_metric[name] = set()
            continue
        cmp = min if direction == "min" else max
        best_val = cmp(v for _sch, v in ranked)
        best_per_metric[name] = {sch for sch, v in ranked if v == best_val}

    # Body rows.
    body_lines = []
    for sch in schedulers:
        row = [_scheduler_display(sch)]
        for (name, _label, _direction, spec) in metrics:
            v = cells.get((sch, name))
            text = _fmt(v, spec)
            if sch in best_per_metric[name] and v is not None:
                text = f"\\textbf{{{text}}}"
            row.append(text)
        body_lines.append(" & ".join(row) + " \\\\")
    body = "\n".join(body_lines)

    scenario_display = {"hom": "all fast", "1f9s": "1 fast", "5f5s": "half fast"}[scenario]
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Real-world throughput (legos / min) and starvation rate — scenario: {scenario_display}.}}\n"
        f"\\label{{tab:real_{scenario}}}\n"
        "\\small\n"
        "\\begin{tblr}{\n"
        f"  colspec = {{{colspec}}},\n"
        "  cells = {c},\n"
        "  row{1} = {font=\\bfseries},\n"
        "  column{1} = {font=\\bfseries},\n"
        "  colsep = 3pt,\n"
        "  rowsep = 2pt,\n"
        "}\n"
        "\\toprule\n"
        f"{header_line}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tblr}\n"
        "\\end{table}\n"
    )


# -------- main ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=pathlib.Path, default=None,
                   help="Plots output dir (default: <data-dir>/plots).")
    p.add_argument("--latex-dir", type=pathlib.Path, default=None,
                   help="LaTeX output dir (default: <data-dir>/latex).")
    p.add_argument("--starv-as-percent", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    plots_dir = (args.output_dir or (data_dir / "plots")).resolve()
    latex_dir = (args.latex_dir or (data_dir / "latex")).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    all_thr: list[ThrSample] = []
    all_starv: list[StarvSample] = []
    fast_maps: dict[str, dict[int, bool]] = {}

    for scenario in SCENARIOS:
        csv_path = data_dir / f"{scenario}.csv"
        if not csv_path.is_file():
            print(f"WARN: {csv_path} missing; skipping scenario")
            continue
        thr_samples, fast_map = _parse_throughput_csv(scenario, csv_path)
        fast_maps[scenario] = fast_map
        all_thr.extend(thr_samples)
        all_starv.extend(_walk_starvation(data_dir, scenario, fast_map))

    if not all_thr:
        raise SystemExit("No throughput samples parsed.")

    schedulers_observed = sorted({s.scheduler for s in all_thr} | {s.scheduler for s in all_starv})
    schedulers = _ordered_schedulers(schedulers_observed)

    # Print a sanity summary.
    print(f"Throughput samples: {len(all_thr)}    Starvation samples: {len(all_starv)}")
    print(f"Schedulers: {schedulers}")

    # ---- plots ----
    starv_scale = 100.0 if args.starv_as_percent else 1.0
    starv_unit = "%" if args.starv_as_percent else "rate"

    for scenario in SCENARIOS:
        thr = [s for s in all_thr if s.scenario == scenario]
        starv = [s for s in all_starv if s.scenario == scenario]
        if not thr and not starv:
            continue

        # Throughput plot (paper-ready)
        scen_label = _scenario_paper_label(scenario)
        if scenario == HOM_SCENARIO:
            fast_vals = [
                _agg([s for s in thr if s.scheduler == sch], "legos_per_minute", True)
                for sch in schedulers
            ]
            _plot_single_bar(
                plots_dir / f"throughput__{scenario}__real.png",
                schedulers, fast_vals,
                title=f"{scen_label} Throughput",
                ylabel="Throughput (legos / minute)",
                color="#37A3D2",
                paper_style=True,
            )
        else:
            fast = [_agg([s for s in thr if s.scheduler == sch], "legos_per_minute", True) for sch in schedulers]
            slow = [_agg([s for s in thr if s.scheduler == sch], "legos_per_minute", False) for sch in schedulers]
            _plot_grouped_bars(
                plots_dir / f"throughput__{scenario}__real.png",
                schedulers,
                series=[("fast tier", fast, "#37A3D2"), ("slow tier", slow, "#F94144")],
                title=f"{scen_label} Throughput",
                ylabel="Throughput (legos / minute)",
                paper_style=True,
            )

        # Starvation plot (per tier, scaled to %).
        if scenario == HOM_SCENARIO:
            vals = [
                _scaled(_agg([s for s in starv if s.scheduler == sch], "starv_rate", True), starv_scale)
                for sch in schedulers
            ]
            _plot_single_bar(
                plots_dir / f"starvation__{scenario}__real.png",
                schedulers, vals,
                title=f"{scenario} starvation (real)",
                ylabel=f"Starvation rate ({starv_unit})",
                color="#d62728",
            )
        else:
            fast = [_scaled(_agg([s for s in starv if s.scheduler == sch], "starv_rate", True), starv_scale) for sch in schedulers]
            slow = [_scaled(_agg([s for s in starv if s.scheduler == sch], "starv_rate", False), starv_scale) for sch in schedulers]
            _plot_grouped_bars(
                plots_dir / f"starvation__{scenario}__real.png",
                schedulers,
                series=[("fast tier", fast, "#1f77b4"), ("slow tier", slow, "#ff7f0e")],
                title=f"{scenario} starvation (real)",
                ylabel=f"Starvation rate ({starv_unit})",
            )

    print(f"Wrote plots under {plots_dir}")

    # ---- LaTeX tables ----
    all_blocks: list[str] = []
    for scenario in SCENARIOS:
        thr = [s for s in all_thr if s.scenario == scenario]
        starv = [s for s in all_starv if s.scenario == scenario]
        scen_scheds = _ordered_schedulers(
            sorted({s.scheduler for s in thr} | {s.scheduler for s in starv})
        )
        if not scen_scheds:
            continue

        cells: dict[tuple[str, str], float | None] = {}
        for sch in scen_scheds:
            thr_s = [s for s in thr if s.scheduler == sch]
            stv_s = [s for s in starv if s.scheduler == sch]
            cells[(sch, "thr_fast")] = _agg(thr_s, "legos_per_minute", True)
            cells[(sch, "thr_slow")] = _agg(thr_s, "legos_per_minute", False)
            # Cluster total = sum of all robots' legos/min within a trial, mean over trials.
            cells[(sch, "thr_total")] = _per_trial_total_throughput(thr_s)
            cells[(sch, "starv_avg")] = _scaled(_agg(stv_s, "starv_rate", None), starv_scale)
            cells[(sch, "starv_fast")] = _scaled(_agg(stv_s, "starv_rate", True), starv_scale)
            cells[(sch, "starv_slow")] = _scaled(_agg(stv_s, "starv_rate", False), starv_scale)

        block = _build_real_table(scenario, scen_scheds, cells)
        if not block:
            continue
        (latex_dir / f"real_{scenario}.tex").write_text(block)
        all_blocks.append(f"% --- real / {scenario} ---")
        all_blocks.append(block)

    combined = data_dir / "tables.tex"
    combined.write_text("\n".join(all_blocks) + "\n")
    print(f"Wrote {len(list(latex_dir.glob('real_*.tex')))} LaTeX tables under {latex_dir}")
    print(f"Wrote combined file: {combined}")


def _scaled(v: float | None, scale: float) -> float | None:
    if v is None:
        return None
    return v * scale


if __name__ == "__main__":
    main()
