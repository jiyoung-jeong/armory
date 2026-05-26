"""Cross-scenario summary tables for interactive sweep runs.

Takes one or more run directories and aggregates per-case results across seeds.
Run dirs that map to the same scenario label are combined into one scenario
column (so you can pass an ``official_<X>_lookahead`` dir alongside an
``official_<X>_maxbatch_rr`` dir and see all schedulers side-by-side).

Outputs:
  * One results table per ``max_batch_size``
      rows    = num_robots
      columns = scenario  ->  scheduler   (lookahead-actions is split per
                                           action_horizon_multiplier value,
                                           e.g. ``lookahead-actions@ahm=1``)
      cells   = mean starvation, fast-tier throughput, slow-tier throughput,
                worst-robot starvation,  n (=#seeds averaged)
  * One missing-cases table listing (scenario, scheduler, num_robots,
    max_batch_size) combos that are missing one or more expected seeds.

Scenario name is inferred from the run-dir basename via a ``hom``/``\\d+f\\d+s``
token. Pass ``--scenario`` once per run_dir to override.

Throughput per tier (per seed)::

    throughput = sum(success) * control_hz / sum(observed_steps)

over all episodes for robots whose ``max_execution_horizon`` matches the
fast/slow horizon. Cells averaged across seeds. Tier cells with no robots in
that scenario (e.g. ``hom`` has no slow robots) render as ``-``.

Run:
    uv run python scripts/interactive/summarize_sweep_runs.py \\
        experiments/sweeps/interactive/official_1f9s_lookahead \\
        experiments/sweeps/interactive/official_1f9s_maxbatch_rr \\
        experiments/sweeps/interactive/official_5f5s_lookahead \\
        experiments/sweeps/interactive/official_5f5s_maxbatch_rr \\
        experiments/sweeps/interactive/official_hom_lookahead \\
        experiments/sweeps/interactive/official_hom_maxbatch_rr \\
        --output-dir experiments/sweeps/interactive/_summary
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from rich.console import Console
from rich.table import Table

FAST_HORIZON_DEFAULT = 6
SLOW_HORIZON_DEFAULT = 10
CONTROL_HZ_DEFAULT = 20.0

SCENARIO_TOKEN_RE = re.compile(r"(hom|\d+f\d+s)", re.IGNORECASE)
LOOKAHEAD_SCHEDULER = "lookahead-actions"
METRICS = (
    "starv", "starv_fast", "starv_slow",
    "thr_fast", "thr_slow", "thr_total",
    "successes", "successes_fast", "successes_slow",
    "worst", "n",
)


@dataclass
class CaseRow:
    scenario: str
    scheduler_label: str
    num_robots: int
    max_batch_size: int
    seed: int
    mean_starvation: float | None
    worst_starvation: float | None
    fast_success_sum: float
    fast_observed_steps_sum: float
    fast_starvation_steps_sum: float
    slow_success_sum: float
    slow_observed_steps_sum: float
    slow_starvation_steps_sum: float


def _infer_scenario(run_root: pathlib.Path, override: str | None) -> str:
    if override:
        return override
    m = SCENARIO_TOKEN_RE.search(run_root.name)
    if m:
        return m.group(1).lower()
    return run_root.name


def _scheduler_label(case: dict) -> str:
    name = case["scheduler"]
    ahm = float(case.get("action_horizon_multiplier", 0.0) or 0.0)
    if name == LOOKAHEAD_SCHEDULER and ahm > 0:
        return f"{name}@ahm={ahm:g}"
    return name


def _load_case(
    case_dir: pathlib.Path,
    scenario: str,
    fast_horizon: int,
    slow_horizon: int,
) -> CaseRow | None:
    case_json = case_dir / "case.json"
    result_json = case_dir / "result.json"
    if not (case_json.is_file() and result_json.is_file()):
        return None
    try:
        case = json.loads(case_json.read_text())
        result = json.loads(result_json.read_text())
    except json.JSONDecodeError as e:
        print(f"WARN: skipping {case_dir.name}: {e}", file=sys.stderr)
        return None
    if result.get("status") != "ok":
        return None

    runtime_json = case_dir / "outputs" / "runtime_metadata.json"
    results_csv = case_dir / "outputs" / "results.csv"
    if not (runtime_json.is_file() and results_csv.is_file()):
        return None
    try:
        runtime = json.loads(runtime_json.read_text())
    except json.JSONDecodeError:
        return None

    horizons = runtime.get("max_execution_horizon") or []
    fast_robot_ids = {i for i, h in enumerate(horizons) if int(h) == fast_horizon}
    slow_robot_ids = {i for i, h in enumerate(horizons) if int(h) == slow_horizon}

    try:
        df = pd.read_csv(results_csv)
    except Exception as e:
        print(f"WARN: bad results.csv in {case_dir.name}: {e}", file=sys.stderr)
        return None
    needed = {"robot_idx", "success", "observed_steps", "starvation_steps"}
    if not needed.issubset(df.columns):
        return None
    df = df[df["robot_idx"].notna()].copy()
    df["robot_idx"] = df["robot_idx"].astype(int)
    if df["success"].dtype == object:
        df["success_num"] = df["success"].astype(str).str.lower().eq("true").astype(float)
    else:
        df["success_num"] = df["success"].astype(float)

    fast_df = df[df["robot_idx"].isin(fast_robot_ids)]
    slow_df = df[df["robot_idx"].isin(slow_robot_ids)]

    return CaseRow(
        scenario=scenario,
        scheduler_label=_scheduler_label(case),
        num_robots=int(case["num_robots"]),
        max_batch_size=int(case["max_batch_size"]),
        seed=int(case["seed"]),
        mean_starvation=result.get("mean_starvation"),
        worst_starvation=result.get("max_starvation"),
        fast_success_sum=float(fast_df["success_num"].sum()),
        fast_observed_steps_sum=float(fast_df["observed_steps"].sum()),
        fast_starvation_steps_sum=float(fast_df["starvation_steps"].sum()),
        slow_success_sum=float(slow_df["success_num"].sum()),
        slow_observed_steps_sum=float(slow_df["observed_steps"].sum()),
        slow_starvation_steps_sum=float(slow_df["starvation_steps"].sum()),
    )


def _load_all(
    run_dirs: list[tuple[pathlib.Path, str]],
    fast_horizon: int,
    slow_horizon: int,
) -> pd.DataFrame:
    rows: list[CaseRow] = []
    for run_root, scenario in run_dirs:
        for case_dir in sorted(run_root.iterdir()):
            if not case_dir.is_dir() or not case_dir.name.startswith("scheduler="):
                continue
            row = _load_case(case_dir, scenario, fast_horizon, slow_horizon)
            if row is not None:
                rows.append(row)
    if not rows:
        raise SystemExit("No ok cases found in any run dir.")
    return pd.DataFrame([row.__dict__ for row in rows])


def _aggregate_cell(group: pd.DataFrame, control_hz: float) -> dict[str, float | None]:
    starvs = group["mean_starvation"].dropna().tolist()
    worsts = group["worst_starvation"].dropna().tolist()

    def per_seed_ratio(numer_col: str, denom_col: str) -> float | None:
        vals: list[float] = []
        for _, r in group.iterrows():
            denom = r[denom_col]
            if denom and denom > 0:
                vals.append(r[numer_col] / denom)
        if not vals:
            return None
        return sum(vals) / len(vals)

    return {
        "starv": sum(starvs) / len(starvs) if starvs else None,
        "worst": sum(worsts) / len(worsts) if worsts else None,
        "starv_fast": per_seed_ratio("fast_starvation_steps_sum", "fast_observed_steps_sum"),
        "starv_slow": per_seed_ratio("slow_starvation_steps_sum", "slow_observed_steps_sum"),
        # Throughput = successes / observed_seconds = successes * control_hz / observed_steps.
        "thr_fast": _tier_throughput(group, "fast_success_sum", "fast_observed_steps_sum", control_hz),
        "thr_slow": _tier_throughput(group, "slow_success_sum", "slow_observed_steps_sum", control_hz),
        # Cluster total throughput: sum of per-robot throughputs across the
        # whole scenario, averaged across seeds. Naive fast+slow weights tiers
        # equally regardless of population, so we reconstruct the true total
        # from raw success/step sums + num_robots for this case.
        "thr_total": _cluster_total_throughput(group, control_hz),
        # Mean success counts per seed: total (fleet-wide), fast tier, slow tier.
        "successes":      _mean_col_sum(group, ("fast_success_sum", "slow_success_sum")),
        "successes_fast": _mean_col_sum(group, ("fast_success_sum",)),
        "successes_slow": _mean_col_sum(group, ("slow_success_sum",)),
        "n": float(len(group)),
    }


def _mean_col_sum(group: pd.DataFrame, cols: tuple[str, ...]) -> float | None:
    per_seed = [sum(float(r[c]) for c in cols) for _, r in group.iterrows()]
    return sum(per_seed) / len(per_seed) if per_seed else None


def _tier_throughput(group: pd.DataFrame, success_col: str, steps_col: str, control_hz: float) -> float | None:
    per_seed: list[float] = []
    for _, r in group.iterrows():
        steps = r[steps_col]
        if steps and steps > 0:
            per_seed.append(r[success_col] * control_hz / steps)
    if not per_seed:
        return None
    return sum(per_seed) / len(per_seed)


def _cluster_total_throughput(group: pd.DataFrame, control_hz: float) -> float | None:
    """Per seed: total_successes * control_hz * num_robots / total_observed_steps.

    This equals the sum of per-robot throughput rates across the whole
    scenario (assuming all robots observe similar total step counts — true
    in trial mode with a fixed wall_clock_time_limit_s).
    """
    per_seed: list[float] = []
    for _, r in group.iterrows():
        total_steps = float(r["fast_observed_steps_sum"]) + float(r["slow_observed_steps_sum"])
        total_succ = float(r["fast_success_sum"]) + float(r["slow_success_sum"])
        nr = int(r["num_robots"])
        if total_steps > 0:
            per_seed.append(total_succ * control_hz * nr / total_steps)
    if not per_seed:
        return None
    return sum(per_seed) / len(per_seed)


def _aggregate(
    df: pd.DataFrame, control_hz: float
) -> dict[tuple[int, int, str, str], dict[str, float | None]]:
    """Return ``{(mbs, num_robots, scenario, scheduler_label): metrics}``."""
    out: dict[tuple[int, int, str, str], dict[str, float | None]] = {}
    for (mbs, num_robots, scenario, sched), grp in df.groupby(
        ["max_batch_size", "num_robots", "scenario", "scheduler_label"], sort=True
    ):
        out[(int(mbs), int(num_robots), scenario, sched)] = _aggregate_cell(grp, control_hz)
    return out


def _fmt(v: float | None, spec: str) -> str:
    if v is None:
        return "-"
    try:
        if pd.isna(v):
            return "-"
    except (TypeError, ValueError):
        pass
    return format(float(v), spec)


def _ordered_schedulers(observed: list[str]) -> list[str]:
    preferred = ["max-batch", "round-robin"]
    head = [s for s in preferred if s in observed]
    rest = sorted(s for s in observed if s not in preferred)
    return head + rest


def _render_results_table(
    console: Console,
    mbs: int,
    scenario: str,
    cells: dict[tuple[int, int, str, str], dict[str, float | None]],
    schedulers: list[str],
    num_robots_list: list[int],
) -> None:
    # Only show schedulers that have at least one cell for this scenario+mbs.
    present_schedulers = [
        sch for sch in schedulers
        if any((mbs, nr, scenario, sch) in cells for nr in num_robots_list)
    ]
    if not present_schedulers:
        return
    title = (
        f"scenario={scenario} | max_batch_size={mbs}  "
        "(cell = starv / thr_fast / thr_slow / worst,  n=#seeds)"
    )
    table = Table(title=title, show_lines=True)
    table.add_column("num_robots", justify="right", style="bold")
    for scheduler in present_schedulers:
        table.add_column(scheduler, justify="left", overflow="fold")

    for nr in num_robots_list:
        row = [str(nr)]
        for scheduler in present_schedulers:
            cell = cells.get((mbs, nr, scenario, scheduler))
            if cell is None:
                row.append("-")
                continue
            txt = (
                f"starv {_fmt(cell['starv'], '.3f')}\n"
                f"thr_f {_fmt(cell['thr_fast'], '.3f')}\n"
                f"thr_s {_fmt(cell['thr_slow'], '.3f')}\n"
                f"worst {_fmt(cell['worst'], '.3f')}\n"
                f"n={int(cell['n']) if cell.get('n') else 0}"
            )
            row.append(txt)
        table.add_row(*row)
    console.print(table)


def _cells_to_wide_df(
    cells: dict[tuple[int, int, str, str], dict[str, float | None]],
    mbs: int,
    scenarios: list[str],
    schedulers: list[str],
    num_robots_list: list[int],
) -> pd.DataFrame:
    """Build a wide DataFrame for CSV export (one MBS slice)."""
    cols = pd.MultiIndex.from_product(
        [scenarios, schedulers, list(METRICS)], names=["scenario", "scheduler", "metric"]
    )
    out = pd.DataFrame(index=num_robots_list, columns=cols, dtype=object)
    out.index.name = "num_robots"
    for (m, nr, sc, sch), metrics in cells.items():
        if m != mbs or sc not in scenarios or sch not in schedulers:
            continue
        for metric in METRICS:
            out.loc[nr, (sc, sch, metric)] = metrics.get(metric)
    return out


def _build_missing_table(
    df: pd.DataFrame, expected_seeds: list[int]
) -> pd.DataFrame:
    """List (scenario, scheduler, num_robots, max_batch_size) combos that are
    missing one or more expected seeds. Expected combos = those observed for
    at least one seed; missing = expected_seeds \\ seeds_seen for that combo."""
    observed: dict[tuple, set[int]] = defaultdict(set)
    for r in df.itertuples():
        key = (r.scenario, r.scheduler_label, r.num_robots, r.max_batch_size)
        observed[key].add(r.seed)

    rows = []
    for key in sorted(observed):
        scenario, scheduler, nr, mbs = key
        seeds_seen = observed[key]
        missing = [s for s in expected_seeds if s not in seeds_seen]
        if missing:
            rows.append({
                "scenario": scenario,
                "scheduler": scheduler,
                "num_robots": nr,
                "max_batch_size": mbs,
                "seeds_present": ",".join(str(s) for s in sorted(seeds_seen)) or "-",
                "seeds_missing": ",".join(str(s) for s in missing),
            })
    return pd.DataFrame(rows)


def _render_missing_table(console: Console, missing_df: pd.DataFrame) -> None:
    if missing_df.empty:
        console.print("[green]No missing cases or seeds.[/green]")
        return
    table = Table(title="Missing cases / seeds", show_lines=False)
    for col in missing_df.columns:
        table.add_column(col, justify="left")
    for _, r in missing_df.iterrows():
        table.add_row(*[str(x) for x in r])
    console.print(table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    p.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Override scenario label per run_dir (repeat positionally).",
    )
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    p.add_argument("--fast-horizon", type=int, default=FAST_HORIZON_DEFAULT)
    p.add_argument("--slow-horizon", type=int, default=SLOW_HORIZON_DEFAULT)
    p.add_argument("--control-hz", type=float, default=CONTROL_HZ_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides = list(args.scenario)
    run_dirs: list[tuple[pathlib.Path, str]] = []
    for i, d in enumerate(args.run_dirs):
        d = d.resolve()
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        scenario = overrides[i] if i < len(overrides) else _infer_scenario(d, None)
        run_dirs.append((d, scenario))

    df = _load_all(run_dirs, args.fast_horizon, args.slow_horizon)

    # Dedup scenarios preserving first-seen order — multiple run dirs that
    # infer to the same scenario share one column group.
    seen: set[str] = set()
    scenarios: list[str] = []
    for _, s in run_dirs:
        if s not in seen:
            seen.add(s)
            scenarios.append(s)

    schedulers = _ordered_schedulers(df["scheduler_label"].unique().tolist())
    expected_seeds = sorted(df["seed"].unique().tolist())
    num_robots_list = sorted(df["num_robots"].unique().tolist())

    console = Console()
    console.print(f"[bold]Loaded[/bold] {len(df)} cases from {len(run_dirs)} run dirs.")
    console.print(
        f"Scenarios: {scenarios}\nSchedulers: {schedulers}\nSeeds observed: {expected_seeds}"
    )

    cells = _aggregate(df, args.control_hz)
    mbs_values = sorted({m for (m, _, _, _) in cells})
    for mbs in mbs_values:
        for scenario in scenarios:
            _render_results_table(
                console, mbs, scenario, cells, schedulers, num_robots_list
            )

    missing_df = _build_missing_table(df, expected_seeds)
    _render_missing_table(console, missing_df)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for mbs in mbs_values:
            wide = _cells_to_wide_df(cells, mbs, scenarios, schedulers, num_robots_list)
            flat = wide.copy()
            flat.columns = [f"{sc}__{sch}__{m}" for sc, sch, m in flat.columns]
            flat.to_csv(args.output_dir / f"results_max_batch_size={mbs}.csv")
        missing_df.to_csv(args.output_dir / "missing_cases.csv", index=False)
        console.print(f"[bold]Wrote CSVs to[/bold] {args.output_dir}")


if __name__ == "__main__":
    main()
