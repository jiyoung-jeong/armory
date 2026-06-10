r"""Generate LaTeX `tblr` summary tables from a ``summarize_sweep_runs`` dir.

For each ``max_batch_size`` value present in the summary CSVs, emits four
tables (rows = num_robots, columns grouped by scenario then scheduler):

  1. Average starvation rate              — single value per cell
  2. Fast / slow tier starvation rate     — ``fast / slow`` per cell
  3. Fast / slow tier throughput          — ``fast / slow`` per cell
  4. System throughput (fast + slow)      — single value per cell

Scenario column headers and abbreviations follow the user's reference table::

    hom  -> "all fast"   (only one tier of robots)
    1f9s -> "1 fast"
    5f5s -> "half fast"

Schedulers are rendered in this order, only if present:

    round-robin            -> RR
    max-batch              -> MB
    lookahead-actions@ahm=W -> LA $w{=}W$

Output: a single ``tables.tex`` file under ``<summary_dir>/`` with all tables
in order; each is also written to its own file under ``<summary_dir>/latex/``
for selective \\input{}-ing.

Run:
    uv run python scripts/interactive/generate_latex_tables.py \\
        experiments/sweeps/interactive/_summary
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
from dataclasses import dataclass

import pandas as pd

# --- formatting / labels -------------------------------------------------------

SCENARIO_DISPLAY = {
    "hom": "10 Fast",
    "1f9s": "One Fast",
    "5f5s": "Half Fast",
}
SCENARIO_ORDER = ["hom", "1f9s", "5f5s"]

SCHEDULER_HEADERS = {
    "round-robin": "RR",
    "max-batch": "EDF",
}
LA_AHM_RE = re.compile(r"^lookahead-actions@ahm=(\d+(?:\.\d+)?)$")

MBS_RE = re.compile(r"results_max_batch_size=(\d+)\.csv$")
COL_RE = re.compile(
    r"^(?P<scenario>[^_]+(?:_[^_]+)*?)__(?P<scheduler>.+?)__"
    r"(?P<metric>starv|starv_fast|starv_slow|thr_fast|thr_slow|thr_total|worst|n)$"
)


def _scheduler_display(name: str) -> str:
    if name in SCHEDULER_HEADERS:
        return SCHEDULER_HEADERS[name]
    m = LA_AHM_RE.match(name)
    if m:
        w = m.group(1)
        if "." in w and w.endswith(".0"):
            w = w.split(".")[0]
        return f"LA $w{{=}}{w}$"
    return name


def _scheduler_sort_key(name: str) -> tuple[int, float, str]:
    # Order: RR (0), MB (1), LA in ascending ahm (2, ahm).
    if name == "round-robin":
        return (0, 0.0, "")
    if name == "max-batch":
        return (1, 0.0, "")
    m = LA_AHM_RE.match(name)
    if m:
        return (2, float(m.group(1)), "")
    return (3, 0.0, name)


# --- summary loading -----------------------------------------------------------


def _load_summary(summary_dir: pathlib.Path) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for path in sorted(summary_dir.glob("results_max_batch_size=*.csv")):
        m = MBS_RE.search(path.name)
        if not m:
            continue
        out[int(m.group(1))] = pd.read_csv(path, index_col="num_robots")
    if not out:
        raise SystemExit(f"No results CSVs found under {summary_dir}")
    return out


def _parse_columns(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for col in df.columns:
        m = COL_RE.match(col)
        if not m:
            continue
        key = (m.group("scenario"), m.group("scheduler"))
        out.setdefault(key, {})[m.group("metric")] = col
    return out


def _present_schedulers(
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    scenario: str,
    require_metrics: list[str],
) -> list[str]:
    """Schedulers with at least one non-NaN value for *all* require_metrics
    (after taking the column-wise union — i.e. some row has both metrics)."""
    out: list[str] = []
    for (sc, sch), m in cols.items():
        if sc != scenario:
            continue
        ok = True
        for metric in require_metrics:
            col = m.get(metric)
            if col is None:
                ok = False
                break
            if df[col].dropna().empty:
                ok = False
                break
        if ok:
            out.append(sch)
    return sorted(out, key=_scheduler_sort_key)


# --- value formatting ----------------------------------------------------------


def _fmt_num(v: float | None, decimals: int, scale: float = 1.0) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v * scale:.{decimals}f}"


def _get_raw(row: pd.Series, cols, scenario: str, sched: str, metric: str) -> float | None:
    col = cols.get((scenario, sched), {}).get(metric)
    if col is None:
        return None
    v = row.get(col)
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def _cell_single(row: pd.Series, cols, scenario: str, sched: str, metric: str,
                 decimals: int, scale: float) -> tuple[str, float | None]:
    raw = _get_raw(row, cols, scenario, sched, metric)
    return (_fmt_num(raw, decimals, scale), raw)


def _cell_pair(row: pd.Series, cols, scenario: str, sched: str,
               metric_a: str, metric_b: str, decimals: int, scale: float,
               rank: str = "mean") -> tuple[str, float | None]:
    """Return ("a / b", rank_value). rank='mean' or 'sum'."""
    a = _get_raw(row, cols, scenario, sched, metric_a)
    b = _get_raw(row, cols, scenario, sched, metric_b)
    text = (
        f"{_fmt_num(a, decimals, scale)} / "
        f"{_fmt_num(b, decimals, scale)}"
    )
    vals = [v for v in (a, b) if v is not None]
    if not vals:
        return (text, None)
    if rank == "sum":
        return (text, sum(vals))
    return (text, sum(vals) / len(vals))


# --- table generation ---------------------------------------------------------


@dataclass
class TableSpec:
    name: str           # filename stem
    caption: str
    label: str
    direction: str      # "min" (lower is better) or "max" (higher is better)
    cell_fn: callable   # (row, cols, scenario, sched) -> (text, rank_value or None)


def _build_table(
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    spec: TableSpec,
    mbs: int,
    require_metrics_per_scenario: dict[str, list[str]],
    num_robots_filter: set[int] | None = None,
) -> str:
    """Emit one tblr table. Schedulers per scenario are auto-discovered from
    the data for the metrics that the cell function needs."""
    scenarios_used: list[tuple[str, list[str]]] = []
    for scenario in SCENARIO_ORDER:
        if scenario not in SCENARIO_DISPLAY:
            continue
        schedulers = _present_schedulers(
            df, cols, scenario, require_metrics_per_scenario[scenario]
        )
        if schedulers:
            scenarios_used.append((scenario, schedulers))
    if not scenarios_used:
        return ""

    col_groups = [len(scheds) for _, scheds in scenarios_used]
    total_data_cols = sum(col_groups)
    colspec = f"Q[c,wd=1.1cm] *{{{total_data_cols}}}{{X[c]}}"

    # cmidrule ranges.
    cmidrules = []
    start = 2
    for n in col_groups:
        end = start + n - 1
        cmidrules.append(f"\\cmidrule[lr]{{{start}-{end}}}")
        start = end + 1

    # Construct the header line manually so the SetCell empties align.
    header_cells = ["Num Robots"]
    for scenario, scheds in scenarios_used:
        n = len(scheds)
        header_cells.append(f"\\SetCell[c={n}]{{c}} {SCENARIO_DISPLAY[scenario]}")
        for _ in range(n - 1):
            header_cells.append("")
    header_line = " & ".join(header_cells) + " \\\\"

    sub_header_cells = [""]
    for _, scheds in scenarios_used:
        sub_header_cells.extend(_scheduler_display(s) for s in scheds)
    sub_header_line = " & ".join(sub_header_cells) + " \\\\"

    # Data rows. For each row, per scenario group, bold the best cell.
    body_lines = []
    cmp = min if spec.direction == "min" else max
    for nr, row in df.iterrows():
        if num_robots_filter is not None and int(nr) not in num_robots_filter:
            continue
        cells = [str(int(nr))]
        for _scenario, scheds in scenarios_used:
            group_results = [
                spec.cell_fn(row, cols, _scenario, sched) for sched in scheds
            ]
            ranked = [(i, rv) for i, (_t, rv) in enumerate(group_results) if rv is not None]
            best_idxs: set[int] = set()
            if ranked:
                best_val = cmp(rv for _i, rv in ranked)
                best_idxs = {i for i, rv in ranked if rv == best_val}
            for i, (text, _rv) in enumerate(group_results):
                if i in best_idxs:
                    cells.append(f"\\textbf{{{text}}}")
                else:
                    cells.append(text)
        body_lines.append(" & ".join(cells) + " \\\\")
    if not body_lines:
        return ""

    body = "\n".join(body_lines)
    cmidrule_line = " ".join(cmidrules)

    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{spec.caption} (max batch size = {mbs}).}}\n"
        f"\\label{{tab:{spec.label}_mbs{mbs}}}\n"
        "\\footnotesize\n"
        "\\begin{tblr}{\n"
        f"  colspec = {{{colspec}}},\n"
        "  cells = {c},\n"
        "  row{1,2} = {font=\\bfseries},\n"
        "  column{1} = {font=\\bfseries},\n"
        "  colsep = 4pt,\n"
        "  rowsep = 5pt,\n"
        "}\n"
        "\\toprule\n"
        f"{header_line}\n"
        f"{cmidrule_line}\n"
        f"{sub_header_line}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tblr}\n"
        "\\end{table*}\n"
    )


def _build_system_combined_table(
    df: pd.DataFrame,
    cols: dict[tuple[str, str], dict[str, str]],
    mbs: int,
    *,
    starv_decimals: int = 1,
    starv_scale: float = 100.0,
    thr_decimals: int = 3,
    thr_scale: float = 1.0,
    thr_unit_label: str = "successes/s",
    num_robots_filter: set[int] | None = None,
) -> str:
    """Combined system table: rows grouped by scenario (Config column with
    rowspan), one row per N, columns = schedulers. Each cell stacks system
    throughput (top, bold = max in row) over average starvation (bottom, gray,
    bold = min in row). Both are bolded independently. Matches the user's
    \\cell{thr}{starv} reference template."""
    require = ["thr_total", "starv"]
    scen_scheds: list[tuple[str, list[str]]] = []
    all_scheds: list[str] = []
    for scenario in SCENARIO_ORDER:
        if scenario not in SCENARIO_DISPLAY:
            continue
        s = _present_schedulers(df, cols, scenario, require)
        if not s:
            continue
        scen_scheds.append((scenario, s))
        for ss in s:
            if ss not in all_scheds:
                all_scheds.append(ss)
    if not scen_scheds:
        return ""

    all_scheds = sorted(all_scheds, key=_scheduler_sort_key)
    n_sched = len(all_scheds)
    colspec = f"l l *{{{n_sched}}}{{c}}"

    header_cells = ["Config", "$N$"] + [_scheduler_display(s) for s in all_scheds]
    header_line = " & ".join(header_cells) + " \\\\"

    body_lines: list[str] = []
    last_scenario = scen_scheds[-1][0]
    for scenario, scheds in scen_scheds:
        nrs: list[int] = []
        for nr in df.index:
            if num_robots_filter is not None and int(nr) not in num_robots_filter:
                continue
            row = df.loc[nr]
            if any(
                _get_raw(row, cols, scenario, sch, "thr_total") is not None
                for sch in scheds
            ):
                nrs.append(int(nr))
        if not nrs:
            continue

        scen_label = SCENARIO_DISPLAY[scenario]
        n_rows = len(nrs)
        for i, nr in enumerate(nrs):
            row = df.loc[nr]
            if i == 0:
                scen_cell = (
                    f"\\SetCell[r={n_rows}]{{}} {scen_label}"
                    if n_rows > 1 else scen_label
                )
            else:
                scen_cell = ""
            cells = [scen_cell, str(int(nr))]

            for sch in all_scheds:
                if sch not in scheds:
                    cells.append("--")
                    continue
                thr_v = _get_raw(row, cols, scenario, sch, "thr_total")
                if thr_v is None:
                    cells.append("--")
                    continue
                starv_v = _get_raw(row, cols, scenario, sch, "starv")
                thr_str = _fmt_num(thr_v, thr_decimals, thr_scale)
                starv_str = _fmt_num(starv_v, starv_decimals, starv_scale)
                cells.append(f"\\cell{{{thr_str}}}{{{starv_str}}}")
            body_lines.append(" & ".join(cells) + " \\\\")
        if scenario != last_scenario:
            body_lines.append("\\midrule")

    body = "\n".join(body_lines)

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{LIBERO: System throughput ({thr_unit_label}) and average "
        f"starvation (\\%, \\textcolor{{gray}}{{gray}}), max batch size = {mbs}.}}\n"
        f"\\label{{tab:sys_thr_starv_mbs{mbs}}}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\newcommand{\\cell}[2]{#1 \\\\ {\\scriptsize\\textcolor{gray}{#2\\%}}}\n"
        "\\begin{tblr}{\n"
        f"  colspec = {{{colspec}}},\n"
        "  row{1} = {font=\\bfseries},\n"
        "  column{1,2} = {font=\\bfseries},\n"
        "  colsep = 4pt,\n"
        "  rowsep = 1.5pt,\n"
        "}\n"
        "\\toprule\n"
        f"{header_line}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tblr}\n"
        "\\end{table}\n"
    )


def _scenario_require_metrics(metrics: list[str]) -> dict[str, list[str]]:
    """For each scenario, list which metrics the cell function will read. For
    metrics with a slow-tier component, hom doesn't need slow (it has none).
    We require at least the *fast* component to consider a scheduler "present"."""
    out: dict[str, list[str]] = {}
    for scenario in SCENARIO_ORDER:
        if scenario == "hom":
            out[scenario] = [m for m in metrics if m != "starv_slow" and m != "thr_slow"]
        else:
            out[scenario] = list(metrics)
    return out


# --- main ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("summary_dir", type=pathlib.Path)
    p.add_argument("--starv-as-percent", action="store_true", default=True,
                   help="Render starvation as percentage (×100). Default on.")
    p.add_argument("--starv-decimals", type=int, default=2)
    p.add_argument("--thr-decimals", type=int, default=2)
    p.add_argument("--thr-scale", type=float, default=100.0,
                   help="Multiply throughput values by this before rendering. "
                        "100 gives '7.25 / 8.43' instead of '0.0725 / 0.0843' "
                        "(units become successes per 100s).")
    p.add_argument(
        "--num-robots",
        type=str,
        default=None,
        help="Comma-separated list of num_robots values to include "
             "(e.g. '2,4,10,12,16,20'). Default: include all present.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.resolve()
    if not summary_dir.is_dir():
        raise SystemExit(f"Not a directory: {summary_dir}")
    latex_dir = summary_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    starv_scale = 100.0 if args.starv_as_percent else 1.0
    starv_dec = args.starv_decimals
    thr_dec = args.thr_decimals
    thr_scale = args.thr_scale
    thr_unit_suffix = " ($\\times 100$)" if abs(thr_scale - 100.0) < 1e-9 else (
        "" if abs(thr_scale - 1.0) < 1e-9 else f" ($\\times {thr_scale:g}$)"
    )

    nr_filter: set[int] | None = None
    if args.num_robots:
        nr_filter = {int(x.strip()) for x in args.num_robots.split(",") if x.strip()}

    by_mbs = _load_summary(summary_dir)

    def make_specs() -> list[tuple[TableSpec, list[str]]]:
        return [
            (
                TableSpec(
                    name="avg_starvation",
                    caption=("Average starvation as the number of robots increases"
                             + (" (\\%)" if args.starv_as_percent else "")),
                    label="avg_starv",
                    direction="min",
                    cell_fn=lambda row, cols, sc, sch: _cell_single(
                        row, cols, sc, sch, "starv", starv_dec, starv_scale
                    ),
                ),
                ["starv"],
            ),
            (
                TableSpec(
                    name="tier_starvation",
                    caption=("Per-tier starvation: fast / slow"
                             + (" (\\%)" if args.starv_as_percent else "")),
                    label="tier_starv",
                    direction="min",
                    cell_fn=lambda row, cols, sc, sch: _cell_pair(
                        row, cols, sc, sch, "starv_fast", "starv_slow",
                        starv_dec, starv_scale, rank="mean",
                    ),
                ),
                ["starv_fast", "starv_slow"],
            ),
            (
                TableSpec(
                    name="tier_throughput",
                    caption=f"Per-tier throughput: fast / slow (successes per second){thr_unit_suffix}",
                    label="tier_thr",
                    direction="max",
                    cell_fn=lambda row, cols, sc, sch: _cell_pair(
                        row, cols, sc, sch, "thr_fast", "thr_slow",
                        thr_dec, thr_scale, rank="sum",
                    ),
                ),
                ["thr_fast", "thr_slow"],
            ),
            (
                TableSpec(
                    name="system_throughput",
                    caption="System throughput: cluster total (successes per second)",
                    label="sys_thr",
                    direction="max",
                    cell_fn=lambda row, cols, sc, sch: _cell_single(
                        row, cols, sc, sch, "thr_total", 4, 1.0
                    ),
                ),
                ["thr_total"],
            ),
        ]

    all_blocks: list[str] = []
    for mbs in sorted(by_mbs):
        df = by_mbs[mbs]
        cols = _parse_columns(df)
        for spec, metrics in make_specs():
            require = _scenario_require_metrics(metrics)
            block = _build_table(df, cols, spec, mbs, require, num_robots_filter=nr_filter)
            if not block:
                continue
            (latex_dir / f"{spec.name}_mbs{mbs}.tex").write_text(block)
            all_blocks.append(f"% --- {spec.name} | max_batch_size={mbs} ---")
            all_blocks.append(block)

        combined = _build_system_combined_table(
            df, cols, mbs,
            starv_decimals=1,
            starv_scale=starv_scale,
            thr_decimals=2,
            thr_scale=60.0,
            thr_unit_label="successes/min",
            num_robots_filter=nr_filter,
        )
        if combined:
            (latex_dir / f"system_combined_mbs{mbs}.tex").write_text(combined)
            all_blocks.append(f"% --- system_combined | max_batch_size={mbs} ---")
            all_blocks.append(combined)

    tables_tex = summary_dir / "tables.tex"
    tables_tex.write_text("\n".join(all_blocks) + "\n")
    print(f"Wrote {len(list(latex_dir.glob('*.tex')))} individual tables under {latex_dir}")
    print(f"Wrote combined file: {tables_tex}")


if __name__ == "__main__":
    main()
