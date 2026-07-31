r"""Generate the aligned fast/slow tier plot from a collected Modal LIBERO sweep.

The input is an artifact directory containing scheduler=* case folders.
Failed, incomplete, and shorter-than-five-minute cases are reported in
<output-dir>/missing_cases.csv and excluded from the aggregate.

Example:
    uv run python scripts/visualization/plot_libero_sweep.py \
        experiments/sweeps/libero_5min/libero_5min_paper_new \
        --num-robots 2,4,6,8,10 --max-batch-size 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAST_HORIZON = 6
SLOW_HORIZON = 10
CONTROL_HZ = 20.0
MIN_DURATION_S = 299.0
LOOKAHEAD = "lookahead-actions"
TIERED_SCENARIOS = ("1f9s", "5f5s")
SCENARIO_DISPLAY = {"1f9s": "One Fast", "5f5s": "Half Fast"}
SCHEDULER_ORDER = (
    "max-batch",
    "round-robin",
    "lookahead-actions@ahm=1",
    "lookahead-actions@ahm=3",
    "lookahead-actions@ahm=5",
)
SCHEDULER_COLORS = {
    "max-batch": "#8E6CA8",
    "round-robin": "#5FA86F",
    "lookahead-actions@ahm=1": "#6FB0D6",
    "lookahead-actions@ahm=3": "#3C86B8",
    "lookahead-actions@ahm=5": "#1E5C84",
}
SCHEDULER_MARKERS = {
    "max-batch": "s",
    "round-robin": "D",
    "lookahead-actions@ahm=1": "o",
    "lookahead-actions@ahm=3": "o",
    "lookahead-actions@ahm=5": "o",
}
SCHEDULER_DISPLAY = {"max-batch": "EDF", "round-robin": "RR"}
LA_AHM_RE = re.compile(r"^lookahead-actions@ahm=(\d+(?:\.\d+)?)$")
TOTAL_TIME_RE = re.compile(r"Total experiment time:\s*([0-9]+(?:\.[0-9]+)?)s")


@dataclass(frozen=True)
class CaseSpec:
    run_id: str
    case_dir: pathlib.Path
    scenario: str
    scheduler: str
    num_robots: int
    max_batch_size: int
    seed: int

    @property
    def logical_key(self) -> tuple[str, str, int, int, int]:
        return (
            self.scenario,
            self.scheduler,
            self.num_robots,
            self.max_batch_size,
            self.seed,
        )


@dataclass(frozen=True)
class CaseMetrics:
    scenario: str
    scheduler: str
    num_robots: int
    max_batch_size: int
    seed: int
    fast_successes: float
    fast_steps: float
    fast_starved: float
    slow_successes: float
    slow_steps: float
    slow_starved: float


def _scenario(experiment: str) -> str:
    if experiment == "hom" or experiment.startswith("hom_"):
        return "hom"
    if experiment == "one_fast" or experiment.startswith("one_fast_"):
        return "1f9s"
    if experiment == "half_fast_half_slow" or experiment.startswith("half_fast_half_slow_"):
        return "5f5s"
    raise ValueError(f"unrecognized LIBERO scenario {experiment!r}")


def _parse_case_dir(case_dir: pathlib.Path) -> CaseSpec:
    fields = dict(token.split("=", 1) for token in case_dir.name.split("__") if "=" in token)
    required = {"scheduler", "experiment", "num_robots", "seed", "max_batch_size"}
    if missing := required - fields.keys():
        raise ValueError(f"run id is missing {', '.join(sorted(missing))}")

    scheduler = fields["scheduler"]
    if scheduler == LOOKAHEAD:
        ahm = fields.get("ahm") or fields.get("action_horizon_multiplier")
        if ahm is None:
            raise ValueError("lookahead-actions run id has no ahm")
        scheduler = f"{scheduler}@ahm={float(ahm):g}"

    return CaseSpec(
        run_id=case_dir.name,
        case_dir=case_dir,
        scenario=_scenario(fields["experiment"]),
        scheduler=scheduler,
        num_robots=int(fields["num_robots"]),
        max_batch_size=int(fields["max_batch_size"]),
        seed=int(fields["seed"]),
    )


def _artifact_roots(path: pathlib.Path) -> list[pathlib.Path]:
    roots = [path]
    artifacts = path / "artifacts"
    if artifacts.is_dir():
        roots.extend(child for child in artifacts.iterdir() if child.is_dir())
    return roots


def _discover(root: pathlib.Path) -> tuple[list[CaseSpec], list[pathlib.Path]]:
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    case_dirs: dict[str, pathlib.Path] = {}
    roots: set[pathlib.Path] = set()
    for candidate in _artifact_roots(root):
        for case_dir in sorted(candidate.glob("scheduler=*")):
            if not case_dir.is_dir():
                continue
            previous = case_dirs.get(case_dir.name)
            if previous is not None and previous.resolve() != case_dir.resolve():
                raise SystemExit(f"Duplicate run id in {previous} and {case_dir}")
            case_dirs[case_dir.name] = case_dir
            roots.add(candidate)

        for manifest in sorted(candidate.glob("_server_pools/pool-*/pool_manifest.json")):
            try:
                payload = json.loads(manifest.read_text())
                run_ids = payload.get("run_ids", [])
            except (AttributeError, json.JSONDecodeError, OSError) as exc:
                print(f"WARN: could not read {manifest}: {exc}")
                continue
            roots.add(candidate)
            for run_id in run_ids:
                case_dirs.setdefault(str(run_id), candidate / str(run_id))

    if not case_dirs:
        raise SystemExit("No scheduler=* cases found.")

    specs: list[CaseSpec] = []
    for case_dir in sorted(case_dirs.values()):
        try:
            specs.append(_parse_case_dir(case_dir))
        except ValueError as exc:
            raise SystemExit(f"Invalid case directory {case_dir}: {exc}") from exc

    seen: dict[tuple[str, str, int, int, int], str] = {}
    for spec in specs:
        if previous := seen.get(spec.logical_key):
            raise SystemExit(
                f"Duplicate logical case {spec.logical_key}: {previous!r} and {spec.run_id!r}"
            )
        seen[spec.logical_key] = spec.run_id
    return specs, sorted(roots)


def _pool_statuses(roots: list[pathlib.Path]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.glob("_server_pools/pool-*/pool_results.json")):
            try:
                rows = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"WARN: could not read {path}: {exc}")
                continue
            for row in rows:
                if row.get("run_id") and row.get("status"):
                    statuses[str(row["run_id"])] = str(row["status"])
    return statuses


def _experiment_config(case_dir: pathlib.Path) -> dict:
    errors: list[str] = []
    for name in ("experiment_args.json", "client_args.json"):
        path = case_dir / name
        if not path.is_file():
            continue
        try:
            return json.loads(path.read_text())["experiment_config"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            errors.append(f"{name}: {exc}")
    detail = "; ".join(errors) if errors else "both files are missing"
    raise ValueError(f"cannot load experiment config ({detail})")


def _duration(case_dir: pathlib.Path) -> float | None:
    path = case_dir / "client.log"
    if not path.is_file():
        return None
    match = TOTAL_TIME_RE.search(path.read_text(errors="replace"))
    return float(match.group(1)) if match else None


def _successes(results: pd.DataFrame) -> pd.Series:
    if pd.api.types.is_bool_dtype(results["success"]):
        return results["success"].astype(float)
    numeric = pd.to_numeric(results["success"], errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    return results["success"].astype(str).str.lower().eq("true").astype(float)


def _tier_sums(results: pd.DataFrame, robot_ids: set[int]) -> tuple[float, float, float]:
    tier = results[results["robot_idx"].isin(robot_ids)]
    return (
        float(tier["success_num"].sum()),
        float(tier["observed_steps"].sum()),
        float(tier["starvation_steps"].sum()),
    )


def _load_case(
    spec: CaseSpec,
    *,
    status: str | None,
    fast_horizon: int,
    slow_horizon: int,
    control_hz: float,
    min_duration: float,
) -> CaseMetrics:
    if status is not None and status != "ok":
        raise ValueError(f"pool status is {status}")
    for required in ("results.csv", "summary.csv"):
        if not (spec.case_dir / required).is_file():
            raise ValueError(f"missing {required}")

    config = _experiment_config(spec.case_dir)
    robots = config.get("robots") or []
    if len(robots) != spec.num_robots:
        raise ValueError(f"config has {len(robots)} robots, expected {spec.num_robots}")

    horizons = [int(robot["execution_horizon"]["max"]) for robot in robots]
    rates = {float(robot.get("control_hz", control_hz)) for robot in robots}
    if rates != {control_hz}:
        raise ValueError(f"control_hz values {sorted(rates)} do not match {control_hz:g}")
    configured_duration = float(config.get("time_limit", 0.0))
    if configured_duration < min_duration:
        raise ValueError(f"configured duration is only {configured_duration:.1f}s")
    actual_duration = _duration(spec.case_dir)
    if actual_duration is None:
        raise ValueError("missing Total experiment time in client.log")
    if actual_duration < min_duration:
        raise ValueError(f"experiment ran only {actual_duration:.1f}s")

    results = pd.read_csv(spec.case_dir / "results.csv")
    needed = {"robot_idx", "success", "observed_steps", "starvation_steps"}
    if missing := needed - set(results.columns):
        raise ValueError(f"results.csv is missing {', '.join(sorted(missing))}")
    results = results.dropna(subset=["robot_idx"]).copy()
    results["robot_idx"] = pd.to_numeric(results["robot_idx"], errors="raise").astype(int)
    for column in ("observed_steps", "starvation_steps"):
        results[column] = pd.to_numeric(results[column], errors="raise").astype(float)
    results["success_num"] = _successes(results)

    expected_ids = set(range(spec.num_robots))
    observed_ids = set(results["robot_idx"].unique())
    if observed_ids != expected_ids:
        raise ValueError(
            "results.csv robot ids differ from expected: "
            f"observed={sorted(observed_ids)}, expected={sorted(expected_ids)}"
        )
    by_robot = results.groupby("robot_idx", sort=True)["observed_steps"].sum()
    if (by_robot <= 0).any():
        raise ValueError("one or more robots have no observed steps")

    fast_ids = {idx for idx, horizon in enumerate(horizons) if horizon == fast_horizon}
    slow_ids = {idx for idx, horizon in enumerate(horizons) if horizon == slow_horizon}
    if unknown_ids := expected_ids - fast_ids - slow_ids:
        unknown = sorted({horizons[idx] for idx in unknown_ids})
        raise ValueError(f"unrecognized execution horizons: {unknown}")
    fast = _tier_sums(results, fast_ids)
    slow = _tier_sums(results, slow_ids)
    return CaseMetrics(
        scenario=spec.scenario,
        scheduler=spec.scheduler,
        num_robots=spec.num_robots,
        max_batch_size=spec.max_batch_size,
        seed=spec.seed,
        fast_successes=fast[0],
        fast_steps=fast[1],
        fast_starved=fast[2],
        slow_successes=slow[0],
        slow_steps=slow[1],
        slow_starved=slow[2],
    )


def _mean_ratio(group: pd.DataFrame, numerator: str, denominator: str) -> float:
    valid = group[group[denominator] > 0]
    if valid.empty:
        return float("nan")
    return float((valid[numerator] / valid[denominator]).mean())


def _aggregate(rows: list[CaseMetrics], control_hz: float) -> pd.DataFrame:
    raw = pd.DataFrame([asdict(row) for row in rows])
    records: list[dict] = []
    keys = ["max_batch_size", "num_robots", "scenario", "scheduler"]
    for (mbs, num_robots, scenario, scheduler), group in raw.groupby(keys, sort=True):
        records.append(
            {
                "max_batch_size": int(mbs),
                "num_robots": int(num_robots),
                "scenario": str(scenario),
                "scheduler": str(scheduler),
                "thr_fast": _mean_ratio(group, "fast_successes", "fast_steps") * control_hz * 60.0,
                "thr_slow": _mean_ratio(group, "slow_successes", "slow_steps") * control_hz * 60.0,
                "starv_fast": _mean_ratio(group, "fast_starved", "fast_steps") * 100.0,
                "starv_slow": _mean_ratio(group, "slow_starved", "slow_steps") * 100.0,
            }
        )
    return pd.DataFrame(records)


def _scheduler_display(name: str) -> str:
    if name in SCHEDULER_DISPLAY:
        return SCHEDULER_DISPLAY[name]
    if match := LA_AHM_RE.match(name):
        weight = match.group(1).removesuffix(".0")
        return "LA" if weight == "1" else f"LA@{weight}"
    return name


def _ordered_schedulers(present: set[str]) -> list[str]:
    return [name for name in SCHEDULER_ORDER if name in present] + sorted(
        present - set(SCHEDULER_ORDER)
    )


def _strip_chrome(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", color="#dddddd", linewidth=0.8)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="major", length=6, width=1.2, labelsize=12)


def _draw_series(ax, data: pd.DataFrame, scheduler: str, metric: str) -> None:
    series = data[data["scheduler"] == scheduler].dropna(subset=[metric])
    if series.empty:
        return
    series = series.sort_values("num_robots")
    ax.plot(
        series["num_robots"],
        series[metric],
        marker=SCHEDULER_MARKERS.get(scheduler, "o"),
        color=SCHEDULER_COLORS.get(scheduler, "#444444"),
        linewidth=1.8,
        markersize=6,
        label=_scheduler_display(scheduler),
    )


def _align_row(axes) -> None:
    values = [
        float(value)
        for ax in axes
        for line in ax.get_lines()
        for value in line.get_ydata()
        if np.isfinite(value)
    ]
    if not values:
        return
    array = np.asarray(values)
    data_min, data_max = float(array.min()), float(array.max())
    q1, q3 = np.percentile(array, [25, 75])
    iqr = q3 - q1
    within = array[(array >= q1 - 1.5 * iqr) & (array <= q3 + 1.5 * iqr)]
    low = float(within.min()) if within.size else data_min
    high = float(within.max()) if within.size else data_max
    span = (high - low) or 1.0
    low_clipped = low > data_min
    high_clipped = high < data_max
    ymin = low - 0.10 * span if low_clipped else min(0.0, data_min)
    ymax = high + 0.12 * span if high_clipped else data_max + 0.05 * span

    for index, ax in enumerate(axes):
        ax.set_ylim(ymin, ymax)
        if index:
            ax.tick_params(axis="y", labelleft=False)
        for line in ax.get_lines():
            color = line.get_color()
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                if not np.isfinite(y):
                    continue
                if high_clipped and y > ymax:
                    marker, boundary, offset, vertical = "^", ymax, 6, "bottom"
                elif low_clipped and y < ymin:
                    marker, boundary, offset, vertical = "v", ymin, -6, "top"
                else:
                    continue
                ax.plot(
                    [x],
                    [boundary],
                    marker=marker,
                    color=color,
                    markersize=8,
                    clip_on=False,
                    zorder=6,
                )
                label = f"{y:.1f}" if abs(y) < 10 else f"{y:.0f}"
                ax.annotate(
                    label,
                    xy=(x, boundary),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va=vertical,
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                    annotation_clip=False,
                )


def _plot(
    data: pd.DataFrame,
    *,
    max_batch_size: int,
    output_dir: pathlib.Path,
    num_robots: set[int] | None,
) -> pathlib.Path | None:
    data = data[data["max_batch_size"] == max_batch_size]
    if num_robots is not None:
        data = data[data["num_robots"].isin(num_robots)]
    present = set(data["scenario"])
    if not set(TIERED_SCENARIOS).issubset(present):
        print(f"WARN: batch size {max_batch_size} does not have both tiered scenarios")
        return None

    schedulers = _ordered_schedulers(set(data["scheduler"]))
    fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.5), sharex=True)
    fig.set_facecolor("white")
    panels = (
        (0, 0, "thr_fast", "Fast tier", "Throughput (successes / min)"),
        (0, 1, "thr_slow", "Slow tier", "Throughput (successes / min)"),
        (1, 0, "starv_fast", "Fast tier", "Starvation rate (%)"),
        (1, 1, "starv_slow", "Slow tier", "Starvation rate (%)"),
    )
    for scenario_index, scenario in enumerate(TIERED_SCENARIOS):
        scenario_data = data[data["scenario"] == scenario]
        for row, tier_offset, metric, tier, ylabel in panels:
            column = scenario_index * 2 + tier_offset
            ax = axes[row, column]
            for scheduler in schedulers:
                _draw_series(ax, scenario_data, scheduler, metric)
            _strip_chrome(ax)
            if row == 0:
                ax.set_title(f"{SCENARIO_DISPLAY[scenario]} — {tier}", fontsize=13)
            if column == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if row == 1:
                ax.set_xlabel("Number of Robots", fontsize=12)

    for row in axes:
        _align_row(row)

    handles: list = []
    labels: list[str] = []
    for ax in axes.flat:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            fontsize=12,
            framealpha=0.95,
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"tier_breakdown_combined_aligned__mbs{max_batch_size}.png"
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def _int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artifact_root", type=pathlib.Path)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Destination (default: <artifact-root>/_summary).",
    )
    parser.add_argument("--num-robots", default=None, help="Comma-separated robot counts.")
    parser.add_argument("--max-batch-size", default=None, help="Comma-separated batch sizes.")
    parser.add_argument("--fast-horizon", type=int, default=FAST_HORIZON)
    parser.add_argument("--slow-horizon", type=int, default=SLOW_HORIZON)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_S)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero if any discovered case is failed, short, or incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = (args.output_dir or artifact_root / "_summary").resolve()
    specs, roots = _discover(artifact_root)
    statuses = _pool_statuses(roots)

    rows: list[CaseMetrics] = []
    missing: list[dict] = []
    for spec in specs:
        try:
            rows.append(
                _load_case(
                    spec,
                    status=statuses.get(spec.run_id),
                    fast_horizon=args.fast_horizon,
                    slow_horizon=args.slow_horizon,
                    control_hz=args.control_hz,
                    min_duration=args.min_duration,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
            missing.append(
                {
                    "run_id": spec.run_id,
                    "scenario": spec.scenario,
                    "scheduler": spec.scheduler,
                    "num_robots": spec.num_robots,
                    "max_batch_size": spec.max_batch_size,
                    "seed": spec.seed,
                    "reason": str(exc),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    missing_columns = [
        "run_id",
        "scenario",
        "scheduler",
        "num_robots",
        "max_batch_size",
        "seed",
        "reason",
    ]
    pd.DataFrame(missing, columns=missing_columns).to_csv(
        output_dir / "missing_cases.csv", index=False
    )
    if not rows:
        raise SystemExit("No complete cases found; is the artifact download still running?")
    if args.require_complete and missing:
        raise SystemExit(
            f"Refusing an incomplete plot: {len(missing)} of {len(specs)} cases are unusable."
        )

    data = _aggregate(rows, args.control_hz)
    requested_mbs = _int_set(args.max_batch_size)
    batch_sizes = sorted(set(data["max_batch_size"]))
    if requested_mbs is not None:
        batch_sizes = [value for value in batch_sizes if value in requested_mbs]
    written = [
        path
        for mbs in batch_sizes
        if (
            path := _plot(
                data,
                max_batch_size=mbs,
                output_dir=output_dir,
                num_robots=_int_set(args.num_robots),
            )
        )
    ]
    if not written:
        raise SystemExit("No aligned tier-breakdown plot could be generated.")

    print(f"Loaded {len(rows)} complete case(s) out of {len(specs)} discovered.")
    if missing:
        print(f"WARNING: excluded {len(missing)} case(s); see {output_dir / 'missing_cases.csv'}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
