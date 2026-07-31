"""Build the paper-plot summary tables from collected Modal sweep artifacts.

The historical paper plotters consume a wide ``results_max_batch_size=M.csv``
table.  This collector recreates that contract from the current Modal layout:

    <artifact-root>/scheduler=.../results.csv
    <artifact-root>/scheduler=.../experiment_args.json
    <artifact-root>/_server_pools/pool-*/pool_results.json

Run this only after ``modal volume get`` has finished.  Failed, incomplete, or
short cases are written to ``missing_cases.csv`` and excluded from averages.

Example:
    uv run python scripts/visualization/summarize_sweep_runs.py \
        experiments/sweeps/libero_5min/libero_5min_paper_new \
        --output-dir experiments/sweeps/libero_5min/libero_5min_paper_new/_summary
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import asdict, dataclass

import pandas as pd

FAST_HORIZON_DEFAULT = 6
SLOW_HORIZON_DEFAULT = 10
CONTROL_HZ_DEFAULT = 20.0
MIN_DURATION_DEFAULT = 299.0
LOOKAHEAD_SCHEDULER = "lookahead-actions"

METRICS = (
    "starv",
    "starv_fast",
    "starv_slow",
    "thr_fast",
    "thr_slow",
    "thr_total",
    "successes",
    "successes_fast",
    "successes_slow",
    "worst",
    "n",
)

SCENARIO_ORDER = ("hom", "1f9s", "5f5s")
SCHEDULER_ORDER = (
    "max-batch",
    "round-robin",
    "lookahead-actions@ahm=1",
    "lookahead-actions@ahm=3",
    "lookahead-actions@ahm=5",
)
TOTAL_TIME_RE = re.compile(r"Total experiment time:\s*([0-9]+(?:\.[0-9]+)?)s")


@dataclass(frozen=True)
class CaseSpec:
    run_id: str
    case_dir: pathlib.Path
    scenario: str
    scheduler_label: str
    num_robots: int
    max_batch_size: int
    seed: int

    @property
    def logical_key(self) -> tuple[str, str, int, int, int]:
        return (
            self.scenario,
            self.scheduler_label,
            self.num_robots,
            self.max_batch_size,
            self.seed,
        )


@dataclass
class CaseRow:
    scenario: str
    scheduler_label: str
    num_robots: int
    max_batch_size: int
    seed: int
    mean_starvation: float
    worst_starvation: float
    fast_success_sum: float
    fast_observed_steps_sum: float
    fast_starvation_steps_sum: float
    slow_success_sum: float
    slow_observed_steps_sum: float
    slow_starvation_steps_sum: float


def _scenario_for_experiment(experiment: str) -> str:
    if experiment == "hom" or experiment.startswith("hom_"):
        return "hom"
    if experiment == "one_fast" or experiment.startswith("one_fast_"):
        return "1f9s"
    if experiment == "half_fast_half_slow" or experiment.startswith("half_fast_half_slow_"):
        return "5f5s"
    raise ValueError(f"unrecognized LIBERO scenario in experiment={experiment!r}")


def _parse_case_dir(case_dir: pathlib.Path) -> CaseSpec:
    fields: dict[str, str] = {}
    for token in case_dir.name.split("__"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value

    required = {"scheduler", "experiment", "num_robots", "seed", "max_batch_size"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"run id is missing {', '.join(sorted(missing))}")

    scheduler = fields["scheduler"]
    scheduler_label = scheduler
    if scheduler == LOOKAHEAD_SCHEDULER:
        ahm = fields.get("ahm") or fields.get("action_horizon_multiplier")
        if ahm is None:
            raise ValueError("lookahead-actions run id has no ahm")
        scheduler_label = f"{scheduler}@ahm={float(ahm):g}"

    return CaseSpec(
        run_id=case_dir.name,
        case_dir=case_dir,
        scenario=_scenario_for_experiment(fields["experiment"]),
        scheduler_label=scheduler_label,
        num_robots=int(fields["num_robots"]),
        max_batch_size=int(fields["max_batch_size"]),
        seed=int(fields["seed"]),
    )


def _candidate_artifact_roots(path: pathlib.Path) -> list[pathlib.Path]:
    candidates = [path]
    artifacts = path / "artifacts"
    if artifacts.is_dir():
        candidates.extend(child for child in artifacts.iterdir() if child.is_dir())
    return candidates


def _discover_specs(paths: list[pathlib.Path]) -> tuple[list[CaseSpec], list[pathlib.Path]]:
    case_dirs: dict[str, pathlib.Path] = {}
    artifact_roots: list[pathlib.Path] = []
    for path in paths:
        if not path.is_dir():
            raise SystemExit(f"Not a directory: {path}")
        for candidate in _candidate_artifact_roots(path):
            found = sorted(candidate.glob("scheduler=*"))
            found = [case_dir for case_dir in found if case_dir.is_dir()]
            if found:
                artifact_roots.append(candidate)
                for case_dir in found:
                    previous = case_dirs.get(case_dir.name)
                    if previous is not None and previous.resolve() != case_dir.resolve():
                        raise SystemExit(
                            f"Duplicate run id in two roots: {previous} and {case_dir}"
                        )
                    case_dirs[case_dir.name] = case_dir

            for manifest in sorted(candidate.glob("_server_pools/pool-*/pool_manifest.json")):
                try:
                    payload = json.loads(manifest.read_text())
                    run_ids = payload.get("run_ids", [])
                except (AttributeError, json.JSONDecodeError, OSError) as exc:
                    print(f"WARN: could not read {manifest}: {exc}")
                    continue
                artifact_roots.append(candidate)
                for run_id in run_ids:
                    case_dirs.setdefault(str(run_id), candidate / str(run_id))

    if not case_dirs:
        raise SystemExit("No scheduler=* case directories found.")

    specs: list[CaseSpec] = []
    for case_dir in sorted(case_dirs.values()):
        try:
            specs.append(_parse_case_dir(case_dir))
        except ValueError as exc:
            raise SystemExit(f"Invalid case directory {case_dir}: {exc}") from exc

    by_key: dict[tuple[str, str, int, int, int], str] = {}
    for spec in specs:
        previous = by_key.get(spec.logical_key)
        if previous is not None:
            raise SystemExit(
                f"Duplicate logical case key {spec.logical_key}: {previous!r} and {spec.run_id!r}"
            )
        by_key[spec.logical_key] = spec.run_id
    return specs, sorted(set(artifact_roots))


def _load_pool_statuses(artifact_roots: list[pathlib.Path]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for root in artifact_roots:
        for path in sorted(root.glob("_server_pools/pool-*/pool_results.json")):
            try:
                rows = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"WARN: could not read {path}: {exc}")
                continue
            for row in rows:
                run_id = row.get("run_id")
                status = row.get("status")
                if run_id and status:
                    statuses[str(run_id)] = str(status)
    return statuses


def _load_experiment_config(case_dir: pathlib.Path) -> dict:
    errors: list[str] = []
    for name in ("experiment_args.json", "client_args.json"):
        path = case_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
            return payload["experiment_config"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            errors.append(f"{name}: {exc}")
    detail = "; ".join(errors) if errors else "both files are missing"
    raise ValueError(f"cannot load experiment config ({detail})")


def _duration_from_log(case_dir: pathlib.Path) -> float | None:
    path = case_dir / "client.log"
    if not path.is_file():
        return None
    match = TOTAL_TIME_RE.search(path.read_text(errors="replace"))
    return float(match.group(1)) if match else None


def _success_series(df: pd.DataFrame) -> pd.Series:
    if pd.api.types.is_bool_dtype(df["success"]):
        return df["success"].astype(float)
    numeric = pd.to_numeric(df["success"], errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    return df["success"].astype(str).str.lower().eq("true").astype(float)


def _tier_sums(df: pd.DataFrame, robot_ids: set[int]) -> tuple[float, float, float]:
    tier = df[df["robot_idx"].isin(robot_ids)]
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
) -> CaseRow:
    if status is not None and status != "ok":
        raise ValueError(f"pool status is {status}")
    for required in ("results.csv", "summary.csv"):
        if not (spec.case_dir / required).is_file():
            raise ValueError(f"missing {required}")

    config = _load_experiment_config(spec.case_dir)
    robots = config.get("robots") or []
    if len(robots) != spec.num_robots:
        raise ValueError(f"config has {len(robots)} robots, expected {spec.num_robots}")
    horizons = [int(robot["execution_horizon"]["max"]) for robot in robots]
    control_rates = {float(robot.get("control_hz", control_hz)) for robot in robots}
    if control_rates != {control_hz}:
        raise ValueError(f"control_hz values {sorted(control_rates)} do not match {control_hz:g}")

    configured_duration = float(config.get("time_limit", 0.0))
    if configured_duration < min_duration:
        raise ValueError(f"configured duration is only {configured_duration:.1f}s")
    actual_duration = _duration_from_log(spec.case_dir)
    if actual_duration is None:
        raise ValueError("missing Total experiment time in client.log")
    if actual_duration < min_duration:
        raise ValueError(f"experiment ran only {actual_duration:.1f}s")

    results = pd.read_csv(spec.case_dir / "results.csv")
    needed = {"robot_idx", "success", "observed_steps", "starvation_steps"}
    missing = needed - set(results.columns)
    if missing:
        raise ValueError(f"results.csv is missing {', '.join(sorted(missing))}")
    results = results.dropna(subset=["robot_idx"]).copy()
    results["robot_idx"] = pd.to_numeric(results["robot_idx"], errors="raise").astype(int)
    for column in ("observed_steps", "starvation_steps"):
        results[column] = pd.to_numeric(results[column], errors="raise").astype(float)
    results["success_num"] = _success_series(results)

    observed_robot_ids = set(results["robot_idx"].unique())
    expected_robot_ids = set(range(spec.num_robots))
    if observed_robot_ids != expected_robot_ids:
        raise ValueError(
            "results.csv robot ids differ from expected: "
            f"observed={sorted(observed_robot_ids)}, expected={sorted(expected_robot_ids)}"
        )

    by_robot = results.groupby("robot_idx", sort=True)[["starvation_steps", "observed_steps"]].sum()
    if (by_robot["observed_steps"] <= 0).any():
        raise ValueError("one or more robots have no observed steps")
    robot_starvation = by_robot["starvation_steps"] / by_robot["observed_steps"]

    fast_ids = {idx for idx, horizon in enumerate(horizons) if horizon == fast_horizon}
    slow_ids = {idx for idx, horizon in enumerate(horizons) if horizon == slow_horizon}
    unknown_ids = expected_robot_ids - fast_ids - slow_ids
    if unknown_ids:
        unknown = sorted({horizons[idx] for idx in unknown_ids})
        raise ValueError(f"unrecognized execution horizons: {unknown}")

    fast_success, fast_steps, fast_starved = _tier_sums(results, fast_ids)
    slow_success, slow_steps, slow_starved = _tier_sums(results, slow_ids)
    return CaseRow(
        scenario=spec.scenario,
        scheduler_label=spec.scheduler_label,
        num_robots=spec.num_robots,
        max_batch_size=spec.max_batch_size,
        seed=spec.seed,
        mean_starvation=float(robot_starvation.mean()),
        worst_starvation=float(robot_starvation.max()),
        fast_success_sum=fast_success,
        fast_observed_steps_sum=fast_steps,
        fast_starvation_steps_sum=fast_starved,
        slow_success_sum=slow_success,
        slow_observed_steps_sum=slow_steps,
        slow_starvation_steps_sum=slow_starved,
    )


def _mean_ratio(group: pd.DataFrame, numerator: str, denominator: str) -> float | None:
    values = [
        float(row[numerator]) / float(row[denominator])
        for _, row in group.iterrows()
        if float(row[denominator]) > 0
    ]
    return sum(values) / len(values) if values else None


def _mean_sum(group: pd.DataFrame, columns: tuple[str, ...]) -> float | None:
    values = [sum(float(row[column]) for column in columns) for _, row in group.iterrows()]
    return sum(values) / len(values) if values else None


def _scaled_mean_ratio(
    group: pd.DataFrame, numerator: str, denominator: str, scale: float
) -> float | None:
    ratio = _mean_ratio(group, numerator, denominator)
    return None if ratio is None else ratio * scale


def _aggregate_cell(group: pd.DataFrame, control_hz: float) -> dict[str, float | None]:
    total_throughputs: list[float] = []
    for _, row in group.iterrows():
        total_steps = float(row["fast_observed_steps_sum"]) + float(row["slow_observed_steps_sum"])
        total_successes = float(row["fast_success_sum"]) + float(row["slow_success_sum"])
        if total_steps > 0:
            total_throughputs.append(
                total_successes * control_hz * int(row["num_robots"]) / total_steps
            )

    return {
        "starv": float(group["mean_starvation"].mean()),
        "starv_fast": _mean_ratio(group, "fast_starvation_steps_sum", "fast_observed_steps_sum"),
        "starv_slow": _mean_ratio(group, "slow_starvation_steps_sum", "slow_observed_steps_sum"),
        "thr_fast": _scaled_mean_ratio(
            group, "fast_success_sum", "fast_observed_steps_sum", control_hz
        ),
        "thr_slow": _scaled_mean_ratio(
            group, "slow_success_sum", "slow_observed_steps_sum", control_hz
        ),
        "thr_total": (
            sum(total_throughputs) / len(total_throughputs) if total_throughputs else None
        ),
        "successes": _mean_sum(group, ("fast_success_sum", "slow_success_sum")),
        "successes_fast": _mean_sum(group, ("fast_success_sum",)),
        "successes_slow": _mean_sum(group, ("slow_success_sum",)),
        "worst": float(group["worst_starvation"].mean()),
        "n": float(len(group)),
    }


def _ordered(values: set[str], preferred: tuple[str, ...]) -> list[str]:
    return [value for value in preferred if value in values] + sorted(values - set(preferred))


def _write_summary(
    rows: list[CaseRow],
    specs: list[CaseSpec],
    missing_rows: list[dict],
    output_dir: pathlib.Path,
    control_hz: float,
) -> list[pathlib.Path]:
    df = pd.DataFrame([asdict(row) for row in rows])
    scenarios = _ordered({spec.scenario for spec in specs}, SCENARIO_ORDER)
    schedulers = _ordered({spec.scheduler_label for spec in specs}, SCHEDULER_ORDER)
    num_robots = sorted({spec.num_robots for spec in specs})
    mbs_values = sorted({spec.max_batch_size for spec in specs})

    cells: dict[tuple[int, int, str, str], dict[str, float | None]] = {}
    for key, group in df.groupby(
        ["max_batch_size", "num_robots", "scenario", "scheduler_label"], sort=True
    ):
        mbs, robots, scenario, scheduler = key
        cells[(int(mbs), int(robots), str(scenario), str(scheduler))] = _aggregate_cell(
            group, control_hz
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    columns = pd.MultiIndex.from_product(
        [scenarios, schedulers, METRICS], names=["scenario", "scheduler", "metric"]
    )
    for mbs in mbs_values:
        wide = pd.DataFrame(index=num_robots, columns=columns, dtype=object)
        wide.index.name = "num_robots"
        for (cell_mbs, robots, scenario, scheduler), metrics in cells.items():
            if cell_mbs != mbs:
                continue
            for metric, value in metrics.items():
                wide.loc[robots, (scenario, scheduler, metric)] = value
        wide.columns = [
            f"{scenario}__{scheduler}__{metric}" for scenario, scheduler, metric in columns
        ]
        path = output_dir / f"results_max_batch_size={mbs}.csv"
        wide.to_csv(path)
        written.append(path)

    missing_columns = [
        "run_id",
        "scenario",
        "scheduler",
        "num_robots",
        "max_batch_size",
        "seed",
        "reason",
    ]
    pd.DataFrame(missing_rows, columns=missing_columns).to_csv(
        output_dir / "missing_cases.csv", index=False
    )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artifact_roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fast-horizon", type=int, default=FAST_HORIZON_DEFAULT)
    parser.add_argument("--slow-horizon", type=int, default=SLOW_HORIZON_DEFAULT)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ_DEFAULT)
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION_DEFAULT,
        help="Exclude cases whose logged wall time is below this many seconds.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero instead of writing summaries when any case is unusable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [path.resolve() for path in args.artifact_roots]
    specs, artifact_roots = _discover_specs(roots)
    statuses = _load_pool_statuses(artifact_roots)

    rows: list[CaseRow] = []
    missing_rows: list[dict] = []
    for spec in specs:
        try:
            row = _load_case(
                spec,
                status=statuses.get(spec.run_id),
                fast_horizon=args.fast_horizon,
                slow_horizon=args.slow_horizon,
                control_hz=args.control_hz,
                min_duration=args.min_duration,
            )
        except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
            missing_rows.append(
                {
                    "run_id": spec.run_id,
                    "scenario": spec.scenario,
                    "scheduler": spec.scheduler_label,
                    "num_robots": spec.num_robots,
                    "max_batch_size": spec.max_batch_size,
                    "seed": spec.seed,
                    "reason": str(exc),
                }
            )
        else:
            rows.append(row)

    if not rows:
        raise SystemExit("No complete cases found; is the artifact download still running?")
    if args.require_complete and missing_rows:
        raise SystemExit(
            f"Refusing an incomplete summary: {len(missing_rows)} of {len(specs)} cases are unusable."
        )

    written = _write_summary(rows, specs, missing_rows, args.output_dir.resolve(), args.control_hz)
    print(f"Loaded {len(rows)} complete case(s) out of {len(specs)} discovered.")
    if missing_rows:
        print(
            f"WARNING: excluded {len(missing_rows)} case(s); details are in "
            f"{args.output_dir.resolve() / 'missing_cases.csv'}"
        )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
