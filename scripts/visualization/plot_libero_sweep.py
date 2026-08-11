r"""Generate the paper's aligned fast/slow tier plot from a Slurm LIBERO sweep.

The input may be either one timestamped run directory or a parent containing
one. Cases that failed, have suspicious infrastructure timing, are incomplete,
ran for less than five minutes, or exceed an optional starvation-rate threshold
are recorded in ``<output-dir>/missing_cases.csv`` and excluded from the aggregate.

Example:
    uv run python scripts/visualization/plot_libero_sweep.py \
        experiments/sweeps/skynet-weighted-libero_new \
        --num-robots 2,4,6,8,10 --max-batch-size 3 \
        --max-starvation-rate 0.9 \
        --lookahead-root experiments/sweeps/libero_5min/libero_5min_paper_new
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

FAST_HORIZON = 6
SLOW_HORIZON = 10
CONTROL_HZ = 20.0
MIN_DURATION_S = 299.0
TIERED_SCENARIOS = ("1f9s", "5f5s")
SCENARIO_DISPLAY = {"1f9s": "One Fast", "5f5s": "Half Fast"}
WEIGHTED_SCHEDULERS = {"weighted-edf", "weighted-round-robin"}
LOOKAHEAD_SCHEDULERS = (
    "lookahead-actions@ahm=1",
    "lookahead-actions@ahm=3",
    "lookahead-actions@ahm=5",
)
PAPER_SCHEDULERS = ("max-batch", "round-robin", *LOOKAHEAD_SCHEDULERS)
SCHEDULER_ORDER = (
    "max-batch",
    "round-robin",
    "weighted-edf@w=1",
    "weighted-edf@w=3",
    "weighted-edf@w=5",
    "weighted-round-robin@w=1",
    "weighted-round-robin@w=3",
    "weighted-round-robin@w=5",
    "deficit-round-robin",
    *LOOKAHEAD_SCHEDULERS,
)
SCHEDULER_COLORS = {
    "max-batch": "#555555",
    "round-robin": "#2A9D8F",
    "weighted-edf@w=1": "#B49ACB",
    "weighted-edf@w=3": "#8E6CA8",
    "weighted-edf@w=5": "#65477E",
    "weighted-round-robin@w=1": "#F2B56B",
    "weighted-round-robin@w=3": "#D9822B",
    "weighted-round-robin@w=5": "#A65313",
    "deficit-round-robin": "#5FA86F",
    "lookahead-actions@ahm=1": "#6FB0D6",
    "lookahead-actions@ahm=3": "#3C86B8",
    "lookahead-actions@ahm=5": "#1E5C84",
}
SCHEDULER_MARKERS = {
    "max-batch": "^",
    "round-robin": "X",
    "weighted-edf@w=1": "s",
    "weighted-edf@w=3": "s",
    "weighted-edf@w=5": "s",
    "weighted-round-robin@w=1": "D",
    "weighted-round-robin@w=3": "D",
    "weighted-round-robin@w=5": "D",
    "deficit-round-robin": "P",
    "lookahead-actions@ahm=1": "o",
    "lookahead-actions@ahm=3": "o",
    "lookahead-actions@ahm=5": "o",
}
SCHEDULER_DISPLAY = {
    "max-batch": "EDF",
    "round-robin": "RR",
    "weighted-edf@w=1": "WEDF",
    "weighted-edf@w=3": "WEDF@3",
    "weighted-edf@w=5": "WEDF@5",
    "weighted-round-robin@w=1": "WRR",
    "weighted-round-robin@w=3": "WRR@3",
    "weighted-round-robin@w=5": "WRR@5",
    "deficit-round-robin": "DRR",
    "lookahead-actions@ahm=1": "LA",
    "lookahead-actions@ahm=3": "LA@3",
    "lookahead-actions@ahm=5": "LA@5",
}
WEIGHT_SUFFIX_RE = re.compile(r"_w(?P<weight>[0-9]+(?:\.[0-9]+)?)$")
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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not contain a JSON object")
    return payload


def _scenario(experiment: str) -> str:
    if experiment == "hom" or experiment.startswith("hom_"):
        return "hom"
    if experiment == "one_fast" or experiment.startswith("one_fast_"):
        return "1f9s"
    if experiment == "half_fast_half_slow" or experiment.startswith("half_fast_half_slow_"):
        return "5f5s"
    raise ValueError(f"unrecognized LIBERO scenario {experiment!r}")


def _format_weight(value: object) -> str:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid fast-tier weight {value!r}") from exc
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"invalid fast-tier weight {value!r}")
    return f"{weight:g}"


def _fast_weight(case: dict[str, Any], experiment: str) -> str:
    if match := WEIGHT_SUFFIX_RE.search(experiment):
        return _format_weight(match.group("weight"))

    raw_weights = case.get("weights")
    if isinstance(raw_weights, str):
        weights = [part.strip() for part in raw_weights.split(",") if part.strip()]
    elif isinstance(raw_weights, list):
        weights = raw_weights
    else:
        weights = []
    if not weights:
        raise ValueError("weighted scheduler case has no robot weights")
    return _format_weight(weights[0])


def _scheduler(case: dict[str, Any], experiment: str) -> str:
    scheduler = str(case.get("scheduler", "")).strip()
    if not scheduler:
        raise ValueError("case metadata has no scheduler")
    if scheduler == "lookahead-actions":
        return f"{scheduler}@ahm={_fast_weight(case, experiment)}"
    if scheduler in WEIGHTED_SCHEDULERS:
        return f"{scheduler}@w={_fast_weight(case, experiment)}"
    return scheduler


def _parse_case(case_dir: pathlib.Path, case: dict[str, Any]) -> CaseSpec:
    required = {"scheduler", "experiment", "num_robots", "seed", "max_batch_size"}
    if missing := required - case.keys():
        raise ValueError(f"case metadata is missing {', '.join(sorted(missing))}")
    experiment = str(case["experiment"])
    try:
        return CaseSpec(
            run_id=str(case.get("run_id") or case_dir.name),
            case_dir=case_dir,
            scenario=_scenario(experiment),
            scheduler=_scheduler(case, experiment),
            num_robots=int(case["num_robots"]),
            max_batch_size=int(case["max_batch_size"]),
            seed=int(case["seed"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid case metadata: {exc}") from exc


def _missing_row(
    *,
    run_id: str,
    reason: str,
    scenario: str = "",
    scheduler: str = "",
    num_robots: int | str = "",
    max_batch_size: int | str = "",
    seed: int | str = "",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "scenario": scenario,
        "scheduler": scheduler,
        "num_robots": num_robots,
        "max_batch_size": max_batch_size,
        "seed": seed,
        "reason": reason,
    }


def _discover(root: pathlib.Path) -> tuple[list[CaseSpec], list[dict[str, object]]]:
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    case_paths = sorted(root.rglob("case.json"))
    if not case_paths:
        raise SystemExit(f"No case.json files found under {root}")

    specs: list[CaseSpec] = []
    missing: list[dict[str, object]] = []
    for path in case_paths:
        try:
            case = _read_json(path)
            specs.append(_parse_case(path.parent, case))
        except ValueError as exc:
            missing.append(
                _missing_row(
                    run_id=path.parent.name,
                    reason=f"invalid case.json: {exc}",
                )
            )

    run_ids: dict[str, pathlib.Path] = {}
    logical_keys: dict[tuple[str, str, int, int, int], CaseSpec] = {}
    for spec in specs:
        if previous := run_ids.get(spec.run_id):
            raise SystemExit(f"Duplicate run id {spec.run_id!r} in {previous} and {spec.case_dir}")
        run_ids[spec.run_id] = spec.case_dir
        if previous_spec := logical_keys.get(spec.logical_key):
            raise SystemExit(
                f"Duplicate logical case {spec.logical_key}: "
                f"{previous_spec.run_id!r} and {spec.run_id!r}"
            )
        logical_keys[spec.logical_key] = spec
    return specs, missing


def _apply_replacements(
    specs: list[CaseSpec], replacement_roots: list[pathlib.Path]
) -> list[CaseSpec]:
    ordered_ids = [spec.run_id for spec in specs]
    by_run_id = {spec.run_id: spec for spec in specs}
    for root in replacement_roots:
        replacements, invalid = _discover(root.resolve())
        if invalid:
            raise SystemExit(
                f"Invalid replacement case metadata under {root}: {invalid[0]['reason']}"
            )
        for replacement in replacements:
            original = by_run_id.get(replacement.run_id)
            if original is None:
                raise SystemExit(
                    f"Replacement run_id {replacement.run_id!r} does not exist in the base sweep."
                )
            if replacement.logical_key != original.logical_key:
                raise SystemExit(
                    f"Replacement {replacement.run_id!r} does not match its base sweep axes."
                )
            by_run_id[replacement.run_id] = replacement
        print(f"Applied {len(replacements)} replacement case(s) from {root.resolve()}.")
    return [by_run_id[run_id] for run_id in ordered_ids]


def _append_roots(
    specs: list[CaseSpec],
    missing: list[dict[str, object]],
    roots: list[pathlib.Path],
) -> tuple[list[CaseSpec], list[dict[str, object]]]:
    run_ids = {spec.run_id for spec in specs}
    logical_keys = {spec.logical_key for spec in specs}
    combined = list(specs)
    for root in roots:
        additional, invalid = _discover(root.resolve())
        for spec in additional:
            if spec.run_id in run_ids:
                raise SystemExit(f"Duplicate run id across sweep roots: {spec.run_id!r}")
            if spec.logical_key in logical_keys:
                raise SystemExit(f"Duplicate logical case across sweep roots: {spec.logical_key}")
            run_ids.add(spec.run_id)
            logical_keys.add(spec.logical_key)
            combined.append(spec)
        missing.extend(invalid)
        print(f"Added {len(additional)} case(s) from {root.resolve()}.")
    return combined, missing


def _experiment_config(case_dir: pathlib.Path) -> dict[str, Any]:
    errors: list[str] = []
    candidates = (case_dir / "client_args.json", case_dir / "output/experiment_args.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
            config = payload["experiment_config"]
            if not isinstance(config, dict):
                raise TypeError("experiment_config is not an object")
            return config
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path.relative_to(case_dir)}: {exc}")
    detail = "; ".join(errors) if errors else "both files are missing"
    raise ValueError(f"cannot load experiment config ({detail})")


def _duration(case_dir: pathlib.Path) -> float | None:
    path = case_dir / "logs/client.stdout.log"
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
    fast_horizon: int,
    slow_horizon: int,
    control_hz: float,
    min_duration: float,
    max_starvation_rate: float | None,
) -> CaseMetrics:
    result_path = spec.case_dir / "result.json"
    if not result_path.is_file():
        raise ValueError("missing result.json")
    result = _read_json(result_path)
    status = result.get("status")
    if status != "ok":
        detail = str(result.get("error") or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"result status is {status!r}{suffix}")
    if result.get("timing_suspicious"):
        detail = str(result.get("timing_flags") or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"result flagged suspicious timing{suffix}")

    output_dir = spec.case_dir / "output"
    for required in ("results.csv", "summary.csv"):
        if not (output_dir / required).is_file():
            raise ValueError(f"missing output/{required}")

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
        raise ValueError("missing Total experiment time in logs/client.stdout.log")
    if actual_duration < min_duration:
        raise ValueError(f"experiment ran only {actual_duration:.1f}s")

    results = pd.read_csv(output_dir / "results.csv")
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

    total_steps = float(results["observed_steps"].sum())
    raw_starvation_rate = float(results["starvation_steps"].sum()) / total_steps
    if max_starvation_rate is not None and raw_starvation_rate >= max_starvation_rate:
        raise ValueError(
            f"aggregate raw starvation_rate={raw_starvation_rate:.6f} >= "
            f"--max-starvation-rate={max_starvation_rate:.6f}"
        )

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


def _ratio_values(
    group: pd.DataFrame, numerator: str, denominator: str
) -> pd.Series:
    valid = group[group[denominator] > 0]
    if valid.empty:
        return pd.Series(dtype=float)
    return valid[numerator] / valid[denominator]


def _mean_ratio(group: pd.DataFrame, numerator: str, denominator: str) -> float:
    values = _ratio_values(group, numerator, denominator)
    return float(values.mean()) if not values.empty else float("nan")


def _sem_ratio(group: pd.DataFrame, numerator: str, denominator: str) -> float:
    values = _ratio_values(group, numerator, denominator)
    return float(values.sem()) if len(values) > 1 else float("nan")


def _aggregate(
    rows: list[CaseMetrics],
    specs: list[CaseSpec],
    control_hz: float,
    *,
    include_sem: bool = False,
) -> pd.DataFrame:
    raw = pd.DataFrame([asdict(row) for row in rows])
    keys = ["max_batch_size", "num_robots", "scenario", "scheduler"]
    groups = {key: group for key, group in raw.groupby(keys, sort=True)}
    expected = sorted(
        {(spec.max_batch_size, spec.num_robots, spec.scenario, spec.scheduler) for spec in specs}
    )
    records: list[dict[str, object]] = []
    for mbs, num_robots, scenario, scheduler in expected:
        group = groups.get((mbs, num_robots, scenario, scheduler))
        if group is None:
            record = {
                "max_batch_size": mbs,
                "num_robots": num_robots,
                "scenario": scenario,
                "scheduler": scheduler,
                "thr_fast": float("nan"),
                "thr_slow": float("nan"),
                "starv_fast": float("nan"),
                "starv_slow": float("nan"),
                "n": 0,
            }
            if include_sem:
                record.update(
                    {
                        "thr_fast_sem": float("nan"),
                        "thr_slow_sem": float("nan"),
                        "starv_fast_sem": float("nan"),
                        "starv_slow_sem": float("nan"),
                    }
                )
            records.append(record)
            continue
        record = {
            "max_batch_size": mbs,
            "num_robots": num_robots,
            "scenario": scenario,
            "scheduler": scheduler,
            "thr_fast": _mean_ratio(group, "fast_successes", "fast_steps")
            * control_hz
            * 60.0,
            "thr_slow": _mean_ratio(group, "slow_successes", "slow_steps")
            * control_hz
            * 60.0,
            "starv_fast": _mean_ratio(group, "fast_starved", "fast_steps") * 100.0,
            "starv_slow": _mean_ratio(group, "slow_starved", "slow_steps") * 100.0,
            "n": len(group),
        }
        if include_sem:
            record.update(
                {
                    "thr_fast_sem": _sem_ratio(group, "fast_successes", "fast_steps")
                    * control_hz
                    * 60.0,
                    "thr_slow_sem": _sem_ratio(group, "slow_successes", "slow_steps")
                    * control_hz
                    * 60.0,
                    "starv_fast_sem": _sem_ratio(group, "fast_starved", "fast_steps")
                    * 100.0,
                    "starv_slow_sem": _sem_ratio(group, "slow_starved", "slow_steps")
                    * 100.0,
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def _paper_scenario(root: pathlib.Path) -> str:
    match = re.search(r"(?:^|_)(1f9s|5f5s)(?:_|$)", root.name)
    if match is None:
        raise SystemExit(f"Cannot infer 1f9s/5f5s scenario from paper run root: {root}")
    return match.group(1)


def _paper_sem(
    roots: list[pathlib.Path],
    batch_sizes: list[int],
    control_hz: float,
    fast_horizon: int,
    slow_horizon: int,
    schedulers: tuple[str, ...],
) -> pd.DataFrame:
    """Recover per-seed SEMs from the historical paper artifacts."""
    rows: list[CaseMetrics] = []
    seen: set[tuple[str, str, int, int, int]] = set()
    for root in roots:
        root = root.resolve()
        scenario = _paper_scenario(root)
        case_paths = sorted(root.glob("scheduler=*/case.json"))
        if not case_paths:
            raise SystemExit(f"No paper cases found under {root}")
        for case_path in case_paths:
            case_dir = case_path.parent
            case = _read_json(case_path)
            scheduler = str(case.get("scheduler", ""))
            if scheduler == "lookahead-actions":
                multiplier = _format_weight(case.get("action_horizon_multiplier"))
                scheduler = f"lookahead-actions@ahm={multiplier}"
            if scheduler not in schedulers:
                continue
            max_batch_size = int(case["max_batch_size"])
            if max_batch_size not in batch_sizes:
                continue
            result = _read_json(case_dir / "result.json")
            if result.get("status") != "ok":
                continue

            num_robots = int(case["num_robots"])
            seed = int(case["seed"])
            key = (scenario, scheduler, num_robots, max_batch_size, seed)
            if key in seen:
                raise SystemExit(f"Duplicate raw paper case: {key}")
            seen.add(key)

            config = _read_json(case_dir / "experiment_config.json")
            robots = config.get("robots")
            if not isinstance(robots, dict) or len(robots) != num_robots:
                raise SystemExit(f"Invalid robot config in {case_dir}")
            try:
                horizons = [
                    int(robots[f"robot_{index}"]["max_execution_horizon"])
                    for index in range(num_robots)
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid execution horizons in {case_dir}: {exc}") from exc

            results_path = case_dir / "outputs/results.csv"
            results = pd.read_csv(results_path)
            needed = {"robot_idx", "success", "observed_steps", "starvation_steps"}
            if missing := needed - set(results.columns):
                raise SystemExit(f"{results_path} is missing {sorted(missing)}")
            results = results.dropna(subset=["robot_idx"]).copy()
            results["robot_idx"] = pd.to_numeric(
                results["robot_idx"], errors="raise"
            ).astype(int)
            for column in ("observed_steps", "starvation_steps"):
                results[column] = pd.to_numeric(results[column], errors="raise").astype(float)
            results["success_num"] = _successes(results)

            fast_ids = {
                index for index, horizon in enumerate(horizons) if horizon == fast_horizon
            }
            slow_ids = {
                index for index, horizon in enumerate(horizons) if horizon == slow_horizon
            }
            fast = _tier_sums(results, fast_ids)
            slow = _tier_sums(results, slow_ids)
            rows.append(
                CaseMetrics(
                    scenario=scenario,
                    scheduler=scheduler,
                    num_robots=num_robots,
                    max_batch_size=max_batch_size,
                    seed=seed,
                    fast_successes=fast[0],
                    fast_steps=fast[1],
                    fast_starved=fast[2],
                    slow_successes=slow[0],
                    slow_steps=slow[1],
                    slow_starved=slow[2],
                )
            )

    if not rows:
        raise SystemExit("No usable raw paper cases found")
    raw = pd.DataFrame([asdict(row) for row in rows])
    keys = ["max_batch_size", "num_robots", "scenario", "scheduler"]
    records: list[dict[str, object]] = []
    for key, group in raw.groupby(keys, sort=True):
        record = dict(zip(keys, key, strict=True))
        record.update(
            {
                "thr_fast_sem": _sem_ratio(group, "fast_successes", "fast_steps")
                * control_hz
                * 60.0,
                "thr_slow_sem": _sem_ratio(group, "slow_successes", "slow_steps")
                * control_hz
                * 60.0,
                "starv_fast_sem": _sem_ratio(group, "fast_starved", "fast_steps")
                * 100.0,
                "starv_slow_sem": _sem_ratio(group, "slow_starved", "slow_steps")
                * 100.0,
                "raw_n": len(group),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _attach_paper_sem(reference: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    keys = ["max_batch_size", "num_robots", "scenario", "scheduler"]
    merged = reference.merge(errors, on=keys, how="left", validate="one_to_one")
    missing = merged["raw_n"].isna()
    if missing.any():
        rows = merged.loc[missing, keys].to_dict("records")
        raise SystemExit(f"Missing raw paper seeds for {rows}")
    mismatched = merged[merged["raw_n"].astype(int) != merged["n"].astype(int)]
    if not mismatched.empty:
        rows = mismatched[keys + ["n", "raw_n"]].to_dict("records")
        raise SystemExit(f"Raw seed counts do not match the paper summary: {rows}")
    return merged.drop(columns="raw_n")


def _paper_reference(
    root: pathlib.Path,
    batch_sizes: list[int],
    schedulers: tuple[str, ...],
) -> pd.DataFrame:
    """Load exact scheduler aggregates used by the paper tier plot."""
    records: list[dict[str, object]] = []
    for max_batch_size in batch_sizes:
        filename = f"results_max_batch_size={max_batch_size}.csv"
        candidates = (root / filename, root / "_summary" / filename)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise SystemExit(
                "Missing paper summary; checked "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        summary = pd.read_csv(path)
        if "num_robots" not in summary:
            raise SystemExit(f"Paper summary has no num_robots column: {path}")

        for scenario in TIERED_SCENARIOS:
            for scheduler in schedulers:
                prefix = f"{scenario}__{scheduler}__"
                columns = {
                    "thr_fast": prefix + "thr_fast",
                    "thr_slow": prefix + "thr_slow",
                    "starv_fast": prefix + "starv_fast",
                    "starv_slow": prefix + "starv_slow",
                    "n": prefix + "n",
                }
                if missing := set(columns.values()) - set(summary.columns):
                    raise SystemExit(
                        f"Paper summary is missing {sorted(missing)}: {path}"
                    )
                for _, row in summary.iterrows():
                    records.append(
                        {
                            "max_batch_size": max_batch_size,
                            "num_robots": int(row["num_robots"]),
                            "scenario": scenario,
                            "scheduler": scheduler,
                            # The historical summary stores throughput per second
                            # and starvation as a fraction.
                            "thr_fast": float(row[columns["thr_fast"]]) * 60.0,
                            "thr_slow": float(row[columns["thr_slow"]]) * 60.0,
                            "starv_fast": float(row[columns["starv_fast"]]) * 100.0,
                            "starv_slow": float(row[columns["starv_slow"]]) * 100.0,
                            "n": int(row[columns["n"]]),
                        }
                    )
    return pd.DataFrame(records)


def _scheduler_display(name: str) -> str:
    return SCHEDULER_DISPLAY.get(name, name)


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


def _draw_series(
    ax,
    data: pd.DataFrame,
    scheduler: str,
    metric: str,
    robot_grid: list[int],
) -> None:
    scheduler_data = data[data["scheduler"] == scheduler]
    if scheduler_data.empty:
        return
    series = scheduler_data.set_index("num_robots")[metric].reindex(robot_grid)
    if not series.notna().any():
        return
    ax.plot(
        series.index,
        series.to_numpy(),
        marker=SCHEDULER_MARKERS.get(scheduler, "o"),
        color=SCHEDULER_COLORS.get(scheduler, "#444444"),
        linewidth=1.8,
        markersize=6,
        label=_scheduler_display(scheduler),
    )


def _draw_error_bars(
    ax,
    data: pd.DataFrame,
    scheduler: str,
    metric: str,
    robot_grid: list[int],
) -> None:
    scheduler_data = data[data["scheduler"] == scheduler]
    if scheduler_data.empty:
        return
    indexed = scheduler_data.set_index("num_robots")
    means = indexed[metric].reindex(robot_grid)
    errors = indexed[f"{metric}_sem"].reindex(robot_grid)
    valid = means.notna() & errors.notna()
    if not valid.any():
        return
    ax.errorbar(
        means.index[valid],
        means[valid].to_numpy(),
        yerr=errors[valid].to_numpy(),
        fmt="none",
        ecolor=SCHEDULER_COLORS.get(scheduler, "#444444"),
        elinewidth=1.1,
        capsize=2.5,
        capthick=1.1,
        alpha=0.8,
        label="_nolegend_",
        zorder=2,
    )


def _draw_ci95_band(
    ax,
    data: pd.DataFrame,
    scheduler: str,
    metric: str,
    robot_grid: list[int],
) -> None:
    scheduler_data = data[data["scheduler"] == scheduler]
    if scheduler_data.empty:
        return
    indexed = scheduler_data.set_index("num_robots")
    means = indexed[metric].reindex(robot_grid)
    errors = indexed[f"{metric}_sem"].reindex(robot_grid)
    counts = indexed["n"].reindex(robot_grid)
    valid = means.notna() & errors.notna() & counts.gt(1)
    if not valid.any():
        return

    critical = pd.Series(np.nan, index=means.index, dtype=float)
    critical[valid] = scipy_stats.t.ppf(0.975, counts[valid].to_numpy() - 1)
    half_width = critical * errors
    lower = means - half_width
    upper = means + half_width
    lower = lower.clip(lower=0.0)
    if metric.startswith("starv_"):
        upper = upper.clip(upper=100.0)

    ax.fill_between(
        means.index.to_numpy(),
        lower.to_numpy(),
        upper.to_numpy(),
        where=valid.to_numpy(),
        color=SCHEDULER_COLORS.get(scheduler, "#444444"),
        alpha=0.10,
        linewidth=0,
        interpolate=False,
        label="_nolegend_",
        zorder=1,
    )


def _align_row(axes, *, full_range: bool = False) -> None:
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
    if full_range:
        span = (data_max - data_min) or 1.0
        ymin = min(0.0, data_min)
        ymax = data_max + 0.15 * span
        for index, ax in enumerate(axes):
            ax.set_ylim(ymin, ymax)
            if index:
                ax.tick_params(axis="y", labelleft=False)
        return

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
            for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True):
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
    filename_suffix: str = "",
    error_bars: bool = False,
    ci95_bands: bool = False,
    full_range: bool = False,
) -> pathlib.Path | None:
    data = data[data["max_batch_size"] == max_batch_size]
    if num_robots is not None:
        data = data[data["num_robots"].isin(num_robots)]
    metrics = ["thr_fast", "thr_slow", "starv_fast", "starv_slow"]
    present = set(data.loc[data[metrics].notna().any(axis=1), "scenario"])
    if not set(TIERED_SCENARIOS).issubset(present):
        print(f"WARN: batch size {max_batch_size} does not have both tiered scenarios")
        return None

    robot_grid = sorted(num_robots if num_robots is not None else set(data["num_robots"]))
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
                _draw_series(ax, scenario_data, scheduler, metric, robot_grid)
            _strip_chrome(ax)
            if row == 0:
                ax.set_title(f"{SCENARIO_DISPLAY[scenario]} — {tier}", fontsize=13)
            if column == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if row == 1:
                ax.set_xlabel("Number of Robots", fontsize=12)
                ax.set_xticks(robot_grid)

    for row in axes:
        _align_row(row, full_range=full_range)

    if error_bars:
        for scenario_index, scenario in enumerate(TIERED_SCENARIOS):
            scenario_data = data[data["scenario"] == scenario]
            for row, tier_offset, metric, _, _ in panels:
                column = scenario_index * 2 + tier_offset
                for scheduler in schedulers:
                    _draw_error_bars(
                        axes[row, column], scenario_data, scheduler, metric, robot_grid
                    )
    if ci95_bands:
        for scenario_index, scenario in enumerate(TIERED_SCENARIOS):
            scenario_data = data[data["scenario"] == scenario]
            for row, tier_offset, metric, _, _ in panels:
                column = scenario_index * 2 + tier_offset
                for scheduler in schedulers:
                    _draw_ci95_band(
                        axes[row, column], scenario_data, scheduler, metric, robot_grid
                    )

    handles: list = []
    labels: list[str] = []
    for ax in axes.flat:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
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
    if error_bars or ci95_bands:
        fig.text(
            0.5,
            0.012,
            (
                "Shading: 95% t interval across seeds"
                if ci95_bands
                else "Error bars: ±1 standard error across seeds"
            ),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    fig.tight_layout(rect=[0, 0.035 if error_bars or ci95_bands else 0, 1, 0.95])
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / (
        f"tier_breakdown_combined_aligned__mbs{max_batch_size}{filename_suffix}.png"
    )
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def _int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _str_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


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
    parser.add_argument("--schedulers", default=None, help="Comma-separated schedulers to plot.")
    parser.add_argument("--fast-horizon", type=int, default=FAST_HORIZON)
    parser.add_argument("--slow-horizon", type=int, default=SLOW_HORIZON)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_S)
    parser.add_argument(
        "--max-starvation-rate",
        type=float,
        default=None,
        help="Exclude cases at or above this aggregate raw rate (fraction in [0, 1]).",
    )
    parser.add_argument(
        "--lookahead-root",
        type=pathlib.Path,
        default=None,
        help="Paper summary directory or artifact root whose reference results are overlaid.",
    )
    parser.add_argument(
        "--lookahead-case-root",
        action="append",
        default=[],
        type=pathlib.Path,
        help=(
            "Raw paper-run root used to recover reference seed uncertainty; "
            "repeat once per tiered scenario."
        ),
    )
    parser.add_argument(
        "--paper-baselines",
        action="store_true",
        help="Load paper EDF and RR curves in addition to Lookahead.",
    )
    parser.add_argument(
        "--replacement-root",
        action="append",
        default=[],
        type=pathlib.Path,
        help="Rerun root whose matching run_ids replace base cases; repeat for later rerun passes.",
    )
    parser.add_argument(
        "--additional-root",
        action="append",
        default=[],
        type=pathlib.Path,
        help="Sweep root containing disjoint cases to append; repeat as needed.",
    )
    parser.add_argument(
        "--wedf-vs-lookahead",
        action="store_true",
        help="Also write a separate plot containing only WEDF and Lookahead curves.",
    )
    parser.add_argument(
        "--error-bars",
        action="store_true",
        help=(
            "Write only separate __errorbars plots with mean ± SEM; leave existing "
            "plots and summary CSVs untouched."
        ),
    )
    parser.add_argument(
        "--ci95-bands",
        action="store_true",
        help=(
            "Write only separate __ci95 plots with shaded 95%% t intervals "
            "across seeds; leave existing plots and summary CSVs untouched."
        ),
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Show the full metric range instead of clipping isolated values.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero if any case is failed, timing-flagged, short, or incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.error_bars and args.ci95_bands:
        raise SystemExit("--error-bars and --ci95-bands are mutually exclusive")
    if args.max_starvation_rate is not None and not 0 <= args.max_starvation_rate <= 1:
        raise SystemExit("--max-starvation-rate must be between 0 and 1")
    if args.control_hz <= 0:
        raise SystemExit("--control-hz must be positive")
    if args.min_duration < 0:
        raise SystemExit("--min-duration must be nonnegative")

    artifact_root = args.artifact_root.resolve()
    output_dir = (args.output_dir or artifact_root / "_summary").resolve()
    specs, missing = _discover(artifact_root)
    specs = _apply_replacements(specs, args.replacement_root)
    specs, missing = _append_roots(specs, missing, args.additional_root)

    rows: list[CaseMetrics] = []
    for spec in specs:
        try:
            rows.append(
                _load_case(
                    spec,
                    fast_horizon=args.fast_horizon,
                    slow_horizon=args.slow_horizon,
                    control_hz=args.control_hz,
                    min_duration=args.min_duration,
                    max_starvation_rate=args.max_starvation_rate,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
            missing.append(
                _missing_row(
                    run_id=spec.run_id,
                    scenario=spec.scenario,
                    scheduler=spec.scheduler,
                    num_robots=spec.num_robots,
                    max_batch_size=spec.max_batch_size,
                    seed=spec.seed,
                    reason=str(exc),
                )
            )

    missing_columns = [
        "run_id",
        "scenario",
        "scheduler",
        "num_robots",
        "max_batch_size",
        "seed",
        "reason",
    ]
    uncertainty_plot = args.error_bars or args.ci95_bands
    if not uncertainty_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(missing, columns=missing_columns).to_csv(
            output_dir / "missing_cases.csv", index=False
        )
    if not rows:
        raise SystemExit("No complete cases found; is the sweep still running?")
    if args.require_complete and missing:
        raise SystemExit(
            f"Refusing an incomplete plot: {len(missing)} of "
            f"{len(specs) + len([row for row in missing if not row['scenario']])} "
            "cases are unusable."
        )

    requested_mbs = _int_set(args.max_batch_size)
    data = _aggregate(rows, specs, args.control_hz, include_sem=uncertainty_plot)
    batch_sizes = sorted(set(data["max_batch_size"]))
    if requested_mbs is not None:
        batch_sizes = [value for value in batch_sizes if value in requested_mbs]
    reference_rows = 0
    if args.lookahead_root is not None:
        reference_schedulers = PAPER_SCHEDULERS if args.paper_baselines else LOOKAHEAD_SCHEDULERS
        reference = _paper_reference(
            args.lookahead_root.resolve(), batch_sizes, reference_schedulers
        )
        if uncertainty_plot:
            if not args.lookahead_case_root:
                raise SystemExit(
                    "uncertainty plots with --lookahead-root also require "
                    "--lookahead-case-root for each tiered scenario"
                )
            errors = _paper_sem(
                args.lookahead_case_root,
                batch_sizes,
                args.control_hz,
                args.fast_horizon,
                args.slow_horizon,
                reference_schedulers,
            )
            reference = _attach_paper_sem(reference, errors)
        keys = ["max_batch_size", "num_robots", "scenario", "scheduler"]
        overlap = data.merge(reference, on=keys, how="inner")
        if not overlap.empty:
            raise SystemExit(
                "Paper reference overlaps locally aggregated points: "
                f"{overlap[keys].to_dict('records')}"
            )
        reference_rows = len(reference)
        data = pd.concat([data, reference], ignore_index=True)
    elif args.paper_baselines:
        raise SystemExit("--paper-baselines requires --lookahead-root")
    selected_schedulers = _str_set(args.schedulers)
    if selected_schedulers is not None:
        unknown = selected_schedulers - set(data["scheduler"])
        if unknown:
            raise SystemExit(f"Requested schedulers are unavailable: {sorted(unknown)}")
        data = data[data["scheduler"].isin(selected_schedulers)].copy()
    if not uncertainty_plot:
        data.to_csv(output_dir / "aggregated_metrics.csv", index=False)
    filename_suffix = (
        "__errorbars" if args.error_bars else "__ci95" if args.ci95_bands else ""
    )
    written = [
        path
        for mbs in batch_sizes
        if (
            path := _plot(
                data,
                max_batch_size=mbs,
                output_dir=output_dir,
                num_robots=_int_set(args.num_robots),
                filename_suffix=filename_suffix,
                error_bars=args.error_bars,
                ci95_bands=args.ci95_bands,
                full_range=args.full_range,
            )
        )
    ]
    if not written:
        raise SystemExit("No aligned tier-breakdown plot could be generated.")
    if args.wedf_vs_lookahead:
        comparison = data[data["scheduler"].str.startswith(("weighted-edf", "lookahead-actions"))]
        written.extend(
            path
            for mbs in batch_sizes
            if (
                path := _plot(
                    comparison,
                    max_batch_size=mbs,
                    output_dir=output_dir,
                    num_robots=_int_set(args.num_robots),
                    filename_suffix=(
                        "__wedf_vs_la__errorbars"
                        if args.error_bars
                        else "__wedf_vs_la__ci95"
                        if args.ci95_bands
                        else "__wedf_vs_la"
                    ),
                    error_bars=args.error_bars,
                    ci95_bands=args.ci95_bands,
                    full_range=args.full_range,
                )
            )
        )

    print(f"Loaded {len(rows)} usable case(s) out of {len(specs)} parsed.")
    if missing:
        detail = (
            "the existing exclusion set was left untouched"
            if uncertainty_plot
            else f"see {output_dir / 'missing_cases.csv'}"
        )
        print(f"WARNING: excluded {len(missing)} case(s); {detail}")
    if reference_rows:
        print(f"Loaded {reference_rows} reference aggregate point(s) from the paper run.")
    if not uncertainty_plot:
        print(f"Wrote {output_dir / 'aggregated_metrics.csv'}")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
