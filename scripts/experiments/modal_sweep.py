"""Unified Modal sweep entrypoint for scheduler experiments.

Examples:
    modal run scripts/experiments/modal_sweep.py \
        --setup mock \
        --experiment starvation \
        --schedulers fixed-max-batch,greedy-deadline,round-robin \
        --experiment-configs configs/client/mock/short.json \
        --num-robots 2,4,6 \
        --seeds 7 \
        --output-dir experiments/sweeps/mock

    modal run scripts/experiments/modal_sweep.py \
        --setup mock \
        --experiment alpha-fairness \
        --schedulers fixed-max-batch,greedy-deadline,round-robin,lookahead-actions,dynamic-action \
        --experiment-configs configs/client/mock/half_fast_half_slow/10_robots.json \
        --alpha-grid 0.0,0.25,0.5,0.75,1.0 \
        --seeds 7,42 \
        --output-dir experiments/sweeps/fairness_alpha
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # _setups, plot modules
sys.path.insert(0, str(_HERE.parent))  # serve, run_libero

import run_libero  # noqa: E402
import serve  # noqa: E402
from _setups import ARTIFACTS_VOLUME_NAME, MOCK, REAL_CPU, REAL_GPU, Case, app  # noqa: E402

SETUP_MOCK = "mock"
SETUP_REAL_CPU = "real-cpu"
SETUP_REAL_GPU = "real-gpu"
SETUPS = (SETUP_MOCK, SETUP_REAL_CPU, SETUP_REAL_GPU)

EXPERIMENT_STARVATION = "starvation"
EXPERIMENT_ALPHA_FAIRNESS = "alpha-fairness"
EXPERIMENTS = (EXPERIMENT_STARVATION, EXPERIMENT_ALPHA_FAIRNESS)

BASELINE_SCHEDULERS = ("fixed-max-batch", "greedy-deadline", "round-robin", "lookahead-actions")
DYNAMIC_SCHEDULER = "dynamic-action"

DEFAULT_MOCK_SERVER_CONFIG = "configs/server/mock.json"
DEFAULT_STARVATION_SCHEDULERS = "fixed-max-batch,greedy-deadline,round-robin"
DEFAULT_ALPHA_FAIRNESS_SCHEDULERS = ",".join((*BASELINE_SCHEDULERS, DYNAMIC_SCHEDULER))
DEFAULT_ALPHA_FAIRNESS_MAX_STEPS = 200
DEFAULT_REAL_MAX_STEPS = 150

MODEL_TO_PROFILE = {
    "pi05": "l40s_pi05",
    "gr00t-n1.7": "l40s_gr00t",
}

# Modal's account-wide GPU cap is shared across server + client. Each real-gpu
# case consumes one L40S + one A10G simultaneously.
GPU_CAP = 10
DEFAULT_REAL_GPU_MAX_CONCURRENT = GPU_CAP // 2


@dataclasses.dataclass(frozen=True)
class SweepCase:
    experiment: str
    setup: str
    scheduler: str
    experiment_config: str
    config_index: int
    seed: int
    num_robots: int
    model: str
    alpha: float | None
    scenario_id: str
    horizons: tuple[int, ...]

    @property
    def run_id(self) -> str:
        cfg_name = f"{self.config_index:02d}-{_safe_id(pathlib.Path(self.experiment_config).stem)}"
        parts = [
            f"experiment={self.experiment}",
            f"setup={self.setup}",
            f"model={self.model}",
            f"scheduler={self.scheduler}",
            f"config={cfg_name}",
            f"robots={self.num_robots}",
        ]
        if self.experiment == EXPERIMENT_ALPHA_FAIRNESS:
            parts.append(f"scenario={self.scenario_id}")
        if self.alpha is not None:
            parts.append(f"alpha={self.alpha:.3f}")
        parts.append(f"seed={self.seed}")
        return "__".join(parts)


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", value).strip("-") or "config"


def _strip_json_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _read_config(path: str) -> dict[str, Any]:
    cfg_path = pathlib.Path(path)
    text = cfg_path.read_text()
    if cfg_path.suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(_strip_json_comments(text))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return loaded


def _read_server_config(server_config: str, setup: str) -> dict[str, Any]:
    if server_config:
        return _read_config(server_config)
    if setup == SETUP_MOCK:
        return _read_config(DEFAULT_MOCK_SERVER_CONFIG)
    return {}


def _validate_experiment_config(exp_cfg: dict[str, Any], path: str) -> None:
    if not isinstance(exp_cfg.get("experiment"), dict) or not isinstance(
        exp_cfg.get("robots"), dict
    ):
        raise ValueError(
            f"{path} must contain 'experiment' and 'robots' mappings. "
            "Real-robot control_hz YAML files are not run_libero experiment configs."
        )


def _config_horizons(exp_cfg: dict[str, Any]) -> tuple[int, ...]:
    robots = exp_cfg["robots"]
    return tuple(int(robots[f"robot_{idx}"]["execution_horizon"]) for idx in range(len(robots)))


def _expand_experiment_config(
    exp_cfg: dict[str, Any],
    num_robots: int,
    *,
    max_steps: int | None,
    replicate_single_robot: bool,
) -> dict[str, Any]:
    robots = exp_cfg["robots"]
    if replicate_single_robot and len(robots) == 1:
        robot_template = robots["robot_0"]
        exp_cfg = {
            **exp_cfg,
            "experiment": {**exp_cfg["experiment"], "num_robots": num_robots},
            "robots": {f"robot_{i}": dict(robot_template) for i in range(num_robots)},
        }
    else:
        actual = len(robots)
        exp_cfg = {
            **exp_cfg,
            "experiment": {**exp_cfg["experiment"], "num_robots": actual},
            "robots": {name: dict(robot) for name, robot in robots.items()},
        }
    if max_steps is not None:
        exp_cfg = {**exp_cfg, "experiment": {**exp_cfg["experiment"], "max_steps": max_steps}}
    return exp_cfg


def _model_from_token(token: str | None) -> serve.ModelFamily:
    if not token:
        return serve.ModelFamily.PI05
    normalized = token.strip()
    for model in serve.ModelFamily:
        if normalized == model.name or normalized == model.value:
            return model
    raise ValueError(
        f"Unknown model {token!r}; expected one of {[m.value for m in serve.ModelFamily]}"
    )


def _build_policy(srv_cfg: dict[str, Any], setup: str, model: serve.ModelFamily) -> Any:
    if setup == SETUP_MOCK or srv_cfg.get("policy_type") == "mock":
        policy_cfg = dict(srv_cfg.get("policy", {}))
        policy_cfg.setdefault("profile", MODEL_TO_PROFILE.get(model.value, "l40s_pi05"))
        return serve.Mock(**policy_cfg)
    return serve.Default()


def _build_server_args(
    *,
    case: SweepCase,
    srv_cfg: dict[str, Any],
    setup: str,
    run_dir: pathlib.Path,
    max_batch_size: int | None,
) -> serve.Args:
    cfg_model = srv_cfg.get("model")
    model = _model_from_token(case.model or cfg_model)
    kwargs: dict[str, Any] = dict(
        env=serve.EnvMode[srv_cfg.get("env", "LIBERO")],
        model=model,
        max_batch_size=max_batch_size
        if max_batch_size is not None
        else srv_cfg.get("max_batch_size", 1),
        scheduling_algorithm=case.scheduler,
        alpha=case.alpha if case.alpha is not None else serve.Args.alpha,
        policy=_build_policy(srv_cfg, setup, model),
        log_dir=str(run_dir / "server_logs"),
    )
    if "num_steps" in srv_cfg:
        kwargs["num_steps"] = srv_cfg["num_steps"]
    return serve.Args(**kwargs)


def _build_client_args(
    *,
    setup: str,
    run_dir: pathlib.Path,
    seed: int,
    max_steps: int | None,
) -> run_libero.Args:
    kwargs: dict[str, Any] = dict(
        env="mock" if setup == SETUP_MOCK else "libero",
        overwrite=True,
        progress_type="logging",
        seed=seed,
        output_dir=run_dir / "output",
        log_dir=run_dir / "client_logs",
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return run_libero.Args(**kwargs)


def _build_case(
    case: SweepCase,
    *,
    exp_cfg: dict[str, Any],
    srv_cfg: dict[str, Any],
    output_dir: pathlib.Path,
    max_batch_size: int | None,
    max_steps: int | None,
) -> Case:
    run_dir = output_dir / "runs" / case.run_id
    return Case(
        run_id=case.run_id,
        run_dir=run_dir,
        server_args=_build_server_args(
            case=case,
            srv_cfg=srv_cfg,
            setup=case.setup,
            run_dir=run_dir,
            max_batch_size=max_batch_size,
        ),
        client_args=_build_client_args(
            setup=case.setup,
            run_dir=run_dir,
            seed=case.seed,
            max_steps=max_steps,
        ),
        experiment_config=exp_cfg,
    )


def _write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _download_artifacts(*, stamp: str, out: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts from volume '{ARTIFACTS_VOLUME_NAME}/{stamp}' -> {artifacts_dir}")
    subprocess.run(
        ["modal", "volume", "get", ARTIFACTS_VOLUME_NAME, stamp, str(artifacts_dir), "--force"],
        check=True,
    )
    for row in rows:
        row["artifact_path"] = str(artifacts_dir / stamp / row["run_id"])


def _default_schedulers(experiment: str) -> str:
    if experiment == EXPERIMENT_ALPHA_FAIRNESS:
        return DEFAULT_ALPHA_FAIRNESS_SCHEDULERS
    return DEFAULT_STARVATION_SCHEDULERS


def _default_max_steps(experiment: str, setup: str) -> int | None:
    if experiment == EXPERIMENT_ALPHA_FAIRNESS:
        return DEFAULT_ALPHA_FAIRNESS_MAX_STEPS
    if setup != SETUP_MOCK:
        return DEFAULT_REAL_MAX_STEPS
    return None


def _scheduler_alpha_pairs(
    schedulers: list[str], alpha_grid: list[float], alpha: float
) -> list[tuple[str, float | None]]:
    pairs: list[tuple[str, float | None]] = []
    for scheduler in schedulers:
        if scheduler == DYNAMIC_SCHEDULER:
            if alpha_grid:
                pairs.extend((scheduler, value) for value in alpha_grid)
            else:
                pairs.append((scheduler, alpha))
        else:
            pairs.append((scheduler, None))
    return pairs


def _scenario_id(path: str, config_index: int, exp_cfg: dict[str, Any]) -> str:
    raw = exp_cfg.get("scenario_id") or exp_cfg.get("name") or pathlib.Path(path).stem
    return f"{config_index:02d}-{_safe_id(str(raw))}"


def _make_cases(
    *,
    experiment: str,
    setup: str,
    schedulers: list[str],
    experiment_configs: list[str],
    num_robots: list[int],
    seeds: list[int],
    models: list[str],
    alpha_grid: list[float],
    alpha: float,
    max_steps: int | None,
) -> tuple[list[SweepCase], dict[str, dict[str, Any]]]:
    cases: list[SweepCase] = []
    configs_by_run_id: dict[str, dict[str, Any]] = {}
    pairs = _scheduler_alpha_pairs(schedulers, alpha_grid, alpha)
    replicate_single_robot = experiment == EXPERIMENT_STARVATION

    for cfg_index, cfg_path in enumerate(experiment_configs):
        raw_cfg = _read_config(cfg_path)
        _validate_experiment_config(raw_cfg, cfg_path)
        robot_counts = (
            num_robots
            if replicate_single_robot and len(raw_cfg["robots"]) == 1
            else [len(raw_cfg["robots"])]
        )
        for n in robot_counts:
            exp_cfg = _expand_experiment_config(
                raw_cfg,
                n,
                max_steps=max_steps,
                replicate_single_robot=replicate_single_robot,
            )
            actual_n = int(exp_cfg["experiment"]["num_robots"])
            horizons = _config_horizons(exp_cfg)
            scenario_id = _scenario_id(cfg_path, cfg_index, exp_cfg)
            for model in models:
                for scheduler, requested_alpha in pairs:
                    for seed in seeds:
                        case = SweepCase(
                            experiment=experiment,
                            setup=setup,
                            scheduler=scheduler,
                            experiment_config=cfg_path,
                            config_index=cfg_index,
                            seed=seed,
                            num_robots=actual_n,
                            model=model,
                            alpha=requested_alpha,
                            scenario_id=scenario_id,
                            horizons=horizons,
                        )
                        cases.append(case)
                        configs_by_run_id[case.run_id] = exp_cfg
    return cases, configs_by_run_id


def _case_meta(case: SweepCase) -> dict[str, Any]:
    return {
        "run_id": case.run_id,
        "experiment": case.experiment,
        "setup": case.setup,
        "model": case.model,
        "scheduler": case.scheduler,
        "experiment_config": case.experiment_config,
        "config_index": case.config_index,
        "scenario_id": case.scenario_id,
        "num_robots": case.num_robots,
        "alpha_requested": case.alpha,
        "seed": case.seed,
        "horizons": json.dumps(list(case.horizons)),
    }


def _run_cases(
    *,
    setup: str,
    cases: list[SweepCase],
    built: list[Case],
    stamp: str,
    cpu: int | None,
    memory: int | None,
    max_concurrent: int | None,
) -> list[dict[str, Any]]:
    meta = {case.run_id: _case_meta(case) for case in cases}
    rows: list[dict[str, Any]] = []
    if setup == SETUP_MOCK:
        kwargs: dict[str, Any] = {}
        if cpu is not None:
            kwargs["cpu"] = cpu
        if memory is not None:
            kwargs["memory"] = memory
        if max_concurrent is not None:
            kwargs["max_containers"] = max_concurrent
        results = MOCK.run(built, stamp=stamp, **kwargs)
    else:
        client_cpu = cpu if cpu is not None else max(case.num_robots for case in cases) + 2
        client_memory = memory if memory is not None else 16384
        concurrency = max_concurrent
        if concurrency is None:
            concurrency = DEFAULT_REAL_GPU_MAX_CONCURRENT if setup == SETUP_REAL_GPU else 5
        runner = REAL_GPU if setup == SETUP_REAL_GPU else REAL_CPU
        print(
            f"Sweeping {len(cases)} cases | setup={setup} "
            f"client_cpu={client_cpu} max_concurrent={concurrency}"
        )
        results = runner.run(
            built,
            stamp=stamp,
            max_concurrent=concurrency,
            client_cpu=client_cpu,
            client_memory=client_memory,
        )

    for result in results:
        row = {**meta[result["run_id"]], **result}
        rows.append(row)
        if row.get("mean_starvation") not in (None, ""):
            msg = (
                f"{row['status']}: {row['run_id']} "
                f"mean_starvation={_safe_float(row.get('mean_starvation')):.4f} "
                f"max_starvation={_safe_float(row.get('max_starvation')):.4f}"
            )
        else:
            msg = f"{row['status']}: {row['run_id']} starvation={_safe_float(row.get('starvation_rate')):.3f}"
        if row.get("status") != "ok":
            msg += f" error={row.get('error')!r}"
        print(msg, flush=True)
    return rows


def _plot(experiment: str, latest_csv: pathlib.Path, out: pathlib.Path, stamp: str) -> None:
    if experiment == EXPERIMENT_ALPHA_FAIRNESS:
        from plot_alpha_fairness import plot_results  # noqa: PLC0415

        plot_results(latest_csv, out / "plots" / stamp)
        return

    from plot_starvation_sweep import DEFAULT_METRICS, plot_results  # noqa: PLC0415

    timing_metrics = [
        "step_interval_p95_ms",
        "inference_p99_ms",
        "inbound_p95_ms",
        "outbound_p95_ms",
    ]
    plot_results(latest_csv, out / "plots" / stamp, metrics=list(DEFAULT_METRICS) + timing_metrics)


@app.local_entrypoint()
def main(
    setup: str = SETUP_MOCK,
    experiment: str = EXPERIMENT_STARVATION,
    schedulers: str = "",
    experiment_configs: str = "configs/client/mock/short.json",
    num_robots: str = "2,4,6",
    server_config: str = "",
    models: str = "pi05",
    seeds: str = "7",
    output_dir: str = "experiments/sweeps/modal",
    max_batch_size: int | None = None,
    max_steps: int | None = None,
    alpha: float = 1.0,
    alpha_grid: str = "",
    cpu: int | None = None,
    memory: int | None = None,
    max_concurrent: int | None = None,
    skip_plots: bool = False,
) -> None:
    """Run scheduler sweeps on Modal.

    For ``experiment=alpha-fairness``, every path in ``experiment_configs`` is an
    explicit scenario config. No robot profiles are generated from shorthand.
    """
    if setup not in SETUPS:
        raise ValueError(f"Unknown setup {setup!r}; expected one of {SETUPS}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")

    out = pathlib.Path(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    scheduler_list = _parse_csv(schedulers or _default_schedulers(experiment))
    config_list = _parse_csv(experiment_configs)
    seed_list = _parse_csv(seeds, cast=int)
    robot_count_list = _parse_csv(num_robots, cast=int)
    alpha_list = _parse_csv(alpha_grid, cast=float)
    srv_cfg = _read_server_config(server_config, setup)
    model_list = _parse_csv(models) or [str(srv_cfg.get("model") or serve.ModelFamily.PI05.value)]
    effective_max_steps = (
        max_steps if max_steps is not None else _default_max_steps(experiment, setup)
    )

    cases, configs_by_run_id = _make_cases(
        experiment=experiment,
        setup=setup,
        schedulers=scheduler_list,
        experiment_configs=config_list,
        num_robots=robot_count_list,
        seeds=seed_list,
        models=model_list,
        alpha_grid=alpha_list,
        alpha=alpha,
        max_steps=effective_max_steps,
    )
    if not cases:
        raise ValueError("sweep is empty")

    built = [
        _build_case(
            case,
            exp_cfg=configs_by_run_id[case.run_id],
            srv_cfg=srv_cfg,
            output_dir=out,
            max_batch_size=max_batch_size,
            max_steps=effective_max_steps,
        )
        for case in cases
    ]
    print(
        f"Submitting {len(cases)} cases "
        f"(experiment={experiment} setup={setup} configs={len(config_list)} "
        f"schedulers={scheduler_list} seeds={seed_list})"
    )

    rows = _run_cases(
        setup=setup,
        cases=cases,
        built=built,
        stamp=stamp,
        cpu=cpu,
        memory=memory,
        max_concurrent=max_concurrent,
    )

    _download_artifacts(stamp=stamp, out=out, rows=rows)
    sweep_csv = out / f"sweep_results_{stamp}.csv"
    latest_csv = out / "sweep_results.csv"
    _write_rows(sweep_csv, rows)
    _write_rows(latest_csv, rows)
    print(f"Wrote {latest_csv}")
    print(f"Wrote {sweep_csv}")

    suspicious = [r for r in rows if r.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for row in suspicious:
            print(f"  {row['run_id']}: {row.get('timing_flags', '')}")

    if not skip_plots:
        _plot(experiment, latest_csv, out, stamp)
