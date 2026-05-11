"""Run a bounded real-robot trial and compute the same metrics as run_libero.

Workflow:
  1. Load FleetConfig (defaults to ``configs/armory-tui.yaml``).
  2. Filter to selected robots; require they're all reachable.
  3. ``FleetDispatcher.run_trial(...)`` → start clients, wait, kill (SIGINT
     with grace), SFTP each robot's RealSaver output back.
  4. Optionally fetch ``server_metrics_history.json`` from the running
     armory server (``/save-metrics``).
  5. Run ``calculate_metrics`` and ``generate_all_plots`` on the assembled
     output dir — the same offline pass run_libero.py uses for sim.

The on-disk layout RealSaver produces matches the sim Saver, so the metrics
code parses the real output without modification.
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import sys
from dataclasses import dataclass, field

import requests
import tyro
import yaml

from armory.real import FleetConfig, FleetController, FleetDispatcher, RobotStatus
from armory_client.schemas import RuntimeMetadata

logger = logging.getLogger("run_real")


@dataclass
class Args:
    #################################################################################
    # Fleet selection
    #################################################################################
    robots: list[int] = field(default_factory=list)
    """Workstation IDs to include. Empty = every robot in the config."""

    config_path: str | None = None
    """Path to the fleet YAML. Defaults to <armory>/configs/armory-tui.yaml."""

    #################################################################################
    # Trial parameters
    #################################################################################
    duration_sec: float = 60.0
    """How long to let the clients run before killing them."""

    grace_sec: float = 5.0
    """SIGINT grace period before SIGKILL — needs to cover RealSaver flush."""

    output_dir: pathlib.Path = pathlib.Path("data/real")
    """Parent dir; this script appends ``trial_<timestamp>``."""

    fetch_video: bool = False
    """Pull MP4s back too. Adds ~50–500MB per episode."""

    remote_subdir: str = "armory_episodes"
    """Subdir under the workstation's user_data bind mount where RealSaver writes."""

    require_status: bool = True
    """If true, skip robots that aren't BOOTED before starting."""

    het_config_path: str | None = None
    """Optional YAML mapping workstation id → control_hz. If set, each
    matching robot is launched with ``--ros-args -p control_hz:=<N>``.
    See configs/heterogeneous_*.yaml for examples."""

    #################################################################################
    # Server metrics (optional)
    #################################################################################
    fetch_server_metrics: bool = True
    """Pull /save-metrics from the running server too (enables Gantt + scheduler plots)."""

    server_host: str | None = "localhost"
    """Server host for /save-metrics. Set to None to skip the server fetch."""

    server_port: int = 8080

    #################################################################################
    # Trial provenance
    #################################################################################
    control_hz: int = 20
    """Used only to estimate max_steps in runtime_metadata.json."""

    broker_type: str = "naive_async"

    execution_horizon: int = 20

    resize_size: int = 224


def _default_config_path() -> str:
    """Locate configs/armory-tui.yaml relative to this script."""
    here = pathlib.Path(__file__).resolve().parent
    return str(here.parent / "configs" / "armory-tui.yaml")


def _load_het_config(path: str) -> dict[int, int]:
    """Parse the heterogeneous control-rate YAML.

    Schema: ``control_hz: {<station_id>: <hz>, ...}``. Returns
    ``{station_id: hz}`` with ints on both sides; raises on malformed input.
    """
    raw = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(raw, dict) or "control_hz" not in raw:
        sys.exit(f"het config {path}: missing top-level 'control_hz' mapping")
    mapping = raw["control_hz"]
    if not isinstance(mapping, dict):
        sys.exit(f"het config {path}: 'control_hz' must be a mapping")
    out: dict[int, int] = {}
    for k, v in mapping.items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            sys.exit(f"het config {path}: bad entry {k!r}: {v!r} (need int → int)")
    return out


def _select_targets(cfg: FleetConfig, args: Args) -> list:
    if args.robots:
        wanted = set(args.robots)
        targets = [r for r in cfg.robots if r.id in wanted]
        missing = wanted - {r.id for r in targets}
        if missing:
            sys.exit(f"unknown workstation id(s) in --robots: {sorted(missing)}")
    else:
        targets = list(cfg.robots)
    return targets


def _filter_to_booted(fleet: FleetController, targets: list, timeout_sec: float = 30.0) -> list:
    """Refresh statuses; keep only booted/online robots."""
    fleet.check_all_status().result(timeout=timeout_sec)
    eligible = [r for r in targets if r.status in (RobotStatus.BOOTED, RobotStatus.ONLINE)]
    skipped = [r for r in targets if r not in eligible]
    if skipped:
        logger.warning(
            "skipping %d robot(s) not in BOOTED/ONLINE: %s",
            len(skipped),
            [f"WS-{r.id}({r.status.value})" for r in skipped],
        )
    return eligible


def _write_runtime_metadata(out: pathlib.Path, robots: list, args: Args) -> None:
    estimated_max_steps = int(round(args.duration_sec * args.control_hz))
    metadata = RuntimeMetadata(
        task_suite_name="real",
        num_trials_per_task=1,
        max_steps=estimated_max_steps,
        seed=0,
        resize_size=args.resize_size,
        num_robots=len(robots),
        control_hz=args.control_hz,
        broker_type=args.broker_type,
        episodes=[f"real_session_{r.name}" for r in robots],
        execution_horizon=[args.execution_horizon] * len(robots),
    )
    metadata.to_json(out / "runtime_metadata.json")


def _fetch_server_metrics(args: Args, out: pathlib.Path) -> None:
    url = f"http://{args.server_host}:{args.server_port}/save-metrics"
    try:
        resp = requests.get(url, timeout=30.0)
        resp.raise_for_status()
        (out / "server_metrics_history.json").write_text(json.dumps(resp.json(), indent=2))
        logger.info("saved server metrics history to %s", out / "server_metrics_history.json")
    except Exception as e:
        logger.warning("could not fetch server metrics from %s: %s", url, e)


def _reset_server_metrics(args: Args) -> None:
    url = f"http://{args.server_host}:{args.server_port}/reset"
    try:
        requests.post(url, timeout=5.0)
        logger.info("reset server metrics at %s", url)
    except Exception as e:
        logger.warning("could not reset server metrics: %s", e)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    cfg = FleetConfig(args.config_path or _default_config_path())
    fleet = FleetController(cfg, logger=logging.getLogger("armory.real"))
    fleet.start()
    try:
        dispatcher = FleetDispatcher(fleet)

        targets = _select_targets(cfg, args)
        if not targets:
            sys.exit("no targets selected")

        if args.require_status:
            targets = _filter_to_booted(fleet, targets)
            if not targets:
                sys.exit("no eligible (BOOTED/ONLINE) robots — boot the fleet first")

        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        out = args.output_dir / f"trial_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        logger.info("output dir: %s", out.resolve())
        logger.info("running trial on %d robot(s): %s",
                    len(targets), [f"WS-{r.id}/{r.name}" for r in targets])

        _write_runtime_metadata(out, targets, args)

        # Reset server metrics so the snapshot we fetch matches this trial.
        if args.fetch_server_metrics and args.server_host:
            _reset_server_metrics(args)

        het_overrides = _load_het_config(args.het_config_path) if args.het_config_path else None
        if het_overrides:
            applied = {r.id: het_overrides[r.id] for r in targets if r.id in het_overrides}
            unmatched = sorted(set(het_overrides) - {r.id for r in targets})
            logger.info("control_hz overrides applied: %s", applied)
            if unmatched:
                logger.warning("control_hz config has entries for non-target ids: %s", unmatched)

        # Run the trial — start, wait, kill, fetch.
        fut = dispatcher.run_trial(
            targets,
            duration_sec=args.duration_sec,
            output_dir=out,
            fetch_video=args.fetch_video,
            grace_sec=args.grace_sec,
            remote_subdir=args.remote_subdir,
            control_hz_overrides=het_overrides,
        )
        # Total time: trial duration + grace + fetch overhead. Add a 60s buffer.
        summary = fut.result(timeout=args.duration_sec + args.grace_sec * 2 + 120)

        logger.info("trial summary:")
        for stage, results in summary.items():
            if isinstance(results, dict):
                for rid, msg in results.items():
                    logger.info("  %s WS-%s: %s", stage, rid, str(msg)[:120])
            else:
                logger.info("  %s: %s", stage, results)

        # Pull the server's metrics snapshot if requested.
        if args.fetch_server_metrics and args.server_host:
            _fetch_server_metrics(args, out)

        # Restructure: each robot's data landed under <out>/<robot.name>/<robot_idx>/...
        # but calculate_metrics expects <out>/<robot_idx>/... directly.
        # _fetch_one_robot writes to <local_dir>/<robot_name>/<remote_subdir>/<robot_idx>/...
        # Flatten: hoist the per-robot trees so the metrics pass sees the layout it expects.
        _flatten_fetched_layout(out, args.remote_subdir)

        # Finally: same offline metrics pass as run_libero.py.
        try:
            from sims.libero.metrics import calculate_metrics, generate_all_plots
            calculate_metrics(out)
            generate_all_plots(out)
            logger.info("metrics + plots written to %s", out)
        except Exception as e:
            logger.warning("metrics/plot pass failed (data is still on disk): %s", e)

        print(f"trial complete: {out.resolve()}")
    finally:
        fleet.stop()


def _flatten_fetched_layout(out: pathlib.Path, remote_subdir: str) -> None:
    """Lift per-robot episode trees so calculate_metrics sees <out>/<robot_idx>/.

    fetch lands data at <out>/<robot.name>/<remote_subdir>/<robot_idx>/<episode>/.
    Move <robot_idx> up so the layout matches the sim Saver: <out>/<robot_idx>/<episode>/.
    """
    for robot_dir in list(out.iterdir()):
        if not robot_dir.is_dir():
            continue
        # Skip already-flattened or non-fetched dirs (e.g. server_metrics dir).
        nested_root = robot_dir / remote_subdir.lstrip("/")
        if not nested_root.is_dir():
            continue
        for robot_idx_dir in list(nested_root.iterdir()):
            if not robot_idx_dir.is_dir():
                continue
            target = out / robot_idx_dir.name
            if target.exists():
                logger.warning(
                    "flatten: %s already exists, merging episodes from %s",
                    target, robot_idx_dir,
                )
                for ep in robot_idx_dir.iterdir():
                    dest = target / ep.name
                    if dest.exists():
                        continue
                    ep.rename(dest)
                robot_idx_dir.rmdir()
            else:
                robot_idx_dir.rename(target)
        # Clean up empty intermediates.
        try:
            nested_root.rmdir()
            robot_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main(tyro.cli(Args))
