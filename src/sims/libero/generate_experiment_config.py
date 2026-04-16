#!/usr/bin/env python3
"""Generate experiment configs from a heterogeneity knob.

This script interpolates between:
1) a homogeneous robot profile, and
2) a generated max-heterogeneity profile defined by user-provided maxima.

Interpolation is controlled by a scalar knob k in [0, 1].
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any
from typing import Dict

from openpi_client.network_emulation import load_experiment_config

NETWORK_FIELDS = (
    "uplink_median_ms",
    "uplink_sigma",
    "downlink_median_ms",
    "downlink_sigma",
)


def load_homogeneous_robot_profile(path: pathlib.Path) -> Dict[str, Any]:
    profile = json.loads(path.read_text())

    uplink_median = float(profile["uplink_median_ms"])
    uplink_sigma = float(profile["uplink_sigma"])
    downlink_median = float(profile["downlink_median_ms"])
    downlink_sigma = float(profile["downlink_sigma"])
    execution_horizon = int(profile["execution_horizon"])

    normalized = dict(profile)
    normalized["uplink_median_ms"] = uplink_median
    normalized["uplink_sigma"] = uplink_sigma
    normalized["downlink_median_ms"] = downlink_median
    normalized["downlink_sigma"] = downlink_sigma
    normalized["execution_horizon"] = execution_horizon
    return normalized


def _lerp(a: float, b: float, k: float) -> float:
    return a + k * (b - a)


def _robot_position(robot_idx: int, num_robots: int) -> float:
    if num_robots <= 1:
        return 0.0
    return float(robot_idx) / float(num_robots - 1)


def _build_max_profile(
    homogeneous_profile: Dict[str, Any],
    *,
    max_uplink_median_ms: float,
    max_uplink_sigma: float,
    max_downlink_median_ms: float,
    max_downlink_sigma: float,
    max_execution_horizon: int,
) -> Dict[str, Any]:
    max_profile = {
        "uplink_median_ms": float(max_uplink_median_ms),
        "uplink_sigma": float(max_uplink_sigma),
        "downlink_median_ms": float(max_downlink_median_ms),
        "downlink_sigma": float(max_downlink_sigma),
        "execution_horizon": int(max_execution_horizon),
    }

    return max_profile


def build_output_config(
    *,
    num_robots: int,
    homogeneous_profile: Dict[str, Any],
    max_profile: Dict[str, Any],
    heterogeneity_k: float,
    action_chunk_broker_type: str,
    trials_per_robot: int,
    toxiproxy_api_url: str,
    toxiproxy_listen_host: str,
    toxiproxy_listen_port_base: int,
    toxiproxy_server_args: list[str],
    sampling_default_seed: int,
    sampling_resample_every_requests: int,
) -> Dict[str, Any]:
    broker_type = action_chunk_broker_type.strip().lower()
    k = float(heterogeneity_k)

    output: Dict[str, Any] = {
        "experiment": {
            "action_chunk_broker_type": broker_type,
            "num_robots": num_robots,
            "trials_per_robot": int(trials_per_robot),
        },
        "toxiproxy": {
            "api_url": toxiproxy_api_url,
            "listen_host": toxiproxy_listen_host,
            "listen_port_base": int(toxiproxy_listen_port_base),
            "server_args": list(toxiproxy_server_args),
        },
        "sampling": {
            "default_seed": int(sampling_default_seed),
            "resample_every_requests": int(sampling_resample_every_requests),
        },
        "robots": {},
    }

    for robot_idx in range(num_robots):
        robot_id = f"robot_{robot_idx}"
        position = _robot_position(robot_idx, num_robots)
        robot_out: Dict[str, Any] = {}

        for field in NETWORK_FIELDS:
            homo_val = float(homogeneous_profile[field])
            max_val = float(max_profile[field])
            target_max_hetero = _lerp(homo_val, max_val, position)
            robot_out[field] = _lerp(homo_val, target_max_hetero, k)

        homo_horizon = float(homogeneous_profile["execution_horizon"])
        max_horizon = float(max_profile["execution_horizon"])
        target_horizon_at_max_hetero = _lerp(homo_horizon, max_horizon, position)
        robot_out["execution_horizon"] = int(
            round(_lerp(homo_horizon, target_horizon_at_max_hetero, k))
        )
        robot_out["execution_horizon"] = max(1, robot_out["execution_horizon"])
        robot_out["seed"] = int(sampling_default_seed) + robot_idx

        output["robots"][robot_id] = robot_out

    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-robots",
        type=int,
        required=True,
        help="Number of robots to generate profiles for",
    )
    parser.add_argument(
        "--homogeneous-robot-profile",
        required=True,
        help=(
            "Path to homogeneous robot profile JSON/JSONC containing: "
            "uplink_median_ms, uplink_sigma, downlink_median_ms, downlink_sigma, execution_horizon"
        ),
    )
    parser.add_argument("--max-uplink-median-ms", type=float, required=True)
    parser.add_argument("--max-uplink-sigma", type=float, required=True)
    parser.add_argument("--max-downlink-median-ms", type=float, required=True)
    parser.add_argument("--max-downlink-sigma", type=float, required=True)
    parser.add_argument("--max-execution-horizon", type=int, required=True)
    parser.add_argument(
        "--heterogeneity-k",
        type=float,
        required=True,
        help="Heterogeneity knob in [0, 1] where 0=homogeneous and 1=max generated heterogeneity",
    )
    parser.add_argument(
        "--action-chunk-broker-type",
        default="rtc",
        help="Broker type for experiment config: rtc or sync",
    )
    parser.add_argument(
        "--trials-per-robot",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--toxiproxy-api-url",
        default="http://127.0.0.1:8474",
    )
    parser.add_argument(
        "--toxiproxy-listen-host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--toxiproxy-listen-port-base",
        type=int,
        default=15000,
    )
    parser.add_argument(
        "--toxiproxy-server-arg",
        action="append",
        default=[],
        help="Optional repeatable server arg for toxiproxy.server_args",
    )
    parser.add_argument(
        "--sampling-default-seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--sampling-resample-every-requests",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output-config",
        required=True,
        help="Path to write generated experiment config JSON/JSONC",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    profile = load_homogeneous_robot_profile(
        pathlib.Path(args.homogeneous_robot_profile)
    )
    max_profile = _build_max_profile(
        profile,
        max_uplink_median_ms=float(args.max_uplink_median_ms),
        max_uplink_sigma=float(args.max_uplink_sigma),
        max_downlink_median_ms=float(args.max_downlink_median_ms),
        max_downlink_sigma=float(args.max_downlink_sigma),
        max_execution_horizon=int(args.max_execution_horizon),
    )
    output = build_output_config(
        num_robots=int(args.num_robots),
        homogeneous_profile=profile,
        max_profile=max_profile,
        heterogeneity_k=float(args.heterogeneity_k),
        action_chunk_broker_type=str(args.action_chunk_broker_type),
        trials_per_robot=int(args.trials_per_robot),
        toxiproxy_api_url=str(args.toxiproxy_api_url),
        toxiproxy_listen_host=str(args.toxiproxy_listen_host),
        toxiproxy_listen_port_base=int(args.toxiproxy_listen_port_base),
        toxiproxy_server_args=[str(x) for x in args.toxiproxy_server_arg],
        sampling_default_seed=int(args.sampling_default_seed),
        sampling_resample_every_requests=int(args.sampling_resample_every_requests),
    )

    output_path = pathlib.Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    load_experiment_config(output_path)
    print(f"Wrote generated experiment config to {output_path}")


if __name__ == "__main__":
    main()
