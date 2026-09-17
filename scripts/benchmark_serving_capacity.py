"""Open-loop 2 Hz snapshot replay through the real server and production action broker.

A new output directory is required. No GPU settings are modified. This measures
serving and action availability, not simulated task success. Each robot has its
own process, periodic control ticks, and independently phased periodic requests.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.local_batch_sweep import ROOT, emit, gpu_processes, stop_group, wait_ready


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def request_tick(step, control_hz, request_hz):
    ratio = control_hz / request_hz
    if ratio < 1 or not ratio.is_integer():
        raise ValueError("control_hz must be an integer multiple of request_hz")
    return step % int(ratio) == 0


def robot_worker(index, case, ready, go, origin):
    import numpy as np

    from armory_client.action_chunk_broker import ActionChunkBroker
    from armory_client.client import BidirectionalWebsocket
    from evaluation.agents.policy_agent import PolicyAgent
    from evaluation.envs.mock import MockEnvironment, MockObservation

    class PeriodicWebsocket(BidirectionalWebsocket):
        # Keep the production agent/broker and ACK path, only gate observations.
        # Sending never waits for the previous inference response.
        def send(self, obs, *args, **kwargs):
            if not self.enabled or not request_tick(
                obs.step, case["control_hz"], case["request_hz"]
            ):
                return None
            return super().send(obs, *args, **kwargs)

    dest = Path(case["output"])
    records = json.loads((Path(case["inputs"]) / "metadata.json").read_text())
    record = records[index % len(records)]
    path = Path(case["inputs"]) / f"robot_{record['robot']}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Input checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        values = {
            key: value.item() if value.ndim == 0 else value.copy() for key, value in archive.items()
        }
    values = {key: values[key] for key in ["state", "image", "wrist_image", "prompt"]}
    ws = PeriodicWebsocket(
        robot_id=f"robot_{index}", host="127.0.0.1", port=8080, control_hz=case["control_hz"]
    )
    ws.enabled = True
    agent = None
    try:
        ws.connect()
        agent = PolicyAgent(
            ws,
            ActionChunkBroker(1, 10),
            MockEnvironment().create_null_action,
            dest / f"broker_{index}.jsonl",
        )
        ready.put(index)
        if not go.wait(timeout=120):
            raise TimeoutError("Fleet start barrier timed out")
        base = origin.value
        phase = case["offsets"][index]
        measure_start = base + case["warmup"] + 1 / case["request_hz"]
        measure_end = measure_start + case["seconds"]
        send_end = measure_end + case["tail"]
        finish = send_end + case["drain"]
        step = 0
        with (dest / f"ticks_{index}.jsonl").open("x") as ticks:
            while True:
                scheduled = base + phase + step / case["control_hz"]
                if scheduled >= finish:
                    break
                time.sleep(max(0, scheduled - time.monotonic()))
                now = time.monotonic()
                lateness = now - scheduled
                if lateness > max(0.2, 2 / case["control_hz"]):
                    raise RuntimeError(f"Load generator stalled for {lateness:.3f}s")
                ws.enabled = now < send_end
                if not agent._background_thread.is_alive():
                    raise RuntimeError("Policy receiver thread failed")
                action = agent.get_action(MockObservation(step=step, **values))
                if action.action.shape != (7,) or not np.isfinite(action.action).all():
                    raise ValueError("Invalid returned action")
                ticks.write(
                    json.dumps(
                        dict(
                            step=step, scheduled=scheduled, actual=now, lateness_ms=lateness * 1000
                        )
                    )
                    + "\n"
                )
                step += 1
        chunks = agent.action_chunks
        with (dest / f"chunks_{index}.jsonl").open("x") as log:
            for chunk in chunks:
                row = {
                    f.name: getattr(chunk, f.name)
                    for f in dataclasses.fields(chunk)
                    if f.name not in ("actions", "noise")
                }
                row["action_shape"] = list(chunk.actions.shape)
                row["action_dtype"] = str(chunk.actions.dtype)
                log.write(json.dumps(row) + "\n")
        write_json(
            dest / f"robot_{index}.json",
            dict(
                robot=index,
                episode_id=agent.episode_id,
                input=record,
                offset=phase,
                ticks=step,
                chunks=len(chunks),
            ),
        )
    finally:
        if agent is not None:
            agent.close()
        else:
            ws.close()


def client(case):
    ctx = mp.get_context("spawn")
    ready, go, origin = ctx.Queue(), ctx.Event(), ctx.Value("d", 0.0)
    workers = [
        ctx.Process(target=robot_worker, args=(i, case, ready, go, origin))
        for i in range(case["robots"])
    ]
    try:
        for process in workers:
            process.start()
        for _ in workers:
            ready.get(timeout=90)
        origin.value = time.monotonic() + 0.5
        # Single conversion anchors monotonic workload windows to same-host wall logs.
        epoch = time.time() + origin.value - time.monotonic()
        case.update(
            start_epoch=epoch,
            measure_start=epoch + case["warmup"] + 1 / case["request_hz"],
            measure_end=epoch + case["warmup"] + 1 / case["request_hz"] + case["seconds"],
        )
        write_json(Path(case["output"]) / "case.json", case)
        go.set()
        deadline = (
            time.monotonic() + case["warmup"] + case["seconds"] + case["tail"] + case["drain"] + 30
        )
        for process in workers:
            process.join(timeout=max(0, deadline - time.monotonic()))
            if process.exitcode != 0:
                raise RuntimeError(f"Robot process {process.pid} failed: {process.exitcode}")
        case["status"] = "complete"
        write_json(Path(case["output"]) / "case.json", case)
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=5)
        ready.close()


def check_gpu(gpu, own_groups):
    foreign = []
    for pid in gpu_processes(gpu):
        try:
            if os.getpgid(pid) not in own_groups:
                foreign.append(pid)
        except ProcessLookupError:
            pass
    if foreign:
        raise RuntimeError(f"Another GPU workload appeared: {foreign}")


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if gpu_processes(args.gpu):
        raise RuntimeError(f"GPU {args.gpu} is occupied")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 8080))
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENPI_DATA_HOME=str(ROOT / ".cache/openpi-private/cache"),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        ARMORY_NVTX="0",
        ARMORY_RECORD_PREDICTIONS="0",
        MPLBACKEND="Agg",
    )
    config = json.loads((ROOT / "configs/server/local_libero_b2.json").read_text())
    config["server"]["max_batch_size"] = args.batch
    config["server"]["scheduler"]["scheduling_algorithm"] = args.algorithm
    write_json(root / "server_input.json", config)
    manifest = dict(
        vars(args),
        output=str(root),
        inputs=str(args.inputs.resolve()),
        status="starting",
        started_at=time.time(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        input_metadata=json.loads((args.inputs / "metadata.json").read_text()),
        slo_reference_ms=156,
        slo_attainment_target=0.98,
        paper="https://arxiv.org/html/2609.12075v1#S5.SS1",
        scope="pi05_libero snapshot replay; real policy/WS/broker; no physics",
    )
    write_json(root / "manifest.json", manifest)
    server = fleet = telemetry = None
    try:
        server_args = [
            sys.executable,
            "-u",
            "-m",
            "scripts.serve",
            "--json-path",
            str(root / "server_input.json"),
            "--server.output-dir",
            str(root / "policy"),
            "--log-dir",
            str(root / "server_logs"),
            "policy:default",
        ]
        with (root / "server.stdout.log").open("x") as log:
            server = subprocess.Popen(
                server_args,
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with (root / "telemetry.stdout.log").open("x") as log:
            telemetry = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.record_gpu_telemetry",
                    str(root),
                    "--gpu",
                    str(args.gpu),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        emit("server_start", pid=server.pid, output=str(root))
        manifest["metadata"] = wait_ready(server, args.batch)
        manifest["status"] = "running"
        write_json(root / "manifest.json", manifest)
        emit("server_ready", metadata=manifest["metadata"])
        # Cyclic order avoids confounding every high-load trial with a hot late run.
        for repeat in range(args.repeats):
            combinations = [(n, hz) for hz in args.control_hz for n in args.robots]
            shift = repeat % len(combinations)
            combinations = combinations[shift:] + combinations[:shift]
            offsets = [
                random.Random(args.seed + repeat * 10000 + i).random() / args.request_hz
                for i in range(max(args.robots))
            ]
            for robots, hz in combinations:
                request_tick(0, hz, args.request_hz)
                if server.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly")
                if telemetry.poll() is not None:
                    raise RuntimeError("GPU telemetry exited unexpectedly")
                check_gpu(args.gpu, {server.pid})
                with urllib.request.urlopen(
                    urllib.request.Request("http://127.0.0.1:8080/reset", data=b"", method="POST"),
                    timeout=5,
                ) as response:
                    json.load(response)
                time.sleep(0.3)
                dest = root / f"r{robots}_hz{hz:g}_rep{repeat}"
                dest.mkdir()
                case = dict(
                    output=str(dest),
                    inputs=str(args.inputs.resolve()),
                    robots=robots,
                    control_hz=hz,
                    request_hz=args.request_hz,
                    offsets=offsets[:robots],
                    max_batch_size=args.batch,
                    repeat=repeat,
                    seed=args.seed,
                    warmup=args.warmup,
                    seconds=args.seconds,
                    tail=1,
                    drain=2,
                    status="starting",
                    algorithm=args.algorithm,
                    slo_ms=156,
                    min_execution_horizon=1,
                    max_execution_horizon=10,
                )
                write_json(dest / "case.json", case)
                emit("case_start", name=dest.name)
                with (dest / "client.stdout.log").open("x") as log:
                    fleet = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "scripts.benchmark_serving_capacity",
                            "client",
                            str(dest / "case.json"),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                timeout = time.monotonic() + args.seconds + args.warmup + 150
                while fleet.poll() is None:
                    time.sleep(2)
                    if time.monotonic() > timeout or server.poll() is not None:
                        raise RuntimeError("Fleet timeout or server failure")
                    check_gpu(args.gpu, {server.pid, fleet.pid})
                if fleet.returncode:
                    raise RuntimeError(f"Fleet exited {fleet.returncode}: {dest}")
                fleet = None
                from scripts.analyze_serving_capacity import analyze_case

                summary = analyze_case(dest, root / "policy/server")
                write_json(dest / "summary.json", summary)
                emit("case_complete", **summary)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["finished_at"] = time.time()
        write_json(root / "manifest.json", manifest)
        stop_group(fleet)
        stop_group(server)
        if telemetry is not None:
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_group(telemetry)
        emit("finished", status=manifest["status"], output=str(root))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "client":
        return client(json.loads(Path(sys.argv[2]).read_text()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("output/static_inputs_20260917"))
    parser.add_argument("--robots", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    parser.add_argument("--control-hz", type=float, nargs="+", default=[20.0])
    parser.add_argument("--request-hz", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--algorithm", default="lookahead-actions")
    args = parser.parse_args()
    if min(args.robots) < 1 or min(args.control_hz) <= 0 or args.request_hz <= 0:
        parser.error("Robot counts and frequencies must be positive")
    if args.seconds <= 0 or args.warmup < 0 or args.repeats < 1 or args.batch < 1:
        parser.error("Invalid duration, repeats or batch")
    run(args)


if __name__ == "__main__":
    main()
