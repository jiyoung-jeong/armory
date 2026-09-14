"""Run isolated local LIBERO trials with one server and linked logs per trial.

The chosen GPU must be idle. CPU sampling and global GPU settings are untouched.
Run from the repository root; outputs must be new paths. See docs/local_batch_experiments.md.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
NSYS = pathlib.Path("/usr/local/cuda-13.2/bin/nsys")


def emit(event, **fields):
    print(json.dumps(dict(time=time.time(), event=event, **fields)), flush=True)


def gpu_processes(gpu):
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def server_process_groups(output_dir):
    """Find our exact server command, including Nsight-created process groups."""
    groups = set()
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if b"scripts.serve" not in args or b"--server.output-dir" not in args:
                continue
            index = args.index(b"--server.output-dir")
            if args[index + 1] == os.fsencode(output_dir):
                group = os.getpgid(int(entry.name))
                if group <= 1 or group == os.getpgrp():
                    raise RuntimeError("Refusing to manage the experiment runner's process group")
                groups.add(group)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return groups


def stop_group(proc, extra_groups=()):
    if proc is None:
        return
    groups = {proc.pid, *extra_groups}
    for sig, timeout in [(signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 3)]:
        for group in list(groups):
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                groups.discard(group)
        deadline = time.monotonic() + timeout
        while groups and time.monotonic() < deadline:
            proc.poll()
            for group in list(groups):
                try:
                    os.killpg(group, 0)
                except ProcessLookupError:
                    groups.discard(group)
            time.sleep(0.2)
        if not groups:
            break
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def wait_ready(server, expected_batch, timeout=360):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Server launcher exited: {server.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/metadata", timeout=1) as response:
                metadata = json.load(response)
            if metadata["max_batch_size"] != expected_batch:
                raise RuntimeError("Unexpected server on port 8080")
            return metadata
        except (OSError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("Server readiness timeout")


def trial(output, robots, batch, repeat, seconds, gpu, profile, resume=False):
    name = f"{'profile' if profile else 'run'}_r{robots}_b{batch}_rep{repeat}"
    dest = output / name
    if resume and (dest / "manifest.json").exists():
        prior = json.loads((dest / "manifest.json").read_text())
        expected = dict(
            robots=robots,
            max_batch_size=batch,
            repeat=repeat,
            seconds=seconds,
            gpu=gpu,
            profiling=profile,
            seed=7,
            status="complete",
        )
        if not all(prior.get(k) == v for k, v in expected.items()):
            raise ValueError(f"Cannot resume incomplete or differently configured trial: {dest}")
        emit("trial_skipped_complete", run=name)
        return
    dest.mkdir(parents=True, exist_ok=False)
    occupied = gpu_processes(gpu)
    if occupied:
        raise RuntimeError(f"GPU {gpu} already has compute processes: {occupied}")
    # A free GPU does not imply that this network port is available.
    import socket

    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 8080))
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OPENPI_DATA_HOME=str(ROOT / ".cache/openpi-private/cache"),
        MUJOCO_EGL_DEVICE_ID="0",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        LIBERO_CONFIG_PATH=str(ROOT / ".cache/libero"),
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        ARMORY_NVTX="1" if profile else "0",
    )
    server_args = [
        sys.executable,
        "-u",
        "-m",
        "scripts.serve",
        "--json-path",
        f"configs/server/local_libero_b{batch}.json",
        "--server.output-dir",
        str(dest / "policy"),
        "--log-dir",
        str(dest / "server_logs"),
        "policy:default",
    ]
    session = f"armory_{name}_{os.getpid()}"
    if profile:
        server_args = [
            str(NSYS),
            "profile",
            "--start-later=true",
            f"--session-new={session}",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace=cuda,nvtx",
            "--trace-fork-before-exec=true",
            "--kill=none",
            f"--output={dest / 'timeline'}",
            *server_args,
        ]
    client_args = [
        sys.executable,
        "-u",
        "-m",
        "scripts.run",
        "--json-path",
        f"configs/local_{robots}_robots_libero.json",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--experiment-config.time-limit",
        str(seconds),
        "--output-dir",
        str(dest / "client"),
        "--server-log-dir",
        str(dest / "policy/server"),
    ]
    manifest = dict(
        name=name,
        robots=robots,
        max_batch_size=batch,
        repeat=repeat,
        seconds=seconds,
        seed=7,
        gpu=gpu,
        profiling=profile,
        server_command=server_args,
        client_command=client_args,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        status="starting",
        started_at=time.time(),
    )
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    server = client = None
    server_groups = set()
    capture_start = capture_end = None
    with (
        (dest / "server.stdout.log").open("w") as sf,
        (dest / "client.stdout.log").open("w") as cf,
        (dest / "gpu_samples.jsonl").open("w") as gf,
    ):
        try:
            server = subprocess.Popen(
                server_args,
                cwd=ROOT,
                env=env,
                stdout=sf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            emit("server_start", run=name, pid=server.pid)
            manifest["server_metadata"] = wait_ready(server, batch)
            server_groups = server_process_groups(dest / "policy")
            manifest["server_process_groups"] = sorted(server_groups)
            emit("server_ready", run=name)
            client = subprocess.Popen(
                client_args,
                cwd=ROOT,
                env=env,
                stdout=cf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            launch_time = time.monotonic()
            next_sample = 0
            first_request_seen = None
            while client.poll() is None:
                now = time.monotonic()
                if now - launch_time > seconds + 180:
                    raise TimeoutError("Client rollout/save timeout")
                if now >= next_sample:
                    processes = gpu_processes(gpu)
                    foreign = []
                    for pid in processes:
                        try:
                            if os.getpgid(pid) not in {server.pid, client.pid, *server_groups}:
                                foreign.append(pid)
                        except ProcessLookupError:
                            pass
                    if foreign:
                        raise RuntimeError(f"Another GPU workload appeared: {foreign}")
                    sample = subprocess.check_output(
                        [
                            "nvidia-smi",
                            f"--id={gpu}",
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=15,
                    ).strip()
                    gf.write(
                        json.dumps(dict(time=time.time(), value=sample, compute_pids=processes))
                        + "\n"
                    )
                    gf.flush()
                    next_sample = now + 5
                if profile:
                    events = dest / "policy/server/events.jsonl"
                    if first_request_seen is None and events.exists() and events.stat().st_size > 0:
                        first_request_seen = now
                    if (
                        capture_start is None
                        and first_request_seen is not None
                        and now - first_request_seen >= 5
                    ):
                        subprocess.run(
                            [
                                str(NSYS),
                                "start",
                                f"--session={session}",
                                "--sample=none",
                                "--cpuctxsw=none",
                            ],
                            check=True,
                            stdout=sf,
                            stderr=subprocess.STDOUT,
                            timeout=30,
                        )
                        capture_start = time.monotonic()
                        emit("capture_start", run=name)
                    if (
                        capture_start is not None
                        and capture_end is None
                        and now - capture_start >= 15
                    ):
                        subprocess.run(
                            [str(NSYS), "stop", f"--session={session}"],
                            check=True,
                            stdout=sf,
                            stderr=subprocess.STDOUT,
                            timeout=90,
                        )
                        capture_end = time.monotonic()
                        emit("capture_stop", run=name)
                elif server.poll() is not None:
                    raise RuntimeError("Server exited during rollout")
                time.sleep(0.5)
            if client.returncode != 0:
                raise RuntimeError(f"Client failed with exit code {client.returncode}")
            if profile and capture_end is None:
                raise RuntimeError("Client finished without a complete Nsight capture")
            manifest.update(status="complete", finished_at=time.time())
            emit("trial_complete", run=name)
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc), finished_at=time.time())
            raise
        finally:
            stop_group(client)
            stop_group(server, server_groups)
            if profile:
                subprocess.run(
                    [str(NSYS), "shutdown", f"--session={session}"],
                    stdout=sf,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
            (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["profile", "repeat", "scale"], required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Skip matching completed trials")
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats < 1:
        parser.error("seconds and repeats must be positive")
    os.chdir(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "profile":
        trial(output, 2, 2, 0, 35, args.gpu, True, resume=args.resume)
    else:
        for rep in range(1, args.repeats + 1):
            if args.phase == "repeat":
                sizes = [1, 2] if rep % 2 else [2, 1]
            else:
                sizes = [1, 2, 4]
                offset = (rep - 1) % len(sizes)
                sizes = sizes[offset:] + sizes[:offset]
            for size in sizes:
                trial(
                    output,
                    2 if args.phase == "repeat" else 4,
                    size,
                    rep,
                    args.seconds,
                    args.gpu,
                    False,
                    resume=args.resume,
                )
    emit("phase_complete", phase=args.phase)


if __name__ == "__main__":
    main()
