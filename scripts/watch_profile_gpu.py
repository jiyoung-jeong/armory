"""Watch a profiled worker from outside the Nsight process tree.

Writes a heartbeat and collision state; records read-only 1 Hz GPU telemetry.
Does not signal other processes or change device settings.
"""

import argparse
import json
import threading
import time
from pathlib import Path

from scripts.local_batch_sweep import gpu_processes
from scripts.record_gpu_telemetry import record


def watch(root, gpu):
    deadline = time.monotonic() + 120
    while not (root / "manifest.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Profile worker did not create its manifest")
        time.sleep(0.25)
    stop = threading.Event()
    error = []

    def guard():
        while not stop.is_set():
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["status"] in ("complete", "failed"):
                break
            try:
                others = sorted(set(gpu_processes(gpu)) - {manifest["pid"]})
                if others:
                    error.append(f"Other GPU compute processes detected: {others}")
            except Exception as exc:
                error.append(repr(exc))
            state = dict(
                checked_at=time.time(),
                worker_pid=manifest["pid"],
                error=error[-1] if error else None,
            )
            temporary = root / "gpu_guard.tmp"
            temporary.write_text(json.dumps(state))
            temporary.replace(root / "gpu_guard.json")
            if error:
                break
            stop.wait(1)

    thread = threading.Thread(target=guard, daemon=True)
    thread.start()
    try:
        record(root, gpu)
    finally:
        stop.set()
        thread.join(timeout=15)
    if error or thread.is_alive():
        raise RuntimeError(error or "GPU guard did not stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    watch(args.root, args.gpu)
