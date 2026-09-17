"""Stream read-only nvidia-smi samples until the linked benchmark finishes.

A persistent nvidia-smi process avoids repeatedly paying driver initialization
cost. Uses nvidia-smi's own sample timestamp rather than pipe receipt time.
"""

import argparse
import csv
import datetime as dt
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

from scripts.benchmark_static_inference import GPU_FIELDS, NUMERIC_FIELDS


def record(root, gpu):
    fields = ["timestamp", *GPU_FIELDS]
    process = subprocess.Popen(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
            "--loop-ms=1000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    messages = queue.Queue()

    def receive():
        for line in process.stdout:
            messages.put(line)

    reader = threading.Thread(target=receive, daemon=True)
    reader.start()
    count = 0
    try:
        with (root / "gpu_stream.jsonl").open("x") as log:
            while True:
                manifest = json.loads((root / "manifest.json").read_text())
                if manifest["status"] in ("complete", "failed"):
                    break
                try:
                    line = messages.get(timeout=2)
                except queue.Empty:
                    if process.poll() is not None:
                        raise RuntimeError(f"nvidia-smi exited {process.returncode}")
                    continue
                values = next(csv.reader([line]))
                if len(values) != len(fields):
                    raise ValueError(f"Unexpected nvidia-smi output: {line!r}")
                raw = dict(zip(fields, [s.strip() for s in values], strict=True))
                sample = dict(
                    epoch=dt.datetime.strptime(
                        raw["timestamp"], "%Y/%m/%d %H:%M:%S.%f"
                    ).timestamp(),
                    received_epoch=time.time(),
                    timestamp=raw["timestamp"],
                )
                for key, label in NUMERIC_FIELDS.items():
                    sample[label] = float(raw[key])
                sample.update(
                    clock_event_reasons=raw["clocks_event_reasons.active"],
                    sw_thermal=raw["clocks_event_reasons.sw_thermal_slowdown"] == "Active",
                    hw_thermal=raw["clocks_event_reasons.hw_thermal_slowdown"] == "Active",
                )
                log.write(json.dumps(sample) + "\n")
                log.flush()
                count += 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        reader.join(timeout=2)
    (root / "gpu_stream_finished.json").write_text(
        json.dumps(
            dict(
                samples=count,
                finished_at=time.time(),
                sample_timestamp="nvidia-smi local timestamp, parsed in host timezone",
                host_timezone=time.tzname,
                target_interval_s=1,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    record(args.root, args.gpu)
