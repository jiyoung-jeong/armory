"""Matched 4-robot trials after mirror alignment: lookahead B2/B3 and tuned RR B2."""

import argparse
import json
import os
import time
from pathlib import Path

from scripts.local_batch_sweep import ROOT, emit, trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    conditions = [("lookahead-actions", 2), ("lookahead-actions", 3), ("round-robin", 2)]
    for rep in range(1, args.repeats + 1):
        # Cyclic orders put each condition in each position once over 3 repeats.
        offset = (rep - 1) % len(conditions)
        for algorithm, cap in conditions[offset:] + conditions[:offset]:
            trial(
                root / algorithm,
                4,
                cap,
                rep,
                args.seconds,
                args.gpu,
                False,
                resume=args.resume,
                algorithm=algorithm,
                record_predictions=True,
            )
    (root / "experiments_finished.json").write_text(
        json.dumps(
            dict(
                status="complete",
                trials=3 * args.repeats,
                seconds=args.seconds,
                conditions=conditions,
                finished_at=time.time(),
            ),
            indent=2,
        )
    )
    emit("scheduler_followup_complete")


if __name__ == "__main__":
    main()
