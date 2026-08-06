from __future__ import annotations

import datetime as dt
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.modal.app import CaseRunner, app
from scripts.modal.images import REMOTE_ROOT
from scripts.modal.utils import download_artifacts, write_rows
from scripts.sweep_cases import build_cases, parse_list_args


@app.local_entrypoint()
def main(
    mode: str = "mock",
    server_config: str = "",
    client_config: str = "",
    output_dir: str = "experiments/sweeps/modal",
    seeds: str = "7",
    stream_logs: bool = False,
) -> None:
    if mode not in {"gpu", "mock"}:
        raise SystemExit("--mode must be 'gpu' or 'mock'; a sweep needs a server.")
    if not server_config or not client_config:
        raise SystemExit("--server-config and --client-config are both required.")

    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d_%H%M%S")
    run_root = pathlib.Path(output_dir) / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    cases = build_cases(server_config, client_config, parse_list_args(seeds, cast=int))
    rows_by_run_id = {case.run_id: case.row(stamp) for case in cases}
    payloads = [
        {
            "mode": mode,
            "run_id": case.run_id,
            "run_dir": str(REMOTE_ROOT / stamp / case.run_id),
            "server_config": case.server.model_dump(mode="json"),
            "client_config": case.client_config(),
            "stream_logs": stream_logs,
        }
        for case in cases
    ]

    rows: list[dict[str, Any]] = []
    print(f"Running {len(cases)} case(s) in {mode} mode")
    for result in CaseRunner().run.map(payloads, order_outputs=False):
        row = {**rows_by_run_id.get(result.get("run_id", ""), {}), **result}
        rows.append(row)
        starvation = row.get("starvation_rate")
        rate = f"{starvation:.3f}" if isinstance(starvation, (int, float)) else "n/a"
        print(f"{row.get('status', '?')}: {row['run_id']} starvation={rate}")

    download_artifacts(stamp=stamp, out=run_root, rows=rows)
    results_csv = run_root / f"sweep_results_{stamp}.csv"
    write_rows(results_csv, rows)

    suspicious = [r for r in rows if r.get("timing_suspicious")]
    if suspicious:
        print(f"WARNING: {len(suspicious)} run(s) flagged for suspicious timings:")
        for row in suspicious:
            print(f"  {row['run_id']}: {row.get('timing_flags', '')}")

    print(
        f"\nPlot with:\n  uv run python scripts/visualization/plot_sweep.py --results {results_csv}"
    )
