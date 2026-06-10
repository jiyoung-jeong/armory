"""Live in-sweep diagnostic for the downstream-latency-toxic bug.

Toxiproxy applies both directions correctly in isolation, yet real sweeps
under-inject the server->client (downstream) delay. This probe attaches to a
RUNNING case and inspects toxiproxy's live state so you can see whether, during
real traffic, the downstream latency toxic is (1) actually present on each
proxy, (2) set to the configured value, and (3) staying that way (not being
removed/overwritten). It is read-only: it only issues GETs to the toxiproxy
REST API and sends NO traffic to the policy server.

It also records topology (local hostname vs the configured server host), since a
client/server split across nodes is the one remaining unreplicated factor.

Usage (run in a THIRD terminal while a case is executing):
    uv run python scripts/interactive/probe_toxiproxy.py \\
        --case-dir experiments/sweeps/interactive/<stamp>/<run_id> \\
        --duration 60 --interval 2

Or point it straight at the toxiproxy API:
    uv run python scripts/interactive/probe_toxiproxy.py \\
        --api-url http://127.0.0.1:8474 --num-robots 10 --duration 60
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import time

import requests


def _load_case(case_dir: pathlib.Path):
    """Pull toxiproxy api/listen/robot info + server host from a case dir."""
    exp = json.loads((case_dir / "experiment_config.json").read_text())
    toxi = exp.get("toxiproxy", {})
    robots = exp.get("robots", {})
    cargs_path = case_dir / "client_args.json"
    server_host = None
    if cargs_path.exists():
        server_host = json.loads(cargs_path.read_text()).get("host")
    # expected per-robot downlink/uplink medians
    expected = {}
    for rid, rc in robots.items():
        expected[rid] = (
            float(rc.get("uplink_median_ms", 0.0)),
            float(rc.get("downlink_median_ms", 0.0)),
        )
    return {
        "api_url": str(toxi.get("api_url", "http://127.0.0.1:8474")).rstrip("/"),
        "listen_host": str(toxi.get("listen_host", "127.0.0.1")),
        "listen_port_base": int(toxi.get("listen_port_base", 18080)),
        "num_robots": int(exp.get("experiment", {}).get("num_robots", len(robots))),
        "server_host": server_host,
        "expected": expected,
    }


def _toxic_latency(toxics, stream):
    """Return the latency-toxic value for a stream, or None if absent."""
    for t in toxics:
        if t.get("type") == "latency" and t.get("stream") == stream:
            return int(t.get("attributes", {}).get("latency"))
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--case-dir", type=pathlib.Path, default=None,
                   help="A running case dir; reads api_url/ports/expected latencies from it.")
    p.add_argument("--api-url", default=None, help="Toxiproxy API (overrides case-dir).")
    p.add_argument("--num-robots", type=int, default=None)
    p.add_argument("--proxy-prefix", default="openpi_robot_", help="Proxy name prefix.")
    p.add_argument("--proxy-suffix", default="_proxy")
    p.add_argument("--duration", type=float, default=60.0, help="Total probe seconds.")
    p.add_argument("--interval", type=float, default=2.0, help="Seconds between polls.")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Per-poll JSONL log (default: <case-dir>/toxic_probe.jsonl).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    info = None
    if args.case_dir:
        info = _load_case(args.case_dir.resolve())
    api_url = (args.api_url or (info and info["api_url"]) or "http://127.0.0.1:8474").rstrip("/")
    num_robots = args.num_robots or (info and info["num_robots"]) or 1
    expected = (info and info["expected"]) or {}
    server_host = info and info["server_host"]
    out = args.out or (args.case_dir / "toxic_probe.jsonl" if args.case_dir else
                       pathlib.Path("toxic_probe.jsonl"))

    local_host = socket.gethostname()
    print(f"toxiproxy api : {api_url}")
    print(f"local host    : {local_host}")
    print(f"server host   : {server_host}  "
          f"({'SAME node' if server_host and server_host.split('.')[0] == local_host.split('.')[0] else 'DIFFERENT node -> topology/clock factor' if server_host else 'unknown'})")
    print(f"probing {num_robots} proxies every {args.interval}s for {args.duration}s\n")

    proxy_names = [f"{args.proxy_prefix}{i}{args.proxy_suffix}" for i in range(num_robots)]
    sess = requests.Session()
    deadline = time.monotonic() + args.duration
    n_polls = 0
    n_ok_polls = 0
    # track, per proxy, whether downstream was ever missing or wrong
    down_missing = {pn: 0 for pn in proxy_names}
    down_values: dict[str, set] = {pn: set() for pn in proxy_names}
    up_values: dict[str, set] = {pn: set() for pn in proxy_names}

    with out.open("a", encoding="utf-8") as fh:
        while time.monotonic() < deadline:
            n_polls += 1
            ts = time.time()
            try:
                proxies = sess.get(f"{api_url}/proxies", timeout=3).json()
            except Exception as exc:  # noqa: BLE001
                print(f"[poll {n_polls}] toxiproxy API unreachable: {exc}")
                time.sleep(args.interval)
                continue
            n_ok_polls += 1

            row = {"ts": ts, "poll": n_polls, "proxies": {}}
            line_bits = []
            for i, pn in enumerate(proxy_names):
                pdata = proxies.get(pn) if isinstance(proxies, dict) else None
                if pdata is None:
                    row["proxies"][pn] = {"present": False}
                    line_bits.append(f"r{i}:NO-PROXY")
                    continue
                toxics = pdata.get("toxics", [])
                up = _toxic_latency(toxics, "upstream")
                down = _toxic_latency(toxics, "downstream")
                enabled = pdata.get("enabled")
                row["proxies"][pn] = {"present": True, "enabled": enabled,
                                      "upstream": up, "downstream": down}
                if up is not None:
                    up_values[pn].add(up)
                if down is None:
                    down_missing[pn] += 1
                else:
                    down_values[pn].add(down)
                exp_up, exp_down = expected.get(f"robot_{i}", (None, None))
                flag = ""
                if down is None:
                    flag = " <DOWN MISSING>"
                elif exp_down is not None and abs(down - exp_down) > 0.5:
                    flag = f" <DOWN={down}!=cfg {exp_down:.0f}>"
                line_bits.append(f"r{i}:up={up} down={down}{flag}")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            # compact console line: only show first 4 robots inline
            print(f"[poll {n_polls:>3}] " + "  ".join(line_bits[:4]) +
                  (" ..." if len(line_bits) > 4 else ""))
            time.sleep(args.interval)

    print("\n===== summary =====")
    if n_ok_polls == 0:
        print("Toxiproxy API was never reachable -- is a case actually running, and")
        print(f"is {api_url} the right API? No conclusion about the toxics.")
        print(f"\nPer-poll log: {out}")
        return
    any_problem = False
    for i, pn in enumerate(proxy_names):
        exp_up, exp_down = expected.get(f"robot_{i}", (None, None))
        dv = sorted(down_values[pn])
        uv = sorted(up_values[pn])
        miss = down_missing[pn]
        problem = (miss > 0) or (exp_down is not None and any(abs(v - exp_down) > 0.5 for v in dv)) or (not dv)
        any_problem = any_problem or problem
        tag = "  <-- PROBLEM" if problem else ""
        print(f"  robot_{i}: upstream set(s)={uv} expected~{exp_up}; "
              f"downstream set(s)={dv} expected~{exp_down}; "
              f"down-missing polls={miss}/{n_polls}{tag}")
    print()
    if any_problem:
        print("Downstream toxic was missing or set to the wrong value on the live")
        print("proxy -> the bug is in how the toxic is installed/maintained during")
        print("real traffic (not toxiproxy itself).")
    else:
        print("Downstream toxic was present and correct on every poll. The configured")
        print("delay IS installed on the live proxy, so the lost latency is NOT a")
        print("missing/overwritten toxic -> look at topology (client/server on")
        print("different nodes => proxy->server hop + distributed clocks) or at how")
        print("the response bytes traverse the proxy under real concurrent traffic.")
    print(f"\nPer-poll log: {out}")


if __name__ == "__main__":
    main()
