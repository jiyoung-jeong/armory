"""Minimal, self-contained repro for the "downstream latency toxic is inert" bug.

Isolates toxiproxy from openpi/LIBERO: a raw-TCP echo "server" (with a fixed
sleep mimicking inference) behind a real ToxiproxyController-managed proxy, and a
synchronous client. The toxiproxy latency toxic operates at the TCP byte layer,
so this exercises the same proxy path the openpi websocket uses.

We measure offset-free one-way delays (all timestamps on one machine/clock):
    up   = server_arrival - client_send
    down = client_recv    - server_send
    rtt  = client_recv    - client_send
and run three conditions to localize the bug:

  A. set_latency ONCE at startup, never re-applied.
  B. set_latency re-applied before EVERY request (mimics before_send with
     resample_every_requests=1 -- the current production behavior).
  C. re-applied before every request but via an attributes-ONLY update body
     (candidate fix that avoids resending name/type/stream).

Run:
    uv run python scripts/interactive/_toxiproxy_repro.py
"""

from __future__ import annotations

import pathlib
import socket
import struct
import sys
import threading
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "armory-client" / "src"))

from armory_evaluation.toxiproxy import (  # noqa: E402
    DEFAULT_TOXIC_DOWNSTREAM,
    DEFAULT_TOXIC_UPSTREAM,
    ToxiproxyController,
)

TOXIPROXY_BIN = "/coc/flash7/rbansal66/vvla/toxiproxy-server-linux-amd64"
LATENCY_MS = 100  # configured each direction
INFER_S = 0.15  # server-side "inference" sleep (keeps resp in down-queue)
N_REQUESTS = 40
N_WARMUP = 3
MSG_BYTES = 200_000  # ~obs-sized upstream payload


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- wire format: [u32 len][8 bytes f64 t_arrival][8 bytes f64 t_send][payload]
def _recv_exactly(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def _handle_conn(conn: socket.socket, stop: threading.Event) -> None:
    with conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not stop.is_set():
                header = _recv_exactly(conn, 4)
                (plen,) = struct.unpack("!I", header)
                _ = _recv_exactly(conn, plen)
                t_arrival = time.time()
                time.sleep(INFER_S)  # mimic inference
                t_send = time.time()
                body = struct.pack("!dd", t_arrival, t_send)
                conn.sendall(struct.pack("!I", len(body)) + body)
        except (ConnectionError, OSError):
            pass


def _echo_server(server_sock: socket.socket, stop: threading.Event) -> None:
    server_sock.settimeout(0.5)
    while not stop.is_set():
        try:
            conn, _ = server_sock.accept()
        except TimeoutError:
            continue
        except OSError:
            break
        threading.Thread(target=_handle_conn, args=(conn, stop), daemon=True).start()


def _run_condition(
    label: str,
    ctrl: ToxiproxyController,
    proxy: str,
    listen_host: str,
    listen_port: int,
    *,
    reapply,  # None | "full" | "attrs"
) -> None:
    # Fresh toxics for every condition.
    for t in (DEFAULT_TOXIC_UPSTREAM, DEFAULT_TOXIC_DOWNSTREAM):
        try:
            ctrl._request("DELETE", f"/proxies/{proxy}/toxics/{t}", expected=(200, 204, 404))
        except Exception:
            pass
    ctrl.set_latency(proxy, LATENCY_MS, LATENCY_MS)
    time.sleep(0.2)

    cli = socket.create_connection((listen_host, listen_port))
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = b"\x00" * MSG_BYTES

    up, down, rtt = [], [], []
    for i in range(N_REQUESTS + N_WARMUP):
        if reapply == "full":
            ctrl.set_latency(proxy, LATENCY_MS, LATENCY_MS)
        elif reapply == "attrs":
            _attrs_only_update(ctrl, proxy, LATENCY_MS)
        t_req = time.time()
        cli.sendall(struct.pack("!I", len(payload)) + payload)
        (plen,) = struct.unpack("!I", _recv_exactly(cli, 4))
        body = _recv_exactly(cli, plen)
        t_recv = time.time()
        t_arrival, t_send = struct.unpack("!dd", body)
        if i >= N_WARMUP:
            up.append((t_arrival - t_req) * 1000.0)
            down.append((t_recv - t_send) * 1000.0)
            rtt.append((t_recv - t_req) * 1000.0)
    cli.close()

    def p(x):
        a = np.asarray(x)
        return f"p50={np.median(a):6.1f}  mean={a.mean():6.1f}  p95={np.percentile(a, 95):6.1f}"

    print(f"\n[{label}]  reapply={reapply}")
    print(f"  upstream   (client->server): {p(up)}")
    print(f"  downstream (server->client): {p(down)}   <-- configured {LATENCY_MS} ms")
    print(f"  client RTT (offset-free):    {p(rtt)}")


def _client_worker(listen_host, listen_port, payload, out, idx):
    """One synchronous request/response client; appends (up,down,rtt) lists to out[idx]."""
    cli = socket.create_connection((listen_host, listen_port))
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    up, down, rtt = [], [], []
    try:
        for i in range(N_REQUESTS + N_WARMUP):
            t_req = time.time()
            cli.sendall(struct.pack("!I", len(payload)) + payload)
            (plen,) = struct.unpack("!I", _recv_exactly(cli, 4))
            body = _recv_exactly(cli, plen)
            t_recv = time.time()
            t_arrival, t_send = struct.unpack("!dd", body)
            if i >= N_WARMUP:
                up.append((t_arrival - t_req) * 1000.0)
                down.append((t_recv - t_send) * 1000.0)
                rtt.append((t_recv - t_req) * 1000.0)
    finally:
        cli.close()
    out[idx] = (up, down, rtt)


def _run_concurrent(label, ctrl, proxies, listen_host, echo_port, *, reapply):
    """K proxies + K parallel synchronous clients, like the real 10-robot run."""
    # Fresh toxics on every proxy.
    for proxy, lport in proxies:
        for t in (DEFAULT_TOXIC_UPSTREAM, DEFAULT_TOXIC_DOWNSTREAM):
            try:
                ctrl._request("DELETE", f"/proxies/{proxy}/toxics/{t}", expected=(200, 204, 404))
            except Exception:
                pass
        ctrl.set_latency(proxy, LATENCY_MS, LATENCY_MS)
    time.sleep(0.2)

    payload = b"\x00" * MSG_BYTES
    out: dict[int, tuple] = {}

    def worker(idx, proxy, lport):
        cli = socket.create_connection((listen_host, lport))
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        up, down, rtt = [], [], []
        try:
            for i in range(N_REQUESTS + N_WARMUP):
                if reapply == "full":
                    ctrl.set_latency(proxy, LATENCY_MS, LATENCY_MS)
                t_req = time.time()
                cli.sendall(struct.pack("!I", len(payload)) + payload)
                (plen,) = struct.unpack("!I", _recv_exactly(cli, 4))
                body = _recv_exactly(cli, plen)
                t_recv = time.time()
                t_arrival, t_send = struct.unpack("!dd", body)
                if i >= N_WARMUP:
                    up.append((t_arrival - t_req) * 1000.0)
                    down.append((t_recv - t_send) * 1000.0)
                    rtt.append((t_recv - t_req) * 1000.0)
        finally:
            cli.close()
        out[idx] = (up, down, rtt)

    threads = [
        threading.Thread(target=worker, args=(i, proxy, lport))
        for i, (proxy, lport) in enumerate(proxies)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    up = [v for i in out for v in out[i][0]]
    down = [v for i in out for v in out[i][1]]
    rtt = [v for i in out for v in out[i][2]]

    def p(x):
        a = np.asarray(x)
        return f"p50={np.median(a):6.1f}  mean={a.mean():6.1f}  p95={np.percentile(a, 95):6.1f}"

    print(f"\n[{label}]  {len(proxies)} concurrent clients, reapply={reapply}")
    print(f"  upstream   (client->server): {p(up)}")
    print(f"  downstream (server->client): {p(down)}   <-- configured {LATENCY_MS} ms")
    print(f"  client RTT (offset-free):    {p(rtt)}")


def _attrs_only_update(ctrl: ToxiproxyController, proxy: str, latency_ms: int) -> None:
    """Update both latency toxics sending ONLY the attributes (no name/type/stream)."""
    for t in (DEFAULT_TOXIC_UPSTREAM, DEFAULT_TOXIC_DOWNSTREAM):
        ctrl._request(
            "POST",
            f"/proxies/{proxy}/toxics/{t}",
            expected=(200, 201),
            json={"attributes": {"latency": int(latency_ms), "jitter": 0}},
        )


def main() -> None:
    api_port = _free_port()
    listen_port = _free_port()
    echo_port = _free_port()
    listen_host = "127.0.0.1"

    stop = threading.Event()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((listen_host, echo_port))
    srv.listen(8)
    th = threading.Thread(target=_echo_server, args=(srv, stop), daemon=True)
    th.start()

    ctrl = ToxiproxyController(
        f"http://127.0.0.1:{api_port}",
        server_bin=TOXIPROXY_BIN,
        server_args=["-host", "127.0.0.1", "-port", str(api_port)],
    )
    print(
        f"toxiproxy api=127.0.0.1:{api_port}  proxy={listen_host}:{listen_port}"
        f"  echo=127.0.0.1:{echo_port}"
    )
    print(
        f"config: latency={LATENCY_MS}ms each way, inference sleep={INFER_S * 1000:.0f}ms, "
        f"{N_REQUESTS} reqs/condition"
    )
    ctrl.start_server()
    proxy = "repro_proxy"
    n_concurrent = 10
    extra_proxies = [(f"repro_proxy_{i}", _free_port()) for i in range(n_concurrent)]
    try:
        ctrl.create_proxy(
            proxy, listen=f"{listen_host}:{listen_port}", upstream=f"127.0.0.1:{echo_port}"
        )
        _run_condition(
            "A: 1 conn, set once, never re-apply",
            ctrl,
            proxy,
            listen_host,
            listen_port,
            reapply=None,
        )
        _run_condition(
            "B: 1 conn, re-apply full body/req (production)",
            ctrl,
            proxy,
            listen_host,
            listen_port,
            reapply="full",
        )
        _run_condition(
            "C: 1 conn, re-apply attrs-only/req",
            ctrl,
            proxy,
            listen_host,
            listen_port,
            reapply="attrs",
        )

        # Concurrency: 10 proxies + 10 parallel synchronous clients, like 10 robots.
        for pn, lport in extra_proxies:
            ctrl.create_proxy(
                pn, listen=f"{listen_host}:{lport}", upstream=f"127.0.0.1:{echo_port}"
            )
        _run_concurrent(
            "D: 10 conns, set once", ctrl, extra_proxies, listen_host, echo_port, reapply=None
        )
        _run_concurrent(
            "E: 10 conns, re-apply full body/req (production)",
            ctrl,
            extra_proxies,
            listen_host,
            echo_port,
            reapply="full",
        )
    finally:
        for pn in [proxy, *[p for p, _ in extra_proxies]]:
            try:
                ctrl.delete_proxy(pn)
            except Exception:
                pass
        ctrl.stop_server()
        stop.set()
        srv.close()
        th.join(timeout=2)

    print("\nVerdict: compare downstream across A-E. Single-conn (A-C) already")
    print("  showed ~100ms. If D/E downstream collapses (~0-30ms) it's a")
    print("  concurrency/load issue on one toxiproxy server; otherwise the real")
    print("  gap is websocket-specific and needs a ws variant.")


if __name__ == "__main__":
    main()
