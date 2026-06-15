# FIXME: this script is broken after AsyncWebsocketClientPolicy was deleted
"""WebSocket variant of the toxiproxy downstream-latency repro.

Raw TCP showed the downstream latency toxic works perfectly (single conn, per-
request re-POST, 10 concurrent clients). The real openpi stack differs only in
the WebSocket layer, so this repro uses the SAME libraries as production:
a starlette/uvicorn WebSocket server and a ``websockets`` client, behind a real
ToxiproxyController-managed proxy.

Two client conditions:
  WS-SYNC : websockets.sync.client, one connection, synchronous req/resp.
  WS-ASYNC: websockets.asyncio.client with an N-connection pool (mirrors
            AsyncWebsocketClientPolicy in production), each request grabs a
            free connection.

All timestamps are on one machine/clock, so up/down/rtt are offset-free:
    up   = server_arrival - client_send
    down = client_recv    - server_send
    rtt  = client_recv    - client_send

Run:
    uv run python scripts/interactive/_toxiproxy_repro_ws.py
"""

from __future__ import annotations

import asyncio
import pathlib
import socket
import struct
import sys
import threading
import time

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

_HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "armory-client" / "src"))

import websockets.asyncio.client  # noqa: E402
import websockets.sync.client  # noqa: E402
from armory_evaluation.network_emulation.toxiproxy import (  # noqa: E402
    DEFAULT_TOXIC_DOWNSTREAM,
    DEFAULT_TOXIC_UPSTREAM,
    ToxiproxyController,
)

TOXIPROXY_BIN = "/coc/flash7/rbansal66/vvla/toxiproxy-server-linux-amd64"
LATENCY_MS = 100
INFER_S = 0.15
N_REQUESTS = 30
N_WARMUP = 3
MSG_BYTES = 200_000


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- server: starlette/uvicorn websocket echo with a fixed "inference" sleep
async def _ws_endpoint(websocket):
    await websocket.accept()
    try:
        while True:
            _ = await websocket.receive_bytes()
            t_arrival = time.time()
            await asyncio.sleep(INFER_S)
            t_send = time.time()
            await websocket.send_bytes(struct.pack("!dd", t_arrival, t_send))
    except Exception:
        pass


def _start_server(port: int) -> uvicorn.Server:
    app = Starlette(routes=[WebSocketRoute("/", _ws_endpoint)])
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", ws_max_size=16 * 1024 * 1024
    )
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server


def _pct(x):
    a = np.asarray(x)
    return f"p50={np.median(a):6.1f}  mean={a.mean():6.1f}  p95={np.percentile(a, 95):6.1f}"


def _report(label, up, down, rtt):
    print(f"\n[{label}]")
    print(f"  upstream   (client->server): {_pct(up)}")
    print(f"  downstream (server->client): {_pct(down)}   <-- configured {LATENCY_MS} ms")
    print(f"  client RTT (offset-free):    {_pct(rtt)}")


def _reset_toxics(ctrl, proxy):
    for t in (DEFAULT_TOXIC_UPSTREAM, DEFAULT_TOXIC_DOWNSTREAM):
        try:
            ctrl._request("DELETE", f"/proxies/{proxy}/toxics/{t}", expected=(200, 204, 404))
        except Exception:
            pass
    ctrl.set_latency(proxy, LATENCY_MS, LATENCY_MS)
    time.sleep(0.2)


def _run_ws_sync(ctrl, proxy, host, port):
    _reset_toxics(ctrl, proxy)
    payload = b"\x00" * MSG_BYTES
    up, down, rtt = [], [], []
    with websockets.sync.client.connect(f"ws://{host}:{port}/", max_size=16 * 1024 * 1024) as ws:
        for i in range(N_REQUESTS + N_WARMUP):
            t_req = time.time()
            ws.send(payload)
            body = ws.recv()
            t_recv = time.time()
            t_arrival, t_send = struct.unpack("!dd", body)
            if i >= N_WARMUP:
                up.append((t_arrival - t_req) * 1000.0)
                down.append((t_recv - t_send) * 1000.0)
                rtt.append((t_recv - t_req) * 1000.0)
    _report("WS-SYNC: 1 connection", up, down, rtt)


def _run_ws_async(ctrl, proxy, host, port, n_pool=10):
    _reset_toxics(ctrl, proxy)
    payload = b"\x00" * MSG_BYTES

    async def go():
        pool = await asyncio.gather(
            *[
                websockets.asyncio.client.connect(f"ws://{host}:{port}/", max_size=16 * 1024 * 1024)
                for _ in range(n_pool)
            ]
        )
        up, down, rtt = [], [], []

        async def one(ws, n):
            for i in range(n):
                t_req = time.time()
                await ws.send(payload)
                body = await ws.recv()
                t_recv = time.time()
                t_arrival, t_send = struct.unpack("!dd", body)
                up.append((t_arrival - t_req) * 1000.0)
                down.append((t_recv - t_send) * 1000.0)
                rtt.append((t_recv - t_req) * 1000.0)

        # warmup (untimed) then timed, all pool conns concurrently
        await asyncio.gather(*[one(ws, N_WARMUP) for ws in pool])
        up.clear()
        down.clear()
        rtt.clear()
        await asyncio.gather(*[one(ws, N_REQUESTS) for ws in pool])
        for ws in pool:
            await ws.close()
        return up, down, rtt

    up, down, rtt = asyncio.run(go())
    _report(f"WS-ASYNC: {n_pool}-connection pool (production client)", up, down, rtt)


def _run_ws_pipelined(ctrl, proxy, host, port):
    """One connection, client FIRES requests at 20Hz without awaiting each
    response; a concurrent reader stamps receive_time. Mimics the action-chunk
    broker's streaming (many in-flight requests, responses come back async)."""
    _reset_toxics(ctrl, proxy)
    payload = b"\x00" * MSG_BYTES
    n = N_REQUESTS + N_WARMUP

    async def go():
        ws = await websockets.asyncio.client.connect(
            f"ws://{host}:{port}/", max_size=16 * 1024 * 1024
        )
        sends: dict[int, float] = {}
        up, down, rtt = [], [], []
        recv_count = 0

        async def reader():
            nonlocal recv_count
            while recv_count < n:
                body = await ws.recv()
                t_recv = time.time()
                t_arrival, t_send = struct.unpack("!dd", body)
                k = recv_count
                if k >= N_WARMUP:
                    up.append((t_arrival - sends[k]) * 1000.0)
                    down.append((t_recv - t_send) * 1000.0)
                    rtt.append((t_recv - sends[k]) * 1000.0)
                recv_count += 1

        rt = asyncio.create_task(reader())
        for k in range(n):
            sends[k] = time.time()
            await ws.send(payload)
            await asyncio.sleep(0.05)  # 20 Hz control rate, do NOT await response
        await rt
        await ws.close()
        return up, down, rtt

    up, down, rtt = asyncio.run(go())
    _report("WS-PIPELINED: fire-at-20Hz, async reader (broker-like)", up, down, rtt)


def main() -> None:
    api_port = _free_port()
    listen_port = _free_port()
    srv_port = _free_port()
    host = "127.0.0.1"

    server = _start_server(srv_port)
    ctrl = ToxiproxyController(
        f"http://127.0.0.1:{api_port}",
        server_bin=TOXIPROXY_BIN,
        server_args=["-host", "127.0.0.1", "-port", str(api_port)],
    )
    print(
        f"toxiproxy api=127.0.0.1:{api_port}  proxy={host}:{listen_port}  ws-server={host}:{srv_port}"
    )
    print(
        f"config: latency={LATENCY_MS}ms each way, inference sleep={INFER_S * 1000:.0f}ms, "
        f"{N_REQUESTS} reqs/condition (starlette/uvicorn + websockets {websockets_version()})"
    )
    ctrl.start_server()
    proxy = "repro_ws_proxy"
    try:
        ctrl.create_proxy(proxy, listen=f"{host}:{listen_port}", upstream=f"127.0.0.1:{srv_port}")
        _run_ws_sync(ctrl, proxy, host, listen_port)
        _run_ws_async(ctrl, proxy, host, listen_port, n_pool=10)
        _run_ws_pipelined(ctrl, proxy, host, listen_port)
    finally:
        try:
            ctrl.delete_proxy(proxy)
        except Exception:
            pass
        ctrl.stop_server()
        server.should_exit = True
        time.sleep(0.3)

    print("\nVerdict: if WS downstream collapses (~0-30ms) while raw-TCP was ~100,")
    print("  the bug is in how toxiproxy delays the websocket server->client frames.")


def websockets_version():
    import websockets

    return websockets.__version__


if __name__ == "__main__":
    main()
