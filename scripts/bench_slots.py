"""Microbenchmark for the WS event-loop hot path.

The WS recv() coroutine in src/armory/serving/server.py:393-477 does, per
inbound InferRequest:

    1. await websocket.receive_bytes()        # yields, fine
    2. msgpack_numpy.unpackb(raw)             # sync, ~300 KB obs
    3. construct InferRequest(**msg)
    4. state.slots.write(SlotData(...))       # pickle SlotData -> shm copy
    5. await scheduler_sock.send_pyobj(...)   # pickle SlotRequest + zmq PUB
    6. state.metrics_store.record_request(...)

Past ~13 robots inbound p95 jumps from <50 ms to several hundred ms. We need
to know which of (2)/(4)/(5) is the actual hot path before refactoring any of
them. This script measures each independently against realistic payloads,
plus the end-to-end cost of the steps the event loop runs synchronously.

Run from repo root:

    PYTHONPATH=src:packages/armory-client/src \
        .venv/bin/python scripts/bench_slots.py

Sections:
  1. Correctness: write/read round-trip on RobotSlot (must keep passing
     after any slots refactor).
  2. slots.write / slots.read latency, single-process.
  3. msgpack_numpy.packb / unpackb latency on a full InferRequest.
  4. pickle.dumps(SlotRequest) + zmq PUB send latency.
  5. End-to-end: unpackb -> SlotData -> slots.write -> pickle SlotRequest,
     i.e. the chain of sync work the WS event loop runs per request.
  6. Cross-process throughput (writer parent, reader child).
  7. Event-loop stall proxy (1 ms ticker thread vs slots.write storm).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import statistics
import threading
import time
from dataclasses import dataclass

import numpy as np
import zmq

from armory.serving.schemas import SlotRequest
from armory.serving.slots import RobotSlot, RobotSlots, SlotData
from armory_client import msgpack_numpy
from armory_client.messages import InferRequest, InferType


# Matches sims.libero.mock_env.MockEnvironment.get_observation():
#   image:        uint8[224, 224, 3]   ~150 KB
#   wrist_image:  uint8[224, 224, 3]   ~150 KB
#   state:        float32[8]           tiny
#   step:         int
#   prompt:       str
IMAGE_SHAPE = (224, 224, 3)
STATE_DIM = 8


def _make_obs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "step": int(seed),
        "state": rng.standard_normal(STATE_DIM).astype(np.float32),
        # uint8 random fills the buffer so any short-write bug is visible.
        "image": rng.integers(0, 255, size=IMAGE_SHAPE, dtype=np.uint8),
        "wrist_image": rng.integers(0, 255, size=IMAGE_SHAPE, dtype=np.uint8),
        "prompt": "mock task",
    }


def _make_slot_data(seed: int = 0) -> SlotData:
    obs = _make_obs(seed)
    now = time.time()
    return SlotData(
        robot_id=f"robot_{seed % 32}",
        obs=obs,
        request_id=seed,
        arrival_timestamp=now,
        observation_step=seed,
        action_index_start=seed * 10,
        request_timestamp=now - 0.001,
        deadline=now + 0.5,
        execution_horizon=10,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=20.0,
    )


def _make_infer_request(seed: int = 0) -> InferRequest:
    """Mirrors what BidirectionalWebsocket.send constructs on the client side."""
    obs = _make_obs(seed)
    now = time.time()
    return InferRequest(
        robot_id=f"robot_{seed % 32}",
        observation=obs,  # type: ignore[arg-type]
        observation_step=seed,
        action_index_start=seed * 10,
        request_timestamp=now,
        deadline=now + 0.5,
        execution_horizon=10,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
    )


def _make_slot_request(seed: int = 0, slot_index: int = 0) -> SlotRequest:
    """No obs payload — that lives in shared memory; only metadata flows over zmq."""
    now = time.time()
    return SlotRequest(
        slot_index=slot_index,
        robot_id=f"robot_{seed % 32}",
        request_id=seed,
        arrival_timestamp=now,
        observation_step=seed,
        action_index_start=seed * 10,
        request_timestamp=now - 0.001,
        deadline=now + 0.5,
        execution_horizon=10,
        infer_type=InferType.SYNC,
        params=None,
        noise=None,
        control_hz=20.0,
    )


def _assert_round_trip(original: SlotData, restored: SlotData) -> None:
    """Same external contract: every field must round-trip; ndarrays byte-equal."""
    assert restored.robot_id == original.robot_id
    assert restored.request_id == original.request_id
    assert restored.observation_step == original.observation_step
    assert restored.action_index_start == original.action_index_start
    assert restored.request_timestamp == original.request_timestamp
    assert restored.arrival_timestamp == original.arrival_timestamp
    assert restored.deadline == original.deadline
    assert restored.execution_horizon == original.execution_horizon
    assert restored.infer_type == original.infer_type
    assert restored.params == original.params
    assert restored.noise == original.noise
    assert restored.control_hz == original.control_hz
    # obs round-trip: same keys, ndarrays byte-equal, scalars equal
    assert set(restored.obs.keys()) == set(original.obs.keys())
    for key, orig_val in original.obs.items():
        got = restored.obs[key]
        if isinstance(orig_val, np.ndarray):
            assert isinstance(got, np.ndarray), f"key={key} expected ndarray, got {type(got)}"
            assert got.dtype == orig_val.dtype, f"key={key} dtype {got.dtype} != {orig_val.dtype}"
            assert got.shape == orig_val.shape, f"key={key} shape {got.shape} != {orig_val.shape}"
            assert np.array_equal(got, orig_val), f"key={key} bytes differ"
        else:
            assert got == orig_val, f"key={key} {got!r} != {orig_val!r}"


# --------------------------------------------------------------------------- #
# Bench 1: correctness round-trip
# --------------------------------------------------------------------------- #


def bench_round_trip() -> None:
    print("== correctness: write -> read round-trip ==")
    slot = RobotSlot()
    for seed in range(5):
        data = _make_slot_data(seed)
        slot.write(data)
        restored = slot.read()
        _assert_round_trip(data, restored)
    print("  OK\n")


# --------------------------------------------------------------------------- #
# Bench 2: single-process write/read latency
# --------------------------------------------------------------------------- #


@dataclass
class LatencyStats:
    n: int
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: float
    mean_us: float

    @classmethod
    def from_seconds(cls, samples_s: list[float]) -> "LatencyStats":
        us = [s * 1e6 for s in samples_s]
        us.sort()
        return cls(
            n=len(us),
            p50_us=us[len(us) // 2],
            p95_us=us[int(len(us) * 0.95)],
            p99_us=us[int(len(us) * 0.99)],
            max_us=us[-1],
            mean_us=statistics.fmean(us),
        )

    def __str__(self) -> str:
        return (
            f"n={self.n:5d}  mean={self.mean_us:8.1f}us  p50={self.p50_us:8.1f}us  "
            f"p95={self.p95_us:8.1f}us  p99={self.p99_us:8.1f}us  max={self.max_us:8.1f}us"
        )


def bench_single_process(iters: int) -> None:
    print(f"== single-process latency (iters={iters}) ==")
    slot = RobotSlot()
    # Pre-build SlotData so allocation/randomness is out of the timed path.
    # Keep a small ring of distinct payloads so memcpy patterns vary.
    payloads = [_make_slot_data(i) for i in range(8)]

    # Warmup
    for i in range(50):
        slot.write(payloads[i % len(payloads)])
        slot.read()

    write_samples: list[float] = []
    for i in range(iters):
        data = payloads[i % len(payloads)]
        t0 = time.perf_counter()
        slot.write(data)
        write_samples.append(time.perf_counter() - t0)

    read_samples: list[float] = []
    # Re-write fresh so the slot is full of valid data for reads.
    slot.write(payloads[0])
    for _ in range(iters):
        t0 = time.perf_counter()
        slot.read()
        read_samples.append(time.perf_counter() - t0)

    print(f"  write  {LatencyStats.from_seconds(write_samples)}")
    print(f"  read   {LatencyStats.from_seconds(read_samples)}")
    print()


# --------------------------------------------------------------------------- #
# Bench 3: msgpack_numpy.packb / unpackb on a full InferRequest
# --------------------------------------------------------------------------- #


def bench_msgpack(iters: int) -> None:
    """What the WS event loop pays per inbound websocket frame.

    `unpackb` here is exactly server.py:397; `packb` is exactly what the
    client does in BidirectionalWebsocket.send (client.py:131-145).
    """
    print(f"== msgpack_numpy on full InferRequest (iters={iters}) ==")
    requests = [_make_infer_request(i) for i in range(8)]
    # Pre-pack so the unpack benchmark times only unpack.
    packed = [msgpack_numpy.packb(r) for r in requests]
    payload_kb = len(packed[0]) / 1024
    print(f"  wire size: {payload_kb:.1f} KB per InferRequest")

    # warmup
    for i in range(50):
        msgpack_numpy.packb(requests[i % len(requests)])
        msgpack_numpy.unpackb(packed[i % len(packed)])

    pack_samples: list[float] = []
    for i in range(iters):
        r = requests[i % len(requests)]
        t0 = time.perf_counter()
        msgpack_numpy.packb(r)
        pack_samples.append(time.perf_counter() - t0)

    unpack_samples: list[float] = []
    for i in range(iters):
        raw = packed[i % len(packed)]
        t0 = time.perf_counter()
        msgpack_numpy.unpackb(raw)
        unpack_samples.append(time.perf_counter() - t0)

    print(f"  packb    {LatencyStats.from_seconds(pack_samples)}")
    print(f"  unpackb  {LatencyStats.from_seconds(unpack_samples)}")
    print()


# --------------------------------------------------------------------------- #
# Bench 4: pickle.dumps(SlotRequest) + zmq PUB send_pyobj
# --------------------------------------------------------------------------- #


def _zmq_sub_drainer(endpoint: str, stop_evt, count_val) -> None:
    """SUB-side: drain messages as fast as possible. Mirrors the scheduler's
    req_sock.recv_pyobj loop (scheduler.py:96-98 + _process_server_messages)."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(endpoint)
    n = 0
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while not stop_evt.is_set():
        if dict(poller.poll(timeout=50)):
            try:
                while True:
                    sock.recv_pyobj(zmq.NOBLOCK)
                    n += 1
            except zmq.Again:
                pass
    count_val.value = n
    sock.close()
    ctx.term()


def bench_zmq_send(iters: int) -> None:
    """Per-call cost of `await scheduler_sock.send_pyobj(slot_req)` (server.py:477).

    SUB is in a forked child so the send actually has somewhere to go (otherwise
    PUB would happily drop and the timing would be misleading). Also breaks out
    pickle.dumps separately so we know how much of the wall time is Python vs
    zmq's own copy/queue.
    """
    print(f"== pickle + zmq PUB send_pyobj on SlotRequest (iters={iters}) ==")
    slot_req = _make_slot_request(seed=0)

    # Pure pickle cost (no IPC).
    pickled = pickle.dumps(slot_req)
    print(f"  pickled SlotRequest size: {len(pickled)} B")

    # warmup
    for _ in range(50):
        pickle.dumps(slot_req)

    pickle_samples: list[float] = []
    for i in range(iters):
        sr = _make_slot_request(seed=i)
        t0 = time.perf_counter()
        pickle.dumps(sr)
        pickle_samples.append(time.perf_counter() - t0)

    # zmq PUB→SUB send. Use ipc:// to match server.py's transport.
    endpoint = f"ipc:///tmp/bench_slots_{os.getpid()}"
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(endpoint)

    stop_evt = mp.Event()
    count_val = mp.Value("q", 0)
    sub_proc = mp.Process(
        target=_zmq_sub_drainer, args=(endpoint, stop_evt, count_val), daemon=True
    )
    sub_proc.start()
    # Let the SUB connect; PUB/SUB drops messages sent before the join.
    time.sleep(0.3)

    send_samples: list[float] = []
    for i in range(iters):
        sr = _make_slot_request(seed=i)
        t0 = time.perf_counter()
        pub.send_pyobj(sr)
        send_samples.append(time.perf_counter() - t0)

    # Drain a bit so the SUB sees most messages before we tear down.
    time.sleep(0.1)
    stop_evt.set()
    sub_proc.join(timeout=5)
    if sub_proc.is_alive():
        sub_proc.terminate()
        sub_proc.join()
    pub.close()
    ctx.term()

    print(f"  pickle.dumps    {LatencyStats.from_seconds(pickle_samples)}")
    print(f"  send_pyobj      {LatencyStats.from_seconds(send_samples)}")
    print(f"  SUB drained {count_val.value}/{iters} messages")
    print()


# --------------------------------------------------------------------------- #
# Bench 5: end-to-end recv() chain (no asyncio, just the sync work)
# --------------------------------------------------------------------------- #


def bench_recv_chain(iters: int) -> None:
    """Time the chain of synchronous work between `await receive_bytes()` and
    the next `await` in server.py:393-477. This is the wall time the WS event
    loop is actually unavailable to other coroutines per inbound request.

    Sequence:
        msgpack_numpy.unpackb(raw)
        InferRequest(**msg)
        slots.write(SlotData(obs=req.observation, ...))
        pickle.dumps(SlotRequest(...))    # zmq send is async, so just pickle here
    """
    print(f"== recv() sync chain per request (iters={iters}) ==")
    # Pre-pack realistic wire bytes.
    requests = [_make_infer_request(i) for i in range(8)]
    packed = [msgpack_numpy.packb(r) for r in requests]
    slot = RobotSlot()
    slot_index = 0

    # warmup
    for i in range(50):
        msg = msgpack_numpy.unpackb(packed[i % len(packed)])
        msg.pop("type", None)
        req = InferRequest(**msg)
        now = time.time()
        slot.write(
            SlotData(
                robot_id=req.robot_id, obs=req.observation, request_id=i,
                arrival_timestamp=now, observation_step=req.observation_step,
                action_index_start=req.action_index_start,
                request_timestamp=req.request_timestamp, deadline=req.deadline,
                execution_horizon=req.execution_horizon, infer_type=req.infer_type,
                params=req.params, noise=req.noise, control_hz=20.0,
            )
        )
        sr = SlotRequest(
            slot_index=slot_index, robot_id=req.robot_id, request_id=i,
            arrival_timestamp=now, observation_step=req.observation_step,
            action_index_start=req.action_index_start,
            request_timestamp=req.request_timestamp, deadline=req.deadline,
            execution_horizon=req.execution_horizon, infer_type=req.infer_type,
            params=req.params, noise=req.noise, control_hz=20.0,
        )
        pickle.dumps(sr)

    samples: list[float] = []
    breakdown: dict[str, list[float]] = {"unpackb": [], "slots.write": [], "pickle.dumps": []}
    for i in range(iters):
        raw = packed[i % len(packed)]

        t0 = time.perf_counter()
        msg = msgpack_numpy.unpackb(raw)
        msg.pop("type", None)
        req = InferRequest(**msg)
        t1 = time.perf_counter()

        now = time.time()
        slot.write(
            SlotData(
                robot_id=req.robot_id, obs=req.observation, request_id=i,
                arrival_timestamp=now, observation_step=req.observation_step,
                action_index_start=req.action_index_start,
                request_timestamp=req.request_timestamp, deadline=req.deadline,
                execution_horizon=req.execution_horizon, infer_type=req.infer_type,
                params=req.params, noise=req.noise, control_hz=20.0,
            )
        )
        t2 = time.perf_counter()

        sr = SlotRequest(
            slot_index=slot_index, robot_id=req.robot_id, request_id=i,
            arrival_timestamp=now, observation_step=req.observation_step,
            action_index_start=req.action_index_start,
            request_timestamp=req.request_timestamp, deadline=req.deadline,
            execution_horizon=req.execution_horizon, infer_type=req.infer_type,
            params=req.params, noise=req.noise, control_hz=20.0,
        )
        pickle.dumps(sr)
        t3 = time.perf_counter()

        samples.append(t3 - t0)
        breakdown["unpackb"].append(t1 - t0)
        breakdown["slots.write"].append(t2 - t1)
        breakdown["pickle.dumps"].append(t3 - t2)

    print(f"  total           {LatencyStats.from_seconds(samples)}")
    for name, ss in breakdown.items():
        print(f"  {name:<15} {LatencyStats.from_seconds(ss)}")
    # Throughput cap on a single event loop, ignoring scheduler/GPU/dashboard contention:
    mean_s = statistics.fmean(samples)
    print(f"  -> single-loop throughput ceiling: {1.0 / mean_s:.0f} req/s "
          f"({mean_s * 1e3:.2f} ms/req)")
    print()


# --------------------------------------------------------------------------- #
# Bench 6: cross-process throughput (writer parent, reader child)
# --------------------------------------------------------------------------- #


def _reader_proc(slots: RobotSlots, slot_index: int, stop_evt, count_val) -> None:
    """Spin reading from the slot until stop_evt is set; counts reads."""
    n = 0
    while not stop_evt.is_set():
        slots.read(slot_index)
        n += 1
        # No sleep — we want the upper bound on read throughput.
    count_val.value = n


def bench_cross_process(duration_s: float) -> None:
    print(f"== cross-process throughput (duration={duration_s:.1f}s) ==")
    # RobotSlots must be created BEFORE fork so both procs share the buffers.
    slots = RobotSlots(max_robots=4)
    slot_index = slots.register("robot_0")

    # Seed so the reader doesn't hit an empty slot.
    slots.write(slot_index, _make_slot_data(0))

    stop_evt = mp.Event()
    count_val = mp.Value("q", 0)
    reader = mp.Process(
        target=_reader_proc, args=(slots, slot_index, stop_evt, count_val), daemon=True
    )
    reader.start()

    payloads = [_make_slot_data(i) for i in range(8)]
    writes = 0
    write_samples: list[float] = []
    deadline = time.perf_counter() + duration_s
    i = 0
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        slots.write(slot_index, payloads[i % len(payloads)])
        write_samples.append(time.perf_counter() - t0)
        writes += 1
        i += 1

    stop_evt.set()
    reader.join(timeout=5)
    if reader.is_alive():
        reader.terminate()
        reader.join()

    elapsed = duration_s
    print(
        f"  writer  {writes:7d} writes/{elapsed:.1f}s = {writes / elapsed:7.0f}/s   "
        f"(per-write {LatencyStats.from_seconds(write_samples)})"
    )
    print(f"  reader  {count_val.value:7d} reads /{elapsed:.1f}s = {count_val.value / elapsed:7.0f}/s")
    print()


# --------------------------------------------------------------------------- #
# Bench 4: how long does write() stall an unrelated thread?
# --------------------------------------------------------------------------- #


def bench_event_loop_stall(iters: int) -> None:
    """Proxy for WS event-loop stall.

    A background thread tries to do work at 1 ms intervals; we measure how
    much its wakeups jitter while the main thread runs slot.write() in a hot
    loop. Pickle releases the GIL on numpy bulk but reacquires it for Python
    framing — anything > a memcpy shows up here.
    """
    print(f"== event-loop stall (iters={iters}) ==")
    slot = RobotSlot()
    payloads = [_make_slot_data(i) for i in range(8)]

    stop = threading.Event()
    jitter_us: list[float] = []

    def ticker() -> None:
        prev = time.perf_counter()
        while not stop.is_set():
            time.sleep(0.001)
            now = time.perf_counter()
            jitter_us.append((now - prev - 0.001) * 1e6)
            prev = now

    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    time.sleep(0.05)  # let the ticker settle

    for i in range(iters):
        slot.write(payloads[i % len(payloads)])

    stop.set()
    t.join(timeout=2)

    if jitter_us:
        # Negative jitter (sleep returned early) is meaningless here; clip to 0.
        positive = [max(0.0, j) for j in jitter_us]
        positive.sort()
        print(
            f"  ticker  n={len(positive):5d}  "
            f"p50={positive[len(positive)//2]:7.1f}us  "
            f"p95={positive[int(len(positive)*0.95)]:7.1f}us  "
            f"p99={positive[int(len(positive)*0.99)]:7.1f}us  "
            f"max={positive[-1]:7.1f}us"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--cross-proc-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--skip",
        choices=["round-trip", "single", "msgpack", "zmq", "recv-chain", "cross", "stall"],
        action="append",
        default=[],
    )
    args = parser.parse_args()

    print(f"pid={os.getpid()}  cpus={os.cpu_count()}  python perf_counter resolution OK")
    print(f"image bytes per obs: {np.prod(IMAGE_SHAPE) * 2 / 1024:.1f} KB "
          f"(image+wrist), state={STATE_DIM * 4} B\n")

    if "round-trip" not in args.skip:
        bench_round_trip()
    if "single" not in args.skip:
        bench_single_process(args.iters)
    if "msgpack" not in args.skip:
        bench_msgpack(args.iters)
    if "zmq" not in args.skip:
        bench_zmq_send(args.iters)
    if "recv-chain" not in args.skip:
        bench_recv_chain(args.iters)
    if "stall" not in args.skip:
        bench_event_loop_stall(args.iters)
    if "cross" not in args.skip:
        bench_cross_process(args.cross_proc_duration_s)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
