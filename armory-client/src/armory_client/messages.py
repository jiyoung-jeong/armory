from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class InferRequest:
    robot_id: str
    observation: dict
    observation_step: int
    action_index_start: int
    request_timestamp: float
    deadline: float
    min_execution_horizon: int
    max_execution_horizon: int
    noise: np.ndarray | None = None  # action_horizon noise_dim
    episode_id: str = ""
    type: Literal["infer"] = "infer"


@dataclass(frozen=True)
class ResetRequest:
    robot_id: str
    episode_id: str = ""
    type: Literal["reset"] = "reset"


@dataclass(frozen=True)
class InferResponse:
    robot_id: str
    request_id: int  # for routing response to correct connection
    chunk_id: int  # for matching with ActionChunk.chunk_id
    observation_step: int  # from request
    action_index_start: int  # from request
    request_timestamp: float  # from request
    actions: np.ndarray  # 1 action_horizon action_dim
    min_execution_horizon: int
    max_execution_horizon: int
    noise: np.ndarray | None = None  # action_horizon noise_dim
    episode_id: str = ""


@dataclass(frozen=True)
class ResponseAck:
    request_id: int  # matches InferResponse.request_id

    chunk_id: int  # matches InferResponse.chunk_id
    observation_step: int  # step when observation was captured
    receive_time: float  # time.time() on client at receipt
    action_index_start: int  # action index of the first action in the chunk
    min_execution_horizon: int
    max_execution_horizon: int
    execution_start_step: int  # client step when new chunk became available
    first_executed_index: int = 0  # index within chunk where actual execution started

    episode_id: str = ""
    type: Literal["ack"] = "ack"


@dataclass(frozen=True)
class ConnectRequest:
    robot_id: str
    control_hz: float
    weight: float = 1.0
    type: Literal["connect"] = "connect"


@dataclass(frozen=True)
class ConnectResponse:
    type: Literal["connect_response"] = "connect_response"


@dataclass(frozen=True)
class WarmupPing:
    client_timestamp: float
    payload: bytes  # dummy bytes, same size as a typical packed InferRequest
    type: Literal["warmup_ping"] = "warmup_ping"


@dataclass(frozen=True)
class WarmupPong:
    client_timestamp: float  # echoed from WarmupPing
    server_receive_time: float
    server_send_time: float
    payload: bytes  # dummy bytes, same size as a typical packed InferResponse
    type: Literal["warmup_pong"] = "warmup_pong"


@dataclass(frozen=True)
class WarmupAck:
    server_send_time: float  # echoed from WarmupPong, so server can compute delivery latency
    client_receive_time: float
    type: Literal["warmup_ack"] = "warmup_ack"
