"""Client-side data schemas.

ServerMetadata and JSONDataclass are re-exported from armory.schemas (the canonical source).
Client-specific types (ActionChunk, Action, Observation, RuntimeMetadata) are defined here.
"""

import csv
import pathlib
import time
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from dataclasses import asdict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar

import numpy as np
import pandas as pd
from jaxtyping import Float

from armory_client import messages

# Re-export from canonical armory location so callers can do:
#   from armory_client.schemas import ServerMetadata
from armory.schemas import JSONDataclass  # noqa: F401
from armory.schemas import ServerMetadata  # noqa: F401

T = TypeVar("T", bound="CSVDataclass")
P = TypeVar("P", bound="ParquetDataclass")


class CSVDataclass:
    """Mixin class that adds CSV serialization to dataclasses."""

    @classmethod
    def to_csv(cls: Type[T], instances: List[T], filepath: pathlib.Path) -> None:
        if not instances:
            return
        with open(filepath, "w", newline="") as f:
            allowed_fields = [f for f in fields(cls) if f.type in (int, float, bool, str)]
            fieldnames = [f.name for f in allowed_fields]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for instance in instances:
                writer.writerow({field.name: getattr(instance, field.name) for field in allowed_fields})

    @classmethod
    def from_csv(cls: Type[T], filepath: pathlib.Path) -> List[T]:
        instances = []
        allowed_fields = [f for f in fields(cls) if f.type in (int, float, bool, str)]
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kwargs = {}
                for field in allowed_fields:
                    value = row[field.name]
                    if field.type in (int, "int"):
                        kwargs[field.name] = int(value)
                    elif field.type in (float, "float"):
                        kwargs[field.name] = float(value)
                    elif field.type in (bool, "bool"):
                        kwargs[field.name] = value.lower() in ("true", "1", "yes")
                    else:
                        kwargs[field.name] = value
                instances.append(cls(**kwargs))
        return instances


class ParquetDataclass:
    """Mixin class that adds Parquet serialization to dataclasses."""

    @classmethod
    def to_parquet(cls: Type[P], instances: List[P], filepath: pathlib.Path) -> None:
        if not instances:
            return
        parquet_fields = [f for f in fields(cls) if f.type is not dict]
        data_dict = {field.name: [] for field in parquet_fields}
        for instance in instances:
            for f in parquet_fields:
                data_dict[f.name].append(getattr(instance, f.name))
        for f in parquet_fields:
            values = data_dict[f.name]
            if values and isinstance(values[0], np.ndarray):
                data_dict[f.name] = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
        df = pd.DataFrame(data_dict)
        df.to_parquet(filepath, engine="pyarrow", index=False)

    @classmethod
    def from_parquet(cls: Type[P], filepath: pathlib.Path) -> List[P]:
        df = pd.read_parquet(filepath, engine="pyarrow")
        instances = []
        for _, row in df.iterrows():
            kwargs = {}
            for f in fields(cls):
                if f.name not in row:
                    if f.default is not None:
                        kwargs[f.name] = f.default
                    elif f.default_factory is not None:
                        kwargs[f.name] = f.default_factory()
                    continue
                value = row[f.name]
                if isinstance(value, list) and value and isinstance(value[0], list):
                    kwargs[f.name] = np.array(value)
                elif value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
                    kwargs[f.name] = None
                else:
                    kwargs[f.name] = value
            instances.append(cls(**kwargs))
        return instances


@dataclass(frozen=True)
class ActionChunk(ParquetDataclass):
    """All actions from one inference chunk, including those past the execution horizon."""

    observation_step: int
    action_start_step: int
    execution_start_step: int
    actions: np.ndarray
    execution_horizon: int
    request_timestamp: float
    response_timestamp: float
    request_id: int = -1
    noise: Optional[np.ndarray] = None

    @classmethod
    def from_infer_response(
        cls,
        infer_response: messages.InferResponse,
        execution_start_step: int,
    ) -> "ActionChunk":
        return ActionChunk(
            observation_step=infer_response.observation_step,
            action_start_step=infer_response.action_start_step,
            execution_start_step=execution_start_step,
            actions=infer_response.actions,
            execution_horizon=infer_response.execution_horizon,
            request_timestamp=infer_response.request_timestamp,
            response_timestamp=time.time(),
            request_id=infer_response.request_id,
            noise=infer_response.noise,
        )

    @property
    def latency(self) -> float:
        return self.response_timestamp - self.request_timestamp

    def get_action(self, index: int) -> Float[np.ndarray, " action_dim"]:
        return self.actions[index]


@dataclass(frozen=True)
class Action:
    """Single action with chunk provenance. action_chunk_index/index_in_chunk are None for null actions."""

    step: int
    action: Float[np.ndarray, " action_dim"]
    action_chunk_index: Optional[int]
    index_in_chunk: Optional[int]


@dataclass(frozen=True)
class Timestamp(CSVDataclass):
    timestamp: float
    env_step: int
    action_chunk_index: Optional[int]
    action_index: Optional[int]


@dataclass
class Observation:
    state: Float[np.ndarray, " state_dim"]
    step: int
    image: Float[np.ndarray, " h w c"]
    wrist_image: Float[np.ndarray, " h w c"]


@dataclass
class LiberoObservation(Observation):
    prompt: str


@dataclass(frozen=True)
class RuntimeMetadata(JSONDataclass):
    """Metadata about the runtime/experiment configuration."""

    task_suite_name: str
    num_trials_per_task: int
    max_steps: int
    seed: int
    resize_size: int
    num_robots: int
    control_hz: int
    broker_type: str
    episodes: List[str] = field(default_factory=list)
    execution_horizon: List[int] = field(default_factory=list)
