import csv
import json
import pathlib
import time
from dataclasses import dataclass, field, fields, asdict
from typing import List, Type, TypeVar, Optional

import numpy as np
from jaxtyping import Float
from openpi_client import messages
import pandas as pd
import requests

T = TypeVar("T", bound="CSVDataclass")
J = TypeVar("J", bound="JSONDataclass")
P = TypeVar("P", bound="ParquetDataclass")


class CSVDataclass:
    """Mixin class that adds CSV serialization to dataclasses."""

    @classmethod
    def to_csv(cls: Type[T], instances: List[T], filepath: pathlib.Path) -> None:
        """Save a list of dataclass instances to a CSV file."""
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
        """Load a list of dataclass instances from a CSV file."""
        instances = []
        allowed_fields = [f for f in fields(cls) if f.type in (int, float, bool, str)]
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types based on field annotations
                kwargs = {}
                for field in allowed_fields:
                    value = row[field.name]
                    # Handle type conversion
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


class JSONDataclass:
    """Mixin class that adds JSON serialization to dataclasses."""

    def to_json(self, filepath: pathlib.Path, indent: int = 4) -> None:
        """Save a dataclass instance to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=indent)

    @classmethod
    def from_json(cls: Type[J], filepath: pathlib.Path) -> J:
        """Load a dataclass instance from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
            return cls(**data)


class ParquetDataclass:
    """Mixin class that adds Parquet serialization to dataclasses."""

    @classmethod
    def to_parquet(cls: Type[P], instances: List[P], filepath: pathlib.Path) -> None:
        """Save a list of dataclass instances to a Parquet file."""
        if not instances:
            return

        # Filter out dict fields as they don't serialize well to Parquet
        parquet_fields = [f for f in fields(cls) if f.type is not dict]

        # Convert instances to dictionary format
        data_dict = {field.name: [] for field in parquet_fields}

        for instance in instances:
            for f in parquet_fields:
                value = getattr(instance, f.name)
                data_dict[f.name].append(value)

        # Convert lists of numpy arrays to a format Parquet can handle
        for f in parquet_fields:
            values = data_dict[f.name]
            if values and isinstance(values[0], np.ndarray):
                # Convert to list of lists for nested array storage
                data_dict[f.name] = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]

        # Create DataFrame and write to Parquet
        df = pd.DataFrame(data_dict)
        df.to_parquet(filepath, engine="pyarrow", index=False)

    @classmethod
    def from_parquet(cls: Type[P], filepath: pathlib.Path) -> List[P]:
        """Load a list of dataclass instances from a Parquet file."""
        df = pd.read_parquet(filepath, engine="pyarrow")

        instances = []
        for _, row in df.iterrows():
            kwargs = {}
            for f in fields(cls):
                # Use default value if field is not in the DataFrame (e.g., dict fields)
                if f.name not in row:
                    if f.default is not None:
                        kwargs[f.name] = f.default
                    elif f.default_factory is not None:
                        kwargs[f.name] = f.default_factory()
                    continue

                value = row[f.name]

                # Convert back to numpy arrays if needed
                # This is a heuristic - you might want to add field annotations
                # to specify which fields should be numpy arrays
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
    """
    We store all actions, including past execution horizon, as they might come in handy for debugging later
    """

    observation_step: int
    action_start_step: int
    execution_start_step: int  # which observation step the execution started on
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
        # NOTE: copy attributes instead of composition to make it easier to serialize for parquet/csv
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


# TODO: rename this
@dataclass(frozen=True)
class Action:
    """
    The action_chunk_index and index_in_chunk will be None for the null action.
    """

    step: int
    action: Float[np.ndarray, " action_dim"]  # TODO: check the shape on this
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


@dataclass
class ServerMetadata(JSONDataclass):
    """Metadata about the policy server and model configuration.

    Sent from server to clients at connection time.
    """

    # Training config info
    config_name: str  # e.g., "pi0_aloha_sim", "pi05_libero"
    checkpoint_dir: str

    # Model configuration
    action_horizon: int
    action_dim: int
    num_steps: int  # sampling steps

    # Server configuration
    max_batch_size: int
    env: str  # environment mode (ALOHA, LIBERO, etc.)
    scheduling_algorithm: str  # TODO: maybe reference the enum from scheduler.py

    # Set by Modal when running behind a tunnel; clients should use this for WebSocket
    tunnel_url: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            info = requests.get("https://ipinfo.io/json", timeout=3).json()
            self.location = f"{info.get('city', '?')}, {info.get('region', '?')}, {info.get('country', '?')}"
        except Exception:
            self.location = "unknown"


@dataclass(frozen=True)
class RuntimeMetadata(JSONDataclass):
    """Metadata about the runtime/experiment configuration."""

    # Environment config
    task_suite_name: str
    num_trials_per_task: int
    max_steps: int
    seed: int
    resize_size: int

    # Multi-robot config
    num_robots: int
    control_hz: int

    # Action chunking config
    broker_type: str

    # Other
    episodes: List[str] = field(default_factory=list)
    execution_horizon: List[int] = field(default_factory=list)
