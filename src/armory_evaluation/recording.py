import csv
import json
import pathlib
from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

T = TypeVar("T", bound="CSVDataclass")
J = TypeVar("J", bound="JSONDataclass")
P = TypeVar("P", bound="ParquetDataclass")


class CSVDataclass:
    """Mixin that adds CSV serialization to dataclasses."""

    @classmethod
    def to_csv(cls: type[T], instances: list[T], filepath: pathlib.Path) -> None:
        if not instances:
            return
        with open(filepath, "w", newline="") as f:
            allowed_fields = [f for f in fields(cls) if f.type in (int, float, bool, str)]
            fieldnames = [f.name for f in allowed_fields]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for instance in instances:
                writer.writerow(
                    {field.name: getattr(instance, field.name) for field in allowed_fields}
                )

    @classmethod
    def from_csv(cls: type[T], filepath: pathlib.Path) -> list[T]:
        instances = []
        allowed_fields = [f for f in fields(cls) if f.type in (int, float, bool, str)]
        with open(filepath) as f:
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


class JSONDataclass:
    """Mixin that adds JSON serialization to dataclasses."""

    def to_json(self, filepath: pathlib.Path, indent: int = 4) -> None:
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=indent)

    @classmethod
    def from_json(cls: type[J], filepath: pathlib.Path) -> J:
        with open(filepath) as f:
            return cls(**json.load(f))


class JSONBaseModel(BaseModel):
    """Mixin that adds JSON file serialization to pydantic BaseModel subclasses."""

    def to_json(self, filepath: pathlib.Path, indent: int = 4) -> None:
        with open(filepath, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=indent)

    @classmethod
    def from_json(cls: type[J], filepath: pathlib.Path) -> J:
        with open(filepath) as f:
            return cls.model_validate_json(f.read())


class ParquetDataclass:
    """Mixin that adds Parquet serialization to dataclasses."""

    @classmethod
    def to_parquet(cls: type[P], instances: list[P], filepath: pathlib.Path) -> None:
        if not instances:
            return
        parquet_fields = [f for f in fields(cls) if f.type is not dict]
        data_dict: dict[str, list[Any]] = {field.name: [] for field in parquet_fields}
        for instance in instances:
            for f in parquet_fields:
                data_dict[f.name].append(getattr(instance, f.name))
        for f in parquet_fields:
            values = data_dict[f.name]
            if values and isinstance(values[0], np.ndarray):
                data_dict[f.name] = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
        pd.DataFrame(data_dict).to_parquet(filepath, engine="pyarrow", index=False)

    @classmethod
    def from_parquet(cls: type[P], filepath: pathlib.Path) -> list[P]:
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
                elif (
                    value is None
                    or value is pd.NA
                    or (isinstance(value, float) and np.isnan(value))
                ):
                    kwargs[f.name] = None
                else:
                    kwargs[f.name] = value
            instances.append(cls(**kwargs))
        return instances


@dataclass(frozen=True)
class Timestamp(CSVDataclass):
    timestamp: float
    env_step: int
    action_chunk_index: int | None
    action_index: int | None


@dataclass
class LiberoObservation:
    state: np.ndarray
    step: int
    image: np.ndarray
    wrist_image: np.ndarray
    prompt: str


@dataclass
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
    episodes: list[str] = None
    max_execution_horizon: list[int] = None

    def __post_init__(self) -> None:
        if self.episodes is None:
            self.episodes = []
        if self.max_execution_horizon is None:
            self.max_execution_horizon = []
