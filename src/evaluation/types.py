import json
import pathlib
from dataclasses import MISSING, dataclass, fields
from enum import Enum
from typing import Any, Self, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.envs.config import EnvironmentConfig, MockConfig

J = TypeVar("J", bound="JSONBaseModel")
P = TypeVar("P", bound="ParquetDataclass")


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
                    if f.default is not MISSING:
                        kwargs[f.name] = f.default
                    elif f.default_factory is not MISSING:
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
class StepRecord(ParquetDataclass):
    """One control step of an episode, the row type of ``steps.parquet``."""

    timestamp: float
    env_step: int
    action_chunk_index: int | None
    action_index: int | None
    # None for agents that keep no action queue, e.g. the mock agent.
    actions_left: int | None = None


class EnvironmentType(str, Enum):
    LIBERO = "libero"
    MOCK = "mock"


class ExecutionHorizon(BaseModel):
    min: int = Field(ge=1, default=1)
    max: int = 10

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.min > self.max:
            raise ValueError("min_execution_horizon must be <= max_execution_horizon")
        return self


class NetworkLatency(BaseModel):
    median: float = Field(ge=0.0, default=0.0)
    sigma: float = Field(ge=0.0, default=0.0)


class Robot(BaseModel):
    execution_horizon: ExecutionHorizon = ExecutionHorizon()
    control_hz: int = Field(gt=0, default=20)

    observation_latency: NetworkLatency = NetworkLatency()
    action_latency: NetworkLatency = NetworkLatency()

    weight: float = 1.0


class ExperimentConfig(JSONBaseModel):
    model_config = ConfigDict(frozen=True)

    environment: EnvironmentConfig = MockConfig()

    robots: list[Robot] = [Robot()]
    time_limit: float = Field(default=10.0, ge=0.0)
    seed: int = Field(default=7, ge=0)
