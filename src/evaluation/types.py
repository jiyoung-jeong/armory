from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from armory_client.action_chunkers import ActionChunkBrokerType
from evaluation.recording import JSONBaseModel


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
    action_chunk_broker_type: ActionChunkBrokerType = ActionChunkBrokerType.NAIVE_ASYNC
    execution_horizon: ExecutionHorizon = ExecutionHorizon()
    control_hz: int = Field(gt=0, default=20)

    observation_latency: NetworkLatency = NetworkLatency()
    action_latency: NetworkLatency = NetworkLatency()

    weight: int = 1


class ExperimentConfig(JSONBaseModel):
    model_config = ConfigDict(frozen=True)

    env: EnvironmentType = EnvironmentType.MOCK
    max_steps_per_episode: int = Field(gt=0, default=100)

    robots: list[Robot] = [Robot()]
    time_limit: float = Field(default=0.0, ge=0.0)
    seed: int = Field(default=7, ge=0)
