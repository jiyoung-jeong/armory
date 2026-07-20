from typing import Literal, Self  # Any used for shared globals

from pydantic import BaseModel, ConfigDict, Field, model_validator

from armory_client.action_chunkers import ActionChunkBrokerType
from evaluation.recording import JSONBaseModel


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
    observation_latency: NetworkLatency = NetworkLatency()
    action_latency: NetworkLatency = NetworkLatency()
    control_hz: int = Field(gt=0, default=20)
    weight: int = 1


class ExperimentConfig(JSONBaseModel):
    model_config = ConfigDict(frozen=True)

    env: Literal["libero", "mock"] = "mock"
    task_suite_name: str = "libero10"
    num_trials_per_task: int = Field(ge=1, default=1)
    max_steps: int = Field(gt=0, default=100)
    action_chunk_broker_type: ActionChunkBrokerType = ActionChunkBrokerType.NAIVE_ASYNC
    robots: list[Robot] = [Robot()]
    # New "trial" mode: when wall_clock_time_limit_s > 0, the seed picks
    # ``subset_size`` tasks from the suite (0 = all tasks), each robot is
    # pinned to one of those tasks, and runs episodes back-to-back until
    # its per-robot wall-clock budget is exhausted. ``max_steps`` still
    # caps each individual episode.
    subset_size: int = 0  # TODO: what is this, can we remove it?
    wall_clock_time_limit_s: float = Field(default=0.0, ge=0.0)
    seed: int = Field(default=7, ge=0)

    @property
    def use_trial_mode(self) -> bool:
        return self.wall_clock_time_limit_s > 0.0

    def execution_horizon_for_robot(self, robot_idx: int) -> ExecutionHorizon:
        return self.robots[robot_idx].execution_horizon

    def max_execution_horizons(self) -> list[int]:
        return [r.execution_horizon.max for r in self.robots]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.use_trial_mode and self.wall_clock_time_limit_s <= 0.0:
            raise ValueError("wall_clock_time_limit_s must be positive in trial mode")
        if not self.use_trial_mode and self.num_trials_per_task <= 0:
            raise ValueError("num_trials_per_task must be positive")
        return self
