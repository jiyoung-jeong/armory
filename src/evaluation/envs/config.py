"""Configuration models for evaluation environment backends.

These models deliberately contain no simulator imports.  The runner can parse
an experiment on a lightweight machine, while each backend owns planning and
construction of its simulator-specific environments.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MockConfig(BaseModel):
    kind: Literal["mock"] = "mock"
    max_steps_per_episode: int = Field(gt=0, default=100)


class LiberoConfig(BaseModel):
    kind: Literal["libero"] = "libero"
    task_suite_name: str = "libero_10"
    max_steps_per_episode: int = Field(gt=0, default=300)
    # Zero preserves the default behavior of sampling a distinct task for each
    # robot. Positive values reproduce the paper experiments, which sampled a
    # task subset once and assigned robots to it round-robin.
    task_subset_size: int = Field(default=0, ge=0)
    # None means use ExperimentConfig.seed. Keeping this independently
    # configurable lets task assignments remain fixed across other sweeps.
    task_seed: int | None = Field(default=None, ge=0)


EnvironmentConfig = Annotated[MockConfig | LiberoConfig, Field(discriminator="kind")]
