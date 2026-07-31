import pathlib

from pydantic import BaseModel, ConfigDict

from armory.serving.protocol import SchedulerConfig


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_steps: int = 10


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_batch_size: int = 1
    scheduler: SchedulerConfig = SchedulerConfig()
    engine: EngineConfig = EngineConfig()
    output_dir: pathlib.Path = pathlib.Path("output/serve")
