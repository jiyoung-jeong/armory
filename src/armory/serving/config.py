import pathlib

from pydantic import BaseModel

from armory.serving.protocol import SchedulerConfig
from armory.serving.rtc import InferType


class EngineConfig(BaseModel):
    num_steps: int = 10
    infer_type: InferType = InferType.SYNC


class ServerConfig(BaseModel):
    max_batch_size: int = 1
    scheduler: SchedulerConfig = SchedulerConfig()
    engine: EngineConfig = EngineConfig()
    output_dir: pathlib.Path = pathlib.Path("output/serve")
