from dataclasses import dataclass
from enum import Enum

import numpy as np


class InferType(Enum):
    SYNC = "sync"
    INFERENCE_TIME_RTC = "inference_time_rtc"
    TRAIN_TIME_RTC = "train_time_rtc"


@dataclass
class RTCParams:
    prev_action: np.ndarray
    s_param: int
    d_param: int
