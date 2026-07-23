"""Dormant server-side RTC protocol types.

Clients no longer select an inference mode. These stay server-local so the
engine and policy RTC implementation can be retained without a wire feature.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class InferType(Enum):
    SYNC = "sync"
    INFERENCE_TIME_RTC = "inference_time_rtc"
    TRAIN_TIME_RTC = "train_time_rtc"
    VLASH = "vlash"


@dataclass
class RTCParams:
    prev_action: np.ndarray
    s_param: int
    d_param: int
