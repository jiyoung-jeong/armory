import random

import numpy as np


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
