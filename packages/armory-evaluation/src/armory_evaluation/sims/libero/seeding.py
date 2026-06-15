import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
