
from pathlib import Path
import numpy as np

episode = Path("/coc/flash7/rbansal66/vvla/0_real_0_failure")
control_hz = 20  # change if this run used a different hz

costs = np.load(episode / "cost_history.npy")
starved = np.isnan(costs)

first_non_starved = np.flatnonzero(~starved)
if first_non_starved.size:
    first = int(first_non_starved[0])
    post_first_starved = int(starved[first:].sum())
    post_first_observed = int(len(costs) - first)
else:
    post_first_starved = 0
    post_first_observed = 0

print("observed_steps:", len(costs))
print("starvation_steps:", int(starved.sum()))
print("starvation_rate:", float(starved.mean()))
print("starvation_seconds:", float(starved.sum() / control_hz))
print("post_first_starvation_steps:", post_first_starved)
print("post_first_observed_steps:", post_first_observed)
print("post_first_starvation_rate:", post_first_starved / post_first_observed if post_first_observed else 0.0)
