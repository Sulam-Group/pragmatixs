from typing import Mapping, Optional

import numpy as np

import wandb


class Monitor:
    def __init__(self):
        self.data: Mapping[str, float] = {}
        self.global_samples: int = 0

    def zero(self):
        self.data = {}

    def update(
        self,
        data: Mapping[str, float],
        num_samples: int,
        increase_global_samples: bool = True,
    ):
        if increase_global_samples:
            self.global_samples += num_samples

        for k, v in data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((v, num_samples))

    def log(self, prefix: str, step: Optional[int] = None):
        data = {k: np.array(v) for k, v in self.data.items()}
        data = {k: np.sum(v[:, 0]) / sum(v[:, 1]) for k, v in data.items()}
        wandb.log(
            {f"{prefix}/{k}": v for k, v in data.items()},
            step=step or self.global_samples,
        )
        self.data = {}
