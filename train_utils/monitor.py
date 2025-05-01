import os
from collections.abc import Mapping

import numpy as np
import torch
import torch.distributed as distributed

import wandb
from configs import Config
from configs import Constants as C
from listeners import Listener
from speaker import ClaimSpeaker
from train_utils.utils import rank_zero_only


class Monitor:
    def __init__(self, config: Config):
        self.config = config
        self.run_name = config.run_name()

        self.rank, self.world_size = 0, 1
        if distributed.is_initialized():
            self.rank = distributed.get_rank()
            self.world_size = distributed.get_world_size()

        self.data: Mapping[str, float] = {}
        self.global_samples: int = 0

        if self.rank == 0:
            self.init_wandb()

    def init_wandb(self):
        wandb.init(
            project="pragmatics", name=self.run_name, config=self.config.to_dict()
        )

    def zero(self):
        self.data = {}

    def update(
        self,
        data: Mapping[str, torch.Tensor],
        num_samples: int,
        increase_global_samples: bool = True,
    ):
        if increase_global_samples:
            self.global_samples += num_samples * self.world_size

        for k, v in data.items():
            v = v.detach().cpu()
            nan = torch.isnan(v)
            safe_v = v[~nan]
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((safe_v.sum(), num_samples - nan.sum()))

    def log(self, prefix: str, step: int | None = None):
        data = {k: np.array(v) for k, v in self.data.items()}
        data = {k: np.sum(v[:, 0]) / sum(v[:, 1]) for k, v in data.items()}
        if self.rank == 0:
            wandb.log(
                {f"{prefix}/{k}": v for k, v in data.items()},
                step=step or self.global_samples,
            )
        self.data = {}

    @rank_zero_only
    def save(
        self,
        speaker: ClaimSpeaker = None,
        listener: Listener = None,
        epoch: int = None,
        workdir=C.workdir,
    ):
        weights_dir = os.path.join(workdir, "weights", self.run_name)
        os.makedirs(weights_dir, exist_ok=True)

        dist = self.config.data.distributed
        state_dict = {
            "speaker": speaker.module.state_dict() if dist else speaker.state_dict(),
            "listener": listener.module.state_dict() if dist else listener.state_dict(),
        }
        if self.rank == 0:
            torch.save(state_dict, os.path.join(weights_dir, f"iteration_{epoch+1}.pt"))
            with open(os.path.join(weights_dir, "latest.txt"), "w") as f:
                f.write(f"iteration_{epoch+1}.pt")
