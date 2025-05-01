import logging
from functools import wraps

import torch
import torch.distributed as distributed
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)

from configs import Config
from configs import Constants as C


def get_rank():
    if not distributed.is_initialized():
        return 0
    return distributed.get_rank()


def is_rank_zero():
    return get_rank() == 0


def rank_zero_only(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if is_rank_zero():
            return f(*args, **kwargs)
        return None

    return wrapped


def setup_logging():
    class RankZeroFilter(logging.Filter):
        def filter(self, record):
            return is_rank_zero()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s", "%H:%M:%S"
        )
    )
    handler.addFilter(RankZeroFilter())

    logging.basicConfig(level=logging.INFO, handlers=[handler])


def get_loader_and_indices(
    config: Config = None,
    dataset: Dataset = None,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 0,
    epoch: int = 0,
):
    if config.data.distributed:
        sampler = DistributedSampler(
            dataset, drop_last=True, shuffle=shuffle, seed=seed
        )
        sampler.set_epoch(epoch)
    else:
        if shuffle:
            g = torch.Generator()
            g.manual_seed(seed + epoch)
            sampler = RandomSampler(dataset, generator=g)
        else:
            sampler = SequentialSampler(dataset)

    return (
        DataLoader(dataset, batch_size=batch_size, sampler=sampler),
        list(sampler),
    )


def truncate_to_shortest_shard(data, device=C.device):
    length = torch.tensor(len(data["image_idx"]), device=device)
    world_size, min_len = 1, length
    if distributed.is_initialized():
        world_size = distributed.get_world_size()
        lengths = [torch.zeros_like(length) for _ in range(world_size)]
        distributed.all_gather(lengths, length)
        min_len = torch.min(torch.stack(lengths)).item()
    data["length"] = min_len
    return data


def initialize_optimizer(model: nn.Module, lr: float, weight_decay: float):
    def exclude(n, p):
        return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n

    def include(n, p):
        return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    if distributed.is_initialized():
        lr = lr * distributed.get_world_size()
    return torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=lr,
    )
