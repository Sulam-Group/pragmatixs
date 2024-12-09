import os
from typing import Iterable, Optional

import pandas as pd
import torch


class Constants:
    WORKDIR = os.path.dirname(os.path.realpath(__file__))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    dataset_name: str

    listener_type: str
    context_length: int
    beta: float
    gamma: float
    alpha: float
    k: int = 8

    def __init__(
        self,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
    ):
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha

    def run_name(self):
        return f"{self.listener_type}_{self.context_length}_{self.beta}_{self.gamma}_{self.alpha}"

    def get_results(self, workdir="./"):
        results_dir = os.path.join(workdir, "results")
        return pd.read_parquet(os.path.join(results_dir, f"{self.run_name()}.parquet"))


configs = {}


def register_config(name: str):
    def register(cls: Config):
        if name in configs:
            raise ValueError(f"Config {name} is already registered")
        configs[name] = cls

    return register


def get_config(name: str, **kwargs) -> Config:
    Config = configs[name]
    return Config(**kwargs)


@register_config(name="cub")
class CUBConfig(Config):
    dataset_name = "CUB"


@register_config(name="ham")
class HAMConfig(Config):
    dataset_name = "HAM"


class ClaimConfig(Config):
    listener_type = "claim"


class TopicConfig(Config):
    listener_type = "topic"

    ignore_topics: Iterable[int]

    def __init__(
        self,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
        ignore_topics: Optional[Iterable[int]] = None,
    ):
        super().__init__(beta=beta, gamma=gamma, alpha=alpha)

        if gamma is not None:
            self.gamma = gamma
        if ignore_topics is not None:
            self.ignore_topics = ignore_topics

    def run_name(self):
        return f"{super().run_name()}_{''.join(map(str, self.ignore_topics))}"


class DistributionConfig(Config):
    listener_type = "distribution"

    prior: Iterable[float]
    temperature_scale: float

    def __init__(
        self,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
        prior: Optional[Iterable[float]] = None,
        temperature_scale: Optional[float] = None,
    ):
        super().__init__(beta=beta, gamma=gamma, alpha=alpha)

        if prior is not None:
            self.prior = prior
        if temperature_scale is not None:
            self.temperature_scale = temperature_scale

    def run_name(self):
        return f"{super().run_name()}_{self.temperature_scale}"


class CUBClaimConfig(ClaimConfig):
    context_length = 12
    beta = 0.4

    def run_name(self):
        return f"cub_{super().run_name()}"


class CUBTopicConfig(TopicConfig):
    context_length = 12
    beta = 0.4

    ignore_topics = [4]

    def run_name(self):
        return f"cub_{super().run_name()}"


class CUBDistributionConfig(DistributionConfig):
    context_length = 12
    beta = 0.4
    gamma = 0.0

    prior = 6 * [1 / 6]
    temperature_scale = 1.0

    def run_name(self):
        return f"cub_{super().run_name()}"


# class CUBPragmaticClaimConfig(CUBClaimConfig):
#     alpha = 0.1


# class CUBPragmaticTopicConfig(CUBTopicConfig):
#     gamma = 0.01
#     alpha = 0.1


# class CUBPragmaticDistributionConfig(CUBDistributionConfig):
#     alpha = 0.1
