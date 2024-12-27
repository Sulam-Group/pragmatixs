import os
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd
import torch
from ml_collections import ConfigDict


@dataclass
class Constants:
    WORKDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataConfig(ConfigDict):
    def __init__(self, **kwargs):
        super().__init__()

        self.dataset: str = kwargs.get("dataset", None)
        self.classifier: str = kwargs.get("classifier", None)
        self.context_length: int = kwargs.get("context_length", None)
        self.listener_type: str = kwargs.get("listener_type", None)


class SpeakerConfig(ConfigDict):
    def __init__(self, **kwargs):
        super().__init__()

        self.beta: float = kwargs.get("beta", None)
        self.alpha: float = kwargs.get("alpha", None)
        self.k: int = kwargs.get("k", None)


class ListenerConfig(ConfigDict):
    def __init__(self, **kwargs):
        super().__init__()

        self.gamma: float = kwargs.get("gamma", 0.0)


class TopicListenerConfig(ListenerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ignore_topics: Iterable[int] = kwargs.get("ignore_topics", None)


class DistributionListenerConfig(ListenerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prior: Iterable[float] = kwargs.get("prior", None)
        self.temperature_scale: float = kwargs.get("temperature_scale", None)


class Config(ConfigDict):
    def __init__(self, **kwargs):
        super().__init__()

        self.name = kwargs.get("name", None)
        self.data = DataConfig(**kwargs.get("data", {}))
        self.speaker = SpeakerConfig(**kwargs.get("speaker", {}))

        if self.data.listener_type == "claim":
            self.listener = ListenerConfig(**kwargs.get("listener", {}))
        elif self.data.listener_type == "topic":
            self.listener = TopicListenerConfig(**kwargs.get("listener", {}))
        elif self.data.listener_type == "distribution":
            self.listener = DistributionListenerConfig(**kwargs.get("listener", {}))

    def run_name(self):
        dataset_name = self.data.dataset.lower()
        listener_type = self.data.listener_type
        context_length = self.data.context_length
        beta = self.speaker.beta
        gamma = self.listener.gamma
        alpha = self.speaker.alpha

        run_name = (
            f"{dataset_name}_{listener_type}_{context_length}_{beta}_{gamma}_{alpha}"
        )
        if listener_type == "topic":
            run_name += f"_{''.join(map(str, self.listener.ignore_topics))}"
        if listener_type == "distribution":
            run_name += f"_{self.listener.temperature_scale}"
        return run_name

    def get_results(self, workdir=Constants.WORKDIR):
        results_dir = os.path.join(workdir, "results")
        return pd.read_parquet(os.path.join(results_dir, f"{self.run_name()}.parquet"))


configs: Mapping[str, Config] = {}


def register_config(name: str):
    def register(cls: Config):
        if name in configs:
            raise ValueError(f"Config {name} is already registered")
        configs[name] = cls

    return register


def get_config(name: str, **kwargs) -> Config:
    Config = configs[name]
    return Config(**kwargs)
