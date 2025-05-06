import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd
import torch
from ml_collections import ConfigDict


@dataclass
class Constants:
    workdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get(dict, key):
    subkeys = key.split(".")
    if len(subkeys) == 1:
        return dict[subkeys[0]]
    else:
        return _get(dict[subkeys[0]], ".".join(subkeys[1:]))


def _set(dict, key, value):
    subkeys = key.split(".")
    if len(subkeys) == 1:
        dict[subkeys[0]] = value
    else:
        _set(dict[subkeys[0]], ".".join(subkeys[1:]), value)


class DataConfig(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] | None = {}):
        super().__init__()

        self.dataset: str = config_dict.get("dataset", None)
        self.classifier: str = config_dict.get("classifier", None)
        self.explanation_length: int = config_dict.get("explanation_length", None)
        self.task: str = config_dict.get("task", None)
        self.distributed: bool = config_dict.get("distributed", False)


class SpeakerConfig(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] | None = {}):
        super().__init__()

        self.width: int = config_dict.get("width", None)
        self.heads: int = config_dict.get("heads", None)
        self.layers: int = config_dict.get("layers", None)
        self.attn_pooler_heads: int = config_dict.get("attn_pooler_heads", None)

        self.use_tokens: bool = config_dict.get("use_tokens", False)
        self.include_prediction: bool = config_dict.get("include_prediction", True)
        self.dropout: float = config_dict.get("dropout", 0.0)
        self.n_queries: int = config_dict.get("n_queries", None)

        self.beta: float = config_dict.get("beta", None)
        self.alpha: float = config_dict.get("alpha", None)
        self.k: int = config_dict.get("k", None)


class ListenerConfig(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] | None = {}):
        super().__init__()

        self.type: str = config_dict.get("type", None)

        self.width: int = config_dict.get("width", None)
        self.heads: int = config_dict.get("heads", None)
        self.layers: int = config_dict.get("layers", None)

        self.gamma: float = config_dict.get("gamma", None)
        self.k: int = config_dict.get("k", None)

        # distributional listener config
        self.prior: Mapping[str, float] = config_dict.get("prior", None)
        self.temperature_scale: float = config_dict.get("temperature_scale", None)


class TopicListenerConfig(ListenerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ignore_topics: Iterable[int] = kwargs.get("ignore_topics", None)


class DistributionListenerConfig(ListenerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prior: Iterable[float] = kwargs.get("prior", None)
        self.temperature_scale: float = kwargs.get("temperature_scale", None)


class TrainingConfig(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] | None = {}):
        super().__init__()

        self.iterations: int = config_dict.get("iterations", None)
        self.batch_size: int = config_dict.get("batch_size", None)
        self.min_lr: float = config_dict.get("min_lr", None)
        self.max_lr: float = config_dict.get("max_lr", None)
        self.wd: float = config_dict.get("wd", None)
        self.max_grad_norm: float = config_dict.get("max_grad_norm", None)


class Config(ConfigDict):
    def __init__(self, config_dict: Mapping[str, Any] | None = {}):
        super().__init__()

        self.data = DataConfig(config_dict.get("data", {}))
        self.speaker = SpeakerConfig(config_dict.get("speaker", {}))
        self.listener = ListenerConfig(config_dict.get("listener", {}))
        self.training = TrainingConfig(config_dict.get("training", {}))

    def classifier_name(self):
        return (
            self.data.classifier.lower()
            .replace(":", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

    def run_name(self):
        dataset_name = self.data.dataset.lower()
        listener_type = self.listener.type
        gamma = self.listener.gamma
        explanation_length = self.data.explanation_length
        alpha = self.speaker.alpha

        run_name = (
            f"{dataset_name}"
            f"_{listener_type}"
            f"_len{explanation_length}"
            f"_gamma{gamma}"
            f"_alpha{alpha}"
        )
        if listener_type in ["topic", "region"]:
            p = [f"{p:.2f}" for p in self.listener.prior]
            run_name += f"_p{','.join(p)}_t{self.listener.temperature_scale}"
        return run_name

    def _path(self, out_dir, workdir=Constants.workdir):
        out_dir = os.path.join(workdir, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{self.run_name()}")

    def train_cache_dir(self, workdir=Constants.workdir):
        train_cache_dir = os.path.join(workdir, "data", "train_cache", self.run_name())
        os.makedirs(train_cache_dir, exist_ok=True)
        return train_cache_dir

    def state_path(self, workdir=Constants.workdir):
        weight_dir = os.path.join(workdir, "weights", self.run_name())
        os.makedirs(weight_dir, exist_ok=True)
        with open(os.path.join(weight_dir, "latest.txt")) as f:
            latest = f.read().strip()
        return os.path.join(weight_dir, latest)

    def results_path(self, workdir=Constants.workdir):
        results_dir = os.path.join(workdir, "results", self.data.dataset.lower())
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, f"{self.run_name()}.pkl")

    def get_results(self, workdir=Constants.workdir) -> pd.DataFrame:
        results_path = self.results_path(workdir=workdir)
        results = pd.read_pickle(results_path)
        results.set_index("idx", inplace=True)
        return results

    def sweep(self, keys: Iterable[str]):
        config_dict = self.to_dict()
        sweep_values = [_get(config_dict, key) for key in keys]
        sweep = list(
            product(*map(lambda x: x if isinstance(x, list) else [x], sweep_values))
        )

        configs: Iterable[Config] = []
        for _sweep in sweep:
            _config_dict = config_dict.copy()
            for key, value in zip(keys, _sweep):
                _set(_config_dict, key, value)
            _config = Config(config_dict=config_dict)
            configs.append(_config)
        return configs


configs: Mapping[str, Config] = {}


def register_config(name: str):
    def register(cls: Config):
        if name in configs:
            raise ValueError(f"Config {name} is already registered")
        configs[name] = cls

    return register


def get_config(
    name: str, config_dict: Mapping[str, Any] | None = {}
) -> Config | Iterable[Config]:
    config: Config = configs[name]()

    if config_dict is not None:
        _config_dict = config.to_dict()
        for key, value in config_dict.items():
            _set(_config_dict, key, value)
        config = Config(config_dict=_config_dict)
    return config
