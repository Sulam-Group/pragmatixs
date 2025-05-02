import logging
import os

import torch
import torch.distributed as distributed
from torch.utils.data import Dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from listeners import Listener
from speaker import ClaimSpeaker
from train_utils.prediction import PredictionDataset
from train_utils.utils import get_loader_and_indices, get_rank

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    def __init__(
        self,
        config: Config = None,
        prediction_dataset: PredictionDataset = None,
        workdir=C.workdir,
    ):
        self.prediction_dataset = prediction_dataset

        self.shard_paths = []
        self.shard_lengths = []
        self.index = []

        world_size = 1
        if distributed.is_initialized():
            world_size = distributed.get_world_size()

        for rank in range(world_size):
            shard_path = os.path.join(
                config.train_cache_dir(workdir=workdir), f"preference_rank{rank}.pt"
            )
            shard_data = torch.load(shard_path, map_location="cpu")
            shard_length = shard_data["length"]

            self.shard_paths.append(shard_path)
            self.shard_lengths.append(shard_length)
            for local_idx in range(shard_length):
                self.index.append((rank, local_idx))

        self.loaded_shard_id = None
        self.loaded_shard = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        shard_id, local_idx = self.index[idx]

        if shard_id != self.loaded_shard_id:
            self.loaded_shard_id = shard_id
            self.loaded_shard = torch.load(
                self.shard_paths[shard_id], map_location="cpu"
            )

        image_idx = self.loaded_shard["image_idx"][local_idx]
        image_tokens = torch.tensor(self.prediction_dataset[image_idx]["image_tokens"])

        return {
            "image_idx": image_idx.long(),
            "image_tokens": image_tokens.float(),
            "chosen": self.loaded_shard["chosen"][local_idx].long(),
            "rejected": self.loaded_shard["rejected"][local_idx].long(),
            "chosen_score": self.loaded_shard["chosen_score"][local_idx],
            "rejected_score": self.loaded_shard["rejected_score"][local_idx],
            "chosen_logp": self.loaded_shard["chosen_logp"][local_idx],
            "rejected_logp": self.loaded_shard["rejected_logp"][local_idx],
            "margin_mask": self.loaded_shard["margin_mask"][local_idx],
        }

    @staticmethod
    @torch.no_grad()
    def make_dataset(
        config: Config = None,
        prediction_dataset: PredictionDataset = None,
        speaker: ClaimSpeaker = None,
        listener: Listener = None,
        epoch: int = 0,
        device=C.device,
    ):
        speaker.eval()
        listener.eval()

        explain = speaker.module.explain if config.data.distributed else speaker.explain
        listen = listener.module.listen if config.data.distributed else listener.listen

        dataloader, indices = get_loader_and_indices(
            config=config,
            dataset=prediction_dataset,
            batch_size=64,
            shuffle=config.data.distributed,
            epoch=epoch,
        )

        k = config.speaker.k
        pair_mask = torch.combinations(torch.arange(k), r=2)
        n_pairs = pair_mask.size(0)
        n_preferences = n_pairs * len(indices)
        context_length = config.data.explanation_length + 1

        data = {
            "image_idx": torch.tensor(indices).repeat_interleave(n_pairs),
            "chosen": -torch.ones(n_preferences, context_length, 2),
            "rejected": -torch.ones(n_preferences, context_length, 2),
            "chosen_score": -torch.ones(n_preferences),
            "rejected_score": -torch.ones(n_preferences),
            "chosen_logp": -torch.ones(n_preferences),
            "rejected_logp": -torch.ones(n_preferences),
            "margin_mask": torch.zeros(n_preferences, dtype=torch.bool),
        }

        start = 0
        for _, _data in enumerate(tqdm(dataloader)):
            _data = {
                n: torch.repeat_interleave(v, k, dim=0).to(device)
                for n, v in _data.items()
            }

            image_tokens = _data["image_tokens"]
            image_attribute = _data["image_attribute"]
            prediction = _data["prediction"]

            length = config.data.explanation_length
            if config.speaker.alpha > 0:
                length = None

            explanation, explanation_logp = explain(image_tokens, length=length)
            consistency, action = listen(image_attribute, explanation)

            action_loss = torch.nn.functional.cross_entropy(
                action, prediction, reduction="none"
            )

            explanation_score = consistency - config.speaker.alpha * action_loss

            b = explanation.size(0) // k
            explanation = explanation.view(b, k, context_length, 2)
            explanation_logp = explanation_logp.view(b, k)
            explanation_score = explanation_score.view(b, k)

            pair_explanation = explanation[:, pair_mask]
            pair_logp = explanation_logp[:, pair_mask]
            pair_score = explanation_score[:, pair_mask]

            sorted_pair_idx = torch.argsort(-pair_score, dim=-1)
            sorted_pair_explanation = torch.take_along_dim(
                pair_explanation, sorted_pair_idx[..., None, None], dim=2
            )
            sorted_pair_score = torch.take_along_dim(
                pair_score, sorted_pair_idx, dim=-1
            )
            sorted_pair_logp = torch.take_along_dim(pair_logp, sorted_pair_idx, dim=-1)

            sorted_pair_explanation = sorted_pair_explanation.flatten(0, 1)
            sorted_pair_score = sorted_pair_score.flatten(0, 1)
            sorted_pair_logp = sorted_pair_logp.flatten(0, 1)

            win_margin = 0.50
            win_rate = torch.softmax(sorted_pair_score, dim=-1)[:, 0]
            margin_mask = win_rate > win_margin

            end = start + sorted_pair_explanation.size(0)

            data["chosen"][start:end] = sorted_pair_explanation[:, 0]
            data["rejected"][start:end] = sorted_pair_explanation[:, 1]
            data["chosen_score"][start:end] = sorted_pair_score[:, 0]
            data["rejected_score"][start:end] = sorted_pair_score[:, 1]
            data["chosen_logp"][start:end] = sorted_pair_logp[:, 0]
            data["rejected_logp"][start:end] = sorted_pair_logp[:, 1]
            data["margin_mask"][start:end] = margin_mask

            start = end

        mask = data["margin_mask"]
        return {k: v[mask] for k, v in data.items()}


def generate_utterances(
    config: Config = None,
    prediction_dataset: PredictionDataset = None,
    speaker: ClaimSpeaker = None,
    listener: Listener = None,
    epoch: int = 0,
    workdir=C.workdir,
    device=C.device,
):
    logger.info("Sampling utterances...")

    data_path = os.path.join(
        config.train_cache_dir(workdir=workdir), f"utterance_rank{get_rank()}.pt"
    )
    if os.path.exists(data_path):
        os.remove(data_path)

    explain = speaker.module.explain if config.data.distributed else speaker.explain

    dataloader, indices = get_loader_and_indices(
        config=config,
        dataset=prediction_dataset,
        shuffle=config.data.distributed,
        epoch=epoch,
    )

    k = config.speaker.k
    n_utterances = k * len(indices)
    data = {
        "image_idx": torch.tensor(indices).repeat_interleave(config.speaker.k),
        "explanation": -torch.ones(n_utterances, speaker.context_length, 2),
    }

    start = 0
    for _, _data in enumerate(tqdm(dataloader)):
        _data = {
            n: torch.repeat_interleave(v, k, dim=0).to(device) for n, v in _data.items()
        }

        image_tokens = _data["image_tokens"]
        image_attribute = _data["image_attribute"]
        prediction = _data["prediction"]


def generate_and_save_preferences(
    config: Config = None,
    prediction_dataset: PredictionDataset = None,
    speaker: ClaimSpeaker = None,
    listener: Listener = None,
    epoch: int = 0,
    workdir=C.workdir,
    device=C.device,
):
    logger.info("Creating preference dataset...")

    data_path = os.path.join(
        config.train_cache_dir(workdir=workdir), f"preference_rank{get_rank()}.pt"
    )
    if os.path.exists(data_path):
        os.remove(data_path)

    data = PreferenceDataset.make_dataset(
        config=config,
        prediction_dataset=prediction_dataset,
        speaker=speaker,
        listener=listener,
        epoch=epoch,
        device=device,
    )
    data["length"] = len(data["image_idx"])
    torch.save(data, data_path)

    logger.info(
        f"Number of preferences: {data['length']}, "
        f"average chosen score: {data['chosen_score'].mean().item():.2f}"
    )
