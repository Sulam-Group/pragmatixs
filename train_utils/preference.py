import logging
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from listeners import Listener
from speaker import ClaimSpeaker
from train_utils.prediction import PredictionDataset
from train_utils.utils import (
    get_loader_and_indices,
    get_rank,
    truncate_to_shortest_shard,
)

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    def __init__(
        self,
        config: Config = None,
        prediction_dataset: PredictionDataset = None,
        workdir=C.workdir,
    ):
        data_path = os.path.join(
            config.train_cache_dir(workdir=workdir), f"preference_rank{get_rank()}.pt"
        )

        self.prediction_dataset = prediction_dataset
        self.data = torch.load(data_path, map_location="cpu")

    def __len__(self):
        return len(self.data["length"])

    def __getitem__(self, idx):
        image_idx = self.data["image_idx"][idx]
        image_tokens = torch.tensor(self.prediction_dataset[image_idx]["image_tokens"])

        return {
            "image_idx": image_idx.long(),
            "image_tokens": image_tokens.float(),
            "chosen": self.data["chosen"][idx].long(),
            "rejected": self.data["rejected"][idx].long(),
            "chosen_score": self.data["chosen_score"][idx],
            "rejected_score": self.data["rejected_score"][idx],
            "chosen_logp": self.data["chosen_logp"][idx],
            "rejected_logp": self.data["rejected_logp"][idx],
            "margin_mask": self.data["margin_mask"][idx],
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
            config=config, dataset=prediction_dataset, shuffle=True, epoch=epoch
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

            length = torch.randint(
                1, config.data.explanation_length, (image_tokens.size(0),)
            )

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

        return data


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
    data = truncate_to_shortest_shard(data, device=device)
    torch.save(data, data_path)

    logger.info(
        f"Number of preferences: {data['length']}, "
        f"average chosen score: {data['chosen_score'].mean().item():.2f}"
    )
