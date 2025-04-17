import logging
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from speaker import ClaimSpeaker
from train_utils.prediction import PredictionDataset
from train_utils.utils import (
    get_loader_and_indices,
    get_rank,
    truncate_to_shortest_shard,
)

logger = logging.getLogger(__name__)


class ExplanationDataset(Dataset):
    def __init__(
        self,
        config: Config = None,
        prediction_dataset: PredictionDataset = None,
        workdir=C.workdir,
    ):
        data_path = os.path.join(
            config.train_cache_dir(workdir=workdir), f"explanation_rank{get_rank()}.pt"
        )

        self.prediction_dataset = prediction_dataset
        self.data = torch.load(data_path, map_location="cpu")

    def __len__(self):
        return len(self.data["length"])

    def __getitem__(self, idx):
        image_idx = self.data["image_idx"][idx]
        prediction = torch.tensor(self.prediction_dataset[image_idx]["prediction"])
        return {
            "image_idx": image_idx.long(),
            "prediction": prediction.long(),
            "explanation": self.data["explanation"][idx].long(),
        }

    @staticmethod
    @torch.no_grad()
    def make_dataset(
        config: Config = None,
        prediction_dataset: PredictionDataset = None,
        speaker: ClaimSpeaker = None,
        epoch: int = 0,
        device=C.device,
    ):
        speaker.eval()

        explain = speaker.module.explain if config.data.distributed else speaker.explain

        k = config.listener.k
        context_length = config.data.explanation_length + 1

        dataloader, indices = get_loader_and_indices(
            config=config, dataset=prediction_dataset, shuffle=True, epoch=epoch
        )

        data = {
            "image_idx": torch.tensor(indices).repeat_interleave(k),
            "explanation": -torch.ones(k * len(indices), context_length, 2),
        }

        start = 0
        for _, _data in enumerate(tqdm(dataloader)):
            image_tokens = _data["image_tokens"]
            image_tokens = torch.repeat_interleave(image_tokens, k, dim=0).to(device)
            explanation, _ = explain(image_tokens)

            end = start + explanation.size(0)
            data["explanation"][start:end] = explanation
            start = end

        return data


def generate_and_save_explanations(
    config: Config = None,
    prediction_dataset: PredictionDataset = None,
    speaker: ClaimSpeaker = None,
    epoch: int = 0,
    workdir=C.workdir,
    device=C.device,
):
    logger.info("Creating explanation dataset...")

    data_path = os.path.join(
        config.train_cache_dir(workdir=workdir), f"explanation_rank{get_rank()}.pt"
    )
    if os.path.exists(data_path):
        os.remove(data_path)

    data = ExplanationDataset.make_dataset(
        config=config,
        prediction_dataset=prediction_dataset,
        speaker=speaker,
        epoch=epoch,
        device=device,
    )
    data = truncate_to_shortest_shard(data, device=device)
    torch.save(data, data_path)
