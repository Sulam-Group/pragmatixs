import logging
import os

import torch
import torch.distributed as distributed
from torch.utils.data import Dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from speaker import ClaimSpeaker
from train_utils.prediction import PredictionDataset
from train_utils.utils import get_loader_and_indices, get_rank

logger = logging.getLogger(__name__)


class ExplanationDataset(Dataset):
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
                config.train_cache_dir(workdir=workdir), f"explanation_rank{rank}.pt"
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
        prediction = torch.tensor(self.prediction_dataset[image_idx]["prediction"])
        return {
            "image_idx": image_idx.long(),
            "prediction": prediction.long(),
            "explanation": self.loaded_shard["explanation"][local_idx].long(),
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
            config=config,
            dataset=prediction_dataset,
            batch_size=64,
            shuffle=config.data.distributed,
            epoch=epoch,
        )

        data = {
            "image_idx": torch.tensor(indices).repeat_interleave(k),
            "explanation": -torch.ones(k * len(indices), context_length, 2),
        }

        start = 0
        for _, _data in enumerate(tqdm(dataloader)):
            image_tokens = _data["image_tokens"]
            image_tokens = torch.repeat_interleave(image_tokens, k, dim=0).to(device)

            length = config.data.explanation_length
            if config.speaker.alpha > 0:
                length = None

            explanation, _ = explain(image_tokens, length=length)

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
    data["length"] = len(data["image_idx"])
    torch.save(data, data_path)
