import logging
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from speaker import ClaimSpeaker
from train_utils.prediction import PredictionDataset
from train_utils.utils import rank_zero_only

logger = logging.getLogger(__name__)


class ExplanationDataset(Dataset):
    def __init__(
        self,
        config: Config = None,
        prediction_dataset: PredictionDataset = None,
        workdir=C.workdir,
    ):
        self.prediction_dataset = prediction_dataset
        data_path = os.path.join(
            config.train_cache_dir(workdir=workdir), f"explanation.pt"
        )
        self.data = torch.load(data_path, map_location="cpu")

    def __len__(self):
        return self.data["length"]

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
        device=C.device,
    ):
        speaker.eval()

        explain = speaker.module.explain if config.data.distributed else speaker.explain

        k = config.listener.k
        context_length = config.data.explanation_length + 1

        dataloader = DataLoader(prediction_dataset, batch_size=64, shuffle=False)
        indices = list(range(len(prediction_dataset)))

        data = {
            "image_idx": torch.tensor(indices).repeat_interleave(k),
            "explanation": -torch.ones(k * len(indices), context_length, 2),
        }

        start = 0
        for _, _data in enumerate(tqdm(dataloader)):
            _data = {
                n: torch.repeat_interleave(v, k, dim=0).to(device)
                for n, v in _data.items()
            }

            image_tokens = _data["image_tokens"]
            image_attribute = _data["image_attribute"]

            length = config.data.explanation_length
            if config.speaker.alpha > 0:
                length = None

            explanation, _ = explain(image_tokens, length=length)

            end = start + explanation.size(0)
            data["explanation"][start:end] = explanation
            start = end

        return data


@rank_zero_only
def generate_and_save_explanations(
    config: Config = None,
    prediction_dataset: PredictionDataset = None,
    speaker: ClaimSpeaker = None,
    workdir=C.workdir,
    device=C.device,
):
    logger.info("Creating explanation dataset...")

    data_path = os.path.join(config.train_cache_dir(workdir=workdir), f"explanation.pt")
    if os.path.exists(data_path):
        os.remove(data_path)

    data = ExplanationDataset.make_dataset(
        config=config,
        prediction_dataset=prediction_dataset,
        speaker=speaker,
        device=device,
    )
    data["length"] = len(data["image_idx"])
    torch.save(data, data_path)
