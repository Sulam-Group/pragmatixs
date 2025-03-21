import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from speaker import ClaimSpeaker
from train_utils.prediction import PredictionDataset


class ExplanationDataset(Dataset):
    def __init__(
        self,
        config: Config,
        dataset: PredictionDataset,
        speaker: ClaimSpeaker,
        device=C.device,
    ):
        data = self.make_dataset(config, dataset, speaker, device)
        self.image_idx = data["image_idx"]
        self.explanation = data["explanation"]

    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, idx):
        return {
            "image_idx": self.image_idx[idx].long(),
            "explanation": self.explanation[idx].long(),
        }

    @torch.no_grad()
    def make_dataset(
        self,
        config: Config,
        dataset: PredictionDataset,
        speaker: ClaimSpeaker,
        device: torch.device,
    ):
        speaker.eval()

        k = config.listener.k
        context_length = speaker.context_length

        data = {
            "image_idx": torch.arange(len(dataset)).repeat_interleave(k),
            "explanation": -torch.ones(k * len(dataset), context_length, 2),
        }

        start = 0
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        for _, _data in enumerate(tqdm(dataloader)):
            image_tokens = _data["image_tokens"]

            image_tokens = torch.repeat_interleave(image_tokens, k, dim=0).to(device)

            explanation, _ = speaker.explain(image_tokens)

            end = start + explanation.size(0)

            data["explanation"][start:end] = explanation

            start = end

        return data
