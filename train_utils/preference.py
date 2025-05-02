import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from listeners import ClaimListener
from speaker import ClaimSpeaker
from train_utils.prediction import PredictionDataset


class PreferenceDataset(Dataset):
    def __init__(
        self,
        config: Config,
        dataset: PredictionDataset,
        speaker: ClaimSpeaker,
        listener: ClaimListener,
        device=C.device,
    ):
        data = self.make_dataset(config, dataset, speaker, listener, device)
        margin_mask = data["margin_mask"]
        data = {k: v[margin_mask] for k, v in data.items()}
        self.image_idx = data["image_idx"]
        self.chosen = data["chosen"]
        self.rejected = data["rejected"]
        self.chosen_score = data["chosen_score"]
        self.rejected_score = data["rejected_score"]
        self.chosen_logp = data["chosen_logp"]
        self.rejected_logp = data["rejected_logp"]
        self.margin_mask = data["margin_mask"]

    def __len__(self):
        return len(self.image_idx)

    def __getitem__(self, idx):
        return {
            "image_idx": self.image_idx[idx].long(),
            "chosen": self.chosen[idx].long(),
            "rejected": self.rejected[idx].long(),
            "chosen_score": self.chosen_score[idx],
            "rejected_score": self.rejected_score[idx],
            "chosen_logp": self.chosen_logp[idx],
            "rejected_logp": self.rejected_logp[idx],
            "margin_mask": self.margin_mask[idx],
        }

    @torch.no_grad()
    def make_dataset(
        self,
        config: Config,
        dataset: PredictionDataset,
        speaker: ClaimSpeaker,
        listener: ClaimListener,
        device: torch.device,
    ):
        speaker.eval()
        listener.eval()

        k = config.speaker.k
        pair_mask = torch.combinations(torch.arange(k), r=2)
        n_pairs = pair_mask.size(0)
        n_preferences = n_pairs * len(dataset)
        context_length = speaker.context_length

        data = {
            "image_idx": torch.arange(len(dataset)).repeat_interleave(n_pairs),
            "chosen": -torch.ones(n_preferences, context_length, 2),
            "rejected": -torch.ones(n_preferences, context_length, 2),
            "chosen_score": -torch.ones(n_preferences),
            "rejected_score": -torch.ones(n_preferences),
            "chosen_logp": -torch.ones(n_preferences),
            "rejected_logp": -torch.ones(n_preferences),
            "margin_mask": torch.zeros(n_preferences, dtype=torch.bool),
        }

        start = 0
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        for _, _data in enumerate(tqdm(dataloader)):
            _data = {
                n: torch.repeat_interleave(v, k, dim=0).to(device)
                for n, v in _data.items()
            }

            image_tokens = _data["image_tokens"]
            image_attribute = _data["image_attribute"]
            prediction = _data["prediction"]

            explanation, explanation_logp = speaker.explain(image_tokens)
            consistency, action = listener.listen(image_attribute, explanation)
            action_loss = torch.nn.functional.cross_entropy(
                action, prediction, reduction="none"
            )
