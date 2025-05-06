import os

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from configs import Config
from configs import Constants as C
from datasets import get_dataset
from train_utils import PredictionDataset
from vip_utils.cub import CUBConceptModel, NetworkCUB

transform = T.Compose(
    [
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
    ]
)


def get_run_name(config: Config, max_queries: int, sampling: str):
    return f"{config.data.dataset.lower()}_vip_query{max_queries}_s{sampling}"


def make_dataset(config: Config, train: bool, workdir=C.workdir, device=C.device):
    dataset = get_dataset(config, train=train, transform=transform, workdir=workdir)
    prediction_dataset = PredictionDataset(
        config, dataset, workdir=workdir, device=device
    )

    class _Dataset(Dataset):
        def __init__(self):
            self.classes = dataset.classes
            self.claims = dataset.claims

        def __len__(self):
            return len(dataset)

        def __getitem__(self, idx):
            image, _ = dataset[idx]
            prediction = prediction_dataset[idx]["prediction"]
            return image, prediction

    return _Dataset()


def load_concept_net(workdir=C.workdir, device=C.device):
    weights_dir = os.path.join(workdir, "weights", "vip")

    concept_net = CUBConceptModel.load_from_checkpoint(
        os.path.join(weights_dir, "cub_concept.pth")
    )
    concept_net.to(device)
    concept_net.requires_grad_(False)
    concept_net.eval()
    return concept_net


def load_vip(
    config: Config, max_queries: int, tau: float, workdir=C.workdir, device=C.device
):
    run_name = get_run_name(config, max_queries, "biased")
    weights_dir = os.path.join(workdir, "weights", run_name)

    concept_net = load_concept_net(workdir=workdir, device=device)

    with open(os.path.join(weights_dir, "latest.txt")) as f:
        latest = f.read().strip()
    state_dict = torch.load(os.path.join(weights_dir, latest), map_location=device)

    querier = NetworkCUB(query_size=312, output_size=312, tau=tau)
    querier.to(device)
    querier.load_state_dict(state_dict["querier"])
    querier.eval()

    classifier = NetworkCUB(query_size=312, output_size=200, tau=None)
    classifier.to(device)
    classifier.load_state_dict(state_dict["classifier"])
    classifier.eval()

    return concept_net, querier, classifier
