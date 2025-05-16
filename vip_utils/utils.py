import os
from functools import partial

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from configs import Config
from configs import Constants as C
from datasets import get_dataset
from train_utils import PredictionDataset
from vip_utils.cub import CUBConceptModel
from vip_utils.imagenet import ConceptNet2
from vip_utils.imagenet import answer_query as imagenet_answer_query
from vip_utils.imagenet import get_dictionary


def get_run_name(config: Config, max_queries: int, sampling: str):
    return f"{config.data.dataset.lower()}_vip_query{max_queries}_s{sampling}"


def make_dataset(
    config: Config,
    train: bool,
    transform: T.Compose = None,
    workdir=C.workdir,
    device=C.device,
):
    dataset = get_dataset(
        config, train=train, transform=transform, workdir=workdir, return_attribute=True
    )
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
            image, _, _ = dataset[idx]
            prediction = prediction_dataset[idx]["prediction"]
            return image, prediction

    return _Dataset()


def get_query_answerer(config: Config = None, workdir=C.workdir, device=C.device):
    weights_dir = os.path.join(workdir, "weights", "vip")

    if config.data.dataset.lower() == "cub":
        from torchvision import transforms as T

        concept_net = CUBConceptModel.load_from_checkpoint(
            os.path.join(weights_dir, "cub_concept.pth")
        )
        concept_net.to(device)
        concept_net.requires_grad_(False)
        concept_net.eval()

        preprocess = T.Compose(
            [
                T.CenterCrop(299),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )

        def answer_query(image: torch.Tensor):
            return concept_net.net(image)

    elif config.data.dataset.lower() == "imagenet":
        import clip

        attribute_dir = os.path.join(workdir, "data", "ImageNet", "attributes")

        clip_backbone = "ViT-B/16"
        weights_name = "imagenet_answers_clip_finetuned_depends_no_classifier_epoch_611"

        model_clip, preprocess = clip.load(clip_backbone, device=device)
        dictionary = get_dictionary(
            clip_backbone=clip_backbone, workdir=workdir, device=device
        )

        concept_net = ConceptNet2(embed_dims=512).to(device)
        state_dict = torch.load(
            os.path.join(weights_dir, f"{weights_name}.pt"), map_location=device
        )
        concept_net.load_state_dict(state_dict)
        concept_net.eval()

        with open(os.path.join(attribute_dir, "top_concepts.txt"), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            top_claim_idx = [int(claim_idx) for claim_idx, _ in lines]
            top_claim_idx = torch.tensor(top_claim_idx, device=device)

        answer_query = partial(
            imagenet_answer_query,
            model_clip=model_clip,
            dictionary=dictionary,
            concept_net=concept_net,
            concept_idx=top_claim_idx,
        )

    else:
        raise NotImplementedError(
            f"Dataset {config.data.dataset} not supported for V-IP training."
        )

    return preprocess, answer_query
