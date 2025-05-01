import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from classifiers import get_classifier
from configs import Config
from configs import Constants as C
from datasets import DatasetWithAttributes
from train_utils.utils import get_loader_and_indices

logger = logging.getLogger(__name__)


class PredictionDataset(Dataset):
    def __init__(
        self,
        config: Config,
        dataset: DatasetWithAttributes,
        workdir=C.workdir,
        device=C.device,
    ):
        cache_dir = os.path.join(workdir, "data", "embed_cache")
        self.embed_dir = embed_dir = os.path.join(
            cache_dir,
            f"{config.data.dataset.lower()}_{config.classifier_name()}",
            "train" if dataset.train else "val",
        )
        os.makedirs(embed_dir, exist_ok=True)

        if len(os.listdir(embed_dir)) != len(dataset):
            logger.info(f"Generating {dataset.op} prediction dataset")

            self.make_dataset(
                config=config, dataset=dataset, workdir=workdir, device=device
            )
            if config.data.distributed:
                torch.distributed.barrier()

        logger.info(f"Loading {dataset.op} prediction dataset from {embed_dir}")
        self.filenames = os.listdir(embed_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.embed_dir, self.filenames[idx])
        return np.load(filepath)

    @torch.no_grad()
    def make_dataset(
        self,
        config: Config = None,
        dataset: DatasetWithAttributes = None,
        workdir=C.workdir,
        device=C.device,
    ):
        classifier = get_classifier(
            config, from_pretrained=True, workdir=workdir, device=device
        )
        classifier.eval()

        classes = dataset.classes
        class_prompts = [f"A photo of a {c}" for c in classes]

        dataloader, indices = get_loader_and_indices(
            config=config, dataset=dataset, shuffle=False
        )

        start = 0
        for _, _data in enumerate(tqdm(dataloader)):
            image, _, image_attribute = _data

            image = image.to(classifier.device)

            output = classifier(image, text=class_prompts)
            image_tokens = output["image_tokens"]
            logits = output["logits"]
            prediction = torch.argmax(logits, dim=-1)

            for i, (_image_tokens, _image_attribute, _prediction) in enumerate(
                zip(image_tokens, image_attribute, prediction)
            ):
                idx = indices[start + i]
                filepath = os.path.join(self.embed_dir, f"{idx}.npz")
                np.savez(
                    filepath,
                    **{
                        "image_tokens": _image_tokens.cpu().numpy(),
                        "image_attribute": _image_attribute.cpu().numpy(),
                        "prediction": _prediction.cpu().numpy(),
                    },
                )

            start += image.size(0)
