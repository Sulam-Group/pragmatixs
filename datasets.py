import os
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset
from tqdm import tqdm

from configs import Config
from configs import Constants as C


class DatasetWithAttributes(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: T.Compose = None,
        return_attribute: bool = False,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.return_attribute = return_attribute

        self.classes: Iterable[str] = None
        self.claims: Iterable[str] = None
        self.samples = None
        self.image_attribute = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_attribute:
            attribute = self.image_attribute[idx]
            return image, label, attribute
        return image, label


datasets: Mapping[str, DatasetWithAttributes] = {}


def register_dataset(name: str) -> DatasetWithAttributes:
    def register(cls: DatasetWithAttributes):
        if name in datasets:
            raise ValueError(f"Dataset {name} is already registered")
        datasets[name] = cls
        return cls

    return register


def get_dataset(
    config: Config,
    train: bool = True,
    transform: T.Compose = None,
    return_attribute: bool = False,
    workdir=C.workdir,
) -> DatasetWithAttributes:
    dataset_name = config.data.dataset.lower()
    root = os.path.join(workdir, "data")
    Dataset = datasets[dataset_name]
    return Dataset(
        root, train=train, transform=transform, return_attribute=return_attribute
    )


@register_dataset(name="cub")
class CUB(DatasetWithAttributes):
    def __init__(
        self,
        root: str,
        train: bool = False,
        transform: T.Compose = None,
        return_attribute: bool = False,
    ):
        super().__init__(
            root, train=train, transform=transform, return_attribute=return_attribute
        )
        self.op = "train" if train else "test"

        self.classes, self.samples = self.get_classes_and_samples()
        self.claims, self.image_attribute = self.get_claims_and_image_attributes()

    def get_classes_and_samples(self):
        image_dir = os.path.join(self.root, "CUB", "images")

        with open(os.path.join(self.root, "CUB", f"{self.op}_filenames.txt"), "r") as f:
            op_filenames = f.readlines()
            op_filenames = [filename.strip() for filename in op_filenames]

        wnids, wnid_to_idx = find_classes(image_dir)
        classes = [wnid.split(".")[1].replace("_", " ") for wnid in wnids]

        samples = make_dataset(image_dir, wnid_to_idx, extensions=".jpg")
        samples = [
            (filename, class_idx)
            for filename, class_idx in samples
            if "/".join(filename.split("/")[-2:]) in op_filenames
        ]
        return classes, samples

    def get_claims_and_image_attributes(self):
        attribute_dir = os.path.join(self.root, "CUB", "attributes")

        with open(os.path.join(attribute_dir, "attributes.txt")) as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            claims = [claim for _, claim in lines]

        op_image_attribute_path = os.path.join(
            self.root, "CUB", f"{self.op}_image_attribute.npy"
        )
        if not os.path.exists(op_image_attribute_path):
            with open(os.path.join(self.root, "CUB", "images.txt"), "r") as f:
                lines = f.readlines()
                lines = [line.strip().split() for line in lines]
                filename_to_idx = {filename: int(idx) for idx, filename in lines}

            image_to_dataset_idx = {
                filename_to_idx["/".join(filename.split("/")[-2:])]: dataset_idx
                for dataset_idx, (filename, _) in enumerate(self.samples)
            }

            image_attribute = -1 * np.ones((len(self.samples), len(self.claims)))
            with open(
                os.path.join(attribute_dir, "image_attribute_labels.txt"), "r"
            ) as f:
                for line in tqdm(f):
                    chunks = line.strip().split()
                    if len(chunks) != 5:
                        continue

                    idx, attribute_idx, attribute_label, confidence, _ = chunks

                    idx = int(idx)
                    attribute_idx = int(attribute_idx) - 1
                    confidence = int(confidence)

                    if idx not in image_to_dataset_idx or confidence < 2:
                        continue

                    dataset_idx = image_to_dataset_idx[idx]
                    image_attribute[dataset_idx, attribute_idx] = attribute_label

            np.save(op_image_attribute_path, image_attribute)

        return claims, np.load(op_image_attribute_path)


@register_dataset(name="imagenette")
class Imagenette(DatasetWithAttributes):
    WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chainsaw", "chain saw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
    }

    def __init__(
        self,
        root: str,
        train: bool = False,
        transform: T.Compose = None,
        return_attribute: bool = False,
    ):
        super().__init__(
            root, train=train, transform=transform, return_attribute=return_attribute
        )
        self.op = "train" if train else "test"

        self.classes, self.samples = self.get_classes_and_samples()
        self.claims, self.image_attribute = self.get_claims_and_image_attributes()

    def get_classes_and_samples(self):
        image_root = os.path.join(self.root, "imagenette", self.op)
        wnids, wnid_to_idx = find_classes(image_root)
        classes = [self.WNID_TO_CLASS[wnid][0] for wnid in wnids]
        samples = make_dataset(image_root, wnid_to_idx, extensions=".jpeg")
