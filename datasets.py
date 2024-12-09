import os
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes, make_dataset
from tqdm import tqdm

from configs.utils import Config
from configs.utils import Constants as c


class DatasetWithAttributes(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[[Image.Image], torch.Tensor] = None,
        return_attribute: bool = False,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.return_attribute = return_attribute

        self.classes, self.samples = None, None
        self.claims, self.image_attribute = None, None

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


def register_dataset(name: str):
    def register(cls: DatasetWithAttributes):
        if name in datasets:
            raise ValueError(f"Dataset {name} is already registered")
        datasets[name] = cls

    return register


def get_dataset(
    config: Config,
    train=True,
    transform=None,
    return_attribute=False,
    workdir=c.WORKDIR,
) -> DatasetWithAttributes:
    dataset_name = config.dataset_name.lower()
    root = os.path.join(workdir, "data")
    Dataset = datasets[dataset_name]
    return Dataset(
        root, train=train, transform=transform, return_attribute=return_attribute
    )


@register_dataset(name="cub")
class CUB(DatasetWithAttributes):
    def __init__(self, root, train=False, transform=None, return_attribute=False):
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


@register_dataset(name="ham")
class HAM(DatasetWithAttributes):
    wnids_to_idx = {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "nv": 4,
        "mel": 5,
        "vasc": 6,
    }

    def __init__(
        self, root, train=False, transform=None, return_attribute=False, finetune=False
    ):
        super().__init__(
            root, train=train, transform=transform, return_attribute=return_attribute
        )
        if finetune:
            self.op = "finetune"
        else:
            self.op = "train" if train else "test"

        self.classes = [
            "actinic keratosis",
            "basal cell carcinoma",
            "benign keratosis",
            "dermatofibroma",
            "melanocytic nevi",
            "melanoma",
            "vascular lesions",
        ]
        self.samples = self.get_samples()
        self.claims, self.image_attribute = self.get_claims_and_image_attributes()

    def get_samples(self):
        ham_dir = os.path.join(self.root, "HAM10000")
        image_dir = os.path.join(ham_dir, "images")

        with open(os.path.join(ham_dir, f"{self.op}_images.txt"), "r") as f:
            op_filenames = f.readlines()
            op_filenames = [filename.strip() for filename in op_filenames]

        metadata = pd.read_csv(os.path.join(ham_dir, "metadata.csv"))
        filenames = metadata["image_id"].values.tolist()
        diagnoses = metadata["dx"].values.tolist()

        return [
            (os.path.join(image_dir, f"{filename}.jpg"), self.wnids_to_idx[dx])
            for filename, dx in zip(filenames, diagnoses)
            if filename in op_filenames
        ]

    def get_claims_and_image_attributes(self):
        skincon_dir = os.path.join(self.root, "SKINCON")
        attribute_dir = os.path.join(self.root, "HAM10000", "attributes")

        with open(os.path.join(skincon_dir, "attributes.txt"), "r") as f:
            lines = f.readlines()
            claims = [line.strip() for line in lines]

        image_attribute = np.load(
            os.path.join(attribute_dir, f"{self.op}_image_attribute.npy")
        )
        return claims, image_attribute


class SKINCON(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        image_root = os.path.join(root, "SKINCON", "Fitz17k", "images")
        metadata_path = os.path.join(root, "SKINCON", "image.csv")
        metadata = pd.read_csv(metadata_path)

        with open(os.path.join(root, "SKINCON", "attributes.txt"), "r") as f:
            self.attributes = f.read().splitlines()

        metadata.rename(
            columns={
                "Brown(Hyperpigmentation)": "Hyperpigmentation",
                "White(Hypopigmentation)": "Hypopigmentation",
            },
            inplace=True,
        )
        metadata = metadata[metadata["Do not consider this image"] == 0]
        image_idx = metadata["ImageID"].values.tolist()
        image_attribute = metadata[self.attributes].values.tolist()

        self.samples = [
            (os.path.join(image_root, f"{idx}"), attribute)
            for idx, attribute in zip(image_idx, image_attribute)
            if os.path.exists(os.path.join(image_root, f"{idx}"))
        ]
        self.image_attribute = [attribute for _, attribute in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, attribute = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, attribute
