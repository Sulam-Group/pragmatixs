import os
from collections.abc import Iterable, Mapping

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

        self.op = None
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

    kwargs = {}
    if dataset_name == "chexpert":
        kwargs = {"task": config.data.task}

    root = os.path.join(workdir, "data")
    Dataset = datasets[dataset_name]
    return Dataset(
        root,
        train=train,
        transform=transform,
        return_attribute=return_attribute,
        **kwargs,
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

        with open(os.path.join(self.root, "CUB", f"{self.op}_filenames.txt")) as f:
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
            with open(os.path.join(self.root, "CUB", "images.txt")) as f:
                lines = f.readlines()
                lines = [line.strip().split() for line in lines]
                filename_to_idx = {filename: int(idx) for idx, filename in lines}

            image_to_dataset_idx = {
                filename_to_idx["/".join(filename.split("/")[-2:])]: dataset_idx
                for dataset_idx, (filename, _) in enumerate(self.samples)
            }

            image_attribute = -1 * np.ones((len(self.samples), len(self.claims)))
            with open(os.path.join(attribute_dir, "image_attribute_labels.txt")) as f:
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


@register_dataset(name="imagenet")
class ImageNet(DatasetWithAttributes):
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
        self.op = "train" if train else "val"

        self.classes, self.samples = self.get_classes_and_samples()
        self.claims, self.image_attribute = self.get_claims_and_image_attributes()

    def get_classes_and_samples(self):
        dataset_dir = os.path.join(self.root, "ImageNet")

        # with open(os.path.join(dataset_dir, "imagenette_classes.txt")) as f:
        #     lines = f.readlines()
        # with open(os.path.join(dataset_dir, "imagewoof_classes.txt")) as f:
        #     lines += f.readlines()
        # with open(os.path.join(dataset_dir, "wnids_to_class.txt")) as f:
        #     lines = f.readlines()
        with open(os.path.join(dataset_dir, "top_classes.txt")) as f:
            lines = f.readlines()

        wnids_to_class = {}
        for line in lines:
            line = line.strip().replace(", ", ",")
            chunks = line.split()
            wnid, class_names = chunks[0], " ".join(chunks[1:])
            class_name = class_names.split(",")[0]
            wnids_to_class[wnid] = class_name

        # with open(os.path.join(dataset_dir, "imagewoof_classes.txt")) as f:
        #     lines = f.readlines()
        #     wnids_to_class = {}
        #     for line in lines:
        #         chunks = [c.strip().replace(",", "") for c in line.split()]
        #         wnid = chunks[0]
        #         classes = chunks[1:]
        #         wnids_to_class[wnid] = classes

        # with open(os.path.join(dataset_dir, "top_classes.txt")) as f:
        #     lines = f.readlines()
        #     lines = [line.strip().split() for line in lines]
        #     wnids = [wnid for wnid, _, _ in lines]
        #     wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        wnids = list(wnids_to_class.keys())
        classes = list(wnids_to_class.values())
        wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        image_dir = os.path.join(dataset_dir, self.op)
        samples = make_dataset(image_dir, wnid_to_idx, extensions=".jpeg")

        samples_per_class = 200
        class_samples = {k: [] for k, _ in enumerate(classes)}
        for filename, class_idx in samples:
            class_samples[class_idx].append((filename, class_idx))
        class_samples = {k: v[:samples_per_class] for k, v in class_samples.items()}

        samples = []
        for class_idx, class_samples in class_samples.items():
            samples.extend(class_samples)
        return classes, samples

    def get_claims_and_image_attributes(self):
        attribute_dir = os.path.join(self.root, "ImageNet", "attributes")
        with open(os.path.join(attribute_dir, "attributes.txt")) as f:
            lines = f.readlines()
            claims = [line.strip() for line in lines]

        op_image_attribute_path = os.path.join(
            attribute_dir, f"{self.op}_image_attribute.npy"
        )
        assert os.path.exists(op_image_attribute_path), (
            f"Image attribute file {op_image_attribute_path} not found. "
            "Make sure to run `preprocess/imagenet/label_concept_vqa.py` first, "
            "or that the path is correct."
        )
        return claims, np.load(op_image_attribute_path)


@register_dataset(name="chexpert")
class CheXpert(DatasetWithAttributes):
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
        self.TASK = config.data.task
        self.op = "train" if train else "val"
        if self.op == "train":
            self.info_df = pd.read_csv(
                os.path.join(self.root, "chexpert", "train_visualCheXbert.csv"),
                header=0,
                index_col=0,
            )
        else:
            self.info_df = pd.read_csv(
                os.path.join(self.root, "chexpert", "val.csv"),
                header=0,
                index_col=0,
            )
        self.classes, self.samples = self.get_classes_and_samples()
        self.claims, self.image_attribute = self.get_claims_and_image_attributes()

    def get_classes_and_samples(self):
        image_dir = os.path.join(self.root, "chexpert", self.op)
        patient_ids = os.listdir(image_dir)
        # only keep the patients with id < 10000 (for training)
        if self.op == "train":
            patient_ids = [
                i for i in patient_ids if int(i.removeprefix("patient")) < 10000
            ]
        image_paths = list(self.info_df.index)
        image_paths = [
            os.path.join(image_dir, i.split("/")[2], "/".join(i.split("/")[3:]))
            for i in image_paths
            if i.split("/")[2] in patient_ids
        ]
        indices = [i for i in self.info_df.index if i.split("/")[2] in patient_ids]
        self.info_df = self.info_df.loc[indices]

        labels = list(self.info_df[self.TASK])
        classes = [f"No signs of {self.TASK}", f"Findings suggesting {self.TASK}"]
        samples = [
            (image_path, label) for image_path, label in zip(image_paths, labels)
        ]
        return classes, samples

    def get_claims_and_image_attributes(self):
        claims = [
            "Enlarged Cardiomediastinum",
            "Lung Opacity",
            "Cardiomegaly",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]
        claims = [i for i in claims if i != self.TASK]
        image_attribute = self.info_df[claims].values
        return claims, image_attribute
