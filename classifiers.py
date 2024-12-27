import os
from abc import abstractmethod
from typing import Mapping

import clip
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.utils import Config
from configs.utils import Constants as c


class ImageClassifier(nn.Module):
    def __init__(self, config: Config, device=c.DEVICE):
        super().__init__()
        self.config = config

        self.embed_dim: int = None
        self.preprocess: T.Compose = None
        self.device = device

        self.to(device)

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, config: Config, workdir=c.WORKDIR, device=c.DEVICE
    ) -> "ImageClassifier":
        pass

    @abstractmethod
    def predict(self, dataset: Dataset):
        pass


classifiers: Mapping[str, ImageClassifier] = {}


def register_classifier(name: str):
    def register(cls: ImageClassifier):
        if name in classifiers:
            raise ValueError(f"Classifier {name} is already registered")
        classifiers[name] = cls

    return register


def get_classifier(name: str, **kwargs) -> ImageClassifier:
    return classifiers[name]


@register_classifier(name="clip")
class CLIPClassifier(ImageClassifier):
    def __init__(self, config: Config, device=c.DEVICE):
        super().__init__(config, device)
        backbone = config.data.classifier.split(":")[1]

        self.model, self.preprocess = clip.load(backbone, device=device)
        self.tokenize = clip.tokenize

        self.embed_dim = self.model.visual.output_dim
        self.device = device

    @classmethod
    def from_pretrained(cls, config: Config, workdir=c.WORKDIR, device=c.DEVICE):
        return cls(config, device=device)

    def encode_text(self, text):
        text = self.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text).float()
        return text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    def encode_image(self, image):
        image_features = self.model.encode_image(image).float()
        return image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

    def forward(self, image, text=None, text_features=None):
        assert text is not None or text_features is not None

        image_features = self.encode_image(image)
        if text_features is None:
            text_features = self.encode_text(text)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits": logits_per_image,
        }

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        classes = dataset.classes
        class_prompts = [f"A photo of a {class_name}" for class_name in classes]
        text_features = self.encode_text(class_prompts)

        results = {"label": [], "prediction": []}
        for data in tqdm(dataloader):
            image, label = data

            image = image.to(self.device)

            output = self(image, text_features=text_features)
            logits = output["logits"]
            prediction = torch.argmax(logits, dim=-1).cpu()

            results["label"].extend(label.squeeze().tolist())
            results["prediction"].extend(prediction.squeeze().tolist())

        return pd.DataFrame(results)


@register_classifier(name="ham_biomedclip")
class HAMBiomedCLIP(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.embed_dim, self.n_classes = 512, 7
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            device=device,
        )

        self.proj = nn.Linear(self.embed_dim, self.n_classes)

        nn.init.xavier_normal_(self.proj.weight)

        self.to(device)
        self.device = device

    @staticmethod
    def from_pretrained(workdir: str, device: torch.device):
        model = HAMBiomedCLIP(device=device)
        state_dict = torch.load(
            os.path.join(workdir, "weights", "ham_biomedclip.pt"), map_location=device
        )
        model.load_state_dict(state_dict)
        return model

    def encode_image(self, image):
        image_features = self.model.encode_image(image).float()
        return image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

    def forward(self, image):
        image_features = self.encode_image(image)
        logits = self.proj(image_features)
        return {
            "image_features": image_features,
            "logits": logits,
        }

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        results = {"label": [], "prediction": []}
        for data in tqdm(dataloader):
            image, label = data

            image = image.to(self.device)

            output = self(image)
            logits = output["logits"]
            prediction = torch.argmax(logits, dim=-1).cpu()

            results["label"].extend(label.squeeze().tolist())
            results["prediction"].extend(prediction.squeeze().tolist())

        return pd.DataFrame(results)


# class MONET(CLIPClassifier):
#     backbone = "ViT-L/14"
#     weight_dir = os.path.join(os.path.dirname(__file__), "weights")
#     state_path = os.path.join(weight_dir, "monet.pt")

#     templates = [
#         "This is dermatoscopy of {}",
#         "This is dermoscopy of {}",
#     ]
#     ref_prompts = [
#         "This is dermatoscopy",
#         "This is dermoscopy",
#     ]

#     def __init__(self, device: torch.device):
#         super().__init__(self.backbone, device)
#         self.preprocess = T.Compose(
#             [
#                 T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
#                 T.CenterCrop(224),
#                 lambda x: x.convert("RGB"),
#                 T.ToTensor(),
#                 T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )

#         state = torch.load(self.state_path, map_location=device)
#         self.model.load_state_dict(state)

#     @torch.no_grad()
#     def predict(self, dataset):
#         self.eval()

#         dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

#         attributes = dataset.claims
#         attributes = list(map(str.lower, attributes))
#         attribute_prompts = [t.format(a) for a in attributes for t in self.templates]

#         attribute_features = self.encode_text(attribute_prompts)
#         ref_features = self.encode_text(self.ref_prompts)

#         results = np.zeros((len(dataset), len(attributes)))
#         start = 0
#         for data in tqdm(dataloader):
#             image, _ = data

#             image = image.to(self.device)

#             attribute_output = self(image, text_features=attribute_features)
#             ref_output = self(image, text_features=ref_features)

#             attribute_logits = attribute_output["logits"]
#             ref_logits = ref_output["logits"]

#             attribute_logits = attribute_logits.view(
#                 -1, len(attributes), len(self.templates)
#             )
#             ref_logits = ref_logits.unsqueeze(1).expand(-1, len(attributes), -1)
#             attribute_probs = torch.sigmoid(attribute_logits - ref_logits)
#             attribute_probs = torch.amax(attribute_probs, dim=-1)

#             end = start + len(image)
#             results[start:end] = attribute_probs.cpu().numpy()
#             start = end

#         return results
