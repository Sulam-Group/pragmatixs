from abc import abstractmethod
from collections.abc import Mapping

import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from health_multimodal.image.data.transforms import (
    create_chest_xray_transform_for_inference,
)
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
from health_multimodal.text.utils import BertEncoderType, get_bert_inference
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.utils import Config
from configs.utils import Constants as C


class ImageClassifier(nn.Module):
    def __init__(self, config: Config, device=C.device):
        super().__init__()
        self.config = config

        self.embed_dim: int = None
        self.width: int = None
        self.preprocess: T.Compose = None
        self.device = device

        self.to(device)

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, config: Config, workdir=C.workdir, device=C.device
    ) -> "ImageClassifier":
        pass

    @abstractmethod
    def encode_image(self, image):
        pass

    @abstractmethod
    def predict(self, dataset: Dataset) -> pd.DataFrame:
        pass


classifiers: Mapping[str, ImageClassifier] = {}


def register_classifier(name: str):
    def register(cls: ImageClassifier):
        if name in classifiers:
            raise ValueError(f"Classifier {name} is already registered")
        classifiers[name] = cls
        return cls

    return register


def get_classifier(
    config: Config, from_pretrained=True, workdir=C.workdir, device=C.device
) -> ImageClassifier:
    classifier_name = config.data.classifier.split(":")[0].lower()
    Classifier = classifiers[classifier_name]
    if from_pretrained:
        return Classifier.from_pretrained(config, workdir=workdir, device=device)
    raise NotImplementedError


@register_classifier(name="open_clip")
class OpenClipClassifier(ImageClassifier):
    OPENCLIP_WEIGHTS = {
        "ViT-B-32": "laion2b_s34b_b79k",
        "ViT-L-14": "laion2b_s32b_b82k",
    }

    def __init__(self, config: Config, device=C.device):
        super().__init__(config, device=device)
        backbone = config.data.classifier.split(":")[1]

        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            backbone,
            pretrained=self.OPENCLIP_WEIGHTS[backbone],
            precision="fp16",
            device=device,
        )
        self.tokenize = open_clip.get_tokenizer(backbone)
        self.model.visual.output_tokens = True

        self.output_tokens = config.speaker.use_tokens
        self.embed_dim = self.model.visual.output_dim
        self.width = self.model.visual.proj.shape[0]

    @classmethod
    def from_pretrained(cls, config: Config, workdir=C.workdir, device=C.device):
        model = cls(config, device=device)
        model.eval()
        return model

    def encode_text(self, text):
        text = self.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text)
        return text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    def encode_image(self, image):
        image_features, image_tokens = self.model.visual(image.half())
        image_features /= torch.linalg.norm(image_features, dim=-1, keepdim=True)
        return image_features, image_tokens

    def forward(self, image, text=None, text_features=None):
        assert text is not None or text_features is not None

        image_features, image_tokens = self.encode_image(image)

        if text_features is None:
            text_features = self.encode_text(text)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return {
            "image_features": image_features.float(),
            "image_tokens": image_tokens.float(),
            "text_features": text_features.float(),
            "logits": logits_per_image.float(),
        }

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        classes = dataset.classes
        class_prompts = [f"A photo of a {class_name}" for class_name in classes]
        text_features = self.encode_text(class_prompts)

        results = {"label": [], "prediction": [], "logits": []}
        for data in tqdm(dataloader):
            image, label = data[0], data[1]

            image = image.to(self.device)

            output = self(image, text_features=text_features)
            logits = output["logits"]
            prediction = torch.argmax(logits, dim=-1).cpu()

            results["label"].extend(label.squeeze().tolist())
            results["prediction"].extend(prediction.squeeze().tolist())
            results["logits"].extend(logits.squeeze().cpu().tolist())

        return pd.DataFrame(results)


@register_classifier(name="biomedvlp")
class BiomedVLP(ImageClassifier):
    def __init__(self, config: Config, device=C.device):
        super().__init__(config, device=device)

        self.preprocess = create_chest_xray_transform_for_inference(
            resize=512, center_crop_size=512
        )
        self.image_encoder = get_biovil_t_image_encoder()
        self.text_encoder = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        self.device = device
        self.task = config.data.task
        self.embed_dim = 128
        self.width = 128

        self.to(device)

    @classmethod
    def from_pretrained(cls, config: Config, workdir=C.workdir, device=C.device):
        model = cls(config, device=device)
        model.eval()
        return model

    def encode_text(self, text):
        text_features = self.text_encoder.get_embeddings_from_prompt(text)
        text_features = text_features.to(self.device)
        text_features = text_features / torch.linalg.norm(
            text_features, dim=-1, keepdim=True
        )
        return text_features

    def encode_image(self, image):
        results = self.image_encoder(image)
        image_tokens = results.projected_patch_embeddings
        image_features = results.projected_global_embedding
        image_tokens = image_tokens.permute(0, 2, 3, 1).reshape(
            image_tokens.shape[0], -1, self.width
        )
        image_features = image_features / torch.linalg.norm(
            image_features, dim=-1, keepdim=True
        )
        return image_features, image_tokens

    def forward(self, image_path, text=None, text_features=None):
        assert text is not None or text_features is not None

        image_features, image_tokens = self.encode_image(image_path)

        if text_features is None:
            text_features = self.encode_text(text)

        logits_per_image = image_features @ text_features.t()

        return {
            "image_features": image_features.float(),
            "image_tokens": image_tokens.float(),
            "text_features": text_features.float(),
            "logits": logits_per_image.float(),
        }

    @torch.no_grad()
    def predict(self, dataset):
        self.eval()

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        class_prompts = [f"No signs of {self.task}", f"Findings suggesting {self.task}"]
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
