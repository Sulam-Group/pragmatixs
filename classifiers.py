import base64
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
from openai import OpenAI
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.utils import Config
from configs.utils import Constants as c


class ImageClassifier(nn.Module):
    def __init__(self, config: Config, device=c.device):
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
        cls, config: Config, workdir=c.workdir, device=c.device
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
    config: Config, from_pretrained=True, workdir=c.workdir, device=c.device
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

    def __init__(self, config: Config, device=c.device):
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
    def from_pretrained(cls, config: Config, workdir=c.workdir, device=c.device):
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

        results = {"label": [], "prediction": []}
        for data in tqdm(dataloader):
            image, label = data[0], data[1]

            image = image.to(self.device)

            output = self(image, text_features=text_features)
            logits = output["logits"]
            prediction = torch.argmax(logits, dim=-1).cpu()

            results["label"].extend(label.squeeze().tolist())
            results["prediction"].extend(prediction.squeeze().tolist())

        return pd.DataFrame(results)


@register_classifier(name="ham_biomedclip")
class HAMBiomedCLIP(ImageClassifier):
    def __init__(self, config: Config, device=c.device):
        super().__init__(config, device)

        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            device=device,
        )

        self.embed_dim = 512
        n_classes = 7
        self.proj = nn.Linear(self.embed_dim, n_classes)

        nn.init.xavier_normal_(self.proj.weight)

        self.to(device)
        self.device = device

    @classmethod
    def from_pretrained(cls, config: Config, workdir=c.workdir, device=c.device):
        model = cls(config, device=device)
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


class MONET(nn.Module):
    templates = [
        "This is dermatoscopy of {}",
        "This is dermoscopy of {}",
    ]
    ref_templates = [
        "This is dermatoscopy",
        "This is dermoscopy",
    ]

    def __init__(self, workdir=c.workdir, device=c.device):
        super().__init__()
        backbone = "ViT-L/14"

        self.model, _ = clip.load(backbone, device=device)
        self.tokenize = clip.tokenize

        self.preprocess = T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                lambda x: x.convert("RGB"),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.embed_dim = self.model.visual.output_dim

        state_path = os.path.join(workdir, "weights", "monet.pt")
        state = torch.load(state_path, map_location=device)
        self.model.load_state_dict(state)

        self.to(device)
        self.device = device

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

        attributes = dataset.attributes
        attributes = list(map(str.lower, attributes))
        attribute_prompts = [t.format(a) for a in attributes for t in self.templates]

        attribute_features = self.encode_text(attribute_prompts)
        ref_features = self.encode_text(self.ref_templates)

        results = np.zeros((len(dataset), len(attributes)))
        start = 0
        for data in tqdm(dataloader):
            image, _ = data

            image = image.to(self.device)

            attribute_output = self(image, text_features=attribute_features)
            ref_output = self(image, text_features=ref_features)

            attribute_logits = attribute_output["logits"]
            ref_logits = ref_output["logits"]

            attribute_logits = attribute_logits.view(
                -1, len(attributes), len(self.templates)
            )
            ref_logits = ref_logits.unsqueeze(1).expand(-1, len(attributes), -1)
            attribute_probs = torch.sigmoid(attribute_logits - ref_logits)
            attribute_probs = torch.amax(attribute_probs, dim=-1)

            end = start + len(image)
            results[start:end] = attribute_probs.cpu().numpy()
            start = end

        return results


class GPTClassifier:
    # prompt = (
    #     "This is a dermatoscopy image of a skin lesion. "
    #     + "Please answer to the best of your ability whether the following attributes"
    #     " are present in the image: "
    #     + {}
    #     + ". "
    #     + "Write your answer as a list with a number between 0 and 1 that represents"
    #     " the likelihood of the presence of the attribute. "
    #     + "For example, if you think the attribute is definitely not present, you can"
    #     " write 0. If you think the attribute is definitely present, you can write 1."
    #     " If you are unsure, you can write a number between 0 and 1. "
    #     + "If you prefer to abstain from answering for a particular attribute, you can"
    #     " write -1."
    # )

    def __init__(self, model):
        self.model = model
        self.client = OpenAI()

    def predict(self, prompt, dataset):
        class AttributeAnnotation(BaseModel):
            attribute: str
            label: float

        class ImageAnnotation(BaseModel):
            annotations: list[AttributeAnnotation]

        def tobase64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # attributes = dataset.attributes
        # attributes = list(map(str.lower, attributes))
        # prompt = prompt.format(", ".join(attributes))

        responses = {}
        for i, (path, _) in enumerate(tqdm(dataset.samples)):
            res = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{tobase64(path)}"
                                },
                            },
                        ],
                    },
                ],
                response_format=ImageAnnotation,
            )

            for choice_idx, choice in enumerate(res.choices):
                if path not in responses:
                    responses[path] = {}

                message = choice.message
                parsed = message.parsed

                if parsed:
                    responses[path][choice_idx] = parsed.model_dump()
                else:
                    responses[path][choice_idx] = message.refusal
