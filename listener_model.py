import os
from typing import Iterable, Mapping

import torch
import torch.nn as nn

from configs import Config
from transformer_model import _build_transformer


class ClaimListener(nn.Module):
    encoder_config = {
        "width": 512,
        "context_length": None,
        "heads": 4,
        "layers": 4,
        "attn_pooler_heads": 4,
    }

    def __init__(
        self,
        config: Config,
        n_classes: int,
        claims: Iterable[str],
        device: torch.device,
    ):
        super().__init__()
        self.encoder_config["context_length"] = config.data.context_length
        self.n_classes = n_classes
        self.claims = claims

        vocab_size = len(claims) + 3
        self.cls_token_id = vocab_size - 3
        self.eos_token_id = vocab_size - 2
        self.pad_token_id = vocab_size - 1

        embed_dim = self.encoder_config["width"]
        self.claim_embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_config = self.encoder_config
        encoder_config["vocab_size"] = vocab_size
        encoder_config["width"] = embed_dim
        self.transformer = _build_transformer(n_classes, encoder_config)

        self.init_parameters()

        self.device = device
        self.to(device)

    def init_parameters(self):
        nn.init.xavier_normal_(self.claim_embedding.weight)
        nn.init.zeros_(self.transformer.text_projection)

    @classmethod
    def from_pretrained(
        cls,
        config: Config,
        n_classes: int,
        claims: Iterable[str],
        device: torch.device,
        workdir: str,
    ):
        listener = cls(config, n_classes, claims, device)
        state_path = os.path.join(workdir, "weights", f"{config.run_name()}.pt")
        state = torch.load(state_path, map_location=device)
        listener.load_state_dict(state["listener"])
        listener.eval()
        return listener

    def build_cls_mask(self, explanation):
        cls_mask = (explanation != self.pad_token_id).unsqueeze(1)
        cls_mask = cls_mask.unsqueeze(1).expand(
            -1, self.encoder_config["heads"], cls_mask.size(-1), -1
        )
        return cls_mask.flatten(0, 1)

    def forward(self, image, explanation):
        attention_mask = self.build_cls_mask(explanation)
        explanation_embedding = self.claim_embedding(explanation)

        if image is None:
            context_length = self.encoder_config["context_length"]
            embed_dim = self.encoder_config["width"]
            image_features = torch.zeros(
                explanation.size(0),
                context_length,
                embed_dim,
                device=explanation.device,
            )

        logits = self.transformer(
            image_features, explanation_embedding, attention_mask=attention_mask
        )
        return logits[:, 0]

    def listen(self, image_attribute, explanation):
        b = explanation.size(1)

        image_attribute = image_attribute.unsqueeze(1).expand(-1, b, -1)
        image_attribute = image_attribute.flatten(0, 1)
        explanation = explanation.flatten(0, 1)

        explanation_attribute = torch.zeros(
            image_attribute.size(0), image_attribute.size(1) + 3
        )
        explanation_attribute = explanation_attribute.scatter(
            1, explanation.to(explanation_attribute.device), 1
        )
        explanation_attribute = explanation_attribute[:, :-3]

        intersection = torch.sum(
            explanation_attribute * (image_attribute > -1) * image_attribute, dim=-1
        )
        consistency = intersection / (torch.sum(explanation_attribute, dim=-1) + 1e-06)
        consistency = consistency.reshape(-1, b)

        action = self(None, explanation)
        action = action.unsqueeze(1).reshape(-1, b, self.n_classes)
        return consistency, action


class CUBTopicListener(ClaimListener):
    attribute_dir = os.path.join(os.path.dirname(__file__), "data", "CUB", "attributes")

    def __init__(
        self,
        config: Config,
        n_classes: int,
        claims: Iterable[str],
        device: torch.device,
    ):
        super().__init__(config, n_classes, claims, device)
        self.ignore_topics = config.ignore_topics

        with open(os.path.join(self.attribute_dir, "attribute_topic.txt"), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            attribute_to_topic = {attribute: int(idx) for attribute, idx in lines}

        self.claim_topic = torch.tensor(
            [attribute_to_topic[claim] for claim in claims] + 3 * [-1],
            device=self.device,
        )
        self.super_forward = super().forward
        self.super_listen = super().listen

    def forward(self, image, explanation):
        claim_topic = self.claim_topic.unsqueeze(0).expand(explanation.size(0), -1)
        explanation_topic = torch.gather(claim_topic, -1, explanation)
        mask = torch.isin(
            explanation_topic,
            torch.tensor(self.ignore_topics, device=explanation_topic.device),
        )
        masked_explanation = explanation.masked_fill(mask, self.pad_token_id)
        return self.super_forward(image, masked_explanation)


class CUBDistributionListener(ClaimListener):
    number_of_topics = 6
    attribute_dir = os.path.join(os.path.dirname(__file__), "data", "CUB", "attributes")

    def __init__(
        self,
        config: Config,
        n_classes: int,
        claims: Iterable[str],
        device: torch.device,
    ):
        super().__init__(config, n_classes, claims, device)
        self.topic_prior = torch.tensor(config.prior, device=device)
        self.temperature_scale = config.listener.temperature_scale

        with open(os.path.join(self.attribute_dir, "attribute_topic.txt"), "r") as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
            attribute_to_topic = {attribute: int(idx) for attribute, idx in lines}

        claim_topic = [attribute_to_topic[claim] for claim in claims]
        claim_topic = claim_topic + 3 * [-1]
        self.claim_topic = torch.tensor(claim_topic, device=self.device)

        self.super_forward = super().forward

    def forward(self, image, explanation):
        action = self.super_forward(image, explanation)

        claim_topic = self.claim_topic.unsqueeze(0).expand(explanation.size(0), -1)
        explanation_topic = torch.gather(claim_topic, -1, explanation)

        topic_mask = (
            torch.arange(self.number_of_topics, device=explanation_topic.device) + 1
        )
        explanation_topic_mask = explanation_topic[..., None] == topic_mask
        explanation_topic_distribution = torch.sum(explanation_topic_mask, dim=-2)
        explanation_topic_distribution = explanation_topic_distribution / torch.sum(
            explanation_topic_distribution, dim=-1, keepdim=True
        )

        prior = self.topic_prior
        topic_kl = torch.sum(
            prior * torch.log(prior / (explanation_topic_distribution + 1e-08)), dim=-1
        )
        temperature = self.temperature_scale * (topic_kl + 1)
        return action / temperature[:, None]


LISTENERS: Mapping[str, ClaimListener] = {
    "claim": ClaimListener,
    "topic": CUBTopicListener,
    "distribution": CUBDistributionListener,
}
