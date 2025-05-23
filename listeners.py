import os
from abc import abstractmethod
from collections.abc import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
from open_clip.model import CLIPTextCfg, _build_text_tower

from configs import Config
from configs import Constants as C


class Listener(nn.Module):
    def __init__(self, config: Config, device=C.device):
        super().__init__()
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        config: Config,
        n_classes: int,
        claims: Iterable[str],
        workdir=C.workdir,
        device=C.device,
    ):
        listener = cls(config, n_classes, claims, device=device)
        state_path = config.state_path(workdir=workdir)
        state = torch.load(state_path, map_location=device)

        listener.load_state_dict(state["listener"])
        listener.eval()
        return listener

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def listen(self):
        pass


listeners: Mapping[str, Listener] = {}


def register_listener(name):
    def _register(cls: Listener) -> Listener:
        listeners[name] = cls
        return cls

    return _register


def get_listener(name: str) -> Listener:
    return listeners[name]


@register_listener("claim")
class ClaimListener(Listener):
    def __init__(
        self,
        config: Config,
        n_classes: int,
        claims: Iterable[str],
        workdir=C.workdir,
        device=C.device,
    ):
        super().__init__(config, device=device)
        self.context_length = context_length = config.data.explanation_length + 1
        self.gamma = config.listener.gamma
        self.claims = claims

        self.speaker_vocab_size = speaker_vocab_size = len(claims) + 3
        self.speaker_bos_token_id = speaker_vocab_size - 3
        self.speaker_eos_token_id = speaker_vocab_size - 2
        self.speaker_pad_token_id = speaker_vocab_size - 1

        self.vocab_size = vocab_size = 2 * len(claims) + 1
        self.pad_token_id = vocab_size - 1

        width = config.listener.width
        heads = config.listener.heads
        layers = config.listener.layers

        text_cfg = CLIPTextCfg(
            context_length=context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            embed_cls=True,
            no_causal_mask=True,
            pad_id=self.pad_token_id,
        )
        self.text = _build_text_tower(n_classes, text_cfg=text_cfg)
        self.text.build_cls_mask = self.build_cls_mask

        self.init_parameters()

        self.to(device)

    def init_parameters(self):
        nn.init.zeros_(self.text.text_projection)

    def build_cls_mask(self, claims, cast_dtype: torch.dtype):
        cls_mask = claims != self.pad_token_id
        cls_mask = torch.cat(
            [
                cls_mask,
                torch.ones(
                    (cls_mask.size(0), 1), device=cls_mask.device, dtype=torch.bool
                ),
            ],
            dim=-1,
        )
        additive_mask = torch.empty(
            cls_mask.shape, dtype=cast_dtype, device=cls_mask.device
        )
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = additive_mask.unsqueeze(1).expand(-1, cls_mask.size(1), -1)
        return torch.repeat_interleave(additive_mask, self.text.heads, dim=0)

    def prepare_claims_for_listener(self, explanation):
        claims = explanation[..., 0]
        claims_cls = explanation[..., 1]

        _claims = torch.clone(claims)
        _claims[_claims == self.speaker_bos_token_id] = self.pad_token_id
        _claims[_claims == self.speaker_eos_token_id] = self.pad_token_id
        _claims[_claims == self.speaker_pad_token_id] = self.pad_token_id
        _claims = _claims + claims_cls
        return _claims

    def forward(self, explanation):
        claims = self.prepare_claims_for_listener(explanation)
        return self.text(claims)

    def consistency(self, image_attribute, explanation):
        claims = explanation[..., 0]
        claims_cls = explanation[..., 1]

        _image_attribute = torch.cat(
            [
                image_attribute,
                torch.zeros(
                    image_attribute.size(0),
                    self.speaker_vocab_size - len(self.claims),
                    device=image_attribute.device,
                ),
            ],
            dim=-1,
        )
        target_cls = torch.gather(_image_attribute, -1, claims)

        claims_mask = claims < len(self.claims)
        tp = claims_mask * (claims_cls == 1) * (target_cls == 1)
        tn = claims_mask * (claims_cls == 0) * (target_cls == 0)
        consistency = torch.sum(tp, dim=-1) + self.gamma * torch.sum(tn, dim=-1)
        return consistency / torch.sum(claims_mask, dim=-1)

    def listen(self, image_attribute, explanation):
        consistency = self.consistency(image_attribute, explanation)
        action = self.forward(explanation)
        return consistency, action


@register_listener("topic")
class TopicListener(ClaimListener):
    def __init__(
        self,
        config: Config,
        n_classes: int,
        claims: Iterable[str],
        workdir=C.workdir,
        device=C.device,
    ):
        super().__init__(config, n_classes, claims, workdir=workdir, device=device)
        self.data_name = config.data.dataset
        self.prior = torch.tensor(config.listener.prior).to(device)
        self.temperature_scale = config.listener.temperature_scale

        claim_topic = self.load_attribute_topic(workdir=workdir)
        claim_topic = claim_topic + 3 * [-1]
        self.claim_topic = torch.tensor(claim_topic).to(device)
        self.number_of_topics = self.claim_topic.max()

        self._super_forward = super().forward

    def load_attribute_topic(self, workdir=C.workdir):
        if self.data_name == "cub":
            attribute_dir = os.path.join(workdir, "data", "CUB", "attributes")
            with open(os.path.join(attribute_dir, "attribute_topic.txt"), "r") as f:
                lines = f.readlines()
                lines = [line.strip().split() for line in lines]
                attribute_topic = [int(idx) for _, idx in lines]
        elif self.data_name == "chexpert":
            attribute_topic = [1] * 12
        elif self.data_name == "chexpert_augmented":
            attribute_topic = [1] * 12 + [2] * 12
        elif self.data_name == "chexpert_augmentedv2":
            attribute_topic = [1] * 12 + [2] * 10
        return attribute_topic

    def get_explanation_topic(self, explanation):
        claims = explanation[..., 0]

        claim_topic = self.claim_topic.unsqueeze(0).expand(explanation.size(0), -1)
        explanation_topic = torch.gather(claim_topic, -1, claims)

        topic_mask = (
            torch.arange(self.number_of_topics, device=explanation_topic.device) + 1
        )
        explanation_topic_mask = explanation_topic[..., None] == topic_mask
        explanation_topic = torch.sum(explanation_topic_mask, dim=-2)
        norm = torch.sum(explanation_topic, dim=-1, keepdim=True)
        return explanation_topic / norm

    def forward(self, explanation):
        action = self._super_forward(explanation)  # batch_size x n_classes

        prior = self.prior  # prior distribution of topics (batch_size, n_topics)
        explanation_topic = self.get_explanation_topic(
            explanation
        )  # empirical distribution of topics (batch_size, n_topics)
        kl = torch.sum(
            explanation_topic
            * torch.log((explanation_topic + 1e-08) / (prior + 1e-08)),
            dim=-1,
        )
        temperature = self.temperature_scale * kl + 1
        return action / temperature[:, None]
