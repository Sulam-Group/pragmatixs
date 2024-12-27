import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from classifiers import ImageClassifier
from configs import Config
from transformer_model import _build_transformer


class ClaimSpeaker(nn.Module):
    multimodal_config = {
        "context_length": None,
        "heads": 4,
        "layers": 4,
        "attn_pooler_heads": 4,
    }

    @classmethod
    def from_pretrained(
        cls,
        config: Config,
        classifier: ImageClassifier,
        n_classes: int,
        claims: Iterable[str],
        device: torch.device,
        workdir: str,
    ):
        speaker = cls(config, classifier, n_classes, claims, device)
        state_path = os.path.join(workdir, "weights", f"{config.run_name()}.pt")
        state = torch.load(state_path, map_location=device)
        speaker.load_state_dict(state["speaker"])
        speaker.eval()
        return speaker

    def __init__(
        self,
        config: Config,
        classifier: ImageClassifier,
        n_classes: int,
        claims: Iterable[str],
        device: torch.device,
    ):
        super().__init__()
        self.multimodal_config["context_length"] = config.data.context_length
        self.n_classes = n_classes
        self.claims = claims

        self.embed_dim = embed_dim = classifier.embed_dim
        self.vocab_size = vocab_size = len(claims) + 3
        self.bos_token_id = vocab_size - 3
        self.eos_token_id = vocab_size - 2
        self.pad_token_id = vocab_size - 1

        self.claim_embedding = nn.Embedding(vocab_size, embed_dim)
        self.class_embedding = nn.Embedding(n_classes, embed_dim)

        multimodal_config = self.multimodal_config
        multimodal_config["vocab_size"] = vocab_size
        multimodal_config["width"] = embed_dim
        self.transformer = _build_transformer(vocab_size, multimodal_config)

        self.init_parameters()

        self.device = device
        self.to(device)

    def init_parameters(self):
        nn.init.xavier_normal_(self.claim_embedding.weight)
        nn.init.xavier_normal_(self.class_embedding.weight)
        nn.init.zeros_(self.transformer.text_projection)

    def forward(self, image_features, prediction, claim):
        claim_embedding = self.claim_embedding(claim)

        # class_embedding = self.class_embedding(prediction)
        # class_embedding = class_embedding.unsqueeze(1).expand(
        #     -1, claim_embedding.size(1), -1
        # )

        # claim_embedding = claim_embedding + class_embedding
        return self.transformer(image_features, claim_embedding)

    def explanation_logp(self, image_features, prediction, explanation):
        logits = self(image_features, prediction, explanation)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = explanation[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=self.pad_token_id,  # ignore padding tokens
        )
        loss = loss.view_as(shift_labels)
        return -torch.sum(loss, dim=-1)

    @torch.no_grad()
    def explain(self, image_features, prediction, b=1):
        m = image_features.size(0)

        image_features = image_features.unsqueeze(1).expand(-1, b, -1, -1)
        image_features = image_features.flatten(0, 1)

        prediction = prediction.unsqueeze(1).expand(-1, b)
        prediction = prediction.flatten()

        explanation = torch.tensor([self.bos_token_id], device=self.device)
        explanation = explanation.unsqueeze(0).expand(image_features.size(0), -1)

        finished = torch.zeros(
            explanation.size(0), dtype=torch.bool, device=explanation.device
        )
        while explanation.size(1) < self.multimodal_config["context_length"]:
            # for _ in range(self.multimodal_config["context_length"] - 1):
            logits = self(image_features, prediction, explanation)
            next_claim_logits = logits[:, -1, :]

            # set pad token to -inf to avoid generating it
            next_claim_logits[:, self.pad_token_id] = float("-inf")

            # set eos token to -inf to avoid generating empty sequence
            if explanation.size(1) == 1:
                next_claim_logits[:, self.eos_token_id] = float("-inf")

            # set previous tokens to -inf to avoid repeating claims
            next_claim_logits = next_claim_logits.scatter(
                -1, explanation, float("-inf")
            )

            # sample next claim
            probs = torch.softmax(next_claim_logits, dim=-1)
            next_claim = torch.multinomial(probs, 1)

            # # nucleus sampling
            # sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
            # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # nucleus_p = 0.9
            # nucleus_mask = cumulative_probs <= nucleus_p
            # nucleus_mask[:, 0] = True  # always include one token
            # nucleus_mask[indices == self.eos_token_id] = True  # always include eos

            # sorted_logits = next_claim_logits.gather(-1, indices)
            # masked_logits = sorted_logits.masked_fill(~nucleus_mask, float("-inf"))
            # masked_probs = torch.softmax(masked_logits, dim=-1)

            # next_claim = torch.multinomial(masked_probs, 1)
            # next_claim = indices.gather(-1, next_claim)

            # if eos is reached, stop generating
            finished += explanation[:, -1] == self.eos_token_id
            next_claim[finished] = self.pad_token_id

            explanation = torch.cat([explanation, next_claim], dim=-1)

        # # set last claim of unfinished sequences to eos
        # explanation[~finished, -1] = self.eos_token_id
        # # set last claim of finished sequences to pad
        # explanation[finished, -1] = self.pad_token_id

        explanation_logp = self.explanation_logp(
            image_features, prediction, explanation
        )
        return explanation.reshape(m, b, -1), explanation_logp.reshape(m, b)
