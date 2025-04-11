from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from open_clip.coca_model import MultimodalCfg, _build_text_decoder_tower
from open_clip.model import CLIPTextCfg, _build_text_tower
from open_clip.transformer import AttentionalPooler, LayerNorm

from classifiers import ImageClassifier
from configs import Config
from configs import Constants as C


class ClaimSpeaker(nn.Module):
    def __init__(
        self,
        config: Config,
        classifier: ImageClassifier,
        claims: Iterable[str],
        device=C.device,
    ):
        super().__init__()
        self.context_length = context_length = config.data.explanation_length + 1 # 12+1
        self.claims = claims

        vocab_size = len(claims) + 3
        width = config.speaker.width # 256
        heads = config.speaker.heads # 4
        unimodal_layers = multimodal_layers = config.speaker.layers // 2
        print(
            f"Number of layers: {unimodal_layers} unimodal layers,"
            f" {multimodal_layers} multimodal layers"
        )
        n_queries = config.data.explanation_length // 2
        attn_pooler_heads = config.speaker.attn_pooler_heads

        self.bos_token_id = vocab_size - 3
        self.eos_token_id = vocab_size - 2
        self.pad_token_id = vocab_size - 1

        text_cfg = CLIPTextCfg(
            context_length=context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=unimodal_layers,
        )
        self.text = _build_text_tower(width, text_cfg=text_cfg)

        multimodal_cfg = MultimodalCfg(
            context_length=context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=multimodal_layers,
            n_queries=n_queries,
        )
        self.decoder = _build_text_decoder_tower(
            vocab_size, multimodal_cfg=multimodal_cfg
        )

        self.attn_pool_gen = AttentionalPooler(
            width, classifier.width, n_head=attn_pooler_heads, n_queries=n_queries # classifier.width = 1024
        )
        self.ln_attn_pool_gen = LayerNorm(width)

        self._attn_pool_cls = AttentionalPooler(
            int(np.ceil(vocab_size / attn_pooler_heads)) * attn_pooler_heads,
            classifier.width,
            n_head=attn_pooler_heads,
            n_queries=1,
        )
        self.attn_pool_cls = lambda x: torch.squeeze(self._attn_pool_cls(x), 1)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07), requires_grad=False
        )

        self.init_parameters()

        self.to(device)

    def init_parameters(self):
        nn.init.zeros_(self.decoder.text_projection)
        nn.init.zeros_(self._attn_pool_cls.attn.q_proj_weight)
        nn.init.zeros_(self._attn_pool_cls.attn.k_proj_weight)
        nn.init.zeros_(self._attn_pool_cls.attn.v_proj_weight)

    @classmethod
    def from_pretrained(
        cls,
        config: Config,
        classifier: ImageClassifier,
        claims: Iterable[str],
        workdir=C.workdir,
        device=C.device,
    ):
        speaker = cls(config, classifier, claims, device=device)
        state_path = config.state_path(workdir=workdir)
        state = torch.load(state_path, map_location=device)
        speaker.load_state_dict(state["speaker"])
        speaker.eval()
        return speaker

    def encode_text(self, text):
        seq_len = text.size(1)

        attn_mask = self.text.attn_mask[:seq_len, :seq_len]

        text = self.text.token_embedding(text)
        text = text + self.text.positional_embedding[:seq_len]
        text = self.text.transformer(text, attn_mask=attn_mask)
        return self.text.ln_final(text)

    def forward(self, image_tokens, claims):
        image_tokens = self.attn_pool_gen(image_tokens)
        image_tokens = self.ln_attn_pool_gen(image_tokens)
        explanation_tokens = self.encode_text(claims)
        return self.decoder(image_tokens, explanation_tokens)

    def explanation_logp(self, image_tokens, explanation):
        claims = explanation[..., 0]
        claims_cls = explanation[..., 1]

        binary_logits = self.logit_scale.exp() * self.attn_pool_cls(image_tokens)
        binary_logits = torch.gather(binary_logits, -1, claims)
        binary_logp = -torch.nn.functional.binary_cross_entropy_with_logits(
            binary_logits, claims_cls.float(), reduction="none"
        )
        binary_logp[claims >= len(self.claims)] = 0

        claim_logits = self(image_tokens, claims)
        shift_logits = claim_logits[..., :-1, :].contiguous()
        shift_labels = claims[..., 1:].contiguous()

        claim_logp = -torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=self.pad_token_id,
        )
        claim_logp = claim_logp.view_as(shift_labels)

        explanation_logp = binary_logp[:, 1:] + claim_logp
        return torch.sum(explanation_logp, dim=-1)

    @torch.no_grad()
    def explain(self, image_tokens):
        m = image_tokens.size(0)

        binary_logits = self.logit_scale.exp() * self.attn_pool_cls(image_tokens)
        binary_probs = torch.sigmoid(binary_logits)
        binary_labels = torch.bernoulli(binary_probs)

        claims = torch.tensor([[self.bos_token_id]] * m, device=image_tokens.device)
        claims_cls = torch.zeros(m, 1, device=image_tokens.device)

        finished = torch.zeros(m, dtype=torch.bool, device=image_tokens.device)
        while claims.size(1) < self.context_length:
            claim_logits = self(image_tokens, claims)
            next_claim_logits = claim_logits[:, -1]

            # set pad token to -inf to avoid generating it
            next_claim_logits[:, self.bos_token_id] = float("-inf")
            next_claim_logits[:, self.pad_token_id] = float("-inf")

            # # set eos token to -inf to avoid generating empty sequence
            if claims.size(1) == 1:
                next_claim_logits[:, self.eos_token_id] = float("-inf")

            # set previous tokens to -inf to avoid repeating claims
            next_claim_logits = next_claim_logits.scatter(-1, claims, float("-inf"))

            # sample next claim
            next_claim_probs = torch.softmax(next_claim_logits, dim=-1)
            next_claim = torch.multinomial(next_claim_probs, 1)
            next_claim_cls = torch.take_along_dim(binary_labels, next_claim, -1)

            next_claim_cls[next_claim == self.eos_token_id] = 0

            finished += claims[:, -1] == self.eos_token_id
            next_claim[finished] = self.pad_token_id
            next_claim_cls[finished] = 0

            claims = torch.cat([claims, next_claim], dim=-1)
            claims_cls = torch.cat([claims_cls, next_claim_cls], dim=-1)

        explanation = torch.stack([claims, claims_cls], dim=-1).long()
        explanation_logp = self.explanation_logp(image_tokens, explanation)
        return explanation, explanation_logp
