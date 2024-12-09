import torch
import torch.nn as nn
from open_clip.coca_model import MultimodalCfg
from open_clip.transformer import LayerNorm, MultimodalTransformer
from torch.utils.checkpoint import checkpoint


class Transformer(MultimodalTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, image_embs, text_embs, attention_mask=None):
        seq_len = text_embs.shape[1]
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else self.attn_mask[:seq_len, :seq_len]
        )

        if not self.batch_first:
            image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
            text_embs = text_embs.permute(1, 0, 2)  # NLD -> LND

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, attention_mask)
                text_embs = checkpoint(
                    cross_attn, text_embs, image_embs, image_embs, None
                )
            else:
                text_embs = resblock(text_embs, attn_mask=attention_mask)
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        if not self.batch_first:
            text_embs = text_embs.permute(1, 0, 2)  # LND -> NLD

        out = self.ln_final(text_embs)
        if self.text_projection is not None:
            out = out @ self.text_projection

        return out


def _build_transformer(output_dim, multimodal_cfg):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg)
    act_layer = nn.GELU
    norm_layer = LayerNorm

    return Transformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=output_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
