import os

from configs.utils import Config, register_config


@register_config(name="cub_claim")
class CUBClaimConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = os.path.basename(__file__).replace(".py", "")

        data = self.data
        data.dataset = "cub"
        data.classifier = "CLIP:ViT-L/14"
        data.context_length = 12
        data.listener_type = "claim"
