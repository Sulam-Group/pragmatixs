import os

from configs.utils import Config, register_config


@register_config(name="cub_distribution")
class CUBClaimConfig(Config):
    config = {
        "data": {
            "dataset": "cub",
            "classifier": "CLIP:ViT-L/14",
            "context_length": 12,
            "listener_type": "distribution",
        },
        "speaker": {"beta": 0.4, "alpha": 0.1, "k": 8},
        "listener": {"temperature_scale": 1.0},
    }

    def __init__(self):
        super().__init__(**self.config)
        self.name = os.path.basename(__file__).replace(".py", "")
        # data = self.data
        # data.dataset = "cub"
        # data.classifier = "CLIP:ViT-L/14"
        # data.context_length = 12
        # data.listener_type = "claim"

        # speaker = self.speaker
        # speaker.beta = 0.4
        # speaker.alpha = 0.0
        # speaker.k = 8
