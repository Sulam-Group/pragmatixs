from configs.utils import Config, register_config


@register_config(name="cub")
class CUBClaimConfig(Config):
    def __init__(self):
        super().__init__()
        self.data.dataset = "cub"
        self.data.classifier = "open_clip:ViT-L-14"
        
        self.data.explanation_length = 12

        self.speaker.beta = 0.6
        self.speaker.alpha = 0.0 # [0.0, 0.2]
        self.speaker.k = 4

        self.speaker.width = 256
        self.speaker.heads = 4
        self.speaker.layers = 12
        self.speaker.n_queries = None
        self.speaker.attn_pooler_heads = 4

        self.speaker.lr = 1e-04
        self.speaker.wd = 1e-02

        # self.listener.type = "claim"
        self.listener.type = "topic"
        self.listener.prior = [0, 0, 1 / 3, 1 / 3, 1 / 3, 0]
        self.listener.temperature_scale = 1.0 #[1.0, 2.0, 4.0, 8.0]
        self.listener.gamma = 0.4
        self.listener.k = 8

        self.listener.width = 256
        self.listener.heads = 4
        self.listener.layers = 12

        self.listener.lr = 1e-04
        self.listener.wd = 1e-02
