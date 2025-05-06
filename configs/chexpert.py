from configs.utils import Config, register_config


@register_config(name="chexpert")
class CheXpertClaimConfig(Config):
    def __init__(self):
        super().__init__()
        self.data.dataset = "chexpert"
        self.data.classifier = "BiomedVLP"
        self.data.explanation_length = [4, 8, 12]
        self.data.task = "Lung Opacity"

        self.speaker.beta = 0.6
        self.speaker.alpha = [0.0, 0.2]
        self.speaker.k = 4

        self.speaker.width = 256
        self.speaker.heads = 4
        self.speaker.layers = 12
        self.speaker.n_queries = None
        self.speaker.attn_pooler_heads = 4

        self.listener.type = "claim"
        self.listener.prior = [1.0, 0.0]
        self.listener.temperature_scale = 1.0
        self.listener.gamma = [0.0, 0.2, 0.4]
        self.listener.k = 8

        self.listener.width = 256
        self.listener.heads = 4
        self.listener.layers = 12

        self.training.iterations = 20
        self.training.batch_size = 16
        self.training.min_lr = 1e-05
        self.training.max_lr = 1e-04
        self.training.wd = 1e-02
        self.training.max_grad_norm = 1.0
