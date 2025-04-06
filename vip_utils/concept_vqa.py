import torch.nn as nn


class ConceptNet2(nn.Module):
    def __init__(self, embed_dims=512):
        super().__init__()
        self.embed_dims = embed_dims
        self.input_dim = self.embed_dims * 2

        # Architecture
        self.layer1 = nn.Linear(self.input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

        self.head = nn.Linear(64, 1)

    def forward(self, x):
        #         img_emb = F.normalize(img_emb, p=2, dim=-1)
        #         txt_emb = F.normalize(txt_emb, p=2, dim=-1)

        # x = torch.hstack([img_emb, txt_emb])
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.relu(self.norm3(self.layer3(x)))
        x = self.relu(self.norm4(self.layer4(x)))
        return self.head(x).squeeze()
