import os

import clip
import torch
import torch.nn as nn
from clip.model import CLIP

from configs import Constants as C


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


@torch.no_grad()
def get_dictionary(clip_backbone: str = None, workdir=C.workdir, device=C.device):
    data_dir = os.path.join(workdir, "data")
    imagenet_dir = os.path.join(data_dir, "ImageNet")

    model_clip, _ = clip.load(clip_backbone, device=device)
    model_clip.eval()

    attribute_dir = os.path.join(imagenet_dir, "attributes")
    with open(os.path.join(attribute_dir, "attributes.txt")) as f:
        lines = f.readlines()
        attributes = [line.strip() for line in lines]

    text = clip.tokenize(attributes).to(device)
    text_features = model_clip.encode_text(text)
    text_features = text_features.T
    return text_features / torch.linalg.norm(text_features, axis=0)


@torch.no_grad()
def answer_query(
    image: torch.Tensor = None,
    model_clip: CLIP = None,
    dictionary: torch.Tensor = None,
    concept_net: ConceptNet2 = None,
    concept_idx: torch.Tensor = None,
):
    image_features = model_clip.encode_image(image)
    image_features /= torch.linalg.norm(image_features, dim=-1, keepdim=True)
    image_features = image_features.repeat(dictionary.size(1), 1, 1).permute(1, 0, 2)

    dictionary_extended = dictionary.T.repeat(image_features.size(0), 1, 1)
    input_features = torch.cat((image_features, dictionary_extended), dim=2)
    input_features = torch.flatten(input_features, 0, 1)

    output = concept_net(input_features.float()).squeeze().view(-1, dictionary.size(1))
    if concept_idx is not None:
        output = output[..., concept_idx]
    return (output > -0.4).float()
