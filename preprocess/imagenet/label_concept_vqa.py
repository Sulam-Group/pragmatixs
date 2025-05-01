import os
import sys
from collections.abc import Iterable

import clip
import numpy as np
import torch
from clip.model import CLIP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)
from datasets import ImageNet
from vip_utils.concept_vqa import ConceptNet2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dir = os.path.join(root_dir, "weights")
data_dir = os.path.join(root_dir, "data")

imagenet_dir = os.path.join(data_dir, "ImageNet")


@torch.no_grad()
def get_dictionary(attributes: Iterable[str], model_clip: CLIP, device=device):
    text = clip.tokenize(attributes).to(device)
    text_features = model_clip.encode_text(text)
    dictionary = text_features.T
    return dictionary / torch.linalg.norm(dictionary, axis=0)


@torch.no_grad()
def answer_query(
    dataset: Dataset,
    model_clip: CLIP,
    dictionary: torch.Tensor,
    model: ConceptNet2,
    device=device,
):
    image_attribute = torch.zeros(
        len(dataset), dictionary.size(1), dtype=float, device=device
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    start = 0
    for i, data in enumerate(tqdm(dataloader)):
        image, _ = data

        image = image.to(device)
        image_features = model_clip.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features = image_features.repeat(dictionary.size(1), 1, 1).permute(
            1, 0, 2
        )
        dictionary_extended = dictionary.T.repeat(image_features.size(0), 1, 1)
        input_features = torch.cat((image_features, dictionary_extended), dim=2)
        input_features = torch.flatten(input_features, 0, 1)

        output = model(input_features.float()).squeeze().view(-1, dictionary.size(1))
        answers = (output > -0.4).float()

        image_attribute[start : start + image.size(0)] = answers
        start += image.size(0)

    return image_attribute.cpu().numpy()


def main():
    clip_backbone = "ViT-B/16"
    model_clip, preprocess = clip.load(clip_backbone, device=device)
    model_clip.eval()

    train_dataset = ImageNet(data_dir, train=True, transform=preprocess)
    val_dataset = ImageNet(data_dir, train=False, transform=preprocess)

    attribute_dir = os.path.join(imagenet_dir, "attributes")
    with open(os.path.join(attribute_dir, "attributes.txt")) as f:
        lines = f.readlines()
        attributes = [line.strip() for line in lines]

    dictionary = get_dictionary(attributes, model_clip, device=device)

    model = ConceptNet2(embed_dims=512).to(device)
    weights_name = "imagenet_answers_clip_finetuned_depends_no_classifier_epoch_611"
    state_dict = torch.load(
        os.path.join(weights_dir, f"{weights_name}.pt"), map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()

    train_image_attribute = answer_query(
        train_dataset, model_clip, dictionary, model, device=device
    )
    np.save(
        os.path.join(attribute_dir, "train_image_attribute.npy"), train_image_attribute
    )

    val_image_attribute = answer_query(
        val_dataset, model_clip, dictionary, model, device=device
    )
    np.save(os.path.join(attribute_dir, "val_image_attribute.npy"), val_image_attribute)


if __name__ == "__main__":
    main()
