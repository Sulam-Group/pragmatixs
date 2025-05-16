import argparse
import os

import clip
import numpy as np
import torch
from clip.model import CLIP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import Constants as C
from datasets import ImageNet
from vip_utils.imagenet import ConceptNet2, answer_query, get_dictionary

device = C.device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


@torch.no_grad()
def label_dataset(
    dataset: Dataset,
    model_clip: CLIP,
    dictionary: torch.Tensor,
    concept_net: ConceptNet2,
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
        answers = answer_query(
            image=image,
            model_clip=model_clip,
            dictionary=dictionary,
            concept_net=concept_net,
        )

        image_attribute[start : start + image.size(0)] = answers
        start += image.size(0)

    return image_attribute.cpu().numpy()


def main(args):
    workdir = args.workdir

    weights_dir = os.path.join(workdir, "weights", "vip")
    data_dir = os.path.join(workdir, "data")
    attribute_dir = os.path.join(data_dir, "ImageNet", "attributes")

    clip_backbone = "ViT-B/16"
    weights_name = "imagenet_answers_clip_finetuned_depends_no_classifier_epoch_611"

    model_clip, preprocess = clip.load(clip_backbone, device=device)
    dictionary = get_dictionary(
        clip_backbone=clip_backbone, workdir=workdir, device=device
    )
    model_clip.eval()

    concept_net = ConceptNet2(embed_dims=512).to(device)
    state_dict = torch.load(
        os.path.join(weights_dir, f"{weights_name}.pt"), map_location=device
    )
    concept_net.load_state_dict(state_dict)
    concept_net.eval()

    train_dataset = ImageNet(data_dir, train=True, transform=preprocess)
    val_dataset = ImageNet(data_dir, train=False, transform=preprocess)

    train_image_attribute = label_dataset(
        train_dataset, model_clip, dictionary, concept_net, device=device
    )
    np.save(
        os.path.join(attribute_dir, "train_image_attribute.npy"), train_image_attribute
    )

    val_image_attribute = label_dataset(
        val_dataset, model_clip, dictionary, concept_net, device=device
    )
    np.save(os.path.join(attribute_dir, "val_image_attribute.npy"), val_image_attribute)


if __name__ == "__main__":
    args = parse_args()
    main(args)
