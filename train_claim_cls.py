import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from classifiers import get_classifier
from configs import Constants as C
from configs import get_config
from datasets import get_dataset
from speaker import ClaimSpeaker
from train_utils import PredictionDataset

device = C.device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


class ClaimClassifier(nn.Module):
    def __init__(self, speaker: ClaimSpeaker):
        super().__init__()
        self.attn_pool_cls = speaker._attn_pool_cls
        self.logit_scale = speaker.logit_scale

    def forward(self, image_tokens):
        logits = self.attn_pool_cls(image_tokens)
        logits = torch.squeeze(logits, 1)
        return self.logit_scale.exp() * logits


def main(args):
    config_name = args.config
    workdir = args.workdir

    config = get_config(config_name)

    classifier = get_classifier(config)

    train_dataset = get_dataset(
        config,
        train=True,
        transform=classifier.preprocess,
        return_attribute=True,
        workdir=workdir,
    )
    val_dataset = get_dataset(
        config,
        train=False,
        transform=classifier.preprocess,
        return_attribute=True,
        workdir=workdir,
    )

    train_prediction_dataset = PredictionDataset(
        config, train_dataset, workdir=workdir, device=device
    )
    val_prediction_dataset = PredictionDataset(
        config, val_dataset, workdir=workdir, device=device
    )

    speaker = ClaimSpeaker(config, classifier, train_dataset.claims, device=device)
    claim_classifier = ClaimClassifier(speaker)

    optimizer = torch.optim.Adam(claim_classifier.parameters(), lr=1e-04)

    wandb.init(project="pragmatics", name=f"{config.data.dataset}_claim_cls")

    datasets = {"train": train_prediction_dataset, "val": val_prediction_dataset}

    total_iterations = 5
    for t in range(total_iterations):
        for op, dataset in datasets.items():
            train = op == "train"
            if train:
                torch.set_grad_enabled(True)
                claim_classifier.train()
            else:
                torch.set_grad_enabled(False)
                claim_classifier.eval()

            dataloader = DataLoader(dataset, batch_size=64, shuffle=train)

            running_samples, running_loss, running_accuracy = 0, 0, 0
            for i, data in enumerate(tqdm(dataloader)):
                image_tokens = data["image_tokens"].to(device)
                image_attribute = data["image_attribute"].float().to(device)

                attribute_mask = image_attribute != -1

                optimizer.zero_grad()
                output = claim_classifier(image_tokens)
                output = output[:, : image_attribute.size(1)]
                loss = F.binary_cross_entropy_with_logits(
                    output, image_attribute.float(), reduction="none"
                )
                loss = torch.sum(attribute_mask * loss)

                if train:
                    loss.backward()
                    optimizer.step()

                prediction = output >= 0.0
                correct = attribute_mask * (prediction == image_attribute)
                accuracy = torch.sum(correct, dim=-1) / torch.sum(
                    attribute_mask, dim=-1
                )

                running_samples += image_tokens.size(0)
                running_loss += loss.item()
                running_accuracy += torch.sum(accuracy).item()

                log_step = 10
                if train and ((i + 1) % log_step == 0):
                    wandb.log(
                        {
                            "train/loss": running_loss / running_samples,
                            "train/accuracy": running_accuracy / running_samples,
                        }
                    )
                    running_samples, running_loss, running_accuracy = 0, 0, 0

            if not train:
                wandb.log(
                    {
                        "val/loss": running_loss / running_samples,
                        "val/accuracy": running_accuracy / running_samples,
                    }
                )

                results_dir = os.path.join(workdir, "results", "claim_cls")
                os.makedirs(results_dir, exist_ok=True)
                with open(
                    os.path.join(results_dir, f"{config.data.dataset}.txt"), "w"
                ) as f:
                    f.write(f"{running_accuracy / running_samples:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
