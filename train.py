import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from classifiers import get_classifier
from configs import Config
from configs import Constants as C
from configs import get_config
from datasets import get_dataset
from listeners import Listener, get_listener
from speaker import ClaimSpeaker
from train_utils import (
    ExplanationDataset,
    Monitor,
    PredictionDataset,
    PreferenceDataset,
    initialize_optimizer,
)

device = C.device
monitor = Monitor()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--explanation_length", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--listener_k", type=int, default=None)
    parser.add_argument("--temperature_scale", type=float, default=None)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def update_speaker(
    config: Config,
    prediction_dataset: PredictionDataset,
    preference_dataset: PreferenceDataset,
    speaker: ClaimSpeaker,
    optimizer: torch.optim.Optimizer,
):
    speaker.train()
    monitor.zero()

    beta = config.speaker.beta

    dataloader = DataLoader(preference_dataset, batch_size=16, shuffle=True)
    for i, data in enumerate(tqdm(dataloader)):
        image_idx = data["image_idx"]

        image_tokens = torch.from_numpy(
            np.stack(
                [prediction_dataset[_idx]["image_tokens"] for _idx in image_idx], axis=0
            )
        ).to(device)
        chosen = data["chosen"].to(device)
        rejected = data["rejected"].to(device)
        ref_chosen_logp = data["chosen_logp"].to(device)
        ref_rejected_logp = data["rejected_logp"].to(device)

        chosen_logp = speaker.explanation_logp(image_tokens, chosen)
        rejected_logp = speaker.explanation_logp(image_tokens, rejected)

        logratios = chosen_logp - rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp
        logits = logratios - ref_logratios
        loss = -F.logsigmoid(beta * logits).sum()
        loss.backward()
        nn.utils.clip_grad_norm_(speaker.parameters(), 1.0)
        optimizer.step()

        monitor.update(
            {
                "chosen logp": chosen_logp.sum().cpu().item(),
                "rejected logp": rejected_logp.sum().cpu().item(),
                "logratios": logratios.sum().cpu().item(),
                "speaker loss": loss.cpu().item(),
            },
            num_samples=image_tokens.size(0),
        )

        log_step = 20
        if (i + 1) % log_step == 0:
            monitor.log(prefix="train")


def update_listener(
    prediction_dataset: PredictionDataset,
    explanation_dataset: ExplanationDataset,
    listener: Listener,
    optimizer: torch.optim.Optimizer,
):
    listener.train()
    monitor.zero()

    dataloader = DataLoader(explanation_dataset, batch_size=16, shuffle=True)
    for i, data in enumerate(tqdm(dataloader)):
        image_idx = data["image_idx"]

        explanation = data["explanation"].to(device)
        prediction = torch.from_numpy(
            np.stack(
                [prediction_dataset[_idx]["prediction"] for _idx in image_idx], axis=0
            )
        ).to(device)

        optimizer.zero_grad()
        action = listener(explanation)
        loss = F.cross_entropy(action, prediction, reduction="sum")
        loss.backward()
        nn.utils.clip_grad_norm_(listener.parameters(), 1.0)
        optimizer.step()

        listener_prediction = torch.argmax(action, dim=-1)
        correct = (listener_prediction == prediction).float()

        monitor.update(
            {
                "listener loss": loss.cpu().item(),
                "listener accuracy": correct.sum().cpu().item(),
            },
            num_samples=explanation.size(0),
        )

        log_step = 20
        if (i + 1) % log_step == 0:
            monitor.log(prefix="train")


def train_iteration(
    config: Config,
    prediction_dataset: PredictionDataset,
    speaker: ClaimSpeaker,
    listener: Listener,
    speaker_optimizer: torch.optim.Optimizer,
    listener_optimizer: torch.optim.Optimizer,
):
    print("Creating preference dataset...")
    preference_dataset = PreferenceDataset(
        config, prediction_dataset, speaker, listener, device=device
    )
    print("Created preference dataset")
    print(
        f"\tNumber of preferences: {len(preference_dataset)}\n"
        f"\tAverage chosen score: {preference_dataset.chosen_score.mean().item():.2f}"
    )

    print("Updating speaker...")
    update_speaker(
        config, prediction_dataset, preference_dataset, speaker, speaker_optimizer
    )

    print("Creating explanation dataset...")
    explanation_dataset = ExplanationDataset(
        config, prediction_dataset, speaker, device=device
    )

    print("Updating listener...")
    update_listener(
        prediction_dataset, explanation_dataset, listener, listener_optimizer
    )


@torch.no_grad()
def evaluate(dataset: PredictionDataset, speaker: ClaimSpeaker, listener: Listener):
    speaker.eval()
    listener.eval()
    monitor.zero()

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    for _, data in enumerate(tqdm(dataloader)):
        image_tokens = data["image_tokens"].to(device)
        image_attribute = data["image_attribute"].to(device)
        prediction = data["prediction"].to(device)

        explanation, explanation_logp = speaker.explain(image_tokens)
        consistency, action = listener.listen(image_attribute, explanation)

        claims = explanation[..., 0]
        claims_cls = explanation[..., 1]

        _image_attribute = torch.cat(
            [
                image_attribute,
                torch.zeros(
                    image_attribute.size(0),
                    listener.speaker_vocab_size - len(listener.claims),
                    device=image_attribute.device,
                ),
            ],
            dim=-1,
        )
        target_cls = torch.gather(_image_attribute, -1, claims)

        claims_mask = claims < len(listener.claims)
        correct = claims_mask * (claims_cls == target_cls)
        accuracy = torch.sum(correct, dim=-1) / torch.sum(claims_mask, dim=-1)

        listener_prediction = torch.argmax(action, dim=-1)
        correct = (listener_prediction == prediction).float()

        explanation_length = torch.sum(
            explanation[..., 0] < len(speaker.claims), dim=-1
        )
        explanation_sentiment = (
            torch.sum(explanation[..., 1], dim=-1) / explanation_length
        )

        monitor.update(
            {
                "explanation accuracy": accuracy.sum().cpu().item(),
                "explanation consistency": consistency.sum().cpu().item(),
                "explanation logp": explanation_logp.sum().cpu().item(),
                "explanation length": explanation_length.sum().cpu().item(),
                "explanation sentiment": explanation_sentiment.sum().cpu().item(),
                "listener accuracy": correct.sum().cpu().item(),
            },
            num_samples=image_tokens.size(0),
            increase_global_samples=False,
        )

    monitor.log(prefix="val")


def main(args):
    config_name = args.config
    explanation_length = args.explanation_length
    k = args.k
    beta = args.beta
    gamma = args.gamma
    alpha = args.alpha
    listener_k = args.listener_k
    temperature_scale = args.temperature_scale
    workdir = args.workdir

    config = get_config(config_name)
    if explanation_length is not None:
        config.data.explanation_length = explanation_length
    if k is not None:
        config.speaker.k = k
    if beta is not None:
        config.speaker.beta = beta
    if gamma is not None:
        config.listener.gamma = gamma
    if alpha is not None:
        config.speaker.alpha = alpha
    if listener_k is not None:
        config.listener.k = listener_k
    if temperature_scale is not None:
        config.listener.temperature_scale = temperature_scale

    classifier = get_classifier(
        config, from_pretrained=True, workdir=workdir, device=device
    )

    train_dataset = get_dataset(
        config, train=True, transform=classifier.preprocess, return_attribute=True
    )
    val_dataset = get_dataset(
        config, train=False, transform=classifier.preprocess, return_attribute=True
    )

    classes, claims = train_dataset.classes, train_dataset.claims
    speaker = ClaimSpeaker(config, classifier, claims, device=device)
    Listener = get_listener(config.listener.type)
    listener = Listener(config, len(classes), claims, workdir=workdir, device=device)

    speaker_optimizer = initialize_optimizer(
        speaker, config.speaker.lr, config.speaker.wd
    )
    listener_optimizer = initialize_optimizer(
        listener, config.listener.lr, config.listener.wd
    )

    print("Creating prediction datasets")
    train_prediction_dataset = PredictionDataset(
        config, train_dataset, workdir=workdir, device=device
    )
    val_prediction_dataset = PredictionDataset(
        config, val_dataset, workdir=workdir, device=device
    )

    run_name = config.run_name()
    wandb.init(project="pragmatics", name=run_name, config=config.to_dict())

    print("Evaluating initialization...")
    evaluate(val_prediction_dataset, speaker, listener)

    total_iterations = 50
    for t in range(total_iterations):
        print(f"Iteration {t+1}")
        train_iteration(
            config,
            train_prediction_dataset,
            speaker,
            listener,
            speaker_optimizer,
            listener_optimizer,
        )

        print("Evaluating speaker...")
        evaluate(val_prediction_dataset, speaker, listener)

        weights_dir = os.path.join(workdir, "weights", run_name)
        os.makedirs(weights_dir, exist_ok=True)

        if (t + 1) % 10 == 0:
            torch.save(
                {"speaker": speaker.state_dict(), "listener": listener.state_dict()},
                os.path.join(weights_dir, f"iteration_{t+1}.pt"),
            )
            with open(os.path.join(weights_dir, "latest.txt"), "w") as f:
                f.write(f"iteration_{t+1}.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
