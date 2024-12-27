import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from classifiers import ImageClassifier, get_classifier
from configs import Config
from configs import Constants as c
from configs import get_config

# from classifiers import CLIPClassifier
from datasets import DatasetWithAttributes, get_dataset
from listener_model import ClaimListener, CUBDistributionListener, CUBTopicListener
from speaker_model import ClaimSpeaker

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = c.DEVICE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument(
        "--beta", type=float, default=None, help="DPO regularization parameter"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Explanation length regularization parameter",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Listener action regularization parameter",
    )
    parser.add_argument(
        "--temperature_scale",
        type=float,
        default=None,
        help="Temperature scale for distribution listener",
    )
    # parser.add_argument(
    #     "--k",
    #     type=int,
    #     default=8,
    #     help="Number of explanations to construct preference pairs",
    # )
    parser.add_argument("--lr", type=float, default=1e-04, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay")
    parser.add_argument("--workdir", type=str, default=c.WORKDIR)
    return parser.parse_args()


class PredictionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["image"])

    def __getitem__(self, idx):
        return {
            "image": torch.tensor(self.data["image"][idx], dtype=torch.float),
            "image_features": torch.tensor(
                self.data["image_features"][idx], dtype=torch.float
            ),
            "image_attribute": torch.tensor(
                self.data["image_attribute"][idx], dtype=torch.long
            ),
            "prediction": torch.tensor(self.data["prediction"][idx], dtype=torch.long),
        }


class PreferenceDataset(Dataset):
    def __init__(self, data):
        margin_mask = data["margin_mask"]
        data = {k: v[margin_mask] for k, v in data.items()}

        check = {k: [np.all(~np.isnan(v)), np.all(v != -1)] for k, v in data.items()}
        assert all([all(v) for v in check.values()])

        self.data = data

    def __len__(self):
        return len(self.data["image_features"])

    def __getitem__(self, idx):
        return {
            "image_features": torch.tensor(
                self.data["image_features"][idx], dtype=torch.float
            ),
            "prediction": torch.tensor(self.data["prediction"][idx], dtype=torch.long),
            "chosen": torch.tensor(self.data["chosen"][idx], dtype=torch.long),
            "rejected": torch.tensor(self.data["rejected"][idx], dtype=torch.long),
            "chosen_logp": torch.tensor(
                self.data["chosen_logp"][idx], dtype=torch.float
            ),
            "rejected_logp": torch.tensor(
                self.data["rejected_logp"][idx], dtype=torch.float
            ),
        }


class ExplanationDataset(Dataset):
    def __init__(self, data):
        check = {k: [np.all(~np.isnan(v)), np.all(v != -1)] for k, v in data.items()}
        assert all([all(v) for v in check.values()])

        self.data = data

    def __len__(self):
        return len(self.data["image_features"])

    def __getitem__(self, idx):
        return {
            "image_features": torch.tensor(
                self.data["image_features"][idx], dtype=torch.float
            ),
            "prediction": torch.tensor(self.data["prediction"][idx], dtype=torch.long),
            "explanation": torch.tensor(
                self.data["explanation"][idx], dtype=torch.long
            ),
        }


class Monitor:
    def __init__(self):
        self.data = {}
        self.global_samples = 0
        self.logger = wandb

    def zero(self):
        self.data = {}

    def update(self, data, num_samples):
        self.global_samples += num_samples

        for key, value in data.items():
            if key in self.data:
                self.data[key].append((value, num_samples))
            else:
                self.data[key] = [(value, num_samples)]

    def reduce(self):
        return {
            key: sum([v[0] for v in value]) / sum([v[1] for v in value])
            for key, value in self.data.items()
        }

    def log(self, prefix, step):
        data = self.reduce()
        self.logger.log({f"{prefix}/{k}": v for k, v in data.items()}, step=step)


@torch.no_grad()
def create_prediction_dataset(
    dataset: DatasetWithAttributes, classifier: ImageClassifier
):
    classifier.eval()

    classes = dataset.classes
    class_prompts = [f"A photo of a {class_name}" for class_name in classes]

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    prediction_data = {
        "image": -np.ones((len(dataset), 3, 224, 224)),
        "image_features": -np.ones((len(dataset), classifier.embed_dim)),
        "image_attribute": -np.ones((len(dataset), len(dataset.claims))),
        "prediction": -np.ones(len(dataset)),
    }

    start = 0
    for _, data in enumerate(tqdm(dataloader)):
        image, _, image_attribute = data

        image = image.to(device)

        cls_output = classifier(image, class_prompts)
        image_features = cls_output["image_features"]
        logits = cls_output["logits"]
        prediction = torch.argmax(logits, dim=-1)

        end = start + image.size(0)

        prediction_data["image"][start:end] = image.cpu().numpy()
        prediction_data["image_features"][start:end] = image_features.cpu().numpy()
        prediction_data["image_attribute"][start:end] = image_attribute.numpy()
        prediction_data["prediction"][start:end] = prediction.cpu().numpy()

        start = end

    return PredictionDataset(prediction_data)


@torch.no_grad()
def create_preference_dataset(
    config: Config,
    prediction_dataset: PredictionDataset,
    speaker: ClaimSpeaker,
    listener: ClaimListener,
):
    speaker.eval()

    gamma = config.listener.gamma
    alpha = config.speaker.alpha
    k = config.speaker.k

    pair_mask = torch.combinations(torch.arange(k), r=2).numpy()

    context_length = speaker.multimodal_config["context_length"]
    preference_data = {
        "image_features": -np.ones(
            (
                pair_mask.shape[0] * len(prediction_dataset),
                context_length,
                prediction_dataset.data["image_features"].shape[-1],
            )
        ),
        "prediction": -np.ones(pair_mask.shape[0] * len(prediction_dataset)),
        "chosen": -np.ones(
            (pair_mask.shape[0] * len(prediction_dataset), context_length)
        ),
        "rejected": -np.ones(
            (pair_mask.shape[0] * len(prediction_dataset), context_length)
        ),
        "chosen_logp": -np.ones(pair_mask.shape[0] * len(prediction_dataset)),
        "rejected_logp": -np.ones(pair_mask.shape[0] * len(prediction_dataset)),
        "margin_mask": np.zeros(
            pair_mask.shape[0] * len(prediction_dataset), dtype=bool
        ),
    }

    logp = -np.ones((len(prediction_dataset), k))
    consistency = -np.ones((len(prediction_dataset), k))
    length = -np.ones((len(prediction_dataset), k))

    dataloader = DataLoader(prediction_dataset, batch_size=16, shuffle=False)
    start, pair_start = 0, 0
    for _, data in enumerate(tqdm(dataloader)):
        image_features = data["image_features"]
        image_attribute = data["image_attribute"]
        prediction = data["prediction"]

        image_features = image_features.to(device)
        prediction = prediction.to(device)

        image_features = image_features.unsqueeze(1).expand(-1, context_length, -1)

        explanation, explanation_logp = speaker.explain(image_features, prediction, k)
        # compute explanation length
        explanation_length = torch.sum(explanation < len(speaker.claims), dim=-1).cpu()
        # MAYBE: randomly trim explanations to better support EOS token
        explanation_consistency, action = listener.listen(image_attribute, explanation)
        # evaluate action taken by listener
        action_loss = (
            F.cross_entropy(
                action.view(-1, action.size(-1)),
                prediction.repeat_interleave(k),
                reduction="none",
            )
            .view_as(explanation_consistency)
            .cpu()
        )
        explanation_score = (
            explanation_consistency - gamma * explanation_length - alpha * action_loss
        )

        explanation = explanation.cpu().numpy()
        explanation_logp = explanation_logp.cpu().numpy()
        explanation_consistency = explanation_consistency.cpu().numpy()
        explanation_score = explanation_score.cpu().numpy()

        explanation_length = np.sum(explanation < len(speaker.claims), axis=-1)

        # save explanation statistics
        end = start + explanation_logp.shape[0]

        logp[start:end] = explanation_logp
        consistency[start:end] = explanation_consistency
        length[start:end] = explanation_length

        start = end

        # compose preference pairs
        pair_explanation = explanation[:, pair_mask]
        pair_score = explanation_score[:, pair_mask]
        pair_logp = explanation_logp[:, pair_mask]

        sorted_pair_idx = np.argsort(-pair_score, axis=-1)
        sorted_pair_explanation = np.take_along_axis(
            pair_explanation, sorted_pair_idx[..., None], axis=-2
        )
        sorted_pair_score = np.take_along_axis(pair_score, sorted_pair_idx, axis=-1)
        sorted_pair_logp = np.take_along_axis(pair_logp, sorted_pair_idx, axis=-1)

        n_pairs = pair_mask.shape[0]
        image_features = image_features.cpu().unsqueeze(1).expand(-1, n_pairs, -1, -1)
        prediction = prediction.cpu().unsqueeze(1).expand(-1, n_pairs)

        image_features = image_features.flatten(0, 1)
        prediction = prediction.flatten()
        sorted_pair_explanation = sorted_pair_explanation.reshape(
            -1, 2, sorted_pair_explanation.shape[-1]
        )
        sorted_pair_score = sorted_pair_score.reshape(-1, 2)
        sorted_pair_logp = sorted_pair_logp.reshape(-1, 2)

        win_margin = 0.50
        win_rate = softmax(sorted_pair_score, axis=-1)[:, 0]
        margin_mask = win_rate > win_margin

        # save preference pair statistics
        pair_end = pair_start + len(sorted_pair_explanation)

        preference_data["image_features"][pair_start:pair_end] = image_features.numpy()
        preference_data["prediction"][pair_start:pair_end] = prediction.numpy()
        preference_data["chosen"][pair_start:pair_end] = sorted_pair_explanation[:, 0]
        preference_data["rejected"][pair_start:pair_end] = sorted_pair_explanation[:, 1]
        preference_data["chosen_logp"][pair_start:pair_end] = sorted_pair_logp[:, 0]
        preference_data["rejected_logp"][pair_start:pair_end] = sorted_pair_logp[:, 1]
        preference_data["margin_mask"][pair_start:pair_end] = margin_mask

        pair_start = pair_end

    preference_dataset = PreferenceDataset(preference_data)
    mean_logp = np.mean(logp)
    mean_consistency = np.mean(consistency)
    mean_length = np.mean(length)
    return preference_dataset, mean_logp, mean_consistency, mean_length


@torch.no_grad()
def create_explanation_dataset(
    prediction_dataset: PredictionDataset, speaker: ClaimSpeaker, k: int
):
    speaker.eval()

    context_length = speaker.multimodal_config["context_length"]
    explanation_data = {
        "image_features": -np.ones(
            (
                len(prediction_dataset) * k,
                context_length,
                prediction_dataset.data["image_features"].shape[-1],
            )
        ),
        "prediction": -np.ones(len(prediction_dataset) * k),
        "explanation": -np.ones((len(prediction_dataset) * k, context_length)),
    }

    dataloader = DataLoader(prediction_dataset, batch_size=16, shuffle=False)
    start = 0
    for _, data in enumerate(tqdm(dataloader)):
        image_features = data["image_features"]
        prediction = data["prediction"]

        image_features = image_features.to(device)
        prediction = prediction.to(device)

        image_features = image_features.unsqueeze(1).expand(-1, context_length, -1)

        explanation, _ = speaker.explain(image_features, prediction, k)

        image_features = image_features.unsqueeze(1).expand(-1, k, -1, -1)
        image_features = image_features.flatten(0, 1)
        image_features = image_features.cpu().numpy()

        prediction = prediction.unsqueeze(1).expand(-1, k)
        prediction = prediction.flatten()
        prediction = prediction.cpu().numpy()

        explanation = explanation.flatten(0, 1)
        explanation = explanation.cpu().numpy()

        end = start + explanation.shape[0]

        explanation_data["image_features"][start:end] = image_features
        explanation_data["prediction"][start:end] = prediction
        explanation_data["explanation"][start:end] = explanation

        start = end

    return ExplanationDataset(explanation_data)


def update_speaker(
    config: Config,
    preference_dataset: PreferenceDataset,
    speaker: ClaimSpeaker,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    monitor: Monitor,
):
    speaker.train()

    beta = config.speaker.beta

    dataloader = DataLoader(preference_dataset, batch_size=16, shuffle=True)

    for _, data in enumerate(tqdm(dataloader)):
        data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        image_features = data["image_features"]
        prediction = data["prediction"]
        chosen = data["chosen"]
        rejected = data["rejected"]
        ref_chosen_logp = data["chosen_logp"]
        ref_rejected_logp = data["rejected_logp"]

        image_features = image_features.unsqueeze(1).expand(-1, 2, -1, -1)
        image_features = image_features.flatten(0, 1)

        prediction = prediction.unsqueeze(1).expand(-1, 2)
        prediction = prediction.flatten()

        preference = torch.stack([chosen, rejected], dim=1)
        preference = preference.flatten(0, 1)

        optimizer.zero_grad()

        explanation_logp = speaker.explanation_logp(
            image_features, prediction, preference
        )
        explanation_logp = explanation_logp.reshape(-1, 2)
        chosen_logp = explanation_logp[:, 0]
        rejected_logp = explanation_logp[:, 1]

        logratios = chosen_logp - rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp
        logits = logratios - ref_logratios
        loss = -F.logsigmoid(beta * logits).sum()
        loss.backward()
        nn.utils.clip_grad_norm_(speaker.parameters(), 1.0)
        optimizer.step()

        monitor.update({"Speaker loss": loss.item()}, explanation_logp.size(0))

    scheduler.step()


def update_listener(
    explanation_dataset: ExplanationDataset,
    listener: ClaimListener,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    monitor: Monitor,
):
    listener.train()

    dataloader = DataLoader(explanation_dataset, batch_size=16, shuffle=True)

    for _, data in enumerate(tqdm(dataloader)):
        data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        explanation = data["explanation"]
        prediction = data["prediction"]

        optimizer.zero_grad()
        action = listener(None, explanation)
        loss = F.cross_entropy(action, prediction, reduction="sum")
        loss.backward()
        nn.utils.clip_grad_norm_(listener.parameters(), 1.0)
        optimizer.step()

        listener_prediction = torch.argmax(action, dim=-1)
        accuracy = torch.sum((listener_prediction == prediction).float())

        monitor.update(
            {"Listener loss": loss.item(), "Listener accuracy": accuracy.item()},
            explanation.size(0),
        )

    scheduler.step()


def train_iteration(
    config: Config,
    prediction_dataset: PredictionDataset,
    speaker: ClaimSpeaker,
    listener: ClaimListener,
    speaker_optimizer: torch.optim.Optimizer,
    speaker_scheduler: torch.optim.lr_scheduler,
    listener_optimizer: torch.optim.Optimizer,
    listener_scheduler: torch.optim.lr_scheduler,
    monitor: Monitor,
    iteration: int,
):
    print("Creating preference dataset...")
    (
        preference_dataset,
        mean_logp,
        mean_consistency,
        mean_length,
    ) = create_preference_dataset(config, prediction_dataset, speaker, listener)
    print(
        f"Created dataset with {len(preference_dataset)} preference pairs,"
        f" {mean_logp:.2f} mean logp, {mean_consistency:.2f} mean consistency,"
        f" {mean_length:.2f} mean length"
    )
    monitor.logger.log(
        {
            "Train/Preference pairs": len(preference_dataset),
            "Train/Explanation logp": mean_logp,
            "Train/Explanation consistency": mean_consistency,
            "Train/Explanation length": mean_length,
        },
        step=iteration,
    )

    print("Updating speaker...")
    update_speaker(
        config,
        preference_dataset,
        speaker,
        speaker_optimizer,
        speaker_scheduler,
        monitor,
    )
    monitor.log(prefix="Train", step=iteration)
    monitor.zero()

    print("Creating explanation dataset...")
    explanation_dataset = create_explanation_dataset(prediction_dataset, speaker, 16)

    print("Updating listener...")
    update_listener(
        explanation_dataset, listener, listener_optimizer, listener_scheduler, monitor
    )
    monitor.log(prefix="Train", step=iteration)
    monitor.zero()


@torch.no_grad()
def evaluate(
    dataset: Dataset,
    classifier: ImageClassifier,
    speaker: ClaimSpeaker,
    listener: ClaimListener,
    monitor: Monitor,
    iteration: int,
):
    print("Evaluating models...")
    classifier.eval()
    speaker.eval()
    listener.eval()

    classes = dataset.classes
    class_prompts = [f"A photo of a {class_name}" for class_name in classes]

    context_length = speaker.multimodal_config["context_length"]

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    for _, data in enumerate(tqdm(dataloader)):
        image, _, image_attribute = data

        image = image.to(device)

        cls_output = classifier(image, class_prompts)
        image_features = cls_output["image_features"]
        logits = cls_output["logits"]
        prediction = torch.argmax(logits, dim=-1)

        image_features = image_features.unsqueeze(1).expand(-1, context_length, -1)

        explanation, explanation_logp = speaker.explain(image_features, prediction)
        explanation_consistency, action = listener.listen(image_attribute, explanation)
        listener_prediction = torch.argmax(action, dim=-1).squeeze()
        correct = listener_prediction == prediction

        monitor.update(
            {
                "Explanation consistency": explanation_consistency.sum().item(),
                "Explanation logp": explanation_logp.sum().item(),
                "Listener accuracy": correct.sum().item(),
            },
            image.size(0),
        )

    monitor.log(prefix="Val", step=iteration)
    monitor.zero()


def initialize_optimizer(model: nn.Module, lr: float, weight_decay: float):
    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=lr,
    )


def main(args):
    config_name = args.config_name
    beta = args.beta
    gamma = args.gamma
    alpha = args.alpha
    lr = args.lr
    wd = args.wd
    workdir = args.workdir

    config = get_config(config_name)
    if beta is not None:
        config.speaker.beta = beta
    if gamma is not None:
        config.listener.gamma = gamma
    if alpha is not None:
        config.speaker.alpha = alpha

    classifier = get_classifier(
        config, from_pretrained=True, workdir=workdir, device=device
    )

    train_dataset = get_dataset(
        config, train=True, transform=classifier.preprocess, return_attribute=True
    )
    val_dataset = get_dataset(
        config, train=False, transform=classifier.preprocess, return_attribute=True
    )

    classes = train_dataset.classes
    claims = train_dataset.claims

    # Initialize speaker model
    speaker = ClaimSpeaker(config, classifier, len(classes), claims, device=device)
    speaker_optimizer = initialize_optimizer(speaker, lr, wd)
    speaker_scheduler = torch.optim.lr_scheduler.StepLR(
        speaker_optimizer, step_size=5, gamma=0.5
    )

    # Initialize listener model
    if config.data.listener_type == "claim":
        Listener = ClaimListener
    elif config.data.listener_type == "topic":
        Listener = CUBTopicListener
    elif config.data.listener_type == "distribution":
        Listener = CUBDistributionListener
    listener = Listener(config, len(classes), claims, device=device)
    listener_optimizer = initialize_optimizer(listener, lr, wd)
    listener_scheduler = torch.optim.lr_scheduler.StepLR(
        listener_optimizer, step_size=5, gamma=0.5
    )

    print("Creating prediction dataset...")
    train_prediction_dataset = create_prediction_dataset(train_dataset, classifier)

    run_name = config.run_name()
    wandb.init(project="llm_pragmatixs", name=run_name, config=args)
    monitor = Monitor()

    evaluate(val_dataset, classifier, speaker, listener, monitor, 0)

    total_iterations = 25
    for iteration in range(total_iterations):
        print(f"Iteration {iteration + 1}")
        train_iteration(
            config,
            train_prediction_dataset,
            speaker,
            listener,
            speaker_optimizer,
            speaker_scheduler,
            listener_optimizer,
            listener_scheduler,
            monitor,
            iteration + 1,
        )
        evaluate(
            val_dataset,
            classifier,
            speaker,
            listener,
            monitor,
            iteration + 1,
        )

    state = {"speaker": speaker.state_dict(), "listener": listener.state_dict()}
    state_path = os.path.join(workdir, "weights", f"{run_name}.pt")
    torch.save(state, state_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
