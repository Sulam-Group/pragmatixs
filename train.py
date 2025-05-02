import argparse
import logging

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from classifiers import get_classifier
from configs import Config
from configs import Constants as C
from configs import get_config
from datasets import get_dataset
from listeners import Listener, get_listener
from speaker import ClaimSpeaker
from train_utils import (
    CosineScheduler,
    ExplanationDataset,
    Monitor,
    PredictionDataset,
    PreferenceDataset,
    generate_and_save_explanations,
    generate_and_save_preferences,
    initialize_optimizer,
    rank_zero_only,
    setup_logging,
)

logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--listener_type", type=str, default=None)
    parser.add_argument("--explanation_length", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--preference", type=str, default=None)
    parser.add_argument("--listener_k", type=int, default=None)
    parser.add_argument("--temperature_scale", type=float, default=None)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    parser.add_argument("--dist", action="store_true", default=False)
    return parser.parse_args()


def update_speaker(
    config: Config = None,
    preference_dataset: PreferenceDataset = None,
    speaker: ClaimSpeaker = None,
    optimizer: torch.optim.Optimizer = None,
    monitor: Monitor = None,
    device=C.device,
):
    logger.info("Updating speaker...")

    speaker.train()
    monitor.zero()

    beta = config.speaker.beta

    sampler, batch_size = None, config.training.batch_size
    if distributed.is_initialized():
        sampler = DistributedSampler(preference_dataset, shuffle=False)

    dataloader = DataLoader(
        preference_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
    )
    for i, data in enumerate(tqdm(dataloader)):
        image_tokens = data["image_tokens"].to(device)
        chosen = data["chosen"].to(device)
        rejected = data["rejected"].to(device)
        ref_chosen_logp = data["chosen_logp"].to(device)
        ref_rejected_logp = data["rejected_logp"].to(device)

        optimizer.zero_grad()
        chosen_output = speaker(image_tokens, chosen)
        rejected_output = speaker(image_tokens, rejected)

        chosen_logp = chosen_output["explanation_logp"]
        rejected_logp = rejected_output["explanation_logp"]

        logratios = chosen_logp - rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp
        logits = logratios - ref_logratios
        loss = -F.logsigmoid(beta * logits).sum()
        loss.backward()
        nn.utils.clip_grad_norm_(speaker.parameters(), config.training.max_grad_norm)
        optimizer.step()

        monitor.update(
            {
                "chosen logp": chosen_logp,
                "rejected logp": rejected_logp,
                "logratios": logratios,
                "speaker loss": loss,
            },
            num_samples=image_tokens.size(0),
        )

        log_step = 20
        if (i + 1) % log_step == 0:
            monitor.log(prefix="train")


def update_listener(
    config: Config = None,
    explanation_dataset: ExplanationDataset = None,
    listener: Listener = None,
    optimizer: torch.optim.Optimizer = None,
    monitor: Monitor = None,
    device=C.device,
):
    logger.info("Updating listener...")

    listener.train()
    monitor.zero()

    sampler, batch_size = None, config.training.batch_size
    if distributed.is_initialized():
        sampler = DistributedSampler(explanation_dataset, shuffle=False)

    dataloader = DataLoader(
        explanation_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
    )
    for i, data in enumerate(tqdm(dataloader)):
        prediction = data["prediction"].to(device)
        explanation = data["explanation"].to(device)

        optimizer.zero_grad()
        action = listener(explanation)
        loss = F.cross_entropy(action, prediction, reduction="sum")
        loss.backward()
        nn.utils.clip_grad_norm_(listener.parameters(), config.training.max_grad_norm)
        optimizer.step()

        listener_prediction = torch.argmax(action, dim=-1)
        correct = (listener_prediction == prediction).float()

        monitor.update(
            {"listener loss": loss, "listener accuracy": correct},
            num_samples=explanation.size(0),
        )

        log_step = 20
        if (i + 1) % log_step == 0:
            monitor.log(prefix="train")


def train_iteration(
    config: Config = None,
    prediction_dataset: PredictionDataset = None,
    speaker: ClaimSpeaker = None,
    listener: Listener = None,
    speaker_optimizer: torch.optim.Optimizer = None,
    listener_optimizer: torch.optim.Optimizer = None,
    speaker_scheduler: CosineScheduler = None,
    listener_scheduler: CosineScheduler = None,
    epoch: int = 0,
    monitor: Monitor = None,
    workdir=C.workdir,
    device=C.device,
):
    logger.info(f"Iteration {epoch+1}")

    generate_and_save_preferences(
        config=config,
        prediction_dataset=prediction_dataset,
        speaker=speaker,
        listener=listener,
        epoch=epoch,
        workdir=workdir,
        device=device,
    )
    if distributed.is_initialized():
        distributed.barrier()

    preference_dataset = PreferenceDataset(
        config=config, prediction_dataset=prediction_dataset, workdir=workdir
    )

    update_speaker(
        config=config,
        preference_dataset=preference_dataset,
        speaker=speaker,
        optimizer=speaker_optimizer,
        monitor=monitor,
        device=device,
    )
    speaker_scheduler.step()
    if distributed.is_initialized():
        distributed.barrier()

    generate_and_save_explanations(
        config=config,
        prediction_dataset=prediction_dataset,
        speaker=speaker,
        epoch=epoch,
        workdir=workdir,
        device=device,
    )
    listener_scheduler.step()
    if distributed.is_initialized():
        distributed.barrier()

    explanation_dataset = ExplanationDataset(
        config=config, prediction_dataset=prediction_dataset, workdir=workdir
    )

    update_listener(
        config=config,
        explanation_dataset=explanation_dataset,
        listener=listener,
        optimizer=listener_optimizer,
        monitor=monitor,
        device=device,
    )
    if distributed.is_initialized():
        distributed.barrier()

    rank = 0
    if distributed.is_initialized():
        rank = distributed.get_rank()
    if rank == 0:
        monitor.logger.log(
            {
                "train/speaker_lr": speaker_optimizer.param_groups[0]["lr"],
                "train/listener_lr": listener_optimizer.param_groups[0]["lr"],
            },
            step=monitor.global_samples,
        )
    
@torch.no_grad()
def evaluate_classifier(
    dataset, prediction_dataset: PredictionDataset, classifier: nn.Module, device: torch.device
):
    classifier.eval()
    samples = dataset.get_classes_and_samples()
    Labels = [i[1] for i in samples[1]]
    
    dataloader = DataLoader(prediction_dataset, batch_size=16, shuffle=False)
    Predictions = []
    for _, data in enumerate(tqdm(dataloader)):
        prediction = data["prediction"]
        Predictions.extend(prediction.tolist())
    # positive and negative ratios
    positive_p = np.sum(np.array(Labels) == 1) 
    negative_n = np.sum(np.array(Labels) == 0)
    print(
        f"Positive: {positive_p}, "
        f"Negative: {negative_n}"
    )
    accuracy = np.sum(np.array(Labels) == np.array(Predictions)) / len(Labels)
    print(f"Accuracy: {accuracy:.2f}")
    # sensitivity and specificity
    tp = np.sum(
        np.logical_and(np.array(Labels) == 1, np.array(Predictions) == 1)
    )
    tn = np.sum(
        np.logical_and(np.array(Labels) == 0, np.array(Predictions) == 0)
    )
    fp = np.sum(
        np.logical_and(np.array(Labels) == 0, np.array(Predictions) == 1)
    )
    fn = np.sum(
        np.logical_and(np.array(Labels) == 1, np.array(Predictions) == 0)
    )
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    f1_score = (
        2 * precision * sensitivity / (precision + sensitivity)
    )
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    print(f"F1 score: {f1_score:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"TPR: {tp / (tp + fn):.2f}, TNR: {tn / (tn + fp):.2f}")
        


@torch.no_grad()
@rank_zero_only
def evaluate(
    config: Config = None,
    dataset: PredictionDataset = None,
    speaker: ClaimSpeaker = None,
    listener: Listener = None,
    monitor: Monitor = None,
    device=C.device,
):
    logger.info("Evaluating...")

    speaker.eval()
    listener.eval()
    monitor.zero()

    dist = config.data.distributed
    explain = speaker.module.explain if dist else speaker.explain
    listen = listener.module.listen if dist else listener.listen
    speaker_claims = speaker.module.claims if dist else speaker.claims
    listener_claims = listener.module.claims if dist else listener.claims
    assert speaker_claims == listener_claims
    claims = speaker_claims

    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=False
    )
    for _, data in enumerate(tqdm(dataloader)):
        image_tokens = data["image_tokens"].to(device) # (batch_size, seq_len, dim) (16, 256, 1024)
        image_attribute = data["image_attribute"].to(device) # (batch_size, num_attributes) (16, 312)
        prediction = data["prediction"].to(device) # (batch_size) (16)

        explanation, explanation_logp = explain(image_tokens)
        consistency, action = listen(image_attribute, explanation)

        explanation_claims = explanation[..., 0]
        explanation_claims_cls = explanation[..., 1]

        _image_attribute = torch.cat(
            [
                image_attribute,
                torch.zeros(image_attribute.size(0), 3, device=device),
            ],
            dim=-1,
        )
        target_cls = torch.gather(_image_attribute, -1, explanation_claims)

        claims_mask = explanation_claims < len(claims)
        target_cls_mask = target_cls != -1

        explanation_length = torch.sum(claims_mask, dim=-1)

        accuracy_mask = claims_mask * target_cls_mask
        correct_claims = accuracy_mask * (explanation_claims_cls == target_cls)
        explanation_accuracy = torch.sum(correct_claims, dim=-1) / torch.sum(
            accuracy_mask, dim=-1
        )

        explanation_sentiment = (
            torch.sum(explanation_claims_cls, dim=-1) / explanation_length
        )

        listener_prediction = torch.argmax(action, dim=-1)
        listener_correct = (listener_prediction == prediction).float()

        monitor.update(
            {
                "explanation accuracy": explanation_accuracy,
                "explanation consistency": consistency,
                "explanation logp": explanation_logp,
                "explanation length": explanation_length,
                "explanation sentiment": explanation_sentiment,
                "listener accuracy": listener_correct,
            },
            num_samples=image_tokens.size(0),
            increase_global_samples=False,
        )

    monitor.log(prefix="val")


def main(args):
    config_name = args.config
    listener_type = args.listener_type
    explanation_length = args.explanation_length
    k = args.k
    beta = args.beta
    gamma = args.gamma
    alpha = args.alpha
    listener_k = args.listener_k
    temperature_scale = args.temperature_scale
    preference = args.preference
    workdir = args.workdir
    dist = args.dist

    config = get_config(config_name)
    config.data.distributed = dist
    if listener_type is not None:
        config.listener.type = listener_type
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

    rank = 0
    if dist:
        distributed.init_process_group(backend="nccl")
        rank = distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = C.device

    classifier = get_classifier(
        config, from_pretrained=True, workdir=workdir, device=device
    )

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

    classes, claims = train_dataset.classes, train_dataset.claims
    speaker = ClaimSpeaker(config, classifier, claims, device=device)
    Listener = get_listener(config.listener.type)
    listener = Listener(config, len(classes), claims, workdir=workdir, device=device)
    if dist:
        speaker = nn.parallel.DistributedDataParallel(
            speaker, device_ids=[device], find_unused_parameters=True
        )
        listener = nn.parallel.DistributedDataParallel(listener, device_ids=[device])

    speaker_optimizer = initialize_optimizer(
        speaker, config.training.max_lr, config.training.wd
    )
    speaker_scheduler = CosineScheduler(
        optimizer=speaker_optimizer,
        total_steps=config.training.iterations,
        min_lr=config.training.min_lr,
        max_lr=config.training.max_lr,
    )

    listener_optimizer = initialize_optimizer(
        listener, config.training.max_lr, config.training.wd
    )
    listener_scheduler = CosineScheduler(
        optimizer=listener_optimizer,
        total_steps=config.training.iterations,
        min_lr=config.training.min_lr,
        max_lr=config.training.max_lr,
    )

    train_prediction_dataset = PredictionDataset(
        config, train_dataset, workdir=workdir, device=device
    )
    val_prediction_dataset = PredictionDataset(
        config, val_dataset, workdir=workdir, device=device
    )

    # train_prediction_dataset = Subset(train_prediction_dataset, range(1000))
    # val_prediction_dataset = Subset(val_prediction_dataset, range(100))

    monitor = Monitor(config)
    evaluate(
        config=config,
        dataset=val_prediction_dataset,
        speaker=speaker,
        listener=listener,
        monitor=monitor,
        device=device,
    )

    total_iterations = config.training.iterations
    for t in range(total_iterations):
        train_iteration(
            config=config,
            prediction_dataset=train_prediction_dataset,
            speaker=speaker,
            listener=listener,
            speaker_optimizer=speaker_optimizer,
            listener_optimizer=listener_optimizer,
            speaker_scheduler=speaker_scheduler,
            listener_scheduler=listener_scheduler,
            epoch=t,
            monitor=monitor,
            device=device,
        )

        evaluate(
            config=config,
            dataset=val_prediction_dataset,
            speaker=speaker,
            listener=listener,
            monitor=monitor,
            device=device,
        )

        if (t + 1) % 10 == 0:
            monitor.save(speaker=speaker, listener=listener, epoch=t, workdir=workdir)

    monitor.save(speaker=speaker, listener=listener, epoch=t, workdir=workdir)


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)
