import os

import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from configs import Config
from configs import Constants as C
from vip_utils.cub import NetworkCUB
from vip_utils.utils import get_run_name, load_concept_net, make_dataset


def random_sampling(n_claims: int, max_queries: int, n_samples: int):
    n_queries = torch.randint(low=0, high=max_queries, size=(n_samples,))
    mask = torch.zeros(n_samples, n_claims)

    for i, n in enumerate(n_queries):
        if n == 0:
            continue
        indices = torch.randperm(n_claims)[:n]
        mask[i, indices] = 1
    return mask


@torch.no_grad()
def biased_sampling(
    image_attribute: torch.Tensor, max_queries: int, querier: nn.Module
):
    querier.requires_grad_(False)
    device = image_attribute.device

    n, d = image_attribute.size()
    n_queries = torch.randint(low=0, high=max_queries, size=(n,)).to(device)
    mask = torch.zeros((n, d), requires_grad=False, device=device)
    for _ in range(max_queries):
        masked_input = image_attribute * mask
        query = querier(masked_input, mask)

        indices = torch.sum(mask, dim=1) <= n_queries
        mask[indices] = mask[indices] + query[indices]

    querier.requires_grad_(True)
    return mask


def train(
    config: Config,
    max_queries: int,
    epochs: int,
    batch_size: int,
    lr: float,
    tau_start: float,
    tau_end: float,
    sampling: str,
    dist: bool = False,
    workdir=C.workdir,
):
    run_name = get_run_name(config, max_queries, sampling)
    weights_dir = os.path.join(workdir, "weights", run_name)
    os.makedirs(weights_dir, exist_ok=True)

    rank = 0
    if dist:
        distributed.init_process_group(backend="nccl")
        rank = distributed.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    train_dataset = make_dataset(config, True, workdir=workdir, device=device)
    test_dataset = make_dataset(config, False, workdir=workdir, device=device)

    sampler = None
    if dist:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    concept_net = load_concept_net(workdir=workdir, device=device)

    n_classes = len(train_dataset.classes)
    n_claims = len(train_dataset.claims)
    querier = NetworkCUB(query_size=n_claims, output_size=n_claims)
    classifier = NetworkCUB(query_size=n_claims, output_size=n_classes, tau=None)
    querier = querier.to(device)
    classifier = classifier.to(device)

    if sampling == "biased":
        random_run_name = run_name.replace("biased", "random")
        random_weights_dir = os.path.join(workdir, "weights", random_run_name)

        with open(os.path.join(random_weights_dir, "latest.txt")) as f:
            latest_weights = f.read().strip()
        state_dict = torch.load(
            os.path.join(random_weights_dir, latest_weights), map_location=device
        )

        querier_state_dict = state_dict["querier"]
        classifier_state_dict = state_dict["classifier"]
        querier.load_state_dict(querier_state_dict)
        classifier.load_state_dict(classifier_state_dict)

    if dist:
        querier = DistributedDataParallel(querier, device_ids=[device])
        classifier = DistributedDataParallel(classifier, device_ids=[device])

    optimizer = optim.Adam(
        list(querier.parameters()) + list(classifier.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    taus = np.linspace(tau_start, tau_end, epochs)

    if rank == 0:
        wandb.init(project="pragmatics_vip", name=run_name)

    for epoch in range(epochs):
        querier.train()
        classifier.train()

        tau = taus[epoch]
        running_loss, running_correct, running_samples = 0.0, 0, 0
        for i, data in enumerate(tqdm(train_dataloader)):
            image, prediction = data

            image = image.to(device)
            prediction = prediction.to(device)

            with torch.no_grad():
                image_attribute = concept_net.net(image)
                image_attribute = torch.where(image_attribute > 0.0, 1.0, -1.0)

            if dist:
                querier.module.update_tau(tau)
            else:
                querier.update_tau(tau)

            optimizer.zero_grad()
            if sampling == "random":
                mask = (
                    random_sampling(n_claims, max_queries, image.size(0))
                    .to(device)
                    .float()
                )
            elif sampling == "biased":
                mask = biased_sampling(image_attribute, max_queries, querier)
            history = image_attribute * mask

            query = querier(history, mask)
            updated_history = history + image_attribute * query

            logits = classifier(updated_history)
            loss = nn.functional.cross_entropy(logits, prediction, reduction="sum")
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (logits.argmax(dim=1) == prediction).sum().item()
            running_samples += image.size(0)

        scheduler.step()

        if rank == 0:
            wandb.log(
                {
                    "Train/loss": running_loss / running_samples,
                    "Train/accuracy": running_correct / running_samples,
                }
            )

            if (epoch + 1) % 25 == 0:
                querier.eval()
                classifier.eval()

                running_correct, running_samples = 0, 0
                for data in tqdm(test_dataloader):
                    image, prediction = data

                    image = image.to(device)
                    prediction = prediction.to(device)

                    with torch.no_grad():
                        image_attribute = concept_net.net(image)
                        image_attribute = torch.where(image_attribute > 0.0, 1.0, -1.0)

                        mask = torch.zeros(image.size(0), n_claims, device=device)
                        for _ in range(50):
                            query = querier(image_attribute * mask, mask)
                            chosen_query = query.argmax(dim=1)
                            mask[np.arange(image.size(0)), chosen_query] = 1.0

                        logits = classifier(image_attribute * mask)

                    running_correct += (logits.argmax(dim=1) == prediction).sum().item()
                    running_samples += image.size(0)

                wandb.log({"Test/accuracy": running_correct / running_samples})
                running_correct = 0
                running_samples = 0

                querier_state_dict = (
                    querier.module.state_dict() if dist else querier.state_dict()
                )
                classifier_state_dict = (
                    classifier.module.state_dict() if dist else classifier.state_dict()
                )
                state_dict = {
                    "querier": querier_state_dict,
                    "classifier": classifier_state_dict,
                }
                torch.save(state_dict, os.path.join(weights_dir, f"epoch_{epoch+1}.pt"))
                with open(os.path.join(weights_dir, "latest.txt"), "w") as f:
                    f.write(f"epoch_{epoch+1}.pt")

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

    if dist:
        distributed.destroy_process_group()
