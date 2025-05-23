import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from vip_utils.network import Network
from vip_utils.utils import get_query_answerer, get_run_name, make_dataset


def get_vip_networks(
    config: Config = None,
    num_classes: int = None,
    num_claims: int = None,
    max_queries: int = None,
    tau: float = None,
    workdir=C.workdir,
    device=C.device,
):
    querier = Network(query_size=num_claims, output_size=num_claims, tau=tau)
    classifier = Network(query_size=num_claims, output_size=num_classes, tau=None)

    querier = querier.to(device)
    classifier = classifier.to(device)

    run_name = get_run_name(config, max_queries, "biased")
    weights_dir = os.path.join(
        workdir, "weights", config.data.dataset.lower(), run_name
    )

    with open(os.path.join(weights_dir, "latest.txt"), "r") as f:
        latest_weights = f.read().strip()
    state_dict = torch.load(
        os.path.join(weights_dir, latest_weights), map_location=device
    )

    querier_state_dict = state_dict["querier"]
    classifier_state_dict = state_dict["classifier"]
    querier.load_state_dict(querier_state_dict)
    classifier.load_state_dict(classifier_state_dict)

    querier.eval()
    classifier.eval()
    return querier, classifier


@torch.no_grad()
def test(
    config: Config,
    max_queries: int,
    tau: float,
    max_test_queries: int,
    workdir=C.workdir,
    device=C.device,
):
    preprocess, answer_query = get_query_answerer(
        config=config, workdir=workdir, device=device
    )

    dataset = make_dataset(
        config, False, transform=preprocess, workdir=workdir, device=device
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    num_classes = len(dataset.classes)
    num_claims = len(dataset.claims)
    querier, classifier = get_vip_networks(
        config=config,
        num_classes=num_classes,
        num_claims=num_claims,
        max_queries=max_queries,
        tau=tau,
        workdir=workdir,
        device=device,
    )

    results_data = {
        "image_idx": np.arange(len(dataset)),
        "prediction": [],
        "logits": [],
        "image_attribute": [],
        "queries": [],
    }
    for data in tqdm(dataloader):
        image, prediction = data

        image = image.to(device)

        image_attribute = answer_query(image)
        image_attribute = torch.where(image_attribute > 0.0, 1.0, -1.0)

        mask = torch.zeros(image.size(0), len(dataset.claims), device=device)
        logits, queries = [], []
        for _ in range(max_test_queries):
            query = querier(image_attribute * mask, mask)
            chosen_query = query.argmax(dim=-1)
            mask[np.arange(image.size(0)), chosen_query] = 1.0
            _logits = classifier(image_attribute * mask)

            logits.append(_logits)
            queries.append(chosen_query)

        logits = torch.stack(logits, dim=1)
        queries = torch.stack(queries, dim=1)

        results_data["prediction"].extend(prediction.tolist())
        results_data["logits"].extend(logits.cpu().tolist())
        results_data["image_attribute"].extend(image_attribute.cpu().tolist())
        results_data["queries"].extend(queries.cpu().tolist())

    results_dir = os.path.join(workdir, "results", config.data.dataset.lower())
    os.makedirs(results_dir, exist_ok=True)

    run_name = get_run_name(config, max_queries, "biased")
    results_path = os.path.join(results_dir, f"{run_name}.pkl")

    df = pd.DataFrame(results_data)
    df.to_pickle(results_path)
