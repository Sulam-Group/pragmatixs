import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from vip_utils.utils import get_run_name, load_vip, make_dataset


@torch.no_grad()
def test(
    config: Config,
    max_queries: int,
    tau: float,
    max_test_queries: int,
    workdir=C.workdir,
    device=C.device,
):
    dataset = make_dataset(config, train=False, workdir=workdir, device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    concept_net, querier, classifier = load_vip(
        config, max_queries, tau, workdir=workdir, device=device
    )

    results_data = {
        "image_idx": np.arange(len(dataset)),
        "prediction": [],
        "logits": [],
        "queries": [],
    }
    for data in tqdm(dataloader):
        image, prediction = data

        image = image.to(device)

        image_attribute = concept_net.net(image)
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
        results_data["queries"].extend(queries.cpu().tolist())

    run_name = get_run_name(config, max_queries, "biased")
    results_path = os.path.join(workdir, "results", f"{run_name}.pkl")

    df = pd.DataFrame(results_data)
    df.to_pickle(results_path)
