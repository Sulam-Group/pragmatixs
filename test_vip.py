import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from configs import Constants as C
from datasets import CUB
from vip_utils import load_cub


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    parser.add_argument("--device", type=str, default=C.device)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    max_queries = args.max_queries
    workdir = args.workdir
    device = args.device

    transform = T.Compose(
        [
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
        ]
    )
    dataset = CUB(os.path.join(workdir, "data"), train=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    concept_net, classifier, querier = load_cub(workdir=workdir, device=device)

    results_data = {
        "image_idx": np.arange(len(dataset)),
        "label": [],
        "logits": [],
        "queries": [],
    }
    for data in tqdm(dataloader):
        image, label = data

        image = image.to(device)
        label = label.to(device)

        image_attribute = concept_net.net(image)
        image_attribute = torch.where(image_attribute > 0.0, 1.0, -1.0)

        mask = torch.zeros(image.size(0), 312, device=device)
        logits, queries = [], []
        for i in range(max_queries):
            query = querier(image_attribute * mask, mask)
            chosen_query = query.argmax(dim=1)
            mask[np.arange(image.size(0)), chosen_query] = 1.0
            label_logits = classifier(image_attribute * mask)

            logits.append(label_logits)
            queries.append(chosen_query)

        logits = torch.stack(logits, dim=1)
        queries = torch.stack(queries, dim=1)

        results_data["label"].extend(label.cpu().tolist())
        results_data["logits"].extend(logits.cpu().tolist())
        results_data["queries"].extend(queries.cpu().tolist())

    df = pd.DataFrame(results_data)
    results_path = os.path.join(workdir, "results", "cub", "vip.pkl")
    df.to_pickle(results_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
