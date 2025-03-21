import argparse
import json
import os

import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from classifiers import MONET, HAMBiomedCLIP
from configs import Constants as C
from datasets import SKINCON

device = C.device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, default=None, help="Method to label dataset"
    )
    parser.add_argument(
        "--workdir", type=str, default=C.workdir, help="Working directory"
    )
    return parser.parse_args()


def predict_hambiomedclip(workdir):
    pass


def predict_monet(workdir):
    data_dir = os.path.join(workdir, "data")

    monet = MONET(workdir=workdir, device=device)
    dataset = SKINCON(data_dir, transform=monet.preprocess)
    image_attribute = monet.predict(dataset)
    return image_attribute


def predict_gpt(model_name, workdir):
    response_path = os.path.join(
        workdir, "results", "skincon", f"{model_name.replace('-', '_')}_responses.json"
    )

    if os.path.exists(response_path):
        responses = json.load(open(response_path, "r"))
    else:
        pass

    data_dir = os.path.join(workdir, "data")

    dataset = SKINCON(data_dir)
    filenames = [os.path.basename(path) for path, _ in dataset.samples]
    attributes = list(map(str.lower, dataset.attributes))

    image_attribute = -2 * np.ones((len(dataset), len(attributes)))
    for path, response in responses.items():
        filename = os.path.basename(path)
        image_idx = filenames.index(filename)

        for _, choice in response.items():
            annotations = choice["annotations"]
            for annotation in annotations:
                attribute = annotation["attribute"].strip().lower()
                label = annotation["label"]

                attribute_idx = attributes.index(attribute)
                image_attribute[image_idx, attribute_idx] = label

    assert np.all(image_attribute != -2)
    n_refusals = np.sum(image_attribute == -1)
    print(f"Number of refusals: {n_refusals} ({n_refusals / image_attribute.size:.2%})")
    return image_attribute


def main(args):
    method = args.method
    workdir = args.workdir

    if method == "hambiomedclip":
        image_attribute = predict_hambiomedclip(workdir)
    elif method == "monet":
        image_attribute = predict_monet(workdir)
    elif "gpt" in method:
        image_attribute = predict_gpt(method, workdir)
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    results_dir = os.path.join(workdir, "results", "skincon")
    os.makedirs(results_dir, exist_ok=True)

    np.save(
        os.path.join(
            workdir,
            "results",
            "skincon",
            f"{method.replace('-', '_')}_image_attribute.npy",
        ),
        image_attribute,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
