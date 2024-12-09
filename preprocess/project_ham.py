import os
import sys

import numpy as np
import torch

workdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(workdir)
from classifiers import MONET
from datasets import HAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join(workdir, "data")
skincon_dir = os.path.join(data_dir, "SKINCON")
ham_dir = os.path.join(data_dir, "HAM10000")

with open(os.path.join(skincon_dir, "attributes.txt"), "r") as f:
    attributes = f.readlines()
    attributes = [attribute.strip() for attribute in attributes]

classifier = MONET(device=device)
for op in ["finetune", "train", "test"]:
    print(f"Processing {op} set")
    dataset = HAM(
        data_dir,
        train=op == "train",
        finetune=op == "finetune",
        transform=classifier.preprocess,
    )

    results = classifier.predict(dataset)

    threshold = 0.5
    image_attribute = results >= threshold
    image_attribute = image_attribute.astype(int)

    np.save(
        os.path.join(ham_dir, "attributes", f"{op}_monet_image_attribute.npy"),
        image_attribute,
    )
