import os

import numpy as np
import pandas as pd

workdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(workdir, "data")
skincon_dir = os.path.join(data_dir, "SKINCON")

metadata_path = os.path.join(skincon_dir, "image.csv")
metadata = pd.read_csv(metadata_path)

attributes = metadata.columns[2:-1]
metadata = metadata[metadata["Do not consider this image"] == 0]
image_attribute = np.array(metadata[attributes].values.tolist())

threshold = 50
n_positive = np.sum(image_attribute, axis=0)
attribute_mask = n_positive >= threshold
filtered_attributes = attributes[attribute_mask].values.tolist()

if "Brown(Hyperpigmentation)" in filtered_attributes:
    filtered_attributes[
        filtered_attributes.index("Brown(Hyperpigmentation)")
    ] = "Hyperpigmentation"

if "White(Hypopigmentation)" in filtered_attributes:
    filtered_attributes[
        filtered_attributes.index("White(Hypopigmentation)")
    ] = "Hypopigmentation"

with open(os.path.join(skincon_dir, "attributes.txt"), "w") as f:
    for attribute in filtered_attributes:
        f.write(f"{attribute}\n")
