import os
import sys

import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_dir = os.path.join(root_dir, "data")
imagenet_dir = os.path.join(data_dir, "ImageNet")
attribute_dir = os.path.join(imagenet_dir, "attributes")

train_image_attribute_path = os.path.join(attribute_dir, "train_image_attribute.npy")
val_image_attribute_path = os.path.join(attribute_dir, "val_image_attribute.npy")

train_image_attribute = np.load(train_image_attribute_path)
val_image_attribute = np.load(val_image_attribute_path)

prevalence = np.mean(train_image_attribute == 1, axis=0)
sorted_idx = np.argsort(prevalence)[::-1]
sorted_prevalence = prevalence[sorted_idx]

k = 400
sorted_idx = sorted_idx[:k]
with open(os.path.join(attribute_dir, "top_concepts.txt"), "w") as f:
    for idx in sorted_idx:
        f.write(f"{idx} {prevalence[idx]}\n")
