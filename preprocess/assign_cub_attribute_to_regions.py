import os

import numpy as np
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(root_dir, "data")
cub_dir = os.path.join(data_dir, "CUB")

with open(os.path.join(cub_dir, "attributes", "attributes.txt"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    attributes = [attribute for _, attribute in lines]

with open(os.path.join(cub_dir, "classes.txt"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    classes = [line[1].split(".")[1] for line in lines]

with open(os.path.join(cub_dir, "class_regions.txt"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    class_regions = np.array([int(region) for _, region in lines])

class_attribute_path = os.path.join(
    cub_dir, "attributes", "class_attribute_labels_continuous.txt"
)
class_attribute_frequency = np.loadtxt(class_attribute_path, delimiter=" ")
class_attribute = class_attribute_frequency > np.mean(
    class_attribute_frequency, axis=0, keepdims=True
)

positive_out_path = os.path.join(
    cub_dir, "attributes", "attribute_positive_regions.txt"
)
negative_out_path = os.path.join(
    cub_dir, "attributes", "attribute_negative_regions.txt"
)

n_regions = 5
attribute_positive_regions = np.zeros((len(attributes), n_regions))
attribute_negative_regions = np.zeros((len(attributes), n_regions))
for attribute_idx, attribute in enumerate(tqdm(attributes)):
    positive_classes = np.where(class_attribute[:, attribute_idx] == 1)[0]
    negative_classes = np.where(class_attribute[:, attribute_idx] == 0)[0]
    assert (
        len(set(positive_classes.tolist() + negative_classes.tolist()))
        == class_attribute.shape[0]
    )
    positive_regions = class_regions[positive_classes]
    negative_regions = class_regions[negative_classes]

    positive_regions = np.unique(positive_regions, return_counts=True)
    negative_regions = np.unique(negative_regions, return_counts=True)

    attribute_positive_regions[
        attribute_idx, positive_regions[0] - 1
    ] = positive_regions[1]
    attribute_negative_regions[
        attribute_idx, negative_regions[0] - 1
    ] = negative_regions[1]

attribute_positive_regions = attribute_positive_regions / np.sum(
    attribute_positive_regions, axis=1, keepdims=True
)
attribute_negative_regions = attribute_negative_regions / np.sum(
    attribute_negative_regions, axis=1, keepdims=True
)
np.savetxt(positive_out_path, attribute_positive_regions, fmt="%.4f", delimiter=" ")
np.savetxt(negative_out_path, attribute_negative_regions, fmt="%.4f", delimiter=" ")
