import os

import numpy as np

rng = np.random.default_rng()

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_dir = os.path.join(root_dir, "data")
results_dir = os.path.join(root_dir, "results")

imagenet_dir = os.path.join(data_dir, "ImageNet")

with open(os.path.join(imagenet_dir, "wnids_to_class.txt")) as f:
    lines = f.readlines()

k = 300
random_indices = rng.permutation(len(lines))[:k]
random_classes = [lines[i] for i in random_indices]
with open(os.path.join(imagenet_dir, "random_classes.txt"), "w") as f:
    for line in random_classes:
        f.write(line)
