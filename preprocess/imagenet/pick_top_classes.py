import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from torchvision.datasets.folder import find_classes

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_dir = os.path.join(root_dir, "data")
results_dir = os.path.join(root_dir, "results")

imagenet_dir = os.path.join(data_dir, "ImageNet")

with open(os.path.join(imagenet_dir, "wnids_to_class.txt")) as f:
    lines = f.readlines()

wnids_to_class = {}
for line in lines:
    line = line.strip().replace(", ", ",")
    chunks = line.split()
    wnid, class_names = chunks[0], " ".join(chunks[1:])
    class_name = class_names.split(",")[0]
    wnids_to_class[wnid] = class_name

image_dir = os.path.join(imagenet_dir, "val")
wnids, wnid_to_idx = find_classes(image_dir)
classes = list(wnids_to_class.values())
idx_to_wnid = {idx: wnid for wnid, idx in wnid_to_idx.items()}

results_df = pd.read_csv(
    os.path.join(results_dir, "imagenet", "open_clip_vit-l-14.csv")
)
confusion = confusion_matrix(
    results_df["label"], results_df["prediction"], normalize="true"
)
class_accuracy = confusion.diagonal()

k = 300
sorted_idx = np.argsort(class_accuracy)[::-1][:k]
sorted_accuracy = class_accuracy[sorted_idx]
sorted_wnids = [idx_to_wnid[idx] for idx in sorted_idx]
sorted_classes = [classes[idx] for idx in sorted_idx]

with open(os.path.join(imagenet_dir, "top_classes.txt"), "w") as f:
    for wnid, class_name, acc in zip(sorted_wnids, sorted_classes, sorted_accuracy):
        f.write(f"{wnid} {class_name}\n")
