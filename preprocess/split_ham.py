import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(root_dir, "data")
ham_dir = os.path.join(data_dir, "HAM10000")

metadata_path = os.path.join(ham_dir, "metadata.csv")
metadata = pd.read_csv(metadata_path)

image = metadata["image_id"].values.tolist()
label = metadata["dx"].values.tolist()

stratified = {
    "akiec": [],
    "bcc": [],
    "bkl": [],
    "df": [],
    "nv": [],
    "mel": [],
    "vasc": [],
}
for i, l in zip(image, label):
    stratified[l].append(i)

print("Label distribution:")
for l, images in stratified.items():
    print(f"\t{l}: {len(images)}")

finetune_size = 55
finetune_image = []
remaining_image, remaining_label = [], []
for class_name, images in stratified.items():
    idx = np.random.choice(images, finetune_size, replace=False)
    finetune_image.extend(idx)

    remaining_image.extend(np.setdiff1d(images, idx).tolist())
    remaining_label.extend([class_name] * (len(images) - finetune_size))

with open(os.path.join(ham_dir, "finetune_images.txt"), "w") as f:
    for filename in finetune_image:
        f.write(filename + "\n")

train_ratio = 0.8
train_image, test_image, train_label, test_label = train_test_split(
    remaining_image,
    remaining_label,
    test_size=1 - train_ratio,
    stratify=remaining_label,
)

with open(os.path.join(ham_dir, "train_images.txt"), "w") as f:
    for filename in train_image:
        f.write(filename + "\n")

with open(os.path.join(ham_dir, "test_images.txt"), "w") as f:
    for filename in test_image:
        f.write(filename + "\n")
