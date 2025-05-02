import os

from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import find_classes, make_dataset
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(root_dir, "data")

image_dir = os.path.join(data_dir, "CUB", "images")
wnids, wnid_to_idx = find_classes(image_dir)

with open(os.path.join(data_dir, "CUB", "images.txt")) as f:
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    filename_to_idx = {filename: int(idx) for idx, filename in lines}

samples = make_dataset(image_dir, wnid_to_idx, extensions=".jpg")
filenames, class_idx = zip(*samples)
filenames = ["/".join(filename.split("/")[-2:]) for filename in filenames]

train_ratio = 0.8
train_filenames, test_filenames, train_class_idx, test_class_idx = train_test_split(
    filenames, class_idx, test_size=1 - train_ratio, stratify=class_idx
)

with open(os.path.join(data_dir, "CUB", "train_filenames.txt"), "w") as f:
    for filename in tqdm(train_filenames):
        f.write(filename + "\n")

with open(os.path.join(data_dir, "CUB", "test_filenames.txt"), "w") as f:
    for filename in tqdm(test_filenames):
        f.write(filename + "\n")
