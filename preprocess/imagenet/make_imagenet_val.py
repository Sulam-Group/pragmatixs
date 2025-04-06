import os
import shutil

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_dir = os.path.join(root_dir, "data")

imagenet_dir = os.path.join(data_dir, "ImageNet")
raw_val_dir = os.path.join(imagenet_dir, "val_raw")
val_dir = os.path.join(imagenet_dir, "val")
os.makedirs(val_dir, exist_ok=True)

label_df = pd.read_csv(os.path.join(imagenet_dir, "LOC_val_solution.csv"))


def copy_image(row):
    filename = row["ImageId"]
    prediction_string = row["PredictionString"]

    wnid = list(filter(lambda x: x.startswith("n"), prediction_string.split(" ")))
    assert len(set(wnid)) == 1
    wnid = wnid[0]

    wnid_dir = os.path.join(val_dir, wnid)
    os.makedirs(wnid_dir, exist_ok=True)
    image_path = os.path.join(raw_val_dir, f"{filename}.JPEG")
    shutil.copy(image_path, wnid_dir)


Parallel(n_jobs=-1)(
    delayed(copy_image)(row)
    for _, row in tqdm(label_df.iterrows(), total=len(label_df))
)
