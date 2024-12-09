import os

import pandas as pd
import requests
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(root_dir, "data")
skincon_dir = os.path.join(data_dir, "SKINCON")
fitz_dir = os.path.join(skincon_dir, "Fitz17k")
image_dir = os.path.join(fitz_dir, "images")

metadata_path = os.path.join(fitz_dir, "fitzpatrick17k.csv")
metadata = pd.read_csv(metadata_path)


def download_image(row):
    image_id = row["md5hash"]
    url = row["url"]
    if not isinstance(url, str):
        return

    res = requests.get(url, headers={"User-Agent": "curl/7.64.1"}, stream=True)
    if res.status_code != 200:
        return

    image = Image.open(res.raw)
    image.save(os.path.join(image_dir, f"{image_id}.jpg"))


Parallel(n_jobs=4)(
    delayed(download_image)(row)
    for _, row in tqdm(metadata.iterrows(), total=len(metadata))
)
