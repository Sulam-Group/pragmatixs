import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers import MONET, CLIPClassifier, HAMBiomedCLIP
from datasets import CUB, HAM, SKINCON

workdir = os.path.dirname(__file__)
data_dir = os.path.join(workdir, "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# backbone = "ViT-L/14"
# backbone_safe = backbone.lower().replace("/", "_")
# classifier = CLIPClassifier(backbone, device=device)
# classifier = HAMBiomedCLIP.from_pretrained(workdir, device)
classifier = MONET(device=device)

# dataset = CUB(root=data_dir, train=False, transform=classifier.preprocess)
# dataset = HAM(root=data_dir, train=False, transform=classifier.preprocess)
dataset = SKINCON(data_dir, transform=classifier.preprocess)

results = classifier.predict(dataset)
# results.to_csv(os.path.join(workdir, "results", f"CUB_{backbone_safe}.csv"))
# results.to_csv(os.path.join(workdir, "results", f"HAM_biomedclip.csv"))
np.save(os.path.join(workdir, "results", "skincon_monet.npy"), results)

# confusion = confusion_matrix(results["label"], results["prediction"], normalize="true")
# accuracy = np.diag(confusion)

# sorted_idx = np.argsort(accuracy)[::-1]
# sorted_classes = [dataset.classes[idx] for idx in sorted_idx]
# sorted_accuracy = accuracy[sorted_idx]

# print("Results:")
# for class_name, acc in zip(sorted_classes, sorted_accuracy):
#     print(f"\t {class_name}: {acc:.2%}")
# print(f"Average accuracy: {accuracy.mean():.2%}")
