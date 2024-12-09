import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from classifiers import CLIPClassifier
from datasets import CUB
from listener_model import LISTENERS
from speaker_model import ClaimSpeaker

config = configs.CUBTopicConfig(beta=0.4, gamma=0.0, alpha=0.0, ignore_topics=[4])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

workdir = "./"
data_dir = os.path.join(workdir, "data")

backbone = "ViT-L/14"
classifier = CLIPClassifier(backbone, device=device)
classifier.eval()

dataset = CUB(
    root=data_dir, train=False, transform=classifier.preprocess, return_attribute=True
)
classes = dataset.classes
claims = dataset.claims

class_prompts = [f"A photo of a {class_name}" for class_name in classes]

speaker = ClaimSpeaker.from_pretrained(
    config, classifier, len(classes), claims, device, workdir
)

listener_type = config.listener_type
Listener = LISTENERS[listener_type]
listener = Listener.from_pretrained(config, len(classes), claims, device, workdir)

context_length = config.context_length
evaluation_data = {
    "idx": -np.ones(len(dataset)),
    "label": -np.ones(len(dataset)),
    "prediction": -np.ones(len(dataset)),
    "explanation": -np.ones((len(dataset), context_length)),
    "explanation_topics": -np.ones((len(dataset), context_length)),
    "explanation_logp": -np.ones(len(dataset)),
    "explanation_consistency": -np.ones(len(dataset)),
    "listener_action": -np.ones((len(dataset), len(classes))),
}

with open(os.path.join(data_dir, "CUB", "attributes", "attribute_topic.txt"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    attribute_to_topic = {attribute: int(idx) for attribute, idx in lines}

claim_topic = -np.ones(speaker.vocab_size)
for idx, claim in enumerate(claims):
    if claim in attribute_to_topic:
        claim_topic[idx] = attribute_to_topic[claim]

dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
start = 0
for _, data in enumerate(tqdm(dataloader)):
    image, label, image_attribute = data

    image = image.to(device)

    with torch.no_grad():
        cls_output = classifier(image, class_prompts)
        image_features = cls_output["image_features"]
        logits = cls_output["logits"]
        prediction = torch.argmax(logits, dim=-1)

        image_features = image_features.unsqueeze(1).expand(-1, context_length, -1)

        explanation, explanation_logp = speaker.explain(image_features, prediction)
        explanation_consistency, action = listener.listen(image_attribute, explanation)

    prediction = prediction.cpu().numpy()
    explanation = explanation.squeeze().cpu().numpy()
    explanation_logp = explanation_logp.squeeze().cpu().numpy()
    explanation_consistency = explanation_consistency.squeeze().cpu().numpy()
    action = action.squeeze().cpu().numpy()

    explanation_topic = claim_topic[explanation]

    end = start + image.size(0)

    evaluation_data["idx"][start:end] = np.arange(start, end)
    evaluation_data["label"][start:end] = label.numpy()
    evaluation_data["prediction"][start:end] = prediction
    evaluation_data["explanation"][start:end] = explanation
    evaluation_data["explanation_topics"][start:end] = explanation_topic
    evaluation_data["explanation_logp"][start:end] = explanation_logp
    evaluation_data["explanation_consistency"][start:end] = explanation_consistency
    evaluation_data["listener_action"][start:end] = action

    start = end

evaluation_data = {k: v.tolist() for k, v in evaluation_data.items()}
df = pd.DataFrame(evaluation_data)

results_dir = os.path.join(workdir, "results")
os.makedirs(results_dir, exist_ok=True)
df.to_parquet(os.path.join(results_dir, f"{config.run_name()}.parquet"))
