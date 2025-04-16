import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers import get_classifier
from configs import Config
from configs import Constants as C
from configs import get_config
from datasets import get_dataset
from listeners import Listener, TopicListener, get_listener
from speaker import ClaimSpeaker

device = C.device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def register_cls_attention_hook(listener: Listener):
    global cls_attn_weights
    cls_attn_weights = []

    def _hook(module, input, output):
        cls_attn_weights.append(output[1][:, -1])

    for resblock in listener.text.transformer.resblocks:
        resblock.attn.register_forward_hook(_hook)


@torch.no_grad()
def evaluate(config: Config, workdir=C.workdir):
    print(f"Evaluating {config.run_name()}")

    classifier = get_classifier(
        config, from_pretrained=True, workdir=workdir, device=device
    )

    dataset = get_dataset(
        config, train=False, transform=classifier.preprocess, return_attribute=True
    )

    classes, claims = dataset.classes, dataset.claims
    if config.data.dataset != 'chexpert':
        class_prompts = [f"A photo of a {c}" for c in classes]
    else:
        class_prompts = classes

    speaker = ClaimSpeaker.from_pretrained(
        config, classifier, claims, workdir=workdir, device=device
    )

    Listener = get_listener(config.listener.type)
    listener = Listener.from_pretrained(
        config, len(classes), claims, workdir=workdir, device=device
    )
    register_cls_attention_hook(listener)

    topic_listener = TopicListener(
        config, len(classes), claims, workdir=workdir, device=device
    )

    evaluation_data = {
        "idx": np.arange(len(dataset)).tolist(),
        "label": [],
        "prediction": [],
        "explanation": [],
        "explanation_topics": [],
        "consistency": [],
        "logp": [],
        "cls_attention": [],
        "action": [],
    }

    # if config.listener.type == "topic":
    #     evaluation_data["explanation_topics"] = []
    # if config.listener.type == "region":
    #     evaluation_data["explanation_regions"] = []

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    for _, data in enumerate(tqdm(dataloader)):
        image, label, image_attribute = data

        image = image.to(device)
        image_attribute = image_attribute.to(device)

        output = classifier(image, text=class_prompts)
        image_tokens = output["image_tokens"]
        logits = output["logits"]
        prediction = torch.argmax(logits, dim=-1)

        explanation, logp = speaker.explain(image_tokens)
        consistency, action = listener.listen(image_attribute, explanation)

        explanation_topic = topic_listener.get_explanation_topic(explanation)
        explanation_topic = explanation_topic.cpu().numpy()
        evaluation_data["explanation_topics"].extend(explanation_topic)
        # if config.listener.type == "region":
        #     explanation_regions = listener.get_explanation_regions(explanation)
        #     explanation_regions = explanation_regions.cpu().numpy()
        #     evaluation_data["explanation_regions"].extend(explanation_regions)

        label = label.numpy()
        prediction = prediction.cpu().numpy()
        explanation = explanation.squeeze().cpu().numpy()
        logp = logp.squeeze().cpu().numpy()
        consistency = consistency.squeeze().cpu().numpy()
        action = action.squeeze().cpu().numpy()

        global cls_attn_weights
        _cls_attn_weights = torch.stack(cls_attn_weights, dim=1)
        _cls_attn_weights = torch.mean(_cls_attn_weights, dim=1)
        _cls_attn_weights = _cls_attn_weights.cpu().numpy()
        cls_attn_weights = []

        evaluation_data["label"].extend(label.tolist())
        evaluation_data["prediction"].extend(prediction.tolist())
        evaluation_data["explanation"].extend(explanation.astype(int))
        evaluation_data["consistency"].extend(consistency.tolist())
        evaluation_data["logp"].extend(logp.tolist())
        evaluation_data["cls_attention"].extend(_cls_attn_weights)
        evaluation_data["action"].extend(action)

    df = pd.DataFrame(evaluation_data)
    results_path = config.results_path(workdir=workdir)
    df.to_pickle(results_path)


def main(args):
    config_name = args.config
    workdir = args.workdir

    config = get_config(config_name)
    sweep_keys = ["data.explanation_length", "listener.gamma", "speaker.alpha"]
    if config.listener.type == "region":
        sweep_keys += ["listener.prior", "listener.temperature_scale"]
    if config.listener.type == "topic":
        sweep_keys += ["listener.temperature_scale"]
    for _config in config.sweep(sweep_keys):
        evaluate(_config, workdir=workdir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
