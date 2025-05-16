import argparse
from typing import Iterable

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
datasets_with_topics = ["cub", "chexpert"]


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


def get_explanation_accuracy(
    explanation: torch.Tensor, image_attribute: torch.Tensor, claims: Iterable[str]
):
    explanation_claims = explanation[..., 0]
    explanation_claims_cls = explanation[..., 1]

    _image_attribute = torch.cat(
        [
            image_attribute,
            torch.zeros(image_attribute.size(0), 3, device=device),
        ],
        dim=-1,
    )
    target_cls = torch.gather(_image_attribute, -1, explanation_claims)

    claims_mask = explanation_claims < len(claims)
    target_cls_mask = target_cls != -1

    accuracy_mask = claims_mask * target_cls_mask
    correct_claims = accuracy_mask * (explanation_claims_cls == target_cls)
    return torch.sum(correct_claims, dim=-1) / torch.sum(accuracy_mask, dim=-1)


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
    if config.data.dataset.startswith("chexpert"):
        class_prompts = classes

    else:
        class_prompts = [f"A photo of a {c}" for c in classes]

    speaker = ClaimSpeaker.from_pretrained(
        config, classifier, claims, workdir=workdir, device=device
    )

    Listener = get_listener(config.listener.type)
    listener = Listener.from_pretrained(
        config, len(classes), claims, workdir=workdir, device=device
    )
    register_cls_attention_hook(listener)

    if config.data.dataset.lower() in datasets_with_topics:
        topic_listener = TopicListener(
            config, len(classes), claims, workdir=workdir, device=device
        )

    evaluation_data = {
        "idx": np.arange(len(dataset)).tolist(),
        "label": [],
        "prediction": [],
        "explanation": [],
        "explanation_consistency": [],
        "explanation_accuracy": [],
        "logp": [],
        "cls_attention": [],
        "action": [],
    }
    if config.data.dataset.lower() in datasets_with_topics:
        evaluation_data["explanation_topics"] = []

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    for _, data in enumerate(tqdm(dataloader)):
        image, label, image_attribute = data

        image = image.to(device)
        image_attribute = image_attribute.to(device)

        output = classifier(image, text=class_prompts)
        image_tokens = output["image_tokens"]
        logits = output["logits"]
        prediction = torch.argmax(logits, dim=-1)

        length = config.data.explanation_length
        if config.speaker.alpha > 0:
            length = None

        explanation, logp = speaker.explain(image_tokens, length=length)
        consistency, action = listener.listen(image_attribute, explanation)

        explanation_accuracy = get_explanation_accuracy(
            explanation, image_attribute, claims
        )
        if config.data.dataset.lower() in datasets_with_topics:
            explanation_topic = topic_listener.get_explanation_topic(explanation)
            explanation_topic = explanation_topic.cpu().numpy()

        label = label.numpy()
        prediction = prediction.cpu().numpy()
        explanation = explanation.squeeze().cpu().numpy()
        consistency = consistency.squeeze().cpu().numpy()
        explanation_accuracy = explanation_accuracy.squeeze().cpu().numpy()
        logp = logp.squeeze().cpu().numpy()
        action = action.squeeze().cpu().numpy()

        global cls_attn_weights
        # cls_attn_weights a list of tensors [batch_size, n_cls] 12 x 16 x 6
        _cls_attn_weights = torch.stack(cls_attn_weights, dim=1)
        _cls_attn_weights = torch.mean(_cls_attn_weights, dim=1)
        _cls_attn_weights = _cls_attn_weights.cpu().numpy()
        cls_attn_weights = []

        evaluation_data["label"].extend(label.tolist())
        evaluation_data["prediction"].extend(prediction.tolist())
        evaluation_data["explanation"].extend(explanation.astype(int))
        evaluation_data["explanation_consistency"].extend(consistency.tolist())
        evaluation_data["explanation_accuracy"].extend(explanation_accuracy.tolist())
        evaluation_data["logp"].extend(logp.tolist())
        evaluation_data["cls_attention"].extend(_cls_attn_weights)
        evaluation_data["action"].extend(action)
        if config.data.dataset.lower() in datasets_with_topics:
            evaluation_data["explanation_topics"].extend(explanation_topic)

    df = pd.DataFrame(evaluation_data)
    results_path = config.results_path(workdir=workdir)
    df.to_pickle(results_path)


def main(args):
    config_name = args.config
    workdir = args.workdir

    config = get_config(config_name)
    sweep_keys = ["data.explanation_length", "listener.gamma", "speaker.alpha"]
    if config.listener.type == "topic":
        sweep_keys += ["listener.temperature_scale"]
    for _config in config.sweep(sweep_keys):
        evaluate(_config, workdir=workdir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
