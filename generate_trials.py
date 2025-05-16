import argparse
import json
import os
import shutil
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs import Config
from configs import Constants as C
from configs import get_config
from datasets import get_dataset
from vip_utils.utils import get_run_name

rng = np.random.default_rng()

conditions = {
    "random": None,
    "vip": {"max_queries": 311},
    "speaker:literal": {"speaker.alpha": 0.0, "listener.type": "claim"},
    "speaker:pragmatic": {"speaker.alpha": 0.2, "listener.type": "claim"},
    "speaker:topic": {
        "speaker.alpha": 0.2,
        "listener.type": "topic",
        "listener.prior": [0, 0, 1 / 3, 1 / 3, 1 / 3, 0],
        "listener.temperature_scale": 4.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def attribute_to_human_readable(attribute: str) -> str:
    first, second = attribute.split("::")
    part = " ".join(first.split("_")[1:-1])
    desc = second.replace("_", " ")

    if part in ["primary", "size"]:
        attribute = f"is {desc}"
    elif part == "bill" and "head" in desc:
        attribute = f"{desc} bill"
    elif part == "wing" and "wings" in desc:
        attribute = desc.replace("-", " ")
    else:
        attribute = f"{desc} {part}"
    return attribute.capitalize().strip()


def init_experiment_dirs(config: Config, workdir=C.workdir):
    data_dir = os.path.join(workdir, "data", "human_evaluation")
    experiment_dir = os.path.join(data_dir, config.data.dataset)

    solutions_dir = os.path.join(experiment_dir, "solutions")
    try:
        shutil.rmtree(solutions_dir)
    except:
        pass
    os.makedirs(solutions_dir, exist_ok=True)

    trials_dir = os.path.join(experiment_dir, "trials")
    try:
        shutil.rmtree(trials_dir)
    except:
        pass
    os.makedirs(trials_dir, exist_ok=True)
    return experiment_dir, solutions_dir, trials_dir


def list_to_uid(list: Iterable, list_name: str, solutions_dir: str):
    solution = {v: str(uuid4()) for v in list}
    with open(os.path.join(solutions_dir, f"{list_name}.txt"), "w") as f:
        for k, v in solution.items():
            f.write(f"{k},{v}\n")
    return solution


def get_random_explanations(
    samples: Iterable[int] = None,
    explanation_length: int = None,
    num_claims: int = None,
):
    explanations = {}
    for idx in samples:
        explanation = np.zeros((explanation_length, 2), dtype=int)
        explanation[:, 0] = rng.choice(
            num_claims, size=explanation_length, replace=False
        )
        explanation[:, 1] = rng.random(size=explanation_length) >= 0.5
        explanations[idx] = explanation
    return explanations


def get_vip_explanations(
    config: Config = None,
    max_queries: int = None,
    samples: Iterable[int] = None,
    workdir=C.workdir,
):
    results_dir = os.path.join(workdir, "results", config.data.dataset.lower())

    run_name = get_run_name(config, max_queries, "biased")
    results_path = os.path.join(results_dir, f"{run_name}.pkl")

    results = pd.read_pickle(results_path)
    results.set_index("image_idx", inplace=True)

    explanations = {}
    for idx in samples:
        idx_results = results.iloc[idx]
        image_attribute = idx_results["image_attribute"]
        queries = idx_results["queries"]

        explanation_length = config.data.explanation_length
        explanation = np.zeros((explanation_length, 2), dtype=int)
        for i, query in enumerate(queries[:explanation_length]):
            explanation[i, 0] = query
            explanation[i, 1] = image_attribute[query]
        explanations[idx] = explanation
    return explanations


def get_speaker_explanations(
    config_name: str = None,
    config_dict: Mapping[str, Any] = None,
    samples: Iterable[int] = None,
    workdir=C.workdir,
):
    config = get_config(config_name, config_dict=config_dict)
    results = config.get_results(workdir=workdir)
    results["listener_prediction"] = results["action"].apply(lambda x: np.argmax(x))

    explanations = {}
    for idx in samples:
        idx_results = results.iloc[idx]
        explanation = idx_results["explanation"]
        print(
            f"{idx}: label {idx_results['label']}, listener prediction"
            f" {idx_results['listener_prediction']}"
        )
        explanation = explanation[1:, :]
        explanations[idx] = explanation
    return explanations


def main(args):
    config_name = args.config
    workdir = args.workdir

    config = get_config(config_name)

    dataset = get_dataset(config, train=False, return_attribute=True, workdir=workdir)
    classes, claims = dataset.classes, dataset.claims

    experiment_dir, solutions_dir, trials_dir = init_experiment_dirs(
        config, workdir=workdir
    )

    trials = pd.read_csv(os.path.join(experiment_dir, "trials.csv"))
    exp_class_idx = trials["label"].tolist()
    exp_samples = trials["idx"].tolist()

    exp_classes = [classes[idx].lower().replace(" ", "_") for idx in exp_class_idx]
    exp_class_samples = {
        class_name: [exp_samples[i]] for i, class_name in enumerate(exp_classes)
    }

    # with open(os.path.join(experiment_dir, "classes.txt"), "r") as f:
    #     experiment_class_idx = f.read().splitlines()
    #     experiment_class_idx = list(map(int, experiment_class_idx))
    #     exp_classes = [classes[idx] for idx in experiment_class_idx]
    #     exp_classes = [
    #         class_name.lower().replace(" ", "_") for class_name in exp_classes
    #     ]

    # exp_class_samples = {class_name: [] for class_name in exp_classes}
    # for idx, (_, label) in enumerate(tqdm(dataset.samples)):
    #     class_name = classes[label]
    #     class_name = class_name.lower().replace(" ", "_")
    #     if class_name in exp_class_samples:
    #         exp_class_samples[class_name].append(idx)

    # n_samples_per_class = [len(v) for v in exp_class_samples.values()]
    # min_samples = min(n_samples_per_class)
    # exp_class_samples = {
    #     class_name: rng.choice(v, size=min_samples, replace=False).tolist()
    #     for class_name, v in exp_class_samples.items()
    # }

    # exp_samples = []
    # exp_target_uid = {}
    # for class_name, class_idx in exp_class_samples.items():
    #     for idx in class_idx:
    #         exp_samples.append(idx)
    #         exp_target_uid[idx] = class_name

    manifest = {
        "trials": 20,
        "conditions": list(conditions.keys()),
        "samples": exp_class_samples,
    }
    json.dump(manifest, open(os.path.join(trials_dir, "manifest.json"), "w"))

    for condition in conditions.keys():
        condition_dir = os.path.join(trials_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        for class_name in exp_classes:
            condition_class_dir = os.path.join(condition_dir, class_name)
            os.makedirs(condition_class_dir, exist_ok=True)

            samples = exp_class_samples[class_name]
            if condition == "random":
                explanations = get_random_explanations(
                    samples=samples,
                    explanation_length=config.data.explanation_length,
                    num_claims=len(claims),
                )
            if condition == "vip":
                vip_config_dict = conditions[condition]
                max_queries = vip_config_dict["max_queries"]

                explanations = get_vip_explanations(
                    config=config,
                    max_queries=max_queries,
                    samples=samples,
                    workdir=workdir,
                )
            if "speaker" in condition:
                speaker_config_dict = conditions[condition]

                explanations = get_speaker_explanations(
                    config_name=config_name,
                    config_dict=speaker_config_dict,
                    samples=samples,
                    workdir=workdir,
                )

            for idx, explanation in explanations.items():
                explanation_claims = explanation[:, 0]
                explanation_cls = explanation[:, 1]

                explanation_claims = list(
                    map(
                        attribute_to_human_readable,
                        [claims[idx] for idx in explanation_claims],
                    )
                )
                explanation_cls = [
                    "yes" if claim_label else "no" for claim_label in explanation_cls
                ]

                trial = {
                    "features": [
                        {"feature": claim, "label": claim_label}
                        for claim, claim_label in zip(
                            explanation_claims, explanation_cls
                        )
                    ],
                    "target": class_name,
                }

                json.dump(
                    trial,
                    open(os.path.join(condition_class_dir, f"{idx}.json"), "w"),
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
