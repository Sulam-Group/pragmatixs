import argparse
import os

import numpy as np
from sklearn.metrics import confusion_matrix

from classifiers import get_classifier
from configs import Constants as C
from configs import get_config
from datasets import get_dataset

device = C.device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="None")
    parser.add_argument("--workdir", type=str, default=C.workdir)
    return parser.parse_args()


def main(args):
    config_name = args.config
    workdir = args.workdir

    config = get_config(config_name)

    classifier = get_classifier(
        config, from_pretrained=True, workdir=workdir, device=device
    )

    dataset = get_dataset(
        config, train=False, transform=classifier.preprocess, workdir=workdir
    )

    results = classifier.predict(dataset)

    confusion = confusion_matrix(
        results["label"], results["prediction"], normalize="true"
    )
    accuracy = np.diag(confusion)

    sorted_idx = np.argsort(accuracy)[::-1]
    sorted_classes = [dataset.classes[idx] for idx in sorted_idx]
    sorted_accuracy = accuracy[sorted_idx]

    print("Results:")
    for class_name, acc in zip(sorted_classes, sorted_accuracy):
        print(f"\t{class_name:<20}: {acc:.2%}")
    print(f"Average accuracy: {accuracy.mean():.2%}")

    results_dir = os.path.join(workdir, "results", config.data.dataset.lower())
    os.makedirs(results_dir, exist_ok=True)

    classifier_safe = config.data.classifier.lower().replace(":", "_").replace("/", "_")
    results_path = os.path.join(results_dir, f"{classifier_safe}.csv")
    results.to_csv(results_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
