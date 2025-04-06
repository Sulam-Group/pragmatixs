import os

import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(root_dir, "data")

attribute_dir = os.path.join(data_dir, "CUB", "attributes")

with open(os.path.join(attribute_dir, "attributes.txt"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    attributes = [attribute for _, attribute in lines]

with open(os.path.join(attribute_dir, "topics.txt"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    topic_to_idx = {topic: int(idx) for idx, topic in lines}

attribute_topic = [-1 for _ in attributes]
for attribute_idx, attribute in enumerate(attributes):
    attribute_name = attribute.split("::")[0]
    # Bill features
    if "bill" in attribute_name:
        attribute_topic[attribute_idx] = topic_to_idx["bill_features"]
    # Tail features
    elif "tail" in attribute_name:
        attribute_topic[attribute_idx] = topic_to_idx["tail_features"]
    # Head features
    elif (
        "head" in attribute_name
        or "crown" in attribute_name
        or "nape" in attribute_name
    ):
        attribute_topic[attribute_idx] = topic_to_idx["head_features"]
    # Coloration
    elif "color" in attribute_name:
        attribute_topic[attribute_idx] = topic_to_idx["coloration"]
    # Pattern features
    elif "pattern" in attribute_name:
        attribute_topic[attribute_idx] = topic_to_idx["patterns"]
    # Shape and size
    else:
        attribute_topic[attribute_idx] = topic_to_idx["shape_and_size"]

assert -1 not in attribute_topic

with open(os.path.join(attribute_dir, "attribute_topic.txt"), "w") as f:
    for attribute, topic in zip(attributes, attribute_topic):
        f.write(f"{attribute} {topic}\n")
