import base64
import json
import os

import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from datasets import SKINCON

client = OpenAI()


class Annotation(BaseModel):
    attribute: str
    label: float


class ImageAnnotation(BaseModel):
    annotations: list[Annotation]


def tobase64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


workdir = "./"
data_dir = os.path.join(workdir, "data")
ham_dir = os.path.join(data_dir, "HAM10000")
results_dir = os.path.join(workdir, "results")

model = "gpt-4o-mini"
response_path = os.path.join(results_dir, f"{model.replace('-', '_')}_responses.json")

if os.path.exists(response_path):
    responses = json.load(open(response_path, "r"))
else:
    dataset = SKINCON(data_dir)
    attributes = dataset.attributes

    prompt = (
        "This is a dermatoscopy image of a skin lesion. "
        + "Please answer to the best of your ability whether the following attributes"
        " are present in the image: "
        + ", ".join(map(str.lower, attributes))
        + ". "
        + "Write your answer as a list with a number between 0 and 1 that represents"
        " the likelihood of the presence of the attribute. "
        + "For example, if you think the attribute is definitely not present, you can"
        " write 0. If you think the attribute is definitely present, you can write 1."
        " If you are unsure, you can write a number between 0 and 1. "
        + "If you prefer to abstain from answering for a particular attribute, you can"
        " write -1. "
    )

    responses = {}
    for i, (path, _) in enumerate(tqdm(dataset.samples)):
        res = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{tobase64(path)}"
                            },
                        },
                    ],
                },
            ],
            response_format=ImageAnnotation,
        )

        for choice_idx, choice in enumerate(res.choices):
            if path not in responses:
                responses[path] = {}

            message = choice.message
            parsed = message.parsed

            if parsed:
                responses[path][choice_idx] = parsed.model_dump()
            else:
                responses[path][choice_idx] = message.refusal

    json.dump(responses, open(response_path, "w"))
