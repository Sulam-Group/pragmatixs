import base64
import json
import os

from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from datasets import SKINCON


class AttributeAnnotation(BaseModel):
    attribute: str
    label: float


class ImageAnnotation(BaseModel):
    annotations: list[AttributeAnnotation]


def tobase64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


workdir = "./"
data_dir = os.path.join(workdir, "data")
batch_dir = os.path.join(workdir, "batches")

dataset = SKINCON(data_dir)
attributes = dataset.attributes

with open(os.path.join(batch_dir, "prompt.txt"), "r") as f:
    prompt = f.read()
prompt = prompt.format(", ".join(map(str.lower, attributes)))

model_name = "gpt-4o-mini"
batch_name = f"{model_name.replace('-', '_')}_skincon.jsonl"
batch_file_path = os.path.join(batch_dir, batch_name)

batch_size = 500
samples = dataset.samples
n_batches = len(samples) // batch_size + 1

for batch_idx in range(n_batches):
    start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size

    batch_name = f"{model_name.replace('-', '_')}_skincon_{batch_idx}.jsonl"

with open(batch_file_path, "w") as f:
    for path, _ in dataset.samples:
        line = {
            "custom_id": path,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
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
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": to_strict_json_schema(ImageAnnotation),
                        "name": ImageAnnotation.__name__,
                        "strict": True,
                    },
                },
            },
        }
        json.dump(line, f)
        f.write("\n")

client = OpenAI()
batch_input_file = client.files.create(
    file=open(batch_file_path, "rb"), purpose="batch"
)

batch_input_file_id = batch_input_file.id
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": f"{model_name} SKINCON concept labeling"},
)
