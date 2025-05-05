import numpy as np
import seaborn as sns


def attribute_to_human_readable(attribute: str) -> str:
    first, second = attribute.split("::")
    part = " ".join(first.split("_")[1:-1])
    desc = second.replace("_", " ")

    if part in ["primary", "size"]:
        return f"is {desc}"
    if part == "bill" and "head" in desc:
        return f"bill {desc}"
    if part == "wing" and "wings" in desc:
        return desc.replace("-", " ")
    else:
        return f"{desc} {part}"


def viz_explanation(dataset, results, idx, ax):
    _, _, image_attribute = dataset[idx]

    classes = dataset.classes
    claims = dataset.claims

    vocab_size = len(claims) + 3
    special_tokens = {
        vocab_size - 3: "[BOS]",
        vocab_size - 2: "[EOS]",
        vocab_size - 1: "[PAD]",
    }

    idx_results = results.iloc[idx]
    explanation = idx_results["explanation"]
    cls_attn_weights = idx_results["cls_attention"]
    cls_attn_weights = cls_attn_weights[:-1]
    cls_attn_weights /= np.sum(cls_attn_weights)

    image_attribute[image_attribute == -1] = 0
    image_attribute = image_attribute * 2 - 1
    explanation_claims = [
        (
            f"{attribute_to_human_readable(claims[claim])} ({cls*2-1}/{image_attribute[claim]:.0f})"
            if claim < len(claims)
            else special_tokens[claim]
        )
        for claim, cls in explanation
    ]

    explanation_claims = explanation_claims[1:]
    cls_attn_weights = cls_attn_weights[1:]

    y = list(map(str, range(len(explanation_claims))))
    sns.barplot(x=cls_attn_weights, y=y, ax=ax)
    ax.set_xlabel("Attention weights")
    ax.set_yticks(y)
    ax.set_yticklabels(explanation_claims)
    ax.set_title(f"Listener prediction:\n{classes[idx_results['listener_prediction']]}")
