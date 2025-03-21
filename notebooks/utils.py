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
