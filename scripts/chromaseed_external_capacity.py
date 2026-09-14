"""Count architecture storage only; no training, images or weight downloads."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b4


def inspect_capacity():
    rows = []
    # Published torchvision ImageNet parameter counts, checked before head changes.
    specs = [
        ("efficientnet_b0", efficientnet_b0, 5_288_548),
        ("efficientnet_b4", efficientnet_b4, 19_341_616),
    ]
    for name, builder, expected_original in specs:
        with torch.device("meta"):
            model = builder(weights=None)
            original = sum(p.numel() for p in model.parameters())
            assert original == expected_original
            features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(features, 3)
        count = sum(p.numel() for p in model.parameters())
        arithmetic = original - (features + 1) * 1000 + (features + 1) * 3
        assert count == arithmetic
        buffers = sum(b.numel() * b.element_size() for b in model.buffers())
        rows.append({
            "architecture": name + "_lab3",
            "parameters": count,
            "fp32_parameter_bytes": count * 4,
            "state_dict_numeric_bytes": count * 4 + buffers,
            "original_imagenet_parameters": original,
            "head_in_features": features,
            "arithmetic_count_verified": True,
        })
    return {
        "scope": "architecture capacity estimate, not a trained checkpoint measurement",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": "meta",
        "images_read": 0,
        "models_trained": 0,
        "pretrained_weights_downloaded": False,
        "byte_convention": "decimal bytes; excludes executable, allocator and activations",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_capacity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
