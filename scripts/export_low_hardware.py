import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def load_model(model_path: Path, num_classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Export trained model for low-hardware inference")
    parser.add_argument("--model_path", type=str, default="models/breed_classifier.pt")
    parser.add_argument("--classes_path", type=str, default="models/class_names.json")
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    classes_path = Path(args.classes_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = load_model(model_path, len(classes))

    # 1) Dynamic quantization (CPU-friendly)
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quantized_path = out_dir / "breed_classifier_int8.pt"
    torch.save(quantized.state_dict(), quantized_path)

    # 2) TorchScript export (fast loading/runtime)
    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)
    ts_path = out_dir / "breed_classifier_ts.pt"
    traced.save(str(ts_path))

    # 3) Optional ONNX export (widely portable)
    onnx_path = out_dir / "breed_classifier.onnx"
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )

    print("Export complete:")
    print(f"- Quantized: {quantized_path}")
    print(f"- TorchScript: {ts_path}")
    print(f"- ONNX: {onnx_path}")
    print("Keep class map file:")
    print(f"- {classes_path}")


if __name__ == "__main__":
    main()
