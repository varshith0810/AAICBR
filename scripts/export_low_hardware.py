import argparse
import json
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torchvision import models


def find_existing_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def load_model(model_path: Path, num_classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def export_dynamic_int8(model, out_dir: Path):
    """Export CPU-friendly int8 model. Falls back safely if API changes."""
    warnings.filterwarnings(
        "ignore",
        message=".*torch.ao.quantization is deprecated.*",
        category=DeprecationWarning,
    )
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quantized_path = out_dir / "breed_classifier_int8.pt"
    torch.save(quantized.state_dict(), quantized_path)
    return quantized_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model for low-hardware inference")
    parser.add_argument("--model_path", type=str, default="models/breed_classifier.pt")
    parser.add_argument("--classes_path", type=str, default="models/class_names.json")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--work_dir", type=str, default=".", help="Project/work directory used during training")
    parser.add_argument("--skip_onnx", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_candidates = [
        Path(args.model_path),
        work_dir / "models" / "breed_classifier.pt",
        Path("/content/models/breed_classifier.pt"),
    ]
    classes_candidates = [
        Path(args.classes_path),
        work_dir / "models" / "class_names.json",
        Path("/content/models/class_names.json"),
    ]

    model_path = find_existing_file(model_candidates)
    classes_path = find_existing_file(classes_candidates)

    if model_path is None or classes_path is None:
        print("ERROR: Required trained artifacts not found.")
        print("Expected files:")
        print("- breed_classifier.pt")
        print("- class_names.json")
        print("Looked in:")
        for p in model_candidates + classes_candidates:
            print(f"  - {p}")
        print("\nFix:")
        print("1) Ensure training completed successfully.")
        print("2) Pass explicit paths:")
        print("   python scripts/export_low_hardware.py --model_path /content/models/breed_classifier.pt --classes_path /content/models/class_names.json --out_dir /content/models")
        raise FileNotFoundError("Missing model/class files for export")

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = load_model(model_path, len(classes))

    quantized_path = export_dynamic_int8(model, out_dir)

    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)
    ts_path = out_dir / "breed_classifier_ts.pt"
    traced.save(str(ts_path))

    onnx_path = None
    if not args.skip_onnx:
        try:
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
        except ModuleNotFoundError as e:
            print(f"ONNX export skipped: missing dependency ({e}).")
            print("Install ONNX exporter deps with: pip install onnx onnxscript")
        except Exception as e:
            print(f"ONNX export skipped due to runtime error: {e}")

    print("Export complete:")
    print(f"- Source model: {model_path}")
    print(f"- Source classes: {classes_path}")
    print(f"- Quantized: {quantized_path}")
    print(f"- TorchScript: {ts_path}")
    if onnx_path and onnx_path.exists():
        print(f"- ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
