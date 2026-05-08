"""
Single-file Google Colab pipeline for Indian cattle & buffalo breed recognition.
Uses ONLY already-unzipped dataset folder from Google Drive.

Colab quick run:
1) Mount Google Drive.
2) !pip install -r requirements.txt
3) !python colab_breed_recognition.py --mode all --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
4) !python colab_breed_recognition.py --mode app --work_dir /content --dataset_dir "/content/drive/MyDrive/datasets/breeds"
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

BREEDS = [
    "vechur", "umblachery", "toda", "tharparkar", "surti", "sahiwal", "redsindhi", "reddane",
    "rathi", "pulikulam", "ongole", "nimari", "niliravi", "nagpuri", "nagori", "murrah",
    "mehsana", "malnadgidda", "krishnavalley", "khillari", "kherigarh", "kenkatha", "kasargod",
    "kankrej", "kangayam", "jersey", "jaffrabadi", "holsteinfriesian", "hariana", "hallikar",
    "guernsey", "gir", "deoni", "dangi", "brownswiss", "bhadawari", "bargur", "banni",
    "ayrshire", "amritmahal", "alambadi"
]


def validate_structure(breeds_root: Path) -> dict:
    report = {"missing_splits": [], "missing_breeds": {}, "counts": {}}
    for split in ["train", "test"]:
        split_dir = breeds_root / split
        if not split_dir.exists():
            report["missing_splits"].append(split)
            continue
        existing = {d.name.lower() for d in split_dir.iterdir() if d.is_dir()}
        report["missing_breeds"][split] = sorted(set(BREEDS) - existing)
        per_breed = Counter()
        for d in split_dir.iterdir():
            if d.is_dir():
                per_breed[d.name.lower()] = len([f for f in d.iterdir() if f.is_file()])
        report["counts"][split] = dict(per_breed)
    return report


def resolve_breeds_root(dataset_dir: str) -> Path:
    """Resolve dataset location from already-unzipped Google Drive folder only."""
    if not dataset_dir:
        raise ValueError("--dataset_dir is required. Example: /content/drive/MyDrive/datasets/breeds")

    d = Path(dataset_dir)
    if (d / "train").exists() and (d / "test").exists():
        return d
    if (d / "breeds").exists() and (d / "breeds" / "train").exists() and (d / "breeds" / "test").exists():
        return d / "breeds"

    raise FileNotFoundError(
        f"Invalid dataset_dir: {d}. Expected either <dir>/train & <dir>/test or <dir>/breeds/train & <dir>/breeds/test"
    )


def get_dataloaders(breeds_root: Path, image_size: int = 224, batch_size: int = 32):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(breeds_root / "train", transform=train_tfms)
    test_ds = datasets.ImageFolder(breeds_root / "test", transform=test_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, train_ds.classes


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_model(breeds_root: Path, out_dir: Path, epochs: int = 8, lr: float = 1e-3, batch_size: int = 32):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_dataloaders(breeds_root, batch_size=batch_size)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"epoch={epoch+1} loss={avg_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "breed_classifier.pt")

    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(classes, f)
    print(f"Training completed. Best validation accuracy = {best_acc:.4f}")


def load_predictor(model_dir: Path):
    with open(model_dir / "class_names.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(model_dir / "breed_classifier.pt", map_location=device))
    model = model.to(device).eval()

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, classes, tfms, device


def predict_with_fields(image, animal_id, gps_coordinates, model, classes, tfms, device):
    if image is None:
        return {"error": "Please upload an animal image."}, None

    start = time.perf_counter()
    x = tfms(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    output = {
        "Predicted Breed": classes[idx.item()],
        "Confidence Score": f"{float(conf.item()*100.0):.2f}%",
        "Classification Time": f"{elapsed_ms:.1f} ms",
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "GPS Coordinates": gps_coordinates.strip() if gps_coordinates else "N/A",
        "Animal ID": animal_id.strip() if animal_id else "N/A",
        "Alert Flag": "LOW CONFIDENCE (<70%)" if float(conf.item()*100.0) < 70 else "OK",
    }
    return output, image


def launch_app(model_dir: Path):
    model, classes, tfms, device = load_predictor(model_dir)

    with gr.Blocks(title="Indian Cattle & Buffalo Breed Recognition") as demo:
        gr.Markdown("## AI Breed Recognition (Software-Only)")
        gr.Markdown("Upload the image of the animal, optionally provide Animal ID/GPS, then click Recognize Breed.")

        with gr.Row():
            in_image = gr.Image(type="pil", label="Upload Animal Image")
            preview = gr.Image(type="pil", label="Image Preview", interactive=False)

        with gr.Row():
            animal_id = gr.Textbox(label="Animal ID (Optional)", placeholder="COW-2024-0042")
            gps = gr.Textbox(label="GPS Coordinates (Optional)", placeholder="30.8717°N, 75.8520°E")

        output_json = gr.JSON(label="Recognition Output Fields")
        gr.Button("Recognize Breed").click(
            fn=lambda img, aid, g: predict_with_fields(img, aid, g, model, classes, tfms, device),
            inputs=[in_image, animal_id, gps],
            outputs=[output_json, preview],
        )

    demo.launch()


def ask_dataset_dir_if_missing(dataset_dir: str) -> str:
    if dataset_dir:
        return dataset_dir
    print("Enter dataset directory path (must contain train/test or breeds/train+test):")
    return input().strip()


def interactive_predict_single_image(model_dir: Path):
    model, classes, tfms, device = load_predictor(model_dir)
    print("Training completed. Enter full image path to predict breed (or press Enter to skip):")
    image_path = input().strip()
    if not image_path:
        print("Skipped single-image prediction.")
        return
    img = Image.open(image_path).convert("RGB")
    out, _ = predict_with_fields(img, "", "", model, classes, tfms, device)
    print("Prediction output:")
    print(json.dumps(out, indent=2))


def main(args):
    work_dir = Path(args.work_dir)
    model_dir = work_dir / "models"
    dataset_dir = ask_dataset_dir_if_missing(args.dataset_dir)
    breeds_root = resolve_breeds_root(dataset_dir)

    if args.mode in {"preprocess", "all"}:
        report = validate_structure(breeds_root)
        print("Validation report:")
        print(json.dumps(report, indent=2))

    if args.mode in {"train", "all"}:
        train_model(breeds_root, model_dir, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        if args.mode == "all":
            interactive_predict_single_image(model_dir)

    if args.mode == "app":
        launch_app(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preprocess", "train", "app", "all"], default="all")
    parser.add_argument("--dataset_dir", type=str, default="", help="Path to already-unzipped breeds folder from Google Drive")
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    main(parser.parse_args())
