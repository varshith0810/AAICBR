import json
from pathlib import Path

import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

from src.config import Paths


class Predictor:
    def __init__(self, model_path: Path, class_path: Path):
        with open(class_path, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(self.classes))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = model.to(self.device).eval()

        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image, top_k: int = 3):
        x = self.tfms(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            vals, idxs = torch.topk(probs, k=min(top_k, len(self.classes)))

        return [
            {"breed": self.classes[i], "confidence": float(v)}
            for v, i in zip(vals.cpu().tolist(), idxs.cpu().tolist())
        ]


def load_predictor():
    paths = Paths()
    return Predictor(paths.model_dir / "breed_classifier.pt", paths.model_dir / "class_names.json")
