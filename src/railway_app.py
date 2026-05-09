import json
import os
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import models, transforms

app = FastAPI(title="Cattle Breed Recognition Frontend + API")

MODEL = None
CLASSES = None
TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_from_bundle(bundle_path: Path):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    tmp_dir = Path(tempfile.gettempdir()) / "cattle_model_bundle"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Support tar bundles where files may be inside subfolders (e.g., models/...)
    model_candidates = list(tmp_dir.rglob("breed_classifier_int8.pt"))
    classes_candidates = list(tmp_dir.rglob("class_names.json"))

    if not model_candidates or not classes_candidates:
        extracted = [str(p.relative_to(tmp_dir)) for p in tmp_dir.rglob("*") if p.is_file()]
        raise FileNotFoundError(
            "Bundle must contain breed_classifier_int8.pt and class_names.json. "
            f"Extracted files: {extracted}"
        )

    model_path = model_candidates[0]
    classes_path = classes_candidates[0]

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(classes))
    base.eval()
    qmodel = torch.quantization.quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8)
    state = torch.load(model_path, map_location="cpu")
    qmodel.load_state_dict(state)
    qmodel.eval()
    return qmodel, classes


def get_model():
    global MODEL, CLASSES
    if MODEL is None:
        bundle = Path(os.getenv("MODEL_BUNDLE", "cattle_model_low_hw.tar.gz"))
        MODEL, CLASSES = _load_from_bundle(bundle)
    return MODEL, CLASSES


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body style='font-family:Arial;max-width:700px;margin:auto;padding:20px;'>
    <h2>Indian Cattle & Buffalo Breed Recognition</h2>
    <form action='/predict' method='post' enctype='multipart/form-data'>
      <label>Upload animal image:</label><br/><input type='file' name='file' required/><br/><br/>
      <label>Animal ID (optional):</label><br/><input type='text' name='animal_id'/><br/><br/>
      <label>GPS Coordinates (optional):</label><br/><input type='text' name='gps_coordinates'/><br/><br/>
      <button type='submit'>Recognize Breed</button>
    </form>
    </body></html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    file: UploadFile = File(...),
    animal_id: str = Form(default=""),
    gps_coordinates: str = Form(default=""),
):
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        model, classes = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")
    x = TFMS(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        vals, idxs = torch.topk(probs, 5)

    top = classes[idxs[0].item()]
    conf = vals[0].item() * 100
    rows = "".join([f"<li>{classes[i]}: {v*100:.2f}%</li>" for v, i in zip(vals.tolist(), idxs.tolist())])

    return f"""
    <html><body style='font-family:Arial;max-width:700px;margin:auto;padding:20px;'>
      <h2>Prediction Result</h2>
      <p><b>Predicted Breed:</b> {top}</p>
      <p><b>Confidence:</b> {conf:.2f}%</p>
      <p><b>Animal ID:</b> {animal_id or 'N/A'}</p>
      <p><b>GPS Coordinates:</b> {gps_coordinates or 'N/A'}</p>
      <h3>Top-5</h3><ul>{rows}</ul>
      <a href='/'>Try another image</a>
    </body></html>
    """
