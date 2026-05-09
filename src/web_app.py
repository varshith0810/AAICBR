import base64
import json
import os
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware
from torchvision import models, transforms

app = FastAPI(title="Cattle Breed Recognition Frontend + API")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-me-please"))

MODEL = None
CLASSES = None
MODEL_META = None
TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

BASE_STYLE = """
<style>
body{font-family:Inter,Arial,sans-serif;background:linear-gradient(120deg,#f5f7ff,#eefaf6);margin:0;color:#1f2937}
.container{max-width:960px;margin:40px auto;padding:24px}
.card{background:white;border-radius:18px;box-shadow:0 10px 30px rgba(17,24,39,.08);padding:24px}
.title{font-size:28px;font-weight:700;margin-bottom:8px}.subtitle{color:#6b7280;margin-bottom:20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}input,button{width:100%;padding:12px;border-radius:10px;border:1px solid #d1d5db}
button{background:#111827;color:#fff;font-weight:600;cursor:pointer}.error{color:#b91c1c;font-size:14px}
.badge{display:inline-block;background:#ecfeff;color:#155e75;border:1px solid #a5f3fc;padding:6px 10px;border-radius:999px;font-size:12px}
.result{margin-top:18px;padding:16px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px}
video{width:100%;max-width:420px;border-radius:12px;border:1px solid #d1d5db}.row{display:flex;gap:12px;align-items:center}
@media(max-width:768px){.grid{grid-template-columns:1fr}.row{flex-direction:column;align-items:stretch}}
</style>
"""


def render_login(error: str = ""):
    err = f"<p class='error'>{error}</p>" if error else ""
    return f"""<html><head>{BASE_STYLE}</head><body><div class='container'><div class='card'>
    <div class='badge'>Secure Login</div><div class='title'>Sign in</div><div class='subtitle'>Login to use breed recognition.</div>{err}
    <form action='/login' method='post'>
      <label>Username</label><input type='text' name='username' required>
      <label style='margin-top:10px;display:block'>Password</label><input type='password' name='password' required>
      <div style='margin-top:16px'><button type='submit'>Login</button></div>
    </form></div></div></body></html>"""


def render_home(user: str):
    return f"""<html><head>{BASE_STYLE}</head><body><div class='container'><div class='card'>
    <div class='row'><div><div class='badge'>Welcome, {user}</div><div class='title'>Indian Cattle & Buffalo Breed Classifier</div></div>
    <div style='margin-left:auto'><a href='/logout'><button>Logout</button></a></div></div>
    <div class='subtitle'>Upload image or capture from camera for prediction.</div>
    <form id='predict-form' action='/predict' method='post' enctype='multipart/form-data'>
      <div class='grid'>
        <div><label>Upload Animal Image</label><input type='file' name='file' id='fileInput'></div>
        <div><label>Animal ID (optional)</label><input type='text' name='animal_id' placeholder='COW-2024-0042'></div>
      </div>
      <div style='margin-top:12px'><label>GPS Coordinates (optional)</label><input type='text' name='gps_coordinates' placeholder='30.8717N, 75.8520E'></div>
      <input type='hidden' name='captured_image' id='capturedImage'>
      <div style='margin-top:14px'><button type='button' onclick='startCamera()'>Allow Camera</button></div>
      <div style='margin-top:12px'><video id='video' autoplay playsinline></video></div>
      <div class='row' style='margin-top:10px'>
        <button type='button' onclick='capturePhoto()'>Capture Photo</button>
        <button type='submit'>Predict Breed</button>
      </div>
    </form></div></div>
    <script>
      let stream;
      async function startCamera() {{
        try {{
          stream = await navigator.mediaDevices.getUserMedia({{ video: true }});
          document.getElementById('video').srcObject = stream;
        }} catch (e) {{ alert('Camera permission denied or unavailable: ' + e); }}
      }}
      function capturePhoto() {{
        const video = document.getElementById('video');
        if (!video.srcObject) {{ alert('Start camera first'); return; }}
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video,0,0);
        document.getElementById('capturedImage').value = canvas.toDataURL('image/jpeg');
        alert('Photo captured. Click Predict Breed.');
      }}
    </script>
    </body></html>"""


def render_result(top, conf, animal_id, gps_coordinates, rows):
    return f"""<html><head>{BASE_STYLE}</head><body><div class='container'><div class='card'>
    <div class='badge'>Prediction Complete</div><div class='title'>Breed Recognition Result</div>
    <div class='result'><p><b>Predicted Breed:</b> {top}</p><p><b>Confidence:</b> {conf:.2f}%</p>
    <p><b>Animal ID:</b> {animal_id or 'N/A'}</p><p><b>GPS Coordinates:</b> {gps_coordinates or 'N/A'}</p><h4>Top-5 Scores</h4><ul>{rows}</ul></div>
    <div style='margin-top:16px'><a href='/'><button>Try Another Image</button></a></div>
    </div></div></body></html>"""


def _load_from_bundle(bundle_path: Path):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    tmp_dir = Path(tempfile.gettempdir()) / "cattle_model_bundle"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(tmp_dir)
    all_files = [p for p in tmp_dir.rglob("*") if p.is_file()]
    int8_candidates = [p for p in all_files if p.name == "breed_classifier_int8.pt"]
    ts_candidates = [p for p in all_files if p.name == "breed_classifier_ts.pt"]
    classes_candidates = [p for p in all_files if p.name == "class_names.json"]
    if not classes_candidates or (not int8_candidates and not ts_candidates):
        extracted = [str(p.relative_to(tmp_dir)) for p in all_files]
        raise FileNotFoundError("Bundle missing required files. Extracted: " + str(extracted))
    with open(classes_candidates[0], "r", encoding="utf-8") as f:
        classes = json.load(f)
    if int8_candidates:
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, len(classes))
        qmodel = torch.quantization.quantize_dynamic(base.eval(), {nn.Linear}, dtype=torch.qint8)
        qmodel.load_state_dict(torch.load(int8_candidates[0], map_location="cpu"))
        return qmodel.eval(), classes, {"type": "int8"}
    model = torch.jit.load(str(ts_candidates[0]), map_location="cpu").eval()
    return model, classes, {"type": "torchscript"}


def _normalize_loaded(loaded):
    if isinstance(loaded, tuple):
        if len(loaded) == 3:
            return loaded[0], loaded[1], loaded[2]
        if len(loaded) == 2:
            return loaded[0], loaded[1], {"type": "unknown"}
    if isinstance(loaded, dict):
        return loaded.get("model"), loaded.get("classes"), loaded.get("meta", {"type": "unknown"})
    raise RuntimeError(f"Unexpected loader output: {type(loaded)}")


def get_model():
    global MODEL, CLASSES, MODEL_META
    if MODEL is None:
        loaded = _load_from_bundle(Path(os.getenv("MODEL_BUNDLE", "cattle_model_low_hw.tar.gz")))
        MODEL, CLASSES, MODEL_META = _normalize_loaded(loaded)
    return MODEL, CLASSES


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_type": (MODEL_META or {}).get("type")}


@app.get("/debug/bundle")
def debug_bundle():
    if os.getenv("DEBUG_BUNDLE", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Enable DEBUG_BUNDLE=true")
    bundle = Path(os.getenv("MODEL_BUNDLE", "cattle_model_low_hw.tar.gz"))
    return {"bundle_path": str(bundle), "exists": bundle.exists()}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = request.session.get("user")
    if not user:
        return render_login()
    return render_home(user)


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    expected_user = os.getenv("APP_USERNAME", "admin")
    expected_pass = os.getenv("APP_PASSWORD", "admin123")
    if username == expected_user and password == expected_pass:
        request.session["user"] = username
        return RedirectResponse(url='/', status_code=303)
    return render_login("Invalid credentials")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url='/', status_code=303)


@app.post("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    file: UploadFile = File(None),
    captured_image: str = Form(default=""),
    animal_id: str = Form(default=""),
    gps_coordinates: str = Form(default=""),
):
    if "user" not in request.session:
        return RedirectResponse(url='/', status_code=303)

    image = None
    try:
        if captured_image:
            b64 = captured_image.split(",", 1)[1] if "," in captured_image else captured_image
            image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        elif file is not None:
            image = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image input: {e}")

    if image is None:
        raise HTTPException(status_code=400, detail="Upload or capture an image first")

    model, classes = get_model()
    x = TFMS(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        vals, idxs = torch.topk(probs, 5)

    top = classes[idxs[0].item()]
    conf = vals[0].item() * 100
    rows = "".join([f"<li>{classes[i]}: {v*100:.2f}%</li>" for v, i in zip(vals.tolist(), idxs.tolist())])
    return render_result(top, conf, animal_id, gps_coordinates, rows)
