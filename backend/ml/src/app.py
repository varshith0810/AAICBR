import base64
import html
import json
import os
import tarfile
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware
from torchvision import models, transforms

APP_TITLE = "Cattle Breed Recognition Frontend + API"
DEFAULT_MODEL_BUNDLE = "cattle_model_low_hw.tar.gz"
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me")
DEBUG_BUNDLE_FLAG = "DEBUG_BUNDLE"
app = FastAPI(title=APP_TITLE)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
MODEL: torch.nn.Module | None = None
CLASSES: list[str] | None = None
MODEL_META: dict[str, Any] | None = None
USERS: dict[str, dict[str, str]] = {}

TFMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

BASE_STYLE = """
<style>
  body{font-family:Inter,Arial,sans-serif;background:#0b1020;color:#e5e7eb;margin:0}
  .shell{max-width:1100px;margin:0 auto;padding:24px}
  .card{background:#121a31;border:1px solid #263252;border-radius:16px;padding:20px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:18px}
  .nav a{color:#c7d2fe;text-decoration:none;margin-right:10px}
  .result{background:#f8fafc;color:#0f172a;border-radius:12px;padding:12px}
  .img-preview{max-width:360px;width:100%;border-radius:12px;border:1px solid #cbd5e1}
  input,button{width:100%;padding:10px;border-radius:10px;border:1px solid #334155;box-sizing:border-box}
  button{background:#6366f1;color:#fff;border:none;cursor:pointer;font-weight:700}
  label{display:block;margin-bottom:6px}
  .muted{color:#94a3b8}
  .flash{padding:10px 12px;border-radius:10px;margin-bottom:12px;background:#1f2937;color:#e5e7eb}
  .flash.error{background:#7f1d1d;color:#fee2e2}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
</style>
"""
@dataclass(frozen=True)
class PredictionResult:
    breed: str
    confidence: float
    rows_html: str
def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)
def _current_user(request: Request) -> str | None:
    user = request.session.get("user")
    return str(user) if user else None
def _normalize_identity(identity: str) -> str:
    return identity.strip().lower()
def _model_bundle_path() -> Path:
    return Path(os.getenv("MODEL_BUNDLE", DEFAULT_MODEL_BUNDLE))
def _flash_html(message: str, *, is_error: bool = False) -> str:
    if not message:
        return ""
    css_class = "flash error" if is_error else "flash"
    return f"<div class='{css_class}'>{_escape(message)}</div>"
def page_template(content: str, request: Request) -> str:
    user = _current_user(request)
    auth_links = "<a href='/logout'>Logout</a>" if user else "<a href='/signin'>Sign In</a> <a href='/create-account'>Create Account</a>"
    signed_in = f"<span class='muted'>Signed in as {_escape(user)}</span>" if user else ""
    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>{_escape(APP_TITLE)}</title>
  {BASE_STYLE}
</head>
<body>
  <main class='shell'>
    <nav class='nav'><a href='/'>Home</a>{auth_links}{signed_in}</nav>
    {content}
  </main>
</body>
</html>"""
def render_home(request: Request):
    if not _current_user(request):
        return RedirectResponse(url="/signin", status_code=303)
    content = """
    <section class='card'>
      <h2>Indian Cattle & Buffalo Breed Classifier</h2>
      <p class='muted'>Prediction is available only after sign in.</p>
      <form action='/predict' method='post' enctype='multipart/form-data'>
        <div class='grid'>
          <div><label>Upload Animal Image</label><input type='file' name='file' accept='image/*' required></div>
          <div><label>Animal ID (optional)</label><input type='text' name='animal_id' placeholder='COW-2026-001'></div>
        </div>
        <div style='margin-top:10px'>
          <label>GPS Coordinates (lat,long)</label>
          <input type='text' name='gps_coordinates' placeholder='30.8717,75.8520'>
        </div>
        <div style='margin-top:12px'><button type='submit'>Predict Breed</button></div>
      </form>
    </section>"""
    return HTMLResponse(page_template(content, request))
def render_signin(request: Request, message: str = "", *, is_error: bool = False) -> str:
    content = f"""
    <section class='card'>
      <h2>Sign In</h2>
      {_flash_html(message, is_error=is_error)}
      <form action='/signin' method='post'>
        <label>Email or Username</label><input name='identity' required>
        <label style='margin-top:8px'>Password</label><input type='password' name='password' required>
        <div style='margin-top:12px'><button type='submit'>Sign In</button></div>
      </form>
      <p>New user? <a href='/create-account'>Create account</a></p>
    </section>"""
    return page_template(content, request)
def render_create_account(request: Request, message: str = "", *, is_error: bool = False) -> str:
    content = f"""
    <section class='card'>
      <h2>Create Account</h2>
      {_flash_html(message, is_error=is_error)}
      <form action='/create-account' method='post'>
        <label>Email</label><input type='email' name='email' required>
        <label style='margin-top:8px'>Username</label><input name='username' required>
        <label style='margin-top:8px'>Password</label><input type='password' name='password' required>
        <div style='margin-top:12px'><button type='submit'>Create Account</button></div>
      </form>
    </section>"""
    return page_template(content, request)
def render_result(
    request: Request,
    prediction: PredictionResult,
    animal_id: str,
    location_label: str,
    image_b64: str,
) -> str:
    content = f"""
    <section class='card'>
      <h2>Prediction Result</h2>
      <div class='result'>
        <p><b>Predicted Breed:</b> {_escape(prediction.breed)}</p>
        <p><b>Confidence:</b> {prediction.confidence:.2f}%</p>
        <p><b>Animal ID:</b> {_escape(animal_id.strip() or 'N/A')}</p>
        <p><b>Detected Location:</b> {_escape(location_label)}</p>
        <h4>Top Scores</h4>
        <ul>{prediction.rows_html}</ul>
      </div>
      <div style='margin-top:12px'>
        <h4>Uploaded Image</h4>
        <img class='img-preview' src='data:image/jpeg;base64,{image_b64}' alt='Uploaded animal'>
      </div>
    </section>"""
    return page_template(content, request)
def resolve_location_label(gps_coordinates: str) -> str:
    gps = gps_coordinates.strip()
    if not gps:
        return "N/A"
    try:
        lat_text, lon_text = [part.strip() for part in gps.split(",", 1)]
        lat = float(lat_text)
        lon = float(lon_text)
    except ValueError:
        return f"Invalid GPS format: {gps}. Use 'lat,long'."
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return f"Invalid GPS range: {gps}. Latitude must be -90..90 and longitude must be -180..180."
    try:
        query = urllib.parse.urlencode(
            {
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "zoom": 14,
                "addressdetails": 1,
            }
        )
        request = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/reverse?{query}",
            headers={"User-Agent": "cattle-breed-app/1.0"},
        )
        with urllib.request.urlopen(request, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return f"{gps} (location lookup unavailable)"
    address = data.get("address", {})
    locality = address.get("village") or address.get("town") or address.get("city") or address.get("hamlet")
    region = address.get("state") or address.get("county") or ""
    country = address.get("country") or ""
    if locality:
        return ", ".join(part for part in [locality, region, country] if part)
    return data.get("display_name") or gps
def _extract_bundle(bundle_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as archive:
        archive.extractall(destination, filter="data")
def _bundle_files(bundle_path: Path, destination: Path) -> list[Path]:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    _extract_bundle(bundle_path, destination)
    return [path for path in destination.rglob("*") if path.is_file()]
def _load_classes(classes_path: Path) -> list[str]:
    with classes_path.open("r", encoding="utf-8") as classes_file:
        classes = json.load(classes_file)
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("class_names.json must contain a list of class-name strings")
    if not classes:
        raise ValueError("class_names.json must contain at least one class name")
    return classes
def _load_int8_model(model_path: Path, classes: list[str]) -> torch.nn.Module:
    base_model = models.efficientnet_b0(weights=None)
    base_model.classifier[1] = nn.Linear(base_model.classifier[1].in_features, len(classes))
    quantized_model = torch.quantization.quantize_dynamic(base_model.eval(), {nn.Linear}, dtype=torch.qint8)
    state = torch.load(model_path, map_location="cpu")
    quantized_model.load_state_dict(state)
    return quantized_model.eval()
def _load_from_bundle(bundle_path: Path) -> tuple[torch.nn.Module, list[str], dict[str, str]]:
    bundle_dir = Path(tempfile.gettempdir()) / "cattle_model_bundle"
    files = _bundle_files(bundle_path, bundle_dir)
    classes_paths = [path for path in files if path.name == "class_names.json"]
    int8_paths = [path for path in files if path.name == "breed_classifier_int8.pt"]
    torchscript_paths = [path for path in files if path.name == "breed_classifier_ts.pt"]
    if not classes_paths:
        raise FileNotFoundError("Model bundle missing class_names.json")
    if not int8_paths and not torchscript_paths:
        raise FileNotFoundError("Model bundle missing breed_classifier_int8.pt or breed_classifier_ts.pt")
    classes = _load_classes(classes_paths[0])
    if int8_paths:
        return _load_int8_model(int8_paths[0], classes), classes, {"type": "int8"}
    torchscript_model = torch.jit.load(str(torchscript_paths[0]), map_location="cpu").eval()
    return torchscript_model, classes, {"type": "torchscript"}
def _normalize_loaded(loaded: Any) -> tuple[torch.nn.Module, list[str], dict[str, Any]]:
    if isinstance(loaded, tuple):
        if len(loaded) == 3:
            model, classes, meta = loaded
            return model, classes, meta
        if len(loaded) == 2:
            model, classes = loaded
            return model, classes, {"type": "unknown"}
    if isinstance(loaded, dict):
        return loaded.get("model"), loaded.get("classes"), loaded.get("meta", {"type": "unknown"})
    raise RuntimeError(f"Unexpected loader output type={type(loaded)}")
def get_model() -> tuple[torch.nn.Module, list[str]]:
    global MODEL, CLASSES, MODEL_META
    if MODEL is None or CLASSES is None:
        MODEL, CLASSES, MODEL_META = _normalize_loaded(_load_from_bundle(_model_bundle_path()))
    return MODEL, CLASSES
def inspect_bundle_files() -> dict[str, Any]:
    bundle_path = _model_bundle_path()
    info: dict[str, Any] = {"bundle_path": str(bundle_path), "exists": bundle_path.exists(), "files": []}
    if not bundle_path.exists():
        return info
    debug_dir = Path(tempfile.gettempdir()) / "cattle_model_bundle_debug"
    files = _bundle_files(bundle_path, debug_dir)
    info["files"] = sorted(str(path.relative_to(debug_dir)) for path in files)
    return info

def _predict_image(image: Image.Image, model: torch.nn.Module, classes: list[str]) -> PredictionResult:
    tensor = TFMS(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        top_k = min(5, len(classes))
        values, indexes = torch.topk(probabilities, top_k)

    top_index = indexes[0].item()
    rows = "".join(
        f"<li>{_escape(classes[index])}: {value * 100:.2f}%</li>"
        for value, index in zip(values.tolist(), indexes.tolist())
    )
    return PredictionResult(breed=classes[top_index], confidence=values[0].item() * 100, rows_html=rows)

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None, "model_type": (MODEL_META or {}).get("type")}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_home(request)

@app.get("/signin", response_class=HTMLResponse)
def signin_page(request: Request) -> HTMLResponse:
    return HTMLResponse(render_signin(request))

@app.post("/signin")
async def signin(request: Request, identity: str = Form(...), password: str = Form(...)):
    user_key = _normalize_identity(identity)
    stored = USERS.get(user_key)
    if not stored or stored["password"] != password:
        return HTMLResponse(render_signin(request, "Invalid credentials", is_error=True), status_code=401)
    request.session["user"] = stored["username"]
    return RedirectResponse(url="/", status_code=303)

@app.get("/create-account", response_class=HTMLResponse)
def create_account_page(request: Request) -> HTMLResponse:
    return HTMLResponse(render_create_account(request))

@app.post("/create-account", response_class=HTMLResponse)
async def create_account(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
) -> HTMLResponse:
    clean_email = email.strip()
    clean_username = username.strip()
    clean_password = password.strip()

    if not clean_email or not clean_username or not clean_password:
        return HTMLResponse(
            render_create_account(request, "Email, username, and password are required.", is_error=True),
            status_code=400,
        )

    username_key = _normalize_identity(clean_username)
    email_key = _normalize_identity(clean_email)
    if username_key in USERS or email_key in USERS:
        return HTMLResponse(render_create_account(request, "Username or email already exists.", is_error=True), status_code=400)

    user_record = {"email": clean_email, "username": clean_username, "password": clean_password}
    USERS[username_key] = user_record
    USERS[email_key] = user_record
    return HTMLResponse(render_signin(request, f"Account created for {clean_username}. Please sign in."))

@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/signin", status_code=303)

@app.get("/debug/bundle")
def debug_bundle() -> dict[str, Any]:
    if os.getenv(DEBUG_BUNDLE_FLAG, "false").lower() != "true":
        raise HTTPException(status_code=403, detail=f"Enable {DEBUG_BUNDLE_FLAG}=true to use this endpoint")
    return inspect_bundle_files()

@app.post("/predict", response_class=HTMLResponse)
async def predict_page(
    request: Request,
    file: UploadFile = File(...),
    animal_id: str = Form(default=""),
    gps_coordinates: str = Form(default=""),
):
    if not _current_user(request):
        return RedirectResponse(url="/signin", status_code=303)
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    try:
        model, classes = get_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}. Ensure latest deployment is active.") from exc

    prediction = _predict_image(image, model, classes)
    location_label = resolve_location_label(gps_coordinates)
    image_b64 = base64.b64encode(content).decode("utf-8")
    return HTMLResponse(render_result(request, prediction, animal_id, location_label, image_b64))
