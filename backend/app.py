import base64
import hashlib
import hmac
import json
import os
import tarfile
import tempfile
import smtplib
from io import BytesIO
from pathlib import Path
from email.message import EmailMessage
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware
from torchvision import models, transforms
from passlib.context import CryptContext
from backend.db import init_db, get_conn
app = FastAPI(title="Cattle Breed Recognition Frontend + API")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-me-please"))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def _hash_password(password: str) -> str:
    return pwd_context.hash(password)
def _verify_password(raw_password: str, stored_password_hash: str) -> bool:
    if stored_password_hash.startswith("$2"):
        return pwd_context.verify(raw_password, stored_password_hash)
    # Backward compatibility for legacy sha256 records.
    expected = hashlib.sha256(raw_password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(expected, stored_password_hash)
from backend.db import init_db, get_conn
app = FastAPI(title="Cattle Breed Recognition Frontend + API")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-me-please"))
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()
def _verify_password(raw_password: str, stored_password_hash: str) -> bool:
    expected = _hash_password(raw_password)
    return hmac.compare_digest(expected, stored_password_hash)
@app.on_event("startup")
def startup():
    init_db()
    conn = get_conn()
    username = os.getenv("APP_USERNAME", "admin")
    password_hash = _hash_password(os.getenv("APP_PASSWORD", "admin123"))
    role = "admin"
    existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if existing:
        conn.execute("UPDATE users SET password_hash=?, role=? WHERE username=?", (password_hash, role, username))
    else:
        conn.execute(
            "INSERT INTO users (username,password_hash,role) VALUES (?,?,?)",
            (username, password_hash, role),
        )
    conn.commit()
    conn.close()
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
    </form>
    <p style='margin-top:12px'>New user? <a href='/signup'>Create an account</a></p>
    </div></div></body></html>"""

def render_signup(error: str = ""):
    err = f"<p class='error'>{error}</p>" if error else ""
    return f"""<html><head>{BASE_STYLE}</head><body><div class='container'><div class='card'>
    <div class='badge'>Create Account</div><div class='title'>Sign up</div><div class='subtitle'>Register to use breed recognition.</div>{err}
    <form action='/signup' method='post'>
      <label>Username</label><input type='text' name='username' minlength='3' required>
      <label style='margin-top:10px;display:block'>Password</label><input type='password' name='password' minlength='6' required>
      <div style='margin-top:16px'><button type='submit'>Create Account</button></div>
    </form>
    <p style='margin-top:12px'>Already have an account? <a href='/'>Sign in</a></p>
    </div></div></body></html>"""
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
      <div style='margin-top:12px'><label>Email for result (optional)</label><input type='email' name='notify_email' placeholder='farmer@example.com'></div>
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
def send_prediction_email(to_email: str, result: dict):
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")
    from_email = os.getenv("FROM_EMAIL", user)
    if not (host and from_email and to_email):
        return {"sent": False, "reason": "SMTP not configured or recipient missing"}
    msg = EmailMessage()
    msg["Subject"] = "Breed Prediction Result"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(
        f"Predicted Breed: {result['predicted_breed']}\n"
        f"Confidence: {result['confidence']:.2f}%\n"
        f"Animal ID: {result.get('animal_id','N/A')}\n"
        f"GPS: {result.get('gps','N/A')}\n"
    )

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return {"sent": True}
    except Exception as e:
        return {"sent": False, "reason": str(e)}
def render_result(top, conf, animal_id, gps_coordinates, rows, email_status=""):
    return f"""<html><head>{BASE_STYLE}</head><body><div class='container'><div class='card'>
    <div class='badge'>Prediction Complete</div><div class='title'>Breed Recognition Result</div>
    <div class='result'><p><b>Predicted Breed:</b> {top}</p><p><b>Confidence:</b> {conf:.2f}%</p>{email_status}
    <p><b>Animal ID:</b> {animal_id or 'N/A'}</p><p><b>GPS Coordinates:</b> {gps_coordinates or 'N/A'}</p><h4>Top-5 Scores</h4><ul>{rows}</ul></div>
    <div style='margin-top:16px'><a href='/'><button>Try Another Image</button></a></div>
    </div></div></body></html>"""
def _safe_extract_tar(tar: tarfile.TarFile, target_dir: Path):
    target_dir = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if not str(member_path).startswith(str(target_dir)):
            raise ValueError(f"Unsafe tar path detected: {member.name}")
    tar.extractall(target_dir)
def _load_from_bundle(bundle_path: Path):
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    tmp_dir = Path(tempfile.gettempdir()) / "cattle_model_bundle"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        _safe_extract_tar(tar, tmp_dir)
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
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if row and _verify_password(password, row["password_hash"]):
        request.session["user"] = username
        request.session["user_id"] = row["id"]
        return RedirectResponse(url='/', status_code=303)
    return render_login("Invalid credentials")
@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    return render_signup()

@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    return render_signup()
@app.post("/signup", response_class=HTMLResponse)
def signup(username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    if len(username) < 3:
        return render_signup("Username must be at least 3 characters")
    if len(password) < 6:
        return render_signup("Password must be at least 6 characters")
    conn = get_conn()
    existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if existing:
        conn.close()
        return render_signup("Username already exists")
    password_hash = _hash_password(password)
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?,?,?)", (username, password_hash, "user"))
    conn.commit()
    conn.close()
    return RedirectResponse(url='/', status_code=303)

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
    notify_email: str = Form(default=""),
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

    email_html = ""
    if notify_email:
        email_result = send_prediction_email(
            notify_email,
            {
                "predicted_breed": top,
                "confidence": conf,
                "animal_id": animal_id or "N/A",
                "gps": gps_coordinates or "N/A",
            },
        )
        if email_result.get("sent"):
            email_html = "<p><b>Email Notification:</b> Sent successfully.</p>"
        else:
            email_html = f"<p><b>Email Notification:</b> Failed ({email_result.get('reason','unknown')}).</p>"

    conn = get_conn()
    conn.execute(
        "INSERT INTO predictions (user_id,predicted_breed,confidence,animal_id,gps_coordinates,notify_email,image_source) VALUES (?,?,?,?,?,?,?)",
        (request.session.get("user_id"), top, conf, animal_id, gps_coordinates, notify_email, "camera" if captured_image else "upload"),
    )
    conn.commit()
    conn.close()
    return render_result(top, conf, animal_id, gps_coordinates, rows, email_status=email_html)
    return render_result(top, conf, animal_id, gps_coordinates, rows, email_status=email_html)
@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/", status_code=303)
    conn = get_conn()
    rows = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 50", (request.session.get("user_id"),)).fetchall()
    conn.close()
    items = "".join([f"<tr><td>{r['id']}</td><td>{r['predicted_breed']}</td><td>{r['confidence']:.2f}%</td><td>{r['animal_id'] or 'N/A'}</td><td>{r['created_at']}</td></tr>" for r in rows])
    return f"<html><body><h2>Prediction History</h2><table border='1'><tr><th>ID</th><th>Breed</th><th>Confidence</th><th>Animal ID</th><th>Time</th></tr>{items}</table><p><a href='/'>Back</a></p></body></html>"
