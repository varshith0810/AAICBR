# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes

This repository contains a software-only AI/ML system for recognizing Indian cattle and buffalo breeds from images. It supports two common workflows:

1. **Run the deployed-style FastAPI web app** for image upload, sign-in, prediction, GPS location display, and health checks.
2. **Train/export the model locally or in Colab** using an already extracted dataset folder.

The project uses PyTorch/EfficientNet-B0 for breed classification and FastAPI for the production-style web interface.

---

## Quick start: run the web app

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Confirm the model bundle exists

The FastAPI app expects this file in the repository root by default:

```text
cattle_model_low_hw.tar.gz
```

You can also point to another bundle with the `MODEL_BUNDLE` environment variable.

### 4. Start the app

Recommended direct command:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Compatibility command used by the current Render config:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open the app:

```text
http://localhost:8000
```

### 5. Use the web app

1. Open `/create-account` and create a username/password.
2. Sign in at `/signin`.
3. Upload an animal image on `/`.
4. Optionally enter:
   - Animal ID
   - GPS coordinates as `lat,long`, for example `30.8717,75.8520`
5. Submit the form to view:
   - Predicted breed
   - Confidence score
   - Top scores
   - Uploaded image preview
   - Location label if GPS lookup is available

---

## FastAPI endpoints

| Route | Method | Purpose |
|---|---:|---|
| `/health` | GET | Basic health check and model-loaded status |
| `/` | GET | Authenticated upload/prediction page |
| `/signin` | GET/POST | Sign-in page and login handler |
| `/create-account` | GET/POST | Account creation page and handler |
| `/logout` | GET | Clear current session |
| `/predict` | POST | Image upload and breed prediction |
| `/debug/bundle` | GET | Model-bundle inspection when `DEBUG_BUNDLE=true` |

---

## Deployment on Render

The repository includes `render.yaml` for Render Blueprint deployment.

Current Render start command:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

`backend.app` is only a compatibility wrapper. The real app lives in `src.app`.

Required deployment files:

```text
render.yaml
requirements.txt
cattle_model_low_hw.tar.gz
src/app.py
backend/app.py
```

Render environment variables configured in `render.yaml`:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_BUNDLE` | `cattle_model_low_hw.tar.gz` | Model bundle loaded by the app |
| `DEBUG_BUNDLE` | `false` | Enables `/debug/bundle` when set to `true` |

Recommended production override:

| Variable | Purpose |
|---|---|
| `SESSION_SECRET` | Secret key for signed browser sessions. Set this to a strong random value in production. |

---

## Project structure

```text
.
├── app.py                         # Optional Gradio demo app for local model testing
├── backend/
│   ├── app.py                     # Compatibility wrapper: imports app from src.app
│   ├── db.py                      # Optional SQLite helper for future DB-backed auth/history
│   ├── schema.sql                 # Optional SQLite schema for users and predictions
│   └── ml/                        # Older/Colab-oriented ML workspace
├── cattle_model_low_hw.tar.gz     # Deployment model bundle used by FastAPI app
├── dataset                        # Text file containing the Kaggle dataset URL
├── frontend/                      # UI notes/mockup assets only; no separate frontend app yet
├── render.yaml                    # Render deployment configuration
├── requirements.txt               # Python dependencies
├── scripts/
│   ├── export_low_hardware.py     # Export trained model to int8/TorchScript/ONNX artifacts
│   └── run_pipeline.sh            # Local preprocess + train + Gradio demo helper
└── src/
    ├── app.py                     # Main FastAPI app and inference web UI
    ├── config.py                  # Breed list and standard local paths
    ├── infer.py                   # Local predictor used by root Gradio app
    ├── preprocess.py              # Dataset structure validation
    ├── train.py                   # Training script and optional Gradio app mode
    └── web_app.py                 # Compatibility wrapper around src.app
```

---

## Model bundle expected by the FastAPI app

By default, `src.app` loads:

```text
cattle_model_low_hw.tar.gz
```

The bundle must contain:

```text
class_names.json
breed_classifier_int8.pt
```

or:

```text
class_names.json
breed_classifier_ts.pt
```

The loader searches recursively inside the extracted tarball, so nested paths are accepted as long as those filenames exist.

To inspect the bundle in a running app:

```bash
DEBUG_BUNDLE=true uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/debug/bundle
```

---

## Dataset setup for training

Training expects an already extracted dataset folder. The code does **not** download or unzip the dataset for you.

Expected folder format, either:

```text
/path/to/breeds/train/<breed_name>/*.jpg
/path/to/breeds/test/<breed_name>/*.jpg
```

or:

```text
/path/to/dataset/breeds/train/<breed_name>/*.jpg
/path/to/dataset/breeds/test/<breed_name>/*.jpg
```

The file named `dataset` in this repo contains the Kaggle dataset URL.

---

## Validate dataset structure

```bash
python -m src.preprocess /path/to/breeds
```

If you do not pass a path, the script will ask for one interactively:

```bash
python -m src.preprocess
```

The validation prints:

- missing train/test splits,
- missing breed folders,
- image counts per breed.

---

## Train a model

Example:

```bash
python -m src.train --mode all --dataset_dir /path/to/breeds --work_dir . --epochs 8 --batch_size 32 --lr 0.001
```

Useful modes:

| Mode | What it does |
|---|---|
| `preprocess` | Validate dataset structure only |
| `train` | Train model only |
| `app` | Launch Gradio app using an existing trained model |
| `all` | Validate, train, then ask for one image path for a quick prediction |

Training outputs are written under:

```text
models/
├── breed_classifier.pt
└── class_names.json
```

`models/` is ignored by git because trained model files can be large.

---

## Export model for low-hardware deployment

After training, export CPU-friendly artifacts:

```bash
python scripts/export_low_hardware.py \
  --model_path models/breed_classifier.pt \
  --classes_path models/class_names.json \
  --out_dir models
```

Generated files:

```text
models/breed_classifier_int8.pt
models/breed_classifier_ts.pt
models/breed_classifier.onnx  # optional; skipped if ONNX dependencies are unavailable
models/class_names.json
```

Create the minimal deployment bundle:

```bash
tar -czf cattle_model_low_hw.tar.gz \
  -C models \
  breed_classifier_int8.pt \
  class_names.json
```

Then run the FastAPI app again:

```bash
MODEL_BUNDLE=./cattle_model_low_hw.tar.gz uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## Optional Gradio demo

The root `app.py` launches a simple Gradio UI that uses `src.infer` and expects these local files:

```text
models/breed_classifier.pt
models/class_names.json
```

Run it with:

```bash
python app.py
```

This is useful for quick local experiments after training. For deployment, prefer the FastAPI app in `src.app`.

---

## One-command local pipeline helper

If your dataset path is provided interactively, you can run:

```bash
./scripts/run_pipeline.sh
```

This runs:

1. `python -m src.preprocess`
2. `python -m src.train`
3. `python app.py`

For reproducibility, explicit commands with `--dataset_dir` are recommended instead.

---

## Colab-oriented single-file workflow

A single-file Colab-oriented script is available at:

```text
backend/ml/colab_breed_recognition.py
```

Example:

```bash
python backend/ml/colab_breed_recognition.py \
  --mode all \
  --dataset_dir /path/to/breeds \
  --work_dir .
```

This script is useful if you want one file that validates, trains, predicts, or launches a Gradio app in a notebook/Colab-style workflow.

---

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `MODEL_BUNDLE` | `src.app` | Path to model tarball. Defaults to `cattle_model_low_hw.tar.gz`. |
| `SESSION_SECRET` | `src.app` | Secret key for browser sessions. Use a strong value in production. |
| `DEBUG_BUNDLE` | `src.app` | Set to `true` to enable `/debug/bundle`. |
| `DATASET_DIR` | `backend/ml/colab_breed_recognition.py` | Optional dataset path fallback for the Colab script. |
| `DB_PATH` | `backend/db.py` | Optional SQLite DB path if DB helpers are wired into the app later. |

---

## Troubleshooting

### `Model load failed` or bundle missing

Check that `cattle_model_low_hw.tar.gz` exists in the repo root or set `MODEL_BUNDLE` explicitly:

```bash
MODEL_BUNDLE=/path/to/cattle_model_low_hw.tar.gz uvicorn src.app:app --host 0.0.0.0 --port 8000
```

You can inspect bundle contents with:

```bash
DEBUG_BUNDLE=true uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Then visit `/debug/bundle`.

### Invalid GPS format

Use this format:

```text
lat,long
```

Example:

```text
30.8717,75.8520
```

### Training cannot find dataset

Make sure your path contains either:

```text
train/
test/
```

or:

```text
breeds/train/
breeds/test/
```

### Root Gradio app cannot find model

`python app.py` expects:

```text
models/breed_classifier.pt
models/class_names.json
```

If you only have `cattle_model_low_hw.tar.gz`, use the FastAPI app instead.

---

## Recommended workflow for new users

If you only want to **try the existing app**:

```bash
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

If you want to **train your own model**:

```bash
pip install -r requirements.txt
python -m src.preprocess /path/to/breeds
python -m src.train --mode train --dataset_dir /path/to/breeds --work_dir .
python scripts/export_low_hardware.py --model_path models/breed_classifier.pt --classes_path models/class_names.json --out_dir models
tar -czf cattle_model_low_hw.tar.gz -C models breed_classifier_int8.pt class_names.json
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
