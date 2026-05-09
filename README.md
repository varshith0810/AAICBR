
# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes

Yes — you can deploy with only `cattle_model_low_hw.tar.gz` (no separate `models/` folder needed).

## What to keep
- `cattle_model_low_hw.tar.gz` containing:
  - `breed_classifier_int8.pt`
  - `class_names.json`

## Railway-ready app added
- `src/railway_app.py` (frontend + inference backend)
- `Procfile`
- `railway.json`

The app automatically extracts the tar.gz bundle and loads the quantized model.

---

## Local test before Railway

```bash
pip install -r requirements.txt
MODEL_BUNDLE=./cattle_model_low_hw.tar.gz uvicorn src.railway_app:app --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000`

---

## Deploy on Railway (step-by-step)

1. Put `cattle_model_low_hw.tar.gz` in repo root.
2. Push repo to GitHub.
3. Create Railway project from GitHub repo.
4. Add env var (optional if filename is default):
   - `MODEL_BUNDLE=cattle_model_low_hw.tar.gz`
5. Deploy.
6. Open Railway domain and upload animal image.

---

## Endpoints
- `GET /health`
- `GET /` (frontend upload form)
- `POST /predict` (form submit)

---

## Notes
- Software-only model.
- If bundle is missing or invalid, startup inference load will fail with clear error.


## Bundle debug tip
If `/predict` gives model bundle error, your tar likely has nested paths.
The app now searches recursively for `breed_classifier_int8.pt` and `class_names.json` inside the extracted bundle.



## Debug update
Bundle loader now accepts either `breed_classifier_int8.pt` **or** `breed_classifier_ts.pt` plus `class_names.json`.
If deploy still fails, verify tar contents include one model file and class map.



## UI Preview
A frontend preview mock is available at `docs/frontend_preview.svg`.


## Debug patch
If you saw `too many values to unpack (expected 2)`, redeploy latest code.
The loader now supports both 2-value and 3-value tuple returns for compatibility.

# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes (Colab Training-Only)

This repo is now simplified to focus on **training in Google Colab** and **exporting low-hardware model artifacts**.
Production deployment files were removed for now.

## Kept files (only required for training/export)
- `colab_breed_recognition.py` (single-file Colab training + prediction flow)
- `src/config.py`
- `src/preprocess.py`
- `src/train.py`
- `src/infer.py`
- `scripts/export_low_hardware.py`
- `requirements.txt`



## Colab training steps

### 1) Clone repo
```bash
!git clone <YOUR_REPO_URL>
%cd AI-ASSISTED-BREED-RECOGNITION-FOR-INDIAN-CATTLE-AND-BUFFALOES
```

### 2) Install dependencies

# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes


## Dataset path prompt behavior (updated)

The code now asks for dataset path if not provided.

- `colab_breed_recognition.py` asks dataset path before training when `--dataset_dir` is missing.
- `src/preprocess.py` asks dataset path interactively if you run without argument.
- `src/train.py` asks dataset path interactively if you run without argument.

## Colab run

## Updated behavior (as requested)

Now the script does the following in sequence:
1. **Asks dataset path** before training (if `--dataset_dir` is not provided).
2. Trains model.
3. **Asks image path** after training and predicts breed immediately.

## Run in Colab


## Google Drive unzipped dataset only (zip code removed)

The pipeline now **does not unzip data**. It expects dataset already extracted in Google Drive.

## Required dataset folder formats
Either of these:
1. `/content/drive/MyDrive/.../breeds/train` and `/content/drive/MyDrive/.../breeds/test`
2. `/content/drive/MyDrive/.../train` and `/content/drive/MyDrive/.../test`

## Colab run order


### 1) Install dependencies


```bash
!pip install -r requirements.txt
```


### 3) Mount Drive



### 2) Mount Google Drive

### 2) Mount Drive


```python
from google.colab import drive
drive.mount('/content/drive')
```


### 4) Run full flow
```bash
!python colab_breed_recognition.py --mode all --work_dir /content
```
It will ask:
- dataset directory path
- image path for a post-training prediction

(Or pass dataset path directly)




### 3) Run full flow (interactive prompts enabled)

```bash
!python colab_breed_recognition.py --mode all --work_dir /content
```


### Full flow with explicit path

It will ask:
- Dataset directory path (`train/test` or `breeds/train/test`)
- Image path for single prediction after training

### Optional (skip prompt by passing dataset path)

```bash
!python colab_breed_recognition.py --mode all --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```


## Modular run

### Option 1 (interactive prompt)
```bash
python -m src.preprocess
python -m src.train
python app.py
```

### Option 2 (pass path explicitly)
```bash
python -m src.preprocess /content/drive/MyDrive/datasets/breeds
python -m src.train /content/drive/MyDrive/datasets/breeds
python app.py
```

## Launch upload-based app anytime
```bash
!python colab_breed_recognition.py --mode app --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

## Dataset folder format expected
Either:
- `/.../breeds/train` and `/.../breeds/test`
- `/.../train` and `/.../test`

## Notes
- Software-only model.
- Upload-based UI remains available through Gradio app mode.

### 3) Validate + Train

```bash
!python colab_breed_recognition.py --mode all --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```



## Extract model for low hardware

After training, run:
```bash
python scripts/export_low_hardware.py --model_path models/breed_classifier.pt --classes_path models/class_names.json --out_dir models
```

Generated files:
- `models/breed_classifier_int8.pt`
- `models/breed_classifier_ts.pt`
- `models/breed_classifier.onnx`
- `models/class_names.json`

Minimal deploy bundle:
```bash
tar -czf cattle_model_low_hw.tar.gz models/breed_classifier_int8.pt models/class_names.json
```

---

## Next phase
After you finalize the trained/extracted model, we can rebuild a clean production deployment project for Railway.

### 4) Launch App
```bash
!python colab_breed_recognition.py --mode app --dataset_dir "/content/drive/MyDrive/datasets/breeds" --work_dir /content
```

## Modular order (optional)
```bash
python -m src.preprocess /content/drive/MyDrive/datasets/breeds

This project implements an **end-to-end software-only breed recognition pipeline** for the following 41 breeds:

`vechur, umblachery, toda, tharparkar, surti, sahiwal, redsindhi, reddane, rathi, pulikulam, ongole, nimari, niliravi, nagpuri, nagori, murrah, mehsana, malnadgidda, krishnavalley, khillari, kherigarh, kenkatha, kasargod, kankrej, kangayam, jersey, jaffrabadi, holsteinfriesian, hariana, hallikar, guernsey, gir, deoni, dangi, brownswiss, bhadawari, bargur, banni, ayrshire, amritmahal, alambadi`.

## What this builds

1. **Data preprocessing**
   - Unzips `dataset.zip`.
   - Expects structure: `breeds/train/<breed_name>/*.jpg` and `breeds/test/<breed_name>/*.jpg`.
   - Validates split and breed coverage.

2. **Model training**
   - Uses transfer learning with `EfficientNet-B0`.
   - Trains multiclass classifier for breed recognition.
   - Saves best weights as `models/breed_classifier.pt` and class map as `models/class_names.json`.

3. **User-facing recognition app**
   - Gradio app asks user to upload an animal image.
   - Returns predicted breed + top-5 confidence scores.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run full pipeline

```bash
./scripts/run_pipeline.sh
```

Or step-by-step:

```bash
python -m src.preprocess

python -m src.train
python app.py
```

## Notes

- Software-only model.
- App asks user to upload image.


- This implementation is strictly **software model only** (no hardware component).
- Place the provided dataset zip at repo root as `dataset.zip` before preprocessing.





