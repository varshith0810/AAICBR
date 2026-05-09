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


## Use dataset path from your computer (not Drive)
You can pass local dataset path directly:
```bash
python colab_breed_recognition.py --mode all --dataset_dir "/Users/yourname/datasets/breeds" --work_dir .
```
Windows example:
```bash
python colab_breed_recognition.py --mode all --dataset_dir "C:/datasets/breeds" --work_dir .
```
Or via environment variable:
```bash
export DATASET_DIR="/Users/yourname/datasets/breeds"
python colab_breed_recognition.py --mode all --work_dir .
```


## Optional bundle debug endpoint
To inspect bundle contents on Railway, set env var:
`DEBUG_BUNDLE=true`
Then open:
`GET /debug/bundle`
