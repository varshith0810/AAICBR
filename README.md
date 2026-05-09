# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes

Render-focused deployment + Colab training/export project.

## Deploy on Render
1. Keep `cattle_model_low_hw.tar.gz` in repo root.
2. Push to GitHub.
3. In Render: New + -> Blueprint -> select repo.
4. Render uses `render.yaml` to deploy `uvicorn src.web_app:app --host 0.0.0.0 --port $PORT`.
5. Open deployed URL and upload image.

## Debug
- `GET /health`
- Optional: set `DEBUG_BUNDLE=true` and call `GET /debug/bundle`.

## Train locally/colab with local dataset path
```bash
python colab_breed_recognition.py --mode all --dataset_dir "/Users/yourname/datasets/breeds" --work_dir .
```


## Login & camera-enabled frontend
- Login is required before prediction.
- Set env vars for credentials:
  - `APP_USERNAME` (default: `admin`)
  - `APP_PASSWORD` (default: `admin123`)
- Browser camera permission is requested with **Allow Camera** button.
