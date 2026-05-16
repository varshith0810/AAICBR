# Backend

## Core service
- `app.py`: FastAPI app and server-rendered UI routes
- `db.py`: SQLite connection/bootstrap
- `schema.sql`: canonical schema for users and predictions

## ML workspace (kept inside backend)
- `ml/src/`: training/inference/preprocessing/config modules
- `ml/export_low_hardware.py`: quantized/TorchScript export utility
- `ml/colab_breed_recognition.py`: Colab-oriented single-file flow

## Entrypoint

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
