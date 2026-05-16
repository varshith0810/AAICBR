# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes

Clean two-tier repository layout:

```text
backend/
  app.py
  db.py
  schema.sql
  README.md
  ml/
    src/
      config.py
      preprocess.py
      train.py
      infer.py
    export_low_hardware.py
    colab_breed_recognition.py

frontend/
  README.md
  docs/
    frontend_preview.svg
```

## Run Backend

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Deploy

Render start command:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```
