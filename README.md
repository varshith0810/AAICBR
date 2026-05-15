# AI-Assisted Breed Recognition for Indian Cattle and Buffaloes

Reorganized into frontend/backend style with database support.

## Structure
- `backend/app.py` FastAPI backend + server-rendered frontend pages
- `backend/db.py` SQLite database initialization and access
- `frontend/` reserved for future separate frontend assets
- `src/` ML/training modules

## Database
SQLite is used by default (`backend/app.db`) with tables:
- `users`
- `predictions`

## Render deploy
Start command:
`uvicorn backend.app:app --host 0.0.0.0 --port $PORT`

## Features
- Login from DB users table
- Image upload/camera capture
- Breed prediction
- Email notification (optional SMTP)
- Prediction history page: `/history`
