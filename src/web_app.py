"""Render entrypoint compatibility module.

Some deploy environments are configured to run:
    uvicorn src.web_app:app

This module re-exports the FastAPI application from backend.app so both
`backend.app:app` and `src.web_app:app` work.
"""

from backend.app import app

__all__ = ["app"]
