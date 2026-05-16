"""Compatibility entrypoint for Render/uvicorn startup.

Use either:
- uvicorn backend.app:app
- uvicorn src.web_app:app
"""

from backend.app import app

__all__ = ["app"]
