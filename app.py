"""
MAIA Beacon - Application Entrypoint (Backward-Compatible Alias).
Exposes 'app' for ASGI runners (e.g. uvicorn app:app) and routes CLI execution to main.py.
"""

from api.routes import app
import main

if __name__ == "__main__":
    main.main()
