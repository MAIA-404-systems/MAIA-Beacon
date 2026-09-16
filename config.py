"""
MAIA Beacon - Central Configuration and Environment Management.
Loads and exposes typed application settings from .env file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

# Auto-inject project's venv site-packages if running with a different Python interpreter
ROOT_DIR = Path(__file__).resolve().parent
_VENV_SITE = ROOT_DIR / "venv" / "Lib" / "site-packages"
if _VENV_SITE.exists() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))

ENV_PATH = ROOT_DIR / ".env"


def load_env() -> None:
    """Parses .env file and sets environment variables if not already defined."""
    if not ENV_PATH.exists():
        return
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


load_env()

# Server Settings
BEACON_HOST: str = os.getenv("BEACON_HOST", "0.0.0.0")
BEACON_PORT: int = int(os.getenv("BEACON_PORT", "11345"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# Hardware & Directory Settings
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models"))).resolve()
LLAMA_SERVER_EXE: Path = Path(os.getenv("LLAMA_SERVER_EXE", str(ROOT_DIR / "llama-server.exe"))).resolve()

# Llama Server Settings (GPU / Turboquant)
LLAMA_SERVER_HOST: str = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT: int = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_BASE_URL: str = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_LOAD_MODE: str = os.getenv("LLAMA_LOAD_MODE", "mmap+mlock")

# Target Device Preference (GPU, NPU, AUTO)
TARGET_DEVICE: str = os.getenv("TARGET_DEVICE", "AUTO").upper()

# Idle Sleep Settings (in seconds, 0 = disabled)
IDLE_TIMEOUT_SECONDS: int = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))

# Logger Setup
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("maia.beacon")
