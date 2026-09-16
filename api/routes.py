"""
MAIA Beacon - FastAPI HTTP Routes & API Endpoints.
Pure routing layer delegating inference and lifecycle to EngineManager.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time
from typing import Optional, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from engines.manager import manager


class SelectionRequest(BaseModel):
    model: str
    context_size: int = 16384
    thinking: Optional[bool] = False
    thinking_effort: Optional[Union[str, int]] = "medium"
    mmproj: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.logger.info(
        "Starting MAIA Beacon on %s:%d (Llama base: %s, Default device: %s)",
        config.BEACON_HOST,
        config.BEACON_PORT,
        config.LLAMA_BASE_URL,
        config.TARGET_DEVICE,
    )
    watchdog_task = asyncio.create_task(manager.idle_watchdog())
    yield
    watchdog_task.cancel()
    manager.stop_all()
    config.logger.info("MAIA Beacon shut down cleanly.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    app = FastAPI(title="MAIA Beacon", version="1.4.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Worker Node Administration Endpoints

    @app.get("/api/status")
    def get_status():
        """Returns node health, hardware telemetry (GPU VRAM & NPU), and active engine."""
        return manager.get_status()

    @app.get("/api/models")
    def list_models():
        """Returns all models (GGUF & OpenVINO IR) available on this Beacon node."""
        return {"models": manager.get_available_models()}

    @app.post("/api/select")
    def select_model(req: SelectionRequest, background_tasks: BackgroundTasks):
        """Switches or pre-warms the active model on its respective engine."""
        avail = manager.get_available_models()
        clean = req.model[:-5] if req.model.lower().endswith(".gguf") else req.model

        matched = None
        for m in avail:
            if m.lower() == clean.lower():
                matched = m
                break
            elif len(clean) > 5 and (clean.lower() in m.lower() or m.lower() in clean.lower()):
                matched = m

        if not matched:
            raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found in {config.MODELS_DIR}")

        with manager.state_lock:
            cur_status = manager.state.get("status")
            cur_model = manager.state.get("active_model")
            if cur_status == "running" and cur_model and cur_model.lower() == matched.lower():
                manager.state["last_activity"] = time.time()
                return {
                    "message": "Model already loaded and running",
                    "status": "running",
                    "active_model": matched,
                    "engine": manager.state.get("active_engine"),
                }

        background_tasks.add_task(
            manager.load_model_task,
            matched,
            req.context_size,
            req.thinking or False,
            req.thinking_effort or "medium",
            req.mmproj,
        )
        return {"message": "Model selection started", "status": "starting", "active_model": matched}

    @app.post("/api/stop")
    def stop_model():
        """Stops active engine (llama-server or NPU) to free memory immediately."""
        manager.stop_all()
        return {"message": "Model stopped successfully"}

    # OpenAI Compatible Endpoints

    @app.get("/v1/models")
    def list_v1_models():
        """OpenAI-compatible models listing."""
        return manager.list_v1_models()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions routed to the optimal engine."""
        return await manager.handle_chat_completions(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        """OpenAI-compatible text completions routed to the optimal engine."""
        return await manager.handle_completions(request)

    return app


app = create_app()
