"""
MAIA Beacon - Central Engine Manager & Model Orchestrator.
Encapsulates global worker state, inactivity watchdog, model discovery, and request dispatching.
No Ollama dependencies - dedicated to GPU/CPU (llama-server) and Intel NPU (OpenVINO).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException, Request

import config
import engines.llama_engine as llama_engine
import engines.npu_engine as npu_engine
from hardware import hw_manager

logger = logging.getLogger("maia.beacon.manager")


class EngineManager:
    """Orchestrates local inference engines (GPU / Turboquant & Intel NPU)."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "process": None,
            "status": "idle",  # "idle", "starting", "running", "sleeping", "error"
            "active_engine": None,  # "npu", "llama-server", None
            "active_model": None,
            "active_mmproj": None,
            "active_context": None,
            "active_thinking": False,
            "active_thinking_effort": None,
            "config": None,
            "error_message": None,
            "startup_log": [],
            "last_activity": time.time(),
        }
        self.state_lock = threading.Lock()
        self.startup_lock = threading.Lock()

    # Catalog & Detection

    def get_available_models(self) -> List[str]:
        """Returns consolidated list of all GGUF and OpenVINO IR models."""
        gguf = llama_engine.get_available_gguf_models(config.MODELS_DIR)
        npu = npu_engine.get_available_openvino_models(config.MODELS_DIR)
        return sorted(list(set(gguf + npu)))

    def detect_engine(self, model_name: str) -> str:
        """Determines which engine ('npu' or 'llama') should handle the specified model."""
        clean = model_name[:-5] if model_name.lower().endswith(".gguf") else model_name
        if npu_engine.is_openvino_model(clean, config.MODELS_DIR):
            return "npu"
        if llama_engine.is_gguf_model(clean, config.MODELS_DIR):
            return "llama"

        # If model is unspecified or ambiguous, fallback according to TARGET_DEVICE
        ov_models = npu_engine.get_available_openvino_models(config.MODELS_DIR)
        if config.TARGET_DEVICE == "NPU" and ov_models:
            return "npu"
        return "llama"

    # --- Engine Lifecycle Management ---

    def stop_all(self) -> None:
        """Stops all running processes and unloads models to release all VRAM/RAM."""
        llama_engine.kill_all_llama_servers()
        npu_engine.unload_npu_model()
        with self.state_lock:
            self.state["process"] = None
            self.state["status"] = "idle"
            self.state["active_engine"] = None
            self.state["active_model"] = None
            self.state["active_mmproj"] = None
            self.state["active_context"] = None
            self.state["active_thinking"] = False
            self.state["config"] = None
            self.state["error_message"] = None
        logger.info("[*] All inference engines stopped.")

    def sleep_active_engine(self) -> None:
        """Puts active model to sleep after inactivity timeout to free VRAM/NPU memory."""
        llama_engine.kill_all_llama_servers()
        npu_engine.unload_npu_model()
        with self.state_lock:
            self.state["process"] = None
            self.state["active_engine"] = None
            self.state["status"] = "sleeping" if self.state.get("active_model") else "idle"
        logger.info("[*] Inactivity timeout: model put to sleep to free VRAM/NPU memory.")

    async def idle_watchdog(self) -> None:
        """Periodic background task that puts model to sleep if idle for IDLE_TIMEOUT_SECONDS."""
        if config.IDLE_TIMEOUT_SECONDS <= 0:
            return
        while True:
            await asyncio.sleep(15.0)
            with self.state_lock:
                cur_status = self.state["status"]
                last_act = self.state["last_activity"]

            if cur_status == "running" and (time.time() - last_act > config.IDLE_TIMEOUT_SECONDS):
                self.sleep_active_engine()

    def load_model_task(
        self,
        model_name: str,
        context_size: int = 16384,
        thinking: bool = False,
        thinking_effort: Union[str, int] = "medium",
        mmproj: Optional[str] = None,
    ) -> bool:
        """Loads a model on its target hardware engine."""
        engine_type = self.detect_engine(model_name)

        if engine_type == "npu":
            logger.info("[*] Loading OpenVINO IR model '%s' on Intel AI Boost NPU...", model_name)
            return npu_engine.load_npu_model(
                model_name=model_name,
                models_dir=config.MODELS_DIR,
                state=self.state,
                state_lock=self.state_lock,
                before_load_fn=llama_engine.kill_all_llama_servers,
            )
        else:
            logger.info("[*] Loading GGUF model '%s' on GPU with Turboquant...", model_name)
            npu_engine.unload_npu_model()
            return llama_engine.start_llama_server_task(
                model_filename=model_name,
                context_size=context_size,
                thinking=thinking,
                thinking_effort=thinking_effort,
                mmproj_filename=mmproj,
                models_dir=config.MODELS_DIR,
                llama_server_exe=config.LLAMA_SERVER_EXE,
                llama_server_host=config.LLAMA_SERVER_HOST,
                llama_server_port=config.LLAMA_SERVER_PORT,
                llama_load_mode=config.LLAMA_LOAD_MODE,
                state=self.state,
                state_lock=self.state_lock,
                startup_lock=self.startup_lock,
            )

    # Telemetry & Status

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry (GPU VRAM, NPU stats, active model, and engine)."""
        hw_status = hw_manager.get_telemetry()
        all_models = self.get_available_models()

        with self.state_lock:
            return {
                "type": "beacon",
                "status": self.state["status"],
                "active_engine": self.state.get("active_engine"),
                "active_model": self.state["active_model"],
                "available_models": all_models,
                "target_device": config.TARGET_DEVICE,
                "active_mmproj": self.state.get("active_mmproj"),
                "active_context": self.state["active_context"],
                "active_thinking": self.state["active_thinking"],
                "config": self.state["config"],
                "error_message": self.state["error_message"],
                "npu": hw_status["npu"],
                "gpu_vram": hw_status["gpu_vram"],
                "pid": self.state["process"].pid if self.state["process"] and self.state["process"].poll() is None else None,
            }

    def list_v1_models(self) -> Dict[str, Any]:
        """Formats available models in OpenAI /v1/models JSON structure."""
        models = self.get_available_models()
        now_ts = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "maia-beacon",
                }
                for m in models
            ],
        }

    # OpenAI Dispatching

    async def handle_chat_completions(self, request: Request):
        """Dispatches chat completions to NPU or GPU according to requested model."""
        data = await request.json() or {}
        raw_model = str(data.get("model", ""))
        clean_model = raw_model.removesuffix(".gguf")

        engine_type = self.detect_engine(clean_model)
        ov_models = npu_engine.get_available_openvino_models(config.MODELS_DIR)

        with self.state_lock:
            cur_engine = self.state.get("active_engine")
            cur_model = self.state.get("active_model")
            cur_status = self.state.get("status")

        if engine_type == "npu":
            target_m = clean_model if npu_engine.is_openvino_model(clean_model, config.MODELS_DIR) else (cur_model or ov_models[0])
            if cur_status != "running" or cur_engine != "npu" or cur_model != target_m:
                logger.info("[*] Auto-loading NPU model '%s' for chat completion...", target_m)
                success = await asyncio.to_thread(
                    npu_engine.load_npu_model,
                    target_m,
                    config.MODELS_DIR,
                    self.state,
                    self.state_lock,
                    llama_engine.kill_all_llama_servers,
                )
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to load NPU model '{target_m}'")
            return await npu_engine.generate_npu(request, data, self.state, self.state_lock)

        # GPU / CPU with Turboquant via llama-server
        return await llama_engine.proxy_to_llama_server(
            path="v1/chat/completions",
            request=request,
            json_data=data,
            llama_base_url=config.LLAMA_BASE_URL,
            models_dir=config.MODELS_DIR,
            llama_server_exe=config.LLAMA_SERVER_EXE,
            llama_server_host=config.LLAMA_SERVER_HOST,
            llama_server_port=config.LLAMA_SERVER_PORT,
            llama_load_mode=config.LLAMA_LOAD_MODE,
            state=self.state,
            state_lock=self.state_lock,
            startup_lock=self.startup_lock,
        )

    async def handle_completions(self, request: Request):
        """Dispatches completions to NPU or GPU according to requested model."""
        data = await request.json() or {}
        raw_model = str(data.get("model", ""))
        clean_model = raw_model.removesuffix(".gguf")

        engine_type = self.detect_engine(clean_model)
        ov_models = npu_engine.get_available_openvino_models(config.MODELS_DIR)

        with self.state_lock:
            cur_engine = self.state.get("active_engine")
            cur_model = self.state.get("active_model")
            cur_status = self.state.get("status")

        if engine_type == "npu":
            target_m = clean_model if npu_engine.is_openvino_model(clean_model, config.MODELS_DIR) else (cur_model or ov_models[0])
            if cur_status != "running" or cur_engine != "npu" or cur_model != target_m:
                logger.info("[*] Auto-loading NPU model '%s' for completion...", target_m)
                success = await asyncio.to_thread(
                    npu_engine.load_npu_model,
                    target_m,
                    config.MODELS_DIR,
                    self.state,
                    self.state_lock,
                    llama_engine.kill_all_llama_servers,
                )
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to load NPU model '{target_m}'")
            return await npu_engine.generate_npu(request, data, self.state, self.state_lock)

        return await llama_engine.proxy_to_llama_server(
            path="v1/completions",
            request=request,
            json_data=data,
            llama_base_url=config.LLAMA_BASE_URL,
            models_dir=config.MODELS_DIR,
            llama_server_exe=config.LLAMA_SERVER_EXE,
            llama_server_host=config.LLAMA_SERVER_HOST,
            llama_server_port=config.LLAMA_SERVER_PORT,
            llama_load_mode=config.LLAMA_LOAD_MODE,
            state=self.state,
            state_lock=self.state_lock,
            startup_lock=self.startup_lock,
        )


# Singleton instance shared across API routes
manager = EngineManager()
