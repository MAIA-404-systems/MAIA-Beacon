"""
MAIA Beacon - Autonomous GPU Worker Node & Model Engine Manager.
Manages local GGUF models, VRAM optimization, llama-server.exe process execution, and local Ollama relay.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel
import psutil
import uvicorn

import optimizer

# Load environment variables from .env file
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"


def load_env() -> None:
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

BEACON_HOST = os.getenv("BEACON_HOST", "0.0.0.0")
BEACON_PORT = int(os.getenv("BEACON_PORT", "11345"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models"))).resolve()
LLAMA_SERVER_EXE = Path(os.getenv("LLAMA_SERVER_EXE", str(ROOT_DIR / "llama-server.exe"))).resolve()

LLAMA_SERVER_HOST = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_BASE_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "300"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("maia.beacon")

# Global State
state: Dict[str, Any] = {
    "process": None,
    "status": "idle",  # "idle", "starting", "running", "sleeping", "error"
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

state_lock = threading.Lock()
startup_lock = threading.Lock()


class SelectionRequest(BaseModel):
    model: str
    context_size: int = 16384
    thinking: Optional[bool] = False
    thinking_effort: Optional[Union[str, int]] = "medium"
    mmproj: Optional[str] = None


def kill_all_llama_servers() -> None:
    """Kills any running llama-server.exe process to free up VRAM."""
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() in ("llama-server.exe", "llama-server"):
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def get_available_gguf_models() -> List[str]:
    """Scans MODELS_DIR for all available .gguf model files."""
    models: List[str] = []
    if MODELS_DIR.exists():
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".gguf") and not any(k in f.lower() for k in ["mmproj", "projector"]):
                base_name = f[:-5]
                if base_name not in models:
                    models.append(base_name)
    return sorted(models)


def get_local_ollama_models() -> List[str]:
    """Fetches list of models from local Ollama instance if available."""
    if not OLLAMA_HOST:
        return []
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=1.0)
        if resp.status_code == 200:
            payload = resp.json()
            return [
                m["name"]
                for m in payload.get("models", [])
                if isinstance(m, dict) and m.get("name")
            ]
    except Exception:
        pass
    return []


def read_process_output(process: subprocess.Popen) -> None:
    """Reads stdout/stderr of llama-server and logs it."""
    if not process.stdout:
        return
    while True:
        line = process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore").strip()
        logger.info("[llama-server] %s", decoded)
        with state_lock:
            state["startup_log"].append(decoded)
            if len(state["startup_log"]) > 200:
                state["startup_log"].pop(0)


def start_llama_server_task(
    model_filename: str,
    context_size: int,
    thinking: bool = False,
    thinking_effort: Union[str, int] = "medium",
    mmproj_filename: Optional[str] = None,
) -> bool:
    global state

    acquired = startup_lock.acquire(blocking=False)
    if not acquired:
        with state_lock:
            cur_model = state["active_model"]
            cur_status = state["status"]
        if cur_status == "starting" and (cur_model == model_filename or cur_model == f"{model_filename}.gguf"):
            logger.info("[*] llama-server is already starting model %s, ignoring duplicate.", model_filename)
            return True
        logger.info("[*] Waiting for previous startup lock to release...")
        startup_lock.acquire(blocking=True)

    try:
        with state_lock:
            state["status"] = "starting"
            state["error_message"] = None
            state["startup_log"] = []

        # Find model file
        target_name = model_filename
        if not target_name.endswith(".gguf"):
            target_name = f"{target_name}.gguf"

        model_path = MODELS_DIR / target_name
        if not model_path.exists() and MODELS_DIR.exists():
            for f in os.listdir(MODELS_DIR):
                if f.lower() == target_name.lower() or f.lower().removesuffix(".gguf") == target_name.lower().removesuffix(".gguf"):
                    model_path = MODELS_DIR / f
                    target_name = f
                    break

        if not model_path.exists():
            with state_lock:
                state["status"] = "error"
                state["error_message"] = f"Model file '{target_name}' not found in {MODELS_DIR}"
            logger.error(state["error_message"])
            return False

        # Auto-detect vision mmproj projector if available
        mmproj_path = None
        if mmproj_filename:
            candidate = MODELS_DIR / mmproj_filename
            if candidate.exists():
                mmproj_path = candidate
        else:
            model_lower = target_name.lower()
            if "mmproj" not in model_lower:
                possible_base = None
                if "gemma" in model_lower:
                    possible_base = "gemma"
                elif "qwen" in model_lower:
                    possible_base = "qwen"
                elif "llava" in model_lower:
                    possible_base = "llava"

                if possible_base and MODELS_DIR.exists():
                    for f in os.listdir(MODELS_DIR):
                        f_lower = f.lower()
                        if f.endswith(".gguf") and "mmproj" in f_lower and possible_base in f_lower:
                            mmproj_path = MODELS_DIR / f
                            logger.info("[Auto-Detect] Found matching vision projector: %s", f)
                            break

        # Kill old servers
        kill_all_llama_servers()
        time.sleep(1.0)

        # Optimize VRAM & layers
        opt_res = optimizer.find_optimal_config(
            str(model_path), context_size, mmproj_path=str(mmproj_path) if mmproj_path else None
        )
        if not opt_res:
            with state_lock:
                state["status"] = "error"
                state["error_message"] = "Optimization failed: model configuration does not fit in VRAM"
            logger.error(state["error_message"])
            return False

        opt_config = opt_res["best_config"]
        parent_dir = LLAMA_SERVER_EXE.parent

        # Build command
        cmd = [
            str(LLAMA_SERVER_EXE),
            "-m", str(model_path),
            "--n-gpu-layers", str(opt_config["ngl"]),
            "--n-cpu-moe", str(opt_config["ncmoe"]),
            "--cache-type-k", opt_config["cache_k"],
            "--cache-type-v", opt_config["cache_v"],
            "-c", str(context_size),
            "--no-mmap",
            "--mlock",
            "--host", LLAMA_SERVER_HOST,
            "--port", str(LLAMA_SERVER_PORT),
        ]

        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])

        if thinking:
            budget = -1
            if isinstance(thinking_effort, int):
                budget = thinking_effort
            elif isinstance(thinking_effort, str):
                eff = thinking_effort.lower()
                if eff == "low": budget = 1024
                elif eff == "medium": budget = 4096
                elif eff == "high": budget = 16384
            cmd.extend(["--reasoning", "on"])
            if budget >= 0:
                cmd.extend(["--reasoning-budget", str(budget)])
        else:
            cmd.extend(["--reasoning", "off", "--reasoning-budget", "0"])

        logger.info("Launching llama-server: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            cwd=str(parent_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        with state_lock:
            state["process"] = proc
            state["active_model"] = model_filename.rstrip(".gguf")
            state["active_mmproj"] = mmproj_path.name if mmproj_path else None
            state["active_context"] = context_size
            state["active_thinking"] = thinking
            state["active_thinking_effort"] = thinking_effort
            state["config"] = opt_config
            state["last_activity"] = time.time()

        threading.Thread(target=read_process_output, args=(proc,), daemon=True).start()

        # Wait for /health
        start_time = time.time()
        timeout = 180.0
        while time.time() - start_time < timeout:
            if proc.poll() is not None:
                with state_lock:
                    state["status"] = "error"
                    state["error_message"] = "llama-server process exited prematurely during startup"
                logger.error(state["error_message"])
                return False

            try:
                resp = httpx.get(f"{LLAMA_BASE_URL}/health", timeout=1.0)
                if resp.status_code == 200:
                    with state_lock:
                        state["status"] = "running"
                        state["last_activity"] = time.time()
                    logger.info("[+] llama-server is ready and running on %s", LLAMA_BASE_URL)
                    return True
            except Exception:
                pass
            time.sleep(1.0)

        proc.terminate()
        with state_lock:
            state["status"] = "error"
            state["error_message"] = "llama-server startup timed out after 180s"
        logger.error(state["error_message"])
        return False

    except Exception as exc:
        with state_lock:
            state["status"] = "error"
            state["error_message"] = str(exc)
        logger.error("Error starting llama-server: %s", exc)
        return False
    finally:
        startup_lock.release()


def sleep_server() -> None:
    """Terminates active llama-server process to free VRAM during idle periods."""
    global state
    kill_all_llama_servers()
    with state_lock:
        state["process"] = None
        if state["active_model"]:
            state["status"] = "sleeping"
        else:
            state["status"] = "idle"
    logger.info("[*] Inactivity timeout: model put to sleep to free VRAM.")


async def idle_watchdog() -> None:
    """Background task putting model to sleep if idle for IDLE_TIMEOUT_SECONDS."""
    if IDLE_TIMEOUT_SECONDS <= 0:
        return
    while True:
        await asyncio.sleep(15.0)
        with state_lock:
            cur_status = state["status"]
            last_act = state["last_activity"]

        if cur_status == "running" and time.time() - last_act > IDLE_TIMEOUT_SECONDS:
            sleep_server()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MAIA Beacon on %s:%d (Llama base: %s, Ollama base: %s)", BEACON_HOST, BEACON_PORT, LLAMA_BASE_URL, OLLAMA_HOST)
    watchdog_task = asyncio.create_task(idle_watchdog())
    yield
    watchdog_task.cancel()
    kill_all_llama_servers()
    logger.info("MAIA Beacon shut down.")


app = FastAPI(title="MAIA Beacon", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_status():
    """Returns worker status, VRAM telemetry, active model, and available models."""
    total, used, free = optimizer.get_gpu_vram()
    gguf_models = get_available_gguf_models()

    with state_lock:
        return {
            "type": "beacon",
            "status": state["status"],
            "active_model": state["active_model"],
            "available_models": gguf_models,
            "active_mmproj": state.get("active_mmproj"),
            "active_context": state["active_context"],
            "active_thinking": state["active_thinking"],
            "config": state["config"],
            "error_message": state["error_message"],
            "gpu_vram": {
                "total_mib": total,
                "used_mib": used,
                "free_mib": free,
            },
            "pid": state["process"].pid if state["process"] and state["process"].poll() is None else None,
        }


@app.get("/api/models")
def list_models():
    """Returns all models (GGUF and local Ollama) available through this Beacon node."""
    gguf = get_available_gguf_models()
    ollama = get_local_ollama_models()
    all_models = sorted(list(set(gguf + ollama)))
    return {"models": all_models}


@app.post("/api/select")
def select_model(req: SelectionRequest, background_tasks: BackgroundTasks):
    """Dynamically loads or switches active GGUF model with specified context and parameters."""
    avail = get_available_gguf_models()
    clean_model = req.model[:-5] if req.model.lower().endswith(".gguf") else req.model
    
    # Case-insensitive lookup in available GGUF models
    matched_model = None
    for m in avail:
        if m.lower() == clean_model.lower():
            matched_model = m
            break
        elif len(clean_model) > 5 and (clean_model.lower() in m.lower() or m.lower() in clean_model.lower()):
            matched_model = m

    if not matched_model:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found in {MODELS_DIR}")

    clean_model = matched_model

    with state_lock:
        if state["status"] == "running" and state["active_model"] and state["active_model"].lower() == clean_model.lower():
            if state["active_context"] and state["active_context"] >= req.context_size:
                state["last_activity"] = time.time()
                return {"message": "Model already loaded and running", "status": "running", "active_model": clean_model}

    background_tasks.add_task(
        start_llama_server_task,
        clean_model,
        req.context_size,
        req.thinking or False,
        req.thinking_effort or "medium",
        req.mmproj,
    )
    return {"message": "Model selection started", "status": "starting", "active_model": clean_model}


@app.post("/api/stop")
def stop_model():
    """Stops active llama-server to free up VRAM immediately."""
    kill_all_llama_servers()
    with state_lock:
        state["process"] = None
        state["status"] = "idle"
        state["active_model"] = None
        state["active_mmproj"] = None
        state["active_context"] = None
        state["active_thinking"] = False
        state["config"] = None
        state["error_message"] = None
    return {"message": "Model stopped successfully"}


# --- OpenAI Proxy Endpoints ---

@app.get("/v1/models")
def list_v1_models():
    gguf = get_available_gguf_models()
    ollama = get_local_ollama_models()
    all_models = sorted(list(set(gguf + ollama)))
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
            for m in all_models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json() or {}
    model = str(data.get("model", ""))
    
    ollama_models = get_local_ollama_models()
    if model in ollama_models and OLLAMA_HOST:
        return await proxy_to_ollama("v1/chat/completions", request, data)
        
    return await proxy_to_llama_server("v1/chat/completions", request, data)


@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json() or {}
    model = str(data.get("model", ""))
    ollama_models = get_local_ollama_models()
    if model in ollama_models and OLLAMA_HOST:
        return await proxy_to_ollama("v1/completions", request, data)
    return await proxy_to_llama_server("v1/completions", request, data)


async def proxy_to_ollama(path: str, request: Request, json_data: dict):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    url = f"{OLLAMA_HOST}/{path}"
    is_stream = json_data.get("stream", False)

    if is_stream:
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=json_data,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                json=json_data,
            )
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


async def proxy_to_llama_server(path: str, request: Request, json_data: dict):
    # If a startup is currently in progress, wait for it to complete
    start_wait = time.time()
    while True:
        with state_lock:
            cur_status = state["status"]
            cur_model = state["active_model"]
            cur_ctx = state["active_context"] or 16384
            cur_think = state["active_thinking"]
            cur_effort = state["active_thinking_effort"] or "medium"
            cur_mmproj = state.get("active_mmproj")

        if cur_status != "starting":
            break
        if time.time() - start_wait > 120.0:
            break
        await asyncio.sleep(0.5)

    # If sleeping, wake up
    if cur_status == "sleeping" and cur_model:
        logger.info("[*] Waking up sleeping model '%s'...", cur_model)
        await asyncio.to_thread(start_llama_server_task, cur_model, cur_ctx, cur_think, cur_effort, cur_mmproj)
        with state_lock:
            cur_status = state["status"]

    # If idle, try to start the requested model automatically
    requested_model = str(json_data.get("model", ""))
    if requested_model.lower().endswith(".gguf"):
        requested_model = requested_model[:-5]
    avail = get_available_gguf_models()
    matched_req = None
    for m in avail:
        if m.lower() == requested_model.lower():
            matched_req = m
            break
        elif len(requested_model) > 5 and (requested_model.lower() in m.lower() or m.lower() in requested_model.lower()):
            matched_req = m

    if cur_status == "idle" and matched_req:
        logger.info("[*] Auto-starting requested model '%s' on demand...", matched_req)
        await asyncio.to_thread(start_llama_server_task, matched_req, 16384)
        with state_lock:
            cur_status = state["status"]

    if cur_status != "running":
        raise HTTPException(status_code=503, detail=f"Llama-server is not ready (Status: {cur_status}).")

    with state_lock:
        state["last_activity"] = time.time()

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    url = f"{LLAMA_BASE_URL}/{path}"
    is_stream = json_data.get("stream", False)

    if is_stream:
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=json_data,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                json=json_data,
            )
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


if __name__ == "__main__":
    GREEN = "\033[92m"
    RESET = "\033[0m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

    LOGO = fr"""
{GREEN}      ###**###      {RESET}
{GREEN}   ##.        .##   {RESET}
{GREEN} ##              ## {RESET}
{GREEN} #.              .# {RESET}   {CYAN}{BOLD} __  __    _    ___    _         ____  _____    _    ____ ___  _   _ {RESET}
{GREEN}#= :####:  :####: =#{RESET}   {CYAN}{BOLD}|  \/  |  / \  |_ _|  / \       | __ )| ____|  / \  / ___/ _ \| \ | |{RESET}
{GREEN}#: ######  ###### :#{RESET}   {CYAN}{BOLD}| |\/| | / _ \  | |  / _ \      |  _ \|  _|   / _ \| |  | | | |  \| |{RESET}
{GREEN}#= :####:  :####: =#{RESET}   {CYAN}{BOLD}| |  | |/ ___ \ | | / ___ \     | |_) | |___ / ___ \ |__| |_| | |\  |{RESET}
{GREEN} #                # {RESET}   {CYAN}{BOLD}|_|  |_/_/   \_\___/_/   \_\    |____/|_____/_/   \_\____\___/|_| \_|{RESET}
{GREEN} ##              ## {RESET}
{GREEN}   ##          ##   {RESET}
{GREEN}     ####++####     {RESET}
"""
    print(LOGO)
    print(f"[*] Starting MAIA Beacon on {BEACON_HOST}:{BEACON_PORT}...")
    uvicorn.run("app:app", host=BEACON_HOST, port=BEACON_PORT, reload=False)
