"""
MAIA Beacon - GPU / CPU Engine Provider (llama-server & Turboquant).
Manages local GGUF models, VRAM optimization, process lifecycle, and HTTP proxying to llama-server.exe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import psutil

import engines.gguf_parser as gguf_parser
import engines.npu_engine as npu_engine

logger = logging.getLogger("maia.beacon.llama")


def get_available_gguf_models(models_dir: Path) -> List[str]:
    """Scans models_dir for all available .gguf model files."""
    models: List[str] = []
    if models_dir.exists():
        for f in os.listdir(models_dir):
            if f.endswith(".gguf") and not any(k in f.lower() for k in ["mmproj", "projector"]):
                base_name = f[:-5]
                if base_name not in models:
                    models.append(base_name)
    return sorted(models)


def is_gguf_model(model_name: str, models_dir: Path) -> bool:
    """Returns True if the model matches an existing .gguf file in models_dir."""
    clean = model_name[:-5] if model_name.lower().endswith(".gguf") else model_name
    for m in get_available_gguf_models(models_dir):
        if m.lower() == clean.lower():
            return True
    return False


def find_matching_mmproj(model_filename: str, models_dir: Path) -> Optional[Path]:
    """Auto-detects matching vision multimodal projector (mmproj) for a model."""
    model_lower = model_filename.lower()
    if "mmproj" in model_lower:
        return None

    possible_base = None
    if "gemma" in model_lower:
        possible_base = "gemma"
    elif "qwen" in model_lower:
        possible_base = "qwen"
    elif "llava" in model_lower:
        possible_base = "llava"

    if possible_base and models_dir.exists():
        for f in os.listdir(models_dir):
            f_lower = f.lower()
            if f.endswith(".gguf") and "mmproj" in f_lower and possible_base in f_lower:
                logger.info("[Auto-Detect] Found matching vision projector: %s", f)
                return models_dir / f

    return None


def kill_all_llama_servers() -> None:
    """Terminates any running llama-server.exe process to free GPU VRAM."""
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() in ("llama-server.exe", "llama-server"):
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def read_process_output(process: subprocess.Popen, state: Dict[str, Any], state_lock: threading.Lock) -> None:
    """Reads stdout/stderr of llama-server and records lines in startup_log."""
    if not process.stdout:
        return
    while True:
        line = process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore").strip()
        logger.info("[llama-server] %s", decoded)
        with state_lock:
            state.setdefault("startup_log", []).append(decoded)
            if len(state["startup_log"]) > 200:
                state["startup_log"].pop(0)


def start_llama_server_task(
    model_filename: str,
    context_size: int,
    thinking: bool,
    thinking_effort: Union[str, int],
    mmproj_filename: Optional[str],
    models_dir: Path,
    llama_server_exe: Path,
    llama_server_host: str,
    llama_server_port: int,
    llama_load_mode: str,
    state: Dict[str, Any],
    state_lock: threading.Lock,
    startup_lock: threading.Lock,
) -> bool:
    """Optimizes VRAM and launches llama-server.exe for the target GGUF model."""
    llama_base_url = f"http://{llama_server_host}:{llama_server_port}"

    acquired = startup_lock.acquire(blocking=False)
    if not acquired:
        with state_lock:
            cur_model = state.get("active_model")
            cur_status = state.get("status")
        if cur_status == "starting" and (cur_model == model_filename or cur_model == f"{model_filename}.gguf"):
            logger.info("[*] llama-server is already starting model %s, ignoring duplicate.", model_filename)
            return True
        logger.info("[*] Waiting for previous startup lock to release...")
        startup_lock.acquire(blocking=True)

    try:
        with state_lock:
            state["status"] = "starting"
            state["active_engine"] = "llama-server"
            state["error_message"] = None
            state["startup_log"] = []

        target_name = model_filename if model_filename.endswith(".gguf") else f"{model_filename}.gguf"
        model_path = models_dir / target_name
        if not model_path.exists() and models_dir.exists():
            for f in os.listdir(models_dir):
                if f.lower() == target_name.lower() or f.lower().removesuffix(".gguf") == target_name.lower().removesuffix(".gguf"):
                    model_path = models_dir / f
                    target_name = f
                    break

        if not model_path.exists():
            with state_lock:
                state["status"] = "error"
                state["error_message"] = f"Model file '{target_name}' not found in {models_dir}"
            logger.error(state["error_message"])
            return False

        mmproj_path = None
        if mmproj_filename:
            candidate = models_dir / mmproj_filename
            if candidate.exists():
                mmproj_path = candidate
        else:
            mmproj_path = find_matching_mmproj(target_name, models_dir)

        kill_all_llama_servers()
        time.sleep(1.0)

        opt_res = gguf_parser.find_optimal_config(
            str(model_path),
            context_size,
            mmproj_path=str(mmproj_path) if mmproj_path else None,
            llama_server_exe=str(llama_server_exe),
        )
        if not opt_res:
            with state_lock:
                state["status"] = "error"
                state["error_message"] = "Optimization failed: model configuration does not fit in VRAM"
            logger.error(state["error_message"])
            return False

        opt_config = opt_res["best_config"]
        logger.info(
            "[Optimizer] Target VRAM limit: %.1f MiB (%.1f%% of %d MiB total). Optimal: ngl=%d, ncmoe=%d, cache_k=%s, cache_v=%s, estimated vram=%.1f MiB",
            opt_res["safe_vram_limit"],
            opt_res.get("max_vram_percent", 95.0),
            opt_res["total_vram_mib"],
            opt_config["ngl"],
            opt_config["ncmoe"],
            opt_config["cache_k"],
            opt_config["cache_v"],
            opt_config["vram"],
        )

        parent_dir = llama_server_exe.parent
        cmd = [
            str(llama_server_exe),
            "-m", str(model_path),
            "--n-gpu-layers", str(opt_config["ngl"]),
            "--n-cpu-moe", str(opt_config["ncmoe"]),
            "--cache-type-k", opt_config["cache_k"],
            "--cache-type-v", opt_config["cache_v"],
            "-c", str(context_size),
            "--load-mode", llama_load_mode,
            "--host", llama_server_host,
            "--port", str(llama_server_port),
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
            state["active_engine"] = "llama-server"
            state["active_model"] = target_name.removesuffix(".gguf")
            state["active_mmproj"] = mmproj_path.name if mmproj_path else None
            state["active_context"] = context_size
            state["active_thinking"] = thinking
            state["active_thinking_effort"] = thinking_effort
            state["config"] = opt_config
            state["last_activity"] = time.time()

        threading.Thread(target=read_process_output, args=(proc, state, state_lock), daemon=True).start()

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
                resp = httpx.get(f"{llama_base_url}/health", timeout=1.0)
                if resp.status_code == 200:
                    with state_lock:
                        state["status"] = "running"
                        state["last_activity"] = time.time()
                    logger.info("[+] llama-server is ready and running on %s", llama_base_url)
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


async def proxy_to_llama_server(
    path: str,
    request: Request,
    json_data: dict,
    llama_base_url: str,
    models_dir: Path,
    llama_server_exe: Path,
    llama_server_host: str,
    llama_server_port: int,
    llama_load_mode: str,
    state: Dict[str, Any],
    state_lock: threading.Lock,
    startup_lock: threading.Lock,
):
    """Handles proxying chat or completion requests to running llama-server instance."""
    start_wait = time.time()
    while True:
        with state_lock:
            cur_status = state["status"]
            cur_engine = state.get("active_engine")
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

    if cur_status == "sleeping" and cur_model:
        logger.info("[*] Waking up sleeping model '%s'...", cur_model)
        await asyncio.to_thread(
            start_llama_server_task,
            cur_model,
            cur_ctx,
            cur_think,
            cur_effort,
            cur_mmproj,
            models_dir,
            llama_server_exe,
            llama_server_host,
            llama_server_port,
            llama_load_mode,
            state,
            state_lock,
            startup_lock,
        )
        with state_lock:
            cur_status = state["status"]
            cur_engine = state.get("active_engine")
            cur_model = state.get("active_model")

    requested_model = str(json_data.get("model", ""))
    if requested_model.lower().endswith(".gguf"):
        requested_model = requested_model[:-5]
    avail = get_available_gguf_models(models_dir)
    matched_req = None
    for m in avail:
        if m.lower() == requested_model.lower():
            matched_req = m
            break
        elif len(requested_model) > 5 and (requested_model.lower() in m.lower() or m.lower() in requested_model.lower()):
            matched_req = m

    target_to_start = matched_req or cur_model or (avail[0] if avail else None)

    need_start = False
    if cur_status != "running" or cur_engine != "llama-server":
        need_start = True
    elif matched_req and cur_model and cur_model.lower() != matched_req.lower():
        need_start = True

    if need_start:
        if not target_to_start:
            raise HTTPException(status_code=404, detail="No GGUF models available to start llama-server.")

        npu_engine.unload_npu_model()

        logger.info("[*] Auto-starting requested GGUF model '%s' on GPU (previous engine: '%s')...", target_to_start, cur_engine)
        success = await asyncio.to_thread(
            start_llama_server_task,
            target_to_start,
            16384,
            False,
            "medium",
            None,
            models_dir,
            llama_server_exe,
            llama_server_host,
            llama_server_port,
            llama_load_mode,
            state,
            state_lock,
            startup_lock,
        )
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to start llama-server for model '{target_to_start}'")

        with state_lock:
            cur_status = state["status"]

    if cur_status != "running":
        raise HTTPException(status_code=503, detail=f"Llama-server is not ready (Status: {cur_status}).")

    with state_lock:
        state["last_activity"] = time.time()

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    url = f"{llama_base_url}/{path}"
    is_stream = json_data.get("stream", False)

    if is_stream:
        async def stream_generator() -> AsyncGenerator[bytes, None]:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream(
                        method=request.method,
                        url=url,
                        headers=headers,
                        json=json_data,
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except (httpx.ConnectError, httpx.RequestError) as exc:
                logger.error("[llama-server] Proxy streaming connection error: %s", exc)
                err_payload = json.dumps({"error": {"message": f"Connection to llama-server failed: {exc}", "type": "connect_error"}})
                yield f"data: {err_payload}\n\n".encode("utf-8")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=json_data,
                )
                return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))
        except (httpx.ConnectError, httpx.RequestError) as exc:
            logger.error("[llama-server] Proxy connection error: %s", exc)
            raise HTTPException(status_code=503, detail=f"Connection to llama-server failed: {exc}")
