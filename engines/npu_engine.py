"""
MAIA Beacon - Intel NPU Engine Provider (OpenVINO GenAI).
Manages OpenVINO IR models, compilation on Intel AI Boost NPU, and SSE generation.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("maia.beacon.npu")

# Internal NPU state and threading lock
npu_state: Dict[str, Any] = {
    "pipeline": None,
    "tokenizer": None,
    "model_name": None,
    "lock": threading.Lock(),
}


def get_available_openvino_models(models_dir: Path) -> List[str]:
    """Scans models_dir for OpenVINO IR model directories (containing openvino_model.xml)."""
    models: List[str] = []
    if models_dir.exists():
        for item in os.listdir(models_dir):
            item_path = models_dir / item
            if item_path.is_dir() and (item_path / "openvino_model.xml").exists():
                models.append(item)
    return sorted(models)


def is_openvino_model(model_name: str, models_dir: Path) -> bool:
    """Returns True if model_name corresponds to an OpenVINO IR model directory."""
    avail = get_available_openvino_models(models_dir)
    clean = model_name[:-5] if model_name.lower().endswith(".gguf") else model_name
    for m in avail:
        if m.lower() == clean.lower():
            return True
    return False


def unload_npu_model() -> None:
    """Frees the OpenVINO NPU pipeline from memory."""
    global npu_state
    with npu_state["lock"]:
        npu_state["pipeline"] = None
        npu_state["tokenizer"] = None
        npu_state["model_name"] = None
    gc.collect()


def load_npu_model(
    model_name: str,
    models_dir: Path,
    state: Dict[str, Any],
    state_lock: threading.Lock,
    before_load_fn: Optional[Callable[[], None]] = None,
) -> bool:
    """Compiles and loads an OpenVINO IR model onto the Intel AI Boost NPU."""
    global npu_state

    clean_name = model_name[:-5] if model_name.lower().endswith(".gguf") else model_name
    matched_dir = None
    for m in get_available_openvino_models(models_dir):
        if m.lower() == clean_name.lower():
            matched_dir = models_dir / m
            clean_name = m
            break

    if not matched_dir or not matched_dir.exists():
        logger.error("OpenVINO model '%s' not found in %s", model_name, models_dir)
        return False

    with state_lock:
        state["status"] = "starting"
        state["error_message"] = None

    if before_load_fn:
        before_load_fn()

    try:
        import openvino_genai as og

        with npu_state["lock"]:
            if npu_state["model_name"] == clean_name and npu_state["pipeline"] is not None:
                with state_lock:
                    state["status"] = "running"
                    state["active_engine"] = "npu"
                    state["active_model"] = clean_name
                    state["last_activity"] = time.time()
                return True

            logger.info("[NPU] Compiling & loading model '%s' on Intel AI Boost NPU...", clean_name)
            t0 = time.time()
            pipe = og.LLMPipeline(str(matched_dir), "NPU")
            tok = pipe.get_tokenizer()
            npu_state["pipeline"] = pipe
            npu_state["tokenizer"] = tok
            npu_state["model_name"] = clean_name
            load_dur = time.time() - t0
            logger.info("[NPU] Model '%s' successfully ready on NPU in %.2fs", clean_name, load_dur)

        with state_lock:
            state["status"] = "running"
            state["active_engine"] = "npu"
            state["active_model"] = clean_name
            state["last_activity"] = time.time()
            state["error_message"] = None
        return True

    except Exception as exc:
        logger.error("[NPU] Failed to load model on NPU: %s", exc)
        with state_lock:
            state["status"] = "error"
            state["error_message"] = f"NPU initialization error: {exc}"
        return False


async def generate_npu(
    request: Request,
    json_data: dict,
    state: Dict[str, Any],
    state_lock: threading.Lock,
):
    """Executes chat or completion generation on Intel AI Boost NPU via OpenVINO GenAI."""
    import openvino_genai as og

    with state_lock:
        state["last_activity"] = time.time()

    pipe = npu_state.get("pipeline")
    tok = npu_state.get("tokenizer")
    if not pipe or not tok:
        raise HTTPException(status_code=503, detail="NPU pipeline is not ready.")

    messages = json_data.get("messages", [])
    prompt = json_data.get("prompt", "")
    stream = bool(json_data.get("stream", False))
    temperature = float(json_data.get("temperature", 0.7))
    max_tokens = int(json_data.get("max_tokens", json_data.get("max_completion_tokens", 1024)))

    if messages:
        try:
            formatted_prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            formatted_prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:\n"
    else:
        formatted_prompt = str(prompt)

    gen_config = og.GenerationConfig()
    gen_config.max_new_tokens = max_tokens
    gen_config.temperature = max(0.01, temperature)

    cmpl_id = f"chatcmpl-npu-{int(time.time()*1000)}"
    created_ts = int(time.time())
    model_name = npu_state.get("model_name") or "npu-model"

    if stream:
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def streamer(subword: str) -> bool:
            loop.call_soon_threadsafe(queue.put_nowait, subword)
            return False

        def run_generation():
            try:
                with npu_state["lock"]:
                    p = npu_state.get("pipeline")
                    if p:
                        p.generate(formatted_prompt, config=gen_config, streamer=streamer)
            except Exception as exc:
                logger.error("[NPU Stream] Error during generation: %s", exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run_generation, daemon=True).start()

        async def sse_generator() -> AsyncGenerator[bytes, None]:
            first_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n".encode("utf-8")

            while True:
                token = await queue.get()
                if token is None:
                    break
                chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

            done_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    else:
        def run_sync():
            with npu_state["lock"]:
                p = npu_state.get("pipeline")
                if not p:
                    raise RuntimeError("NPU pipeline unavailable")
                return p.generate(formatted_prompt, config=gen_config)

        res = await asyncio.to_thread(run_sync)
        res_str = str(res)

        return JSONResponse({
            "id": cmpl_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": res_str,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(formatted_prompt.split()),
                "completion_tokens": len(res_str.split()),
                "total_tokens": len(formatted_prompt.split()) + len(res_str.split()),
            },
        })
