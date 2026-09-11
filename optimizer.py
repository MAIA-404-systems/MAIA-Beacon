"""
VRAM Optimizer and GGUF Architecture Analyzer for MAIA Beacon.
Calculates optimal GPU layer offloading and KV cache quantization based on available VRAM.
"""

import json
import os
from pathlib import Path
import re
import struct
import subprocess
from typing import Optional, Union

# Load environment variables from .env file if available
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

# Default margin in MiB left free for OS & display
DEFAULT_VRAM_MARGIN = 500
# Default max VRAM usage ratio (95% safety default)
DEFAULT_MAX_VRAM_RATIO = 0.95


def get_max_vram_ratio() -> float:
    """
    Returns the maximum allowable VRAM usage ratio (e.g., 0.95 for 95%).
    Parsed from MAX_VRAM_PERCENT (or MAX_VRAM_USAGE). Defaults to 0.95 (95%) if empty, missing, or invalid.
    Supports formats: "95%", "95", "0.95", etc.
    """
    raw = os.getenv("MAX_VRAM_PERCENT", os.getenv("MAX_VRAM_USAGE", ""))
    if raw is None:
        return DEFAULT_MAX_VRAM_RATIO
    raw_str = str(raw).strip()
    if not raw_str:
        return DEFAULT_MAX_VRAM_RATIO

    has_percent = "%" in raw_str
    cleaned = raw_str.replace("%", "").strip().strip("'\"")
    if not cleaned:
        return DEFAULT_MAX_VRAM_RATIO

    try:
        val = float(cleaned)
        if has_percent or val > 1.0:
            ratio = val / 100.0
        else:
            ratio = val
        # Clamp ratio between 0.10 (10%) and 1.0 (100%)
        return max(0.10, min(1.0, ratio))
    except (ValueError, TypeError):
        return DEFAULT_MAX_VRAM_RATIO


def read_gguf_metadata(file_path: str):
    """Parses GGUF metadata keys directly (offline, extremely fast)."""
    metadata = {}
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return None

            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def read_string():
                length = struct.unpack("<Q", f.read(8))[0]
                return f.read(length).decode("utf-8", errors="ignore")

            def read_val(val_type):
                if val_type == 0: return struct.unpack("<B", f.read(1))[0]
                elif val_type == 1: return struct.unpack("<b", f.read(1))[0]
                elif val_type == 2: return struct.unpack("<H", f.read(2))[0]
                elif val_type == 3: return struct.unpack("<h", f.read(2))[0]
                elif val_type == 4: return struct.unpack("<I", f.read(4))[0]
                elif val_type == 5: return struct.unpack("<i", f.read(4))[0]
                elif val_type == 6: return struct.unpack("<f", f.read(4))[0]
                elif val_type == 7: return struct.unpack("<?", f.read(1))[0]
                elif val_type == 8: return read_string()
                elif val_type == 9:
                    item_type = struct.unpack("<I", f.read(4))[0]
                    length = struct.unpack("<Q", f.read(8))[0]
                    return [read_val(item_type) for _ in range(length)]
                elif val_type == 10: return struct.unpack("<Q", f.read(8))[0]
                elif val_type == 11: return struct.unpack("<q", f.read(8))[0]
                elif val_type == 12: return struct.unpack("<d", f.read(8))[0]
                return None

            for _ in range(kv_count):
                key = read_string()
                val_type = struct.unpack("<I", f.read(4))[0]
                val = read_val(val_type)
                metadata[key] = val
    except Exception as e:
        print(f"Warning: Failed parsing GGUF: {e}")
    return metadata


def get_gpu_vram(llama_server_exe: Optional[str] = None):
    """
    Returns (total_vram_mib, used_vram_mib, free_vram_mib) across NVIDIA, AMD, and Intel GPUs.
    Tries querying via llama-server.exe --list-devices (Vulkan/CUDA/Kompute/SYCL).
    Falls back to nvidia-smi, rocm-smi, xpu-smi, Windows WMI, or DRM sysfs on Linux.
    Returns (0, 0, 0) if no GPU VRAM is detected.
    """
    # Try querying via llama-server --list-devices first (queries Vulkan/CUDA/Kompute memory directly)
    llama_exe = llama_server_exe or os.getenv("LLAMA_SERVER_EXE")
    if llama_exe and os.path.exists(llama_exe):
        try:
            cmd = [llama_exe, "--list-devices"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0)
            if res.returncode == 0:
                matches = re.findall(r"\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)", res.stdout)
                if matches:
                    tot_mib = int(matches[0][0])
                    free_mib = int(matches[0][1])
                    used_mib = max(0, tot_mib - free_mib)
                    if tot_mib > 0:
                        return tot_mib, used_mib, free_mib
        except Exception:
            pass

    # Try NVIDIA (nvidia-smi)
    nvsmi_candidates = ["nvidia-smi"]
    if os.name == "nt":
        nvsmi_candidates.extend([
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
        ])
    for nvsmi_bin in nvsmi_candidates:
        try:
            cmd = [nvsmi_bin, "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            parts = res.stdout.strip().split(",")
            if len(parts) >= 3:
                return int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
        except Exception:
            pass

    # Try AMD (rocm-smi)
    try:
        cmd = ["rocm-smi", "--showmeminfo", "vram", "--json"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
        for card_id, card_data in data.items():
            tot = card_data.get("VRAM Total Memory (B)")
            used = card_data.get("VRAM Total Used Memory (B)")
            if tot and used:
                tot_mib = int(int(tot) / (1024 * 1024))
                used_mib = int(int(used) / (1024 * 1024))
                return tot_mib, used_mib, max(0, tot_mib - used_mib)
    except Exception:
        pass

    # Try Intel (xpu-smi)
    try:
        cmd = ["xpu-smi", "discovery", "-j"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = json.loads(res.stdout)
        device_list = data if isinstance(data, list) else data.get("device_list", [])
        for dev in device_list:
            mem = dev.get("memory_physical_size_byte")
            if mem and int(mem) > 0:
                tot_mib = int(int(mem) / (1024 * 1024))
                return tot_mib, 0, tot_mib
    except Exception:
        pass

    # Try Linux sysfs DRM (AMD / Intel / NVIDIA Linux fallback)
    if os.name != "nt" and os.path.exists("/sys/class/drm"):
        try:
            for card in sorted(os.listdir("/sys/class/drm")):
                if not card.startswith("card"):
                    continue
                vram_file = os.path.join("/sys/class/drm", card, "device", "mem_info_vram_total")
                if os.path.exists(vram_file):
                    with open(vram_file, "r") as f:
                        vram_bytes = int(f.read().strip())
                        if vram_bytes > 0:
                            tot_mib = int(vram_bytes / (1024 * 1024))
                            return tot_mib, 0, tot_mib
        except Exception:
            pass

    # Try Windows WMI / CIM (Generic for Intel Graphics, AMD, NVIDIA on Windows)
    if os.name == "nt":
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json",
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            if res.stdout.strip():
                items = json.loads(res.stdout)
                if isinstance(items, dict):
                    items = [items]
                for gpu in items:
                    name = str(gpu.get("Name", ""))
                    ram_bytes = gpu.get("AdapterRAM")
                    # Ignore virtual display adapters (e.g. Parsec, spacedesk, RDP)
                    if any(v in name.lower() for v in ["parsec", "spacedesk", "rdp", "virtual", "basic display"]):
                        continue
                    if ram_bytes and isinstance(ram_bytes, (int, float)) and ram_bytes > 0:
                        tot_mib = int(ram_bytes / (1024 * 1024))
                        if tot_mib > 0:
                            return tot_mib, 0, tot_mib
        except Exception as e:
            print(f"Warning: Could not query VRAM via Windows WMI: {e}")

    print("Warning: No GPU VRAM detected (NVIDIA/AMD/Intel). Defaulting to 0 MiB VRAM (CPU mode).")
    return 0, 0, 0


def extract_model_properties(model_path: str):
    """Extracts layers, experts, and details needed to compute KV Cache memory."""
    meta = read_gguf_metadata(model_path)
    if not meta:
        return None

    arch = meta.get("general.architecture", "llama")
    layers = meta.get(f"{arch}.block_count", 32)
    head_count = meta.get(f"{arch}.attention.head_count", 32)

    head_count_kv_val = meta.get(f"{arch}.attention.head_count_kv", 8)
    if isinstance(head_count_kv_val, list):
        total_kv_heads = sum(head_count_kv_val)
    else:
        total_kv_heads = layers * head_count_kv_val

    embedding_length = meta.get(f"{arch}.embedding_length", 4096)
    head_dim = embedding_length // head_count

    expert_count = meta.get(f"{arch}.expert_count", 0)
    is_moe = expert_count > 0
    file_size_mib = os.path.getsize(model_path) / (1024 * 1024)

    return {
        "layers": layers,
        "experts": expert_count,
        "is_moe": is_moe,
        "total_kv_heads": total_kv_heads,
        "head_dim": head_dim,
        "architecture": arch,
        "file_size_mib": file_size_mib,
        "filename": os.path.basename(model_path)
    }


def calculate_kv_cache_size_mib(total_kv_heads: int, head_dim: int, num_tokens: int, bytes_per_elem: float) -> float:
    """Calculates KV Cache memory size in MiB."""
    total_elements = 2 * total_kv_heads * head_dim * num_tokens
    bytes_total = total_elements * bytes_per_elem
    return bytes_total / (1024 * 1024)


def simulate_vram_and_speed(props: dict, ngl: int, ncmoe: int, cache_bytes: float, target_ctx: int):
    """Simulates VRAM usage and generation speed mathematically."""
    model_size = props["file_size_mib"]

    if props["is_moe"]:
        non_expert_vram = model_size * 0.25
        expert_total_vram = model_size * 0.75
        expert_weight = expert_total_vram / props["experts"]
        weights_on_gpu = non_expert_vram + (props["experts"] - ncmoe) * expert_weight
    else:
        weights_on_gpu = model_size * (ngl / props["layers"])

    kv_vram = calculate_kv_cache_size_mib(props["total_kv_heads"], props["head_dim"], target_ctx, cache_bytes)
    total_gpu_vram = weights_on_gpu + kv_vram

    if props["is_moe"]:
        gpu_ratio = (props["experts"] - ncmoe) / props["experts"]
        speed = 10.0 + 30.0 * (gpu_ratio ** 1.5)
    else:
        gpu_ratio = ngl / props["layers"]
        speed = 5.0 + 35.0 * (gpu_ratio ** 2)

    return total_gpu_vram, speed


def find_optimal_config(
    model_path: str,
    target_ctx: int,
    max_vram_ratio: Optional[Union[float, int]] = None,
    mmproj_path: Optional[str] = None,
    vram_margin: Optional[int] = None,
    llama_server_exe: Optional[str] = None,
):
    """Finds the optimal settings using mathematical simulation."""
    props = extract_model_properties(model_path)
    if not props:
        return None

    # Handle backward compatibility if 3rd positional argument was vram_margin (e.g. 500 MiB)
    if isinstance(max_vram_ratio, (int, float)) and max_vram_ratio > 100:
        if vram_margin is None:
            vram_margin = int(max_vram_ratio)
        max_vram_ratio = None

    if max_vram_ratio is None:
        effective_ratio = get_max_vram_ratio()
    else:
        try:
            if isinstance(max_vram_ratio, str):
                has_percent = "%" in max_vram_ratio
                cleaned = max_vram_ratio.replace("%", "").strip()
                val = float(cleaned)
                effective_ratio = val / 100.0 if (has_percent or val > 1.0) else val
            elif max_vram_ratio > 1.0:
                effective_ratio = float(max_vram_ratio) / 100.0
            else:
                effective_ratio = float(max_vram_ratio)
            effective_ratio = max(0.0, min(1.0, effective_ratio))
        except (ValueError, TypeError):
            effective_ratio = get_max_vram_ratio()

    total_vram, _, _ = get_gpu_vram(llama_server_exe)

    mmproj_size_mib = 0.0
    if mmproj_path and os.path.exists(mmproj_path):
        base_size = os.path.getsize(mmproj_path) / (1024 * 1024)
        mmproj_size_mib = base_size * 3.0

    max_allowed_vram = total_vram * effective_ratio
    safe_vram_limit = max_allowed_vram - mmproj_size_mib

    if vram_margin is not None:
        safe_vram_limit = min(safe_vram_limit, total_vram - vram_margin - mmproj_size_mib)

    safe_vram_limit = max(0.0, safe_vram_limit)

    cache_types = [
        {"k": "f16", "v": "f16", "bytes": 2.0, "desc": "Standard (Uncompressed)"},
        {"k": "q4_0", "v": "q4_0", "bytes": 0.5625, "desc": "Standard Quantized (4-bit)"},
        {"k": "turbo4", "v": "turbo3", "bytes": 0.4375, "desc": "Turboquant (4-bit K, 3-bit V)"},
        {"k": "turbo3", "v": "turbo3", "bytes": 0.375, "desc": "Turboquant High Compression (3-bit)"}
    ]

    best_config = None
    best_speed = -1.0

    for cache in cache_types:
        target_kv_size = calculate_kv_cache_size_mib(props["total_kv_heads"], props["head_dim"], target_ctx, cache["bytes"])

        if target_kv_size + (props["file_size_mib"] * 0.25 if props["is_moe"] else 0) > safe_vram_limit:
            continue

        opt_ngl = 0
        opt_ncmoe = 0
        opt_speed = 0.0
        opt_vram = 0
        trial_success = False

        if props["is_moe"]:
            low = 0
            high = props["experts"]
            best_trial_speed = -1.0

            while low <= high:
                mid = (low + high) // 2
                sim_vram, sim_speed = simulate_vram_and_speed(props, 999, mid, cache["bytes"], target_ctx)
                if sim_vram <= safe_vram_limit:
                    trial_success = True
                    if sim_speed > best_trial_speed:
                        best_trial_speed = sim_speed
                        opt_ngl = 999
                        opt_ncmoe = mid
                        opt_speed = sim_speed
                        opt_vram = sim_vram
                    high = mid - 1
                else:
                    low = mid + 1
        else:
            low = 0
            high = props["layers"]
            best_trial_speed = -1.0

            while low <= high:
                mid = (low + high) // 2
                sim_vram, sim_speed = simulate_vram_and_speed(props, mid, 0, cache["bytes"], target_ctx)
                if sim_vram <= safe_vram_limit:
                    trial_success = True
                    if sim_speed > best_trial_speed:
                        best_trial_speed = sim_speed
                        opt_ngl = mid
                        opt_ncmoe = 0
                        opt_speed = sim_speed
                        opt_vram = sim_vram
                    low = mid + 1
                else:
                    high = mid - 1

        if trial_success and opt_speed > best_speed:
            best_speed = opt_speed
            best_config = {
                "ngl": opt_ngl,
                "ncmoe": opt_ncmoe,
                "cache_k": cache["k"],
                "cache_v": cache["v"],
                "speed": opt_speed,
                "vram": opt_vram
            }

    if not best_config:
        # Fallback to CPU-only execution (ngl=0) when no layers fit in VRAM or total_vram is 0
        best_config = {
            "ngl": 0,
            "ncmoe": props["experts"] if props["is_moe"] else 0,
            "cache_k": "f16",
            "cache_v": "f16",
            "speed": 5.0,
            "vram": 0.0,
        }

    return {
        "model_properties": props,
        "best_config": best_config,
        "total_vram_mib": total_vram,
        "safe_vram_limit": safe_vram_limit,
        "max_vram_percent": round(effective_ratio * 100.0, 1),
    }
