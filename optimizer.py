"""
VRAM Optimizer and GGUF Architecture Analyzer for MAIA Beacon.
Calculates optimal GPU layer offloading and KV cache quantization based on available VRAM.
"""

import os
import struct
import subprocess

# Default margin in MiB left free for OS & display
DEFAULT_VRAM_MARGIN = 500


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


def get_gpu_vram():
    """Returns (total_vram_mib, used_vram_mib, free_vram_mib) using nvidia-smi."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        parts = res.stdout.strip().split(",")
        if len(parts) >= 3:
            return int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
    except Exception as e:
        print(f"Warning: Could not query VRAM via nvidia-smi: {e}")
    return 12288, 0, 12288


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


def find_optimal_config(model_path: str, target_ctx: int, vram_margin: int = DEFAULT_VRAM_MARGIN, mmproj_path: str = None):
    """Finds the optimal settings using mathematical simulation."""
    props = extract_model_properties(model_path)
    if not props:
        return None

    total_vram, _, _ = get_gpu_vram()

    mmproj_size_mib = 0
    if mmproj_path and os.path.exists(mmproj_path):
        base_size = os.path.getsize(mmproj_path) / (1024 * 1024)
        mmproj_size_mib = base_size * 3.0
        vram_margin += 500

    safe_vram_limit = total_vram - vram_margin - mmproj_size_mib

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

    if best_config:
        return {
            "model_properties": props,
            "best_config": best_config,
            "total_vram_mib": total_vram,
            "safe_vram_limit": safe_vram_limit
        }
    return None
