"""
MAIA Beacon - GPU Hardware Management & VRAM Telemetry.
Provides VRAM detection across NVIDIA, AMD, and Intel GPUs via Vulkan, nvidia-smi, rocm-smi, xpu-smi, or WMI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Optional, Tuple

DEFAULT_VRAM_MARGIN = 500
DEFAULT_MAX_VRAM_RATIO = 0.95


def get_max_vram_ratio() -> float:
    """
    Returns the maximum allowable VRAM usage ratio (e.g. 0.95 for 95%).
    Parsed from MAX_VRAM_PERCENT (or MAX_VRAM_USAGE). Defaults to 0.95 (95%).
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
        return max(0.10, min(1.0, ratio))
    except (ValueError, TypeError):
        return DEFAULT_MAX_VRAM_RATIO


def get_gpu_vram(llama_server_exe: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Returns (total_vram_mib, used_vram_mib, free_vram_mib) across GPUs.
    Tries querying via llama-server.exe --list-devices (Vulkan/CUDA memory directly),
    nvidia-smi, rocm-smi, xpu-smi, Windows WMI, or DRM sysfs on Linux.
    Returns (0, 0, 0) if no GPU VRAM is detected.
    """
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

    # Try Linux sysfs DRM
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

    # Try Windows WMI / CIM
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
                    if any(v in name.lower() for v in ["parsec", "spacedesk", "rdp", "virtual", "basic display"]):
                        continue
                    if ram_bytes and isinstance(ram_bytes, (int, float)) and ram_bytes > 0:
                        tot_mib = int(ram_bytes / (1024 * 1024))
                        if tot_mib > 0:
                            return tot_mib, 0, tot_mib
        except Exception:
            pass

    return 0, 0, 0
