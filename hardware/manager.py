"""
MAIA Beacon - Hardware Manager.
Unified facade for GPU VRAM telemetry, NPU inspection, and hardware capacity checks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import hardware.gpu as gpu_hw
import hardware.npu as npu_hw


class HardwareManager:
    """Unified entrypoint for physical hardware resources (GPU & Intel NPU)."""

    def get_gpu_vram(self, llama_server_exe: Optional[str] = None) -> Tuple[int, int, int]:
        """Returns (total_mib, used_mib, free_mib) for GPU VRAM."""
        return gpu_hw.get_gpu_vram(llama_server_exe)

    def get_max_vram_ratio(self) -> float:
        """Returns maximum allowed VRAM usage ratio (e.g. 0.95)."""
        return gpu_hw.get_max_vram_ratio()

    def get_npu_info(self) -> Dict[str, Any]:
        """Returns Intel AI Boost NPU status and capabilities."""
        return npu_hw.get_npu_info()

    def get_telemetry(self, llama_server_exe: Optional[str] = None) -> Dict[str, Any]:
        """Consolidates GPU VRAM and NPU telemetry into a single status dict."""
        total, used, free = self.get_gpu_vram(llama_server_exe)
        max_ratio = self.get_max_vram_ratio()
        npu_info = self.get_npu_info()

        return {
            "npu": npu_info,
            "gpu_vram": {
                "total_mib": total,
                "used_mib": used,
                "free_mib": free,
                "max_percent": round(max_ratio * 100.0, 1),
                "max_limit_mib": round(total * max_ratio, 1),
            },
        }


# Singleton instance
hw_manager = HardwareManager()
