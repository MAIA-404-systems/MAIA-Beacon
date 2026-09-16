"""
MAIA Beacon - Intel NPU Telemetry & Inspection.
Detects presence and capabilities of Intel AI Boost NPU via OpenVINO Core.
"""

from __future__ import annotations

from typing import Any, Dict


def get_npu_info() -> Dict[str, Any]:
    """
    Detects presence and capabilities of the Intel NPU (Intel AI Boost) via OpenVINO.
    Returns status and hardware telemetry dictionary.
    """
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "NPU" in devices:
            full_name = "Intel(R) AI Boost"
            arch = "unknown"
            caps = []
            try:
                full_name = str(core.get_property("NPU", "FULL_DEVICE_NAME"))
            except Exception:
                pass
            try:
                arch = str(core.get_property("NPU", "DEVICE_ARCHITECTURE"))
            except Exception:
                pass
            try:
                caps = list(core.get_property("NPU", "OPTIMIZATION_CAPABILITIES"))
            except Exception:
                pass
            return {
                "available": True,
                "device": "NPU",
                "device_name": full_name,
                "architecture": arch,
                "capabilities": caps,
            }
    except Exception:
        pass

    return {
        "available": False,
        "device": None,
        "device_name": None,
        "architecture": None,
        "capabilities": [],
    }
