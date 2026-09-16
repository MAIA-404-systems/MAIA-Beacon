"""
MAIA Beacon - Hardware Abstraction Layer.
Contains hardware telemetry, VRAM monitoring, and NPU inspection.
"""

from hardware.manager import hw_manager, HardwareManager

__all__ = ["hw_manager", "HardwareManager"]
