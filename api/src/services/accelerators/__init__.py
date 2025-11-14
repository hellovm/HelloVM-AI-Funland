"""
Hardware acceleration backends
"""

from .base import BaseAccelerator
from .cpu_accelerator import CPUAccelerator
from .openvino_accelerator import OpenVINOAccelerator
from .cuda_accelerator import CUDAAccelerator

__all__ = [
    "BaseAccelerator",
    "CPUAccelerator", 
    "OpenVINOAccelerator",
    "CUDAAccelerator"
]