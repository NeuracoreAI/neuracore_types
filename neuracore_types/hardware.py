"""Shared hardware type definitions."""

from enum import Enum


class GPUType(str, Enum):
    """GPU types available for training and deployment."""

    NVIDIA_H100_80GB = "NVIDIA_H100_80GB"
    NVIDIA_A100_80GB = "NVIDIA_A100_80GB"
    NVIDIA_TESLA_A100 = "NVIDIA_TESLA_A100"
    NVIDIA_TESLA_V100 = "NVIDIA_TESLA_V100"
    NVIDIA_TESLA_P100 = "NVIDIA_TESLA_P100"
    NVIDIA_TESLA_T4 = "NVIDIA_TESLA_T4"
    NVIDIA_TESLA_P4 = "NVIDIA_TESLA_P4"
    NVIDIA_L4 = "NVIDIA_L4"
