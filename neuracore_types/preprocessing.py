"""Configuration models for slot-based preprocessing pipelines."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from neuracore_types.nc_data import DataType


class PreprocessingPhase(str, Enum):
    """When a preprocessing method is allowed to run."""

    TRAINING = "training"
    INFERENCE = "inference"


class PreProcessingConfiguration(BaseModel):
    """Slot-based preprocessing configuration.

    The configuration is keyed by DataType and slot index, where slot indices
    come from the cross-embodiment description. Each slot has an ordered list
    of preprocessing methods to apply.

    Example (conceptual structure):

    steps = {
        DataType.RGB_IMAGES: {
            0: [ResizePad(...), NormalizeResnet(...),
                NormalizeResNet(...)],
            1: [...],
        },
        DataType.DEPTH_IMAGES: {
            0: [...],
        },
    }
    """

    steps: dict[DataType, dict[int, list[Any]]] = Field(
        default_factory=dict,
        description="Per-data-type, per-slot preprocessing pipelines.",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("steps", mode="before")
    @classmethod
    def _steps_slot_keys_to_int(cls, v: Any) -> dict[DataType, dict[int, list[Any]]]:
        """Accept slot indices as strings (from JSON) and convert to int."""
        # if v is not a dict, return it as is
        # so pydantic can handle/raise validation errors.
        if not isinstance(v, dict):
            return v
        out: dict[DataType, dict[int, list[Any]]] = {}
        for dt_key, slots in v.items():
            if not isinstance(slots, dict):
                out[dt_key] = slots  # type: ignore
                continue
            out[dt_key] = {
                int(slot_k) if isinstance(slot_k, str) else slot_k: slot_v
                for slot_k, slot_v in slots.items()
            }
        return out  # type: ignore


__all__ = [
    "PreprocessingPhase",
    "PreProcessingConfiguration",
]
