"""Configuration models for slot-based preprocessing pipelines.

These models describe how to preprocess different data types (e.g. RGB and
depth images) on a per-slot basis, where slots come from the cross-embodiment
description (robot_id, DataType, slot_index → stream name).

The configuration is phase-aware so that training and inference can share the
same structure while selectively enabling or disabling individual steps.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neuracore_types.nc_data import DataType

if TYPE_CHECKING:
    from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData


class PreprocessingPhase(str, Enum):
    """When a preprocessing method should be applied."""

    TRAIN = "train"
    INFERENCE = "inference"
    TRAIN_INFERENCE = "train_inference"


class PreProcessingMethod(BaseModel):
    """Single preprocessing operation in a slot pipeline.

    Attributes:
        name: Identifier of the preprocessing operation. For built-in methods
            this should match a known registry key (e.g. "resize_pad").
            The special value "custom" can be used together with
            ``custom_callable`` to point to user-defined functions.
        phase: When this method should be applied. Defaults to both train and
            inference.
        args: Keyword arguments passed to the underlying implementation.
        custom_callable: Optional dotted path to a user-defined preprocessing
            function. Only valid when ``name == "custom"``.
    """

    name: str = Field(..., description="Identifier of the preprocessing op.")
    phase: PreprocessingPhase = Field(
        default=PreprocessingPhase.TRAIN_INFERENCE,
        description="When to apply this step.",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the preprocessing op.",
    )
    custom_callable: str | None = Field(
        default=None,
        description="Optional dotted path to a custom preprocessing function.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_custom_callable(self) -> PreProcessingMethod:
        """Ensure custom_callable is only used with name='custom'."""
        if self.custom_callable is not None and self.name != "custom":
            raise ValueError(
                "custom_callable is only allowed when name='custom'. "
                f"Got name='{self.name}'."
            )
        return self


class MethodSpec(BaseModel):
    """Runtime preprocessing method specification."""

    handler: Callable[..., BatchedNCData]
    allowed_data_types: set[DataType]

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class PreProcessingConfiguration(BaseModel):
    """Slot-based preprocessing configuration.

    The configuration is keyed by DataType and slot index, where slot indices
    come from the cross-embodiment description. Each slot has an ordered list
    of preprocessing methods to apply.

    Example (conceptual structure):

    steps = {
        DataType.RGB_IMAGES: {
            0: [PreProcessingMethod(name="resize_pad", ...),
                PreProcessingMethod(name="normalize_resnet", ...)],
            1: [...],
        },
        DataType.DEPTH_IMAGES: {
            0: [...],
        },
    }
    """

    steps: dict[DataType, dict[int, list[PreProcessingMethod]]] = Field(
        default_factory=dict,
        description="Per-data-type, per-slot preprocessing pipelines.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("steps", mode="before")
    @classmethod
    def _steps_slot_keys_to_int(
        cls, v: Any
    ) -> dict[DataType, dict[int, list[PreProcessingMethod]]]:
        """Accept slot indices as strings (from JSON) and convert to int."""
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
    "PreProcessingMethod",
    "MethodSpec",
    "PreProcessingConfiguration",
]
