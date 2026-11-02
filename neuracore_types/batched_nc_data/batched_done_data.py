"""Data models for batched episode done flag data."""

from typing import Any, Literal, cast

import torch
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class BatchedDoneData(BatchedNCData):
    """Batched episode done flag data."""

    type: Literal["BatchedDoneData"] = Field(
        default="BatchedDoneData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    done: torch.Tensor  # (B, T, 1) float32 of 0.0 or 1.0

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @field_validator("done", mode="before")
    @classmethod
    def decode_done(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode done field to torch.Tensor."""
        return cls._create_tensor_handlers("done")[0](v)

    @field_serializer("done", when_used="json")
    def serialize_done(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize done field to base64 string."""
        return self._create_tensor_handlers("done")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedDoneData from input nc_data.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        from neuracore_types.nc_data.done_data import DoneData

        done_data: DoneData = cast(DoneData, nc_data)
        done = (
            torch.tensor([float(done_data.done)], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return cls(done=done)

    @classmethod
    def from_nc_data_list(cls, nc_data_list: list[NCData]) -> "BatchedDoneData":
        """Create BatchedDoneData from list of DoneData.

        Args:
            nc_data_list: List of DoneData instances to convert

        Returns:
            BatchedDoneData with shape (1, T, 1)
        """
        from neuracore_types.nc_data.done_data import DoneData

        done_values = [float(cast(DoneData, nc).done) for nc in nc_data_list]
        # Shape: (1, T, 1)
        done_tensor = (
            torch.tensor(done_values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        )
        return cls(done=done_tensor)

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedDoneData":
        """Sample an example instance of BatchedDoneData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedDoneData: Sampled instance
        """
        return cls(done=torch.zeros((batch_size, time_steps, 1), dtype=torch.float32))
