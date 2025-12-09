"""Custom 1D numerical data for specialized applications."""

from typing import Any, Literal, cast

import torch
from pydantic import field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData


class BatchedCustom1DData(BatchedNCData):
    """Batched custom 1D numerical data."""

    type: Literal["BatchedCustom1DData"] = "BatchedCustom1DData"
    data: torch.Tensor  # (B, T, N) float32

    @field_validator("data", mode="before")
    @classmethod
    def decode_data(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode data field to torch.Tensor."""
        return cls._create_tensor_handlers("data")[0](v)

    @field_serializer("data", when_used="json")
    def serialize_data(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize data field to base64 string."""
        return self._create_tensor_handlers("data")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedCustom1DData from Custom1DData.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        from neuracore_types.nc_data.custom_1d_data import Custom1DData

        custom_1d_data: Custom1DData = cast(Custom1DData, nc_data)
        data = (
            torch.tensor(custom_1d_data.data, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return cls(data=data)

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedCustom1DData":
        """Sample an example instance of BatchedCustom1DData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedCustom1DData: Sampled BatchedCustom1DData instance
        """
        return cls(data=torch.zeros((batch_size, time_steps, 10), dtype=torch.float32))
