"""Data models for parallel gripper open amount data."""

from typing import Any, Literal, cast

import torch
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class BatchedParallelGripperOpenAmountData(BatchedNCData):
    """Batched parallel gripper open amount data."""

    type: Literal["BatchedParallelGripperOpenAmountData"] = Field(
        default="BatchedParallelGripperOpenAmountData",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    open_amount: torch.Tensor  # (B, T, 1) float32

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @field_validator("open_amount", mode="before")
    @classmethod
    def decode_open_amount(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode open_amount field to torch.Tensor."""
        return cls._create_tensor_handlers("open_amount")[0](v)

    @field_serializer("open_amount", when_used="json")
    def serialize_open_amount(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize open_amount field to base64 string."""
        return self._create_tensor_handlers("open_amount")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedParallelGripperOpenAmountData from input nc_data.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        from neuracore_types.nc_data.parallel_gripper_open_amount_data import (
            ParallelGripperOpenAmountData,
        )

        gripper_data: ParallelGripperOpenAmountData = cast(
            ParallelGripperOpenAmountData, nc_data
        )
        open_amount = (
            torch.tensor([gripper_data.open_amount], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return cls(open_amount=open_amount)

    @classmethod
    def from_nc_data_list(
        cls, nc_data_list: list[NCData]
    ) -> "BatchedParallelGripperOpenAmountData":
        """Create BatchedParallelGripperOpenAmountData from list of data.

        Args:
            nc_data_list: List of ParallelGripperOpenAmountData instances to convert

        Returns:
            BatchedParallelGripperOpenAmountData with shape (1, T, 1)
        """
        from neuracore_types.nc_data.parallel_gripper_open_amount_data import (
            ParallelGripperOpenAmountData,
        )

        open_amounts = [
            cast(ParallelGripperOpenAmountData, nc).open_amount for nc in nc_data_list
        ]
        # Shape: (1, T, 1)
        open_amount_tensor = (
            torch.tensor(open_amounts, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        )
        return cls(open_amount=open_amount_tensor)

    @classmethod
    def sample(
        cls, batch_size: int = 1, time_steps: int = 1
    ) -> "BatchedParallelGripperOpenAmountData":
        """Sample an example instance of BatchedParallelGripperOpenAmountData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedParallelGripperOpenAmountData: Sampled instance
        """
        return cls(
            open_amount=torch.zeros((batch_size, time_steps, 1), dtype=torch.float32)
        )
