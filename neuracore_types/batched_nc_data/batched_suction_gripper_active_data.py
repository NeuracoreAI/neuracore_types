"""Data models for batched suction gripper active data."""

from typing import Any, Literal, cast

import torch
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class BatchedSuctionGripperActiveData(BatchedNCData):
    """Batched suction gripper active data."""

    type: Literal["BatchedSuctionGripperActiveData"] = Field(
        default="BatchedSuctionGripperActiveData",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    active: torch.Tensor  # (B, T, 1) bool

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @field_validator("active", mode="before")
    @classmethod
    def decode_active(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode active field to torch.Tensor."""
        return cls._create_tensor_handlers("active")[0](v)

    @field_serializer("active", when_used="json")
    def serialize_active(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize active field to base64 string."""
        return self._create_tensor_handlers("active")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedSuctionGripperActiveData from input nc_data.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        from neuracore_types.nc_data.suction_gripper_active_data import (
            SuctionGripperActiveData,
        )

        gripper_data: SuctionGripperActiveData = cast(SuctionGripperActiveData, nc_data)
        active = (
            torch.tensor([gripper_data.active], dtype=torch.bool)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return cls(active=active)

    @classmethod
    def from_nc_data_list(
        cls, nc_data_list: list[NCData]
    ) -> "BatchedSuctionGripperActiveData":
        """Create BatchedSuctionGripperActiveData from list of data.

        Args:
            nc_data_list: List of SuctionGripperActiveData instances to convert

        Returns:
            BatchedSuctionGripperActiveData with shape (1, T, 1)
        """
        from neuracore_types.nc_data.suction_gripper_active_data import (
            SuctionGripperActiveData,
        )

        active_states = [
            cast(SuctionGripperActiveData, nc).active for nc in nc_data_list
        ]
        # Shape: (1, T, 1)
        active_tensor = (
            torch.tensor(active_states, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
        )
        return cls(active=active_tensor)

    @classmethod
    def sample(
        cls, batch_size: int = 1, time_steps: int = 1
    ) -> "BatchedSuctionGripperActiveData":
        """Sample an example instance of BatchedSuctionGripperActiveData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedSuctionGripperActiveData: Sampled instance
        """
        return cls(active=torch.zeros((batch_size, time_steps, 1), dtype=torch.bool))
