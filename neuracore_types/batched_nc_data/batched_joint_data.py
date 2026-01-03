"""Joint data types for robot joint states."""

from typing import Any, Literal, cast

import torch
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class BatchedJointData(BatchedNCData):
    """Batched joint data for multiple time steps and batch sizes."""

    type: Literal["BatchedJointData"] = Field(
        default="BatchedJointData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    value: torch.Tensor  # (B, T, 1) float32

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @field_validator("value", mode="before")
    @classmethod
    def decode_value(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode value field to torch.Tensor."""
        return cls._create_tensor_handlers("value")[0](v)

    @field_serializer("value", when_used="json")
    def serialize_value(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize value field to base64 string."""
        return self._create_tensor_handlers("value")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedJointData from JointData.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        from neuracore_types.nc_data.joint_data import JointData

        joint_data: JointData = cast(JointData, nc_data)
        value = (
            torch.tensor([joint_data.value], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return cls(value=value)

    @classmethod
    def from_nc_data_list(cls, nc_data_list: list[NCData]) -> "BatchedJointData":
        """Create BatchedJointData from list of JointData.

        Args:
            nc_data_list: List of JointData instances to convert

        Returns:
            BatchedJointData with shape (1, T, 1) where T = len(nc_data_list)
        """
        from neuracore_types.nc_data.joint_data import JointData

        values = [cast(JointData, nc).value for nc in nc_data_list]
        # Shape: (1, T, 1)
        value_tensor = (
            torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        )
        return cls(value=value_tensor)

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedJointData":
        """Sample an example instance of BatchedJointData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedJointData: Sampled BatchedJointData instance
        """
        return cls(value=torch.zeros((batch_size, time_steps, 1), dtype=torch.float32))
