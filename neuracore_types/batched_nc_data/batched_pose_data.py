"""Pose data types for 6DOF poses."""

from typing import Any, Literal, cast

import torch
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class BatchedPoseData(BatchedNCData):
    """Batched pose data for multiple time steps and batch sizes."""

    type: Literal["BatchedPoseData"] = Field(
        default="BatchedPoseData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    pose: torch.Tensor  # (B, T, 7) float32

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @field_validator("pose", mode="before")
    @classmethod
    def decode_pose(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode pose field to torch.Tensor."""
        return cls._create_tensor_handlers("pose")[0](v)

    @field_serializer("pose", when_used="json")
    def serialize_pose(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize pose field to base64 string."""
        return self._create_tensor_handlers("pose")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedPoseData from PoseData.

        Args:
            nc_data: Input NCData instance

        Returns:
            BatchedNCData: BatchedPoseData instance
        """
        from neuracore_types.nc_data.pose_data import PoseData

        pose_data: PoseData = cast(PoseData, nc_data)
        pose = (
            torch.tensor(pose_data.pose, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        return cls(pose=pose)

    @classmethod
    def from_nc_data_list(cls, nc_data_list: list[NCData]) -> "BatchedPoseData":
        """Create BatchedPoseData from list of PoseData.

        Args:
            nc_data_list: List of PoseData instances to convert

        Returns:
            BatchedPoseData with shape (1, T, 7) where T = len(nc_data_list)
        """
        from neuracore_types.nc_data.pose_data import PoseData

        poses = [cast(PoseData, nc).pose for nc in nc_data_list]
        # Shape: (1, T, 7)
        pose_tensor = torch.tensor(poses, dtype=torch.float32).unsqueeze(0)
        return cls(pose=pose_tensor)

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedPoseData":
        """Sample an example instance of BatchedPoseData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedPoseData: Sampled BatchedPoseData instance
        """
        return cls(pose=torch.zeros((batch_size, time_steps, 7), dtype=torch.float32))
