"""Data models for end-effector pose data."""

from typing import Literal, Union

import numpy as np
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class EndEffectorPoseDataStats(NCDataStats):
    """Statistics for EndEffectorPoseData."""

    type: Literal["EndEffectorPoseDataStats"] = Field(
        default="EndEffectorPoseDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    pose: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class EndEffectorPoseData(NCData):
    """End-effector pose data.

    Contains the pose of end-effectors as a 7-element list containing the
    position and unit quaternion orientation [x, y, z, qx, qy, qz, qw].
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=fix_required_with_defaults
    )

    type: Literal["EndEffectorPoseData"] = Field(
        default="EndEffectorPoseData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    pose: np.ndarray

    @field_validator("pose")
    @classmethod
    def validate_pose_length(cls, v: np.ndarray) -> np.ndarray:
        """Validate that pose has exactly 7 values (position + quaternion)."""
        if len(v) != 7:
            raise ValueError(
                "Pose must have exactly 7 values "
                f"(x, y, z, qx, qy, qz, qw), got {len(v)}"
            )
        return v

    @field_validator("pose", mode="before")
    @classmethod
    def decode_pose(cls, v: Union[list, np.ndarray]) -> np.ndarray:
        """Decode pose to NumPy array."""
        return np.array(v, dtype=np.float32) if isinstance(v, list) else v

    @field_serializer("pose", when_used="json")
    def serialize_pose(self, v: np.ndarray) -> list[float]:
        """Serialize pose to JSON list."""
        return v.tolist()

    @classmethod
    def sample(cls) -> "EndEffectorPoseData":
        """Sample an example EndEffectorPoseData instance.

        Returns:
            EndEffectorPoseData: Sampled EndEffectorPoseData instance
        """
        return cls(pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    def calculate_statistics(self) -> EndEffectorPoseDataStats:
        """Calculate the statistics for this data type.

        Returns:
            Dictionary attribute names to their corresponding DataItemStats.
        """
        stats = DataItemStats(
            mean=np.array(self.pose, dtype=np.float32),
            std=np.zeros_like(np.array(self.pose, dtype=np.float32)),
            count=np.array([1] * len(self.pose), dtype=np.int32),
            min=np.array(self.pose, dtype=np.float32),
            max=np.array(self.pose, dtype=np.float32),
        )
        return EndEffectorPoseDataStats(pose=stats)
