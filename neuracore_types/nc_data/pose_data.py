"""Pose data types for 6DOF poses."""

from typing import Literal, Union

import numpy as np
from pydantic import ConfigDict, field_serializer, field_validator

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats


class PoseDataStats(NCDataStats):
    """Statistics for PoseData."""

    type: Literal["PoseDataStats"] = "PoseDataStats"
    pose: DataItemStats


class PoseData(NCData):
    """6DOF pose data for objects, end-effectors, or coordinate frames.

    Represents position and orientation information for tracking objects
    or robot components in 3D space. Poses are stored as dictionaries
    mapping pose names to [x, y, z, rx, ry, rz] values.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["PoseData"] = "PoseData"
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
    def sample(cls) -> "PoseData":
        """Sample an example PoseData instance.

        Returns:
            PoseData: Sampled PoseData instance
        """
        return cls(pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    def calculate_statistics(self) -> PoseDataStats:
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
        return PoseDataStats(pose=stats)
