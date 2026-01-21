"""Pose data types for 6DOF poses."""

import copy
from typing import Literal, Union

import numpy as np
from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from neuracore_types.importer.config import AngleConfig, PoseConfig, RotationConfig
from neuracore_types.importer.transform import (
    DataTransform,
    DataTransformSequence,
    Pose,
)
from neuracore_types.nc_data.nc_data import (
    DataItemStats,
    NCData,
    NCDataImportConfig,
    NCDataStats,
)
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class PoseDataStats(NCDataStats):
    """Statistics for PoseData."""

    type: Literal["PoseDataStats"] = Field(
        default="PoseDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    pose: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class PoseDataImportConfig(NCDataImportConfig):
    """Import configuration for PoseData."""

    @model_validator(mode="after")
    def validate_orientation_required(self) -> "PoseDataImportConfig":
        """Validate orientation is provided when format is position_orientation."""
        if self.format.pose_type == PoseConfig.POSITION_ORIENTATION:
            if self.format.orientation is None:
                raise ValueError(
                    "orientation must be provided when format is 'position_orientation'"
                )
            if self.format.orientation.type == RotationConfig.QUATERNION:
                if not self.format.orientation.quaternion_order:
                    raise ValueError(
                        "quaternion_order must be provided when type is 'quaternion'"
                    )
            if self.format.orientation.type == RotationConfig.EULER:
                if not self.format.orientation.euler_order:
                    raise ValueError(
                        "euler_order must be provided when type is 'euler'"
                    )
        return self

    @model_validator(mode="after")
    def validate_index_range(self) -> "PoseDataImportConfig":
        """Validate that index range length matches the format."""
        if self.format.pose_type == PoseConfig.MATRIX:
            return self

        for item in self.mapping:
            if item.index_range is None:
                raise ValueError("index_range is required for pose data points")
            index_length = item.index_range.end - item.index_range.start

            if self.format.pose_type == PoseConfig.MATRIX:
                if index_length != 16:
                    raise ValueError(
                        "Index range length must be 16 for matrix format, "
                        f"got {index_length}"
                    )
            elif self.format.pose_type == PoseConfig.POSITION_ORIENTATION:
                if self.format.orientation is None:
                    raise ValueError(
                        "orientation is required when pose_type is "
                        "'position_orientation'"
                    )
                if self.format.orientation.type == RotationConfig.QUATERNION:
                    expected_length = 7  # 3 position + 4 quaternion
                elif self.format.orientation.type in [
                    RotationConfig.EULER,
                    RotationConfig.AXIS_ANGLE,
                ]:  # euler or axis_angle
                    expected_length = 6  # 3 position + 3 euler or axis_angle
                elif self.format.orientation.type == RotationConfig.MATRIX:
                    expected_length = 9  # 3 position + 3x3 matrix
                else:
                    raise ValueError(
                        f"Unsupported orientation type: {self.format.orientation.type}"
                    )
                if index_length != expected_length:
                    raise ValueError(
                        f"Index range length must be {expected_length} for "
                        f"orientation type {self.format.orientation.type}, "
                        f"got {index_length}"
                    )

        return self

    def _populate_transforms(self) -> None:
        """Populate transforms based on flags if not already set."""
        transform_list: list[DataTransform] = []
        # Add Pose transform based on format and orientation settings
        for item in self.mapping:
            item_transforms = copy.deepcopy(transform_list)
            if self.format.pose_type == PoseConfig.MATRIX:
                item_transforms.append(Pose(pose_type=PoseConfig.MATRIX))
            elif self.format.pose_type == PoseConfig.POSITION_ORIENTATION:
                if self.format.orientation is None:
                    raise ValueError(
                        "orientation is required when pose_type is "
                        "'position_orientation'"
                    )
                # Determine sequence
                seq: str = "xyzw"
                if self.format.orientation.type == RotationConfig.QUATERNION:
                    seq = self.format.orientation.quaternion_order.value
                elif self.format.orientation.type == RotationConfig.EULER:
                    seq = self.format.orientation.euler_order.value

                item_transforms.append(
                    Pose(
                        pose_type=PoseConfig.POSITION_ORIENTATION,
                        rotation_type=RotationConfig(self.format.orientation.type),
                        angle_type=AngleConfig(self.format.orientation.angle_units),
                        seq=seq,
                    )
                )
            item.transforms = DataTransformSequence(transforms=item_transforms)


class PoseData(NCData):
    """6DOF pose data for objects, end-effectors, or coordinate frames.

    Represents position and orientation information for tracking objects
    or robot components in 3D space. Poses are stored as dictionaries
    mapping pose names to [x, y, z, rx, ry, rz] values.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=fix_required_with_defaults
    )

    type: Literal["PoseData"] = Field(
        default="PoseData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
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
