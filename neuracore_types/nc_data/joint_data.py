"""Joint data types for robot joint states."""

import copy
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from neuracore_types.importer.config import (
    AngleConfig,
    JointPositionTypeConfig,
    PoseConfig,
    RotationConfig,
    TorqueUnitsConfig,
    VisualJointTypeConfig,
)
from neuracore_types.importer.data_config import MappingItem
from neuracore_types.importer.transform import (
    Clip,
    DataTransform,
    DataTransformSequence,
    DegreesToRadians,
    FlipSign,
    Normalize,
    NumpyToScalar,
    Offset,
    Pose,
    Scale,
    Unnormalize,
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


def _validate_index_provided(mapping: list[MappingItem], class_name: str) -> None:
    """Validate that either all or no indexes are provided for mapping items."""
    indexes = [item.index for item in mapping]
    if any(idx is not None for idx in indexes) and any(idx is None for idx in indexes):
        raise ValueError(
            f"All or none of the mapping items in {class_name} must have an "
            "'index' specified"
        )
    # If no indexes are provided, assign them sequentially
    if not any(idx is not None for idx in indexes):
        for i, item in enumerate(mapping):
            item.index = i


def _apply_common_joint_item_transforms(
    item: MappingItem, transforms_list: list[DataTransform]
) -> list[DataTransform]:
    """Apply common transforms for joint data items (flip and offset)."""
    item_transforms = copy.deepcopy(transforms_list)
    if item.inverted:
        item_transforms.append(FlipSign())
    if getattr(item, "offset", 0.0) != 0.0:
        item_transforms.append(Offset(value=item.offset))
    item_transforms.append(NumpyToScalar())
    return item_transforms


class JointDataStats(NCDataStats):
    """Statistics for JointData."""

    type: Literal["JointDataStats"] = Field(
        default="JointDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    value: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class JointPositionsDataImportConfig(NCDataImportConfig):
    """Import configuration for JointPositionsData."""

    @model_validator(mode="after")
    def validate_orientation_required(self) -> "JointPositionsDataImportConfig":
        """Validate orientation is when converting joint position type from TCP."""
        if self.format.joint_position_type == JointPositionTypeConfig.END_EFFECTOR:
            if self.format.pose_type == PoseConfig.POSITION_ORIENTATION:
                if self.format.orientation is None:
                    raise ValueError(
                        "orientation must be provided when format is "
                        "'position_orientation'"
                    )
                if self.format.orientation.type == RotationConfig.QUATERNION:
                    if not self.format.orientation.quaternion_order:
                        raise ValueError(
                            "quaternion_order must be provided when type is "
                            "'quaternion'"
                        )
                if self.format.orientation.type == RotationConfig.EULER:
                    if not self.format.orientation.euler_order:
                        raise ValueError(
                            "euler_order must be provided when type is 'euler'"
                        )
        return self

    @model_validator(mode="after")
    def validate_index_provided(self) -> "JointPositionsDataImportConfig":
        """Validate that indexes provided are valid.

        When directly importing joint positions, check either all or no indexes
        are provided. When converting from TCP to joint positions, check that
        the index range length matches the orientation format.
        """
        if self.format.joint_position_type == JointPositionTypeConfig.CUSTOM:
            _validate_index_provided(self.mapping, self.__class__.__name__)
        elif self.format.joint_position_type == JointPositionTypeConfig.END_EFFECTOR:
            if len(self.mapping) != 1:
                raise ValueError(
                    "Only one mapping item is allowed when converting from TCP "
                    "to joint positions"
                )

            if self.format.pose_type == PoseConfig.MATRIX:
                return self
            else:
                if self.mapping[0].index_range is None:
                    raise ValueError("index_range is required for pose data points")
                index_length = (
                    self.mapping[0].index_range.end - self.mapping[0].index_range.start
                )

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
                            f"Unsupported orientation type: "
                            f"{self.format.orientation.type}"
                        )
                    if index_length != expected_length:
                        raise ValueError(
                            f"Index range length must be {expected_length} for "
                            f"orientation type {self.format.orientation.type}, "
                            f"got {index_length}"
                        )
        return self

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        if self.format.joint_position_type == JointPositionTypeConfig.CUSTOM:
            # Add DegreesToRadians transform if needed
            if self.format.angle_units == AngleConfig.DEGREES:
                transform_list.append(DegreesToRadians())

            for item in self.mapping:
                item_transforms = _apply_common_joint_item_transforms(
                    item, transform_list
                )
                item.transforms = DataTransformSequence(transforms=item_transforms)
        elif self.format.joint_position_type == JointPositionTypeConfig.END_EFFECTOR:
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
        else:
            raise ValueError(
                f"Invalid joint position type: {self.format.joint_position_type}"
            )


class JointVelocitiesDataImportConfig(NCDataImportConfig):
    """Import configuration for JointVelocitiesData."""

    @model_validator(mode="after")
    def validate_index_provided(self) -> "JointVelocitiesDataImportConfig":
        """Validate that either all or no indexes are provided."""
        _validate_index_provided(self.mapping, self.__class__.__name__)
        return self

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        if self.format.angle_units == AngleConfig.DEGREES:
            transform_list.append(DegreesToRadians())

        for item in self.mapping:
            item_transforms = _apply_common_joint_item_transforms(item, transform_list)
            item.transforms = DataTransformSequence(transforms=item_transforms)


class JointTorquesDataImportConfig(NCDataImportConfig):
    """Import configuration for JointTorquesData."""

    @model_validator(mode="after")
    def validate_index_provided(self) -> "JointTorquesDataImportConfig":
        """Validate that either all or no indexes are provided."""
        _validate_index_provided(self.mapping, self.__class__.__name__)
        return self

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        if self.format.torque_units == TorqueUnitsConfig.NCM:
            transform_list.append(Scale(factor=0.01))

        for item in self.mapping:
            item_transforms = _apply_common_joint_item_transforms(item, transform_list)
            item.transforms = DataTransformSequence(transforms=item_transforms)


class VisualJointPositionsDataImportConfig(NCDataImportConfig):
    """Import configuration for VisualJointPositionsData."""

    @model_validator(mode="after")
    def validate_index_provided(self) -> "VisualJointPositionsDataImportConfig":
        """Validate that either all or no indexes are provided."""
        if self.format.visual_joint_type == VisualJointTypeConfig.GRIPPER:
            indexes = [item.index for item in self.mapping]
            if any(idx is None for idx in indexes):
                raise ValueError(
                    "All indexes must be provided for gripper visual joint positions"
                )
        elif self.format.visual_joint_type == VisualJointTypeConfig.CUSTOM:
            _validate_index_provided(self.mapping, self.__class__.__name__)
        else:
            raise ValueError(
                f"Invalid visual joint type: {self.format.visual_joint_type}"
            )

        return self

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        if self.format.visual_joint_type == VisualJointTypeConfig.GRIPPER:
            # Use gripper open amount to calculate the visual joint positions
            # from joint limits

            # Add Normalize transform if needed
            if self.format.normalize:
                transform_list.append(
                    Normalize(
                        min=self.format.normalize.min, max=self.format.normalize.max
                    )
                )

            # Clip the value to 0-1
            transform_list.append(Clip(min=0.0, max=1.0))

            # Convert from close amount to open amount if needed
            if self.format.invert_gripper_amount:
                transform_list.append(FlipSign())
                transform_list.append(Offset(value=1.0))

            # Joint limits to be filled in during data import from URDF
            transform_list.append(Unnormalize(min=0.0, max=1.0))

        elif self.format.visual_joint_type == VisualJointTypeConfig.CUSTOM:
            # Add DegreesToRadians transform if needed
            if self.format.angle_units == AngleConfig.DEGREES:
                transform_list.append(DegreesToRadians())
        else:
            raise ValueError(
                f"Invalid visual joint type: {self.format.visual_joint_type}"
            )

        for item in self.mapping:
            item_transforms = _apply_common_joint_item_transforms(item, transform_list)
            item.transforms = DataTransformSequence(transforms=item_transforms)


class JointData(NCData):
    """Robot joint state data including positions, velocities, or torques."""

    type: Literal["JointData"] = Field(
        default="JointData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    value: float

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    def calculate_statistics(self) -> JointDataStats:
        """Calculate the statistics for this data type.

        Returns:
            Dictionary attribute names to their corresponding DataItemStats.
        """
        stats = DataItemStats(
            mean=np.array([self.value], dtype=np.float32),
            std=np.array([0.0], dtype=np.float32),
            count=np.array([1], dtype=np.int32),
            min=np.array([self.value], dtype=np.float32),
            max=np.array([self.value], dtype=np.float32),
        )
        return JointDataStats(value=stats)

    @classmethod
    def sample(cls) -> "JointData":
        """Sample an example JointData instance.

        Returns:
            JointData: Sampled JointData instance
        """
        return cls(value=0.0)
