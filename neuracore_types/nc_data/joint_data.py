"""Joint data types for robot joint states."""

import copy
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from neuracore_types.importer.config import AngleConfig, TorqueUnitsConfig
from neuracore_types.importer.data_config import MappingItem
from neuracore_types.importer.transform import (
    DataTransform,
    DataTransformSequence,
    DegreesToRadians,
    FlipSign,
    NumpyToScalar,
    Offset,
    Scale,
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
    def validate_index_provided(self) -> "JointPositionsDataImportConfig":
        """Validate that either all or no indexes are provided."""
        _validate_index_provided(self.mapping, self.__class__.__name__)
        return self

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        # Add DegreesToRadians transform if needed
        if self.format.angle_units == AngleConfig.DEGREES:
            transform_list.append(DegreesToRadians())

        for item in self.mapping:
            item_transforms = _apply_common_joint_item_transforms(item, transform_list)
            item.transforms = DataTransformSequence(transforms=item_transforms)


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
