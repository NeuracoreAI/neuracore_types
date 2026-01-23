"""Data models for parallel gripper open amount data."""

import copy
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field

from neuracore_types.importer.transform import (
    Clip,
    DataTransform,
    DataTransformSequence,
    FlipSign,
    Normalize,
    NumpyToScalar,
    Offset,
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


class ParallelGripperOpenAmountDataStats(NCDataStats):
    """Statistics for ParallelGripperOpenAmountData."""

    type: Literal["ParallelGripperOpenAmountDataStats"] = Field(
        default="ParallelGripperOpenAmountDataStats",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    open_amount: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class ParallelGripperOpenAmountDataImportConfig(NCDataImportConfig):
    """Import configuration for ParallelGripperOpenAmountData."""

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        # Add Normalize transform if needed
        if self.format.normalize:
            transform_list.append(
                Normalize(min=self.format.normalize.min, max=self.format.normalize.max)
            )

        # Clip the value to 0-1
        transform_list.append(Clip(min=0.0, max=1.0))

        for item in self.mapping:
            item_transforms = copy.deepcopy(transform_list)
            if item.inverted:
                item_transforms.append(FlipSign())
            if item.offset != 0.0:
                item_transforms.append(Offset(value=item.offset))
            item_transforms.append(NumpyToScalar())
            item.transforms = DataTransformSequence(transforms=item_transforms)


class ParallelGripperOpenAmountData(NCData):
    """Open amount data for parallel end effector gripper.

    Contains the state of parallel gripper opening amounts.
    """

    type: Literal["ParallelGripperOpenAmountData"] = Field(
        default="ParallelGripperOpenAmountData",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    open_amount: float

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @classmethod
    def sample(cls) -> "ParallelGripperOpenAmountData":
        """Sample an example ParallelGripperOpenAmountData instance.

        Returns:
            ParallelGripperOpenAmountData: Sampled instance
        """
        return cls(open_amount=0.0)

    def calculate_statistics(self) -> ParallelGripperOpenAmountDataStats:
        """Calculate the statistics for this data type.

        Returns:
            Dictionary attribute names to their corresponding DataItemStats.
        """
        stats = DataItemStats(
            mean=np.array([self.open_amount], dtype=np.float32),
            std=np.array([0.0], dtype=np.float32),
            count=np.array([1], dtype=np.int32),
            min=np.array([self.open_amount], dtype=np.float32),
            max=np.array([self.open_amount], dtype=np.float32),
        )
        return ParallelGripperOpenAmountDataStats(open_amount=stats)
