"""Data models for episode done flag data."""

import copy
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field

from neuracore_types.importer.transform import (
    DataTransform,
    DataTransformSequence,
    NumpyToScalar,
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


class DoneDataStats(NCDataStats):
    """Statistics for DoneData."""

    type: Literal["DoneDataStats"] = Field(
        default="DoneDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    done: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class DoneDataImportConfig(NCDataImportConfig):
    """Import configuration for DoneData."""

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = [NumpyToScalar()]
        for item in self.mapping:
            item.transforms = DataTransformSequence(
                transforms=copy.deepcopy(transform_list)
            )


class DoneData(NCData):
    """Episode done flag.

    False (0) while the episode is running and True (1) once the task is
    complete.
    """

    type: Literal["DoneData"] = Field(
        default="DoneData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    done: bool

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @classmethod
    def sample(cls) -> "DoneData":
        """Sample an example DoneData instance.

        Returns:
            DoneData: Sampled instance
        """
        return cls(done=False)

    def calculate_statistics(self) -> DoneDataStats:
        """Calculate the statistics for this data type.

        Returns:
            Statistics for the done flag as a 0/1 value.
        """
        value = float(self.done)
        stats = DataItemStats(
            mean=np.array([value], dtype=np.float32),
            std=np.array([0.0], dtype=np.float32),
            count=np.array([1], dtype=np.int32),
            min=np.array([value], dtype=np.float32),
            max=np.array([value], dtype=np.float32),
            q01=np.array([value], dtype=np.float32),
            q99=np.array([value], dtype=np.float32),
        )
        return DoneDataStats(done=stats)
