"""Joint data types for robot joint states."""

from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class JointDataStats(NCDataStats):
    """Statistics for JointData."""

    type: Literal["JointDataStats"] = Field(
        default="JointDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    value: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


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
