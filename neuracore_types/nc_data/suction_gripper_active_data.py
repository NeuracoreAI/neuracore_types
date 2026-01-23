"""Data models for suction gripper active data."""

from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class SuctionGripperActiveDataStats(NCDataStats):
    """Statistics for SuctionGripperActiveData."""

    type: Literal["SuctionGripperActiveDataStats"] = Field(
        default="SuctionGripperActiveDataStats",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    active: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class SuctionGripperActiveData(NCData):
    """Active state data for suction gripper end effector.

    Contains the on/off state of a suction gripper.
    """

    type: Literal["SuctionGripperActiveData"] = Field(
        default="SuctionGripperActiveData",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    active: bool

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @classmethod
    def sample(cls) -> "SuctionGripperActiveData":
        """Sample an example SuctionGripperActiveData instance.

        Returns:
            SuctionGripperActiveData: Sampled instance
        """
        return cls(active=False)

    def calculate_statistics(self) -> SuctionGripperActiveDataStats:
        """Calculate the statistics for this data type.

        Returns:
            Dictionary attribute names to their corresponding DataItemStats.
        """
        # Convert boolean to float for statistics (False=0.0, True=1.0)
        value = 1.0 if self.active else 0.0
        stats = DataItemStats(
            mean=np.array([value], dtype=np.float32),
            std=np.array([0.0], dtype=np.float32),
            count=np.array([1], dtype=np.int32),
            min=np.array([value], dtype=np.float32),
            max=np.array([value], dtype=np.float32),
        )
        return SuctionGripperActiveDataStats(active=stats)
