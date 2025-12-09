"""Data models for parallel gripper open amount data."""

from typing import Literal

import numpy as np

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats


class ParallelGripperOpenAmountDataStats(NCDataStats):
    """Statistics for ParallelGripperOpenAmountData."""

    type: Literal["ParallelGripperOpenAmountDataStats"] = (
        "ParallelGripperOpenAmountDataStats"
    )
    open_amount: DataItemStats


class ParallelGripperOpenAmountData(NCData):
    """Open amount data for parallel end effector gripper.

    Contains the state of parallel gripper opening amounts.
    """

    type: Literal["ParallelGripperOpenAmountData"] = "ParallelGripperOpenAmountData"
    open_amount: float

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
