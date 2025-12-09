"""Joint data types for robot joint states."""

from typing import Literal

import numpy as np

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats


class JointDataStats(NCDataStats):
    """Statistics for JointData."""

    type: Literal["JointDataStats"] = "JointDataStats"
    value: DataItemStats


class JointData(NCData):
    """Robot joint state data including positions, velocities, or torques."""

    type: Literal["JointData"] = "JointData"
    value: float

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
