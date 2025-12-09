"""Models for episodes and synchronized data points."""

import time
from datetime import datetime
from enum import Enum
from typing import Optional, Union

from names_generator import generate_name
from pydantic import BaseModel, Field, NonNegativeInt

from neuracore_types.nc_data import DataType, NCDataUnion
from neuracore_types.nc_data.nc_data import DataItemStats, NCData

DataSpec = dict[DataType, list[str]]
RobotDataSpec = dict[str, DataSpec]

NAME_MAX_LENGTH = 60
NOTES_MAX_LENGTH = 1000


class SynchronizedPoint(BaseModel):
    """Synchronized collection of all sensor data at a single time point.

    Represents a complete snapshot of robot state and sensor information
    at a specific timestamp. Used for creating temporally aligned datasets
    and ensuring consistent data relationships across different sensors.
    """

    timestamp: float = Field(default_factory=lambda: time.time())
    robot_id: Optional[str] = None
    data: dict[DataType, dict[str, NCDataUnion]] = Field(default_factory=dict)

    def order(self, order_spec: dict[DataType, list[str]]) -> "SynchronizedPoint":
        """Return a new SynchronizedPoint with all dictionary data ordered."""
        if not set(self.data.keys()).issubset(set(order_spec.keys())):
            raise ValueError(
                "SynchronizedPoint contains DataTypes not present in order_spec.\n"
                f"Keys in synchronized point: {set(self.data.keys())}\n"
                f"Keys in order_spec: {set(order_spec.keys())}\n"
            )
        # Check that all specified keys are present
        for dt, keys in order_spec.items():
            if dt in self.data:
                missing_keys = set(keys) - set(self.data[dt].keys())
                if missing_keys:
                    raise ValueError(
                        f"SynchronizedPoint missing keys for DataType {dt}: "
                        f"{missing_keys}"
                    )
        return SynchronizedPoint(
            timestamp=self.timestamp,
            robot_id=self.robot_id,
            data={
                dt: {name: data_dict[name] for name in order_spec[dt]}
                for dt, data_dict in self.data.items()
            },
        )

    def __getitem__(self, key: Union[DataType, str]) -> dict[str, NCData]:
        """Get item by DataType or field name."""
        # If key is a DataType enum, access the nested data dict
        if isinstance(key, DataType):
            return self.data[key]
        # Otherwise, fall back to default Pydantic behavior for field names
        return super().__getitem__(key)

    def __setitem__(self, key: Union[DataType, str], value: dict[str, NCData]) -> None:
        """Set item by DataType or field name."""
        # Same for setting
        if isinstance(key, DataType):
            self.data[key] = value
        else:
            super().__setitem__(key, value)


class SynchronizedEpisode(BaseModel):
    """Synchronized episode of time-ordered synchronized observations."""

    observations: list[SynchronizedPoint]
    start_time: float
    end_time: float
    robot_id: str

    def order(self, order_spec: dict[DataType, list[str]]) -> "SynchronizedEpisode":
        """Return a new SynchronizedEpisode with all synchronized observations ordered.

        Returns:
            New SynchronizedEpisode with all synchronized observations ordered.
        """
        return SynchronizedEpisode(
            observations=[
                observation.order(order_spec) for observation in self.observations
            ],
            start_time=self.start_time,
            end_time=self.end_time,
            robot_id=self.robot_id,
        )


class EpisodeStatistics(BaseModel):
    """Description of a single episode with statistics and counts.

    Provides metadata about an individual episode including data statistics,
    sensor counts, and episode length for analysis and processing.
    """

    # Episode metadata
    episode_length: int = 0
    data: dict[DataType, dict[str, DataItemStats]] = Field(default_factory=dict)

    def get_data_types(self) -> list[DataType]:
        """Determine which data types are present in the recording.

        Analyzes the recording statistics to identify which data modalities
        contain actual data (non-zero lengths/counts).

        Returns:
            List of DataType enums representing the data modalities
            present in this recording.
        """
        return list(self.data.keys())


class RecordingStatus(str, Enum):
    """Recording status options."""

    NORMAL = "normal"
    FLAGGED = "flagged"


class RecordingMetadata(BaseModel):
    """Metadata details for a recording.

    Attributes:
        name: Name of the recording.
        notes: Optional notes about the recording.
        status: Current RecordingStatus of the recording
    """

    name: str = Field(
        default_factory=lambda: generate_name(style="capital"),
        max_length=NAME_MAX_LENGTH,
        strip_whitespace=True,
        min_length=1,
    )
    notes: str = Field(
        default="",
        max_length=NOTES_MAX_LENGTH,
        strip_whitespace=True,
    )
    status: RecordingStatus = RecordingStatus.NORMAL


class Recording(BaseModel):
    """Represents a robot recording with flat storage.

    Attributes:
        id: Unique identifier for the recording
        robot_id: ID of the robot being recorded
        instance: The physical robot being recorded
        org_id: ID of the organization owning the recording
        created_by: ID of the user who created the recording
        created_at: Unix timestamp when recording started
        end_time: Unix timestamp when recording ended (if not active)
        metadata: Additional metadata about the recording
        total_bytes: Total size of all recorded data in bytes
        is_shared: Whether the recording is shared across organizations
        data_types: Set of data types recorded (e.g., joint positions, images)
    """

    id: str
    robot_id: str
    instance: NonNegativeInt = 0
    org_id: str
    created_by: str = ""
    start_time: float = Field(default_factory=lambda: datetime.now().timestamp())
    end_time: float | None = None
    metadata: RecordingMetadata = Field(default_factory=RecordingMetadata)
    total_bytes: int = 0
    is_shared: bool = False
    data_types: set[DataType] = Field(default_factory=set)
