"""Models for episodes and synchronized data points."""

import time
from datetime import datetime
from enum import Enum
from typing import Optional, Union

from names_generator import generate_name
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from neuracore_types.nc_data import DataType, NCDataUnion
from neuracore_types.nc_data.nc_data import DataItemStats, NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)

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

    timestamp: float = Field(
        default_factory=lambda: time.time(),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    robot_id: Optional[str] = None
    data: dict[DataType, dict[str, NCDataUnion]] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    def order(self, order_spec: dict[DataType, list[str]]) -> "SynchronizedPoint":
        """Return a new SynchronizedPoint with all dictionary data ordered.

        Uses model_construct() to skip validation for better performance,
        since we're just reordering existing validated data.
        """
        if not set(self.data.keys()).issubset(set(order_spec.keys())):
            raise ValueError(
                "SynchronizedPoint contains DataTypes not present in order_spec.\n"
                f"Keys in synchronized point: {set(self.data.keys())}\n"
                f"Keys in order_spec: {set(order_spec.keys())}\n"
            )
        # Check that all specified keys are present
        for data_type, keys in order_spec.items():
            if data_type in self.data:
                missing_keys = set(keys) - set(self.data[data_type].keys())
                if missing_keys:
                    raise ValueError(
                        f"SynchronizedPoint missing keys for DataType {data_type}: "
                        f"{missing_keys}"
                    )
        # Use model_construct to skip validation - data is already validated
        return SynchronizedPoint.model_construct(
            timestamp=self.timestamp,
            robot_id=self.robot_id,
            data={
                data_type: {name: data_dict[name] for name in order_spec[data_type]}
                for data_type, data_dict in self.data.items()
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
    episode_length: int = Field(default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)

    data: dict[DataType, dict[str, DataItemStats]] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

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

    NORMAL = "NORMAL"
    FLAGGED = "FLAGGED"


class RecordingMetadata(BaseModel):
    """Metadata details for a recording.

    Attributes:
        name: Name of the recording.
        notes: Optional notes about the recording.
        status: Current RecordingStatus of the recording
    """

    name: str = Field(
        default_factory=lambda: generate_name(style="capital"),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
        max_length=NAME_MAX_LENGTH,
        strip_whitespace=True,
        min_length=1,
    )
    notes: str = Field(
        default="",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
        max_length=NOTES_MAX_LENGTH,
        strip_whitespace=True,
    )
    status: RecordingStatus = Field(
        default=RecordingStatus.NORMAL, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


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
    robot_id: Optional[str] = None
    instance: NonNegativeInt = Field(
        default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    org_id: str
    created_by: str = Field(default="", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    start_time: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    end_time: float | None = None
    metadata: RecordingMetadata = Field(
        default_factory=RecordingMetadata, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    total_bytes: int = Field(default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    is_shared: bool = Field(default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    data_types: set[DataType] = Field(
        default_factory=set, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class PendingRecordingStatus(str, Enum):
    """Pending recording status.

    STARTED: recording in progress.
    UPLOADING: at least one trace has upload_progress between 0 and 99.
    UPLOADED: all traces are completely uploaded.
    """

    STARTED = "STARTED"
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"


class PendingRecording(Recording):
    """Represents a pending recording.

    Attributes:
        saved_dataset_id: ID of the dataset where the recording is saved
        status: Current status of the pending recording
        progress: Upload progress percentage (0-100)
        expected_trace_count: Number of traces expected (set by register_traces API)
        total_bytes: Total bytes expected across all traces (for progress bar)
        traces_registered: Whether the register_traces API has been called
        save_triggered: Whether save process has been initiated (prevents duplicates)
    """

    saved_dataset_id: Optional[str] = None
    status: PendingRecordingStatus = Field(
        default=PendingRecordingStatus.STARTED,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    progress: int
    expected_trace_count: int = Field(
        default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    total_bytes: int = Field(default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    traces_registered: bool = Field(
        default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    save_triggered: bool = Field(
        default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)
