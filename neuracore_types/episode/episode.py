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

EmbodimentDescription = dict[DataType, dict[int, str]]
CrossEmbodimentDescription = dict[str, EmbodimentDescription]
CrossEmbodimentUnion = dict[str, dict[DataType, list[str]]]

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

    def order(
        self, embodiment_description: EmbodimentDescription
    ) -> "SynchronizedPoint":
        """Return a new sync point ordered by indexed embodiment specification.

        The `embodiment_description` defines the exact set of `DataType` entries
        and per-type sensor names that must be present in this sync point.
        Ordering is determined by sorting each data type's integer indices in the
        embodiment description and then mapping those indices to sensor names.

        This method is strict by design to mirror the training data pipeline:
        the sync point and the embodiment description must describe the same
        information. Extra or missing data types raise an error. Extra or missing
        sensor names within a data type also raise an error.

        Uses `model_construct()` to skip validation for better performance,
        since this only reorders already validated data.

        Args:
            embodiment_description: Mapping of `DataType -> {index: sensor_name}`
                describing the exact names to include and their order.

        Returns:
            A new `SynchronizedPoint` with each data type's dictionary rebuilt in
            sorted index order from the embodiment description.

        Raises:
            ValueError: If the sync point and embodiment description do not have
                exactly matching data types or sensor names.
        """
        sync_point_data_types = set(self.data.keys())
        expected_data_types = set(embodiment_description.keys())
        if sync_point_data_types != expected_data_types:
            extra_data_types = sync_point_data_types - expected_data_types
            missing_data_types = expected_data_types - sync_point_data_types
            raise ValueError(
                "SynchronizedPoint data types must exactly match "
                "embodiment_description.\n"
                f"Extra data types in synchronized point: {extra_data_types}\n"
                f"Missing data types from synchronized point: {missing_data_types}\n"
            )

        for data_type, indexed_names in embodiment_description.items():
            sync_point_names = set(self.data[data_type].keys())
            expected_names = set(indexed_names.values())
            if sync_point_names != expected_names:
                extra_names = sync_point_names - expected_names
                missing_names = expected_names - sync_point_names
                raise ValueError(
                    f"SynchronizedPoint names for DataType {data_type} must exactly "
                    "match embodiment_description.\n"
                    f"Extra names in synchronized point: {extra_names}\n"
                    f"Missing names from synchronized point: {missing_names}\n"
                )

        return SynchronizedPoint.model_construct(
            timestamp=self.timestamp,
            robot_id=self.robot_id,
            data={
                data_type: {
                    indexed_names[index]: self.data[data_type][indexed_names[index]]
                    for index in sorted(indexed_names)
                }
                for data_type, indexed_names in embodiment_description.items()
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

    def order(self, order_spec: EmbodimentDescription) -> "SynchronizedEpisode":
        """Return a new episode with observations ordered by index specification.

        Args:
            order_spec: Mapping of `DataType -> {index: sensor_name}` used to
                reorder every observation.

        Returns:
            New `SynchronizedEpisode` with each observation ordered according to
            the provided indexed embodiment description.
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
    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)
