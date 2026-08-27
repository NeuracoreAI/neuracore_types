"""Models for datasets and synchronized datasets."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neuracore_types.episode.episode import CrossEmbodimentDescription
from neuracore_types.nc_data import DataType, NCDataStatsUnion
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class SynchronizedDataset(BaseModel):
    """Represents a synchronized dataset of episodes.

    A Synchronized dataset groups related robot demonstrations together
    and maintains metadata about the collection as a whole.

    Attributes:
        id: Unique identifier for the synced dataset.
        parent_id: Unique identifier of the corresponding dataset.
        name: Human-readable name for the dataset.
        created_at: Unix timestamp of dataset creation.
        modified_at: Unix timestamp of last modification.
        description: Optional description of the dataset.
        num_demonstrations: Total number of demonstrations.
        total_duration_seconds: Total duration of all demonstrations.
        is_shared: Whether the dataset is shared with other users.
        metadata: Additional arbitrary metadata.
        all_data_types: Dictionary of all data types and their counts.
        common_data_types: Dictionary of common data types and their counts.
        frequency: Frequency at which dataset was processed.
        max_delay_s: Maximum allowed delay for synchronization.
        allow_duplicates: Whether duplicate data points are allowed.
        trim_start_end: Whether to trim the start and end of the episode
            when synchronizing.
        trim_no_movement_at_start_threshold: Threshold below which leading
            frames without joint movement were dropped, or None if the
            dataset was synchronized without that trimming.
    """

    id: str
    parent_id: str
    name: str
    created_at: float
    modified_at: float
    description: str | None
    num_demonstrations: int
    total_duration_seconds: float
    is_shared: bool
    metadata: dict[str, Any]
    all_data_types: dict[DataType, int]
    common_data_types: dict[DataType, int]
    frequency: float
    max_delay_s: float
    allow_duplicates: bool
    trim_start_end: bool = True
    trim_no_movement_at_start_threshold: float | None = None


class SynchronizationProgress(BaseModel):
    """Progress of synchronization for a synchronized dataset.

    Attributes:
        synchronized_dataset_id: Unique identifier for the synced dataset.
        num_synchronized_demonstrations: Number of demonstrations synchronized so far.
        has_failures: Whether any recording synchronization has failed.
        num_failed_recordings: Number of failed recordings.
        failed_recording_ids: IDs of recordings that failed synchronization.
    """

    synchronized_dataset_id: str
    num_synchronized_demonstrations: int
    has_failures: bool = False
    num_failed_recordings: int = 0
    failed_recording_ids: list[str] = Field(default_factory=list)


class CalculateDatasetStatisticsRequest(BaseModel):
    """Request to start (or join) a dataset statistics calculation.

    Attributes:
        synchronized_dataset_id: Synchronized dataset to compute statistics over.
        input_cross_embodiment_description: Mapping of robot IDs to the canonical
            index of each input data item.
        output_cross_embodiment_description: Mapping of robot IDs to the canonical
            index of each output data item.
    """

    synchronized_dataset_id: str
    input_cross_embodiment_description: CrossEmbodimentDescription
    output_cross_embodiment_description: CrossEmbodimentDescription


class DatasetStatisticsJobStatus(str, Enum):
    """Lifecycle stage of an asynchronous dataset statistics calculation."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AGGREGATING = "AGGREGATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class DatasetStatisticsJob(BaseModel):
    """State of an asynchronous dataset statistics calculation.

    Returned both when starting a job and when polling it, so a caller reads the
    total, the progress and the completion signal from one consistent snapshot.

    Attributes:
        job_id: Identifier derived from the dataset, its recording set and both
            cross-embodiment descriptions. Stable across repeat requests, and
            different whenever any of those change.
        synchronized_dataset_id: Synchronized dataset the job covers.
        status: Current lifecycle stage.
        num_recordings: Recordings the job must process.
        num_completed_recordings: Recordings whose statistics are done. Reaching
            num_recordings does not mean the result is fetchable: the aggregate
            stage runs afterwards, so completion is read from ``status``.
        error: Failure message when status is FAILED.
    """

    job_id: str
    synchronized_dataset_id: str
    status: DatasetStatisticsJobStatus
    num_recordings: int
    num_completed_recordings: int
    error: str | None = None


class SynchronizedDatasetStatistics(BaseModel):
    """Statistics for a synchronized dataset.

    Attributes:
        synchronized_dataset_id: Unique identifier for the synced dataset.
        input_cross_embodiment_description: Mapping of robot IDs to the canonical
            index of each input data item.
        output_cross_embodiment_description: Mapping of robot IDs to the canonical
            index of each output data item.
        dataset_statistics: Statistics for each robot and data type, keyed by
            "input" and "output" and dense by canonical index.
    """

    synchronized_dataset_id: str
    input_cross_embodiment_description: CrossEmbodimentDescription
    output_cross_embodiment_description: CrossEmbodimentDescription
    dataset_statistics: dict[str, dict[DataType, list[NCDataStatsUnion]]]


class Dataset(BaseModel):
    """Represents a dataset of unsynchronized episodes.

    Attributes:
        id: Unique identifier for the dataset.
        name: Human-readable name for the dataset.
        created_at: Unix timestamp of dataset creation.
        modified_at: Unix timestamp of last modification.
        description: Optional description of the dataset.
        tags: List of tags for categorizing the dataset.
        num_demonstrations: Total number of demonstrations.
        total_duration_seconds: Total duration of all demonstrations.
        size_bytes: Total size of all demonstrations.
        is_shared: Whether the dataset is shared with other users.
        metadata: Additional arbitrary metadata.
        synced_dataset_ids: List of synced dataset IDs in this dataset.
                            They point to synced datasets that synchronized
                            this dataset at a particular frequency.
        all_data_types: Dictionary of all data types and their counts.
                        A union of all datatypes in the recordings which
                        make up this dataset.
        common_data_types: Dictionary of common data types and their counts.
                           All datatypes common to every recording which
                           make up this dataset.
        is_creating: Whether the dataset is still being created.
        deleted: Whether the dataset has been deleted.
    """

    id: str
    name: str
    created_at: float
    modified_at: float
    description: str | None = None
    tags: list[str] = Field(
        default_factory=list, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    num_demonstrations: int = Field(
        default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    total_duration_seconds: float = Field(
        default=0.0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    size_bytes: int = Field(default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    is_shared: bool = Field(default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    metadata: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    synced_dataset_ids: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    all_data_types: dict[DataType, int] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    common_data_types: dict[DataType, int] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    is_creating: bool = Field(
        default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    deleted: bool = Field(default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class DatasetUpdateRequest(BaseModel):
    """Request model for dataset updates.

    Attributes:
        name: Optional new name for the dataset.
        description: Optional new description.
        tags: Optional new list of tags.
    """

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None


class DatasetCloneRequest(BaseModel):
    """Request model for cloning a dataset.

    Attributes:
        name: Name for the new cloned dataset.
        sourceDatasetId: ID of the dataset to clone.
        description: Optional description for the new dataset.
        tags: Optional tags for the new dataset.
    """

    name: str
    sourceDatasetId: str
    description: str | None = None
    tags: list[str] | None = None


class DatasetSplitRequest(BaseModel):
    """Request model for splitting recordings from a dataset into a new one.

    Attributes:
        name: Name for the new dataset.
        sourceDatasetId: ID of the dataset to split from.
        recordingIds: IDs of the recordings to copy into the new dataset.
        description: Optional description for the new dataset.
        tags: Optional tags for the new dataset.
    """

    name: str
    sourceDatasetId: str
    recordingIds: list[str]
    description: str | None = None
    tags: list[str] | None = None
