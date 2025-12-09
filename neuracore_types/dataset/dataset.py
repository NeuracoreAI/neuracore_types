"""Models for datasets and synchronized datasets."""

from typing import Any

from pydantic import BaseModel, Field

from neuracore_types.nc_data import DataType, NCDataStatsUnion


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


class SynchronizationProgress(BaseModel):
    """Progress of synchronization for a synchronized dataset.

    Attributes:
        synchronized_dataset_id: Unique identifier for the synced dataset.
        num_synchronized_demonstrations: Number of demonstrations synchronized so far.
    """

    synchronized_dataset_id: str
    num_synchronized_demonstrations: int


class SynchronizedDatasetStatistics(BaseModel):
    """Statistics for a synchronized dataset.

    Attributes:
        synchronized_dataset_id: Unique identifier for the synced dataset.
        robot_data_spec: Mapping of robot IDs to data type names.
        dataset_statistics: Statistics for each robot and data type.
    """

    synchronized_dataset_id: str
    robot_data_spec: dict[str, dict[DataType, list[str]]]
    dataset_statistics: dict[DataType, list[NCDataStatsUnion]] = Field(
        default_factory=dict
    )


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
    """

    id: str
    name: str
    created_at: float
    modified_at: float
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    num_demonstrations: int = 0
    total_duration_seconds: float = 0.0
    size_bytes: int = 0
    is_shared: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    synced_dataset_ids: dict[str, Any] = Field(default_factory=dict)
    all_data_types: dict[DataType, int] = Field(default_factory=dict)
    common_data_types: dict[DataType, int] = Field(default_factory=dict)
    deleted: bool = False
