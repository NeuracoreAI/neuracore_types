"""Request and response models for training jobs."""

from datetime import datetime
from enum import Enum
from typing import Any

from ordered_set import OrderedSet
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from neuracore_types.episode.episode import CrossEmbodimentDescription
from neuracore_types.hardware import GPUType
from neuracore_types.nc_data import DataType, NCDataStatsUnion
from neuracore_types.synchronization.synchronization import SynchronizationDetails
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class MetricsData(BaseModel):
    """Response model for metrics data.

    Attributes:
        data: A dictionary that maps the values at each step
        metaData: A dictionary that maps out meta-data related
        to the metric
    """

    data: dict[int, float]
    metadata: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class Metrics(BaseModel):
    """Response model for metrics data.

    Attributes:
        metrics: A dictionary mapping metric names to their values/metaData
    """

    metrics: dict[str, MetricsData] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class ModelInitDescription(BaseModel):
    """Configuration specification for initializing Neuracore models.

    Defines the model architecture requirements including dataset characteristics,
    input/output data types, and prediction horizons for model initialization
    and training configuration.

    Example:
        ModelInitDescription(
            input_data_types=[DataType.RGB_IMAGES, DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            input_dataset_statistics={
                DataType.RGB_IMAGES: DataItemStats(...),
                DataType.JOINT_POSITIONS: DataItemStats(...),
            },
            output_dataset_statistics={
                DataType.JOINT_TARGET_POSITIONS: DataItemStats(...),
            },
        )
    """

    input_data_types: OrderedSet[DataType]
    output_data_types: OrderedSet[DataType]

    input_dataset_statistics: dict[DataType, list[NCDataStatsUnion]]
    output_dataset_statistics: dict[DataType, list[NCDataStatsUnion]]
    output_prediction_horizon: int = Field(
        default=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(
        json_schema_extra=fix_required_with_defaults,
        arbitrary_types_allowed=True,
    )

    @field_validator("input_data_types", "output_data_types", mode="before")
    @classmethod
    def convert_to_ordered_set(cls, v: Any) -> OrderedSet[DataType]:
        """Convert list, set, or other iterable to OrderedSet."""
        if isinstance(v, OrderedSet):
            return v
        return OrderedSet(v)

    @field_serializer("input_data_types", "output_data_types")
    def serialize_ordered_set(self, value: OrderedSet[DataType]) -> list[DataType]:
        """Serialize OrderedSet to list for JSON compatibility."""
        return list(value)


class TrainingJobStatus(str, Enum):
    """Training job status."""

    QUEUED = "QUEUED"
    PROVISIONING = "PROVISIONING"
    STARTING = "STARTING"
    SYNCING_DATA = "SYNCING_DATA"
    FETCHING_DATA = "FETCHING_DATA"
    CALCULATING_STATISTICS = "CALCULATING_STATISTICS"
    TUNING_BATCH_SIZE = "TUNING_BATCH_SIZE"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    CANCELLING = "CANCELLING"

    # Legacy values kept for deserializing in-flight jobs. Those jobs keep
    # these statuses until they finish; they are not rewritten to the new
    # lifecycle values.
    PREPARING_DATA = "PREPARING_DATA"
    PENDING = "PENDING"
    RUNNING = "RUNNING"


# Statuses the training process owns (including legacy RUNNING). The cloud
# poller must not overwrite these with a coarser GCE-derived status.
VM_REPORTED_TRAINING_STATUSES = frozenset({
    TrainingJobStatus.SYNCING_DATA,
    TrainingJobStatus.FETCHING_DATA,
    TrainingJobStatus.CALCULATING_STATISTICS,
    TrainingJobStatus.TUNING_BATCH_SIZE,
    TrainingJobStatus.TRAINING,
    # Legacy in-flight jobs stay on RUNNING until they complete.
    TrainingJobStatus.RUNNING,
})


class TrainingPhaseProgress(BaseModel):
    """Progress within the current pre-training phase.

    Attributes:
        num_completed_items: Completed units for the current phase
            (e.g. synced or downloaded recordings).
        num_total_items: Total units for the current phase.
    """

    num_completed_items: int | None = None
    num_total_items: int | None = None


class TrainingProgress(BaseModel):
    """Progress / phase update reported by the training process.

    Attributes:
        epoch: Current training epoch.
        step: Current training step.
        seconds_per_epoch: Wall-clock seconds for the latest completed
            post-warmup epoch, if measured.
        status: Optional lifecycle phase (e.g. SYNCING_DATA, TRAINING).
        phase_progress: Optional progress within the current phase.
    """

    epoch: int | None = None
    step: int | None = None
    seconds_per_epoch: float | None = None
    status: TrainingJobStatus | None = None
    phase_progress: TrainingPhaseProgress | None = None


class TrainingJob(BaseModel):
    """Training job record.

    Attributes:
        id: The unique identifier for the job.
        name: The name of the job.
        dataset_id: The ID of the dataset being used.
        sync_freq: The frequency the dataset should be synced on.
        synced_dataset_id: The ID of the synced dataset, if applicable.
        algorithm: The name of the algorithm being used.
        algorithm_id: The ID of the algorithm, if applicable.
        status: The current status of the job.
        cloud_compute_job_id: The ID of the cloud compute job, if applicable.
        zone: The GCP zone where the job is running, if applicable.
        launch_time: The time the job was launched.
        start_time: The time the job started, if applicable.
        end_time: The time the job ended, if applicable.
        epoch: The current epoch of the training job.
        step: The current step of the training job.
        algorithm_config: Configuration parameters for the algorithm.
        gpu_type: The type of GPU used for the job.
        num_gpus: The number of GPUs used for the job.
        resumed_at: The time the job was resumed, if applicable.
        resumed_from_job_id: The ID of the job whose checkpoint this job started
            from, if applicable.
        resumed_from_checkpoint: The name of the checkpoint this job started
            from, if applicable.
        root_job_id: The ID of the first job in this job's family of runs, or
            None if this job is the root.
        start_epoch: The epoch this run started from, taken from the checkpoint
            it resumed from. 0 for runs trained from scratch.
        previous_training_time: The time spent on the previous training, if applicable.
        seconds_per_epoch: Wall-clock seconds for the latest completed post-warmup
            epoch, if measured. Used by clients to estimate remaining time.
        phase_progress_done: Completed units for the current pre-training phase
            (synced recordings or downloaded videos), if applicable.
        phase_progress_total: Total units for the current pre-training phase,
            if applicable.
        error: Any error message associated with the job, if applicable.
        resume_points: Epochs at which the job was resumed in place.
        input_cross_embodiment_description: List of data types for the input data.
        output_cross_embodiment_description: List of data types for the output data.
        deleted: True if the job is marked for deletion and its resources are
            not yet removed.
        deleted_at: The time the job was marked for deletion, if applicable.
    """

    id: str
    name: str
    dataset_id: str
    synced_dataset_id: str | None = None
    algorithm: str
    algorithm_id: str | None = None
    status: TrainingJobStatus
    cloud_compute_job_id: str | None = None
    zone: str | None = None
    launch_time: float
    start_time: float | None = None
    end_time: float | None = None
    epoch: int = Field(default=-1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    step: int = Field(default=-1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    algorithm_config: dict[str, Any] = Field(
        default_factory=lambda: {}, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    gpu_type: GPUType = Field(
        default=GPUType.NVIDIA_TESLA_T4, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    num_gpus: int = Field(default=1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    disk_size_gb: int = Field(default=500, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    resumed_at: float | None = None
    resumed_from_job_id: str | None = None
    resumed_from_checkpoint: str | None = None
    root_job_id: str | None = None
    start_epoch: int = Field(default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    previous_training_time: float | None = None
    seconds_per_epoch: float | None = None
    phase_progress_done: int | None = None
    phase_progress_total: int | None = None
    error: str | None = None
    resume_points: list[float] = Field(
        default_factory=lambda: [], json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    input_cross_embodiment_description: CrossEmbodimentDescription = Field(
        default_factory=lambda: {}, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    output_cross_embodiment_description: CrossEmbodimentDescription = Field(
        default_factory=lambda: {}, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    synchronization_details: SynchronizationDetails
    deleted: bool = Field(default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    deleted_at: datetime | None = Field(
        default=None, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)
