"""Request and response models for training jobs."""

from enum import Enum
from typing import Any

from ordered_set import OrderedSet
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from neuracore_types.episode.episode import RobotDataSpec
from neuracore_types.nc_data import DataType, NCDataStatsUnion
from neuracore_types.synchronization.synchronization import SynchronizationDetails
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class GPUType(str, Enum):
    """GPU types available for training."""

    NVIDIA_H100_80GB = "NVIDIA_H100_80GB"
    NVIDIA_A100_80GB = "NVIDIA_A100_80GB"
    NVIDIA_TESLA_A100 = "NVIDIA_TESLA_A100"
    NVIDIA_TESLA_V100 = "NVIDIA_TESLA_V100"
    NVIDIA_TESLA_P100 = "NVIDIA_TESLA_P100"
    NVIDIA_TESLA_T4 = "NVIDIA_TESLA_T4"
    NVIDIA_TESLA_P4 = "NVIDIA_TESLA_P4"
    NVIDIA_L4 = "NVIDIA_L4"


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
            dataset_statistics={
                DataType.RGB_IMAGES: DataItemStats(...),
                DataType.JOINT_POSITIONS: DataItemStats(...),
                DataType.JOINT_TARGET_POSITIONS: DataItemStats(...),
            },
        )
    """

    input_data_types: OrderedSet[DataType]
    output_data_types: OrderedSet[DataType]
    # Dataset statistics for all data types, where the len of the list corresponds
    # to the max number of data items for that data type (across all robots)
    dataset_statistics: dict[DataType, list[NCDataStatsUnion]]
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

    PREPARING_DATA = "PREPARING_DATA"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


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
        previous_training_time: The time spent on the previous training, if applicable.
        error: Any error message associated with the job, if applicable.
        resume_points: List of timestamps where the job can be resumed.
        input_robot_data_spec: List of data types for the input data.
        output_robot_data_spec: List of data types for the output data.
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
    resumed_at: float | None = None
    previous_training_time: float | None = None
    error: str | None = None
    resume_points: list[float] = Field(
        default_factory=lambda: [], json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    input_robot_data_spec: RobotDataSpec = Field(
        default_factory=lambda: {}, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    output_robot_data_spec: RobotDataSpec = Field(
        default_factory=lambda: {}, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    synchronization_details: SynchronizationDetails

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)
