"""Records and configuration models for data synthesis jobs.

Data synthesis expands an existing dataset by generating new recordings from
its recordings. Only the RGB video is regenerated; every other trace is copied
from the source recording byte for byte, so action labels stay valid.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class DataSynthesisAlgorithmSpec(BaseModel):
    """What one synthesis algorithm offers.

    Read out of the algorithm package the worker runs, rather than declared
    here, so an algorithm can be added to that package without any type being
    added to this one.

    Attributes:
        name: What a request names the algorithm, such as "greenaug".
        config_name: The config group the worker's command line selects the
            algorithm by, which is not the same string.
        title: A short name for the algorithm.
        description: What the algorithm does.
        json_schema: The algorithm's parameters, as JSON Schema: their types,
            defaults, choices, which are required, and what each one means.
            Whatever collects them renders a form from this.
    """

    name: str
    config_name: str
    title: str
    description: str
    json_schema: dict[str, Any]


class DataSynthesisAlgorithmConfig(BaseModel):
    """The algorithm a synthesis job runs, and how it is configured.

    The parameters are not typed here on purpose. What an algorithm offers is
    described by its own spec, and the synthesis service checks them against
    it, so adding an algorithm changes nothing in this package.

    Attributes:
        name: What the algorithm is named by its spec.
        parameters: The parameters to run it with. Anything the algorithm
            defaults may be left out.
    """

    name: str
    parameters: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class DataSynthesisJobStatus(str, Enum):
    """Data synthesis job status.

    There is no data preparation state: synthesis reads recordings directly and
    never synchronizes, so a job goes straight from queued to running.
    """

    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    CANCELLING = "CANCELLING"


class DataSynthesisStage(str, Enum):
    """The step of the pipeline a running synthesis job is currently on."""

    STARTING = "STARTING"
    LOADING_MODELS = "LOADING_MODELS"
    SYNTHESIZING = "SYNTHESIZING"
    FINALIZING = "FINALIZING"


class DataSynthesisJob(BaseModel):
    """Data synthesis job record.

    Attributes:
        id: The unique identifier for the job.
        name: The name of the job.
        dataset_id: The ID of the dataset being expanded.
        output_dataset_id: The ID of the dataset the job writes to.
        output_dataset_name: The name of the dataset the job writes to.
        algorithm: Configuration of the algorithm the job runs.
        scaling_factor: How many recordings to produce per original.
        include_original: Whether the output dataset also contains the source
            recordings. They are referenced, never copied.
        status: The current status of the job.
        stage: The pipeline step the job is on, while it is running.
        cloud_compute_job_id: The ID of the cloud compute job, if applicable.
        zone: The GCP zone where the job is running, if applicable.
        launch_time: The time the job was launched.
        start_time: The time the job started, if applicable.
        end_time: The time the job ended, if applicable.
        recordings_total: How many recordings the job expects to synthesize.
        recordings_completed: How many recordings the job has synthesized.
        error: Any error message associated with the job, if applicable.
        deleted: True if the job is marked for deletion and its resources are
            not yet removed.
        deleted_at: The time the job was marked for deletion, if applicable.
    """

    id: str
    name: str
    dataset_id: str
    output_dataset_id: str | None = None
    output_dataset_name: str
    algorithm: DataSynthesisAlgorithmConfig
    scaling_factor: int
    include_original: bool = Field(
        default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    status: DataSynthesisJobStatus
    stage: DataSynthesisStage | None = None
    cloud_compute_job_id: str | None = None
    zone: str | None = None
    launch_time: float
    start_time: float | None = None
    end_time: float | None = None
    recordings_total: int = Field(
        default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    recordings_completed: int = Field(
        default=0, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    error: str | None = None
    deleted: bool = Field(default=False, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG)
    deleted_at: datetime | None = Field(
        default=None, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)
