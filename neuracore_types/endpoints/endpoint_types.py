"""Endpoint response models shared across the monorepo."""

from enum import Enum

from pydantic import BaseModel, Field

from neuracore_types.episode import EmbodimentDescription
from neuracore_types.hardware import GPUType
from neuracore_types.utils.pydantic_to_ts import REQUIRED_WITH_DEFAULT_FLAG


class EndpointStatus(str, Enum):
    """Model endpoint status."""

    CREATING = "creating"
    ACTIVE = "active"
    FAILED = "failed"
    DEACTIVATED = "deactivated"


class Endpoint(BaseModel):
    """Response model for endpoint operations.

    Attributes:
        id: Unique identifier for the endpoint
        name: Display name of the endpoint
        training_job_id: ID of the training job used for this endpoint
        status: Current status of the endpoint
        gpu_type: Type of GPU used by the endpoint
        create_time: Timestamp when the endpoint was created
        update_time: Timestamp when the endpoint was last updated
        endpoint_url: URL for accessing the endpoint (if active)
        error: Error message if the endpoint failed
        error_code: Error code if the endpoint failed
        expiration_time: Timestamp when the endpoint expires (if set)
        cloud_compute_job_id: ID of the cloud compute job running this endpoint
        zone: The GCP zone where the endpoint is running
        input_embodiment_description: Optional model input data order
        output_embodiment_description: Optional model output data order
    """

    id: str
    name: str
    training_job_id: str
    zone: str | None = None
    status: EndpointStatus
    gpu_type: GPUType = Field(
        default=GPUType.NVIDIA_TESLA_T4,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    create_time: float
    update_time: float | None = None
    endpoint_url: str | None = None
    error: str | None = None
    error_code: int | None = None
    expiration_time: float | None = None
    cloud_compute_job_id: str | None = None
    input_embodiment_description: EmbodimentDescription | None = None
    output_embodiment_description: EmbodimentDescription | None = None

    class Config:
        """Configuration for Pydantic models."""

        use_enum_values = True
