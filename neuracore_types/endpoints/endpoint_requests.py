"""Endpoint request models for deploying models."""

from pydantic import BaseModel, Field

from neuracore_types.episode import DataSpec
from neuracore_types.training.training import GPUType


class DeploymentConfig(BaseModel):
    """Configuration for model deployment.

    Attributes:
        machine_type: Type of machine to use for deployment.
        gpu_type: Type of GPU to use for deployment.
        gpu_count: Number of GPUs to use for deployment.
    """

    machine_type: str | None = "n1-standard-4"
    gpu_type: GPUType | None = GPUType.NVIDIA_TESLA_T4
    gpu_count: int | None = 1


class DeploymentRequest(BaseModel):
    """Request model for deploying a model.

    Attributes:
        training_id: Identifier of the trained model to deploy.
        name: Optional name for the endpoint.
        ttl: Optional time-to-live in seconds for the endpoint.
        model_input_order: Specification of the model input data order.
        model_output_order: Specification of the model output data order.
        config: Deployment configuration parameters.
    """

    training_id: str
    name: str | None = None
    ttl: int | None = None
    model_input_order: DataSpec
    model_output_order: DataSpec
    config: DeploymentConfig = Field(default_factory=DeploymentConfig)
