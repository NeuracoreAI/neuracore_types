"""Training request models."""

from typing import Any

from pydantic import BaseModel

from neuracore_types.episode.episode import RobotDataSpec
from neuracore_types.synchronization.synchronization import SynchronizationDetails
from neuracore_types.training.training import GPUType


class TrainingJobRequest(BaseModel):
    """Request model for starting a training job.

    Attributes:
        dataset_id: Identifier of the dataset to use for training.
        name: Optional name for the training job.
        algorithm_id: Identifier of the training algorithm to use.
        algorithm_config: Configuration parameters for the training algorithm.
        frequency: Synchronization frequency in Hz.
        gpu_type: Type of GPU to use for training.
        num_gpus: Number of GPUs to use for training.
        input_robot_data_spec: The robust mapping of robot to input
            data types to names
        output_robot_data_spec: The robust mapping of robot to output
            data types to names
    """

    dataset_id: str
    name: str
    algorithm_id: str
    algorithm_config: dict[str, Any]
    gpu_type: GPUType
    num_gpus: int
    synchronization_details: SynchronizationDetails
    input_robot_data_spec: RobotDataSpec
    output_robot_data_spec: RobotDataSpec


class InternalStartTrainingJobRequest(BaseModel):
    """Request model for starting a training job.

    Attributes:
        org_id: The ID of the organization.
        user_id: The ID of the user.
        job_uuid: The UUID of the job.
        dataset_id: The ID of the dataset.
        synchronization_details: Details for dataset synchronization.
        job_name: The name of the job.
        algorithm_id: The ID of the algorithm to use.
        algorithm_config: Configuration parameters for the algorithm.
        gpu_type: The type of GPU to use for the job.
        num_gpus: The number of GPUs to allocate for the job.
        input_robot_data_spec: Mapping of robot to input data types to names
        output_robot_data_spec: Mapping of robot to output data types to names
    """

    org_id: str
    user_id: str
    job_uuid: str
    dataset_id: str
    job_name: str
    algorithm_id: str
    algorithm_config: dict[str, Any]
    gpu_type: GPUType
    num_gpus: int
    synchronization_details: SynchronizationDetails
    input_robot_data_spec: RobotDataSpec
    output_robot_data_spec: RobotDataSpec
