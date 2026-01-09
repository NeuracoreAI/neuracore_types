"""Init."""

from typing import Annotated, Union

from pydantic import Field

from neuracore_types.batched_nc_data.batched_camera_data import (
    BatchedDepthData,
    BatchedRGBData,
)
from neuracore_types.batched_nc_data.batched_custom_1d_data import BatchedCustom1DData
from neuracore_types.batched_nc_data.batched_end_effector_pose_data import (
    BatchedEndEffectorPoseData,
)
from neuracore_types.batched_nc_data.batched_joint_data import BatchedJointData
from neuracore_types.batched_nc_data.batched_language_data import BatchedLanguageData
from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.batched_nc_data.batched_parallel_gripper_open_amount_data import (
    BatchedParallelGripperOpenAmountData,
)
from neuracore_types.batched_nc_data.batched_point_cloud_data import (
    BatchedPointCloudData,
)
from neuracore_types.batched_nc_data.batched_pose_data import BatchedPoseData
from neuracore_types.nc_data import DataType

BatchedNCDataUnion = Annotated[
    Union[
        BatchedJointData,
        BatchedRGBData,
        BatchedDepthData,
        BatchedPoseData,
        BatchedEndEffectorPoseData,
        BatchedParallelGripperOpenAmountData,
        BatchedPointCloudData,
        BatchedLanguageData,
        BatchedCustom1DData,
    ],
    Field(discriminator="type"),
]


DATA_TYPE_TO_BATCHED_NC_DATA_CLASS: dict[DataType, type[BatchedNCData]] = {
    DataType.JOINT_POSITIONS: BatchedJointData,
    DataType.JOINT_VELOCITIES: BatchedJointData,
    DataType.JOINT_TORQUES: BatchedJointData,
    DataType.JOINT_TARGET_POSITIONS: BatchedJointData,
    DataType.END_EFFECTOR_POSES: BatchedEndEffectorPoseData,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: BatchedParallelGripperOpenAmountData,
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: BatchedParallelGripperOpenAmountData,
    DataType.RGB_IMAGES: BatchedRGBData,
    DataType.DEPTH_IMAGES: BatchedDepthData,
    DataType.POINT_CLOUDS: BatchedPointCloudData,
    DataType.POSES: BatchedPoseData,
    DataType.LANGUAGE: BatchedLanguageData,
    DataType.CUSTOM_1D: BatchedCustom1DData,
}
