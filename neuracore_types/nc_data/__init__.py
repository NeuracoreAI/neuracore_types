"""Init."""

from enum import Enum
from typing import Annotated, Union

from pydantic import Field

from neuracore_types.nc_data.camera_data import CameraData  # noqa: F401
from neuracore_types.nc_data.camera_data import (
    CameraDataStats,
    DepthCameraData,
    RGBCameraData,
)
from neuracore_types.nc_data.custom_1d_data import Custom1DData, Custom1DDataStats
from neuracore_types.nc_data.end_effector_pose_data import (
    EndEffectorPoseData,
    EndEffectorPoseDataStats,
)
from neuracore_types.nc_data.joint_data import JointData, JointDataStats
from neuracore_types.nc_data.language_data import LanguageData, LanguageDataStats
from neuracore_types.nc_data.nc_data import NCData, NCDataStats  # noqa: F401
from neuracore_types.nc_data.parallel_gripper_open_amount_data import (
    ParallelGripperOpenAmountData,
    ParallelGripperOpenAmountDataStats,
)
from neuracore_types.nc_data.point_cloud_data import PointCloudData, PointCloudDataStats
from neuracore_types.nc_data.pose_data import PoseData, PoseDataStats

NCDataUnion = Annotated[
    Union[
        JointData,
        RGBCameraData,
        DepthCameraData,
        PoseData,
        EndEffectorPoseData,
        ParallelGripperOpenAmountData,
        PointCloudData,
        LanguageData,
        Custom1DData,
    ],
    Field(discriminator="type"),
]

NCDataStatsUnion = Annotated[
    Union[
        JointDataStats,
        CameraDataStats,
        PoseDataStats,
        EndEffectorPoseDataStats,
        ParallelGripperOpenAmountDataStats,
        PointCloudDataStats,
        LanguageDataStats,
        Custom1DDataStats,
    ],
    Field(discriminator="type"),
]


class DataType(str, Enum):
    """Enumeration of supported data types in the Neuracore system.

    Defines the standard data categories used for dataset organization,
    model training, and data processing pipelines.
    """

    # Robot state
    JOINT_POSITIONS = "JOINT_POSITIONS"
    JOINT_VELOCITIES = "JOINT_VELOCITIES"
    JOINT_TORQUES = "JOINT_TORQUES"
    JOINT_TARGET_POSITIONS = "JOINT_TARGET_POSITIONS"
    END_EFFECTOR_POSES = "END_EFFECTOR_POSES"
    PARALLEL_GRIPPER_OPEN_AMOUNTS = "PARALLEL_GRIPPER_OPEN_AMOUNTS"

    # Vision
    RGB_IMAGES = "RGB_IMAGES"
    DEPTH_IMAGES = "DEPTH_IMAGES"
    POINT_CLOUDS = "POINT_CLOUDS"

    # Other
    POSES = "POSES"
    LANGUAGE = "LANGUAGE"
    CUSTOM_1D = "CUSTOM_1D"


DATA_TYPE_TO_NC_DATA_CLASS: dict[DataType, type[NCData]] = {
    DataType.JOINT_POSITIONS: JointData,
    DataType.JOINT_VELOCITIES: JointData,
    DataType.JOINT_TORQUES: JointData,
    DataType.JOINT_TARGET_POSITIONS: JointData,
    DataType.END_EFFECTOR_POSES: EndEffectorPoseData,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: ParallelGripperOpenAmountData,
    DataType.RGB_IMAGES: RGBCameraData,
    DataType.DEPTH_IMAGES: DepthCameraData,
    DataType.POINT_CLOUDS: PointCloudData,
    DataType.POSES: PoseData,
    DataType.LANGUAGE: LanguageData,
    DataType.CUSTOM_1D: Custom1DData,
}
