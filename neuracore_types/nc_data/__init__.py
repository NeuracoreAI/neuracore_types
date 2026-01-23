"""Init."""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Union

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from neuracore_types.importer.config import (
    DatasetTypeConfig,
    OutputDatasetConfig,
    RobotConfig,
)
from neuracore_types.nc_data.camera_data import CameraData  # noqa: F401
from neuracore_types.nc_data.camera_data import (
    CameraDataStats,
    DepthCameraData,
    DepthCameraDataImportConfig,
    RGBCameraData,
    RGBCameraDataImportConfig,
)
from neuracore_types.nc_data.custom_1d_data import (
    Custom1DData,
    Custom1DDataImportConfig,
    Custom1DDataStats,
)
from neuracore_types.nc_data.end_effector_pose_data import (
    EndEffectorPoseData,
    EndEffectorPoseDataStats,
)
from neuracore_types.nc_data.joint_data import (
    JointData,
    JointDataStats,
    JointPositionsDataImportConfig,
    JointTorquesDataImportConfig,
    JointVelocitiesDataImportConfig,
)
from neuracore_types.nc_data.language_data import (
    LanguageData,
    LanguageDataImportConfig,
    LanguageDataStats,
)
from neuracore_types.nc_data.nc_data import (  # noqa: F401
    NCData,
    NCDataImportConfig,
    NCDataStats,
)
from neuracore_types.nc_data.parallel_gripper_open_amount_data import (
    ParallelGripperOpenAmountData,
    ParallelGripperOpenAmountDataImportConfig,
    ParallelGripperOpenAmountDataStats,
)
from neuracore_types.nc_data.point_cloud_data import (
    PointCloudData,
    PointCloudDataImportConfig,
    PointCloudDataStats,
)
from neuracore_types.nc_data.pose_data import (
    PoseData,
    PoseDataImportConfig,
    PoseDataStats,
)

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

NCDataImportConfigUnion = Union[
    RGBCameraDataImportConfig,
    DepthCameraDataImportConfig,
    PointCloudDataImportConfig,
    JointPositionsDataImportConfig,
    JointVelocitiesDataImportConfig,
    JointTorquesDataImportConfig,
    PoseDataImportConfig,
    ParallelGripperOpenAmountDataImportConfig,
    LanguageDataImportConfig,
    Custom1DDataImportConfig,
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
    PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS = "PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS"

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
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: ParallelGripperOpenAmountData,
    DataType.RGB_IMAGES: RGBCameraData,
    DataType.DEPTH_IMAGES: DepthCameraData,
    DataType.POINT_CLOUDS: PointCloudData,
    DataType.POSES: PoseData,
    DataType.LANGUAGE: LanguageData,
    DataType.CUSTOM_1D: Custom1DData,
}

DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS: dict[DataType, type[NCDataImportConfig]] = {
    DataType.JOINT_POSITIONS: JointPositionsDataImportConfig,
    DataType.JOINT_VELOCITIES: JointVelocitiesDataImportConfig,
    DataType.JOINT_TORQUES: JointTorquesDataImportConfig,
    DataType.JOINT_TARGET_POSITIONS: JointPositionsDataImportConfig,
    DataType.END_EFFECTOR_POSES: PoseDataImportConfig,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: (ParallelGripperOpenAmountDataImportConfig),
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: (
        ParallelGripperOpenAmountDataImportConfig
    ),
    DataType.RGB_IMAGES: RGBCameraDataImportConfig,
    DataType.DEPTH_IMAGES: DepthCameraDataImportConfig,
    DataType.POINT_CLOUDS: PointCloudDataImportConfig,
    DataType.POSES: PoseDataImportConfig,
    DataType.LANGUAGE: LanguageDataImportConfig,
    DataType.CUSTOM_1D: Custom1DDataImportConfig,
}


class DatasetImportConfig(BaseModel):
    """Main dataset configuration model.

    Specifies the configuration for importing data to Neuracore.
    For each data type, define the format of the incoming data which will be
    converted to the format expected by Neuracore.
    """

    input_dataset_name: str
    output_dataset: OutputDatasetConfig
    robot: RobotConfig
    frequency: float | None = None
    data_import_config: dict[DataType, NCDataImportConfigUnion] = Field(
        default_factory=dict
    )
    dataset_type: DatasetTypeConfig | None = None

    @classmethod
    def from_file(cls, config_path: Path) -> "DatasetImportConfig":
        """Load dataset configuration from a YAML or JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()
        try:
            with config_path.open("r") as f:
                if suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(f)
                elif suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {suffix}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load config file: {exc}") from exc

        # Populate import config for each data type
        data["data_import_config"] = {
            key: DATA_TYPE_TO_NC_DATA_IMPORT_CONFIG_CLASS[DataType(key)](**value)
            for key, value in data["data_import_config"].items()
        }

        return cls(**data)
