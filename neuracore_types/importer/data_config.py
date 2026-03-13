"""Configuration classes for specifying the format of the input data.

This module defines configuration classes to specify the format of the input
data when importing it into Neuracore.

"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuracore_types.importer.config import (
    ActionSpaceConfig,
    ActionTypeConfig,
    AngleConfig,
    DistanceUnitsConfig,
    EndEffectorPoseInputTypeConfig,
    ImageChannelOrderConfig,
    ImageConventionConfig,
    IndexRangeConfig,
    IntrinsicsConfig,
    JointPositionInputTypeConfig,
    LanguageConfig,
    NormalizeConfig,
    OrientationConfig,
    PoseConfig,
    TorqueUnitsConfig,
    VisualJointInputTypeConfig,
)
from neuracore_types.importer.transform import DataTransformSequence


class MappingItem(BaseModel):
    """Mapping information to extract individual items from a data source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    source_name: str | None = None
    index: int | None = None
    index_range: IndexRangeConfig | None = None
    offset: float = 0.0
    inverted: bool = False
    transforms: DataTransformSequence = Field(
        default_factory=lambda: DataTransformSequence(transforms=[])
    )

    @model_validator(mode="after")
    def validate_index_provided(self) -> "MappingItem":
        """Validate that index an index_range are not provided together."""
        if self.index is not None and self.index_range is not None:
            raise ValueError("index and index_range cannot be provided together")
        return self


class RGBCameraDataMappingItem(MappingItem):
    """Mapping item for RGB camera streams, with optional calibration sources."""

    extrinsics_source: str | None = None
    intrinsics_source: str | None = None
    extrinsics_transforms: DataTransformSequence = Field(
        default_factory=lambda: DataTransformSequence(transforms=[])
    )
    intrinsics_transforms: DataTransformSequence = Field(
        default_factory=lambda: DataTransformSequence(transforms=[])
    )


class DepthCameraDataMappingItem(MappingItem):
    """Mapping item for depth camera streams, with optional calibration sources."""

    extrinsics_source: str | None = None
    intrinsics_source: str | None = None
    extrinsics_transforms: DataTransformSequence = Field(
        default_factory=lambda: DataTransformSequence(transforms=[])
    )
    intrinsics_transforms: DataTransformSequence = Field(
        default_factory=lambda: DataTransformSequence(transforms=[])
    )


class PointCloudDataMappingItem(MappingItem):
    """Mapping item for point cloud streams, with optional calibration sources."""

    extrinsics_source: str | None = None
    intrinsics_source: str | None = None
    extrinsics_transforms: DataTransformSequence = Field(
        default_factory=lambda: DataTransformSequence(transforms=[])
    )
    intrinsics_transforms: DataTransformSequence = Field(
        default_factory=lambda: DataTransformSequence(transforms=[])
    )


class DataFormat(BaseModel):
    """Per datatype format specifications.

    This class is used for the 'format:' section in YAML config files and
    when importing data. Relevant fields depend on the data type.
    """

    # RGB image format fields
    image_convention: ImageConventionConfig = ImageConventionConfig.CHANNELS_LAST
    order_of_channels: ImageChannelOrderConfig = ImageChannelOrderConfig.RGB
    normalized_pixel_values: bool = False

    # Units format fields
    angle_units: AngleConfig = AngleConfig.RADIANS
    torque_units: TorqueUnitsConfig = TorqueUnitsConfig.NM
    distance_units: DistanceUnitsConfig = DistanceUnitsConfig.M

    # Pose format fields
    pose_type: PoseConfig = PoseConfig.MATRIX
    orientation: OrientationConfig | None = None
    ee_pose_input_type: EndEffectorPoseInputTypeConfig = (
        EndEffectorPoseInputTypeConfig.CUSTOM
    )

    # Language format fields
    language_type: LanguageConfig = LanguageConfig.STRING

    # Normalize format fields
    normalize: NormalizeConfig | None = None

    # Gripper amount format fields
    invert_gripper_amount: bool = False

    # Visual joint type fields
    visual_joint_input_type: VisualJointInputTypeConfig = (
        VisualJointInputTypeConfig.CUSTOM
    )

    # Joint position type fields
    joint_position_input_type: JointPositionInputTypeConfig = (
        JointPositionInputTypeConfig.CUSTOM
    )
    ik_init_config: list[float] | None = None

    # Joint target position type fields
    action_type: ActionTypeConfig = ActionTypeConfig.ABSOLUTE
    action_space: ActionSpaceConfig = ActionSpaceConfig.JOINT

    # Camera extrinsics/intrinsics format fields
    extrinsics_format: PoseConfig = PoseConfig.MATRIX
    extrinsics_orientation: OrientationConfig | None = None
    intrinsics_format: IntrinsicsConfig = IntrinsicsConfig.MATRIX
