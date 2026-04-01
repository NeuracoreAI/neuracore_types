"""Configuration options for importing data into Neuracore.

This module defines options of the configuration for specifying the format
of the input data.

"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class DatasetTypeConfig(str, Enum):
    """Enumeration of supported dataset types."""

    RLDS = "RLDS"
    LEROBOT = "LEROBOT"
    TFDS = "TFDS"


class OutputDatasetConfig(BaseModel):
    """Output dataset configuration."""

    name: str
    tags: list[str] = Field(default_factory=list)
    description: str = ""


class RobotConfig(BaseModel):
    """Robot configuration."""

    name: str
    urdf_path: str | None = None
    mjcf_path: str | None = None
    overwrite_existing: bool = False


class VisualJointInputTypeConfig(str, Enum):
    """Types of visual joint positions.

    GRIPPER: Extract gripper joint positions from the gripper open amount.
    CUSTOM: Directly use the specified joint positions from the source.
    """

    GRIPPER = "GRIPPER"
    CUSTOM = "CUSTOM"


class JointPositionInputTypeConfig(str, Enum):
    """Types of joint positions.

    END_EFFECTOR: Extract joint positions from the end effector pose using
    Inverse Kinematics.
    CUSTOM: Directly use the specified joint positions from the source.
    """

    END_EFFECTOR = "END_EFFECTOR"
    CUSTOM = "CUSTOM"


class EndEffectorPoseInputTypeConfig(str, Enum):
    """Types of end effector poses.

    JOINT_POSITIONS: Extract end effector pose from the joint positions using
    Forward Kinematics.
    CUSTOM: Directly use the specified end effector pose from the source.
    """

    JOINT_POSITIONS = "JOINT_POSITIONS"
    CUSTOM = "CUSTOM"


class RotationConfig(str, Enum):
    """Types of rotations."""

    QUATERNION = "QUATERNION"
    MATRIX = "MATRIX"
    EULER = "EULER"
    AXIS_ANGLE = "AXIS_ANGLE"


class AngleConfig(str, Enum):
    """Types of angles."""

    DEGREES = "DEGREES"
    RADIANS = "RADIANS"


class PoseConfig(str, Enum):
    """Types of poses."""

    MATRIX = "MATRIX"
    POSITION_ORIENTATION = "POSITION_ORIENTATION"


class QuaternionOrderConfig(str, Enum):
    """Order of quaternion."""

    XYZW = "XYZW"
    WXYZ = "WXYZ"


class EulerOrderConfig(str, Enum):
    """Order of euler angles."""

    XYZ = "XYZ"
    ZYX = "ZYX"
    YXZ = "YXZ"
    ZXY = "ZXY"
    YZX = "YZX"
    XZY = "XZY"


class ActionTypeConfig(str, Enum):
    """Action type configuration.

    Specifies whether actions are absolute or relative to the current state.
    """

    ABSOLUTE = "ABSOLUTE"
    RELATIVE = "RELATIVE"


class ActionSpaceConfig(str, Enum):
    """Action space configuration.

    Specifies whether actions are in joint space or end effector space.
    """

    END_EFFECTOR = "END_EFFECTOR"
    JOINT = "JOINT"


class IndexRangeConfig(BaseModel):
    """Configuration for index range of data extraction."""

    start: int
    end: int

    @model_validator(mode="after")
    def validate_index_range(self) -> "IndexRangeConfig":
        """Validate that index range is valid."""
        if self.start > self.end:
            raise ValueError("Index range start must be less than end")
        return self


class NormalizeConfig(BaseModel):
    """Configuration for normalizing data."""

    min: float = 0.0
    max: float = 1.0


class ImageConventionConfig(str, Enum):
    """Convention of image channels."""

    CHANNELS_LAST = "CHANNELS_LAST"
    CHANNELS_FIRST = "CHANNELS_FIRST"


class ImageChannelOrderConfig(str, Enum):
    """Order of image channels."""

    RGB = "RGB"
    BGR = "BGR"


class LanguageConfig(str, Enum):
    """Types of languages."""

    STRING = "STRING"
    BYTES = "BYTES"


class TorqueUnitsConfig(str, Enum):
    """Types of torque units."""

    NM = "NM"
    NCM = "NCM"


class DistanceUnitsConfig(str, Enum):
    """Types of distance units."""

    M = "M"
    MM = "MM"


class OrientationConfig(BaseModel):
    """Configuration for orientation of poses."""

    type: RotationConfig = RotationConfig.QUATERNION
    quaternion_order: QuaternionOrderConfig = QuaternionOrderConfig.XYZW
    euler_order: EulerOrderConfig = EulerOrderConfig.XYZ
    angle_units: AngleConfig = AngleConfig.RADIANS
    align_frame_roll: float = 0.0
    align_frame_pitch: float = 0.0
    align_frame_yaw: float = 0.0


class IntrinsicsConfig(str, Enum):
    """Format of camera intrinsics data in the source dataset.

    MATRIX: 3x3 camera intrinsics matrix.
    FLAT: Flat array of [fx, fy, cx, cy].
    """

    MATRIX = "MATRIX"
    FLAT = "FLAT"
