"""Configuration options for importing data into Neuracore.

This module defines options of the configuration for specifying the format
of the input data.

"""

# cspell:ignore MCAP

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class DatasetTypeConfig(str, Enum):
    """Enumeration of supported dataset types."""

    MCAP = "MCAP"
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


class Frame(str, Enum):
    """Which frame a constant SE(3) transform X composes in.

    WORLD: pre-multiply, T' = X @ T -- re-express the pose in a new
        reference/world frame (e.g. dataset poses recorded in a world frame
        rotated/translated from the robot's base frame).
    TOOL: post-multiply, T' = T @ X -- apply a fixed offset in the body's own
        (tool) frame (e.g. the dataset's tool identity is offset from the URDF
        link's identity; bridge: gripper-down vs gripper-forward).

    A conjugation X @ T @ X^-1 (re-expressing an action *delta* in another
    frame) is expressed as a WORLD entry followed by a TOOL entry holding the
    inverse transform.
    """

    WORLD = "WORLD"
    TOOL = "TOOL"


class RollPitchYaw(BaseModel):
    """Euler angles in radians: roll about X, pitch about Y, yaw about Z."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class XYZ(BaseModel):
    """Cartesian translation in metres."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class FrameTransformConfig(BaseModel):
    """A constant SE(3) transform applied to a pose.

    `rotation` is XYZ-euler (roll about X, pitch about Y, yaw about Z) in
    radians and `translation` is (x, y, z); together they form the homogeneous
    transform X = [R | t]. `frame` selects how X composes with the pose
    (see Frame).
    """

    frame: Frame
    rotation: RollPitchYaw = Field(default_factory=RollPitchYaw)
    translation: XYZ = Field(default_factory=XYZ)


class OrientationConfig(BaseModel):
    """Configuration for orientation of poses."""

    type: RotationConfig = RotationConfig.QUATERNION
    quaternion_order: QuaternionOrderConfig = QuaternionOrderConfig.XYZW
    euler_order: EulerOrderConfig = EulerOrderConfig.XYZ
    angle_units: AngleConfig = AngleConfig.RADIANS
    # When EULER, interpret euler_order as extrinsic (fixed-axis) rotations,
    # matching the ROS tf default (static-axis xyz). Default False preserves
    # the historical intrinsic interpretation; opt in for datasets that
    # recorded orientation via ROS tf or any other extrinsic convention.
    extrinsic_euler: bool = False
    # Constant SE(3) transforms applied to the pose after conversion, composed
    # in list order. Each entry chooses WORLD (pre-multiply, X @ T) or BODY
    # (post-multiply, T @ X); see FrameTransformConfig and Frame.
    frame_transforms: list[FrameTransformConfig] = Field(default_factory=list)


class IntrinsicsConfig(str, Enum):
    """Format of camera intrinsics data in the source dataset.

    MATRIX: 3x3 camera intrinsics matrix.
    FLAT: Flat array of [fx, fy, cx, cy].
    """

    MATRIX = "MATRIX"
    FLAT = "FLAT"
