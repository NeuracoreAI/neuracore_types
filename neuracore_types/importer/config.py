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
