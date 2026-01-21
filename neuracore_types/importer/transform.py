"""Tools to convert data for use in Neuracore datasets.

This module provides tools to convert and standardize raw input data for use
in Neuracore datasets. The transformations are applied to the data in the
order they are specified.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.spatial.transform import Rotation as R

from neuracore_types.importer.config import (
    AngleConfig,
    EulerOrderConfig,
    ImageChannelOrderConfig,
    ImageConventionConfig,
    PoseConfig,
    QuaternionOrderConfig,
    RotationConfig,
)


class DataTransform(BaseModel):
    """Base class for data transformations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Transform the data."""
        raise NotImplementedError("Subclasses must implement __call__")


class DataTransformSequence(DataTransform):
    """Sequence of data transformations."""

    transforms: list[DataTransform] = Field(default_factory=list)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply all transforms in sequence to the data."""
        for transform in self.transforms:
            data = transform(data)
        return data


class Rotation(DataTransform):
    """Convert rotations to quaternion xyzw."""

    rotation_type: RotationConfig = RotationConfig.QUATERNION
    angle_type: AngleConfig = AngleConfig.RADIANS
    seq: QuaternionOrderConfig | EulerOrderConfig = QuaternionOrderConfig.XYZW

    @property
    def degrees(self) -> bool:
        """Whether angles are in degrees."""
        return self.angle_type == AngleConfig.DEGREES

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Convert rotation to quaternion xyzw format."""
        if self.rotation_type == RotationConfig.QUATERNION:
            if self.seq == QuaternionOrderConfig.XYZW:
                return values
            elif self.seq == QuaternionOrderConfig.WXYZ:
                return values[[1, 2, 3, 0]]
            else:
                raise ValueError(f"Unsupported quaternion order: {self.seq}")
        elif self.rotation_type == RotationConfig.MATRIX:
            return R.from_matrix(values).as_quat()
        elif self.rotation_type == RotationConfig.EULER:
            return R.from_euler(
                self.seq.value.lower(), values, degrees=self.degrees
            ).as_quat()
        elif self.rotation_type == RotationConfig.AXIS_ANGLE:
            return R.from_rotvec(values, degrees=self.degrees).as_quat()
        else:
            raise ValueError(f"Unsupported rotation type: {self.rotation_type}")


class Pose(DataTransform):
    """Convert pose to position and rotation."""

    pose_type: PoseConfig
    rotation_type: RotationConfig = RotationConfig.QUATERNION
    angle_type: AngleConfig = AngleConfig.RADIANS
    seq: QuaternionOrderConfig | EulerOrderConfig = QuaternionOrderConfig.XYZW

    def model_post_init(self, __context: object) -> None:
        """Initialize rotation_transform after model initialization."""
        if self.pose_type == PoseConfig.POSITION_ORIENTATION:
            object.__setattr__(
                self,
                "rotation_transform",
                Rotation(
                    rotation_type=self.rotation_type,
                    angle_type=self.angle_type,
                    seq=self.seq,
                ),
            )
        elif self.pose_type == PoseConfig.MATRIX:
            object.__setattr__(
                self,
                "rotation_transform",
                Rotation(rotation_type=RotationConfig.MATRIX),
            )
        else:
            raise ValueError(f"Unsupported pose type: {self.pose_type}")

    def __call__(self, pose: np.ndarray) -> np.ndarray:
        """Convert pose to position and quaternion rotation."""
        if self.pose_type == PoseConfig.POSITION_ORIENTATION:
            return np.concatenate([pose[:3], self.rotation_transform(pose[3:])])
        elif self.pose_type == PoseConfig.MATRIX:
            if pose.ndim == 1:
                pose = pose.reshape(4, 4)
            rot_mat = pose[:3, :3]
            return np.concatenate([pose[:3, 3], self.rotation_transform(rot_mat)])


class ImageFormat(DataTransform):
    """Convert image format to HWC."""

    format: ImageConventionConfig

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Convert image format to HWC."""
        if self.format == ImageConventionConfig.CHANNELS_LAST:
            return image
        else:
            return image.transpose(1, 2, 0)


class ImageChannelOrder(DataTransform):
    """Convert image channel order to RGB.

    Expects image to be in HWC format.
    """

    order: ImageChannelOrderConfig

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Convert image channel order to RGB."""
        if self.order == ImageChannelOrderConfig.RGB:
            return image
        else:
            return image[..., [2, 1, 0]]


class CastToNumpyDtype(DataTransform):
    """Cast the data to a given numpy dtype."""

    dtype: np.dtype

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: np.dtype) -> np.dtype:
        """Convert dtype class to dtype instance if needed."""
        if isinstance(v, type) and issubclass(v, np.generic):
            return np.dtype(v)
        elif isinstance(v, np.dtype):
            return v
        else:
            return np.dtype(v)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Cast the data to the specified numpy dtype."""
        return data.astype(self.dtype)


class NumpyToScalar(DataTransform):
    """Convert the numpy array to a scalar."""

    def __call__(self, data: np.ndarray) -> float:
        """Convert numpy array to scalar."""
        return data.item()


class Scale(DataTransform):
    """Scale the data by a factor."""

    factor: float

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Scale the data by the factor."""
        return data * self.factor


class Clip(DataTransform):
    """Clip the data to a range."""

    min: float
    max: float

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Clip the data to the specified range."""
        return np.clip(data, self.min, self.max)


class Normalize(DataTransform):
    """Normalize the data from [min, max] to [0, 1]."""

    min: float
    max: float

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Normalize the data from [min, max] to [0, 1]."""
        return (data - self.min) / (self.max - self.min)


class Unnormalize(DataTransform):
    """Unnormalize the data from [0, 1] to [min, max]."""

    min: float
    max: float

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Unnormalize the data from [0, 1] to [min, max]."""
        return data * (self.max - self.min) + self.min


class FlipSign(DataTransform):
    """Flip the sign of the data."""

    def __call__(self, data: float) -> float:
        """Flip the sign of the data."""
        return data * -1.0


class Offset(DataTransform):
    """Offset the data by a value."""

    value: float

    def __call__(self, data: float) -> float:
        """Offset the data by the specified value."""
        return data + self.value


class NanToNum(DataTransform):
    """Convert NaN to 0."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Convert NaN, positive infinity, and negative infinity to 0."""
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


class DegreesToRadians(DataTransform):
    """Convert degrees to radians."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Convert degrees to radians."""
        return data * np.pi / 180.0


class LanguageFromBytes(DataTransform):
    """Convert language from bytes."""

    def __call__(self, data: bytes) -> str:
        """Convert bytes to UTF-8 string."""
        return data.decode("utf-8")
