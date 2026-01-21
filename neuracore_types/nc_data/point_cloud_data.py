"""3D point cloud data with optional RGB colouring and camera parameters."""

import base64
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.importer.config import DistanceUnitsConfig
from neuracore_types.importer.transform import (
    DataTransform,
    DataTransformSequence,
    Scale,
)
from neuracore_types.nc_data.nc_data import (
    DataItemStats,
    NCData,
    NCDataImportConfig,
    NCDataStats,
)
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class PointCloudDataStats(NCDataStats):
    """Statistics for PointCloudData."""

    type: Literal["PointCloudDataStats"] = Field(
        default="PointCloudDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    points: DataItemStats
    rgb_points: DataItemStats
    extrinsics: DataItemStats
    intrinsics: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class PointCloudDataImportConfig(NCDataImportConfig):
    """Import configuration for PointCloudData."""

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        # Add Scale transform to convert mm to m
        if self.format.distance_units == DistanceUnitsConfig.MM:
            transform_list.append(Scale(factor=0.001))

        for item in self.mapping:
            item.transforms = DataTransformSequence(transforms=transform_list)


class PointCloudData(NCData):
    """3D point cloud data with optional RGB colouring and camera parameters.

    Represents 3D spatial data from depth sensors or LiDAR systems with
    optional colour information and camera calibration for registration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["PointCloudData"] = Field(
        default="PointCloudData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    points: Optional[np.ndarray] = None  # (N, 3) float16
    rgb_points: Optional[np.ndarray] = None  # (N, 3) uint8
    extrinsics: Optional[np.ndarray] = None  # (4, 4) float16
    intrinsics: Optional[np.ndarray] = None  # (3, 3) float16

    @field_validator("points", "rgb_points")
    @classmethod
    def validate_points(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate that points have correct shape."""
        if v is not None and len(v) == 0:
            raise ValueError("Points array cannot be empty.")
        if v is not None and (v.ndim != 2 or v.shape[1] != 3):
            raise ValueError(
                "Points must have shape (N, 3), "
                f"got {v.shape if v is not None else None}"
            )
        return v

    @classmethod
    def sample(cls) -> "PointCloudData":
        """Sample an example PointCloudData instance."""
        return cls(
            points=np.zeros((1000, 3), dtype=np.float16),
            rgb_points=np.zeros((1000, 3), dtype=np.uint8),
            extrinsics=np.eye(4, dtype=np.float16),
            intrinsics=np.eye(3, dtype=np.float16),
        )

    @staticmethod
    def _compute_stats(
        data: Optional[np.ndarray], default_shape: tuple[int, ...]
    ) -> DataItemStats:
        """Compute statistics for a data array."""
        if data is None:
            zeros = np.zeros(default_shape, dtype=np.float32)
            return DataItemStats(
                mean=zeros,
                std=zeros,
                count=np.zeros(len(default_shape), dtype=np.int32),
                min=zeros,
                max=zeros,
            )

        return DataItemStats(
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0),
            count=np.array([data.shape[0]] * data.shape[1], dtype=np.int32),
            min=np.min(data, axis=0),
            max=np.max(data, axis=0),
        )

    @staticmethod
    def _matrix_stats(shape: tuple[int, ...], dtype: Any) -> DataItemStats:
        """Create placeholder statistics for matrix data."""
        zeros = np.zeros(shape, dtype=dtype)
        return DataItemStats(
            mean=zeros, std=zeros, count=np.array([1]), min=zeros, max=zeros
        )

    def calculate_statistics(self) -> PointCloudDataStats:
        """Calculate the statistics for this data type."""
        return PointCloudDataStats(
            points=self._compute_stats(self.points, (3,)),
            rgb_points=self._compute_stats(self.rgb_points, (3,)),
            extrinsics=self._matrix_stats((4, 4), np.float16),
            intrinsics=self._matrix_stats((3, 3), np.float16),
        )

    # Validators for point data (base64)
    @field_validator("points", mode="before")
    @classmethod
    def decode_points(cls, v: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Decode points to NumPy array."""
        if isinstance(v, str):
            return np.frombuffer(
                base64.b64decode(v.encode("utf-8")), dtype=np.float16
            ).reshape(-1, 3)
        return v

    @field_validator("rgb_points", mode="before")
    @classmethod
    def decode_rgb_points(cls, v: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Decode rgb_points to NumPy array."""
        if isinstance(v, str):
            return np.frombuffer(
                base64.b64decode(v.encode("utf-8")), dtype=np.uint8
            ).reshape(-1, 3)
        return v

    # Validators for camera matrices (tolist)
    @field_validator("extrinsics", mode="before")
    @classmethod
    def decode_extrinsics(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode extrinsics to NumPy array."""
        return np.array(v, dtype=np.float16) if isinstance(v, list) else v

    @field_validator("intrinsics", mode="before")
    @classmethod
    def decode_intrinsics(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode intrinsics to NumPy array."""
        return np.array(v, dtype=np.float16) if isinstance(v, list) else v

    # Serializers for point data (base64)
    @field_serializer("points", when_used="json")
    def serialize_points(self, v: Optional[np.ndarray]) -> Optional[str]:
        """Serialize points to base64 string."""
        if v is not None:
            return base64.b64encode(v.astype(np.float16).tobytes()).decode("utf-8")
        return None

    @field_serializer("rgb_points", when_used="json")
    def serialize_rgb_points(self, v: Optional[np.ndarray]) -> Optional[str]:
        """Serialize rgb_points to base64 string."""
        if v is not None:
            return base64.b64encode(v.astype(np.uint8).tobytes()).decode("utf-8")
        return None

    # Serializers for camera matrices (tolist)
    @field_serializer("extrinsics", when_used="json")
    def serialize_extrinsics(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize extrinsics to JSON list."""
        return v.tolist() if v is not None else None

    @field_serializer("intrinsics", when_used="json")
    def serialize_intrinsics(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize intrinsics to JSON list."""
        return v.tolist() if v is not None else None
