"""3D point cloud data with optional RGB colouring and camera parameters."""

from __future__ import annotations

import base64
import json
import struct
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.importer.config import DistanceUnitsConfig
from neuracore_types.importer.data_config import PointCloudDataMappingItem
from neuracore_types.importer.transform import (
    CastToNumpyDtype,
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
from neuracore_types.utils.numpy_array import NumpyArray
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

    mapping: list[PointCloudDataMappingItem] = Field(default_factory=list)

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []

        # Add Scale transform to convert mm to m
        if self.format.distance_units == DistanceUnitsConfig.MM:
            transform_list.append(Scale(factor=0.001))

        # Convert to float16
        transform_list.append(CastToNumpyDtype(dtype=np.float16))

        for item in self.mapping:
            item.transforms = DataTransformSequence(transforms=transform_list)


class PointCloudData(NCData):
    """3D point cloud data with optional RGB colouring and camera parameters.

    Represents 3D spatial data from depth sensors or LiDAR systems with
    optional colour information and camera calibration for registration.
    The points and rgb_points fields are populated during dataset iteration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["PointCloudData"] = Field(
        default="PointCloudData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    frame_idx: int = Field(
        default=0,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    points: NumpyArray | None = None  # (N, 3) float16
    rgb_points: NumpyArray | None = None  # (N, 3) uint8
    extrinsics: NumpyArray | None = None  # (4, 4) float16
    intrinsics: NumpyArray | None = None  # (3, 3) float16

    @field_validator("points", "rgb_points")
    @classmethod
    def validate_points(cls, v: np.ndarray | None) -> np.ndarray | None:
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
    def sample(cls) -> PointCloudData:
        """Sample an example PointCloudData instance."""
        return cls(
            points=np.zeros((1000, 3), dtype=np.float16),
            rgb_points=np.zeros((1000, 3), dtype=np.uint8),
            extrinsics=np.eye(4, dtype=np.float16),
            intrinsics=np.eye(3, dtype=np.float16),
        )

    @staticmethod
    def _compute_stats(
        data: np.ndarray | None, default_shape: tuple[int, ...]
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
    def decode_points(cls, v: str | np.ndarray) -> np.ndarray | None:
        """Decode points to NumPy array."""
        if isinstance(v, str):
            return np.frombuffer(
                base64.b64decode(v.encode("utf-8")), dtype=np.float16
            ).reshape(-1, 3)
        return v

    @field_validator("rgb_points", mode="before")
    @classmethod
    def decode_rgb_points(cls, v: str | np.ndarray) -> np.ndarray | None:
        """Decode rgb_points to NumPy array."""
        if isinstance(v, str):
            return np.frombuffer(
                base64.b64decode(v.encode("utf-8")), dtype=np.uint8
            ).reshape(-1, 3)
        return v

    # Validators for camera matrices (tolist)
    @field_validator("extrinsics", mode="before")
    @classmethod
    def decode_extrinsics(cls, v: list | np.ndarray) -> np.ndarray | None:
        """Decode extrinsics to NumPy array."""
        return np.array(v, dtype=np.float16) if isinstance(v, list) else v

    @field_validator("intrinsics", mode="before")
    @classmethod
    def decode_intrinsics(cls, v: list | np.ndarray) -> np.ndarray | None:
        """Decode intrinsics to NumPy array."""
        return np.array(v, dtype=np.float16) if isinstance(v, list) else v

    # Serializers for point data (base64)
    @field_serializer("points", when_used="json")
    def serialize_points(self, v: np.ndarray | None) -> str | None:
        """Serialize points to base64 string."""
        if v is not None:
            return base64.b64encode(v.astype(np.float16).tobytes()).decode("utf-8")
        return None

    @field_serializer("rgb_points", when_used="json")
    def serialize_rgb_points(self, v: np.ndarray | None) -> str | None:
        """Serialize rgb_points to base64 string."""
        if v is not None:
            return base64.b64encode(v.astype(np.uint8).tobytes()).decode("utf-8")
        return None

    # Serializers for camera matrices (tolist)
    @field_serializer("extrinsics", when_used="json")
    def serialize_extrinsics(self, v: np.ndarray | None) -> list | None:
        """Serialize extrinsics to JSON list."""
        return v.tolist() if v is not None else None

    @field_serializer("intrinsics", when_used="json")
    def serialize_intrinsics(self, v: np.ndarray | None) -> list | None:
        """Serialize intrinsics to JSON list."""
        return v.tolist() if v is not None else None


POINT_CLOUD_WIRE_VERSION = 1
POINT_CLOUD_TRACE_BIN_FILENAME = "trace.bin"

_METADATA_LEN_FORMAT = "<I"


def encode_point_cloud_frame_parts(
    data: PointCloudData,
) -> tuple[bytes, bytes, memoryview, memoryview | None]:
    """Encode one frame into parts suitable for zero-copy transport.

    Returns:
        Tuple of (header, metadata_json, points_view, optional rgb_view).
    """
    if data.points is None:
        raise ValueError("Point cloud points are required for wire encoding")

    points = np.ascontiguousarray(data.points, dtype=np.float16)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

    rgb: np.ndarray | None = None
    if data.rgb_points is not None:
        rgb = np.ascontiguousarray(data.rgb_points, dtype=np.uint8)
        if rgb.shape != points.shape:
            raise ValueError(
                "RGB points must have the same shape as points, "
                f"got {rgb.shape} vs {points.shape}"
            )

    metadata: dict[str, Any] = {
        "type": "PointCloudData",
        "version": POINT_CLOUD_WIRE_VERSION,
        "timestamp": data.timestamp,
        "num_points": int(points.shape[0]),
        "points_dtype": "float16",
        "points_nbytes": int(points.nbytes),
        "rgb_nbytes": int(rgb.nbytes) if rgb is not None else 0,
    }
    if data.extrinsics is not None:
        metadata["extrinsics"] = np.asarray(data.extrinsics, dtype=np.float16).tolist()
    if data.intrinsics is not None:
        metadata["intrinsics"] = np.asarray(data.intrinsics, dtype=np.float16).tolist()

    metadata_json = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header = struct.pack(_METADATA_LEN_FORMAT, len(metadata_json))
    points_view = memoryview(points).cast("B")
    rgb_view = memoryview(rgb).cast("B") if rgb is not None else None
    return header, metadata_json, points_view, rgb_view


def decode_point_cloud_wire_metadata(payload: bytes) -> tuple[dict[str, Any], int]:
    """Parse wire metadata and return metadata dict plus end offset."""
    if len(payload) < struct.calcsize(_METADATA_LEN_FORMAT):
        raise ValueError("Point cloud wire payload is too short")

    metadata_len = struct.unpack(_METADATA_LEN_FORMAT, payload[:4])[0]
    metadata_start = 4
    metadata_end = metadata_start + metadata_len
    if metadata_len <= 0 or metadata_end > len(payload):
        raise ValueError("Invalid point cloud wire metadata length")

    try:
        metadata = json.loads(payload[metadata_start:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid point cloud wire metadata JSON") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Point cloud wire metadata must be a JSON object")

    version = metadata.get("version")
    if version != POINT_CLOUD_WIRE_VERSION:
        raise ValueError(f"Unsupported point cloud wire version: {version}")

    return metadata, metadata_end


def decode_point_cloud_frame(payload: bytes) -> PointCloudData:
    """Decode one wire frame into PointCloudData."""
    metadata, metadata_end = decode_point_cloud_wire_metadata(payload)

    num_points = metadata.get("num_points")
    points_nbytes = metadata.get("points_nbytes")
    rgb_nbytes = metadata.get("rgb_nbytes", 0)
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("Invalid num_points in point cloud wire metadata")
    if not isinstance(points_nbytes, int) or points_nbytes <= 0:
        raise ValueError("Invalid points_nbytes in point cloud wire metadata")
    if not isinstance(rgb_nbytes, int) or rgb_nbytes < 0:
        raise ValueError("Invalid rgb_nbytes in point cloud wire metadata")

    expected_points_nbytes = num_points * 3 * np.dtype(np.float16).itemsize
    if points_nbytes != expected_points_nbytes:
        raise ValueError("points_nbytes does not match num_points for float16 XYZ data")

    offset = metadata_end
    points_end = offset + points_nbytes
    if points_end > len(payload):
        raise ValueError("Point cloud wire payload truncated for points array")

    points = np.frombuffer(
        payload[offset:points_end], dtype=np.float16, count=num_points * 3
    ).reshape(num_points, 3)
    offset = points_end

    rgb_points = None
    if rgb_nbytes > 0:
        expected_rgb_nbytes = num_points * 3 * np.dtype(np.uint8).itemsize
        if rgb_nbytes != expected_rgb_nbytes:
            raise ValueError("rgb_nbytes does not match num_points for uint8 RGB data")
        rgb_end = offset + rgb_nbytes
        if rgb_end > len(payload):
            raise ValueError("Point cloud wire payload truncated for rgb array")
        rgb_points = np.frombuffer(
            payload[offset:rgb_end], dtype=np.uint8, count=num_points * 3
        ).reshape(num_points, 3)
        offset = rgb_end

    if offset != len(payload):
        raise ValueError("Point cloud wire payload has trailing bytes")

    extrinsics = None
    if "extrinsics" in metadata and metadata["extrinsics"] is not None:
        extrinsics = np.asarray(metadata["extrinsics"], dtype=np.float16)

    intrinsics = None
    if "intrinsics" in metadata and metadata["intrinsics"] is not None:
        intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float16)

    timestamp = metadata.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        raise ValueError("Invalid timestamp in point cloud wire metadata")

    return PointCloudData(
        timestamp=float(timestamp),
        points=points.copy(),
        rgb_points=rgb_points.copy() if rgb_points is not None else None,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
