"""Camera data including images and camera parameters."""

from typing import Literal, Optional, Union

import numpy as np
from PIL import Image
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats
from neuracore_types.utils.depth_utils import depth_to_rgb, rgb_to_depth
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)

RGB_URI_PREFIX = "data:image/png;base64,"


class CameraDataStats(NCDataStats):
    """Statistics for CameraData."""

    type: Literal["CameraDataStats"] = Field(
        default="CameraDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    frame: DataItemStats
    extrinsics: DataItemStats
    intrinsics: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class CameraData(NCData):
    """Camera sensor data including images and calibration information.

    Contains image data along with camera intrinsic and extrinsic parameters
    for 3D reconstruction and computer vision applications. The frame field
    is populated during dataset iteration for efficiency.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=fix_required_with_defaults
    )

    frame_idx: int = Field(
        default=0,  # Needed so we can index video after sync
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    extrinsics: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    frame: Optional[Union[np.ndarray, str]] = Field(
        default=None,
        exclude=True,  # Exclude from JSON serialization to avoid base64 overhead
        description="Only filled in when using dataset iter. Frames are manually added to JSON when needed for remote streaming."
    )

    def calculate_statistics(self) -> CameraDataStats:
        """Calculate the statistics for this data type.

        Returns:
            Dictionary attribute names to their corresponding DataItemStats.
        """
        # Frame stats will averaged over all pixels, but just return image as is
        if isinstance(self.frame, np.ndarray):
            frame_stats = DataItemStats(
                mean=self.frame.copy(),
                std=np.zeros_like(self.frame),
                count=np.array([1], dtype=np.int32),
                min=self.frame.copy(),
                max=self.frame.copy(),
            )
        else:
            # TODO: When we calculate stats on backend, we may not have access to the
            #   frame. so just return dummy stats for now.
            frame_stats = DataItemStats(
                mean=np.array([1], dtype=np.uint8),
                std=np.array([0], dtype=np.uint8),
                count=np.array([1], dtype=np.int32),
                min=np.array([1], dtype=np.uint8),
                max=np.array([1], dtype=np.uint8),
            )
        zero_ext = np.zeros((4, 4), dtype=np.float16)
        zero_intr = np.zeros((3, 3), dtype=np.float16)
        extrinsics_stats = DataItemStats(
            mean=zero_ext,
            std=zero_ext,
            count=np.array([1]),
            min=zero_ext,
            max=zero_ext,
        )
        intrinsics_stats = DataItemStats(
            mean=zero_intr,
            std=zero_intr,
            count=np.array([1]),
            min=zero_intr,
            max=zero_intr,
        )
        return CameraDataStats(
            frame=frame_stats,
            extrinsics=extrinsics_stats,
            intrinsics=intrinsics_stats,
        )

    @field_validator("extrinsics", mode="before")
    @classmethod
    def decode_extrinsics(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode extrinsics to NumPy array.

        Args:
            v: List of lists or NumPy array

        Returns:
            Decoded NumPy array or None
        """
        return np.array(v, dtype=np.float16) if isinstance(v, list) else v

    @field_validator("intrinsics", mode="before")
    @classmethod
    def decode_intrinsics(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode intrinsics to NumPy array.

        Args:
            v: List of lists or NumPy array

        Returns:
            Decoded NumPy array or None
        """
        return np.array(v, dtype=np.float16) if isinstance(v, list) else v

    @field_serializer("extrinsics", when_used="json")
    def serialize_extrinsics(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Encode NumPy array to JSON list.

        Args:
            v: NumPy array to encode

        Returns:
            Nested list or None
        """
        return v.tolist() if v is not None else None

    @field_serializer("intrinsics", when_used="json")
    def serialize_intrinsics(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Encode NumPy array to JSON list.

        Args:
            v: NumPy array to encode

        Returns:
            Nested list or None
        """
        return v.tolist() if v is not None else None


class RGBCameraData(CameraData):
    """RGB camera data subclass.

    Specialization of CameraData for RGB images.
    """

    type: Literal["RGBCameraData"] = Field(
        default="RGBCameraData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @classmethod
    def sample(cls) -> "CameraData":
        """Sample an example RGBCameraData instance.

        Returns:
            CameraData: Sampled instance
        """
        return cls(
            extrinsics=np.eye(4, dtype=np.float16),
            intrinsics=np.eye(3, dtype=np.float16),
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
        )


class DepthCameraData(CameraData):
    """Depth camera data subclass.

    Specialization of CameraData for depth images.
    """

    type: Literal["DepthCameraData"] = Field(
        default="DepthCameraData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @classmethod
    def sample(cls) -> "CameraData":
        """Sample an example DepthCameraData instance.

        Returns:
            CameraData: Sampled instance
        """
        return cls(
            extrinsics=np.eye(4, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
            frame=np.zeros((480, 640), dtype=np.float32),
        )
