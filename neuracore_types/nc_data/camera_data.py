"""Camera data including images and camera parameters."""

import base64
from io import BytesIO
from typing import Literal, Optional, Union

import numpy as np
from PIL import Image
from pydantic import ConfigDict, field_serializer, field_validator

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats
from neuracore_types.utils.depth_utils import depth_to_rgb, rgb_to_depth

RGB_URI_PREFIX = "data:image/png;base64,"


class CameraDataStats(NCDataStats):
    """Statistics for CameraData."""

    type: Literal["CameraDataStats"] = "CameraDataStats"

    frame: DataItemStats
    extrinsics: DataItemStats
    intrinsics: DataItemStats


class CameraData(NCData):
    """Camera sensor data including images and calibration information.

    Contains image data along with camera intrinsic and extrinsic parameters
    for 3D reconstruction and computer vision applications. The frame field
    is populated during dataset iteration for efficiency.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_idx: int = 0  # Needed so we can index video after sync
    extrinsics: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    frame: Optional[Union[np.ndarray, str]] = (
        None  # Only filled in when using dataset iter
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

    @staticmethod
    def _encode_image(arr: np.ndarray) -> str:
        pil_image = Image.fromarray(arr)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return RGB_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _decode_image(data: str) -> np.ndarray:
        img_bytes = base64.b64decode(data.removeprefix(RGB_URI_PREFIX))
        buffer = BytesIO(img_bytes)
        pil_image = Image.open(buffer)
        return np.array(pil_image)

    @field_validator("frame", mode="before")
    @classmethod
    def decode_frame(cls, v: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Decode base64 string to NumPy array if needed.

        Args:
            v: Base64 encoded string or NumPy array

        Returns:
            Decoded NumPy array or None
        """
        return cls._decode_image(v) if isinstance(v, str) else v

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

    # --- Serializers (encode on dump) ---
    @field_serializer("frame", when_used="json")
    def serialize_frame(self, v: Optional[np.ndarray]) -> Optional[str]:
        """Encode NumPy array to base64 string if needed.

        Args:
            v: NumPy array to encode

        Returns:
            Base64 encoded string or None
        """
        return self._encode_image(v) if v is not None else None

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

    type: Literal["RGBCameraData"] = "RGBCameraData"

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

    type: Literal["DepthCameraData"] = "DepthCameraData"

    @staticmethod
    def _encode_image(arr: np.ndarray) -> str:
        arr = depth_to_rgb(arr)
        pil_image = Image.fromarray(arr)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return RGB_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _decode_image(data: str) -> np.ndarray:
        img_bytes = base64.b64decode(data.removeprefix(RGB_URI_PREFIX))
        buffer = BytesIO(img_bytes)
        pil_image = Image.open(buffer)
        depth = rgb_to_depth(np.array(pil_image))
        assert depth.ndim == 2
        return depth

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
