"""Camera data including images and camera parameters."""

from typing import Any, Literal, cast

import numpy as np
import torch
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)

RGB_URI_PREFIX = "data:image/png;base64,"


class BatchedRGBData(BatchedNCData):
    """Batched RGB camera data."""

    type: Literal["BatchedRGBData"] = Field(
        default="BatchedRGBData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    frame: torch.Tensor  # (B, T, 3, H, W) uint8
    extrinsics: torch.Tensor  # (B, T, 4, 4) float16
    intrinsics: torch.Tensor  # (B, T, 3, 3) float16

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @field_validator("frame", mode="before")
    @classmethod
    def decode_frame(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode frame field to torch.Tensor."""
        return cls._create_tensor_handlers("frame")[0](v)

    @field_serializer("frame", when_used="json")
    def serialize_frame(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize frame field to base64 string."""
        return self._create_tensor_handlers("frame")[1](v)

    @field_validator("extrinsics", mode="before")
    @classmethod
    def decode_extrinsics(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode extrinsics field to torch.Tensor."""
        return cls._create_tensor_handlers("extrinsics")[0](v)

    @field_serializer("extrinsics", when_used="json")
    def serialize_extrinsics(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize extrinsics field to base64 string."""
        return self._create_tensor_handlers("extrinsics")[1](v)

    @field_validator("intrinsics", mode="before")
    @classmethod
    def decode_intrinsics(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode intrinsics field to torch.Tensor."""
        return cls._create_tensor_handlers("intrinsics")[0](v)

    @field_serializer("intrinsics", when_used="json")
    def serialize_intrinsics(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize intrinsics field to base64 string."""
        return self._create_tensor_handlers("intrinsics")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedRGBData from input nc_data.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        from neuracore_types.nc_data.camera_data import RGBCameraData

        rgb_data: RGBCameraData = cast(RGBCameraData, nc_data)
        # Need to change from (H, W, 3) to (3, H, W)
        frame = np.array(rgb_data.frame)
        frame = (
            torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if rgb_data.extrinsics is not None:
            extrinsics = (
                torch.tensor(rgb_data.extrinsics, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            extrinsics = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        if rgb_data.intrinsics is not None:
            intrinsics = (
                torch.tensor(rgb_data.intrinsics, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            intrinsics = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        return cls(frame=frame, extrinsics=extrinsics, intrinsics=intrinsics)

    def transform_nc_data(self) -> None:
        """Apply in-place transformations, e.g. reshaping, reordering dimensions, etc.

        Example usage might include:
        - Reordering dimensions from (B, T, C, H, W) to (B, T, H, W, C) for images
        - Normalizing values to a specific range
        - Resizing images to a common resolution

        Args:
            nc_data: BatchedNCData instance to transform

        Returns:
            BatchedNCData: Transformed instance
        """
        # TODO: In future, the resizing spec should come from the algorithm
        #   or from user specification, not be hardcoded here.

        # Resize each frame to 224x224 while preserving (B, T, C, H, W).
        if self.frame.shape[-2:] != (224, 224):
            batch_size, time_steps, channels, _, _ = self.frame.shape
            reshaped = self.frame.reshape(
                batch_size * time_steps, channels, *self.frame.shape[-2:]
            )
            resized = torch.nn.functional.interpolate(
                reshaped, size=(224, 224), mode="bilinear", align_corners=False
            )
            self.frame = resized.reshape(batch_size, time_steps, channels, 224, 224)

    @classmethod
    def from_nc_data_list(cls, nc_data_list: list[NCData]) -> "BatchedRGBData":
        """Create BatchedRGBData from list of RGBCameraData.

        Args:
            nc_data_list: List of RGBCameraData instances to convert

        Returns:
            BatchedRGBData with shape (1, T, 3, H, W) where T = len(nc_data_list)
        """
        from neuracore_types.nc_data.camera_data import RGBCameraData

        frames = []
        extrinsics_list = []
        intrinsics_list = []

        for nc in nc_data_list:
            rgb_data: RGBCameraData = cast(RGBCameraData, nc)
            # (H, W, 3) -> (3, H, W)
            frame = np.array(rgb_data.frame).transpose(2, 0, 1)
            frames.append(frame)

            if rgb_data.extrinsics is not None:
                extrinsics_list.append(rgb_data.extrinsics)
            else:
                extrinsics_list.append(np.zeros((4, 4), dtype=np.float32))

            if rgb_data.intrinsics is not None:
                intrinsics_list.append(rgb_data.intrinsics)
            else:
                intrinsics_list.append(np.zeros((3, 3), dtype=np.float32))

        # Shape: (1, T, 3, H, W)
        frame_tensor = torch.from_numpy(np.stack(frames)).to(torch.float32).unsqueeze(0)
        # Shape: (1, T, 4, 4)
        extrinsics_tensor = (
            torch.from_numpy(np.stack(extrinsics_list)).to(torch.float32).unsqueeze(0)
        )
        # Shape: (1, T, 3, 3)
        intrinsics_tensor = (
            torch.from_numpy(np.stack(intrinsics_list)).to(torch.float32).unsqueeze(0)
        )

        return cls(
            frame=frame_tensor,
            extrinsics=extrinsics_tensor,
            intrinsics=intrinsics_tensor,
        )

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedRGBData":
        """Sample an example instance of BatchedRGBData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedRGBData: Sampled instance
        """
        return cls(
            frame=torch.zeros(
                (batch_size, time_steps, 3, 224, 224), dtype=torch.float32
            ),
            extrinsics=torch.zeros((batch_size, time_steps, 4, 4), dtype=torch.float32),
            intrinsics=torch.zeros((batch_size, time_steps, 3, 3), dtype=torch.float32),
        )


class BatchedDepthData(BatchedNCData):
    """Batched depth camera data."""

    type: Literal["BatchedDepthData"] = Field(
        default="BatchedDepthData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    frame: torch.Tensor  # (B, T, 1, H, W) uint16 units
    depth_scale: torch.Tensor  # (B, T) float32 metres per unit
    extrinsics: torch.Tensor  # (B, T, 4, 4) float16
    intrinsics: torch.Tensor  # (B, T, 3, 3) float16

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    def to_compute_dtype(self) -> None:
        """Convert frames to float32 meters, with 0 for no return.

        Zero, negative and non-finite depth map to 0, and the depth scale
        becomes 1.0, so the conversion is idempotent.
        """
        depth_scale_m = self.depth_scale.to(
            device=self.frame.device, dtype=torch.float32
        )
        depth_m = self.frame.to(torch.float32) * depth_scale_m[..., None, None, None]
        depth_m = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        self.frame = depth_m.clamp_min(0.0)
        self.depth_scale = torch.ones_like(depth_scale_m)

    @field_validator("depth_scale", mode="before")
    @classmethod
    def decode_depth_scale(cls, encoded: dict[str, Any]) -> torch.Tensor:
        """Decode depth_scale field to torch.Tensor."""
        return cls._create_tensor_handlers("depth_scale")[0](encoded)

    @field_serializer("depth_scale", when_used="json")
    def serialize_depth_scale(self, depth_scale: torch.Tensor) -> dict[str, Any]:
        """Serialize depth_scale field to a JSON tensor encoding."""
        return self._create_tensor_handlers("depth_scale")[1](depth_scale)

    @field_validator("frame", mode="before")
    @classmethod
    def decode_frame(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode frame field to torch.Tensor."""
        return cls._create_tensor_handlers("frame")[0](v)

    @field_serializer("frame", when_used="json")
    def serialize_frame(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize frame field to base64 string."""
        return self._create_tensor_handlers("frame")[1](v)

    @field_validator("extrinsics", mode="before")
    @classmethod
    def decode_extrinsics(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode extrinsics field to torch.Tensor."""
        return cls._create_tensor_handlers("extrinsics")[0](v)

    @field_serializer("extrinsics", when_used="json")
    def serialize_extrinsics(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize extrinsics field to base64 string."""
        return self._create_tensor_handlers("extrinsics")[1](v)

    @field_validator("intrinsics", mode="before")
    @classmethod
    def decode_intrinsics(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode intrinsics field to torch.Tensor."""
        return cls._create_tensor_handlers("intrinsics")[0](v)

    @field_serializer("intrinsics", when_used="json")
    def serialize_intrinsics(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize intrinsics field to base64 string."""
        return self._create_tensor_handlers("intrinsics")[1](v)

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedDepthData from input nc_data.

        Args:
            nc_data: NCData instance to convert

        Returns:
            BatchedNCData: Converted BatchedNCData instance
        """
        from neuracore_types.nc_data.camera_data import DepthCameraData

        depth_data: DepthCameraData = cast(DepthCameraData, nc_data)
        # Need to change from (H, W) to (1, H, W)
        frame = np.array(depth_data.frame)
        frame = (
            torch.tensor(frame, dtype=torch.uint16)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        depth_scale = torch.tensor([[depth_data.depth_scale_m]], dtype=torch.float32)
        if depth_data.extrinsics is not None:
            extrinsics = (
                torch.tensor(depth_data.extrinsics, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            extrinsics = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        if depth_data.intrinsics is not None:
            intrinsics = (
                torch.tensor(depth_data.intrinsics, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            intrinsics = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        return cls(
            frame=frame,
            depth_scale=depth_scale,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

    @classmethod
    def from_nc_data_list(cls, nc_data_list: list[NCData]) -> "BatchedDepthData":
        """Create BatchedDepthData from list of DepthCameraData.

        Args:
            nc_data_list: List of DepthCameraData instances to convert

        Returns:
            BatchedDepthData with shape (1, T, 1, H, W) where T = len(nc_data_list)
        """
        from neuracore_types.nc_data.camera_data import DepthCameraData

        frames = []
        depth_scales = []
        extrinsics_list = []
        intrinsics_list = []

        for nc in nc_data_list:
            depth_data: DepthCameraData = cast(DepthCameraData, nc)
            # (H, W) -> (1, H, W)
            frame = np.array(depth_data.frame)
            frames.append(frame[np.newaxis, ...])
            depth_scales.append(depth_data.depth_scale_m)
            if depth_data.extrinsics is not None:
                extrinsics_list.append(depth_data.extrinsics)
            else:
                extrinsics_list.append(np.zeros((4, 4), dtype=np.float32))
            if depth_data.intrinsics is not None:
                intrinsics_list.append(depth_data.intrinsics)
            else:
                intrinsics_list.append(np.zeros((3, 3), dtype=np.float32))

        # Shape: (1, T, 1, H, W)
        frame_tensor = torch.from_numpy(np.stack(frames)).unsqueeze(0)
        # Shape: (1, T)
        depth_scale_tensor = torch.tensor([depth_scales], dtype=torch.float32)
        # Shape: (1, T, 4, 4)
        extrinsics_tensor = (
            torch.from_numpy(np.stack(extrinsics_list)).to(torch.float32).unsqueeze(0)
        )
        # Shape: (1, T, 3, 3)
        intrinsics_tensor = (
            torch.from_numpy(np.stack(intrinsics_list)).to(torch.float32).unsqueeze(0)
        )

        return cls(
            frame=frame_tensor,
            depth_scale=depth_scale_tensor,
            extrinsics=extrinsics_tensor,
            intrinsics=intrinsics_tensor,
        )

    @classmethod
    def sample(cls, batch_size: int = 1, time_steps: int = 1) -> "BatchedDepthData":
        """Sample an example instance of BatchedDepthData.

        Args:
            batch_size: Number of samples in the batch
            time_steps: Number of time steps in the sequence

        Returns:
            BatchedDepthData: Sampled instance
        """
        return cls(
            frame=torch.zeros(
                (batch_size, time_steps, 1, 224, 224), dtype=torch.uint16
            ),
            depth_scale=torch.full((batch_size, time_steps), 1e-4, dtype=torch.float32),
            extrinsics=torch.zeros((batch_size, time_steps, 4, 4), dtype=torch.float32),
            intrinsics=torch.zeros((batch_size, time_steps, 3, 3), dtype=torch.float32),
        )

    def transform_nc_data(self) -> None:
        """Apply in-place transformations, e.g. reshaping, reordering dimensions, etc.

        Example usage might include:
        - Reordering dimensions from (B, T, C, H, W) to (B, T, H, W, C) for images
        - Normalizing values to a specific range
        - Resizing images to a common resolution

        Args:
            nc_data: BatchedNCData instance to transform
        """
        # Resize each frame to 224x224 while preserving (B, T, C, H, W).
        if self.frame.shape[-2:] != (224, 224):
            batch_size, time_steps, channels, _, _ = self.frame.shape
            reshaped = self.frame.reshape(
                batch_size * time_steps, channels, *self.frame.shape[-2:]
            )
            resized = torch.nn.functional.interpolate(
                reshaped.to(torch.float32), size=(224, 224), mode="nearest"
            ).to(self.frame.dtype)
            self.frame = resized.reshape(batch_size, time_steps, channels, 224, 224)
