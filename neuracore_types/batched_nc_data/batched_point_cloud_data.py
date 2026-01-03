"""3D point cloud data with optional RGB colouring and camera parameters."""

from typing import Any, Literal, Optional, Union, cast

import torch
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.batched_nc_data.batched_nc_data import BatchedNCData
from neuracore_types.nc_data.nc_data import NCData
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class BatchedPointCloudData(BatchedNCData):
    """Batched 3D point cloud data with optional RGB colouring and camera parameters."""

    type: Literal["BatchedPointCloudData"] = Field(
        default="BatchedPointCloudData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    points: torch.Tensor  # (B, T, N, 3) float32
    rgb_points: Optional[torch.Tensor] = None  # (B, T, N, 3) uint8
    extrinsics: Optional[torch.Tensor] = None  # (B, T, 4, 4) float32
    intrinsics: Optional[torch.Tensor] = None  # (B, T, 3, 3) float32

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @field_validator("points", mode="before")
    @classmethod
    def decode_points(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode points field to torch.Tensor."""
        return cls._create_tensor_handlers("points")[0](v)

    @field_serializer("points", when_used="json")
    def serialize_points(self, v: torch.Tensor) -> dict[str, Any]:
        """Serialize points field to base64 string."""
        return self._create_tensor_handlers("points")[1](v)

    @field_validator("rgb_points", mode="before")
    @classmethod
    def decode_rgb_points(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode rgb_points field to torch.Tensor."""
        return (
            cls._create_tensor_handlers("rgb_points")[0](v) if v is not None else None
        )

    @field_serializer("rgb_points", when_used="json")
    def serialize_rgb_points(self, v: torch.Tensor) -> Union[dict[str, Any], None]:
        """Serialize rgb_points field to base64 string."""
        return (
            self._create_tensor_handlers("rgb_points")[1](v) if v is not None else None
        )

    @field_validator("extrinsics", mode="before")
    @classmethod
    def decode_extrinsics(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode extrinsics field to torch.Tensor."""
        return (
            cls._create_tensor_handlers("extrinsics")[0](v) if v is not None else None
        )

    @field_serializer("extrinsics", when_used="json")
    def serialize_extrinsics(self, v: torch.Tensor) -> Union[dict[str, Any], None]:
        """Serialize extrinsics field to base64 string."""
        return (
            self._create_tensor_handlers("extrinsics")[1](v) if v is not None else None
        )

    @field_validator("intrinsics", mode="before")
    @classmethod
    def decode_intrinsics(cls, v: dict[str, Any]) -> torch.Tensor:
        """Decode intrinsics field to torch.Tensor."""
        return (
            cls._create_tensor_handlers("intrinsics")[0](v) if v is not None else None
        )

    @field_serializer("intrinsics", when_used="json")
    def serialize_intrinsics(self, v: torch.Tensor) -> Union[dict[str, Any], None]:
        """Serialize intrinsics field to base64 string."""
        return (
            self._create_tensor_handlers("intrinsics")[1](v) if v is not None else None
        )

    @classmethod
    def from_nc_data(cls, nc_data: NCData) -> "BatchedNCData":
        """Create BatchedPointCloudData from PointCloudData."""
        from neuracore_types.nc_data.point_cloud_data import PointCloudData

        pc_data: PointCloudData = cast(PointCloudData, nc_data)
        # swap axes to (3, N)
        points = (
            torch.tensor(pc_data.points, dtype=torch.float32)
            .permute(1, 0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        rgb_points = extrinsics = intrinsics = None
        if pc_data.rgb_points is not None:
            rgb_points = (
                torch.tensor(pc_data.rgb_points, dtype=torch.uint8)
                .permute(1, 0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        if pc_data.extrinsics is not None:
            extrinsics = (
                torch.tensor(pc_data.extrinsics, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        if pc_data.intrinsics is not None:
            intrinsics = (
                torch.tensor(pc_data.intrinsics, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        return cls(
            points=points,
            rgb_points=rgb_points,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

    @classmethod
    def from_nc_data_list(cls, nc_data_list: list[NCData]) -> "BatchedPointCloudData":
        """Create BatchedPointCloudData from list of PointCloudData.

        Args:
            nc_data_list: List of PointCloudData instances to convert

        Returns:
            BatchedPointCloudData with shape (1, T, 3, N) where T = len(nc_data_list)
        """
        import numpy as np

        from neuracore_types.nc_data.point_cloud_data import PointCloudData

        points_list = []
        rgb_points_list = []
        extrinsics_list = []
        intrinsics_list = []
        has_rgb = False
        has_extrinsics = False
        has_intrinsics = False

        for nc in nc_data_list:
            pc_data: PointCloudData = cast(PointCloudData, nc)
            # (N, 3) -> (3, N)
            points_list.append(np.array(pc_data.points).T)

            if pc_data.rgb_points is not None:
                has_rgb = True
                rgb_points_list.append(np.array(pc_data.rgb_points).T)
            if pc_data.extrinsics is not None:
                has_extrinsics = True
                extrinsics_list.append(pc_data.extrinsics)
            if pc_data.intrinsics is not None:
                has_intrinsics = True
                intrinsics_list.append(pc_data.intrinsics)

        # Shape: (1, T, 3, N)
        points_tensor = torch.tensor(
            np.stack(points_list), dtype=torch.float32
        ).unsqueeze(0)

        rgb_points_tensor = None
        if has_rgb and len(rgb_points_list) == len(nc_data_list):
            rgb_points_tensor = torch.tensor(
                np.stack(rgb_points_list), dtype=torch.uint8
            ).unsqueeze(0)

        extrinsics_tensor = None
        if has_extrinsics and len(extrinsics_list) == len(nc_data_list):
            extrinsics_tensor = torch.tensor(
                np.stack(extrinsics_list), dtype=torch.float32
            ).unsqueeze(0)

        intrinsics_tensor = None
        if has_intrinsics and len(intrinsics_list) == len(nc_data_list):
            intrinsics_tensor = torch.tensor(
                np.stack(intrinsics_list), dtype=torch.float32
            ).unsqueeze(0)

        return cls(
            points=points_tensor,
            rgb_points=rgb_points_tensor,
            extrinsics=extrinsics_tensor,
            intrinsics=intrinsics_tensor,
        )

    @classmethod
    def sample(
        cls, batch_size: int = 1, time_steps: int = 1
    ) -> "BatchedPointCloudData":
        """Sample an example instance of BatchedPointCloudData."""
        shape_3d = (batch_size, time_steps, 3, 1000)

        return cls(
            points=torch.zeros(shape_3d, dtype=torch.float32),
            rgb_points=torch.zeros(shape_3d, dtype=torch.uint8),
            extrinsics=torch.zeros((batch_size, time_steps, 4, 4), dtype=torch.float32),
            intrinsics=torch.zeros((batch_size, time_steps, 3, 3), dtype=torch.float32),
        )
