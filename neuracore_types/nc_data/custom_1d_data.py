"""Custom 1D numerical data for specialized applications."""

import copy
from typing import Literal, Optional, Union

import numpy as np
from pydantic import ConfigDict, Field, field_serializer, field_validator

from neuracore_types.importer.transform import (
    DataTransform,
    DataTransformSequence,
    FlipSign,
    Offset,
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


class Custom1DDataStats(NCDataStats):
    """Statistics for Custom1DData."""

    type: Literal["Custom1DDataStats"] = Field(
        default="Custom1DDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    data: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class Custom1DDataImportConfig(NCDataImportConfig):
    """Import configuration for Custom1DData."""

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        transform_list: list[DataTransform] = []
        for item in self.mapping:
            item_transforms: list[DataTransform] = copy.deepcopy(transform_list)
            if item.inverted:
                item_transforms.append(FlipSign())
            if item.offset != 0.0:
                item_transforms.append(Offset(value=item.offset))
            item.transforms = DataTransformSequence(transforms=item_transforms)


class Custom1DData(NCData):
    """Custom 1D numerical data for specialized applications.

    Used for representing custom sensor data or application-specific
    numerical information that is one-dimensional.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=fix_required_with_defaults
    )

    type: Literal["Custom1DData"] = Field(
        default="Custom1DData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    data: np.ndarray  # 1D array of float32

    @classmethod
    def sample(cls) -> "Custom1DData":
        """Sample an example Custom1DData instance.

        Returns:
            Custom1DData: Sampled Custom1DData instance
        """
        return cls(data=np.zeros((10,), dtype=np.float32))

    @field_validator("data", mode="before")
    @classmethod
    def decode_data(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode data to NumPy array.

        Args:
            v: List or NumPy array
        Returns:
            Decoded NumPy array or None
        """
        return np.array(v, dtype=np.float32) if isinstance(v, list) else v

    @field_serializer("data", when_used="json")
    def serialize_data(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Encode NumPy array to JSON list.

        Args:
            v: NumPy array to encode

        Returns:
            List or None
        """
        return v.tolist() if v is not None else None

    def calculate_statistics(self) -> Custom1DDataStats:
        """Calculate the statistics for this data type.

        Returns:
            Dictionary attribute names to their corresponding DataItemStats.
        """
        stats = DataItemStats(
            mean=np.array(self.data, dtype=np.float32),
            std=np.zeros_like(self.data, dtype=np.float32),
            count=np.array([1], dtype=np.int32),
            min=np.array(self.data, dtype=np.float32),
            max=np.array(self.data, dtype=np.float32),
        )
        return Custom1DDataStats(data=stats)
