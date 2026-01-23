"""Base classes for Neuracore data types."""

import time
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from neuracore_types.importer.data_config import DataFormat, MappingItem
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class NCDataStats(BaseModel):
    """Base class for statistics of Neuracore data types."""

    pass


class NCDataImportConfig(BaseModel):
    """Configuration for importing data to Neuracore."""

    source: str = Field(
        default="",
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    mapping: list[MappingItem] = Field(
        default_factory=list,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    format: DataFormat = Field(
        default_factory=DataFormat,
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the NCDataImportConfig."""
        super().__init__(**data)
        self._populate_transforms()

    def _populate_transforms(self) -> None:
        """Populate transforms based on configuration."""
        raise NotImplementedError(
            "Subclasses must implement _populate_transforms() method."
        )


class NCData(BaseModel):
    """Base class for all Neuracore data with automatic timestamping.

    Provides a common base for all data types in the system with automatic
    timestamp generation for temporal synchronization and data ordering.
    """

    timestamp: float = Field(
        default_factory=lambda: time.time(),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    def calculate_statistics(self) -> NCDataStats:
        """Calculate the statistics for this data type."""
        raise NotImplementedError(
            "Subclasses must implement calculate_statistics() method."
        )

    @classmethod
    def sample(cls) -> "NCData":
        """Sample an example NCData instance."""
        raise NotImplementedError("sample method must be implemented in subclasses.")


class DataItemStats(BaseModel):
    """Statistical summary of data dimensions and distributions.

    Contains statistical information about data arrays including means,
    standard deviations, counts, and maximum lengths for normalization
    and model configuration purposes.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=fix_required_with_defaults
    )

    mean: np.ndarray = Field(
        default_factory=lambda: np.array([]),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    std: np.ndarray = Field(
        default_factory=lambda: np.array([]),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    count: np.ndarray = Field(
        default_factory=lambda: np.array([]),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    min: np.ndarray = Field(
        default_factory=lambda: np.array([]),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )
    max: np.ndarray = Field(
        default_factory=lambda: np.array([]),
        json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG,
    )

    def concatenate(self, other: "DataItemStats") -> "DataItemStats":
        """Concatenate two DataItemStats objects along the data dimension."""
        if not isinstance(other, DataItemStats):
            raise ValueError("Can only concatenate with another DataItemStats object.")

        return DataItemStats(
            mean=np.concatenate((self.mean, other.mean)),
            std=np.concatenate((self.std, other.std)),
            count=np.concatenate((self.count, other.count)),
            min=np.concatenate((self.min, other.min)),
            max=np.concatenate((self.max, other.max)),
        )

    @classmethod
    def _decode_field(
        cls, v: Union[list, np.ndarray], dtype: Any
    ) -> Optional[np.ndarray]:
        """Decode field to NumPy array with specified dtype."""
        return np.array(v, dtype=dtype) if isinstance(v, list) else v

    @staticmethod
    def _serialize_field(v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize NumPy array to list."""
        return v.tolist() if v is not None else None

    @field_validator("mean", mode="before")
    @classmethod
    def decode_mean(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode mean field to NumPy array."""
        return cls._decode_field(v, np.float32)

    @field_serializer("mean", when_used="json")
    def serialize_mean(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize mean field to JSON list."""
        return self._serialize_field(v)

    @field_validator("std", mode="before")
    @classmethod
    def decode_std(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode std field to NumPy array."""
        return cls._decode_field(v, np.float32)

    @field_serializer("std", when_used="json")
    def serialize_std(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize std field to JSON list."""
        return self._serialize_field(v)

    @field_validator("count", mode="before")
    @classmethod
    def decode_count(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode count field to NumPy array."""
        return cls._decode_field(v, np.int64)

    @field_serializer("count", when_used="json")
    def serialize_count(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize count field to JSON list."""
        return self._serialize_field(v)

    @field_validator("min", mode="before")
    @classmethod
    def decode_min(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode min field to NumPy array."""
        return cls._decode_field(v, np.float32)

    @field_serializer("min", when_used="json")
    def serialize_min(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize min field to JSON list."""
        return self._serialize_field(v)

    @field_validator("max", mode="before")
    @classmethod
    def decode_max(cls, v: Union[list, np.ndarray]) -> Optional[np.ndarray]:
        """Decode max field to NumPy array."""
        return cls._decode_field(v, np.float32)

    @field_serializer("max", when_used="json")
    def serialize_max(self, v: Optional[np.ndarray]) -> Optional[list]:
        """Serialize max field to JSON list."""
        return self._serialize_field(v)
