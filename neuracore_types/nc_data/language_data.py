"""Data models for natural language data."""

from typing import Literal

from pydantic import ConfigDict, Field

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class LanguageDataStats(NCDataStats):
    """Statistics for LanguageData."""

    type: Literal["LanguageDataStats"] = Field(
        default="LanguageDataStats", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    text: DataItemStats

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)


class LanguageData(NCData):
    """Natural language instruction or description data.

    Contains text-based information such as task descriptions, voice commands,
    or other linguistic data associated with robot demonstrations.
    """

    type: Literal["LanguageData"] = Field(
        default="LanguageData", json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    text: str

    model_config = ConfigDict(json_schema_extra=fix_required_with_defaults)

    @classmethod
    def sample(cls) -> "LanguageData":
        """Sample an example LanguageData instance.

        Returns:
            LanguageData: Sampled LanguageData instance
        """
        return cls(text="Sample instruction.")

    def calculate_statistics(self) -> LanguageDataStats:
        """Calculate the statistics for this data type."""
        return LanguageDataStats(text=DataItemStats())
