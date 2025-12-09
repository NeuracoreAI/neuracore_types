"""Data models for natural language data."""

from typing import Literal

from neuracore_types.nc_data.nc_data import DataItemStats, NCData, NCDataStats


class LanguageDataStats(NCDataStats):
    """Statistics for LanguageData."""

    type: Literal["LanguageDataStats"] = "LanguageDataStats"
    text: DataItemStats


class LanguageData(NCData):
    """Natural language instruction or description data.

    Contains text-based information such as task descriptions, voice commands,
    or other linguistic data associated with robot demonstrations.
    """

    type: Literal["LanguageData"] = "LanguageData"
    text: str

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
