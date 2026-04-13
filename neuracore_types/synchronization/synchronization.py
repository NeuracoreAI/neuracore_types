"""Request models for dataset and recording synchronization operations."""

from pydantic import BaseModel, ConfigDict, Field

from neuracore_types.episode.episode import CrossEmbodimentUnion
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class SynchronizationDetails(BaseModel):
    """Details for synchronization requests.

    Attributes:
        frequency: Synchronization frequency in Hz.
        cross_embodiment_description: Specification of robot data to include
            in the synchronization.
        max_delay_s: Maximum allowable delay (in seconds) for synchronization.
        allow_duplicates: Whether to allow duplicate data points in the synchronization.
        trim_start_end: Whether to trim the start and end of the episode
            when synchronizing.
    """

    frequency: int
    cross_embodiment_union: CrossEmbodimentUnion | None
    max_delay_s: float = Field(
        default=0.1, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    allow_duplicates: bool = Field(
        default=True, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )
    trim_start_end: bool = Field(
        default=True, json_schema_extra=REQUIRED_WITH_DEFAULT_FLAG
    )

    model_config = ConfigDict(frozen=True, json_schema_extra=fix_required_with_defaults)

    def __hash__(self) -> int:
        """Compute a hash value for the SynchronizationDetails instance.

        Returns:
            int: The computed hash value.
        """
        # Convert the nested dict structure to something hashable
        cross_embodiment_union = None
        if self.cross_embodiment_union is not None:
            # Convert CrossEmbodimentUnion to a frozen structure
            cross_embodiment_union = tuple(
                sorted(
                    (
                        robot_name,
                        tuple(
                            sorted(
                                (data_type, tuple(fields))
                                for data_type, fields in data_spec.items()
                            )
                        ),
                    )
                    for robot_name, data_spec in self.cross_embodiment_union.items()
                )
            )

        return hash((
            self.frequency,
            cross_embodiment_union,
            self.max_delay_s,
            self.allow_duplicates,
            self.trim_start_end,
        ))
