"""Request models for dataset and recording synchronization operations."""

from pydantic import BaseModel, ConfigDict, Field

from neuracore_types.episode.episode import RobotDataSpec
from neuracore_types.utils.pydantic_to_ts import (
    REQUIRED_WITH_DEFAULT_FLAG,
    fix_required_with_defaults,
)


class SynchronizationDetails(BaseModel):
    """Details for synchronization requests.

    Attributes:
        frequency: Synchronization frequency in Hz.
        robot_data_spec: Specification of robot data to include in the synchronization.
        max_delay_s: Maximum allowable delay (in seconds) for synchronization.
        allow_duplicates: Whether to allow duplicate data points in the synchronization.
        trim_start_end: Whether to trim the start and end of the episode
            when synchronizing.
    """

    frequency: int
    robot_data_spec: RobotDataSpec | None
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
        robot_data_spec_hashable = None
        if self.robot_data_spec is not None:
            # Convert dict[str, dict[DataType, list[str]]] to a frozen structure
            robot_data_spec_hashable = tuple(
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
                    for robot_name, data_spec in self.robot_data_spec.items()
                )
            )

        return hash((
            self.frequency,
            robot_data_spec_hashable,
            self.max_delay_s,
            self.allow_duplicates,
            self.trim_start_end,
        ))
